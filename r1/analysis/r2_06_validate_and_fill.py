"""Validate same-source GEE layers against the stored grid and complete the province grid.

1. Exact-source layers (LCZ, LCZ probability, CORINE 2018, GLC_FCS30D 2022, SRTM-derived terrain):
   agreement with the stored values on the stored cells.
2. 2025 spectral/thermal layers: for the cells missing from the stored composite, values are
   predicted from the re-exported Landsat 8/9 and Sentinel-2 JJA 2025 composites with a gradient-
   boosting calibration trained on stored cells near the gaps; accuracy by spatial block CV.
3. Writes the completed grid (all cells inside the province) to data/paper1_model_grid_100m_complete.csv
   and complete 2025 rasters to outputs/cache/complete/*.npy.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from scipy import ndimage
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

NR, NC = 1744, 1981
G = CACHE / "gee"
gee = {p.stem: np.load(p) for p in G.glob("*.npy")}
mask = np.load(CACHE / "province_mask.npy")
g = load_grid(valid_only=False)
r, c = g.row.to_numpy(), g.col.to_numpy()
stored = np.zeros((NR, NC), bool); stored[r, c] = True
new = mask & ~stored
print("province cells", mask.sum(), "stored rows", stored.sum(), "stored outside mask", (stored & ~mask).sum(), "cells to add", new.sum())
rep = []

# ---- 1. exact-source agreement
def agree(name, key, tol=None):
    a = gee[key][r, c]; s = g[name].to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(s)
    if tol is None:
        v = np.mean(a[ok] == s[ok])
        rep.append(dict(layer=name, metric="identical share", value=v))
    else:
        v = np.corrcoef(a[ok], s[ok])[0, 1]
        rep.append(dict(layer=name, metric="pearson r", value=v, mad=np.median(np.abs(a[ok] - s[ok]))))
    print(name, rep[-1])


agree("lcz", "lcz"); agree("lcz_probability", "lcz_probability")
agree("corine_2018", "corine_2018")
_gs = g.glc_fcs_2022.to_numpy(float); _ga = gee["glc_2022"][r, c]; _ok = np.isfinite(_gs) & np.isfinite(_ga)
GLC_MAP = pd.crosstab(_ga[_ok], _gs[_ok]).idxmax(axis=1).to_dict()
gee["glc_2022_idx"] = np.vectorize(lambda v: GLC_MAP.get(v, np.nan))(gee["glc_2022"]).astype(np.float32)
gee["glc_2015_idx"] = np.vectorize(lambda v: GLC_MAP.get(v, np.nan))(gee["glc_2015"]).astype(np.float32)
agree("glc_fcs_2022", "glc_2022_idx")
print("GLC code map", GLC_MAP)
agree("srtm_elevation_m", "srtm", tol=1)

# terrain from GEE SRTM (sampled at the 100 m cell) with the stored derivation (numpy gradient on the 100 m DEM)
dem_s = np.full((NR, NC), np.nan); dem_s[r, c] = g.srtm_elevation_m.to_numpy(float)
_b = np.nanmedian(dem_s[stored] - gee["srtm"][stored])
dem = np.where(np.isfinite(dem_s), dem_s, gee["srtm"] + _b).astype(float)
dz_dy, dz_dx = np.gradient(dem, 100.0, 100.0)
slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
aspect = np.mod(np.arctan2(-dz_dx, dz_dy), 2 * np.pi)
terrain = {"srtm_elevation_m": dem, "srtm_slope_deg": slope, "srtm_aspect_sin": np.sin(aspect), "srtm_aspect_cos": np.cos(aspect)}
for _k in ["srtm_slope_deg", "srtm_aspect_sin", "srtm_aspect_cos"]:
    _st = np.full((NR, NC), np.nan); _st[r, c] = g[_k].to_numpy(float)
    rep.append(dict(layer=_k + " (hybrid DEM)", metric="pearson r", value=np.corrcoef(terrain[_k][stored & np.isfinite(_st)], _st[stored & np.isfinite(_st)])[0, 1]))
    print(rep[-1])
for k in ["srtm_slope_deg", "srtm_aspect_sin", "srtm_aspect_cos"]:
    s = g[k].to_numpy(float); a = terrain[k][r, c]; ok = np.isfinite(a) & np.isfinite(s)
    rep.append(dict(layer=k, metric="pearson r", value=np.corrcoef(a[ok], s[ok])[0, 1], mad=np.median(np.abs(a[ok] - s[ok]))))
    print(k, rep[-1])

# ---- 2. calibration of the 2025 layers
c25 = xr.open_dataset(CUBE_2025)
targets = {"lst_2025": "lst_2025", "albedo_2025": "albedo_2025", "ndvi_2025": "ndvi_2025", "ndbi_2025": "ndbi_2025", "evi_2025": "evi_2025"}
feat_names = ["l_lst", "l_albedo", "l_ndvi", "l_ndbi", "l_evi", "s2_ndvi", "s2_ndbi", "s2_evi", "srtm"]
F = np.stack([gee[k] for k in feat_names], -1).astype(np.float32)
lczA = gee["lcz"]
Fx = np.concatenate([F, lczA[..., None]], -1)
dist_gap = ndimage.distance_transform_edt(~new)  # distance (cells) to nearest gap cell
train_zone = mask & stored & (dist_gap <= 100)  # stored cells within 10 km of a gap
yy, xx = np.where(train_zone)
blk = (yy // 50) * 1000 + (xx // 50)  # 5 km blocks
complete = {}
for name in targets:
    tgt = c25[name].values
    ok = np.isfinite(tgt[yy, xx]) & np.isfinite(Fx[yy, xx]).all(1)
    X, y, grp = Fx[yy, xx][ok], tgt[yy, xx][ok], blk[ok]
    sub = np.random.default_rng(0).choice(len(y), min(len(y), 150_000), replace=False)
    X, y, grp = X[sub], y[sub], grp[sub]
    pred = np.empty_like(y)
    for tr, te in GroupKFold(5).split(X, y, grp):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, categorical_features=[X.shape[1] - 1], random_state=0)
        m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    r2 = 1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rmse = np.sqrt(((pred - y) ** 2).mean())
    base_r = np.corrcoef(X[:, feat_names.index("l_" + name.split("_")[0]) if name != "lst_2025" else 0], y)[0, 1]
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, categorical_features=[X.shape[1] - 1], random_state=0).fit(X, y)
    out = tgt.copy()
    ny, nx = np.where(new)
    okn = np.isfinite(Fx[ny, nx]).all(1)
    out[ny[okn], nx[okn]] = m.predict(Fx[ny, nx][okn])
    _miss = mask & ~np.isfinite(out)
    if _miss.any():
        _idx = ndimage.distance_transform_edt(~np.isfinite(out), return_distances=False, return_indices=True)
        out[_miss] = out[_idx[0][_miss], _idx[1][_miss]]
    out[~mask] = np.nan
    complete[name] = out
    rep.append(dict(layer=name, metric="calibration block-CV R2", value=r2, rmse=rmse, sd=y.std(), r_single_source=base_r,
                    n_train=len(y), filled=int(okn.sum()), unfilled=int((~okn).sum())))
    print(name, rep[-1], flush=True)

# UHI and UTFVI follow from LST (province-standardised anomaly and thermal field variance index)
lst_s = c25["lst_2025"].values.astype(np.float64); uhi_s = c25["uhi_2025"].values.astype(np.float64); utf_s = c25["utfvi_2025"].values.astype(np.float64)
okc = np.isfinite(lst_s) & np.isfinite(uhi_s)
mu, sd = np.nanmean(lst_s[okc]), np.nanstd(lst_s[okc])
print("UHI check r:", np.corrcoef(uhi_s[okc], (lst_s[okc] - mu) / sd)[0, 1], "max diff", np.nanmax(np.abs(uhi_s[okc] - (lst_s[okc] - mu) / sd)))
print("UTFVI check max diff:", np.nanmax(np.abs(utf_s[okc] - (lst_s[okc] - mu) / lst_s[okc])))
L = complete["lst_2025"]
_a, _b0 = np.polyfit(lst_s[okc], uhi_s[okc], 1)
_ok2 = np.isfinite(lst_s) & np.isfinite(utf_s)
_c1, _c0 = np.polyfit(1.0 / lst_s[_ok2], utf_s[_ok2], 1)
print("UHI = %.5f*LST + %.4f ; UTFVI = %.4f/LST + %.4f" % (_a, _b0, _c1, _c0))
rep.append(dict(layer="uhi_2025", metric="linear fit r", value=np.corrcoef(uhi_s[okc], _a * lst_s[okc] + _b0)[0, 1]))
rep.append(dict(layer="utfvi_2025", metric="fit r", value=np.corrcoef(utf_s[_ok2], _c1 / lst_s[_ok2] + _c0)[0, 1]))
complete["uhi_2025"] = np.where(mask, np.where(np.isfinite(uhi_s), uhi_s, _a * L + _b0), np.nan)
complete["utfvi_2025"] = np.where(mask, np.where(np.isfinite(utf_s), utf_s, _c1 / L + _c0), np.nan)
complete["tree_canopy_height_2025"] = np.where(mask, c25["tree_canopy_height_2025"].values, np.nan)
(CACHE / "complete").mkdir(exist_ok=True)
for k, v in complete.items():
    np.save(CACHE / "complete" / f"{k}.npy", v.astype(np.float32))
pd.DataFrame(rep).to_csv(TABLES / "gapfill_validation.csv", index=False)

# ---- 3. completed grid
ny, nx = np.where(new)
add = pd.DataFrame({"row": ny, "col": nx, "x": 432600.0 + (nx + 0.5) * 100, "y": 4359700.0 - (ny + 0.5) * 100})
for k in ["lst_2025", "albedo_2025", "ndvi_2025", "ndbi_2025", "evi_2025", "tree_canopy_height_2025"]:
    add[k] = complete[k][ny, nx]
add["uhi_2025_diagnostic"] = complete["uhi_2025"][ny, nx]
add["utfvi_2025_diagnostic"] = complete["utfvi_2025"][ny, nx]
add["lcz"] = gee["lcz"][ny, nx]; add["lcz_probability"] = gee["lcz_probability"][ny, nx]
add["corine_2018"] = gee["corine_2018"][ny, nx]; add["glc_fcs_2022"] = gee["glc_2022_idx"][ny, nx]
for k, v in terrain.items():
    add[k] = v[ny, nx]
add["gapfilled"] = 1
mah = gpd.read_file(MAHALLE).to_crs(CRS)
pts = gpd.GeoDataFrame(add[["row"]], geometry=gpd.points_from_xy(add.x, add.y), crs=CRS)
jj = gpd.sjoin(pts, mah[["pop_total", "female_share", "child_0_14_share", "elderly_65_plus_share", "age_dependency_proxy", "geometry"]],
               predicate="within", how="left")
jj = jj[~jj.index.duplicated()]
for k in ["pop_total", "female_share", "child_0_14_share", "elderly_65_plus_share", "age_dependency_proxy"]:
    add[k] = jj[k].to_numpy()
g2 = g.copy(); g2["gapfilled"] = 0
full = pd.concat([g2, add[g2.columns]], ignore_index=True).sort_values(["row", "col"]).reset_index(drop=True)
full.to_csv(DATA / "paper1_model_grid_100m_complete.csv", index=False)
print("complete grid rows", len(full), "valid model rows", full.dropna(subset=MODEL_COLS).shape[0])

cs = (gee["corine_2012"][mask] == gee["corine_2018"][mask]).mean()
gs_ = (gee["glc_2015"][mask] == gee["glc_2022"][mask]).mean()
print("CORINE 2012-2018 unchanged", cs, "GLC 2015-2022 unchanged", gs_)
write_values("r2_06_gapfill", {
    "n_province_cells": f"{int(mask.sum()):,}".replace(",", "{,}"), "n_gapfilled": f"{int(new.sum()):,}".replace(",", "{,}"),
    "gapfill_share": fmt(100 * new.sum() / mask.sum(), 2),
    "corine_stable_pct": fmt(100 * cs, 1), "glc_stable_pct": fmt(100 * gs_, 1),
    "corine_changed_pct": fmt(100 - 100 * cs, 1), "glc_changed_pct": fmt(100 - 100 * gs_, 1),
    "gf_r2_lst": fmt([d for d in rep if d["layer"] == "lst_2025"][0]["value"], 3),
    "gf_rmse_lst": fmt([d for d in rep if d["layer"] == "lst_2025"][0]["rmse"], 2),
    "gf_r2_albedo": fmt([d for d in rep if d["layer"] == "albedo_2025"][0]["value"], 3),
    "gf_r2_ndvi": fmt([d for d in rep if d["layer"] == "ndvi_2025"][0]["value"], 2),
    "gf_r2_ndbi": fmt([d for d in rep if d["layer"] == "ndbi_2025"][0]["value"], 2),
    "gf_r2_evi": fmt([d for d in rep if d["layer"] == "evi_2025"][0]["value"], 2),
})
pd.DataFrame(rep).to_csv(TABLES / "gapfill_validation.csv", index=False)
