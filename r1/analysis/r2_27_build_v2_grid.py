"""Analysis grid v2: the 2025 Landsat layers from the documented Earth Engine export.

The spectral indices of the alternative 2025 layers could not be traced to a recipe
(r2_01-r2_04, r2_17, r2_25, r2_26). LST, NDVI, NDBI, albedo and EVI of every province cell are
the documented June-August 2025 Landsat 8/9 Collection 2 Level-2 composite of r2_05 (scene cloud cover < 10%,
QA_PIXEL bits 0-5 and QA_RADSAT mask, per-pixel mean; LST = ST_B10 in degrees Celsius), sampled on the analysis grid.
Cells without a valid composite value (shoreline) take the value of the nearest valid cell.
The province-standardised LST anomaly and UTFVI (kelvin) follow from LST. Slope and aspect are derived for every cell
with the same finite-difference operator on the 100 m SRTM DEM completed outside the province with the same source,
so that cells next to the province edge use real neighbours.

Writes data/paper1_model_grid_100m_v2.csv (same columns as the complete grid, plus nearest_fill) and keeps the
alternative layers in data/paper1_layers_2025_previous_export.csv for the sensitivity analysis.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

NR, NC = 1744, 1981
g = pd.read_csv(DATA / "paper1_model_grid_100m_complete.csv", encoding="utf-8-sig")
r, c = g.row.to_numpy(int), g.col.to_numpy(int)
prov = np.zeros((NR, NC), bool); prov[r, c] = True
G = {k: np.load(CACHE / "gee" / f"l_{k}.npy").astype(float) for k in ["lst", "ndvi", "ndbi", "albedo", "evi"]}

prev = g[["row", "col", "lst_2025", "ndvi_2025", "ndbi_2025", "albedo_2025", "evi_2025", "uhi_2025_diagnostic",
          "srtm_slope_deg", "srtm_aspect_sin", "srtm_aspect_cos"]].copy()
prev.to_csv(DATA / "paper1_layers_2025_previous_export.csv", index=False)

valid = prov & np.isfinite(G["lst"]) & np.isfinite(G["ndvi"]) & np.isfinite(G["ndbi"]) & np.isfinite(G["albedo"]) & np.isfinite(G["evi"])
_, (iy, ix) = ndimage.distance_transform_edt(~valid, return_indices=True)
g["nearest_fill"] = (~valid[r, c]).astype(int)
for k in G:
    g[f"{k}_2025"] = G[k][iy, ix][r, c]
T = g["lst_2025"]
g["uhi_2025_diagnostic"] = (T - T.mean()) / T.std(ddof=0)
TK = T + 273.15
g["utfvi_2025_diagnostic"] = (TK - TK.mean()) / TK

# terrain: stored 100 m elevation inside the province, same-source SRTM (bias-matched) outside, then the stored operator
dem_s = np.full((NR, NC), np.nan); dem_s[r, c] = g.srtm_elevation_m.to_numpy(float)
srtm = np.load(CACHE / "gee" / "srtm.npy").astype(float)
b = np.nanmedian(dem_s[prov] - srtm[prov])
dem = np.where(np.isfinite(dem_s), dem_s, srtm + b)
dem = np.where(np.isfinite(dem), dem, np.nanmedian(dem_s))  # sea beyond the SRTM footprint
dz_dy, dz_dx = np.gradient(dem, 100.0, 100.0)
slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
aspect = np.mod(np.arctan2(-dz_dx, dz_dy), 2 * np.pi)
g["srtm_slope_deg"] = slope[r, c]; g["srtm_aspect_sin"] = np.sin(aspect)[r, c]; g["srtm_aspect_cos"] = np.cos(aspect)[r, c]

g.to_csv(DATA / "paper1_model_grid_100m_v2.csv", index=False)
v = {"v2_n_nearest": int(g.nearest_fill.sum())}
for k in ["lst", "ndvi", "ndbi", "albedo", "evi"]:
    a, p = g[f"{k}_2025"].to_numpy(), prev[f"{k}_2025"].to_numpy()
    ok = np.isfinite(a) & np.isfinite(p)
    v[f"prev_r_{k}"] = fmt(np.corrcoef(a[ok], p[ok])[0, 1], 3)
    v[f"v2_mean_{k}"] = fmt(np.nanmean(a), 3); v[f"prev_mean_{k}"] = fmt(np.nanmean(p), 3)
ok = np.isfinite(prev.srtm_slope_deg.to_numpy())
v["prev_r_slope"] = fmt(np.corrcoef(g.srtm_slope_deg[ok], prev.srtm_slope_deg[ok])[0, 1], 3)
v["slope_changed_pct"] = fmt(100 * (np.abs(g.srtm_slope_deg[ok] - prev.srtm_slope_deg[ok]) > 0.01).mean(), 2)
print(v)
write_values("r2_27_build_v2_grid", v)
