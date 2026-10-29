"""Reported MGWR on the v2 layers: sample columns, local R2 and the 500 m coefficient atlas.

Input: outputs/cache/mgwr_orig.npz (r2_28 remote fit of the 4,000-cell calibration sample on the v2 layers).
* data/paper1_mgwr_sample_v2.gpkg gains the columns of the earlier reported sample: mgwr_intercept, mgwr_coef_*,
  mgwr_t_*, mgwr_local_r2, mgwr_cond, mgwr_residual_scaled, mgwr_residual_lst. Local R2 uses the method of the
  earlier run (adaptive bisquare neighbourhood of k = median predictor bandwidth, weighted 1 - RSS/TSS).
* data/paper1_mgwr_coefficient_grid_idw_v2.gpkg: the same 500 m atlas points as before, inverse-distance
  interpolated (12 neighbours, power 2) from the calibration cells; dominant driver = largest absolute coefficient.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
from mgwr_core import idw  # noqa

f = np.load(CACHE / "mgwr_orig.npz", allow_pickle=True)
s = gpd.read_file(MGWR_SAMPLE)
assert (s.row.to_numpy() == f["rows"]).all() and (s.col.to_numpy() == f["cols"]).all()
params, tv, bws = f["params"], f["tvalues"], f["bws"]
coords, e, yS = f["coords"], f["resid"], f["yS"]
k_ref = max(int(round(np.median(bws[1:]))), 5)
k_ref = min(k_ref, len(coords) - 1)
dnn, inn = cKDTree(coords).query(coords, k=k_ref + 1)
lr2 = np.empty(len(coords))
for i in range(len(coords)):
    dd, nb = dnn[i], inn[i]
    w = (1 - (dd / (dd[-1] if dd[-1] > 0 else 1.0)) ** 2) ** 2
    yb = np.average(yS[nb], weights=w)
    tss = np.sum(w * (yS[nb] - yb) ** 2)
    lr2[i] = 1 - np.sum(w * e[nb] ** 2) / tss if tss > 0 else np.nan
s["mgwr_intercept"] = params[:, 0]
for j, p in enumerate(PREDICTORS, start=1):
    s[f"mgwr_coef_{p}"] = params[:, j]
    s[f"mgwr_t_{p}"] = tv[:, j]
s["mgwr_local_r2"] = lr2
s["mgwr_cond"] = f["cond"]
s["mgwr_residual_scaled"] = e
s["mgwr_residual_lst"] = e * float(f["ys"])
s.to_file(MGWR_SAMPLE, driver="GPKG")

a1 = gpd.read_file(DATA / "mgwr_run" / "paper1_mgwr_coefficient_grid_idw.gpkg")
dst = np.column_stack([a1.geometry.x, a1.geometry.y])
at = gpd.GeoDataFrame(geometry=a1.geometry.copy(), crs=a1.crs)
for j, nm in enumerate(["ndvi", "ndbi", "albedo"], start=1):
    at[f"mgwr_coef_{nm}"] = idw(coords, params[:, j], dst)
B = at[["mgwr_coef_ndvi", "mgwr_coef_ndbi", "mgwr_coef_albedo"]].abs().to_numpy()
at["dominant_driver"] = np.array(["ndvi", "ndbi", "albedo"])[B.argmax(1)]
at["local_r2"] = idw(coords, lr2, dst)
at["local_cond"] = idw(coords, f["cond"], dst)
at = at[["mgwr_coef_ndvi", "mgwr_coef_ndbi", "mgwr_coef_albedo", "dominant_driver", "local_r2", "local_cond", "geometry"]]
at.to_file(DATA / "paper1_mgwr_coefficient_grid_idw_v2.gpkg", driver="GPKG")
v = {"mgwr_local_r2_k": k_ref, "mgwr_local_r2_med": fmt(np.nanmedian(lr2), 3)}
print(v, at.dominant_driver.value_counts(normalize=True).round(3).to_dict())
write_values("r2_29_finalize_v2_fit", v)
