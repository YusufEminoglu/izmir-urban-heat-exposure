"""Audit: reproduce the R0 numbers the revision builds on before changing anything.

Checks Table 4 (LCZ-LST), the KW statistic, the MGWR sample representativeness
against the full 1.19 M-cell grid, the grid geometry, and the mahalle
morphology-pressure component from the stored IDW coefficient grid.
"""
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *  # noqa

g = load_grid()
print("valid cells", len(g))

# grid geometry: x = x0 + (col+0.5)*100 ; y = y0 - (row+0.5)*100
x0 = (g["x"] - (g["col"] + 0.5) * 100).unique()
y0 = (g["y"] + (g["row"] + 0.5) * 100).unique()
print("x0", x0[:5], "y0", y0[:5])

# Table 4
t = g.groupby("lcz")[DEP].agg(["count", "mean", "std", "median"]).round(2)
print(t)
groups = [v[DEP].to_numpy() for _, v in g.groupby("lcz")]
H, p = stats.kruskal(*groups)
K, N = len(groups), len(g)
print("KW H", H, "eta2", (H - K + 1) / (N - K))

# MGWR sample representativeness against the full grid
s = gpd.read_file(MGWR_SAMPLE)
for v in [DEP] + PREDICTORS + ["tree_canopy_height_2025", "srtm_elevation_m", "srtm_slope_deg"]:
    ks = stats.ks_2samp(g[v], s[v])
    print(f"{v:28s} full {g[v].mean():8.4f} samp {s[v].mean():8.4f} KS p={ks.pvalue:.3f}")
sh_full = g["lcz"].value_counts(normalize=True)
sh_samp = s["lcz"].value_counts(normalize=True)
d = (sh_full - sh_samp.reindex(sh_full.index).fillna(0)) * 100
print("LCZ share diff (pp):\n", d.round(3))
print("sample LCZ counts:\n", s["lcz"].value_counts().sort_index())
# is the sample on the grid?
m = s.merge(g[["row", "col", DEP]], on=["row", "col"], how="left", suffixes=("", "_g"))
print("sample rows matched in grid:", m[DEP + "_g"].notna().sum(), "max abs diff", (m[DEP] - m[DEP + "_g"]).abs().max())

# IDW grid vs sample coefficient ranges
idw = gpd.read_file(IDW_GRID)
print(idw.describe().T)

# mahalle morphology pressure reproduction
mah = gpd.read_file(MAHALLE).to_crs(CRS)
pri = pd.read_csv(PRIORITY_CSV, encoding="utf-8-sig")
print("priority rows", len(pri), pri[["lst_mean", "morph_pressure", "vulnerable_share", "pop_total", "priority_score_0_100"]].describe().T)

# values for the manuscript
from scipy.spatial import cKDTree
kps = [stats.ks_2samp(g[v], s[v]).pvalue for v in [DEP] + PREDICTORS + ["tree_canopy_height_2025", "srtm_elevation_m", "srtm_slope_deg"]]
dd, _ = cKDTree(s[["x", "y"]].to_numpy(float)).query(g[["x", "y"]].to_numpy(float))
vc = s["lcz"].value_counts()
write_values("r1_00_audit", {
    "mgwr_sample_ks_minp": fmt(min(kps), 2),
    "mgwr_sample_coverage_pp": fmt(d.abs().max(), 2),
    "mgwr_sample_dist_median": int(round(np.median(dd))), "mgwr_sample_dist_p95": f"{int(round(np.percentile(dd, 95))):,}".replace(",", "{,}"),
    "mgwr_sample_dist_max": f"{int(round(dd.max())):,}".replace(",", "{,}"),
    "mgwr_sample_n_lcz": int(s["lcz"].nunique()), "mgwr_sample_lcz2": int(vc.get(2, 0)), "mgwr_sample_lcz10": int(vc.get(10, 0)),
    "mgwr_sample_lcz5": int(vc.get(5, 0)), "mgwr_sample_lcz3": int(vc.get(3, 0)), "mgwr_sample_lcz8": int(vc.get(8, 0)),
})
