"""Summary values of the reported MGWR on the v2 layers (Table 6, Table S3 and the text), from the r2_28 fit and the
v2 calibration sample (r2_29): bandwidths, fit, ENP, AICc, coefficient ranges, local condition numbers, local R2 and
residual Moran's I (8 nearest neighbours, row-standardised, 999 permutations)."""
import sys
from pathlib import Path
import numpy as np
import geopandas as gpd
from libpysal.weights import KNN
from esda.moran import Moran

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

f = np.load(CACHE / "mgwr_orig.npz", allow_pickle=True)
s = gpd.read_file(MGWR_SAMPLE)
P, bws, cond = f["params"], f["bws"], f["cond"]
lr2 = s["mgwr_local_r2"].to_numpy(float)
v = {"mgwr_r2": fmt(float(f["R2"]), 3), "mgwr_adj_r2": fmt(float(f["adjR2"]), 3), "mgwr_aicc": fmt(float(f["aicc"]), 1),
     "mgwr_enp": fmt(float(f["ENP"]), 1)}
for j, nm in enumerate(["intercept", "ndvi", "ndbi", "albedo"]):
    v[f"mgwr_bw_{nm}"] = int(bws[j])
for j, nm in [(1, "ndvi"), (2, "ndbi"), (3, "albedo")]:
    v[f"mgwr_{nm}_min"] = fmt(P[:, j].min(), 3)
    v[f"mgwr_{nm}_max"] = fmt(P[:, j].max(), 3)
v["mgwr_cond_median"] = fmt(np.median(cond), 2)
v["mgwr_cond_max"] = fmt(cond.max(), 2)
v["mgwr_cond_gt30"] = fmt(100 * (cond > 30).mean(), 1)
v["mgwr_local_r2_median"] = fmt(np.nanmedian(lr2), 3)
v["mgwr_local_r2_min"] = fmt(np.nanmin(lr2), 3)
v["mgwr_local_r2_max"] = fmt(np.nanmax(lr2), 3)
v["mgwr_local_r2_sd"] = fmt(np.nanstd(lr2), 3)
w = KNN.from_array(f["coords"], k=8); w.transform = "r"
np.random.seed(0)
m = Moran(f["resid"], w, permutations=999)
v["mgwr_resid_morans_i"] = fmt(m.I, 3)
v["mgwr_resid_morans_p"] = "<0.001" if m.p_sim <= 0.001 else fmt(m.p_sim, 3)
for j, nm in [(1, "ndvi"), (2, "ndbi"), (3, "albedo")]:
    v[f"mgwr_{nm}_pos"] = fmt(100 * (P[:, j] > 0).mean(), 1)
# grid-wide Spearman correlations quoted in the Discussion (cells with complete covariates)
from scipy import stats
g = load_grid()
v["rho_ndvi_lst"] = fmt(stats.spearmanr(g.ndvi_2025, g[DEP]).statistic, 2)
v["rho_ndvi_ndbi"] = fmt(stats.spearmanr(g.ndvi_2025, g.ndbi_2025).statistic, 2)
print(v)
write_values("r2_31_mgwr_summary_v2", v)
