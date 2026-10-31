"""Stability of the MGWR solution across independent samples and sample sizes.

Fits: orig (reported sample, refit), rep1-rep5 (new stratified 4,000-cell
samples), n2000, n6000, and aug (reported sample with elevation and canopy
height added as local covariates).
For each fit: bandwidths, R2, ENP, AICc, residual Moran's I (k=8).
Every fit is evaluated with its own kernel-consistent estimator on the common
500 m atlas grid, and compared with orig: Pearson r of each coefficient
surface, sign agreement, dominant-driver agreement and LCZ-class medians.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from libpysal.weights import KNN
from esda.moran import Moran

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
from mgwr_core import local_betas, load_r0_fit

atlas = gpd.read_file(IDW_GRID)
axy = np.column_stack([atlas.geometry.x, atlas.geometry.y])
g = load_grid()
_, ii = cKDTree(g[["x", "y"]].to_numpy(float)).query(axy, k=1)
lcz_a = g.lcz.to_numpy()[ii]
NAMES = ["ndvi", "ndbi", "albedo"]

fits = {"r0": load_r0_fit()}
for name in ["orig", "rep1", "rep2", "rep3", "rep4", "rep5", "n2000", "n6000", "aug"]:
    p = CACHE / f"mgwr_{name}.npz"
    if p.exists():
        fits[name] = dict(np.load(p))
print("fits available:", list(fits))
stored = gpd.read_file(MGWR_SAMPLE)
fo = fits["r0"]
rep = {"refit_bw": fits["orig"]["bws"].tolist(), "refit_R2": fits["orig"]["R2"], "refit_AICc": fits["orig"]["aicc"]}
print("orig reproduction:", rep)
# kernel-consistent estimator reproduces fitted coefficients at calibration points
kc = local_betas(fo, fo["coords"])
kc_err = float(np.abs(kc - fo["params"]).max())
kc_r = min(np.corrcoef(kc[:, j], fo["params"][:, j])[0, 1] for j in range(1, 4))
kc_mad = float(np.median(np.abs(kc - fo["params"])[:, 1:], 0).max())
print("kernel-consistent max abs diff at calibration points:", kc_err)

surf = {k: local_betas(f, axy) for k, f in fits.items()}
base = surf["r0"]
dom = lambda A: np.argmax(np.abs(A[:, 1:4]), 1)
rows = []
for k, f in fits.items():
    w = KNN.from_array(f["coords"], k=8); w.transform = "r"
    mi = Moran(f["resid"], w, permutations=199).I
    A = surf[k]
    bw = f["bws"]
    row = dict(fit=k, n=len(f["coords"]), bw_int=bw[0], bw_ndvi=bw[1], bw_ndbi=bw[2], bw_alb=bw[3],
               R2=f["R2"], adjR2=f["adjR2"], ENP=f["ENP"], AICc=f["aicc"], moran=mi,
               cond_med=np.nanmedian(f["cond"]), cond_max=np.nanmax(f["cond"]))
    for j, n in enumerate(NAMES, start=1):
        row[f"r_{n}"] = np.corrcoef(A[:, j], base[:, j])[0, 1]
        row[f"sign_{n}"] = np.mean(np.sign(A[:, j]) == np.sign(base[:, j]))
        row[f"med_{n}"] = np.median(f["params"][:, j]); row[f"p05_{n}"] = np.percentile(f["params"][:, j], 5)
        row[f"p95_{n}"] = np.percentile(f["params"][:, j], 95)
    row["dom_agree"] = np.mean(dom(A) == dom(base))
    rows.append(row)
T = pd.DataFrame(rows)
T.to_csv(TABLES / "mgwr_stability.csv", index=False)
pd.set_option("display.width", 250)
print(T.round(3).to_string())

# LCZ-class medians of the atlas coefficients per fit
med = []
for k, A in surf.items():
    for c in [3, 6, 8, 9, 11, 12, 13, 14]:
        m = lcz_a == c
        med.append(dict(fit=k, lcz=c, **{n: np.median(A[m, j]) for j, n in enumerate(NAMES, start=1)}))
MED = pd.DataFrame(med)
MED.to_csv(TABLES / "mgwr_stability_lcz_medians.csv", index=False)
print(MED.pivot(index="lcz", columns="fit", values="ndbi").round(3))

reps = T[T.fit.str.startswith("rep")]
refit = T.set_index("fit").loc["orig"]
sizes = T[T.fit.isin(["n2000", "n6000"])]
allalt = T[T.fit != "r0"]
vals = {
    "kc_reproduce_r": fmt(kc_r), "kc_reproduce_mad": fmt(kc_mad), "kc_reproduce_err": fmt(kc_err, 2),
    "refit_bw": f"{int(refit.bw_int)}/{int(refit.bw_ndvi)}/{'global' if refit.bw_ndbi >= refit.n - 2 else int(refit.bw_ndbi)}/{int(refit.bw_alb)}",
    "refit_r2": fmt(refit.R2), "refit_aicc": fmt(refit.AICc, 1), "refit_enp": fmt(refit.ENP, 1),
    "refit_r_ndvi": fmt(refit.r_ndvi, 2), "refit_r_ndbi": fmt(refit.r_ndbi, 2), "refit_r_alb": fmt(refit.r_albedo, 2), "refit_dom": fmt(100 * refit.dom_agree, 1),
    "rep_n": len(reps),
    "rep_r2_min": fmt(reps.R2.min()), "rep_r2_max": fmt(reps.R2.max()),
    "rep_bw_int": f"{int(reps.bw_int.min())}--{int(reps.bw_int.max())}",
    "rep_bw_ndvi": f"{int(reps.bw_ndvi.min())}--{int(reps.bw_ndvi.max())}",
    "rep_bw_ndbi": f"{int(reps.bw_ndbi.min())}--{int(reps.bw_ndbi.max())}",
    "rep_bw_alb": f"{int(reps.bw_alb.min())}--{int(reps.bw_alb.max())}",
    "rep_moran_min": fmt(reps.moran.min()), "rep_moran_max": fmt(reps.moran.max()),
    "rep_r_ndvi_min": fmt(reps.r_ndvi.min(), 2), "rep_r_ndbi_min": fmt(reps.r_ndbi.min(), 2), "rep_r_alb_min": fmt(reps.r_albedo.min(), 2),
    "rep_r_ndvi_med": fmt(reps.r_ndvi.median(), 2), "rep_r_ndbi_med": fmt(reps.r_ndbi.median(), 2), "rep_r_alb_med": fmt(reps.r_albedo.median(), 2),
    "rep_sign_ndbi_min": fmt(100 * reps.sign_ndbi.min(), 1), "rep_sign_alb_min": fmt(100 * reps.sign_albedo.min(), 1),
    "rep_sign_ndvi_min": fmt(100 * reps.sign_ndvi.min(), 1),
    "rep_dom_min": fmt(100 * reps.dom_agree.min(), 1), "rep_dom_med": fmt(100 * reps.dom_agree.median(), 1),
    "rep_cond_max": fmt(reps.cond_max.max(), 2),
}
for k in ["n2000", "n6000", "aug"]:
    if k in set(T.fit):
        r = T.set_index("fit").loc[k]
        vals.update({f"{k}_r2": fmt(r.R2), f"{k}_bw": f"{int(r.bw_int)}/{int(r.bw_ndvi)}/{int(r.bw_ndbi)}/{int(r.bw_alb)}",
                     f"{k}_r_ndvi": fmt(r.r_ndvi, 2), f"{k}_r_ndbi": fmt(r.r_ndbi, 2), f"{k}_r_alb": fmt(r.r_albedo, 2),
                     f"{k}_dom": fmt(100 * r.dom_agree, 1), f"{k}_moran": fmt(r.moran), f"{k}_enp": fmt(r.ENP, 1),
                     f"{k}_cond_max": fmt(r.cond_max, 2)})
if "aug" in fits:
    fa = fits["aug"]
    vals["aug_bw_elev"] = int(fa["bws"][4]); vals["aug_bw_tch"] = int(fa["bws"][5])
    vals["aug_elev_med"] = fmt(np.median(fa["params"][:, 4]), 3); vals["aug_tch_med"] = fmt(np.median(fa["params"][:, 5]), 3)
    vals["aug_aicc"] = fmt(fa["aicc"], 1)
write_values("r1_04_mgwr_stability", vals)
np.savez_compressed(CACHE / "atlas_surfaces.npz", **{k: v.astype(np.float32) for k, v in surf.items()}, lcz=lcz_a, xy=axy)
