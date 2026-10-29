"""Local inference for the reported MGWR (bandwidths 17/64/140/97, intercept/NDVI/NDBI/albedo).

The multiple-testing-adjusted critical t of da Silva and Fotheringham (2016) depends on the
covariate-specific effective number of parameters ENP_j, which the original run did not store.
ENP_j is obtained from a back-fitting refit with the bandwidths fixed at the reported values (same
data, kernel and options); the shares of significant locations are computed from the local t-values
stored with the reported fit.
"""
import os
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(k, "4")
import sys
from pathlib import Path
import numpy as np
import geopandas as gpd
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import MGWR
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

s = gpd.read_file(MGWR_SAMPLE)
P = ["ndvi_2025", "ndbi_2025", "albedo_2025"]
coords = s[["x", "y"]].to_numpy(float)
ys = StandardScaler().fit_transform(s[["lst_2025"]].to_numpy(float))
Xs = StandardScaler().fit_transform(s[P].to_numpy(float))
from mgwr_core import load_r0_fit
bws = [int(b) for b in load_r0_fit()["bws"]]
_f = load_r0_fit()
if "tcrit" in _f:  # v2: the reported fit stores ENP_j and the adjusted critical t
    class _R: pass
    res = _R(); res.ENP_j = _f["ENP_j"]; crit = _f["tcrit"]
else:
    sel = Sel_BW(coords, ys, Xs, multi=True, kernel="bisquare", fixed=False, constant=True)
    sel.search(multi_bw_min=bws, multi_bw_max=bws, verbose=False)
    res = MGWR(coords, ys, Xs, sel, kernel="bisquare", fixed=False, sigma2_v1=True, n_jobs=1).fit()
    stored = s[["mgwr_intercept", "mgwr_coef_ndvi_2025", "mgwr_coef_ndbi_2025", "mgwr_coef_albedo_2025"]].to_numpy(float)
    print("refit bandwidths", sel.bw[0], "max |params - stored|", np.abs(res.params - stored).max(), "R2", res.R2)
    crit = res.critical_tval()
v = {}
for j, (nm, col) in enumerate([("ndvi", "ndvi_2025"), ("ndbi", "ndbi_2025"), ("alb", "albedo_2025")], start=1):
    t = s[f"mgwr_t_{col}"].to_numpy(float)
    sig = np.abs(t) > crit[j]
    v[f"mgwr_enp_{nm}"] = fmt(res.ENP_j[j], 1)
    v[f"mgwr_tcrit_{nm}"] = fmt(crit[j], 2)
    v[f"mgwr_sig_{nm}"] = fmt(100 * sig.mean(), 1)
    v[f"mgwr_sig_pos_{nm}"] = fmt(100 * (sig & (t > 0)).mean(), 1)
    v[f"mgwr_sig_neg_{nm}"] = fmt(100 * (sig & (t < 0)).mean(), 1)
    print(nm, v[f"mgwr_enp_{nm}"], v[f"mgwr_tcrit_{nm}"], v[f"mgwr_sig_{nm}"], v[f"mgwr_sig_pos_{nm}"], v[f"mgwr_sig_neg_{nm}"])
write_values("r2_14_local_inference", v)
