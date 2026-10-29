"""How much of the MGWR hold-out accuracy comes from the local intercept and how much from the varying slopes?

Two constrained fits on the reported calibration sample (bandwidths fixed, same data and options):
  local intercept only : intercept bandwidth 17, slopes global (bandwidth = n)
  global intercept     : intercept global, slopes at the reported 64 / 140 / 97
Each is evaluated, with the reported model, on the same random 200,000 grid cells outside the sample.
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

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
from mgwr_core import local_betas, load_r0_fit

s = gpd.read_file(MGWR_SAMPLE)
coords = s[["x", "y"]].to_numpy(float)
X = s[PREDICTORS].to_numpy(float); y = s[DEP].to_numpy(float)
ym, ys = y.mean(), y.std(ddof=0); xm, xs = X.mean(0), X.std(0, ddof=0)
Ys = ((y - ym) / ys).reshape(-1, 1); Xs = (X - xm) / xs
n = len(s)


def fixed_fit(bws, name):
    cp = CACHE / f"mgwr_dec_{name}.npz"
    if cp.exists():
        return dict(np.load(cp))
    sel = Sel_BW(coords, Ys, Xs, kernel="bisquare", fixed=False, multi=True, n_jobs=1)
    sel.search(multi_bw_min=bws, multi_bw_max=bws, verbose=False)
    res = MGWR(coords, Ys, Xs, sel, kernel="bisquare", fixed=False, sigma2_v1=True, n_jobs=1).fit()
    Z = np.column_stack([np.ones(n), Xs])
    P = np.asarray(res.params)
    out = dict(coords=coords, ym=ym, ys=ys, xm=xm, xs=xs, Z=Z, yS=Ys.ravel(), params=P, resid=Ys.ravel() - (Z * P).sum(1),
               bws=np.asarray(bws, float), R2=float(res.R2), aicc=float(res.aicc))
    np.savez(cp, **out)
    return out


fits = {"int_only": fixed_fit([17, n, n, n], "int_only"), "glob_int": fixed_fit([n, 64, 140, 97], "glob_int"), "reported": load_r0_fit()}
g = load_grid()
key = g.row.astype(np.int64) * 10000 + g.col
skey = set((s.row.astype(np.int64) * 10000 + s.col).tolist())
ho = np.where(~key.isin(skey).to_numpy())[0]
sub = np.random.default_rng(4).choice(ho, 200_000, replace=False)
gs = g.iloc[sub]
yt = gs[DEP].to_numpy(float)
v = {}
for k, f in fits.items():
    B = local_betas(f, gs[["x", "y"]].to_numpy(float))
    Xn = (gs[PREDICTORS].to_numpy(float) - f["xm"]) / f["xs"]
    pred = (B[:, 0] + (B[:, 1:] * Xn).sum(1)) * f["ys"] + f["ym"]
    r2 = 1 - ((yt - pred) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
    v[f"dec_{k}_r2"] = fmt(r2, 3); v[f"dec_{k}_insample_r2"] = fmt(f["R2"], 3)
    print(k, "in-sample R2", round(f["R2"], 3), "hold-out R2", round(r2, 3), "AICc", round(f.get("aicc", np.nan), 1))
write_values("r2_18_intercept_decomposition", v)
