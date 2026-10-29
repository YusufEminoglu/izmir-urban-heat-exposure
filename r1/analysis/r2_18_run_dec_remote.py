"""Remote run of the intercept/slope decomposition (same computation as scripts/r2_18_intercept_decomposition.py).
Inputs: dec_inputs.npz, r0_fit.npz, optional cached mgwr_dec_*.npz. Output: dec_results.csv."""
import os
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(k, "6")
from pathlib import Path
import time
import numpy as np
from mgwr.sel_bw import Sel_BW
from mgwr.gwr import MGWR
from mgwr_core import local_betas

D = Path(__file__).parent
d = np.load(D / "dec_inputs.npz")
coords, X, y = d["coords"], d["X"], d["y"]
ym, ys = y.mean(), y.std(ddof=0); xm, xs = X.mean(0), X.std(0, ddof=0)
Ys = ((y - ym) / ys).reshape(-1, 1); Xs = (X - xm) / xs
n = len(y)


def fixed_fit(bws, name):
    cp = D / f"mgwr_dec_{name}.npz"
    if cp.exists():
        return dict(np.load(cp))
    t = time.time()
    sel = Sel_BW(coords, Ys, Xs, kernel="bisquare", fixed=False, multi=True, n_jobs=1)
    sel.search(multi_bw_min=bws, multi_bw_max=bws, verbose=False)
    res = MGWR(coords, Ys, Xs, sel, kernel="bisquare", fixed=False, sigma2_v1=True, n_jobs=1).fit()
    Z = np.column_stack([np.ones(n), Xs]); P = np.asarray(res.params)
    out = dict(coords=coords, ym=ym, ys=ys, xm=xm, xs=xs, Z=Z, yS=Ys.ravel(), params=P, resid=Ys.ravel() - (Z * P).sum(1),
               bws=np.asarray(bws, float), R2=float(res.R2), aicc=float(res.aicc))
    np.savez(cp, **out)
    print(name, "fitted in", round(time.time() - t), "s", flush=True)
    return out


fits = {"int_only": fixed_fit([17, n, n, n], "int_only"), "glob_int": fixed_fit([n, 64, 140, 97], "glob_int"), "reported": dict(np.load(D / "r0_fit.npz"))}
hxy, hX, yt = d["hxy"], d["hX"], d["hy"]
rows = ["key,value"]
for k, f in fits.items():
    t = time.time()
    B = local_betas(f, hxy, chunk=20_000)
    Xn = (hX - f["xm"]) / f["xs"]
    pred = (B[:, 0] + (B[:, 1:] * Xn).sum(1)) * f["ys"] + f["ym"]
    r2 = 1 - ((yt - pred) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
    rows += [f"dec_{k}_r2,{r2:.3f}", f"dec_{k}_insample_r2,{float(f['R2']):.3f}"]
    if "aicc" in f:
        rows.append(f"dec_{k}_aicc,{float(f['aicc']):.1f}")
    print(k, "hold-out R2", round(r2, 3), "in-sample", round(float(f["R2"]), 3), round(time.time() - t), "s", flush=True)
(D / "dec_results.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
print("done")
