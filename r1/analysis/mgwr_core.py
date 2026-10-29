"""MGWR fitting and kernel-consistent local estimation used by the R1 analyses.

`fit_mgwr` reproduces the settings of the submitted model (adaptive bisquare,
AICc golden-section back-fitting, standardised y and X, tol 1e-4).

`local_betas` estimates MGWR coefficients at arbitrary locations from a fitted
model without spatial interpolation: for every covariate k it applies the final
back-fitting update of MGWR at the new location, i.e. a single-covariate local
regression of the partial residual r_k = y - sum_{j!=k} X_j beta_j on X_k, with
the adaptive bisquare kernel of bandwidth b_k (the distance to the b_k-th
nearest calibration point). At calibration points this reproduces the fitted
coefficients (checked in r1_10), so it is the model's own estimator rather than
an interpolation of its output.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.spatial import cKDTree


def stratified_sample(df, n, seed, tile=10_000.0, col_lcz="lcz"):
    """Proportional allocation over LCZ x tile strata (largest remainder), seeded."""
    rng = np.random.default_rng(seed)
    tx = np.floor(df["x"].to_numpy() / tile).astype(int)
    ty = np.floor(df["y"].to_numpy() / tile).astype(int)
    key = df[col_lcz].astype(int).astype(str).to_numpy() + "_" + tx.astype(str) + "_" + ty.astype(str)
    uk, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    quota = counts / counts.sum() * n
    base = np.floor(quota).astype(int)
    rem = n - base.sum()
    order = np.argsort(-(quota - base) + rng.random(len(quota)) * 1e-9)
    base[order[:rem]] += 1
    idx = []
    by = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    for s, c, q in zip(starts, counts, base):
        if q > 0:
            idx.append(rng.choice(by[s:s + c], size=q, replace=False))
    idx = np.concatenate(idx)
    return df.iloc[np.sort(idx)].reset_index(drop=True)


def fit_mgwr(coords, y, X, verbose=False, bw_min=None):
    from mgwr.gwr import MGWR
    from mgwr.sel_bw import Sel_BW
    ym, ys = y.mean(), y.std(ddof=0)
    xm, xs = X.mean(0), X.std(0, ddof=0)
    Ys = ((y - ym) / ys).reshape(-1, 1)
    Xs = (X - xm) / xs
    t0 = time.time()
    sel = Sel_BW(coords, Ys, Xs, kernel="bisquare", fixed=False, multi=True, n_jobs=1)
    bws = sel.search(criterion="AICc", multi_bw_min=bw_min or [2], init_multi=(200 if bw_min else None), search_method="golden_section",
                     tol_multi=1e-4, max_iter_multi=200, verbose=verbose)
    res = MGWR(coords, Ys, Xs, sel, kernel="bisquare", fixed=False, sigma2_v1=True, n_jobs=1).fit()
    t1 = time.time() - t0
    params = np.asarray(res.params)
    Z = np.column_stack([np.ones(len(Xs)), Xs])
    resid = Ys.ravel() - (Z * params).sum(1)
    out = dict(coords=coords, ym=ym, ys=ys, xm=xm, xs=xs, Z=Z, yS=Ys.ravel(), params=params,
               bws=np.asarray(bws, float), resid=resid, R2=float(res.R2), adjR2=float(res.adj_R2),
               aicc=float(res.aicc), aic=float(res.aic), ENP=float(res.ENP),
               llf=float(res.llf), seconds=t1)
    try:
        lc = res.local_collinearity()
        out["cond"] = np.asarray(lc[0]).ravel()
    except Exception:  # pragma: no cover
        out["cond"] = np.full(len(Xs), np.nan)
    return out


def local_betas(fit, new_coords, chunk=200_000):
    """Kernel-consistent MGWR coefficients (standardised scale) at new locations."""
    coords, Z, params, bws = fit["coords"], fit["Z"], fit["params"], fit["bws"]
    resid = fit["resid"]
    tree = cKDTree(coords)
    K = Z.shape[1]
    out = np.empty((len(new_coords), K))
    kmax = int(bws.max())
    chunk = max(500, min(chunk, int(1.5e7 // kmax)))
    for s in range(0, len(new_coords), chunk):
        nc = new_coords[s:s + chunk]
        d, ii = tree.query(nc, k=kmax)
        for k in range(K):
            b = int(bws[k])
            dk, ik = d[:, :b], ii[:, :b]
            h = dk[:, -1:] * 1.0000001
            w = (1 - (dk / h) ** 2) ** 2
            xk = Z[ik, k]
            rk = resid[ik] + Z[ik, k] * params[ik, k]  # partial residual for covariate k
            out[s:s + chunk, k] = (w * xk * rk).sum(1) / (w * xk * xk).sum(1)
    return out


def predict(fit, new_coords, Xnew, betas=None):
    """Predicted y (original units) at new locations from kernel-consistent betas."""
    if betas is None:
        betas = local_betas(fit, new_coords)
    Xs = (Xnew - fit["xm"]) / fit["xs"]
    Zn = np.column_stack([np.ones(len(Xs)), Xs])
    return (Zn * betas).sum(1) * fit["ys"] + fit["ym"], betas


def idw(src_xy, src_val, dst_xy, k=12, power=2.0, min_d=1.0):
    tree = cKDTree(src_xy)
    d, ii = tree.query(dst_xy, k=k)
    if k == 1:
        return src_val[ii]
    d = np.maximum(d, min_d)
    w = d ** (-power)
    v = src_val[ii]
    if v.ndim == 3:
        return (w[..., None] * v).sum(1) / w.sum(1)[:, None]
    return (w * v).sum(1) / w.sum(1)


def load_r0_fit():
    """The reported MGWR as a fit dict. v2 (documented 2025 layers): the r2_28 fit of the calibration sample,
    outputs/cache/mgwr_orig.npz. The v1 branch below rebuilt the earlier reported fit from its stored sample."""
    from common import CACHE
    _v2 = CACHE / "mgwr_orig.npz"
    if _v2.exists():
        z = np.load(_v2, allow_pickle=True)
        f = {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}
        f["stored_resid"] = f["resid"]
        return f
    import geopandas as gpd
    from common import MGWR_SAMPLE, PREDICTORS, DEP
    s = gpd.read_file(MGWR_SAMPLE)
    coords = s[["x", "y"]].to_numpy(float)
    y = s[DEP].to_numpy(float)
    X = s[PREDICTORS].to_numpy(float)
    ym, ys = y.mean(), y.std(ddof=0)
    xm, xs = X.mean(0), X.std(0, ddof=0)
    Z = np.column_stack([np.ones(len(X)), (X - xm) / xs])
    params = np.column_stack([s["mgwr_intercept"]] + [s[f"mgwr_coef_{p}"] for p in PREDICTORS]).astype(float)
    yS = (y - ym) / ys
    resid = yS - (Z * params).sum(1)
    n = len(y)
    s2 = (resid ** 2).sum() / n
    return dict(coords=coords, ym=ym, ys=ys, xm=xm, xs=xs, Z=Z, yS=yS, params=params,
                bws=np.array([17.0, 64.0, 140.0, 97.0]), resid=resid, R2=0.8801787941147169, adjR2=0.8461195511633072,
                aicc=5141.794708110976, aic=4636.73311791551, ENP=885.1216226125914,
                llf=-n / 2 * (np.log(2 * np.pi * s2) + 1), cond=s["mgwr_cond"].to_numpy(float),
                stored_resid=s["mgwr_residual_scaled"].to_numpy(float), lcz=s["lcz"].to_numpy(),
                rows=s["row"].to_numpy(), cols=s["col"].to_numpy())
