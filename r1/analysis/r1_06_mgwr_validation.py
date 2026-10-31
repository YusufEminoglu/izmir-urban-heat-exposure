"""Out-of-sample performance of the MGWR and the global benchmarks.

A. Spatial block cross-validation: 10 km blocks of the reported sample are
   split into five folds; MGWR is re-calibrated on four folds (bandwidths
   re-selected) and predicts the held-out fold with its kernel-consistent
   estimator. Two test sets per fold: the held-out sample cells, and every
   100 m cell of the full grid that lies in the held-out blocks.
B. Random 5-fold cross-validation of the same sample (re-calibrated per fold).
C. Hold-out of the full grid: the reported model (orig) predicts all
   1.18 M cells that were not in its sample, with the kernel-consistent
   estimator and with the published IDW atlas.
Benchmarks fitted on exactly the same training cells: OLS with the three
surface predictors, OLS with the full global specification (terrain, canopy,
LCZ probability and LCZ dummies), and the ML spatial error model's trend X b.
Metrics: R2 = 1 - SSE/SST on the test cells, RMSE and MAE in degrees C, and
Moran's I of the held-out residuals.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import KNN
from esda.moran import Moran
import spreg

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
from mgwr_core import predict, idw, load_r0_fit

s = gpd.read_file(MGWR_SAMPLE)
g = load_grid()
LCZS = sorted(g.lcz.unique())


def design(df, full):
    X = df[PREDICTORS].to_numpy(float)
    if full:
        extra = df[[c for c in OLS_CONT if c not in PREDICTORS]].to_numpy(float)
        D = np.column_stack([(df.lcz.to_numpy() == k).astype(float) for k in LCZS if k != 14])
        X = np.column_stack([X, extra, D])
    return np.column_stack([np.ones(len(X)), X])


def ols_fit(tr, full):
    A = design(tr, full)
    b, *_ = np.linalg.lstsq(A, tr[DEP].to_numpy(float), rcond=None)
    return lambda te: design(te, full) @ b


def sem_fit(tr):
    y = tr[DEP].to_numpy(float).reshape(-1, 1)
    X = tr[PREDICTORS].to_numpy(float)
    w = KNN.from_array(tr[["x", "y"]].to_numpy(float), k=8); w.transform = "r"
    m = spreg.ML_Error(y, X, w=w, method="full")
    b = m.betas[:-1].ravel()
    return lambda te: np.column_stack([np.ones(len(te)), te[PREDICTORS].to_numpy(float)]) @ b


def metrics(y, p):
    e = y - p
    return 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(), np.sqrt((e ** 2).mean()), np.abs(e).mean()


def moran_te(df, e):
    w = KNN.from_array(df[["x", "y"]].to_numpy(float), k=8); w.transform = "r"
    return Moran(e, w, permutations=199).I


bx = np.floor(s.x / 10_000).astype(int); by = np.floor(s.y / 10_000).astype(int)
s["blk"] = bx * 10_000 + by
g["blk"] = np.floor(g.x / 10_000).astype(int) * 10_000 + np.floor(g.y / 10_000).astype(int)
skey = set(zip(s.row, s.col))
g_out = g[[(r, c) not in skey for r, c in zip(g.row, g.col)]].reset_index(drop=True)

rows = []
store = {}
for scheme in ["blk", "rnd"]:
    for fo in range(5):
        p = CACHE / f"mgwr_{scheme}_f{fo}.npz"
        if not p.exists():
            print("missing", p.name); continue
        f = dict(np.load(p))
        te_idx = f["test_idx"]
        te = s.iloc[te_idx].reset_index(drop=True)
        tr = s.drop(index=te_idx).reset_index(drop=True)
        assert len(tr) == len(f["coords"])
        sets = {"sample": te}
        if scheme == "blk":
            sets["grid"] = g_out[g_out.blk.isin(set(te.blk))].reset_index(drop=True)
        models = {"MGWR": lambda d: predict(f, d[["x", "y"]].to_numpy(float), d[PREDICTORS].to_numpy(float))[0],
                  "OLS3": ols_fit(tr, False), "OLSfull": ols_fit(tr, True), "SEMtrend": sem_fit(tr)}
        for setname, d in sets.items():
            y = d[DEP].to_numpy(float)
            for mname, fn in models.items():
                pr = fn(d)
                r2, rmse, mae = metrics(y, pr)
                rec = dict(scheme=scheme, fold=fo, set=setname, model=mname, n=len(d), r2=r2, rmse=rmse, mae=mae)
                if setname == "sample":
                    rec["moran"] = moran_te(d, y - pr)
                rows.append(rec)
                store.setdefault((scheme, setname, mname), []).append((y, pr))
CV = pd.DataFrame(rows)
CV.to_csv(TABLES / "cv_folds.csv", index=False)
# pooled metrics across folds
pool = []
for (scheme, setname, mname), lst in store.items():
    y = np.concatenate([a for a, _ in lst]); p = np.concatenate([b for _, b in lst])
    r2, rmse, mae = metrics(y, p)
    mor = CV[(CV.scheme == scheme) & (CV.set == setname) & (CV.model == mname)]["moran"].mean() if setname == "sample" else np.nan
    pool.append(dict(scheme=scheme, set=setname, model=mname, n=len(y), r2=r2, rmse=rmse, mae=mae, moran_mean=mor))
PO = pd.DataFrame(pool)
print(PO.round(3).to_string())

# C. full-grid hold-out with the reported model
fo = load_r0_fit()
y = g_out[DEP].to_numpy(float)
pm, B = predict(fo, g_out[["x", "y"]].to_numpy(float), g_out[PREDICTORS].to_numpy(float))
Bs = fo["params"]
Bi = idw(fo["coords"], Bs, g_out[["x", "y"]].to_numpy(float), 12, 2)
Xs = (g_out[PREDICTORS].to_numpy(float) - fo["xm"]) / fo["xs"]
pi = (np.column_stack([np.ones(len(Xs)), Xs]) * Bi).sum(1) * fo["ys"] + fo["ym"]
hold = []
for name, pr in [("MGWR_kernel", pm), ("MGWR_idw_atlas", pi), ("OLS3", ols_fit(s, False)(g_out)), ("OLSfull", ols_fit(s, True)(g_out)),
                 ("SEMtrend", sem_fit(s)(g_out))]:
    r2, rmse, mae = metrics(y, pr)
    hold.append(dict(set="grid_holdout", model=name, n=len(y), r2=r2, rmse=rmse, mae=mae))
# by distance to the nearest sample cell
from scipy.spatial import cKDTree
dist, _ = cKDTree(s[["x", "y"]].to_numpy(float)).query(g_out[["x", "y"]].to_numpy(float))
for lo, hi in [(0, 500), (500, 1000), (1000, 2000), (2000, 1e9)]:
    m = (dist >= lo) & (dist < hi)
    r2, rmse, _ = metrics(y[m], pm[m])
    hold.append(dict(set=f"dist_{int(lo)}_{int(min(hi, 99999))}", model="MGWR_kernel", n=int(m.sum()), r2=r2, rmse=rmse))
HO = pd.DataFrame(hold)
print(HO.round(3).to_string())
pd.concat([PO, HO]).to_csv(TABLES / "validation_summary.csv", index=False)

P = PO.set_index(["scheme", "set", "model"])
H = HO.set_index(["set", "model"])
vals = {}
for (sc, st, m), r in P.iterrows():
    k = f"cv_{sc}_{st}_{m}"
    vals[k + "_r2"] = fmt(r.r2); vals[k + "_rmse"] = fmt(r.rmse, 2); vals[k + "_mae"] = fmt(r.mae, 2)
    if st == "sample":
        vals[k + "_moran"] = fmt(r.moran_mean)
    vals[k + "_n"] = fmt_int(r.n)
for (st, m), r in H.iterrows():
    k = f"ho_{st}_{m}"
    vals[k + "_r2"] = fmt(r.r2); vals[k + "_rmse"] = fmt(r.rmse, 2)
    vals[k + "_n"] = fmt_int(r.n)
blk = CV[(CV.scheme == "blk") & (CV.model == "MGWR") & (CV.set == "sample")]
vals["cv_blk_mgwr_r2_foldmin"] = fmt(blk.r2.min()); vals["cv_blk_mgwr_r2_foldmax"] = fmt(blk.r2.max())
vals["n_grid_holdout"] = fmt_int(len(g_out))
vals["share_within_2km"] = fmt(100 * np.mean(dist < 2000), 1)
vals["dist_median_holdout"] = int(round(np.median(dist)))
write_values("r1_06_mgwr_validation", vals)
