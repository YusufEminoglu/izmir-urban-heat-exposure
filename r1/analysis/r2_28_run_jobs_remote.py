"""Remote MGWR fits on the v2 layers: same settings as mgwr_core.fit_mgwr (adaptive bisquare, golden-section AICc,
minimum bandwidth 2, init 200 when a minimum is set, tolerance 1e-4, at most 200 back-fitting iterations, sigma2_v1).
For the reported fit ('orig') the local standard errors, t-values, covariate-specific ENP and the
da Silva-Fotheringham adjusted critical t are stored as well.

Usage: python run_jobs.py <workers> job1 job2 ...   (inputs <job>.npz in this folder; outputs mgwr_<job>.npz)
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import sys
import time
import traceback
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def run(job):
    from mgwr.gwr import MGWR
    from mgwr.sel_bw import Sel_BW
    out = HERE / f"mgwr_{job}.npz"
    if out.exists():
        return job, "cached"
    t0 = time.time()
    try:
        d = np.load(HERE / f"{job}.npz")
        coords, y, X = d["coords"], d["y"], d["X"]
        bw_min = [int(b) for b in d["bwmin"]]
        bw_min = None if bw_min == [2] else bw_min
        init = int(d["init"]) if "init" in d.files else None  # restart of the search from a given initial bandwidth
        ym, ys = y.mean(), y.std(ddof=0)
        xm, xs = X.mean(0), X.std(0, ddof=0)
        Ys = ((y - ym) / ys).reshape(-1, 1)
        Xs = (X - xm) / xs
        sel = Sel_BW(coords, Ys, Xs, kernel="bisquare", fixed=False, multi=True, n_jobs=1)
        bws = sel.search(criterion="AICc", multi_bw_min=bw_min or [2], init_multi=(init if init else (200 if bw_min else None)), search_method="golden_section",
                         tol_multi=1e-4, max_iter_multi=200, verbose=False)
        res = MGWR(coords, Ys, Xs, sel, kernel="bisquare", fixed=False, sigma2_v1=True, n_jobs=1).fit()
        params = np.asarray(res.params)
        Z = np.column_stack([np.ones(len(Xs)), Xs])
        resid = Ys.ravel() - (Z * params).sum(1)
        o = dict(coords=coords, ym=ym, ys=ys, xm=xm, xs=xs, Z=Z, yS=Ys.ravel(), params=params, bws=np.asarray(bws, float),
                 resid=resid, R2=float(res.R2), adjR2=float(res.adj_R2), aicc=float(res.aicc), aic=float(res.aic),
                 ENP=float(res.ENP), llf=float(res.llf), seconds=time.time() - t0)
        try:
            o["cond"] = np.asarray(res.local_collinearity()[0]).ravel()
        except Exception:
            o["cond"] = np.full(len(Xs), np.nan)
        if job.startswith("orig"):
            o["bse"] = np.asarray(res.bse); o["tvalues"] = np.asarray(res.tvalues)
            o["ENP_j"] = np.asarray(res.ENP_j); o["tcrit"] = np.asarray(res.critical_tval())
        np.savez_compressed(out, **o, rows=d["rows"], cols=d["cols"], lcz=d["lcz"], preds=d["preds"], test_idx=d["test_idx"])
        msg = f"n={len(y)} bws={list(map(int, bws))} R2={o['R2']:.4f} ENP={o['ENP']:.1f} AICc={o['aicc']:.1f} t={time.time() - t0:.0f}s"
    except Exception:
        msg = "FAILED " + traceback.format_exc()
    with open(HERE / "jobs.log", "a") as f:
        f.write(f"{job}: {msg}\n")
    return job, msg


if __name__ == "__main__":
    from multiprocessing import Pool
    workers = int(sys.argv[1])
    jobs = sys.argv[2:]
    with Pool(workers, maxtasksperchild=1) as pool:
        for job, msg in pool.imap_unordered(run, jobs):
            print(job, msg, flush=True)
