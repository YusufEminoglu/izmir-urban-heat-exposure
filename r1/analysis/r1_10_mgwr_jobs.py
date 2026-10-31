"""Run every MGWR fit the revision needs, in parallel worker processes.

Jobs
  orig            the reported 4,000-cell sample, refit (reproducibility check)
  rep1..rep5      five new stratified 4,000-cell samples (LCZ x 10 km tile), seeds 101-105
  n2000, n6000    sample-size variants of the same design
  aug             reported sample + elevation + tree canopy height as local covariates
  blk_f0..f4      spatial block cross-validation: 10 km blocks, 5 folds, refit per fold
  rnd_f0..f4      random 5-fold cross-validation, refit per fold
Each job stores outputs/cache/mgwr_<job>.npz.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, json, time, traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

THREADS = 1


def job_list():
    jobs = ["orig"] + [f"blk_f{i}" for i in range(5)] + [f"rep{i}" for i in range(1, 6)] \
        + ["aug", "n2000", "n6000"] + [f"rnd_f{i}" for i in range(5)]
    return jobs


def get_data(job):
    import geopandas as gpd
    from common import load_grid, MGWR_SAMPLE, PREDICTORS, DEP
    from mgwr_core import stratified_sample
    preds = list(PREDICTORS)
    s = gpd.read_file(MGWR_SAMPLE)
    s = s[["row", "col", "x", "y", "lcz", DEP] + PREDICTORS + ["srtm_elevation_m", "tree_canopy_height_2025"]].copy()
    test_idx = None
    if job == "orig":
        d = s
    elif job == "aug":
        d = s
        preds = preds + ["srtm_elevation_m", "tree_canopy_height_2025"]
    elif job.startswith("blk_f") or job.startswith("rnd_f"):
        f = int(job[-1])
        rng = np.random.default_rng(7)
        if job.startswith("blk"):
            bx = np.floor(s["x"].to_numpy() / 10_000).astype(int)
            by = np.floor(s["y"].to_numpy() / 10_000).astype(int)
            key = bx * 10_000 + by
            ub = np.unique(key)
            fold_of_block = dict(zip(ub, rng.permutation(len(ub)) % 5))
            fold = np.array([fold_of_block[k] for k in key])
        else:
            fold = rng.permutation(len(s)) % 5
        test_idx = np.where(fold == f)[0]
        d = s.iloc[np.where(fold != f)[0]].reset_index(drop=True)
    else:
        g = load_grid()
        key = set(zip(s["row"], s["col"]))
        mask = np.array([(r, c) not in key for r, c in zip(g["row"], g["col"])])
        g = g[mask].reset_index(drop=True)
        if job.startswith("rep"):
            d = stratified_sample(g, 4000, seed=100 + int(job[3:]))
        elif job == "n2000":
            d = stratified_sample(g, 2000, seed=201)
        elif job == "n6000":
            d = stratified_sample(g, 6000, seed=202)
        d = d[["row", "col", "x", "y", "lcz", DEP] + PREDICTORS + ["srtm_elevation_m", "tree_canopy_height_2025"]]
    return d, preds, test_idx


def run(job):
    import os
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = str(THREADS)
    from common import CACHE, DEP
    from mgwr_core import fit_mgwr
    out = CACHE / f"mgwr_{job}.npz"
    if out.exists():
        return job, "cached"
    t0 = time.time()
    try:
        d, preds, test_idx = get_data(job)
        bwmin = [2, 2, 2, 2, 50, 50] if job == "aug" else None
        fit = fit_mgwr(d[["x", "y"]].to_numpy(float), d[DEP].to_numpy(float), d[preds].to_numpy(float), bw_min=bwmin)
        np.savez_compressed(out, **{k: v for k, v in fit.items()},
                            rows=d["row"].to_numpy(), cols=d["col"].to_numpy(), lcz=d["lcz"].to_numpy(),
                            preds=np.array(preds), test_idx=(test_idx if test_idx is not None else np.array([], int)))
        msg = f"n={len(d)} bws={fit['bws'].tolist()} R2={fit['R2']:.4f} ENP={fit['ENP']:.1f} AICc={fit['aicc']:.1f} t={time.time()-t0:.0f}s"
    except Exception:
        msg = "FAILED " + traceback.format_exc()
    with open(CACHE / "mgwr_jobs.log", "a") as f:
        f.write(f"{job}: {msg}\n")
    return job, msg


if __name__ == "__main__":
    from multiprocessing import Pool
    jobs = sys.argv[1:] or job_list()
    workers = 8
    with Pool(workers, maxtasksperchild=1) as pool:
        for job, msg in pool.imap_unordered(run, jobs):
            print(job, msg, flush=True)
