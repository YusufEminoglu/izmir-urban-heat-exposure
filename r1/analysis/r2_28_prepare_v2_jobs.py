"""Inputs of every MGWR fit on the v2 layers (r2_27), bundled for machines without geodata libraries.

* data/paper1_mgwr_sample_v2.gpkg: the same 4,000 calibration cells as before, with LST, NDVI, NDBI and albedo
  from the v2 grid (the coefficient columns are added by r2_29 from the reported v2 fit).
* outputs/cache/v2_jobs/<job>.npz: the job list of r1_10 (reported fit 'orig', five new samples, n2000, augmented
  model, spatial-block and random 5-fold cross-validation) with coordinates, response, predictors and test indices.
"""
import sys
import shutil
from pathlib import Path
import numpy as np
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

s = gpd.read_file(MGWR_SAMPLE_V1)
g = load_grid(valid_only=False)
cols = [DEP] + PREDICTORS + ["srtm_slope_deg", "srtm_aspect_sin", "srtm_aspect_cos"]
s = s.drop(columns=[c for c in s.columns if c.startswith("mgwr_")])
m = s[["row", "col"]].merge(g[["row", "col"] + cols], on=["row", "col"], how="left")
for k in cols:
    s[k] = m[k].to_numpy()
assert s[cols].notna().all().all()
s.to_file(MGWR_SAMPLE, driver="GPKG")
print("sample", len(s), MGWR_SAMPLE.name)

import r1_10_mgwr_jobs as J  # noqa  (reads MGWR_SAMPLE and the v2 grid through common)
OUTD = CACHE / "v2_jobs"; OUTD.mkdir(exist_ok=True)
jobs = [j for j in J.job_list() if j != "n6000"]
for job in jobs:
    d, preds, test_idx = J.get_data(job)
    np.savez_compressed(OUTD / f"{job}.npz", coords=d[["x", "y"]].to_numpy(float), y=d[DEP].to_numpy(float),
                        X=d[preds].to_numpy(float), preds=np.array(preds), rows=d["row"].to_numpy(), cols=d["col"].to_numpy(),
                        lcz=d["lcz"].to_numpy(), test_idx=(test_idx if test_idx is not None else np.array([], int)),
                        bwmin=np.array([2, 2, 2, 2, 50, 50] if job == "aug" else [2]))
    print(job, len(d), preds)
shutil.copy2(Path(__file__).parent / "r2_28_run_jobs_remote.py", OUTD / "run_jobs.py")
