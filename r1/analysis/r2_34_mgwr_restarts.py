"""Restarts of the MGWR bandwidth search on the reported calibration sample: the same data and
settings as the reported fit, with the back-fitting search started from a global (4,000) and a local (50) bandwidth
for every term (r2_28_run_jobs_remote.py, jobs orig_init4000 and orig_init50)."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

v = {}
for tag, job in [("glob", "orig_init4000"), ("loc", "orig_init50")]:
    f = np.load(CACHE / f"mgwr_{job}.npz")
    v[f"restart_{tag}_bw"] = "/".join(str(int(b)) for b in f["bws"])
    v[f"restart_{tag}_aicc"] = fmt(float(f["aicc"]), 1)
    v[f"restart_{tag}_r2"] = fmt(float(f["R2"]), 3)
print(v)
write_values("r2_34_mgwr_restarts", v)
