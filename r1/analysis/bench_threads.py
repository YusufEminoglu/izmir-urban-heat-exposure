import os, sys
T = sys.argv[1]
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = T
import time
import numpy as np
import geopandas as gpd
from mgwr.gwr import GWR
sys.path.insert(0, os.path.dirname(__file__))
from common import MGWR_SAMPLE, PREDICTORS, DEP
s = gpd.read_file(MGWR_SAMPLE)
c = s[["x", "y"]].to_numpy(float)
y = s[[DEP]].to_numpy(float); y = (y - y.mean()) / y.std()
X = s[PREDICTORS].to_numpy(float); X = (X - X.mean(0)) / X.std(0)
t = time.time()
for bw in (60, 120, 300):
    GWR(c, y, X, bw, kernel="bisquare", fixed=False, n_jobs=1).fit(lite=True)
print("threads", T, "3 lite fits s", round(time.time() - t, 2))
