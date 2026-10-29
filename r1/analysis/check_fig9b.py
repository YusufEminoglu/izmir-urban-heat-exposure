"""Check that Figure 9b draws the hold-out predictions whose R2/RMSE it reports (r1_06), and describe the residuals."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
from mgwr_core import load_r0_fit

V = dict(l.strip().partition(",")[::2] for l in open(VALUES / "r1_06_mgwr_validation.csv", encoding="utf-8-sig"))
f = load_r0_fit()
g = load_grid()
B = np.load(CACHE / "betas_fullgrid_r0.npy").astype(float)
Xs = (g[PREDICTORS].to_numpy(float) - f["xm"]) / f["xs"]
pred = (np.column_stack([np.ones(len(Xs)), Xs]) * B).sum(1) * f["ys"] + f["ym"]
obs = g[DEP].to_numpy(float)
key = g.row.to_numpy(np.int64) * 10000 + g.col.to_numpy(np.int64)
cal = set((f["rows"].astype(np.int64) * 10000 + f["cols"].astype(np.int64)).tolist())
ho = ~np.isin(key, list(cal))
e = obs[ho] - pred[ho]
r2 = 1 - (e ** 2).sum() / ((obs[ho] - obs[ho].mean()) ** 2).sum()
print("hold-out n", ho.sum(), "| reported", V.get("ho_grid_holdout_MGWR_kernel_r2"), V.get("ho_grid_holdout_MGWR_kernel_rmse"))
print("recomputed R2 %.3f RMSE %.2f bias %.3f" % (r2, np.sqrt((e ** 2).mean()), e.mean()))
print("pred range %.1f-%.1f  obs range %.1f-%.1f" % (pred[ho].min(), pred[ho].max(), obs[ho].min(), obs[ho].max()))
lo, hi = np.percentile(obs[ho], [0.2, 99.8])
print("axis range", round(lo, 1), round(hi, 1), "| share of predictions outside it %.2f%%" % (100 * np.mean((pred[ho] < lo) | (pred[ho] > hi))))
sl, ic = np.polyfit(obs[ho], pred[ho], 1)
print("regression of predicted on observed: slope %.3f intercept %.2f" % (sl, ic))
for q in (1, 5, 50, 95, 99):
    print("abs error p%d %.2f" % (q, np.percentile(np.abs(e), q)))
