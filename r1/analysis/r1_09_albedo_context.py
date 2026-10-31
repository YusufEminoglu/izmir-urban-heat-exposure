"""What the positive albedo-LST association represents.

* albedo by LCZ class and its correlation with NDVI and NDBI
* within-class standardised partial slopes of LST on albedo, NDVI and NDBI
  (OLS inside each LCZ class), so the albedo sign can be read separately for
  built and natural classes
* the MGWR albedo coefficient by LCZ class (sample locations)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

g = load_grid()
rho_alb_ndvi = stats.spearmanr(g.albedo_2025, g.ndvi_2025).statistic
rho_alb_ndbi = stats.spearmanr(g.albedo_2025, g.ndbi_2025).statistic
rows = []
rng = np.random.default_rng(9)


def slopes(s):
    Z = s[["albedo_2025", "ndvi_2025", "ndbi_2025"]].to_numpy()
    Z = (Z - Z.mean(0)) / Z.std(0)
    yy = ((s[DEP] - s[DEP].mean()) / s[DEP].std()).to_numpy()
    A = np.column_stack([np.ones(len(Z)), Z])
    return np.linalg.lstsq(A, yy, rcond=None)[0]


for k, s in g.groupby("lcz"):
    if len(s) < 200:
        continue
    b = slopes(s)
    # 95% interval of the albedo slope from a 2 km spatial block bootstrap (200 resamples of blocks)
    blk = (s.y // 2000).astype(int) * 100000 + (s.x // 2000).astype(int)
    ub = blk.unique(); idx = {u: np.where(blk.to_numpy() == u)[0] for u in ub}
    bb = []
    for _ in range(200):
        pick = rng.choice(ub, len(ub), replace=True)
        ii = np.concatenate([idx[u] for u in pick])
        bb.append(slopes(s.iloc[ii])[1])
    rows.append(dict(lcz=k, label=LCZ_LABEL[k], n=len(s), blocks=len(ub), albedo_med=s.albedo_2025.median(), ndvi_med=s.ndvi_2025.median(),
                     rho_alb_lst=stats.spearmanr(s.albedo_2025, s[DEP]).statistic,
                     b_alb=b[1], b_alb_lo=np.percentile(bb, 2.5), b_alb_hi=np.percentile(bb, 97.5), b_ndvi=b[2], b_ndbi=b[3]))
T = pd.DataFrame(rows)
T.to_csv(TABLES / "albedo_within_lcz.csv", index=False)
print("rho albedo-ndvi", rho_alb_ndvi, "albedo-ndbi", rho_alb_ndbi)
print(T.round(3).to_string())
s = gpd.read_file(MGWR_SAMPLE)
mg = s.groupby("lcz")["mgwr_coef_albedo_2025"].agg(["median", lambda v: (v > 0).mean(), "size"])
print(mg.round(3))
built = T[T.lcz.isin([3, 6, 8, 9])]
vals = {
    "rho_alb_ndvi": fmt(rho_alb_ndvi, 2), "rho_alb_ndbi": fmt(rho_alb_ndbi, 2),
    "alb_pos_share_mgwr": fmt(100 * (s.mgwr_coef_albedo_2025 > 0).mean(), 1),
    "alb_med_lcz8": fmt(T.set_index("lcz").loc[8, "albedo_med"], 3), "alb_med_lczA": fmt(T.set_index("lcz").loc[11, "albedo_med"], 3),
    "alb_med_lczF": fmt(T.set_index("lcz").loc[16, "albedo_med"], 3), "alb_med_lczD": fmt(T.set_index("lcz").loc[14, "albedo_med"], 3),
    "b_alb_lcz8": fmt(T.set_index("lcz").loc[8, "b_alb"], 2), "b_alb_lcz3": fmt(T.set_index("lcz").loc[3, "b_alb"], 2),
    "b_alb_lczD": fmt(T.set_index("lcz").loc[14, "b_alb"], 2), "b_alb_lczF": fmt(T.set_index("lcz").loc[16, "b_alb"], 2),
    "b_alb_lczA": fmt(T.set_index("lcz").loc[11, "b_alb"], 2),
}
Ti = T.set_index("lcz")
for k, nm in [(3, "lcz3"), (8, "lcz8"), (10, "lcz10"), (14, "lczD"), (16, "lczF")]:
    if k in Ti.index:
        vals[f"b_alb_{nm}"] = fmt(Ti.loc[k, "b_alb"], 2)
        vals[f"b_alb_{nm}_lo"] = fmt(Ti.loc[k, "b_alb_lo"], 2); vals[f"b_alb_{nm}_hi"] = fmt(Ti.loc[k, "b_alb_hi"], 2)
        vals[f"n_alb_{nm}"] = f"{int(Ti.loc[k, 'n']):,}".replace(",", "{,}")
write_values("r1_09_albedo_context", vals)
