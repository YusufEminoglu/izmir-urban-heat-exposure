"""Temporal consistency of the LCZ-LST structure across the 13 summers 2014-2026: annual cube slices
(p1_11 method; 2025 and 2026 added in r2_40) with the analysed composite as the 2025 summer.

For every annual JJA LST slice of the cloud-local cube (same 100 m grid):
  * LCZ class means, Kruskal-Wallis eta^2 (same estimator as Table 4)
  * Spearman rank correlation of the 14-type class-mean vector with 2025
  * Kendall's W concordance of class rankings over the 13 summers
  * the 2018 slice (the LCZ map's nominal epoch) reported on its own
Per-cell monotonic trend 2014-2026: Mann-Kendall S/Z/p (tie-free normal
approximation) and Theil-Sen slope, the same estimators as the cube's
temporal-trend module, summarised by LCZ type.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

g = load_grid(valid_only=False).dropna(subset=[DEP, "lcz"]).reset_index(drop=True)
g["lcz"] = g["lcz"].astype(int)
r, c = g["row"].to_numpy(), g["col"].to_numpy()
years, cube = load_summer_series()  # (13, y, x); 2025 = analysed composite
A = cube[:, r, c].T  # cells x years
A = np.where(np.isfinite(A) & (A > -50), A, np.nan)
print("annual cube valid share per year", np.isfinite(A).mean(0).round(4))

sup_types = [2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
supported = [3, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17]  # from r1_01 support rule
lcz = g["lcz"].to_numpy()


def eta2(vals, labels):
    ok = np.isfinite(vals)
    v, l = vals[ok], labels[ok]
    grp = [v[l == k] for k in np.unique(l)]
    H, _ = stats.kruskal(*grp)
    K, N = len(grp), len(v)
    return (H - K + 1) / (N - K)


rows = []
means = {}
all_years = years
for j, yr in enumerate(all_years):
    v = A[:, j]
    e = eta2(v, lcz)
    m = pd.Series(v).groupby(lcz).mean()
    means[yr] = m
    rows.append(dict(year=yr, eta2=e, valid=np.isfinite(v).mean()))
M = pd.DataFrame(means)  # lcz x year
M.to_csv(TABLES / "lcz_mean_lst_by_year.csv")
E = pd.DataFrame(rows)
for yr in all_years:
    E.loc[E.year == yr, "rho_2025_types"] = stats.spearmanr(M.loc[sup_types, yr], M.loc[sup_types, 2025]).statistic
    E.loc[E.year == yr, "rho_2025_supported"] = stats.spearmanr(M.loc[supported, yr], M.loc[supported, 2025]).statistic
E.to_csv(TABLES / "lcz_eta2_by_year.csv", index=False)
print(E.round(3).to_string())

# Kendall's W over the 13 summers (rankings of the 14 LCZ types)
R = M.loc[sup_types].rank(axis=0).to_numpy()  # types x years
m_, n_ = R.shape[1], R.shape[0]
Rs = R.sum(1)
W = 12 * ((Rs - Rs.mean()) ** 2).sum() / (m_ ** 2 * (n_ ** 3 - n_))
Rsup = M.loc[supported].rank(axis=0).to_numpy()
Rs2 = Rsup.sum(1)
W_sup = 12 * ((Rs2 - Rs2.mean()) ** 2).sum() / (Rsup.shape[1] ** 2 * (Rsup.shape[0] ** 3 - Rsup.shape[0]))
print("Kendall W types", W, "supported", W_sup)

# contrasts in 2018
v18 = A[:, years.index(2018)]
s18 = pd.Series(v18).groupby(lcz).mean()
gap23A_2018 = (np.nanmean(v18[np.isin(lcz, [2, 3])]) - np.nanmean(v18[lcz == 11]))
gap8A_2018 = s18[8] - s18[11]
# cell-level agreement between 2018 and 2025 maps
ok = np.isfinite(v18)
rho_cell_18_25 = stats.spearmanr(v18[ok], g[DEP].to_numpy()[ok]).statistic
# multi-year mean of the other summers vs 2025
mm = np.nanmean(A[:, [j for j, y in enumerate(years) if y != 2025]], 1)
ok2 = np.isfinite(mm)
rho_cell_mean_25 = stats.spearmanr(mm[ok2], g[DEP].to_numpy()[ok2]).statistic
e_mean = eta2(mm, lcz)

# Mann-Kendall + Theil-Sen per cell, cells with all 13 summers
full = np.isfinite(A).all(1)
X = A[full]
n = X.shape[1]
iu, ju = np.triu_indices(n, 1)
D = X[:, ju] - X[:, iu]
S = np.sign(D).sum(1)
varS = n * (n - 1) * (2 * n + 5) / 18.0
Z = np.where(S > 0, (S - 1) / np.sqrt(varS), np.where(S < 0, (S + 1) / np.sqrt(varS), 0.0))
P = 2 * (1 - stats.norm.cdf(np.abs(Z)))
slope = np.median(D / (ju - iu), axis=1)
tr = pd.DataFrame(dict(lcz=lcz[full], slope=slope, p=P, Z=Z))
tr["sig_up"] = (tr.p < 0.05) & (tr.slope > 0)
tr["sig_dn"] = (tr.p < 0.05) & (tr.slope < 0)
ts = tr.groupby("lcz").agg(n=("slope", "size"), sen_median=("slope", "median"), sen_q25=("slope", lambda s: s.quantile(.25)),
                           sen_q75=("slope", lambda s: s.quantile(.75)), share_up=("sig_up", "mean"), share_dn=("sig_dn", "mean"))
ts.to_csv(TABLES / "lcz_mk_sen_2014_2026.csv")
print(ts.round(4).to_string())
np.savez_compressed(CACHE / "mk_sen_cells.npz", row=r[full], col=c[full], slope=slope, p=P)

# how typical is 2025? province-mean LST of every summer with the cube's own method (2025 cube-method slice)
dsc = xr.open_dataset(CUBE_ANNUAL_2026)
pm = pd.Series([float(np.nanmean(dsc["lst_jja"].values[j][r, c])) for j in range(len(years))], index=years)
dsc.close()
pm.rename("province_mean_lst").to_csv(TABLES / "province_mean_lst_by_summer.csv")
print(pm.round(2).to_string())

vals = {
    "y2025_rank_warm": str(int(pm.rank(ascending=False)[2025])), "y2025_cube_mean": fmt(pm[2025], 1),
    "summers_mean": fmt(pm.mean(), 1), "summers_min": fmt(pm.min(), 1), "summers_max": fmt(pm.max(), 1),
    "y2025_dev_sd": fmt((pm[2025] - pm.mean()) / pm.std(ddof=1), 2), "y2026_cube_mean": fmt(pm[2026], 1),
    "cube_years": f"{years[0]}--{years[-1]}",
    "eta2_2018": fmt(E.loc[E.year == 2018, "eta2"].item()),
    "eta2_min_year": fmt(E.loc[E.year != 2025, "eta2"].min()), "eta2_max_year": fmt(E.loc[E.year != 2025, "eta2"].max()),
    "eta2_mean_2014_2024": fmt(e_mean),
    "rho_types_2018_2025": fmt(E.loc[E.year == 2018, "rho_2025_types"].item()),
    "rho_types_min": fmt(E.loc[E.year != 2025, "rho_2025_types"].min()),
    "rho_sup_min": fmt(E.loc[E.year != 2025, "rho_2025_supported"].min()),
    "n_summers": str(len(years)), "n_cube_composites": str(len(years) - 1),
    "kendall_w_types": fmt(W), "kendall_w_supported": fmt(W_sup),
    "gap_c23_A_2018": fmt(gap23A_2018, 2), "gap_c8_A_2018": fmt(gap8A_2018, 2),
    "rho_cell_2018_2025": fmt(rho_cell_18_25), "rho_cell_mean_2025": fmt(rho_cell_mean_25),
    "mk_cells": fmt_int(full.sum()), "mk_share_up": fmt(100 * tr.sig_up.mean(), 1), "mk_share_dn": fmt(100 * tr.sig_dn.mean(), 1),
    "sen_median_all": fmt(tr.slope.median(), 3),
}
for k in [3, 6, 8, 9, 11, 14]:
    vals[f"sen_lcz{k}"] = fmt(ts.loc[k, "sen_median"], 3)
    vals[f"mkup_lcz{k}"] = fmt(100 * ts.loc[k, "share_up"], 1)
write_values("r1_02_lcz_temporal", vals)
