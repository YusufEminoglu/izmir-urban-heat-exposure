"""Mahalle priority index (R1-6, R2-5): revised morphology term, coverage, and dependence on construction.

Revised baseline: M_m = mahalle mean of C_i = sum_k beta_k(i) x~_k(i) over every 100 m cell,
with kernel-consistent MGWR coefficients (Eq. mahalle_morph). Weights 0.38/0.30/0.22/0.10.
Alternatives compared with the baseline (Spearman rho; overlap of top 10 / 50 / 80):
  r0_coef   coefficient-magnitude term of the submitted index (Eq. mahalle_morph_r0)
  equal     equal weights
  hev       equal hazard / sensitivity / exposure weights (T, M share the hazard third)
  entropy   entropy weights; pca  first-principal-component weights
  noM       no MGWR term (weights renormalised)
Weight uncertainty: 1000 +/-20% perturbations; rank intervals of the top units.
Writes the revised mahalle table used by Figure 10 and Supplementary Tables S4-S5.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
from mgwr_core import local_betas, load_r0_fit

rng = np.random.default_rng(20)
W0 = np.array([0.38, 0.30, 0.22, 0.10])


def z(v):
    v = np.asarray(v, float)
    return (v - np.nanmean(v)) / np.nanstd(v)


def score(T, M, V, P, w=W0):
    Q = w[0] * z(T) + w[1] * z(M) + w[2] * z(V) + w[3] * z(np.log1p(np.asarray(P, float)))
    return 100 * (Q - np.nanmin(Q)) / (np.nanmax(Q) - np.nanmin(Q))


pri = pd.read_csv(PRIORITY_CSV, encoding="utf-8-sig")
pri = pri[pri.lst_mean.notna()].reset_index(drop=True)
P_r0 = score(pri.lst_mean, pri.morph_pressure, pri.vulnerable_share, pri.pop_total)
print("R0 reproduction max abs diff", np.nanmax(np.abs(P_r0 - pri.priority_score_0_100)))

mah = gpd.read_file(MAHALLE).to_crs(CRS)
g = load_grid()
pts = gpd.GeoDataFrame(g[["row"]], geometry=gpd.points_from_xy(g.x, g.y), crs=CRS)
j = gpd.sjoin(pts, mah[["join_key", "geometry"]], predicate="within", how="left")
j = j[~j.index.duplicated()]
g["join_key"] = j["join_key"].to_numpy()
mah["in_rank"] = mah.join_key.isin(pri.join_key)
area_in = mah.loc[mah.in_rank].area.sum() / mah.area.sum()

wb = pd.read_excel(DATA / "izmir_population_data.xlsx", header=None)
wb_pop = pd.to_numeric(wb.iloc[:, 9], errors="coerce")
wb_rows = wb_pop.notna() & pd.to_numeric(wb.iloc[:, 3], errors="coerce").notna()
pop_wb, n_wb, pop_rank = wb_pop[wb_rows].sum(), int(wb_rows.sum()), pri.pop_total.sum()

f = load_r0_fit()
Bp = CACHE / "betas_fullgrid_r0.npy"
if Bp.exists():
    B = np.load(Bp)
else:
    B = local_betas(f, g[["x", "y"]].to_numpy(float)); np.save(Bp, B.astype(np.float32))
Xs = (g[PREDICTORS].to_numpy(float) - f["xm"]) / f["xs"]
contrib = B[:, 1:] * Xs
g["C"] = contrib.sum(1)
for k, nm in enumerate(["ndvi", "ndbi", "alb"]):
    g[f"C_{nm}"] = contrib[:, k]
agg = g.groupby("join_key")[["C", "C_ndvi", "C_ndbi", "C_alb"]].mean().add_suffix("_area")
pri = pri.merge(agg, left_on="join_key", right_index=True, how="left")

# residential support: the heat terms describe where people live, i.e. the cells of each mahalle that
# contain at least one dwelling of the municipal inventory; a unit without any dwelling uses its cells with
# any building (common.residential_mask)
inv = pd.read_pickle(CACHE / "grid_inventory.pkl")[["row", "col", "bsf", "bsf_res"]]
g = g.merge(inv, on=["row", "col"], how="left")
thr = np.nanpercentile(load_grid(valid_only=False)[DEP].dropna(), 90)
g["res"] = residential_mask(g)
res = g[g.res].groupby("join_key").agg(n_res=(DEP, "size"), lst_res=(DEP, "mean"), hot_res=(DEP, lambda v: np.mean(v >= thr)),
                                           C=("C", "mean"), C_ndvi=("C_ndvi", "mean"), C_ndbi=("C_ndbi", "mean"), C_alb=("C_alb", "mean"))
pri = pri.merge(res, left_on="join_key", right_index=True, how="left")
pri = pri.rename(columns={"lst_mean": "lst_mean_area", "hot_cell_share": "hot_cell_share_area", "hot_exposed_pop_proxy": "hot_pop_area"})
pri["lst_mean"] = pri.lst_res
pri["hot_cell_share"] = pri.hot_res
pri["hot_exposed_pop_proxy"] = pri.pop_total * pri.hot_res
n_res_cells = int(g.res.sum()); n_res_cells_ranked = int(pri.n_res.sum())
print("mahalle without residential cells:", pri.n_res.isna().sum(), "residential cells", n_res_cells, "in ranked units", n_res_cells_ranked)

P0 = score(pri.lst_mean, pri.C, pri.vulnerable_share, pri.pop_total)
pri["P0"] = P0
base_rank = pd.Series(P0).rank(ascending=False, method="first").to_numpy()
rho_C_T = stats.spearmanr(pri.C, pri.lst_mean).statistic
rho_C_r0M = stats.spearmanr(pri.C, pri.morph_pressure).statistic
print("rho(C, LST)", rho_C_T, "rho(C, R0 M)", rho_C_r0M)

alts = {"r0_coef": P_r0}
C4 = lambda M: np.column_stack([z(pri.lst_mean), z(M), z(pri.vulnerable_share), z(np.log1p(pri.pop_total))])
C = C4(pri.C)
alts["equal"] = score(pri.lst_mean, pri.C, pri.vulnerable_share, pri.pop_total, np.full(4, .25))
alts["hev"] = score(pri.lst_mean, pri.C, pri.vulnerable_share, pri.pop_total, np.array([1/6, 1/6, 1/3, 1/3]))
Cn = (C - C.min(0)) / (C.max(0) - C.min(0)) + 1e-12
pr = Cn / Cn.sum(0)
Ent = -(pr * np.log(pr)).sum(0) / np.log(len(Cn))
w_ent = (1 - Ent) / (1 - Ent).sum()
alts["entropy"] = score(pri.lst_mean, pri.C, pri.vulnerable_share, pri.pop_total, w_ent)
ev, evec = np.linalg.eigh(np.corrcoef(C.T))
w_pca = np.abs(evec[:, -1]) / np.abs(evec[:, -1]).sum()
alts["pca"] = score(pri.lst_mean, pri.C, pri.vulnerable_share, pri.pop_total, w_pca)
noM = W0.copy(); noM[1] = 0; noM /= noM.sum()
alts["noM"] = score(pri.lst_mean, pri.C, pri.vulnerable_share, pri.pop_total, noM)
area_alt = score(pri.lst_mean_area, pri.C_area, pri.vulnerable_share, pri.pop_total)  # heat terms over all cells (support check, not in consensus)
noalb_alt = score(pri.lst_mean, pri.C_ndvi + pri.C_ndbi, pri.vulnerable_share, pri.pop_total)  # surface term without its albedo part
print("entropy w", w_ent.round(3), "pca w", w_pca.round(3))


def top(rr, k):
    return set(np.where(rr <= k)[0])


def compare(P):
    r = pd.Series(P).rank(ascending=False, method="first").to_numpy()
    return (stats.spearmanr(P, P0).statistic, len(top(r, 10) & top(base_rank, 10)),
            len(top(r, 50) & top(base_rank, 50)), len(top(r, 80) & top(base_rank, 80)))


rows = [dict(alt=k, rho=a, top10=b, top50=c, top80=d) for k, P in {**alts, "area": area_alt, "noalb": noalb_alt}.items() for a, b, c, d in [compare(P)]]
T = pd.DataFrame(rows)
T.to_csv(TABLES / "priority_alternatives.csv", index=False)
print(T.round(3).to_string())

# consensus: units in the top decile (80) under the baseline and at least 5 of the 6 alternatives
ranks_alt = {k: pd.Series(P).rank(ascending=False, method="first").to_numpy() for k, P in alts.items()}
in_top = np.column_stack([ranks_alt[k] <= 80 for k in alts])
consensus = (base_rank <= 80) & (in_top.sum(1) >= 5)
print("consensus top-decile units", consensus.sum())

Rpm = []
for _ in range(1000):
    w = W0 * rng.uniform(0.8, 1.2, 4); w /= w.sum()
    Rpm.append(score(pri.lst_mean, pri.C, pri.vulnerable_share, pri.pop_total, w))
Rpm = np.array(Rpm)
ranks = np.array([pd.Series(p).rank(ascending=False, method="first").to_numpy() for p in Rpm])
rhos = [stats.spearmanr(p, P0).statistic for p in Rpm]
t10 = [(ranks[i] <= 10)[base_rank <= 10].sum() for i in range(len(Rpm))]
t80 = [(ranks[i] <= 80)[base_rank <= 80].sum() for i in range(len(Rpm))]
print("pm20 rho med/min", np.median(rhos), np.min(rhos), "top10 med/min", np.median(t10), np.min(t10), "top80 med/min", np.median(t80), np.min(t80))

pri["rank"] = base_rank
pri["rank_p05"] = np.percentile(ranks, 5, 0); pri["rank_p95"] = np.percentile(ranks, 95, 0)
pri["consensus_top"] = consensus
pri["P_r0"] = P_r0
for k, P in alts.items():
    pri[f"P_{k}"] = P
pri["P_area"] = area_alt
pri["P_noalb"] = noalb_alt
pri.to_csv(TABLES / "priority_revised_by_mahalle.csv", index=False, encoding="utf-8-sig")

# district aggregation (population-weighted where appropriate)
d = pri.groupby("district_name_final").apply(lambda s: pd.Series({
    "n": len(s), "pop": s.pop_total.sum(), "lst": np.average(s.lst_mean, weights=s.pop_total),
    "C": np.average(s.C, weights=s.pop_total), "hot_pop": s.hot_exposed_pop_proxy.sum(),
    "score": s.P0.mean(), "n_top80": int((s["rank"] <= 80).sum())}), include_groups=False).sort_values("score", ascending=False)
d.to_csv(TABLES / "priority_revised_by_district.csv", encoding="utf-8-sig")
print(d.head(15).round(2).to_string())
top10 = pri.sort_values("rank").head(15)[["district_name_final", "mahalle_name_final", "pop_total", "lst_mean", "C", "vulnerable_share", "hot_exposed_pop_proxy", "P0", "rank_p05", "rank_p95"]]
print(top10.round(2).to_string())

Tt = T.set_index("alt")
vals = {
    "n_mahalle_rank": len(pri), "n_wb_rows": n_wb, "pop_rank": fmt_int(pop_rank), "pop_wb": fmt_int(pop_wb),
    "pop_rank_share": fmt(100 * pop_rank / pop_wb, 2), "rank_area_share": fmt(100 * area_in, 1),
    "w_ent": "/".join(f"{x:.2f}" for x in w_ent), "w_pca": "/".join(f"{x:.2f}" for x in w_pca),
    "priority_rank_rho_median": fmt(np.median(rhos)), "priority_rank_rho_min": fmt(np.min(rhos)),
    "priority_top10_retention_median": int(np.median(t10)), "priority_top10_retention_min": int(np.min(t10)),
    "priority_top80_retention_median": int(np.median(t80)), "priority_top80_retention_min": int(np.min(t80)),
    "rho_C_T": fmt(rho_C_T, 2), "rho_C_r0M": fmt(rho_C_r0M, 2), "n_consensus": int(consensus.sum()),
    "n_consensus_districts": int(pri.loc[consensus, "district_name_final"].nunique()),
}
for k in Tt.index:
    vals[f"alt_{k}_rho"] = fmt(Tt.loc[k, "rho"], 2); vals[f"alt_{k}_top10"] = int(Tt.loc[k, "top10"])
    vals[f"alt_{k}_top50"] = int(Tt.loc[k, "top50"]); vals[f"alt_{k}_top80"] = int(Tt.loc[k, "top80"])
oth = Tt.drop(index=["r0_coef", "area", "noalb"])
vals["alt_rho_min"] = fmt(oth.rho.min(), 2); vals["alt_rho_max"] = fmt(oth.rho.max(), 2)
vals["alt_top80_min"] = int(oth.top80.min()); vals["alt_top80_max"] = int(oth.top80.max())
vals["alt_top10_min"] = int(oth.top10.min()); vals["alt_top10_max"] = int(oth.top10.max())
vals["n_res_cells"] = fmt_int(n_res_cells); vals["n_res_cells_ranked"] = fmt_int(n_res_cells_ranked)
vals["n_res_cells_med"] = int(pri.n_res.median()); vals["n_res_cells_min"] = int(pri.n_res.min())
r_area = pd.Series(area_alt).rank(ascending=False, method="first").to_numpy()
vals["area_top15_zero_hot"] = int(((r_area <= 15) & (pri.hot_pop_area.to_numpy() == 0)).sum())
r_na = pd.Series(noalb_alt).rank(ascending=False, method="first").to_numpy()
bay = (pri.district_name_final == "BAYINDIR").to_numpy()
vals["bay_top15_base"] = int(((base_rank <= 15) & bay).sum()); vals["bay_top15_noalb"] = int(((r_na <= 15) & bay).sum())
vals["n_consensus_in_top15"] = int(((base_rank <= 15) & consensus).sum())
vals["res_top15_zero_hot"] = int(((base_rank <= 15) & (pri.hot_exposed_pop_proxy.to_numpy() == 0)).sum())
write_values("r1_08_priority_alternatives", vals)
