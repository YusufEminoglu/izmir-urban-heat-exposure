"""Values requested by the internal five-reviewer stress test (2026-10-28).

- grid accounting: province cells, cells with an LCZ class, modelling cells (Table 4 vs Table 5 n)
- MGWR sample share of the modelling grid
- pairwise Mann-Whitney summary (FDR) from the pipeline test table
- heat-justice OLS (descriptive check): n, terms, adjusted R2, demographic coefficients
- conditional LCZ contrast (compact LCZ 3 vs dense trees) from the full OLS dummies
- residents in the top decile of the revised (residential-support) ranking
- mahalle bivariate Moran with residential-cell mean LST, and local bivariate Moran (LISA) clusters
- provincial ADNKS total vs listed residents
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen
from esda.moran import Moran_BV, Moran_Local_BV

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

PIPE = ROOT / "pipeline" / "outputs" / "tables" / "paper1"
v = {}

# grid accounting
g_all = load_grid(valid_only=False)
g_mod = load_grid()
n_lcz = int((g_all.lcz.notna() & g_all[DEP].notna()).sum())
v["n_cells_lcz"] = fmt_int(n_lcz)
v["n_cells_no_lcz"] = int(len(g_all) - n_lcz)
v["n_cells_no_spectral"] = int(n_lcz - len(g_mod))
v["n_cells_excluded_model"] = int(len(g_all) - len(g_mod))
v["mgwr_sample_pct"] = fmt(100 * 4000 / len(g_mod), 2)

# pairwise Mann-Whitney
T = pd.read_csv(PIPE / "paper1_lcz_lst_stat_tests.csv", encoding="utf-8-sig")
pw = T[T.test == "pairwise_mannwhitney"]
v["mw_pairs"] = len(pw)
v["mw_pairs_sig"] = int((pw.p_value_fdr < 0.05).sum())
ns = pw[pw.p_value_fdr >= 0.05]
v["mw_ns_pair"] = "; ".join(f"LCZ {int(a)}/{int(b)}" for a, b in zip(ns.lcz_a, ns.lcz_b)) or "none"

# heat-justice OLS
S = pd.read_csv(PIPE / "paper1_ols_model_summary.csv", encoding="utf-8-sig").set_index("model")
C = pd.read_csv(PIPE / "paper1_ols_coefficients.csv", encoding="utf-8-sig")
hj = S.loc["heat_justice_with_population"]
v["hj_n"] = fmt_int(hj.n); v["hj_adj_r2"] = fmt(hj.adj_r2, 3)
cj = C[C.model == "heat_justice_with_population"].set_index("term")
for t, k in [("elderly_65_plus_share", "eld"), ("child_0_14_share", "child"), ("female_share", "fem"), ("pop_total", "pop")]:
    v[f"hj_b_{k}"] = fmt(cj.loc[t, "coef"], 2)
phys_on_hj = None

# conditional LCZ contrast (dummies relative to the most common class)
cp = C[C.model == "physical_environmental"].set_index("term").coef
v["lcz_cond_3_A"] = fmt(cp["lcz_3"] - cp["lcz_11"], 2)
v["lcz_cond_8_A"] = fmt(cp["lcz_8"] - cp["lcz_11"], 2)

# top decile residents (revised ranking)
pri = pd.read_csv(TABLES / "priority_revised_by_mahalle.csv", encoding="utf-8-sig")
v["top80_pop"] = fmt_int(pri.loc[pri["rank"] <= 80, "pop_total"].sum())
v["top15_bayindir"] = int(((pri["rank"] <= 15) & (pri.district_name_final == "BAYINDIR")).sum())
bay = pri[(pri["rank"] <= 15) & (pri.district_name_final == "BAYINDIR")]
v["bay_lst_min"] = fmt(bay.lst_mean.min(), 1); v["bay_lst_max"] = fmt(bay.lst_mean.max(), 1)
v["mahalle_lst_res_median"] = fmt(pri.lst_mean.median(), 1)
v["rho_res_area_lst"] = fmt(pri[["lst_mean", "lst_mean_area"]].corr("spearman").iloc[0, 1], 2)

# bivariate Moran with residential-cell LST, and local bivariate clusters (Queen, 999 permutations)
mah = gpd.read_file(MAHALLE).to_crs(CRS)
M = mah[["join_key", "geometry"]].merge(pri[["join_key", "lst_mean", "lst_mean_area", "vulnerable_share", "district_name_final"]], on="join_key")
M = M.dropna(subset=["lst_mean"]).reset_index(drop=True)
w = Queen.from_dataframe(M, use_index=False, silence_warnings=True); w.transform = "r"
np.random.seed(0)
mb = Moran_BV(M.lst_mean.to_numpy(), M.vulnerable_share.to_numpy(), w, permutations=999)
v["mahalle_bivar_i_queen_res"] = fmt(mb.I, 3); v["mahalle_bivar_p_queen_res"] = fmt(mb.p_sim, 3)
v["n_islands_queen"] = len(w.islands)
lb = Moran_Local_BV(M.vulnerable_share.to_numpy(), M.lst_mean.to_numpy(), w, permutations=999, seed=0)
sig = lb.p_sim < 0.05
# quadrant: 1 HH (high vulnerability, hot neighbours), 3 LL, 2 LH, 4 HL
for q, nm in [(1, "hh"), (2, "lh"), (3, "ll"), (4, "hl")]:
    v[f"lisa_bv_{nm}"] = int((sig & (lb.q == q)).sum())
hh = M[sig & (lb.q == 1)]
v["lisa_bv_hh_districts"] = int(hh.district_name_final.nunique())
print(hh.district_name_final.value_counts().head(8))
hl = M[sig & (lb.q == 4)]
print("HL districts", hl.district_name_final.value_counts().head(8))

# provincial population totals (all mahalle in the population layer)
P = gpd.read_file(Path(r"EXTERNAL_DATA_DIR\DM\data\01_raw\nufus\mahalle_population_controls_2024.gpkg"))
v["pop_layer_total"] = fmt_int(pd.to_numeric(P.pop_total, errors="coerce").sum())
print(v)
write_values("r2_13_review_values", v)

# share of the official provincial population (TUIK ADNKS, 31 Dec 2024: 4,493,242) covered by the ranked units
POP_PROVINCE = 4493242
v2 = {"pop_province": fmt_int(POP_PROVINCE), "pop_rank_share_province": fmt(100 * pri.pop_total.sum() / POP_PROVINCE, 1),
      "pop_unranked_province": fmt_int(POP_PROVINCE - pri.pop_total.sum())}
print(v2)
v.update(v2)
write_values("r2_13_review_values", v)

# round-2 requests: separate child and elderly tests, unlisted residents, letter notation, joined-vs-model cells
LETTER = {11: "A", 12: "B", 13: "C", 14: "D", 15: "E", 16: "F", 17: "G"}
lc = lambda c: LETTER.get(int(c), str(int(c)))
v["mw_ns_pair"] = "; ".join(f"LCZ {lc(a)} and {lc(b)}" for a, b in zip(ns.lcz_a, ns.lcz_b)) or "none"
M2 = M.merge(pri[["join_key", "child_0_14_share", "elderly_65_plus_share"]], on="join_key", how="left")
from scipy import stats as _st
for col, nm in [("child_0_14_share", "child"), ("elderly_65_plus_share", "eld")]:
    np.random.seed(0)
    mbv = Moran_BV(M2.lst_mean_area.to_numpy(), M2[col].to_numpy(), w, permutations=999)
    v[f"bv_{nm}_i"] = fmt(mbv.I, 3); v[f"bv_{nm}_p"] = fmt(mbv.p_sim, 3)
    v[f"rho_{nm}_lst"] = fmt(_st.spearmanr(M2.lst_mean_area, M2[col]).statistic, 2)
v["rho_child_eld"] = fmt(_st.spearmanr(M2.child_0_14_share, M2.elderly_65_plus_share).statistic, 2)
vals08 = dict(l.rstrip("\n").split(",", 1) for l in open(VALUES / "r1_08_priority_alternatives.csv", encoding="utf-8"))
pop_wb = int(vals08["pop_wb"].replace("{,}", ""))
v["pop_unlisted"] = fmt_int(POP_PROVINCE - pop_wb)
gj = load_grid(valid_only=False)
pts2 = gpd.GeoDataFrame(gj[["row"]], geometry=gpd.points_from_xy(gj.x, gj.y), crs=CRS)
jj = gpd.sjoin(pts2, mah[["join_key", "geometry"]], predicate="within", how="left"); jj = jj[~jj.index.duplicated()]
gj["join_key"] = jj["join_key"].to_numpy()
inm = gj.join_key.isin(pri.join_key) & gj[DEP].notna()
compl = gj[["ndvi_2025", "ndbi_2025", "albedo_2025", "tree_canopy_height_2025", "lcz", "pop_total"]].notna().all(1)
v["n_joined_incomplete"] = int((inm & ~compl).sum())
print({k: v[k] for k in ["mw_ns_pair", "bv_child_i", "bv_child_p", "bv_eld_i", "bv_eld_p", "rho_child_lst", "rho_eld_lst", "rho_child_eld", "pop_unlisted", "n_joined_incomplete"]})
write_values("r2_13_review_values", v)

# round-3 requests
from libpysal.weights import DistanceBand
cent = np.column_stack([M2.geometry.centroid.x, M2.geometry.centroid.y])
wd = DistanceBand(cent, threshold=5000, binary=True, silence_warnings=True); wd.transform = "r"
for col, nm in [("child_0_14_share", "child"), ("elderly_65_plus_share", "eld")]:
    np.random.seed(0)
    mbv = Moran_BV(M2.lst_mean_area.to_numpy(), M2[col].to_numpy(), wd, permutations=999)
    v[f"bv_{nm}_i_d5"] = fmt(mbv.I, 3); v[f"bv_{nm}_p_d5"] = fmt(mbv.p_sim, 3)
inv_ = pd.read_pickle(CACHE / "grid_inventory.pkl")[["row", "col", "bsf", "bsf_res", "fa_res"]]
gr_ = gj.merge(inv_, on=["row", "col"], how="left")
resid = gr_[residential_mask(gr_) & gr_[DEP].notna()]
v["lst_res_ranked"] = fmt(resid.loc[resid.join_key.isin(pri.join_key), DEP].mean(), 2)
v["lst_res_unranked"] = fmt(resid.loc[~resid.join_key.isin(pri.join_key), DEP].mean(), 2)
v["n_res_unranked"] = fmt_int((~resid.join_key.isin(pri.join_key)).sum())
r_na = pri.P_noalb.rank(ascending=False, method="first")
v["consensus_in_top80_noalb"] = int(((r_na <= 80) & pri.consensus_top).sum())
from r1_22_tables_priority import nm as _nm
_c = pri.loc[pri.consensus_top].sort_values("rank")
v["consensus_names"] = "; ".join(f"{_nm(n)} ({_nm(d)})" for n, d in zip(_c.mahalle_name_final, _c.district_name_final))
print({k: v[k] for k in ["bv_child_i_d5", "bv_child_p_d5", "bv_eld_i_d5", "bv_eld_p_d5", "lst_res_ranked", "lst_res_unranked", "n_res_unranked", "consensus_in_top80_noalb", "consensus_names"]})
write_values("r2_13_review_values", v)

# round-3 requests (2): residential-support group tests, FDR-controlled local counts, group-weighted exposure,
# built-area-weighted index sensitivity, units resting on few residential cells
from statsmodels.stats.multitest import multipletests
for col, nm in [("child_0_14_share", "child"), ("elderly_65_plus_share", "eld")]:
    np.random.seed(0)
    mbv = Moran_BV(M2.lst_mean.to_numpy(), M2[col].to_numpy(), w, permutations=999)
    v[f"bv_{nm}_i"] = fmt(mbv.I, 3); v[f"bv_{nm}_p"] = fmt(mbv.p_sim, 3)
    np.random.seed(0)
    mbd = Moran_BV(M2.lst_mean.to_numpy(), M2[col].to_numpy(), wd, permutations=999)
    v[f"bv_{nm}_i_d5"] = fmt(mbd.I, 3); v[f"bv_{nm}_p_d5"] = fmt(mbd.p_sim, 3)
    v[f"rho_{nm}_lst"] = fmt(_st.spearmanr(M2.lst_mean, M2[col]).statistic, 2)
rej = multipletests(lb.p_sim, alpha=0.05, method="fdr_bh")[0]
for q, nm in [(1, "hh"), (2, "lh"), (3, "ll"), (4, "hl")]:
    v[f"lisa_bv_{nm}_fdr"] = int((rej & (lb.q == q)).sum())
v["lisa_bv_sig_total"] = int(sig.sum()); v["lisa_bv_fdr_total"] = int(rej.sum())
# group-weighted residential exposure (residents assumed evenly spread over the residential cells of their mahalle)
P2 = pri.dropna(subset=["lst_mean"])
grp = {"pop": P2.pop_total, "child": P2.pop_total * P2.child_0_14_share, "eld": P2.pop_total * P2.elderly_65_plus_share}
rng_b = np.random.default_rng(11)
for k, wts in grp.items():
    m = np.average(P2.lst_mean, weights=wts)
    bs = []
    for _ in range(1000):
        i = rng_b.integers(0, len(P2), len(P2))
        bs.append(np.average(P2.lst_mean.to_numpy()[i], weights=wts.to_numpy()[i]))
    v[f"exp_{k}"] = fmt(m, 2); v[f"exp_{k}_lo"] = fmt(np.percentile(bs, 2.5), 2); v[f"exp_{k}_hi"] = fmt(np.percentile(bs, 97.5), 2)
dd = []
for _ in range(1000):
    i = rng_b.integers(0, len(P2), len(P2))
    l = P2.lst_mean.to_numpy()[i]
    dd.append(np.average(l, weights=grp["child"].to_numpy()[i]) - np.average(l, weights=grp["eld"].to_numpy()[i]))
v["exp_child_minus_eld"] = fmt(np.average(P2.lst_mean, weights=grp["child"]) - np.average(P2.lst_mean, weights=grp["eld"]), 2)
v["exp_cme_lo"] = fmt(np.percentile(dd, 2.5), 2); v["exp_cme_hi"] = fmt(np.percentile(dd, 97.5), 2)
# units resting on few residential cells
v["n_units_res_lt10"] = int((pri.n_res < 10).sum()); v["n_top80_res_lt10"] = int(((pri["rank"] <= 80) & (pri.n_res < 10)).sum())
v["res_cells_q25"] = int(pri.n_res.quantile(0.25))
# built-area-weighted heat terms (weights = building surface fraction of each residential cell)
from mgwr_core import load_r0_fit
gg = load_grid().merge(inv_, on=["row", "col"], how="left")
pts3 = gpd.GeoDataFrame(gg[["row"]], geometry=gpd.points_from_xy(gg.x, gg.y), crs=CRS)
j3 = gpd.sjoin(pts3, mah[["join_key", "geometry"]], predicate="within", how="left"); j3 = j3[~j3.index.duplicated()]
gg["join_key"] = j3["join_key"].to_numpy()
fr = load_r0_fit(); Bf = np.load(CACHE / "betas_fullgrid_r0.npy").astype(float)
gg["C"] = (Bf[:, 1:] * ((gg[PREDICTORS].to_numpy(float) - fr["xm"]) / fr["xs"])).sum(1)
rr = gg[residential_mask(gg) & gg.join_key.isin(pri.join_key) & (gg.fa_res > 0)]
wa = rr.groupby("join_key").apply(lambda s: pd.Series({"T": np.average(s[DEP], weights=s.fa_res), "M": np.average(s.C, weights=s.fa_res)}), include_groups=False)
Pw = pri.set_index("join_key").join(wa)
zz = lambda x: (x - np.nanmean(x)) / np.nanstd(x)
Q = 0.38 * zz(Pw["T"]) + 0.30 * zz(Pw["M"]) + 0.22 * zz(Pw.vulnerable_share) + 0.10 * zz(np.log1p(Pw.pop_total))
rk_w = Q.rank(ascending=False, method="first"); rk_b = Pw["rank"]
v["alt_bsfw_rho"] = fmt(_st.spearmanr(Q, Pw.P0).statistic, 2); v["alt_bsfw_top80"] = int(((rk_w <= 80) & (rk_b <= 80)).sum())
print({k: v[k] for k in v if k.startswith(("bv_", "rho_child", "rho_eld", "lisa_bv", "exp_", "n_units", "n_top80_res", "res_cells", "alt_bsfw"))})
write_values("r2_13_review_values", v)

# Queen bivariate Moran without the units that have no contiguous neighbour
keep = ~np.isin(np.arange(len(M)), list(w.islands))
Mk = M[keep].reset_index(drop=True)
wk = Queen.from_dataframe(Mk, use_index=False, silence_warnings=True); wk.transform = "r"
np.random.seed(0)
mk = Moran_BV(Mk.lst_mean.to_numpy(), Mk.vulnerable_share.to_numpy(), wk, permutations=999)
v["bv_noislands_i"] = fmt(mk.I, 3)
print("no-islands I", v["bv_noislands_i"], "vs residential", v["mahalle_bivar_i_queen_res"])
write_values("r2_13_review_values", v)
