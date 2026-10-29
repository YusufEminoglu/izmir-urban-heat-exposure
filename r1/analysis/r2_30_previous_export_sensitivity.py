"""Sensitivity of the results to the source of the 2025 spectral layers (round-4 decision, v2).

The analysis now uses the documented Earth Engine composite (r2_27). The previously analysed export, whose NDVI, NDBI
and EVI could not be regenerated from their recipe, is kept as a sensitivity case. This script collects the
corresponding results of the earlier, fully computed version of the analysis (outputs/v1_values, v1_tables: same
code, same settings, previous layers) and compares them with the current ones: layer agreement, LCZ effect size,
OLS fit and standardised surface coefficients, MGWR bandwidths and fit, the mahalle surface heat term and the index.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

V1 = OUT / "v1_values"
T1 = OUT / "v1_tables"


def val(folder, name, key):
    for line in open(folder / f"{name}.csv", encoding="utf-8-sig"):
        k, _, x = line.strip().partition(",")
        if k == key:
            return x
    raise KeyError(key)


v = {}
v["prev_eta2"] = val(V1, "r1_01_lcz_uncertainty", "lcz_eta_sq")
v["prev_gap_c23_A"] = val(V1, "r1_01_lcz_uncertainty", "gap_c23_A")
v["prev_ols_r2_adj"] = val(V1, "r2_08_ols_moran", "ols_r2_adj")
for k in ["ndvi", "ndbi", "albedo"]:
    v[f"prev_ols_b_{k}"] = val(V1, "r2_21_scripted_ols", f"ref_ols_b_{k}")
v["prev_mgwr_bw"] = "17/64/140/97"
v["prev_mgwr_r2"] = "0.880"
p1 = pd.read_csv(T1 / "priority_revised_by_mahalle.csv", encoding="utf-8-sig").set_index("join_key")
p2 = pd.read_csv(TABLES / "priority_revised_by_mahalle.csv", encoding="utf-8-sig").set_index("join_key")
j = p2.join(p1[["P0", "rank", "C", "consensus_top"]], rsuffix="_prev", how="inner")
v["prev_mah_r_C"] = fmt(np.corrcoef(j.C, j.C_prev)[0, 1], 2)
v["prev_prio_rho"] = fmt(stats.spearmanr(j.P0, j.P0_prev).statistic, 2)
v["prev_prio_top80"] = int(((j["rank"] <= 80) & (j["rank_prev"] <= 80)).sum())
v["prev_cons_shared"] = int((j.consensus_top.astype(bool) & j.consensus_top_prev.astype(bool)).sum())
v["prev_n_consensus"] = int(j.consensus_top_prev.astype(bool).sum())
print(v)
write_values("r2_30_previous_export_sensitivity", v)
