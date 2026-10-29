"""OLS benchmark on the complete province grid: residual Moran's I (rook and queen, binary weights,
analytical randomisation inference) and Table 5 / S1 / S2 from the pipeline outputs (p1_07)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
from r1_20_tables import TEX

PT = ROOT / "pipeline" / "outputs" / "tables" / "paper1"
g = load_grid()
n = len(g)
cont = OLS_CONT
Xc = g[cont].to_numpy(float)
Xc = (Xc - Xc.mean(0)) / Xc.std(0)
lczs = sorted(g.lcz.unique())
D = np.column_stack([(g.lcz.to_numpy() == k).astype(float) for k in lczs if k != 14])
A = np.column_stack([np.ones(n), Xc, D])
beta, *_ = np.linalg.lstsq(A, g[DEP].to_numpy(float), rcond=None)
e = g[DEP].to_numpy(float) - A @ beta
NR, NC = 1744, 1981
E = np.full((NR, NC), np.nan); E[g.row, g.col] = e
V = np.isfinite(E); E0 = np.where(V, E, 0.0)


def moran_grid(offsets):
    num, S0 = 0.0, 0.0
    for dr, dc in offsets:
        a = E0[max(0, -dr):NR - max(0, dr), max(0, -dc):NC - max(0, dc)]
        b = E0[max(0, dr):NR - max(0, -dr) if dr < 0 else NR, max(0, dc):NC]
        va = V[max(0, -dr):NR - max(0, dr), max(0, -dc):NC - max(0, dc)]
        vb = V[max(0, dr):NR, max(0, dc):NC]
        b = E0[max(0, dr):NR + min(0, dr), max(0, dc):NC + min(0, dc)]
        vb = V[max(0, dr):NR + min(0, dr), max(0, dc):NC + min(0, dc)]
        m = va & vb
        num += (a * b)[m].sum(); S0 += m.sum()
    I = (n / S0) * num / (e ** 2).sum()
    EI = -1 / (n - 1)
    # normal approximation (randomisation is indistinguishable at this n)
    S1 = 2 * S0  # binary symmetric weights: S1 = 2*S0 when each directed link counted once per direction
    k = len(offsets)
    deg = np.zeros((NR, NC))
    for dr, dc in offsets:
        vb = np.zeros((NR, NC), bool)
        vb[max(0, -dr):NR + min(0, -dr), max(0, -dc):NC + min(0, -dc)] = V[max(0, dr):NR + min(0, dr), max(0, dc):NC + min(0, dc)]
        deg += (V & vb)
    S2 = (4 * deg[V] ** 2).sum()
    VarI = (n * n * S1 - n * S2 + 3 * S0 ** 2) / ((n * n - 1) * S0 ** 2) - EI ** 2
    z = (I - EI) / np.sqrt(VarI)
    return I, z, int(S0)


rook = [(0, 1), (0, -1), (1, 0), (-1, 0)]
queen = rook + [(1, 1), (1, -1), (-1, 1), (-1, -1)]
Ir, zr, Lr = moran_grid(rook)
Iq, zq, Lq = moran_grid(queen)
print("rook I", Ir, "z", zr, "links", Lr, "| queen I", Iq, "z", zq, "links", Lq)

S = pd.read_csv(PT / "paper1_ols_model_summary.csv", encoding="utf-8-sig").set_index("model")
C = pd.read_csv(PT / "paper1_ols_coefficients.csv", encoding="utf-8-sig")
C = C[C.model == "physical_environmental"].set_index("term")
VI = pd.read_csv(PT / "paper1_vif_diagnostics.csv", encoding="utf-8-sig")
VI = VI[VI.model == "physical_environmental"].set_index("term").vif
lab = {"const": "Intercept", "ndvi_2025": "NDVI", "ndbi_2025": "NDBI", "albedo_2025": "Albedo", "tree_canopy_height_2025": "Tree canopy height",
       "srtm_elevation_m": "SRTM elevation", "srtm_slope_deg": "SRTM slope", "srtm_aspect_sin": "Aspect sine", "srtm_aspect_cos": "Aspect cosine",
       "lcz_probability": "LCZ probability"}
lczlab = {0: "LCZ 0 / other", 2: "LCZ 2 compact mid-rise", 3: "LCZ 3 compact low-rise", 5: "LCZ 5 open mid-rise", 6: "LCZ 6 open low-rise",
          8: "LCZ 8 large low-rise", 9: "LCZ 9 sparsely built", 10: "LCZ 10 heavy industry", 11: "LCZ A dense trees", 12: "LCZ B scattered trees",
          13: "LCZ C bush/scrub", 15: "LCZ E bare rock/paved", 16: "LCZ F bare soil/sand", 17: "LCZ G water"}


def row(term, name):
    r_ = C.loc[term]; v = "--" if term == "const" else f"{VI[term]:.2f}"
    return f"{name} & {r_.coef:.3f} & {r_.std_error_hc3:.3f} & {r_.t:.2f} & {v} \\\\".replace("-", "$-$", 1) if r_.coef < 0 else \
        f"{name} & {r_.coef:.3f} & {r_.std_error_hc3:.3f} & {r_.t:.2f} & {v} \\\\"


phys = "\n".join(row(t, lab[t]) for t in lab)
lcz_rows = "\n".join(row(f"lcz_{k}", lczlab[k]) for k in lczlab if f"lcz_{k}" in C.index)
adj = S.loc["physical_environmental", "adj_r2"]; aic = S.loc["physical_environmental", "aic"]
t5 = r"""\begin{table}[H]
\caption{Baseline physical-environmental OLS model and VIF diagnostics ($n=\var{n_cells}$). Coefficients use HC3 robust standard errors.}
\label{tab:ols}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabularx}{\textwidth}{Xrrrr}
\toprule
\textbf{Variable} & \textbf{Coef.} & \textbf{SE} & \textbf{$t$} & \textbf{VIF} \\
\midrule
""" + phys + r"""
\midrule
\multicolumn{5}{l}{Adjusted $R^2=\var{ols_r2_adj}$; AIC $=\var{ols_aic}$; maximum VIF $=\var{ols_vif_max}$} \\
\multicolumn{5}{l}{OLS residual Moran's $I=\var{ols_morans_i}$ (rook contiguity; $p\var{ols_morans_p}$)} \\
\bottomrule
\end{tabularx}
\begin{tablenotes}
\small
\item LCZ category dummies are included in the model but omitted from this display for readability; full coefficients are reported in Supplementary Table~\ref{stab:ols_full}. HC3 standard errors address heteroskedasticity, not spatial dependence. Because residual Moran's $I$ shows strong spatial autocorrelation, OLS is interpreted as a global benchmark rather than an independent-error spatial model, and coefficient $p$-values are not treated as causal spatial inference.
\end{tablenotes}
\end{table}
"""
(TEX / "table_5_ols.tex").write_text(t5, encoding="utf-8")
s2 = r"""\begin{table}[H]
\caption{Full physical-environmental OLS coefficient table including LCZ controls (reference: LCZ~D low plants). Coefficients use HC3 robust standard errors.}
\label{stab:ols_full}
\scriptsize
\begin{tabularx}{\textwidth}{Xrrrr}
\toprule
\textbf{Term} & \textbf{Coef.} & \textbf{SE} & \textbf{$t$} & \textbf{VIF} \\
\midrule
""" + phys + "\n" + lcz_rows + r"""
\bottomrule
\end{tabularx}
\end{table}
"""
(TEX / "table_s2_ols_full.tex").write_text(s2, encoding="utf-8")
s1rows = "\n".join(f"{lab[t]} & {VI[t]:.2f} & Yes \\\\" for t in lab if t != "const")
s1 = r"""\begin{table}[H]
\caption{VIF diagnostics for the displayed physical-environmental OLS covariates.}
\label{stab:vif}
\small
\begin{tabularx}{\textwidth}{Xrl}
\toprule
\textbf{Variable} & \textbf{VIF} & \textbf{Included?} \\
\midrule
""" + s1rows + r"""
\bottomrule
\end{tabularx}
\begin{tablenotes}
\item VIF values above 5 are treated as a warning band; no displayed predictor exceeds the exclusion threshold of 10.
\end{tablenotes}
\end{table}
"""
(TEX / "table_s1_vif.tex").write_text(s1, encoding="utf-8")

ndvi = C.loc["ndvi_2025"]
write_values("r2_08_ols_moran", {
    "n_cells": f"{n:,}".replace(",", "{,}"), "n_cells_pct": fmt(100 * n / 1197430, 2),
    "ols_r2_adj": fmt(adj, 3), "ols_r2_adj_pct": fmt(100 * adj, 1), "ols_aic": f"{aic:,.1f}".replace(",", "{,}"),
    "ols_vif_max": fmt(VI[[t for t in lab if t != "const"]].max(), 2),
    "ols_ndvi_coef": fmt(ndvi.coef, 3), "ols_ndvi_p": fmt(ndvi.p_value, 2), "ols_ndbi_coef": fmt(C.loc["ndbi_2025"].coef, 3),
    "ols_morans_i": fmt(Ir, 3), "ols_morans_p": "<0.001", "ols_morans_i_queen": fmt(Iq, 3),
    "ols_morans_links_rook": f"{Lr:,}".replace(",", "{,}"), "ols_morans_links_queen": f"{Lq:,}".replace(",", "{,}"),
    "ols_morans_z": f"{zr:,.0f}".replace(",", "{,}"),
    "ols_hj_r2_adj": fmt(S.loc["heat_justice_with_population", "adj_r2"], 3),
})
