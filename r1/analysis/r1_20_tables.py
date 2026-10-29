"""Write the revised and new manuscript tables (LaTeX) from the analysis outputs."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

TEX = Path(r"MANUSCRIPT_TABLES_DIR")
V = {}
for f in VALUES.glob("*.csv"):
    for line in open(f, encoding="utf-8"):
        k, v = line.rstrip("\n").split(",", 1)
        V[k] = v


def n(x):
    return f"{int(x):,}"


def table4():
    t = pd.read_csv(TABLES / "lcz_lst_uncertainty.csv")
    order = [2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0]
    t = t.set_index("lcz").loc[order].reset_index()
    rows = []
    for _, r in t.iterrows():
        lab = "--" if r.lcz == 0 else LCZ_LABEL[r.lcz]
        name = LCZ_NAME[r.lcz] + ("" if r.supported else "$^{\\dagger}$")
        rows.append(f"{lab} & {name} & {n(r.n)} & {n(r.blocks)} & {r["mean"]:.2f} & {r.ci_lo:.2f}--{r.ci_hi:.2f} & {r.sd:.2f} & {r['median']:.2f} \\\\")
        if r.lcz == 10 or r.lcz == 17:
            rows.append("\\addlinespace")
    body = "\n".join(rows)
    s = r"""\begin{table}[H]
\caption{LCZ--LST summary statistics and Kruskal--Wallis test. Values are summer 2025 LST (\textdegree C) at 100~m resolution; 95\% intervals of the class mean from a 2~km spatial block bootstrap (2,000 replicates).}
\label{tab:lcz_lst}
\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabularx}{\textwidth}{lXrrrcrr}
\toprule
\textbf{LCZ} & \textbf{Class} & \textbf{Cells} & \textbf{Blocks} & \textbf{Mean} & \textbf{95\% CI} & \textbf{SD} & \textbf{Median} \\
\midrule
""" + body + r"""
\bottomrule
\end{tabularx}
\begin{tablenotes}
\item Kruskal--Wallis over the 15 categories: $H(\var{kw_df})=\var{kw_H}$, $p<0.001$, $\eta_H^2=\var{lcz_eta_sq}$; over the 14 LCZ types (unclassified removed): $H(\var{kw_df_types})=\var{kw_H_types}$, $\eta_H^2=\var{lcz_eta_sq_types}$; over well-supported types only: $\eta_H^2=\var{lcz_eta_sq_supported}$.
\item $^{\dagger}$Limited support: fewer than \var{lcz_min_cells} cells or fewer than \var{lcz_min_blocks} occupied 2~km blocks (Section~\ref{sec:lcz_checks}). Their statistics are reported but are not used as the basis of class-specific recommendations; Section~\ref{sec:results_lcz_checks} checks them against the municipal building inventory. The product codes LCZ A--G as 11--17. Blocks: number of distinct 2~km blocks occupied by the class.
\end{tablenotes}
\end{table}
"""
    (TEX / "table_4_lcz_lst.tex").write_text(s, encoding="utf-8")


def table_s_inventory():
    T1 = pd.read_csv(TABLES / "lcz_inventory_morphology.csv").set_index("lcz")
    T3 = pd.read_csv(TABLES / "inventory_class_lst.csv"); T3["inv"] = T3["inv"].astype(str); T3 = T3.set_index("inv")
    T4 = pd.read_csv(TABLES / "lcz_confirmed_lst.csv").set_index("lcz")
    rows = []
    for k in [2, 3, 5, 6, 8, 9, 10]:
        a, c = T1.loc[k], T4.loc[k]
        cm = f"{c.mean_conf:.2f}" if np.isfinite(c.mean_conf) and c.n_conf >= 20 else "--"
        rows.append(f"{k} {LCZ_NAME[k]} & {n(a.n)} & {100*a.bsf3_med:.0f} ({100*a.bsf3_q25:.0f}--{100*a.bsf3_q75:.0f}) & {str(a.so_bsf).replace("-", "--")} & "
                    f"{a.h3_med:.1f} ({a.h3_q25:.1f}--{a.h3_q75:.1f}) & {str(a.so_h).replace("-", "--")} & {100*c.share_conf:.1f} & {cm} \\\\")
    body1 = "\n".join(rows)
    rows = []
    nm = {"1": "Compact high-rise", "2": "Compact mid-rise", "3": "Compact low-rise", "4": "Open high-rise", "5": "Open mid-rise",
          "6": "Open low-rise", "8": "Large low-rise", "9": "Sparsely built"}
    U = pd.read_csv(TABLES / "lcz_lst_uncertainty.csv").set_index("lcz")
    for k in ["2", "3", "5", "6", "8", "9"]:
        r = T3.loc[k]; u = U.loc[int(k)]
        rows.append(f"{nm[k]} & {n(r.n)} & {int(r.blocks)} & {r['mean']:.2f} ({r.ci_lo:.2f}--{r.ci_hi:.2f}) & {n(u.n)} & {u['mean']:.2f} ({u.ci_lo:.2f}--{u.ci_hi:.2f}) \\\\")
    body2 = "\n".join(rows)
    s = r"""\begin{table}[H]
\caption{WUDAPT built classes against the municipal building inventory. (a) Inventory morphology in the 300~m window of each class, with the \citet{stewart2012local} ranges, and the share of cells whose inventory compactness class agrees with the WUDAPT class. (b) Summer 2025 LST by inventory-derived class and by the corresponding WUDAPT class (95\% block-bootstrap intervals).}
\label{stab:lcz_inventory}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{p{2.25cm}r>{\centering\arraybackslash}Xc>{\centering\arraybackslash}Xc>{\centering\arraybackslash}X>{\centering\arraybackslash}X}
\toprule
\multicolumn{8}{l}{\textbf{(a) Morphology by WUDAPT class}} \\
\textbf{WUDAPT class} & \textbf{Cells} & \textbf{BSF \%, median (IQR)} & \textbf{S\&O BSF} & \textbf{Height m, median (IQR)} & \textbf{S\&O height} & \textbf{Confirmed \%} & \textbf{LST confirmed} \\
\midrule
""" + body1 + r"""
\bottomrule
\end{tabularx}

\vspace{6pt}
\begin{tabularx}{\textwidth}{p{2.6cm}>{\centering\arraybackslash}X>{\centering\arraybackslash}X>{\centering\arraybackslash}X>{\centering\arraybackslash}X>{\centering\arraybackslash}X}
\toprule
\multicolumn{6}{l}{\textbf{(b) LST by inventory-derived class and by WUDAPT class}} \\
\textbf{Class} & \textbf{Inventory cells} & \textbf{Blocks} & \textbf{Inventory LST (95\% CI)} & \textbf{WUDAPT cells} & \textbf{WUDAPT LST (95\% CI)} \\
\midrule
""" + body2 + r"""
\bottomrule
\end{tabularx}
\begin{tablenotes}
\item BSF: building surface fraction. Height: footprint-weighted mean of above-ground floors $\times$ 3~m, over cells with at least one inventory building. Confirmed: inventory compactness class (compact, open, sparsely built, large low-rise) equals the WUDAPT class. The inventory typology uses BSF and height only (Section~\ref{sec:lcz_checks}); it is a cross-check of the global product, not a replacement for it.
\end{tablenotes}
\end{table}
"""
    (TEX / "table_s7_lcz_inventory.tex").write_text(s, encoding="utf-8")


def table_s_temporal():
    M = pd.read_csv(TABLES / "lcz_mean_lst_by_year.csv", index_col=0); M.columns = M.columns.astype(int)
    E = pd.read_csv(TABLES / "lcz_eta2_by_year.csv").set_index("year")
    S = pd.read_csv(TABLES / "lcz_mk_sen_2014_2026.csv").set_index("lcz")
    rows = []
    for k in [2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:
        lab = LCZ_LABEL[k]
        rank = lambda yr: int(M.loc[[2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], yr].rank(ascending=False)[k])
        s = S.loc[k]
        rows.append(f"{lab} {LCZ_NAME[k]} & {M.loc[k, 2018]:.2f} & {M.loc[k, 2025]:.2f} & {rank(2018)} & {rank(2025)} & "
                    f"{s.sen_median:.3f} & {100*s.share_up:.1f} \\\\")
    body = "\n".join(rows)
    eta = " / ".join(f"{E.loc[y, 'eta2']:.2f}" for y in [2014, 2018, 2022, 2025])
    s = r"""\begin{table}[H]
\caption{Temporal consistency of the LCZ thermal ordering in the cloud-local LST cube. Class-mean summer LST in 2018 (the LCZ epoch) and 2025, rank of the class (1 = hottest of 14 types), and per-cell monotonic change over 2014--2026.}
\label{stab:lcz_temporal}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabularx}{\textwidth}{p{2.5cm}>{\centering\arraybackslash}X>{\centering\arraybackslash}X>{\centering\arraybackslash}X>{\centering\arraybackslash}X>{\centering\arraybackslash}X>{\centering\arraybackslash}X}
\toprule
\textbf{LCZ} & \textbf{LST 2018} & \textbf{LST 2025} & \textbf{Rank 2018} & \textbf{Rank 2025} & \textbf{Sen slope (\textdegree C yr$^{-1}$)} & \textbf{Increasing (\%)} \\
\midrule
""" + body + r"""
\bottomrule
\end{tabularx}
\begin{tablenotes}
\item $\eta_H^2$ in 2014 / 2018 / 2022 / 2025: """ + eta + r"""; Kendall's $W$ over 13 summers: \var{kendall_w_types}; Spearman $\rho$ of class means with 2025: \var{rho_types_2018_2025} in 2018, at least \var{rho_types_min} in every year.
\item Sen slope: median per-cell Theil--Sen slope 2014--2026. Increasing: share of cells with a significant increasing Mann--Kendall trend ($p<0.05$). Slopes summarise composite-to-composite change and include inter-annual weather and sensor differences between summers; they are compared across classes, not read as a climate trend.
\end{tablenotes}
\end{table}
"""
    (TEX / "table_s8_lcz_temporal.tex").write_text(s, encoding="utf-8")


if __name__ == "__main__":
    which = sys.argv[1:] or ["table4", "table_s_inventory", "table_s_temporal"]
    for w in which:
        globals()[w]()
        print("wrote", w)
