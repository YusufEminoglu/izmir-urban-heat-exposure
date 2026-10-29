"""Write Supplementary Tables S4 (top mahalle), S5 (districts) and S11 (index alternatives) for the revised index."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
from r1_20_tables import TEX, n

TR = str.maketrans({"İ": "I", "ı": "i", "Ş": "S", "ş": "s", "Ğ": "G", "ğ": "g", "Ü": "U", "ü": "u", "Ö": "O", "ö": "o", "Ç": "C", "ç": "c"})


def nm(s):
    s = str(s).translate(TR).title()
    return s.replace("Ii", "II")


def s4():
    allp = pd.read_csv(TABLES / "priority_revised_by_mahalle.csv", encoding="utf-8-sig").sort_values("rank")
    p = allp.head(15)
    extra = allp[(allp["rank"] > 15) & allp.consensus_top]

    rk_na = allp.P_noalb.rank(ascending=False, method="first")

    SR = pd.read_csv(TABLES / "priority_rank_by_sample.csv", index_col=0)
    reps = [c for c in SR.columns if c.startswith("rep")]
    k_in = (SR[reps] <= 80).sum(1)
    _na_ranks = sorted(int(rk_na.loc[j]) for j in allp.index if allp.loc[j, "consensus_top"] and rk_na.loc[j] > 80)
    na_txt = (f"; without it they rank {_na_ranks[0]}--{_na_ranks[-1]}" if len(_na_ranks) > 1 else
              (f"; without it it ranks {_na_ranks[0]}" if _na_ranks else ""))

    def row(r):
        _k = int(k_in.get(r.join_key, 0))
        cons = f"$^{{\\ast {_k}}}$" if r.consensus_top else ""
        if r.consensus_top and rk_na.loc[r.name] > 80:
            cons = f"$^{{\\ast {_k},a}}$"
        if r.n_res < 10:
            cons += "$^{\\dagger}$"
        return (f"{int(r['rank'])} & {nm(r.district_name_final)} & {nm(r.mahalle_name_final)}{cons} & {n(r.pop_total)} & {n(r.n_cells)}/{n(r.n_res)} & {r.lst_mean:.2f} & {r.C:.2f} & "
                f"{r.vulnerable_share:.2f} & {n(r.hot_exposed_pop_proxy)} & {r.P0:.1f} & {int(r.rank_p05)}--{int(r.rank_p95)} \\\\")
    rows = [row(r) for _, r in p.iterrows()]
    if len(extra):
        rows.append("\\midrule\n\\multicolumn{11}{l}{\\textit{Further consensus units}} \\\\")
        rows += [row(r) for _, r in extra.iterrows()]
    s = r"""\begin{table}[H]
\caption{Top mahalle-level planning-priority units of the daytime heat-infrastructure screening index.}
\label{stab:mahalle_priority}
\scriptsize
\setlength{\tabcolsep}{1.8pt}
\begin{tabularx}{\textwidth}{rllrrrrrrrr}
\toprule
\textbf{Rank} & \textbf{District} & \textbf{Mahalle} & \textbf{Pop.} & \textbf{Cells} & \textbf{LST} & \textbf{$M_m$} & \textbf{Vuln.} & \textbf{Hot pop.} & \textbf{Score} & \textbf{5--95\%} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\begin{tablenotes}
\item Cells: 100~m cells of the unit / residential cells (with at least one dwelling of the municipal inventory); small village units consist of residential cells only, so their residential and whole-polygon values coincide. LST: mean summer 2025 LST of the unit's residential cells (\textdegree C). $M_m$: MGWR surface-related heat term over the same cells (Eq.~\ref{eq:mahalle_morph}; standard deviations of provincial LST). Vuln.: child (0--14) plus elderly (65+) share. Hot pop.: residents multiplied by the share of residential cells above the provincial 90th-percentile LST (\var{mahalle_hot_threshold}~\textdegree C), a descriptive threshold indicator that is not part of the score, so that a unit whose residential cells are warm but just below the threshold shows zero. Rank 5--95\%: range of the unit's rank under 1000 $\pm$20\% weight perturbations. $^{\ast}$In the top decile under the baseline and at least five of the six alternative constructions of Supplementary Table~\ref{stab:priority_alternatives}; all \var{n_consensus} such units are listed, and the number after the asterisk gives in how many of the five independent MGWR samples (Supplementary Table~\ref{stab:mgwr_stability_scale}) the unit stays in the top decile; $^{a}$not in the top decile when the albedo part of the surface heat term is dropped""" + na_txt + r""". $^{\dagger}$Fewer than ten residential cells.
\end{tablenotes}
\end{table}
"""
    (TEX / "table_s4_mahalle_priority.tex").write_text(s, encoding="utf-8")


def s5():
    d = pd.read_csv(TABLES / "priority_revised_by_district.csv", encoding="utf-8-sig").head(15)
    rows = [f"{nm(r.district_name_final)} & {int(r.n)} & {n(r['pop'])} & {r.lst:.2f} & {r.C:.2f} & {n(r.hot_pop)} & {int(r.n_top80)} & {r.score:.1f} \\\\"
            for _, r in d.iterrows()]
    s = r"""\begin{table}[H]
\caption{District-level aggregation of the mahalle planning-priority index (15 highest mean scores).}
\label{stab:district_priority}
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{tabularx}{\textwidth}{Xrrrrrrr}
\toprule
\textbf{District} & \textbf{Mahalle} & \textbf{Pop.} & \textbf{LST} & \textbf{$M_m$} & \textbf{Hot pop.} & \textbf{In top decile} & \textbf{Mean score} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\begin{tablenotes}
\item LST and $M_m$ are population-weighted district means of the mahalle values; In top decile: number of the district's mahalle among the 80 highest-ranked units.
\end{tablenotes}
\end{table}
"""
    (TEX / "table_s5_district_priority.tex").write_text(s, encoding="utf-8")


def s11():
    T = pd.read_csv(TABLES / "priority_alternatives.csv").set_index("alt")
    V = {}
    for line in open(VALUES / "r1_08_priority_alternatives.csv", encoding="utf-8"):
        k, v = line.rstrip("\n").split(",", 1); V[k] = v
    lab = [("r0_coef", "Coefficient-magnitude morphology term (Eq.~\\ref{eq:mahalle_morph_r0})", "0.38/0.30/0.22/0.10"),
           ("equal", "Equal weights", "0.25/0.25/0.25/0.25"),
           ("hev", "Equal hazard / sensitivity / exposure", "0.17/0.17/0.33/0.33"),
           ("entropy", "Entropy weights", V["w_ent"]), ("pca", "First principal component", V["w_pca"]),
           ("noM", "Without the MGWR term", "0.54/--/0.31/0.14"),
           ("area", "Heat terms averaged over all cells of the unit, not its residential cells", "0.38/0.30/0.22/0.10"),
           ("noalb", "Surface heat term without its albedo part", "0.38/0.30/0.22/0.10")]
    rows = [f"{l} & {w} & {T.loc[k, 'rho']:.2f} & {int(T.loc[k, 'top10'])} & {int(T.loc[k, 'top50'])} & {int(T.loc[k, 'top80'])} \\\\" for k, l, w in lab]
    s = r"""\begin{table}[H]
\caption{Dependence of the mahalle ranking on its construction: agreement of alternative indices with the baseline index (\var{n_mahalle_rank} units).}
\label{stab:priority_alternatives}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{p{4.0cm}>{\centering\arraybackslash}p{3.0cm}>{\centering\arraybackslash}X>{\centering\arraybackslash}X>{\centering\arraybackslash}X>{\centering\arraybackslash}X}
\toprule
\textbf{Construction} & \textbf{Weights $\bar T$/$M$/$V$/Pop} & \textbf{Spearman $\rho$} & \textbf{Top 10} & \textbf{Top 50} & \textbf{Top 80} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\begin{tablenotes}
\item 1000 $\pm$20\% perturbations of the baseline weights: median $\rho=\var{priority_rank_rho_median}$ (minimum \var{priority_rank_rho_min}); top 10 retained, median \var{priority_top10_retention_median} (minimum \var{priority_top10_retention_min}); top 80 retained, median \var{priority_top80_retention_median} (minimum \var{priority_top80_retention_min}).
\item Top $k$: number of the baseline's $k$ highest-ranked units that are also among the $k$ highest under the alternative. Entropy and principal-component weights are derived from the four standardised components. The consensus set of Supplementary Table~\ref{stab:mahalle_priority} uses the first six alternatives; the last two rows check the residential support of the heat terms and the albedo part of the surface heat term.
\end{tablenotes}
\end{table}
"""
    (TEX / "table_s11_priority_alternatives.tex").write_text(s, encoding="utf-8")


if __name__ == "__main__":
    for w in sys.argv[1:] or ["s4", "s5", "s11"]:
        globals()[w]()
        print("wrote", w)
