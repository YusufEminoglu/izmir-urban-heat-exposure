"""Write Table 7 (model comparison), Table 8 (validation) and Supplementary Tables S9-S11."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
from r1_20_tables import TEX, n


def table7():
    s = r"""\begin{table}[H]
\caption{Global and local models on the common calibration sample ($n=\var{n_mgwr_sample}$; standardised LST and the three surface predictors; row-standardised $k=8$ nearest-neighbour $\mathbf{W}$ for estimation and diagnostics).}
\label{tab:spatial_comparison}
\scriptsize
\setlength{\tabcolsep}{1.9pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrrlllr}
\toprule
\textbf{Model} & \textbf{Spatial term} & \textbf{Param.} & \textbf{Log-lik.} & \textbf{AIC} & \textbf{In-sample fit} & \textbf{Residual} & \textbf{Moran's $I$} & \textbf{CV RMSE} \\
\midrule
OLS & -- & 5 & $\var{ols_sample_ll}$ & \var{ols_sample_aic} & $R^2$ \var{ols_sample_r2} & $y-\hat y$ & \var{ols_sample_resid_morans_i} & \var{cv_blk_sample_OLS3_rmse} \\
SEM & $\lambda=\var{sem_lambda}$ & 6 & $\var{sem_ll}$ & \var{sem_aic} & pseudo-$R^2$ \var{sem_pseudo_r2} & trend $\hat u$ & \var{sem_resid_morans_i} & \var{cv_blk_sample_SEMtrend_rmse} \\
 & & & & & & innovation $\hat\varepsilon$ & $\var{sem_filtered_morans_i}$ & \\
SAR & $\rho=\var{sar_rho}$ & 6 & $\var{sar_ll}$ & \var{sar_aic} & pseudo-$R^2$ \var{sar_pseudo_r2} & innovation $\hat\varepsilon$ & \var{sar_resid_morans_i} & -- \\
 & & & & & & reduced form & \var{sar_rf_morans_i} & \\
MGWR & 4 bandwidths & ENP \var{mgwr_enp} & $\var{mgwr_ll}$ & \var{mgwr_aic} & $R^2$ \var{mgwr_r2} & $y-\hat y$ & $\var{mgwr_resid_morans_i}$ & \var{cv_blk_sample_MGWR_rmse} \\
\bottomrule
\end{tabular*}
\begin{tablenotes}
\item Log-likelihoods are Gaussian log-likelihoods of the same standardised response (for SEM and SAR including the Jacobian term), so log-likelihood and AIC are comparable across rows; MGWR is penalised by its effective number of parameters (ENP; AICc \var{mgwr_aicc}); for OLS, SEM and SAR the small-sample correction is negligible ($n=\var{n_mgwr_sample}$, at most six parameters), adding at most \var{aicc_corr_max} to their AIC. In-sample fit measures differ by model family and are not compared across rows. Residuals: SEM trend $\hat u=y-\mathbf{X}\hat\beta$, which contains the modelled process $\hat\lambda\mathbf{W}\hat u$ by construction, and filtered innovation $\hat\varepsilon=\hat u-\hat\lambda\mathbf{W}\hat u$; SAR innovation $y-\hat\rho\mathbf{W}y-\mathbf{X}\hat\beta$ and reduced-form residual $y-(\mathbf{I}-\hat\rho\mathbf{W})^{-1}\mathbf{X}\hat\beta$. Moran's $I$: 999 permutations; the SEM innovation has $p=\var{sem_filtered_morans_p}$, every other value $p\leq0.001$. CV RMSE: spatial block cross-validation on the sample cells, \textdegree C (Table~\ref{tab:validation}); SAR is not cross-validated because its predictions require the outcomes of neighbouring held-out cells.
\end{tablenotes}
\end{table}
"""
    (TEX / "table_7_spatial_comparison.tex").write_text(s, encoding="utf-8")


def table8():
    CV = pd.read_csv(TABLES / "validation_summary.csv")

    def cell(sc, st, m):
        r = CV[(CV["set"] == st) & (CV.model == m)] if sc is None else CV[(CV.scheme == sc) & (CV["set"] == st) & (CV.model == m)]
        return (f"{r.r2.iloc[0]:.3f} & {r.rmse.iloc[0]:.2f}") if len(r) else "-- & --"
    rows = []
    for m, lab, mh in [("MGWR", "MGWR (kernel-consistent)", "MGWR_kernel"), ("OLSfull", "OLS, full specification", "OLSfull"),
                       ("SEMtrend", "SEM trend", "SEMtrend"), ("OLS3", "OLS, three surface predictors", "OLS3")]:
        rows.append(f"{lab} & {cell('rnd', 'sample', m)} & {cell('blk', 'sample', m)} & {cell('blk', 'grid', m)} & {cell(None, 'grid_holdout', mh)} \\\\")
    ia = CV[(CV["set"] == "grid_holdout") & (CV.model == "MGWR_idw_atlas")]
    rows.append(f"MGWR (reported IDW atlas) & -- & -- & -- & -- & -- & -- & {ia.r2.iloc[0]:.3f} & {ia.rmse.iloc[0]:.2f} \\\\")
    body = "\n".join(rows)
    s = r"""\begin{table}[H]
\caption{Out-of-sample performance: $R^2=1-\mathrm{SSE}/\mathrm{SST}$ and RMSE (\textdegree C) on cells not used for calibration. In every fold all models are fitted on the same training cells.}
\label{tab:validation}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{Xcccccccc}
\toprule
 & \multicolumn{2}{c}{\textbf{Random 5-fold}} & \multicolumn{2}{c}{\textbf{Spatial blocks}} & \multicolumn{2}{c}{\textbf{Spatial blocks}} & \multicolumn{2}{c}{\textbf{Full-grid hold-out}} \\
 & \multicolumn{2}{c}{sample cells} & \multicolumn{2}{c}{sample cells} & \multicolumn{2}{c}{all 100~m cells} & \multicolumn{2}{c}{\var{n_grid_holdout} cells} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
\textbf{Model} & $R^2$ & RMSE & $R^2$ & RMSE & $R^2$ & RMSE & $R^2$ & RMSE \\
\midrule
""" + body + r"""
\bottomrule
\end{tabularx}
\begin{tablenotes}
\item Random and spatial-block folds re-calibrate every model, MGWR with bandwidths re-selected, on four fifths of the \var{n_mgwr_sample}-cell sample. Spatial blocks are 10~km squares assigned at random to five folds; ``all 100~m cells'' evaluates every grid cell inside the held-out blocks (\var{cv_blk_grid_MGWR_n} cells). The full-grid hold-out uses the reported calibration and predicts every grid cell outside the sample. OLS, full specification: Eq.~\ref{eq:ols} with terrain, canopy height, LCZ probability and LCZ dummies. SEM trend: $\mathbf{X}\hat\beta$ of the spatial error model.
\end{tablenotes}
\end{table}
"""
    (TEX / "table_8_validation.tex").write_text(s, encoding="utf-8")


def table_s_stability():
    ST = pd.read_csv(TABLES / "mgwr_stability.csv")
    ST = ST[ST.fit != "orig"]  # v2: the reported model is the fit of this sample
    lab = {"r0": "Reported model", "orig": "Refit of the reported sample", "rep1": "New sample 1", "rep2": "New sample 2", "rep3": "New sample 3", "rep4": "New sample 4",
           "rep5": "New sample 5", "n2000": "Sample of 2,000", "n6000": "Sample of 6,000", "aug": "Augmented (+ elevation, canopy height)"}
    rows = []
    for _, r in ST.iterrows():
        bw = f"{int(r.bw_int)}/{int(r.bw_ndvi)}/{int(r.bw_ndbi)}/{int(r.bw_alb)}"
        rr = "--" if r.fit == "r0" else f"{r.r_ndvi:.2f} / {r.r_ndbi:.2f} / {r.r_albedo:.2f}"
        dg = "--" if r.fit == "r0" else f"{100*r.dom_agree:.1f}"
        rows.append(f"{lab[r.fit]} & {n(r.n)} & {bw} & {r.R2:.3f} & {r.ENP:.1f} & {r.moran:.3f} & {r.cond_max:.1f} & {rr} & {dg} \\\\")
    body = "\n".join(rows)
    s = r"""\begin{table}[H]
\caption{Stability of the MGWR solution across independent samples, sample sizes and an augmented local specification. Coefficient surfaces are evaluated with each fit's kernel-consistent estimator on the 500~m atlas grid and compared with the reported model.}
\label{stab:mgwr_stability}
\scriptsize
\setlength{\tabcolsep}{1.8pt}
\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}Xrlrrrrcr}
\toprule
\textbf{Fit} & \textbf{$n$} & \textbf{Bandwidths} & \textbf{$R^2$} & \textbf{ENP} & \textbf{Moran's $I$} & \textbf{Max.} & \textbf{$r$ NDVI / NDBI /} & \textbf{Same} \\
 & & & & & & \textbf{cond.} & \textbf{albedo} & \textbf{driver (\%)} \\
\midrule
""" + body + r"""
\bottomrule
\end{tabularx}
\begin{tablenotes}
\item Bandwidths (nearest neighbours): intercept/NDVI/NDBI/albedo; the augmented fit adds elevation (\var{aug_bw_elev}) and canopy height (\var{aug_bw_tch}). New samples were drawn from the full grid with the stratified design of Section~\ref{sec:gwr}, excluding the calibration cells of the reported model. Moran's $I$ of the residuals with $k=8$ neighbours. $r$: Pearson correlation of the coefficient surface with that of the reported model.
\end{tablenotes}
\end{table}
"""
    (TEX / "table_s9_mgwr_stability.tex").write_text(s, encoding="utf-8")


def table_s_interp():
    IS = pd.read_csv(TABLES / "atlas_interpolation_sensitivity.csv").set_index("method")
    MM = pd.read_csv(TABLES / "atlas_interp_mahalle.csv").set_index("method")
    lab = {"idw_p2_k12": "IDW, power 2, 12 neighbours (reported)", "idw_p1_k12": "IDW, power 1, 12 neighbours", "idw_p2_k6": "IDW, power 2, 6 neighbours",
           "idw_p2_k24": "IDW, power 2, 24 neighbours", "nearest": "Nearest neighbour", "linear": "Delaunay linear",
           "lcz_strat_idw": "IDW within built / within natural LCZ", "kernel_direct": "Kernel-consistent estimator (no interpolation)"}
    rows = []
    for m in lab:
        r = IS.loc[m]; q = MM.loc[m]
        loo = "--" if not np.isfinite(r.r2_ndvi) else f"{r.r2_ndvi:.2f} / {r.r2_ndbi:.2f} / {r.r2_alb:.2f}"
        rows.append(f"{lab[m]} & {r.r_ndvi:.2f} / {r.r_ndbi:.2f} / {r.r_alb:.2f} & {100*r.sign_ndbi:.1f} / {100*r.sign_alb:.1f} & {100*r.dom_agree:.1f} & {loo} & {q.rho_M:.2f} \\\\")
    body = "\n".join(rows)
    s = r"""\begin{table}[H]
\caption{Sensitivity of the coefficient atlas to the interpolation scheme: agreement with the reported atlas on the 500~m grid, leave-one-out reproduction of the sampled coefficients, and agreement of the mahalle coefficient-magnitude term computed at every 100~m cell.}
\label{stab:atlas_interp}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{p{3.4cm}>{\centering\arraybackslash}X>{\centering\arraybackslash}X>{\centering\arraybackslash}X>{\centering\arraybackslash}X>{\centering\arraybackslash}X}
\toprule
\textbf{Scheme} & \textbf{$r$ NDVI / NDBI / albedo} & \textbf{Same sign NDBI / albedo (\%)} & \textbf{Same dominant driver (\%)} & \textbf{LOO $R^2$ NDVI / NDBI / albedo} & \textbf{Mahalle $\rho$} \\
\midrule
""" + body + r"""
\bottomrule
\end{tabularx}
\begin{tablenotes}
\item LOO $R^2$: share of the variance of the sampled local coefficients recovered when each calibration cell is predicted from the others. Mahalle $\rho$: Spearman correlation of the mahalle mean of the coefficient-magnitude term of Eq.~\ref{eq:mahalle_morph_r0} with its value under the reported scheme.
\end{tablenotes}
\end{table}
"""
    (TEX / "table_s10_atlas_interpolation.tex").write_text(s, encoding="utf-8")


def table_s_stability_scale():
    T = pd.read_csv(TABLES / "mgwr_stability_planning_scale.csv")
    T = T[T.fit != "orig"]  # v2: the reported model is the fit of this sample
    lab = {"orig": "Refit of the reported sample", "rep1": "New sample 1", "rep2": "New sample 2", "rep3": "New sample 3", "rep4": "New sample 4",
           "rep5": "New sample 5", "n2000": "Sample of 2,000", "n6000": "Sample of 6,000", "aug": "Augmented (+ elevation, canopy height)"}
    rows = []
    for _, r in T.iterrows():
        rows.append(f"{lab[r.fit]} & {r.mah_r_ndvi:.2f} & {r.mah_r_ndbi:.2f} & {r.mah_r_alb:.2f} & {r.mah_r_C:.2f} & {r.lcz_rho_alb:.2f} & {r.prio_rho:.3f} & {int(r.prio_top80)} \\\\")
    s = r"""\begin{table}[H]
\caption{Stability of the MGWR outputs at the scales at which they are used: agreement of each alternative fit with the reported model across the ranked mahalle.}
\label{stab:mgwr_stability_scale}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{p{2.8cm}>{\centering\arraybackslash}X>{\centering\arraybackslash}X>{\centering\arraybackslash}X>{\centering\arraybackslash}X>{\centering\arraybackslash}X>{\centering\arraybackslash}X>{\centering\arraybackslash}X}
\toprule
 & \multicolumn{4}{c}{\textbf{Mahalle mean, Pearson $r$}} & \textbf{LCZ medians} & \multicolumn{2}{c}{\textbf{Priority ranking}} \\
\cmidrule(lr){2-5}\cmidrule(lr){7-8}
\textbf{Fit} & \textbf{NDVI} & \textbf{NDBI} & \textbf{Albedo} & \textbf{Surface heat term} & \textbf{$\rho$ albedo} & \textbf{Spearman $\rho$} & \textbf{Top 80 shared} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\begin{tablenotes}
\item Kernel-consistent coefficients of each fit at a fixed subsample of 100~m cells, averaged over the \var{n_mahalle_rank} ranked mahalle. Correlations of the NDBI coefficient are lower in the samples whose NDBI bandwidth is broad, because the coefficient then varies little, although its values agree closely (Supplementary Table~\ref{stab:mgwr_stability}). Surface heat term: $M_m$ of Eq.~\ref{eq:mahalle_morph}. Priority ranking: the index of Eq.~\ref{eq:mahalle_priority_q} recomputed with each fit's surface heat term.
\end{tablenotes}
\end{table}
"""
    (TEX / "table_s9b_mgwr_stability_scale.tex").write_text(s, encoding="utf-8")


if __name__ == "__main__":
    which = sys.argv[1:] or ["table7", "table8", "table_s_stability", "table_s_interp", "table_s_stability_scale"]
    for w in which:
        globals()[w]()
        print("wrote", w)
