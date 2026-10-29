"""Supplementary Table S13: assembly of the province-wide grid (v2): source of each layer and its agreement with the
earlier processing of the data stack (same-source layers on the cells the earlier grid covered; earlier 2025 export)."""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
from r1_20_tables import TEX

R = pd.read_csv(TABLES / "gapfill_validation.csv")
get = lambda layer, metric: R[(R.layer == layer) & (R.metric == metric)].iloc[0]
V = {}
for line in open(VALUES / "r2_27_build_v2_grid.csv", encoding="utf-8-sig"):
    k, _, x = line.strip().partition(",")
    V[k] = x
_d = get("lst_2025", "calibration block-CV R2")
NG = f"{int(_d.filled + _d.unfilled):,}"
NP = f"{len(load_grid(valid_only=False)):,}"
rows = []
for layer, name in [("lcz", "LCZ class"), ("lcz_probability", "LCZ probability"), ("corine_2018", "CORINE 2018"), ("glc_fcs_2022", "GLC\\_FCS30D 2022")]:
    cells = f"{int(_d.filled):,}" if layer.startswith("lcz") else NG  # 49 shoreline cells lie outside the LCZ product
    rows.append(f"{name} & same Earth Engine asset & identical on {100 * get(layer, 'identical share').value:.1f}\\% of covered cells & {cells} \\\\")
rows.append(f"SRTM elevation & same source, 100~m sample & $r={get('srtm_elevation_m', 'pearson r').value:.4f}$ & {NG} \\\\")
rows.append(f"Slope, aspect (sine, cosine) & finite differences on the completed DEM & $r={get('srtm_slope_deg', 'pearson r').value:.3f}$, "
            f"${get('srtm_aspect_sin', 'pearson r').value:.3f}$, ${get('srtm_aspect_cos', 'pearson r').value:.3f}$ & {NP} \\\\")
for k, name in [("lst", "LST 2025"), ("albedo", "Albedo 2025"), ("ndvi", "NDVI 2025"), ("ndbi", "NDBI 2025"), ("evi", "EVI 2025")]:
    rows.append(f"{name} & documented Landsat~8/9 composite & $r={V[f'prev_r_{k}']}$ with the earlier export & {NP} \\\\")
rows.append(f"LST anomaly, UTFVI & derived from LST & -- & {NP} \\\\")
s = r"""\begin{table}[H]
\caption{Assembly of the province-wide grid: source of each layer and agreement with the earlier processing of the data stack, which did not cover \var{n_gapfilled} cells (\var{gapfill_share}\% of the province).}
\label{stab:gapfill}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{p{2.6cm}p{4.4cm}Xr}
\toprule
\textbf{Layer} & \textbf{Source} & \textbf{Agreement with the earlier processing} & \textbf{Cells} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\begin{tablenotes}
\item Cells: for the same-source layers, the cells the earlier grid did not cover; for the derived and 2025 layers, all province cells. Agreement is computed on the cells both versions cover. Slope and aspect differ from the earlier derivation only at the \var{slope_changed_pct}\% of cells next to the province edge or the uncovered patches, where that derivation had filled missing neighbours with the province median elevation. The earlier 2025 export and the reasons it is not used are described in Supplementary Note~\ref{snote:grid}.
\end{tablenotes}
\end{table}
"""
(TEX / "table_s13_gapfill.tex").write_text(s, encoding="utf-8")
print("ok")
