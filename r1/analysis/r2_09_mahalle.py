"""Mahalle aggregates, heat--vulnerability bivariate Moran's I and join quality on the complete grid.

The aggregation (checked against the pipeline table): cells are assigned to mahalle by centre-in-polygon; T_m mean LST, p90 LST,
hot-cell share above the provincial 90th percentile, hot-exposed population proxy = population x
hot-cell share; V_m = child + elderly share. Bivariate Moran: esda Moran_BV(mean LST, V) with
row-standardised Queen contiguity and a 5 km centroid distance band (999 permutations, seed 0).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import Queen, DistanceBand
from esda.moran import Moran_BV
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
from r1_20_tables import TEX

R0P = pd.read_csv(PRIORITY_CSV_R0, encoding="utf-8-sig")
mah = gpd.read_file(MAHALLE).to_crs(CRS)


def aggregate(grid):
    pts = gpd.GeoDataFrame(grid[["row"]], geometry=gpd.points_from_xy(grid.x, grid.y), crs=CRS)
    j = gpd.sjoin(pts, mah[["join_key", "geometry"]], predicate="within", how="left")
    j = j[~j.index.duplicated()]
    grid = grid.assign(join_key=j["join_key"].to_numpy())
    thr = np.nanpercentile(grid[DEP], 90)
    sub = grid[grid.join_key.isin(R0P.join_key)]
    a = sub.groupby("join_key").agg(n_cells=(DEP, "size"), lst_mean=(DEP, "mean"), lst_p90=(DEP, lambda v: np.nanpercentile(v, 90)),
                                    hot_cell_share=(DEP, lambda v: np.mean(v >= thr)))
    return a, thr, len(sub)


g_old = pd.read_pickle(CACHE / "grid_R0.pkl").dropna(subset=[DEP])
a_old, thr_old, n_old = aggregate(g_old)
chk = R0P.set_index("join_key")[["lst_mean", "lst_p90", "hot_cell_share", "n_cells"]].join(a_old, rsuffix="_new")
print("reproduction on original grid: max |dLST|", np.nanmax(np.abs(chk.lst_mean - chk.lst_mean_new)),
      "max |dp90|", np.nanmax(np.abs(chk.lst_p90 - chk.lst_p90_new)), "max |dhot|", np.nanmax(np.abs(chk.hot_cell_share - chk.hot_cell_share_new)),
      "threshold", thr_old, "cells joined", n_old)
hp = R0P.hot_exposed_pop_proxy / (R0P.pop_total * R0P.hot_cell_share)
print("hot_exposed_pop_proxy / (pop x hot share): median", np.nanmedian(hp))

g = load_grid(valid_only=False).dropna(subset=[DEP])
a_new, thr, n_join = aggregate(g)
P = R0P.drop(columns=["n_cells", "lst_mean", "lst_p90", "hot_cell_share", "hot_exposed_pop_proxy"]).merge(a_new, left_on="join_key", right_index=True, how="left")
P["hot_exposed_pop_proxy"] = P.pop_total * P.hot_cell_share
P.to_csv(TABLES / "mahalle_components_complete.csv", index=False, encoding="utf-8-sig")
print("mahalle with LST", P.lst_mean.notna().sum(), "threshold", thr, "cells joined", n_join,
      "changed lst_mean >0.01:", int((np.abs(P.set_index('join_key').lst_mean - R0P.set_index('join_key').lst_mean) > 0.01).sum()))

# bivariate Moran on mahalle polygons
M = mah.merge(P[["join_key", "lst_mean", "lst_p90", "vulnerable_share"]], on="join_key", how="inner").dropna(subset=["lst_mean"]).reset_index(drop=True)
wq = Queen.from_dataframe(M, use_index=False, silence_warnings=True); wq.transform = "r"
cent = np.column_stack([M.geometry.centroid.x, M.geometry.centroid.y])
wd = DistanceBand(cent, threshold=5000, binary=True, silence_warnings=True); wd.transform = "r"
np.random.seed(0)
res = {}
for nm, w in [("queen", wq), ("dist5km", wd)]:
    for var, key in [("lst_mean", ""), ("lst_p90", "_p90")]:
        m = Moran_BV(M[var].to_numpy(), M.vulnerable_share.to_numpy(), w, permutations=999)
        res[f"mahalle_bivar{key}_i_{nm}"] = fmt(m.I, 3); res[f"mahalle_bivar{key}_p_{nm}"] = fmt(m.p_sim, 3)
        print(nm, var, round(m.I, 3), m.p_sim)
rho = stats.spearmanr(M.lst_mean, M.vulnerable_share).statistic
res.update({"mahalle_heat_vuln_spearman": fmt(rho, 3), "mahalle_hot_threshold": fmt(thr, 2), "n_mahalle_grid": int(P.lst_mean.notna().sum()),
            "n_cells_mahalle_join": f"{n_join:,}".replace(",", "{,}")})
write_values("r2_09_mahalle", res)
s6 = r"""\begin{table}[H]
\caption{Mahalle analysis coverage and join-quality diagnostics.}
\label{stab:mahalle_join_quality}
\small
\begin{tabularx}{\textwidth}{Xr}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Mahalle geometries & 1,317 \\
Mahalle with matched 2024 population (all units listed in the ADNKS workbook) & \var{n_mahalle_matched} \\
Mahalle with LST grid support & \var{n_mahalle_grid} \\
Listed residents covered by the ranked units & \var{pop_rank} of \var{pop_wb} (\var{pop_rank_share}\%) \\
100~m grid cells joined to matched mahalle & \var{n_cells_mahalle_join} \\
Province-wide 90th percentile LST threshold (\textdegree C) & \var{mahalle_hot_threshold} \\
\bottomrule
\end{tabularx}
\end{table}
"""
(TEX / "table_s6_mahalle_join_quality.tex").write_text(s6, encoding="utf-8")
