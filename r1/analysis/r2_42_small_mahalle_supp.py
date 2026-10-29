"""Mahalle outside the ranking (Supplementary Note S2, Table S14).

The ADNKS 2024 age-group mahalle table lists mahalle with at least 500 residents, so 513 of the 1,317 mahalle
geometries have no population record and are not ranked. The 2023 TurkStat totals on the Izmir open-data portal
('Cinsiyete Gore Mahalle Nufus Bilgileri', mahalle with more than 10 residents; data/izmir_portal_tuik_mahalle_population_2023.csv,
downloaded 2026-10-30 from acikveri.bizizmir.com) give the size of these units. Their heat conditions are summarised with
the same residential-cell support and MGWR surface heat term C as the ranked units (r1_08); they are not scored,
because the portal gives no age structure at mahalle level.
Writes outputs/tables/small_mahalle_outside_ranking.csv, docs/.../tables/table_s14_small_mahalle.tex and
outputs/values/r2_42_small_mahalle.csv."""
import sys
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
from mgwr_core import load_r0_fit

PORTAL = DATA / "izmir_portal_tuik_mahalle_population_2023.csv"
TEX = Path(r"MANUSCRIPT_TABLES_DIR\table_s14_small_mahalle.tex")
TR = str.maketrans({"Ç": "C", "Ğ": "G", "İ": "I", "Ö": "O", "Ş": "S", "Ü": "U", "Â": "A", "Î": "I", "Û": "U",
                    "ç": "C", "ğ": "G", "ı": "I", "i": "I", "ö": "O", "ş": "S", "ü": "U"})


def norm(s):
    s = str(s).strip().upper().translate(TR)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    for suf in (" MAHALLESI", " MAH.", " MAH"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return " ".join(s.replace(".", ". ").split())


p = pd.read_csv(PORTAL, sep=";", encoding="utf-8")
p["pop2023"] = pd.to_numeric(p.NUFUS_ERKEK_TOPLAM, errors="coerce").fillna(0) + pd.to_numeric(p.NUFUS_KADIN_TOPLAM, errors="coerce").fillna(0)
p["key"] = p.ILCE_ADI.map(norm) + "|" + p.MAHALLE_ADI.map(norm)
pk = p.drop_duplicates("key").set_index("key")["pop2023"]

mah = gpd.read_file(MAHALLE).to_crs(CRS)
mah["key"] = mah.district_name_geom.map(norm) + "|" + mah.mahalle_name_geom.map(norm)
pri = pd.read_csv(TABLES / "priority_revised_by_mahalle.csv", encoding="utf-8-sig")
out = mah[~mah.join_key.isin(pri.join_key)].copy()
out["pop2023"] = out.key.map(pk)
osb = out.mahalle_name_geom.str.upper().str.contains("OSB|ORGANİZE|SERBEST|PLASTİKOSB|SB$|TDİOSB", regex=True)

# residential-cell LST and surface heat term C, exactly as for the ranked units (r1_08)
g = load_grid()
pts = gpd.GeoDataFrame(g[["row"]], geometry=gpd.points_from_xy(g.x, g.y), crs=CRS)
j = gpd.sjoin(pts, mah[["join_key", "geometry"]], predicate="within", how="left")
j = j[~j.index.duplicated()]
g["join_key"] = j["join_key"].to_numpy()
f = load_r0_fit()
B = np.load(CACHE / "betas_fullgrid_r0.npy")
Xs = (g[PREDICTORS].to_numpy(float) - f["xm"]) / f["xs"]
g["C"] = (B[:, 1:] * Xs).sum(1)
inv = pd.read_pickle(CACHE / "grid_inventory.pkl")[["row", "col", "bsf", "bsf_res"]]
g = g.merge(inv, on=["row", "col"], how="left")
g["res"] = residential_mask(g)
r = g[g.res].groupby("join_key").agg(n_res=(DEP, "size"), lst_res=(DEP, "mean"), C=("C", "mean"))
a = g.groupby("join_key").agg(n_cells=(DEP, "size"), lst_all=(DEP, "mean"))
out = out.merge(r, left_on="join_key", right_index=True, how="left").merge(a, left_on="join_key", right_index=True, how="left")

nolist = out.pop_total.isna()  # no ADNKS record (below the 500-resident listing threshold or no residents)
small = out[nolist & out.pop2023.notna() & ~osb]
other = out[nolist & ~out.index.isin(small.index)]
print("ranked-ineligible units with an ADNKS record (no LST support):", int((~nolist).sum()))
print("outside ranking", len(out), "| small listed in portal", len(small), "| OSB / free zones / no portal record", len(other))
print(other[["district_name_geom", "mahalle_name_geom", "pop2023"]].to_string())
out.drop(columns="geometry").to_csv(TABLES / "small_mahalle_outside_ranking.csv", index=False, encoding="utf-8-sig")

wavg = lambda d, c, w: float(np.average(d[c].dropna(), weights=d.loc[d[c].notna(), w])) if d[c].notna().any() else np.nan
rk = pri.dropna(subset=["lst_res"])
vals = {
    "n_mahalle_outside": int(nolist.sum()), "n_small_portal": len(small), "n_outside_other": len(other),
    "n_outside_osb": int(osb.sum()),
    "small_pop_total": fmt_int(small.pop2023.sum()), "small_pop_med": fmt_int(small.pop2023.median()), "small_pop_max": fmt_int(small.pop2023.max()),
    "small_pop_share_prov": fmt(100 * small.pop2023.sum() / p.pop2023.sum(), 1),
    "small_with_res": int(small.n_res.notna().sum()),
    "small_lst_res": fmt(wavg(small, "lst_res", "pop2023"), 1), "ranked_lst_res": fmt(wavg(rk, "lst_res", "pop_total"), 1),
    "small_C": fmt(wavg(small, "C", "pop2023"), 2), "ranked_C": fmt(wavg(rk, "C", "pop_total"), 2),
    "small_hot_units": int((small.lst_res >= rk.lst_res.quantile(0.9)).sum()),
    "portal_adnks_ratio": fmt(float((mah.pop_total / mah.key.map(pk)).replace([np.inf, -np.inf], np.nan).dropna().median()), 2),
}
print(vals)
write_values("r2_42_small_mahalle", vals)

# Table S14: by district
d = small.groupby("district_name_geom").agg(units=("join_key", "size"), residents=("pop2023", "sum"), pop_med=("pop2023", "median"),
                                             lst=("lst_res", "mean"), C=("C", "mean"), n_res=("n_res", lambda v: v.notna().sum()))
d = d.sort_values("residents", ascending=False)
def num(x, nd):
    return "--" if not np.isfinite(x) else f"{x:.{nd}f}".replace("-", "$-$")


def title_tr(s):
    return " ".join(w[:1] + w[1:].replace("I", "ı").lower().replace("i̇", "i") for w in str(s).split())


lines = [f"{title_tr(name)} & {int(rr.units)} & {int(rr.residents):,} & {int(rr.pop_med):,} & {num(rr.lst, 1)} & {num(rr.C, 2)} \\\\"
         for name, rr in d.iterrows()]
tot = (f"\\midrule\nAll small units & {len(small)} & {int(small.pop2023.sum()):,} & {int(small.pop2023.median()):,} & "
       f"{num(small.lst_res.mean(), 1)} & {num(small.C.mean(), 2)} \\\\\n"
       f"Ranked units (for comparison) & {len(rk)} & {int(rk.pop_total.sum()):,} & {int(rk.pop_total.median()):,} & "
       f"{num(rk.lst_res.mean(), 1)} & {num(rk.C.mean(), 2)} \\\\")
write_values("r2_42_small_mahalle_b", {"n_outside_nopop": int(len(other) - osb.sum())})
tex = r"""\begin{table}[H]
\caption{Mahalle outside the ranking: small residential units by district. Residents from the 2023 TurkStat mahalle totals on the Izmir open-data portal; mean daytime LST and MGWR surface heat term $C$ over the residential cells, as for the ranked units (unweighted means of the units).}
\label{stab:small_mahalle}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabularx}{\textwidth}{Xrrrrr}
\toprule
\textbf{District} & \textbf{Units} & \textbf{Residents} & \textbf{Median} & \textbf{LST (\textdegree C)} & \textbf{$C$} \\
\midrule
""" + "\n".join(lines) + "\n" + tot + r"""
\bottomrule
\end{tabularx}
\begin{tablenotes}
\scriptsize
\item The ADNKS 2024 age-group table used for the index lists mahalle with at least 500 residents, so these units have no age structure and are not scored. Not listed: \var{n_outside_other} geometries without residents in the portal table (\var{n_outside_osb} organised industrial and free zones and \var{n_outside_nopop} units absent from the portal table, which omits mahalle with 10 or fewer residents). Unit-level values are in the data deposit.
\end{tablenotes}
\end{table}
"""
TEX.write_text(tex, encoding="utf-8")
print("wrote", TEX)
