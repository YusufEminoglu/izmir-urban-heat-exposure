"""Stability of the MGWR outputs at the scales at which they are used (R1-1, R1-2).

For every fit (reported model, refit, five new samples, other sizes, augmented),
kernel-consistent coefficients are computed at a fixed random subsample of 100 m
cells (seeded), and compared with the reported model after aggregation to
(i) mahalle means, (ii) 5 km blocks and (iii) LCZ classes; for the mahalle
surface heat term C and the priority score built from it, rank agreement and
top-decile overlap are reported.
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

rng = np.random.default_rng(5)
g = load_grid()
mah = gpd.read_file(MAHALLE).to_crs(CRS)
pts = gpd.GeoDataFrame(g[["row"]], geometry=gpd.points_from_xy(g.x, g.y), crs=CRS)
j = gpd.sjoin(pts, mah[["join_key", "geometry"]], predicate="within", how="left")
j = j[~j.index.duplicated()]
g["join_key"] = j["join_key"].to_numpy()
pri = pd.read_csv(TABLES / "priority_revised_by_mahalle.csv", encoding="utf-8-sig")
inr = g.join_key.isin(pri.join_key).to_numpy()
# subsample: all ranked-mahalle cells thinned to 1 in 6, plus 1 in 20 elsewhere
sel = np.where(inr)[0]
sel = np.r_[rng.choice(sel, len(sel) // 6, replace=False), rng.choice(np.where(~inr)[0], (~inr).sum() // 20, replace=False)]
s = g.iloc[np.sort(sel)].reset_index(drop=True)
xy = s[["x", "y"]].to_numpy(float)
print("cells used", len(s))
# mahalle terms follow the index: all residential cells (inventory buildings) of the ranked mahalle
inv = pd.read_pickle(CACHE / "grid_inventory.pkl")[["row", "col", "bsf", "bsf_res"]]
gr = g.merge(inv, on=["row", "col"], how="left")
sr = gr[gr.join_key.isin(pri.join_key) & residential_mask(gr)].reset_index(drop=True)
xyr = sr[["x", "y"]].to_numpy(float)
print("residential cells of ranked mahalle", len(sr))

fits = {"r0": load_r0_fit()}
for name in ["orig", "rep1", "rep2", "rep3", "rep4", "rep5", "n2000", "n6000", "aug"]:
    p = CACHE / f"mgwr_{name}.npz"
    if p.exists():
        fits[name] = dict(np.load(p))

W0 = np.array([0.38, 0.30, 0.22, 0.10])
z = lambda v: (v - np.nanmean(v)) / np.nanstd(v)
res = {}
for k, f in fits.items():
    cp = CACHE / f"stab_betas_{k}.npy"
    if cp.exists():
        B = np.load(cp)
    else:
        B = local_betas(f, xy)
        np.save(cp, B.astype(np.float32))
    Xs = (s[PREDICTORS].to_numpy(float) - f["xm"][:3]) / f["xs"][:3]
    C = (B[:, 1:4] * Xs).sum(1)
    d = pd.DataFrame({"join_key": s.join_key, "lcz": s.lcz, "blk": (s.x // 5000) * 1e5 + (s.y // 5000),
                      "ndvi": B[:, 1], "ndbi": B[:, 2], "alb": B[:, 3], "C": C})
    cpr = CACHE / f"stab_betas_dwell_{k}.npy"
    if cpr.exists():
        Br = np.load(cpr)
    else:
        Br = local_betas(f, xyr)
        np.save(cpr, Br.astype(np.float32))
    Xr = (sr[PREDICTORS].to_numpy(float) - f["xm"][:3]) / f["xs"][:3]
    dr = pd.DataFrame({"join_key": sr.join_key, "ndvi": Br[:, 1], "ndbi": Br[:, 2], "alb": Br[:, 3], "C": (Br[:, 1:4] * Xr).sum(1)})
    res[k] = (d, dr)
    print(k, "done")

base, base_r = res["r0"]
rows = []
bm = base_r.groupby("join_key").mean(numeric_only=True)
bb = base.groupby("blk").mean(numeric_only=True)
bl = base.groupby("lcz").median(numeric_only=True)
P = pri.set_index("join_key")


def score(Cm):
    Cm = Cm.reindex(P.index)
    Q = W0[0] * z(P.lst_mean) + W0[1] * z(Cm) + W0[2] * z(P.vulnerable_share) + W0[3] * z(np.log1p(P.pop_total))
    return Q


Q0 = score(bm.C); r0rank = Q0.rank(ascending=False)
sample_ranks = {}
for k, (d, dr) in res.items():
    if k == "r0":
        continue
    m = dr.groupby("join_key").mean(numeric_only=True)
    b = d.groupby("blk").mean(numeric_only=True)
    l = d.groupby("lcz").median(numeric_only=True)
    Q = score(m.C); rr = Q.rank(ascending=False)
    sample_ranks[k] = rr
    row = dict(fit=k)
    for v in ["ndvi", "ndbi", "alb", "C"]:
        row[f"mah_r_{v}"] = np.corrcoef(m[v].reindex(bm.index), bm[v])[0, 1]
        row[f"blk_r_{v}"] = np.corrcoef(b[v].reindex(bb.index), bb[v])[0, 1]
    row["lcz_rho_ndvi"] = stats.spearmanr(l.ndvi.reindex(bl.index), bl.ndvi).statistic
    row["lcz_rho_alb"] = stats.spearmanr(l.alb.reindex(bl.index), bl.alb).statistic
    row["mah_sign_ndvi"] = np.mean(np.sign(m.ndvi.reindex(bm.index)) == np.sign(bm.ndvi))
    row["prio_rho"] = stats.spearmanr(Q, Q0, nan_policy="omit").statistic
    row["prio_top80"] = int(((rr <= 80) & (r0rank <= 80)).sum())
    row["prio_top10"] = int(((rr <= 10) & (r0rank <= 10)).sum())
    rows.append(row)
T = pd.DataFrame(rows)
pd.DataFrame(sample_ranks).to_csv(TABLES / "priority_rank_by_sample.csv")  # rank of each mahalle per fit
T.to_csv(TABLES / "mgwr_stability_planning_scale.csv", index=False)
pd.set_option("display.width", 250)
print(T.round(3).to_string())
reps = T[T.fit.str.startswith("rep")]
vals = {}
for v in ["ndvi", "ndbi", "alb", "C"]:
    vals[f"rep_mah_r_{v}_med"] = fmt(reps[f"mah_r_{v}"].median(), 2); vals[f"rep_mah_r_{v}_min"] = fmt(reps[f"mah_r_{v}"].min(), 2)
    vals[f"rep_blk_r_{v}_med"] = fmt(reps[f"blk_r_{v}"].median(), 2); vals[f"rep_blk_r_{v}_min"] = fmt(reps[f"blk_r_{v}"].min(), 2)
vals["rep_prio_rho_med"] = fmt(reps.prio_rho.median(), 3); vals["rep_prio_rho_min"] = fmt(reps.prio_rho.min(), 3)
vals["rep_prio_top80_med"] = int(reps.prio_top80.median()); vals["rep_prio_top80_min"] = int(reps.prio_top80.min())
vals["rep_prio_top10_med"] = int(reps.prio_top10.median()); vals["rep_prio_top10_min"] = int(reps.prio_top10.min())
vals["rep_lcz_rho_ndvi_med"] = fmt(reps.lcz_rho_ndvi.median(), 2); vals["rep_lcz_rho_alb_med"] = fmt(reps.lcz_rho_alb.median(), 2)
vals["rep_mah_sign_ndvi_med"] = fmt(100 * reps.mah_sign_ndvi.median(), 1)
for k in ["orig", "n2000", "n6000", "aug"]:
    if k in set(T.fit):
        r = T.set_index("fit").loc[k]
        vals[f"{k}_prio_rho"] = fmt(r.prio_rho, 3); vals[f"{k}_prio_top80"] = int(r.prio_top80)
        vals[f"{k}_mah_r_C"] = fmt(r.mah_r_C, 2); vals[f"{k}_mah_r_ndvi"] = fmt(r.mah_r_ndvi, 2)
write_values("r1_11_stability_planning_scale", vals)
