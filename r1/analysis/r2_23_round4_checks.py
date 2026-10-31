"""Supplementary checks.

(a) Is the lower 2025 eta^2 an effect of the stricter 10% scene cloud threshold? The 2025 June-August Landsat
    composite is re-exported on the analysis grid with the 30% threshold of the annual cube, and eta^2_H by LCZ is
    computed for it and for the 10% re-export on the same cells as the reported 2025 value.
(b) Share of cells with fewer than five clear observations in the 10% composite.
(c) Child minus older-resident residential exposure with a 95% interval from resampling districts (cluster
    bootstrap), which keeps the spatial dependence between mahalle of the same district.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import ee

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
from r2_01_gee_probe import coll  # noqa
from r2_05_gee_fullgrid import fetch_tile, TILES, NR, NC, OUTD  # noqa
from scipy import stats


def eta2(vals, labels):  # same estimator as r1_02 / Table 4
    ok = np.isfinite(vals)
    vv, ll = vals[ok], labels[ok]
    grp = [vv[ll == k] for k in np.unique(ll)]
    H, _ = stats.kruskal(*grp)
    return (H - len(grp) + 1) / (len(vv) - len(grp))


ee.Initialize(project=__import__("os").environ["EE_PROJECT"])


def grid_layer(name, img):
    out = OUTD / f"{name}.npy"
    if not out.exists():
        full = np.full((NR, NC), np.nan, np.float32)
        for r, c, h, w in TILES:
            full[r:r + h, c:c + w] = fetch_tile(img, r, c, h, w, False)
        np.save(out, full)
    return np.load(out)


v = {}
g = load_grid(valid_only=False).dropna(subset=[DEP, "lcz"]).reset_index(drop=True)
g["lcz"] = g["lcz"].astype(int)
r, c = g["row"].to_numpy(int), g["col"].to_numpy(int)
L10 = np.load(OUTD / "l_lst.npy")[r, c]
L30 = grid_layer("l_lst_cc30", coll(30).mean().select("lst"))[r, c]
N10 = grid_layer("l_nclear_cc10", coll(10).select("lst").count())[r, c]
ok = np.isfinite(L10) & np.isfinite(L30)
lab = g["lcz"].to_numpy()
v["eta2_scr10"] = fmt(eta2(np.where(ok, L10, np.nan), lab), 3)
v["eta2_scr30"] = fmt(eta2(np.where(ok, L30, np.nan), lab), 3)
v["r_scr10_scr30"] = fmt(np.corrcoef(L10[ok], L30[ok])[0, 1], 3)
fin = np.isfinite(N10)
v["ls_clear_lt5_pct"] = fmt(100 * (N10[fin] < 5).mean(), 2)
print(v, flush=True)

# (c) district cluster bootstrap of the child minus older-resident exposure contrast
pri = pd.read_csv(TABLES / "priority_revised_by_mahalle.csv", encoding="utf-8-sig")
P2 = pri.dropna(subset=["lst_mean"]).reset_index(drop=True)
wc = (P2.pop_total * P2.child_0_14_share).to_numpy()
we = (P2.pop_total * P2.elderly_65_plus_share).to_numpy()
T = P2.lst_mean.to_numpy()
dist = P2.district_name_final.to_numpy()
ud = np.unique(dist)
idx = {d: np.flatnonzero(dist == d) for d in ud}
rng = np.random.default_rng(23)
dd = []
for _ in range(2000):
    i = np.concatenate([idx[d] for d in rng.choice(ud, len(ud))])
    dd.append(np.average(T[i], weights=wc[i]) - np.average(T[i], weights=we[i]))
v["exp_cme_dist_lo"] = fmt(np.percentile(dd, 2.5), 2)
v["exp_cme_dist_hi"] = fmt(np.percentile(dd, 97.5), 2)
v["n_districts_boot"] = len(ud)
print(v)
write_values("r2_23_round4_checks", v)

# (d) agreement of the re-exported EVI with the analysed layer, on the same cells as scr_r_* (r2_21)
gm = load_grid()
ev = np.load(OUTD / "l_evi.npy")[gm.row.to_numpy(int), gm.col.to_numpy(int)]
evi_col = [k for k in gm.columns if k.startswith("evi")][0]
ok = np.isfinite(ev) & np.isfinite(gm[evi_col].to_numpy(float))
v["scr_r_evi"] = fmt(np.corrcoef(ev[ok], gm[evi_col].to_numpy(float)[ok])[0, 1], 3)
print("evi", evi_col, v["scr_r_evi"])
write_values("r2_23_round4_checks", v)
