"""LCZ-LST statistics with spatial uncertainty and a minimum-support rule.

* Table 4 statistics on every cell with valid LST and LCZ (n = 1,187,870).
* Spatial block bootstrap (2 km blocks, 2,000 replicates) for class means and
  class contrasts: blocks, not cells, are resampled, so the intervals respect
  spatial autocorrelation.
* Support: cells, 2 km blocks occupied, and 8-connected patches per class.
  A class is "well supported" when it has >= 1,000 cells (10 km2) and occupies
  >= 30 distinct 2 km blocks; otherwise its statistics are reported but flagged.
* Kruskal-Wallis on all categories, on the 14 LCZ types (unclassified removed),
  and on well-supported classes only.
* The thermal contrasts quoted in the text, with block-bootstrap intervals.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats, ndimage

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

B = 2000
BLOCK = 20  # cells -> 2 km
MIN_CELLS, MIN_BLOCKS = 1000, 30
rng = np.random.default_rng(2026)

g = load_grid(valid_only=False)
g = g.dropna(subset=[DEP, "lcz"]).reset_index(drop=True)
g["lcz"] = g["lcz"].astype(int)
g["blk"] = (g["row"] // BLOCK).astype(np.int64) * 100_000 + (g["col"] // BLOCK).astype(np.int64)
print("cells", len(g))

# patches per class (8-connected) on the row/col raster
nr, nc = g["row"].max() + 1, g["col"].max() + 1
lab = np.full((nr, nc), -1, np.int16)
lab[g["row"], g["col"]] = g["lcz"]


def n_patches(c):
    m = lab == c
    _, n = ndimage.label(m, structure=np.ones((3, 3)))
    return n


def block_boot_mean(sub, reps=B):
    bs = sub.groupby("blk")[DEP].agg(["sum", "count"])
    s, n = bs["sum"].to_numpy(), bs["count"].to_numpy()
    k = len(s)
    idx = rng.integers(0, k, size=(reps, k))
    return s[idx].sum(1) / n[idx].sum(1)


rows = []
boot = {}
for c, sub in g.groupby("lcz"):
    bm = block_boot_mean(sub)
    boot[c] = bm
    nb = sub["blk"].nunique()
    v = sub[DEP]
    rows.append(dict(lcz=c, label=LCZ_LABEL[c], name=LCZ_NAME[c], n=len(sub), blocks=nb, patches=n_patches(c),
                     mean=v.mean(), sd=v.std(), median=v.median(), q25=v.quantile(.25), q75=v.quantile(.75),
                     ci_lo=np.percentile(bm, 2.5), ci_hi=np.percentile(bm, 97.5),
                     supported=(len(sub) >= MIN_CELLS) and (nb >= MIN_BLOCKS)))
t = pd.DataFrame(rows)
t.to_csv(TABLES / "lcz_lst_uncertainty.csv", index=False)
print(t.round(2).to_string())


def kw(df):
    grp = [v[DEP].to_numpy() for _, v in df.groupby("lcz")]
    H, p = stats.kruskal(*grp)
    K, N = len(grp), len(df)
    return H, K - 1, (H - K + 1) / (N - K)


H_all, df_all, e_all = kw(g)
H_14, df_14, e_14 = kw(g[g.lcz != 0])
sup = t.loc[t.supported, "lcz"].tolist()
H_s, df_s, e_s = kw(g[g.lcz.isin(sup) & (g.lcz != 0)])
print("KW all", H_all, df_all, e_all, "| 14 types", H_14, df_14, e_14, "| supported", H_s, df_s, e_s)

# block-bootstrap of area-weighted group means: resample blocks jointly across classes
blk_tab = g.groupby(["blk", "lcz"])[DEP].agg(["sum", "count"]).unstack(fill_value=0)
S, Nn = blk_tab["sum"], blk_tab["count"]
cls = list(S.columns)
Sarr, Narr = S.to_numpy(), Nn.to_numpy()
idx = rng.integers(0, len(S), size=(B, len(S)))


def grp_mean(codes, ii=None):
    j = [cls.index(c) for c in codes if c in cls]
    if ii is None:
        return Sarr[:, j].sum() / Narr[:, j].sum()
    return Sarr[ii][:, :, j].sum((1, 2)) / Narr[ii][:, :, j].sum((1, 2))


def contrast(a, b):
    point = grp_mean(a) - grp_mean(b)
    bt = np.array([grp_mean(a, idx[i:i + 1]) - grp_mean(b, idx[i:i + 1]) for i in range(B)]).ravel()
    return point, np.nanpercentile(bt, 2.5), np.nanpercentile(bt, 97.5)


C = {}
C["c3_A"] = contrast([3], [11])
C["c2_A"] = contrast([2], [11])
C["c23_A"] = contrast([2, 3], [11])
C["c23_AD"] = contrast([2, 3], [11, 12, 13, 14])
C["c23_AB"] = contrast([2, 3], [11, 12])
C["c8_A"] = contrast([8], [11])
C["c10_G"] = contrast([10], [17])
C["built_nat"] = contrast(BUILT, [11, 12, 13, 14])
C["c8_AD"] = contrast([8], [11, 12, 13, 14])
for k, v in C.items():
    print(k, np.round(v, 2))

# Cliff's delta compact (2,3) vs A, and 3 vs D (subsampled for speed)
def cliffs(a, b, m=20000):
    a = rng.choice(a, min(m, len(a)), replace=False); b = rng.choice(b, min(m, len(b)), replace=False)
    u = stats.mannwhitneyu(a, b).statistic
    return 2 * u / (len(a) * len(b)) - 1


d23A = cliffs(g.loc[g.lcz.isin([2, 3]), DEP].to_numpy(), g.loc[g.lcz == 11, DEP].to_numpy())
print("cliff 23 vs A", d23A)

tt = t.set_index("lcz")
vals = {
    "kw_H": f"{H_all:,.1f}".replace(",", "{,}"), "kw_df": df_all, "lcz_eta_sq": fmt(e_all),
    "kw_H_types": f"{H_14:,.1f}".replace(",", "{,}"), "kw_df_types": df_14, "lcz_eta_sq_types": fmt(e_14),
    "kw_H_supported": f"{H_s:,.1f}".replace(",", "{,}"), "kw_df_supported": df_s, "lcz_eta_sq_supported": fmt(e_s),
    "n_lcz_cells": fmt_int(len(g)),
    "lcz_min_cells": fmt_int(MIN_CELLS), "lcz_min_blocks": MIN_BLOCKS,
    "n_lcz_supported": int(t.supported.sum()), "n_lcz_limited": int((~t.supported).sum()),
    "lcz_cliffs_23_A": fmt(d23A, 2), "n_compact_cells": fmt_int(g.lcz.isin([2, 3]).sum()),
}
for k, (p, lo, hi) in C.items():
    vals[f"gap_{k}"] = fmt(p, 2); vals[f"gap_{k}_lo"] = fmt(lo, 2); vals[f"gap_{k}_hi"] = fmt(hi, 2)
for c in [2, 3, 5, 8, 10, 11, 14, 17]:
    r = tt.loc[c]
    vals[f"lcz{c}_n"] = fmt_int(r.n); vals[f"lcz{c}_blocks"] = int(r.blocks); vals[f"lcz{c}_patches"] = int(r.patches)
    vals[f"lcz{c}_ci_lo"] = fmt(r.ci_lo, 2); vals[f"lcz{c}_ci_hi"] = fmt(r.ci_hi, 2)
for c_, key in [(3, "lcz3_lst_mean"), (8, "lcz8_lst_mean"), (17, "lczG_lst_mean"), (16, "lczF_lst_mean"), (11, "lczA_lst_mean"), (2, "lcz2_lst_mean"), (10, "lcz10_lst_mean")]:
    vals[key] = fmt(tt.loc[c_, "mean"], 2)
write_values("r1_01_lcz_uncertainty", vals)
