"""Check the 2018 global LCZ map against the municipal building inventory.

The Izmir Metropolitan Municipality building inventory (901,609 footprints with
above-ground floor counts, 2026 extract) provides an independent, local and
current description of three-dimensional urban form. For every 100 m cell we
compute building surface fraction (BSF), the footprint-weighted mean height
(floors x 3 m) and the mean footprint size, at the cell and in a 3 x 3 (300 m)
window that matches the neighbourhood scale of LCZ classification.

Outputs
  * inventory morphology by WUDAPT class (median BSF, height) versus the
    Stewart and Oke (2012) property ranges
  * a rule-based inventory typology for built cells and its agreement with
    WUDAPT (built/non-built; compact/open/sparse; low/mid/high)
  * LST by inventory class with block-bootstrap intervals, and the WUDAPT class
    statistics restricted to morphology-confirmed cells
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyogrio
import shapely
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

X0, Y0, CELL = 432600.0, 4359700.0, 100.0
FLOOR_H = 3.0
rng = np.random.default_rng(11)

cache = CACHE / "buildings_cells.npz"
if not cache.exists():
    b = pyogrio.read_dataframe(BUILDINGS, columns=["ZEMINUSTUK", "YAPITIP"])
    geom = b.geometry.values
    area = shapely.area(geom)
    cen = shapely.centroid(geom)
    bx, by = shapely.get_x(cen), shapely.get_y(cen)
    fl = b["ZEMINUSTUK"].to_numpy(float)
    fl = np.where(fl < 1, 1, fl)
    np.savez_compressed(cache, x=bx, y=by, area=area, floors=fl,
                        typ=b["YAPITIP"].astype(str).to_numpy())
z = np.load(cache, allow_pickle=True)
bx, by, area, fl, typ = z["x"], z["y"], z["area"], z["floors"], z["typ"]
ok = np.isfinite(area) & (area > 5) & (area < 200_000)
bx, by, area, fl, typ = bx[ok], by[ok], area[ok], fl[ok], typ[ok]
print("buildings used", ok.sum(), "of", len(ok), "median floors", np.median(fl))

g = load_grid(valid_only=False).dropna(subset=[DEP, "lcz"]).reset_index(drop=True)
g["lcz"] = g["lcz"].astype(int)
nr, nc = int(g.row.max()) + 1, int(g.col.max()) + 1
col = np.floor((bx - X0) / CELL).astype(int)
row = np.floor((Y0 - by) / CELL).astype(int)
inb = (row >= 0) & (row < nr) & (col >= 0) & (col < nc)
print("buildings inside grid extent", inb.mean())
row, col, area, fl = row[inb], col[inb], area[inb], fl[inb]


def rast(w):
    a = np.zeros((nr, nc))
    np.add.at(a, (row, col), w)
    return a


A_fp = rast(area)               # footprint m2
A_h = rast(area * fl * FLOOR_H)  # footprint-weighted height
N_b = rast(np.ones_like(area))
k3 = np.ones((3, 3))
A_fp3 = ndimage.convolve(A_fp, k3, mode="constant")
A_h3 = ndimage.convolve(A_h, k3, mode="constant")
N_b3 = ndimage.convolve(N_b, k3, mode="constant")
r, c = g.row.to_numpy(), g.col.to_numpy()
g["bsf"] = A_fp[r, c] / 1e4
g["bsf3"] = A_fp3[r, c] / 9e4
g["h3"] = np.where(A_fp3[r, c] > 0, A_h3[r, c] / np.maximum(A_fp3[r, c], 1e-9), 0)
g["fp3"] = np.where(N_b3[r, c] > 0, A_fp3[r, c] / np.maximum(N_b3[r, c], 1), 0)
g["nb3"] = N_b3[r, c]

built = g.lcz.isin(BUILT)
cov = (g.loc[built, "nb3"] > 0).mean()
print("WUDAPT built cells with >=1 inventory building in 300 m window:", cov)


def inv_class(bsf, h, fp):
    cls = np.full(len(bsf), "none", object)
    sparse = (bsf >= 0.05) & (bsf < 0.20)
    dense = bsf >= 0.20
    large = dense & (fp >= 1500) & (h <= 12)
    cls[sparse] = "9"
    comp = dense & ~large & (bsf >= 0.40)
    opn = dense & ~large & (bsf < 0.40)
    lo, mid, hi = h < 10, (h >= 10) & (h < 25), h >= 25
    cls[comp & hi] = "1"; cls[comp & mid] = "2"; cls[comp & lo] = "3"
    cls[opn & hi] = "4"; cls[opn & mid] = "5"; cls[opn & lo] = "6"
    cls[large] = "8"
    return cls


g["inv"] = inv_class(g.bsf3.to_numpy(), g.h3.to_numpy(), g.fp3.to_numpy())

# 1. inventory morphology by WUDAPT built class
SO = {2: ("40-70", "10-25"), 3: ("40-70", "3-10"), 5: ("20-40", "10-25"), 6: ("20-40", "3-10"),
      8: ("30-50", "3-10"), 9: ("10-20", "3-10"), 10: ("20-30", "5-15")}
rows = []
for k in [2, 3, 5, 6, 8, 9, 10]:
    s = g[g.lcz == k]
    rows.append(dict(lcz=k, n=len(s), cov=(s.nb3 > 0).mean(), bsf3_med=s.bsf3.median(), bsf3_q25=s.bsf3.quantile(.25),
                     bsf3_q75=s.bsf3.quantile(.75), h3_med=s.h3[s.nb3 > 0].median(), h3_q25=s.h3[s.nb3 > 0].quantile(.25),
                     h3_q75=s.h3[s.nb3 > 0].quantile(.75), so_bsf=SO[k][0], so_h=SO[k][1]))
T1 = pd.DataFrame(rows)
T1.to_csv(TABLES / "lcz_inventory_morphology.csv", index=False)
print(T1.round(3).to_string())

# 2. agreement
inv_built = g.bsf3 >= 0.05
bb = pd.crosstab(built, inv_built)
print(bb)
agree_built = (built == inv_built).mean()
prec_built = (built & inv_built).sum() / built.sum()          # WUDAPT built confirmed by inventory
recall_built = (built & inv_built).sum() / inv_built.sum()    # inventory fabric labelled built by WUDAPT
ct = pd.crosstab(g.loc[built, "lcz"], g.loc[built, "inv"])
ct.to_csv(TABLES / "lcz_inventory_confusion.csv")
print(ct)
# compactness dimension agreement for WUDAPT 2,3 (compact) / 5,6 (open) / 9 (sparse)
dim_w = g.lcz.map({2: "compact", 3: "compact", 5: "open", 6: "open", 9: "sparse", 8: "large", 10: "large"})
dim_i = g.inv.map({"1": "compact", "2": "compact", "3": "compact", "4": "open", "5": "open", "6": "open", "9": "sparse", "8": "large", "none": "none"})
mask = dim_w.notna()
dim_agree = (dim_w[mask] == dim_i[mask]).mean()
# height dimension for compact/open classes
hw = g.lcz.map({2: "mid", 5: "mid", 3: "low", 6: "low"})
hi_ = pd.cut(g.h3, [-1, 10, 25, 1e9], labels=["low", "mid", "high"]).astype(str)
m2 = hw.notna() & (g.nb3 > 0)
h_agree = (hw[m2] == hi_[m2]).mean()
# inventory-built cells per class beyond WUDAPT
inv_counts = g.inv.value_counts()
print(inv_counts)
mid_inv = g.inv.isin(["2", "5"]).sum()
print("dimension agreement", dim_agree, "height agreement", h_agree)


# 3. LST by inventory class with 2 km block bootstrap
g["blk"] = (g.row // 20) * 100000 + (g.col // 20)


def boot_ci(s, B=2000):
    bs = s.groupby("blk")[DEP].agg(["sum", "count"])
    a, n = bs["sum"].to_numpy(), bs["count"].to_numpy()
    idx = rng.integers(0, len(a), size=(B, len(a)))
    m = a[idx].sum(1) / n[idx].sum(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5), len(a)


rows = []
for k in ["1", "2", "3", "4", "5", "6", "8", "9"]:
    s = g[g.inv == k]
    if len(s) < 5:
        continue
    lo, hi, nb = boot_ci(s)
    rows.append(dict(inv=k, n=len(s), blocks=nb, mean=s[DEP].mean(), ci_lo=lo, ci_hi=hi, sd=s[DEP].std(),
                     wudapt_same=(g.loc[g.inv == k, "lcz"].astype(str) == k).mean()))
T3 = pd.DataFrame(rows)
T3.to_csv(TABLES / "inventory_class_lst.csv", index=False)
print(T3.round(3).to_string())

# 4. WUDAPT classes restricted to morphology-confirmed cells (compactness dimension agrees)
conf = mask & (dim_w == dim_i)
rows = []
for k in [2, 3, 5, 6, 8, 9, 10]:
    s = g[(g.lcz == k) & conf]
    rows.append(dict(lcz=k, n_conf=len(s), share_conf=len(s) / max((g.lcz == k).sum(), 1),
                     mean_conf=s[DEP].mean() if len(s) else np.nan, mean_all=g.loc[g.lcz == k, DEP].mean()))
T4 = pd.DataFrame(rows)
T4.to_csv(TABLES / "lcz_confirmed_lst.csv", index=False)
print(T4.round(3).to_string())

mA = g.loc[g.lcz == 11, DEP].mean()
inv_compact = g.inv.isin(["2", "3"])
gap_inv_compact_A = g.loc[inv_compact, DEP].mean() - mA
t3 = T3.set_index("inv")
vals = {
    "inv_buildings": fmt_int(ok.sum()), "inv_cov_built": fmt(100 * cov, 1),
    "inv_agree_built": fmt(100 * agree_built, 1), "inv_prec_built": fmt(100 * prec_built, 1),
    "inv_recall_built": fmt(100 * recall_built, 1), "inv_dim_agree": fmt(100 * dim_agree, 1),
    "inv_height_agree": fmt(100 * h_agree, 1),
    "inv_n_compact": fmt_int(inv_compact.sum()), "inv_n_mid": fmt_int(mid_inv),
    "gap_inv_compact_A": fmt(gap_inv_compact_A, 2),
}
for k in ["2", "3", "5", "6", "8", "9"]:
    if k in t3.index:
        vals[f"inv{k}_n"] = fmt_int(t3.loc[k, "n"]); vals[f"inv{k}_mean"] = fmt(t3.loc[k, "mean"], 2)
        vals[f"inv{k}_lo"] = fmt(t3.loc[k, "ci_lo"], 2); vals[f"inv{k}_hi"] = fmt(t3.loc[k, "ci_hi"], 2)
for _, rr in T1.iterrows():
    k = int(rr.lcz)
    vals[f"lcz{k}_bsf_med"] = fmt(100 * rr.bsf3_med, 1); vals[f"lcz{k}_h_med"] = fmt(rr.h3_med, 1)
for _, rr in T4.iterrows():
    vals[f"lcz{int(rr.lcz)}_conf_share"] = fmt(100 * rr.share_conf, 1)
    vals[f"lcz{int(rr.lcz)}_conf_mean"] = fmt(rr.mean_conf, 2) if np.isfinite(rr.mean_conf) else "--"
write_values("r1_03_lcz_building_validation", vals)
g[["row", "col", "bsf", "bsf3", "h3", "fp3", "nb3", "inv"]].to_pickle(CACHE / "grid_inventory.pkl")
