"""Sensitivity of the coefficient atlas to the interpolation scheme.

Coefficients of the reported MGWR (refit 'orig') are mapped to the 500 m atlas
grid and to every 100 m cell with eight schemes:
  IDW p=2 k=12 (baseline); IDW p=1 k=12; IDW p=2 k=6; IDW p=2 k=24;
  nearest neighbour; linear (Delaunay); LCZ-stratified IDW (built cells from
  built sample points, natural from natural); and kernel-consistent direct
  estimation with the model's own per-covariate bandwidths (no interpolation).
Metrics against the baseline: Pearson r, sign agreement, dominant-driver
agreement, LCZ-class medians; leave-one-out reproduction of the sampled
coefficients for each interpolator; mahalle-level agreement of the
coefficient-derived morphology term.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
from mgwr_core import idw, local_betas, load_r0_fit

f = load_r0_fit()
src = f["coords"]; B = f["params"][:, 1:]  # NDVI, NDBI, albedo (standardised)
lcz_s = f["lcz"]
atlas = gpd.read_file(IDW_GRID)
axy = np.column_stack([atlas.geometry.x, atlas.geometry.y])
g = load_grid()
gxy = g[["x", "y"]].to_numpy(float)
NAMES = ["ndvi", "ndbi", "albedo"]

# LCZ of atlas points (from nearest 100 m cell)
t = cKDTree(gxy)
_, ii = t.query(axy, k=1)
lcz_a = g.lcz.to_numpy()[ii]


def lin(dst):
    out = np.empty((len(dst), 3))
    nn = idw(src, B, dst, k=1)
    for k in range(3):
        v = LinearNDInterpolator(src, B[:, k])(dst)
        out[:, k] = np.where(np.isfinite(v), v, nn[:, k])
    return out


def strat(dst, lcz_d):
    out = np.empty((len(dst), 3))
    bs, bd = np.isin(lcz_s, BUILT), np.isin(lcz_d, BUILT)
    out[bd] = idw(src[bs], B[bs], dst[bd], k=min(12, bs.sum()))
    out[~bd] = idw(src[~bs], B[~bs], dst[~bd], k=12)
    return out


def methods(dst, lcz_d):
    return {
        "idw_p2_k12": idw(src, B, dst, 12, 2), "idw_p1_k12": idw(src, B, dst, 12, 1),
        "idw_p2_k6": idw(src, B, dst, 6, 2), "idw_p2_k24": idw(src, B, dst, 24, 2),
        "nearest": idw(src, B, dst, k=1), "linear": lin(dst), "lcz_strat_idw": strat(dst, lcz_d),
        "kernel_direct": local_betas(f, dst)[:, 1:],
    }


M = methods(axy, lcz_a)
base = M["idw_p2_k12"]
print("stored atlas vs recomputed baseline r:", [np.corrcoef(base[:, k], atlas[f"mgwr_coef_{n}"])[0, 1].round(4) for k, n in enumerate(NAMES)])
dom = lambda A: np.argmax(np.abs(A), 1)
rows = []
for name, A in M.items():
    r = [np.corrcoef(A[:, k], base[:, k])[0, 1] for k in range(3)]
    sg = [np.mean(np.sign(A[:, k]) == np.sign(base[:, k])) for k in range(3)]
    rows.append(dict(method=name, r_ndvi=r[0], r_ndbi=r[1], r_alb=r[2], sign_ndvi=sg[0], sign_ndbi=sg[1], sign_alb=sg[2],
                     dom_agree=np.mean(dom(A) == dom(base))))
T = pd.DataFrame(rows)
print(T.round(3).to_string())

# LCZ-class medians of the coefficients per method (atlas points)
med = []
for name, A in M.items():
    for k in [3, 6, 8, 9, 11, 12, 13, 14]:
        m = lcz_a == k
        if m.sum() < 20:
            continue
        med.append(dict(method=name, lcz=k, **{n: np.median(A[m, j]) for j, n in enumerate(NAMES)}))
MED = pd.DataFrame(med)
MED.to_csv(TABLES / "atlas_lcz_medians_by_method.csv", index=False)
rng_med = MED.groupby("lcz")[NAMES].agg(lambda v: v.max() - v.min())
print(rng_med.round(3))

# leave-one-out reproduction of the sampled coefficients
tree = cKDTree(src)
def loo(kind):
    if kind == "nearest":
        d, i = tree.query(src, k=2); return B[i[:, 1]]
    k, p = {"idw_p2_k12": (12, 2), "idw_p1_k12": (12, 1), "idw_p2_k6": (6, 2), "idw_p2_k24": (24, 2)}[kind]
    d, i = tree.query(src, k=k + 1)
    d, i = np.maximum(d[:, 1:], 1), i[:, 1:]
    w = d ** (-p)
    return (w[..., None] * B[i]).sum(1) / w.sum(1)[:, None]
loo_rows = []
for kind in ["idw_p2_k12", "idw_p1_k12", "idw_p2_k6", "idw_p2_k24", "nearest"]:
    P = loo(kind)
    r2 = [1 - ((P[:, k] - B[:, k]) ** 2).sum() / ((B[:, k] - B[:, k].mean()) ** 2).sum() for k in range(3)]
    loo_rows.append(dict(method=kind, r2_ndvi=r2[0], r2_ndbi=r2[1], r2_alb=r2[2]))
LOO = pd.DataFrame(loo_rows)
print(LOO.round(3).to_string())
T = T.merge(LOO, on="method", how="left")
T.to_csv(TABLES / "atlas_interpolation_sensitivity.csv", index=False)

# mahalle-level coefficient term (coefficient-only definition) under each scheme at 100 m cells
mah = gpd.read_file(MAHALLE).to_crs(CRS)
pri = pd.read_csv(PRIORITY_CSV, encoding="utf-8-sig")
pri = pri[pri.lst_mean.notna()]
pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(g.x, g.y), crs=CRS)
j = gpd.sjoin(pts, mah[["join_key", "geometry"]], predicate="within", how="left")
j = j[~j.index.duplicated()]
key = j["join_key"].to_numpy()
inr = pd.Series(key).isin(pri.join_key).to_numpy()
G = methods(gxy[inr], g.lcz.to_numpy()[inr])
zz = lambda v: (v - v.mean()) / v.std()
mrows = []
Mterm = {}
for name, A in G.items():
    term = 0.45 * zz(np.maximum(A[:, 1], 0)) + 0.35 * zz(np.maximum(A[:, 2], 0)) + 0.20 * zz(np.maximum(-A[:, 0], 0))
    Mterm[name] = pd.Series(term).groupby(key[inr]).mean()
bm = Mterm["idw_p2_k12"]
W0 = np.array([0.38, 0.30, 0.22, 0.10])
def score(Mv):
    d = pri.set_index("join_key")
    Mv = Mv.reindex(d.index)
    q = lambda v: (v - v.mean()) / v.std()
    Q = W0[0] * q(d.lst_mean) + W0[1] * q(Mv) + W0[2] * q(d.vulnerable_share) + W0[3] * q(np.log1p(d.pop_total))
    return Q
Qb = score(bm)
rb = Qb.rank(ascending=False)
for name, mv in Mterm.items():
    Q = score(mv); r = Q.rank(ascending=False)
    mrows.append(dict(method=name, rho_M=stats.spearmanr(mv.reindex(bm.index), bm).statistic,
                      rho_P=stats.spearmanr(Q, Qb).statistic, top10=int(((r <= 10) & (rb <= 10)).sum()),
                      top80=int(((r <= 80) & (rb <= 80)).sum())))
MM = pd.DataFrame(mrows)
MM.to_csv(TABLES / "atlas_interp_mahalle.csv", index=False)
print(MM.round(3).to_string())

alt = T[T.method != "idw_p2_k12"]
mm = MM[MM.method != "idw_p2_k12"]
kd = T.set_index("method").loc["kernel_direct"]
vals = {
    "interp_n_methods": len(M),
    "interp_r_min": fmt(alt[["r_ndvi", "r_ndbi", "r_alb"]].min().min(), 3),
    "interp_r_min_nonn": fmt(alt[alt.method != "nearest"][["r_ndvi", "r_ndbi", "r_alb"]].min().min(), 2),
    "interp_sign_min": fmt(100 * alt[["sign_ndvi", "sign_ndbi", "sign_alb"]].min().min(), 1),
    "interp_sign_ndbi_min": fmt(100 * alt.sign_ndbi.min(), 1), "interp_sign_alb_min": fmt(100 * alt.sign_alb.min(), 1),
    "interp_dom_min": fmt(100 * alt.dom_agree.min(), 1), "interp_dom_median": fmt(100 * alt.dom_agree.median(), 1),
    "kd_r_ndvi": fmt(kd.r_ndvi, 2), "kd_r_ndbi": fmt(kd.r_ndbi, 2), "kd_r_alb": fmt(kd.r_alb, 2), "kd_dom": fmt(100 * kd.dom_agree, 1),
    "loo_r2_ndvi": fmt(LOO.set_index("method").loc["idw_p2_k12", "r2_ndvi"], 3),
    "loo_r2_ndbi": fmt(LOO.set_index("method").loc["idw_p2_k12", "r2_ndbi"], 3),
    "loo_r2_alb": fmt(LOO.set_index("method").loc["idw_p2_k12", "r2_alb"], 3),
    "interp_mah_rhoM_min": fmt(mm.rho_M.min(), 3), "interp_mah_rhoP_min": fmt(mm.rho_P.min(), 3),
    "interp_mah_top10_min": int(mm.top10.min()), "interp_mah_top80_min": int(mm.top80.min()),
    "lcz_med_range_max": fmt(rng_med.to_numpy().max(), 3),
}
write_values("r1_05_interpolation_sensitivity", vals)
np.save(CACHE / "atlas_kernel_direct.npy", M["kernel_direct"].astype(np.float32))
