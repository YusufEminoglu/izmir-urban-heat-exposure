"""Supplementary data-cube figures in the logic of the cube workbench (space-time cube, trend and
emerging hot-spot analysis) for the 2014-2026 summer LST cube.

Fig S2: (a) 3-D voxel stack of the thirteen summer slices on a 1 km grid; (b) space-time heatmap
(1 km cells x summers, cells ordered by dominant LCZ group and mean LST); (c) selected-cell series.
Fig S3: (a) Theil-Sen slope 2014-2026; (b) Mann-Kendall significance; (c) emerging hot-spot classes
(Getis-Ord Gi* per summer, Mann-Kendall on the Gi* z-series); (d) class shares by LCZ group.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from scipy import stats, ndimage

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

INK, MUTED, GRID = "#1F2933", "#667085", "#E6E8EB"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#B9BEC5", "xtick.color": MUTED, "ytick.color": MUTED})
HEAT = LinearSegmentedColormap.from_list("heat", ["#667F95", "#BFD7D0", "#F1F3F1", "#E5C8BF", "#B97872", "#8E4F49"])  # the LST colours of Figure 3
F = 10  # 1 km aggregation
mask = np.load(CACHE / "province_mask.npy")
years, cube = load_summer_series()  # 2014-2026; 2025 = analysed composite (v2)
cube[:, ~mask] = np.nan
T, NR, NC = cube.shape
nr, nc = NR // F, NC // F
c = cube[:, :nr * F, :nc * F].reshape(T, nr, F, nc, F)
cnt = np.isfinite(c).sum((2, 4))
agg = np.where(cnt >= 50, np.nanmean(c, (2, 4)), np.nan)  # 1 km cells at least half inside the province
valid = np.isfinite(agg).all(0)
print("1 km cells", valid.sum())

# dominant LCZ group per 1 km cell
g = load_grid()
lg = np.full((NR, NC), -1); lg[g.row, g.col] = np.select([g.lcz.isin(BUILT), g.lcz.isin([11, 12]), g.lcz.isin([13, 14, 15, 16]), g.lcz == 17], [0, 1, 2, 3], -1)
lgc = lg[:nr * F, :nc * F].reshape(nr, F, nc, F)
grp = np.full((nr, nc), -1)
for k in range(4):
    share = (lgc == k).sum((1, 3))
    grp = np.where((share > 0) & (share >= np.max([(lgc == j).sum((1, 3)) for j in range(4)], 0)), k, grp)
GNAME = {0: "Built LCZ 2-10", 1: "Trees LCZ A-B", 2: "Low vegetation and bare LCZ C-F", 3: "Water LCZ G"}

# ---- Mann-Kendall + Sen on 1 km series
Y = agg[:, valid].T  # cells x T
n = T
iu, ju = np.triu_indices(n, 1)
D = Y[:, ju] - Y[:, iu]
S = np.sign(D).sum(1); varS = n * (n - 1) * (2 * n + 5) / 18
Zmk = np.where(S > 0, (S - 1) / np.sqrt(varS), np.where(S < 0, (S + 1) / np.sqrt(varS), 0))
Pmk = 2 * (1 - stats.norm.cdf(np.abs(Zmk)))
sen = np.median(D / (ju - iu), 1)

# ---- Getis-Ord Gi* per summer on the 1 km grid (queen neighbourhood incl. self)
def gistar(a):
    v = np.isfinite(a); x = np.where(v, a, 0.0)
    k = np.ones((3, 3))
    s = ndimage.convolve(x, k, mode="constant"); w = ndimage.convolve(v.astype(float), k, mode="constant")
    N = v.sum(); xb = x[v].mean(); sd = np.sqrt((x[v] ** 2).mean() - xb ** 2)
    den = sd * np.sqrt((N * w - w ** 2) / (N - 1))
    z = (s - xb * w) / den
    return np.where(v, z, np.nan)


Gz = np.stack([gistar(agg[t]) for t in range(T)])[:, valid].T  # cells x T
Dg = Gz[:, ju] - Gz[:, iu]
Sg = np.sign(Dg).sum(1)
Zg = np.where(Sg > 0, (Sg - 1) / np.sqrt(varS), np.where(Sg < 0, (Sg + 1) / np.sqrt(varS), 0))
up, dn = (Zg > 1.96), (Zg < -1.96)
hot = Gz > 1.96; cold = Gz < -1.96
last_hot, last_cold = hot[:, -1], cold[:, -1]
nhot, ncold = hot.sum(1), cold.sum(1)
cls = np.full(len(Gz), "No pattern", object)
# hot-spot patterns (simplified emerging hot-spot taxonomy)
cls[last_hot & (nhot == 1)] = "New hot spot"
cls[last_hot & (nhot >= 0.9 * T) & ~up & ~dn] = "Persistent hot spot"
cls[last_hot & (nhot >= 0.9 * T) & up] = "Intensifying hot spot"
cls[last_hot & (nhot >= 0.9 * T) & dn] = "Diminishing hot spot"
cls[last_hot & (nhot < 0.9 * T) & (nhot > 1) & (cls == "No pattern")] = "Sporadic hot spot"
cls[~last_hot & (nhot >= 0.9 * (T - 1)) & (nhot > 0)] = "Historical hot spot"
cls[last_cold & (ncold >= 0.9 * T) & ~up & ~dn] = "Persistent cold spot"
cls[last_cold & (ncold >= 0.9 * T) & dn] = "Intensifying cold spot"
cls[last_cold & (ncold >= 0.9 * T) & up] = "Diminishing cold spot"
cls[last_cold & (ncold < 0.9 * T) & (cls == "No pattern")] = "Sporadic cold spot"
ORDER = ["Intensifying hot spot", "Persistent hot spot", "Diminishing hot spot", "New hot spot", "Sporadic hot spot", "Historical hot spot",
         "No pattern", "Sporadic cold spot", "Diminishing cold spot", "Persistent cold spot", "Intensifying cold spot"]
COL = ["#7F1D1D", "#B5432C", "#E0A07A", "#F59E0B", "#F2C9B0", "#C9A9A6", "#E6E8EB", "#C7D7E8", "#9DB4C0", "#4F6D8F", "#23395B"]
code = np.array([ORDER.index(c_) for c_ in cls])
counts = pd.Series(cls).value_counts()
print(counts)
gv = grp[valid]
tab = pd.crosstab(pd.Series(gv).map(GNAME), pd.Series(cls), normalize="index").reindex(columns=ORDER, fill_value=0)
tab.to_csv(TABLES / "ehsa_by_lcz_group.csv")
write_values("fig_r2_datacube", {
    "ehsa_cells": f"{valid.sum():,}".replace(",", "{,}"),
    "ehsa_intens_hot_pct": fmt(100 * np.mean(cls == "Intensifying hot spot"), 1),
    "ehsa_persist_hot_pct": fmt(100 * np.mean(cls == "Persistent hot spot"), 1),
    "ehsa_hot_any_pct": fmt(100 * np.mean(np.isin(cls, ORDER[:6])), 1),
    "ehsa_built_hot_pct": fmt(100 * tab.loc[GNAME[0], ORDER[:6]].sum(), 1),
    "ehsa_trees_cold_pct": fmt(100 * tab.loc[GNAME[1], ORDER[7:]].sum(), 1),
    "mk1km_up_pct": fmt(100 * np.mean((Pmk < 0.05) & (sen > 0)), 1), "sen1km_median": fmt(np.median(sen), 3),
})


def to_map(v):
    m = np.full((nr, nc), np.nan); m[valid] = v; return m


ext = [432600, 432600 + nc * 1000, 4359700 - nr * 1000, 4359700]
np.savez_compressed(CACHE / "ehsa_1km.npz", code=to_map(code.astype(float)), ext=np.asarray(ext, float), order=np.array(ORDER), col=np.array(COL),
                    present=np.array([o in counts.index for o in ORDER]))

# ---------------- Fig S2: space-time cube
fig = plt.figure(figsize=(13.5, 10))
ax = fig.add_axes([0.0, 0.45, 0.56, 0.53], projection="3d")
vmin, vmax = np.nanpercentile(agg, 2), np.nanpercentile(agg, 98)
Xg, Yg = np.meshgrid(np.arange(nc), np.arange(nr)[::-1])
for t in range(T):
    zt = np.where(np.isfinite(agg[t]), agg[t], np.nan)
    col = HEAT((np.clip(zt, vmin, vmax) - vmin) / (vmax - vmin))
    col[~np.isfinite(zt)] = (1, 1, 1, 0)
    ax.plot_surface(Xg, Yg, np.full_like(Xg, t, dtype=float), facecolors=col, rstride=2, cstride=2, shade=False, linewidth=0, antialiased=False)
ax.set_zticks(range(0, T, 2)); ax.set_zticklabels([years[i] for i in range(0, T, 2)], fontsize=8)
ax.set_xticks([]); ax.set_yticks([]); ax.view_init(elev=18, azim=-58); ax.set_box_aspect((nc, nr, 330))
ax.set_zlabel("summer", color=MUTED, fontsize=9)
for s in ("x", "y", "z"):
    getattr(ax, f"{s}axis").pane.set_alpha(0)
fig.text(0.02, 0.965, "(a) Space\u2013time cube of summer LST, 2014\u20132026 (1 km voxels)", fontsize=13, fontweight="bold", color=INK)
sm = plt.cm.ScalarMappable(cmap=HEAT, norm=plt.Normalize(vmin, vmax))
cax = fig.add_axes([0.08, 0.425, 0.38, 0.013]); cb = fig.colorbar(sm, cax=cax, orientation="horizontal"); cb.set_label("JJA LST (\u00b0C)", color=MUTED, fontsize=9); cb.outline.set_visible(False)

ax2 = fig.add_axes([0.62, 0.45, 0.35, 0.5])
order = np.lexsort((np.nanmean(Y, 1), gv))
im = ax2.imshow(Y[order], aspect="auto", cmap=HEAT, vmin=vmin, vmax=vmax, interpolation="nearest")
ax2.set_xticks(range(T)); ax2.set_xticklabels([str(y)[2:] for y in years], fontsize=8); ax2.set_yticks([])
bounds = np.cumsum([np.sum(gv == k) for k in range(-1, 4)])
for k, b0 in enumerate(bounds[:-1]):
    ax2.axhline(b0 - 0.5, color="white", lw=1.2)
for k in range(4):
    m = (gv[order] == k)
    if m.any():
        ax2.text(-0.6, np.mean(np.where(m)[0]), GNAME[k].replace(" LCZ", "\nLCZ"), ha="right", va="center", fontsize=8, color=INK)
ax2.set_xlabel("summer (20xx)")
fig.text(0.55, 0.965, "(b) Space\u2013time heatmap: 1 km cells \u00d7 summers", fontsize=13, fontweight="bold", color=INK)

ax3 = fig.add_axes([0.07, 0.115, 0.88, 0.222])
picks = {}
for k, colr in zip(range(4), ["#B5432C", "#1E8A5A", "#B98A1E", "#2F6FB0"]):
    idx = np.where(gv == k)[0]
    if len(idx):
        mid = idx[np.argsort(np.nanmean(Y[idx], 1))[len(idx) // 2]]
        ax3.plot(years, Y[mid], marker="o", lw=2, color=colr, label=f"median cell, {GNAME[k]}")
        ax3.plot(years, np.nanmean(Y[idx], 0), lw=1, ls=(0, (3, 3)), color=colr)
ax3.set_xticks(years); ax3.grid(True, color=GRID); ax3.set_ylabel("JJA LST (\u00b0C)")
for s in ("top", "right"):
    ax3.spines[s].set_visible(False)
ax3.legend(fontsize=8.4, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
ax3.text(0.0, 1.06, "(c) Selected-cell series (solid) and group means (dashed)", transform=ax3.transAxes, fontsize=13, fontweight="bold", color=INK)
for d in [FIGS, Path(r"REPO_ROOT\outputs\figures\paper1\final")]:
    fig.savefig(d / "figS2_spacetime_cube.png", dpi=220)
plt.close(fig)

# ---------------- Fig S3: trends and emerging hot spots
import geopandas as gpd
bnd = gpd.read_file(DATA / "izmir_province_boundary_filled.gpkg").to_crs(CRS)
fig, axs = plt.subplots(1, 2, figsize=(13, 5.9))
fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.92, wspace=0.02)
UP, NO, DN = "#A65F58", "#E6E8EB", "#667F95"
ax = axs[0]
lim = np.nanpercentile(np.abs(sen), 98)
im = ax.imshow(to_map(sen), extent=ext, cmap=HEAT, vmin=-lim, vmax=lim, interpolation="nearest")
cax = ax.inset_axes([0.60, 0.86, 0.34, 0.032])
cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.outline.set_visible(False); cb.ax.tick_params(colors=INK, length=0, labelsize=8)
cax.set_title("Theil\u2013Sen slope (\u00b0C per summer)", fontsize=8.6, color=MUTED, pad=3)
ax = axs[1]
mk = np.where(Pmk < 0.05, np.where(sen > 0, 2, 0), 1).astype(float)
ax.imshow(to_map(mk), extent=ext, cmap=ListedColormap([DN, NO, UP]), vmin=-0.5, vmax=2.5, interpolation="nearest")
h = [plt.Rectangle((0, 0), 1, 1, color=cc) for cc in [UP, NO, DN]]
ax.legend(h, ["significant increase", "no significant trend", "significant decrease"], loc="upper right", bbox_to_anchor=(0.97, 0.95),
          fontsize=8.6, frameon=False, title="Mann\u2013Kendall, p < 0.05", title_fontsize=8.6)
for ax in axs:
    bnd.boundary.plot(ax=ax, color="#2F3D4A", linewidth=0.6)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal"); ax.set_axis_off()
ytop = max(a.get_position().y1 for a in axs) + 0.012
fig.text(axs[0].get_position().x0 + 0.005, ytop, "(a) Monotonic change, 2014\u20132026", fontweight="bold", color=INK, fontsize=11.5, va="bottom")
fig.text(axs[1].get_position().x0 + 0.005, ytop, "(b) Mann\u2013Kendall trend test", fontweight="bold", color=INK, fontsize=11.5, va="bottom")
for d in [FIGS, Path(r"REPO_ROOT\outputs\figures\paper1\final")]:
    fig.savefig(d / "figS3_trend_emerging_hotspots.png", dpi=220)
print("done")
