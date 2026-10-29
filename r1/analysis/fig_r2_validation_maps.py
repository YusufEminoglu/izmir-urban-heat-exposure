"""Figure 9 (map-centred): how well the reported MGWR transfers to the cells it was not calibrated on, and which
parts of the coefficient surfaces recur in independent samples.
(a) mean absolute hold-out error per 1 km cell, with the calibration cells;
(b) observed against predicted LST for all hold-out cells (density);
(c) hold-out R2 against distance to the nearest calibration cell, with the global benchmarks;
(d) NDBI and albedo coefficient surfaces of the reported model and two independent samples.
Values quoted in the text come from r1_06 and r1_04; this script only draws."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
from mgwr_core import load_r0_fit, local_betas

INK, MUTED, GRID = "#344150", "#7A8590", "#E6E8EB"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#B9BEC5", "xtick.color": MUTED, "ytick.color": MUTED})
ERR = LinearSegmentedColormap.from_list("err", ["#F4F6F3", "#D8E2DE", "#E8C2B4", "#C2665A", "#7F1D1D"])
DIV = LinearSegmentedColormap.from_list("div", ["#3D5A80", "#9DB4C0", "#F2F2F0", "#E0A07A", "#A53E2B"])
OUT = Path(r"REPO_ROOT\outputs\figures\paper1\final\fig09_mgwr_validation_stability.png")


def tag(ax, s):
    ax.text(0.0, 1.02, s, transform=ax.transAxes, fontsize=12, fontweight="bold", color=INK, ha="left", va="bottom")


V = {}
for line in open(VALUES / "r1_06_mgwr_validation.csv", encoding="utf-8-sig"):
    k, _, x = line.strip().partition(",")
    V[k] = x
f = load_r0_fit()
g = load_grid()
B = np.load(CACHE / "betas_fullgrid_r0.npy").astype(float)
assert len(B) == len(g)
Xs = (g[PREDICTORS].to_numpy(float) - f["xm"]) / f["xs"]
pred = (np.column_stack([np.ones(len(Xs)), Xs]) * B).sum(1) * f["ys"] + f["ym"]
obs = g[DEP].to_numpy(float)
key = g.row.to_numpy(np.int64) * 10000 + g.col.to_numpy(np.int64)
cal = set((f["rows"].astype(np.int64) * 10000 + f["cols"].astype(np.int64)).tolist())
ho = ~np.isin(key, list(cal))
dist = cKDTree(f["coords"]).query(g[["x", "y"]].to_numpy(float))[0]
bnd = gpd.read_file(DATA / "izmir_province_boundary_filled.gpkg").to_crs(CRS)

fig = plt.figure(figsize=(13.2, 11.4))
gs = fig.add_gridspec(2, 2, left=0.04, right=0.93, top=0.955, bottom=0.06, hspace=0.22, wspace=0.30, width_ratios=[1.12, 1])

# (a) mean absolute hold-out error, 1 km
ax = fig.add_subplot(gs[0, 0])
rr, cc = g.row.to_numpy(int)[ho] // 10, g.col.to_numpy(int)[ho] // 10
nr, nc = rr.max() + 1, cc.max() + 1
acc = np.zeros((nr, nc)); cnt = np.zeros((nr, nc))
np.add.at(acc, (rr, cc), np.abs(pred - obs)[ho]); np.add.at(cnt, (rr, cc), 1)
mae = np.where(cnt >= 20, acc / np.maximum(cnt, 1), np.nan)
x0, y0 = 432600.0, 4359700.0
im = ax.imshow(mae, extent=[x0, x0 + nc * 1000, y0 - nr * 1000, y0], cmap=ERR, vmin=0, vmax=float(np.nanpercentile(mae, 98)), interpolation="nearest")
ax.scatter(f["coords"][:, 0], f["coords"][:, 1], s=0.35, color=INK, alpha=0.35, linewidth=0, label="calibration cells")
bnd.boundary.plot(ax=ax, color=INK, linewidth=0.8)
b = bnd.total_bounds; ax.set_xlim(b[0] - 2000, b[2] + 2000); ax.set_ylim(b[1] - 2000, b[3] + 2000); ax.set_aspect("equal"); ax.set_axis_off()
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01); cb.set_label("mean absolute error (°C), 1 km cells", color=INK); cb.outline.set_visible(False)
ax.legend(loc="lower left", fontsize=8, frameon=False, markerscale=6)
tag(ax, "(a) Hold-out error of the reported model")

# (b) observed vs predicted
ax = fig.add_subplot(gs[0, 1])
hb = ax.hexbin(obs[ho], pred[ho], gridsize=70, bins="log", cmap=LinearSegmentedColormap.from_list("d", ["#F4F6F3", "#9DB4C0", "#3D5A80", "#1F2933"]), mincnt=1, linewidths=0)
lo, hi = np.percentile(obs[ho], [0.2, 99.8])
ax.plot([lo, hi], [lo, hi], color="#A53E2B", lw=1.2)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("observed LST (°C)"); ax.set_ylabel("predicted LST (°C)")
ax.text(0.04, 0.96, f"{ho.sum():,} hold-out cells\n$R^2$ = {V['ho_grid_holdout_MGWR_kernel_r2']}, RMSE = {V['ho_grid_holdout_MGWR_kernel_rmse']} °C",
        transform=ax.transAxes, va="top", fontsize=9, color=INK)
cb = fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.01); cb.set_label("cells (log scale)", color=INK); cb.outline.set_visible(False)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(True, color=GRID, lw=0.6)
tag(ax, "(b) Observed against predicted, hold-out cells")

# (c) R2 against distance to the nearest calibration cell
ax = fig.add_subplot(gs[1, 0])
edges = np.array([0, 250, 500, 750, 1000, 1500, 2000, 3000, 5000, 1e9])
dd, oo, pp = dist[ho], obs[ho], pred[ho]
r2s, xs, ns = [], [], []
for a, c in zip(edges[:-1], edges[1:]):
    m = (dd >= a) & (dd < c)
    if m.sum() < 200:
        continue
    r2s.append(1 - ((oo[m] - pp[m]) ** 2).sum() / ((oo[m] - oo[m].mean()) ** 2).sum()); xs.append((a + min(c, 6000)) / 2 / 1000); ns.append(m.sum())
ax.plot(xs, r2s, "o-", color="#A53E2B", lw=2, ms=6, label="MGWR (kernel-consistent)")
for k, lab, col, ls in [("ho_grid_holdout_OLSfull_r2", "OLS, full specification", "#B98A1E", "--"), ("ho_grid_holdout_OLS3_r2", "OLS, 3 surface predictors", "#2F6FB0", ":"),
                        ("ho_grid_holdout_SEMtrend_r2", "SEM trend", "#1E8A5A", "-.")]:
    ax.axhline(float(V[k]), color=col, ls=ls, lw=1.4, label=f"{lab} (all hold-out cells)")
for x_, n_, r_ in zip(xs, ns, r2s):
    ax.text(x_, r_ + 0.012, f"{100 * n_ / ho.sum():.0f}%", ha="center", fontsize=7.5, color=MUTED)
ax.set_xlabel("distance to the nearest calibration cell (km)"); ax.set_ylabel("hold-out $R^2$")
ax.set_ylim(0.4, 1.0); ax.grid(True, color=GRID, lw=0.6)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.legend(fontsize=8, frameon=False, loc="lower left")
ax.text(0.99, 0.97, "labels: share of hold-out cells in each distance band", transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)
tag(ax, "(c) Accuracy with distance from the calibration cells")

# (d) NDBI and albedo surfaces across samples (atlas points, kernel-consistent)
sub = gs[1, 1].subgridspec(2, 3, wspace=0.02, hspace=0.06)
at = gpd.read_file(IDW_GRID)
dst = np.column_stack([at.geometry.x, at.geometry.y])
fits = [("Reported", f)] + [(f"Sample {i}", dict(np.load(CACHE / f"mgwr_rep{i}.npz", allow_pickle=True))) for i in (1, 3)]
Bs = [local_betas(ff, dst) for _, ff in fits]
xs_, ys_ = np.unique(np.round(dst[:, 0], 1)), np.unique(np.round(dst[:, 1], 1))
ix, iy = np.searchsorted(xs_, np.round(dst[:, 0], 1)), np.searchsorted(ys_, np.round(dst[:, 1], 1))
ext = [xs_.min() - 250, xs_.max() + 250, ys_.min() - 250, ys_.max() + 250]
first = None
for r, (j, nm) in enumerate([(2, "NDBI"), (3, "Albedo")]):
    vmax = float(max(np.percentile(np.abs(B_[:, j]), 99) for B_ in Bs))
    for c, ((lab, _), B_) in enumerate(zip(fits, Bs)):
        a_ = fig.add_subplot(sub[r, c])
        grid_ = np.full((len(ys_), len(xs_)), np.nan); grid_[iy, ix] = B_[:, j]
        im = a_.imshow(grid_, origin="lower", extent=ext, cmap=DIV, vmin=-vmax, vmax=vmax, interpolation="nearest")
        bnd.boundary.plot(ax=a_, color=INK, linewidth=0.4)
        a_.set_aspect("equal"); a_.set_axis_off()
        if r == 0:
            a_.set_title(lab, fontsize=9, color=INK)
        if c == 0:
            a_.text(-0.06, 0.5, nm, transform=a_.transAxes, rotation=90, ha="right", va="center", fontsize=9, color=INK)
        if r == 0 and c == 0:
            first = a_
    cax = a_.inset_axes([1.03, 0.1, 0.05, 0.8])
    cbb = fig.colorbar(im, cax=cax); cbb.ax.tick_params(labelsize=7); cbb.outline.set_visible(False)
_p = first.get_position(); fig.text(_p.x0, _p.y1 + 0.035, "(d) Coefficient surfaces in independent samples", fontsize=12, fontweight="bold", color=INK, ha="left", va="bottom")
fig.savefig(OUT, dpi=250, facecolor="white")
fig.savefig(FIGS / OUT.name, dpi=120, facecolor="white")
print("wrote", OUT)
