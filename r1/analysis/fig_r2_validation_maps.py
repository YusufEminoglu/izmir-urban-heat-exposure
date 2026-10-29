"""Figure 9 (map-centred): how well the reported MGWR transfers to the cells it was not calibrated on, and which
parts of the coefficient surfaces recur in independent samples. Palette of the other figures, compact layout.
(a) mean absolute hold-out error per 1 km cell, with the calibration cells;
(b) observed against predicted LST for all hold-out cells: cell density (power-scaled, so the sparse margin does not
    look as dense as the core), median prediction by 1 degC bin of observed LST with its interquartile range, 1:1 line;
(c) hold-out R2 against distance to the nearest calibration cell, with the global benchmarks;
(d) NDBI and albedo coefficient surfaces of the reported model and two independent samples.
Values quoted in the text come from r1_06 and r1_04; the slope of predicted on observed is written to
outputs/values/fig_r2_validation_maps.csv."""
import sys
from pathlib import Path
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
from mgwr_core import load_r0_fit, local_betas

INK, MUTED, GRID, LIGHT = "#344150", "#7A8590", "#E7EBEA", "#F1F3F1"
TEAL, NAVY, ROSE, PEACH, BOUNDARY = "#86B7AA", "#667F95", "#B97872", "#E5C8BF", "#2F3D4A"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#B9BEC5", "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.labelcolor": INK})
ERR = LinearSegmentedColormap.from_list("err", [LIGHT, "#EAD8D1", "#D8AAA0", ROSE, "#8E4F49"])
DIV = LinearSegmentedColormap.from_list("div_soft", [NAVY, "#BFD7D0", LIGHT, PEACH, ROSE, "#8E4F49"])
DENS = LinearSegmentedColormap.from_list("dens", ["#F6F8F6", "#BFD7D0", TEAL, NAVY, "#3B4A5A"])
C_MGWR, C_OLSF, C_OLS3, C_SEM = "#A65F58", "#6F7D84", "#4F8F85", NAVY
OUT = Path(r"REPO_ROOT\outputs\figures\paper1\final\fig09_mgwr_validation_stability.png")

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


def hcbar(ax, im, label, box=(0.60, 0.88, 0.33, 0.03), ticks=None):
    cax = ax.inset_axes(list(box))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=ticks)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.6, colors=INK, length=0, pad=1.5)
    cax.set_title(label, fontsize=8, color=MUTED, pad=3)
    return cb


def clean(ax):
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


fig = plt.figure(figsize=(13.2, 9.9))
gs = fig.add_gridspec(2, 2, left=0.055, right=0.962, top=0.95, bottom=0.065, hspace=0.20, wspace=0.13, width_ratios=[1.12, 1],
                      height_ratios=[1, 0.84])

# (a) mean absolute hold-out error, 1 km
ax_a = ax = fig.add_subplot(gs[0, 0])
rr, cc = g.row.to_numpy(int)[ho] // 10, g.col.to_numpy(int)[ho] // 10
nr, nc = rr.max() + 1, cc.max() + 1
acc = np.zeros((nr, nc)); cnt = np.zeros((nr, nc))
np.add.at(acc, (rr, cc), np.abs(pred - obs)[ho]); np.add.at(cnt, (rr, cc), 1)
mae = np.where(cnt >= 20, acc / np.maximum(cnt, 1), np.nan)
x0, y0 = 432600.0, 4359700.0
im = ax.imshow(mae, extent=[x0, x0 + nc * 1000, y0 - nr * 1000, y0], cmap=ERR, vmin=0, vmax=float(np.nanpercentile(mae, 98)), interpolation="nearest")
ax.scatter(f["coords"][:, 0], f["coords"][:, 1], s=0.35, color=INK, alpha=0.35, linewidth=0, label="calibration cells")
bnd.boundary.plot(ax=ax, color=BOUNDARY, linewidth=0.7)
b = bnd.total_bounds; ax.set_xlim(b[0] - 1500, b[2] + 1500); ax.set_ylim(b[1] - 1500, b[3] + 1500); ax.set_aspect("equal"); ax.set_axis_off()
ax.set_anchor("W")
hcbar(ax, im, "mean absolute error (\u00b0C), 1 km cells", box=(0.66, 0.86, 0.30, 0.03))
ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.02), fontsize=8, frameon=False, markerscale=6, labelcolor=INK)

# (b) observed vs predicted
ax_b = ax = fig.add_subplot(gs[0, 1])
oo, pp = obs[ho], pred[ho]
lo, hi = np.percentile(oo, [0.2, 99.8])
hb = ax.hexbin(oo, pp, gridsize=64, extent=(lo, hi, lo, hi), cmap=DENS, norm=PowerNorm(gamma=0.45), mincnt=5, linewidths=0, zorder=2)
edges = np.arange(np.floor(lo), np.ceil(hi) + 1, 1.0)
mid, med, q1, q3 = [], [], [], []
for a, c in zip(edges[:-1], edges[1:]):
    m = (oo >= a) & (oo < c)
    if m.sum() >= 200:
        mid.append((a + c) / 2); med.append(np.median(pp[m])); q1.append(np.quantile(pp[m], 0.25)); q3.append(np.quantile(pp[m], 0.75))
ax.fill_between(mid, q1, q3, color=C_MGWR, alpha=0.16, linewidth=0, zorder=3)
ax.plot(mid, med, color=C_MGWR, lw=2.0, zorder=4)
ax.plot([lo, hi], [lo, hi], color=INK, lw=1.0, ls=(0, (4, 3)), zorder=5)
slope, icpt = np.polyfit(oo, pp, 1)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
ax.set_xlabel("observed LST (\u00b0C)"); ax.set_ylabel("predicted LST (\u00b0C)")
clean(ax)
ax.text(0.04, 0.97, f"{ho.sum():,} hold-out cells\n$R^2$ = {V['ho_grid_holdout_MGWR_kernel_r2']}, RMSE = {V['ho_grid_holdout_MGWR_kernel_rmse']} \u00b0C\n"
        f"slope of predicted on observed = {slope:.2f}", transform=ax.transAxes, va="top", fontsize=8.6, color=INK, linespacing=1.35)
h = [plt.Line2D([], [], color=C_MGWR, lw=2, label="median prediction, 1 \u00b0C bins"),
     plt.Rectangle((0, 0), 1, 1, color=C_MGWR, alpha=0.25, label="interquartile range"),
     plt.Line2D([], [], color=INK, lw=1, ls=(0, (4, 3)), label="1:1")]
ax.legend(handles=h, loc="lower right", bbox_to_anchor=(1.0, 0.13), fontsize=7.8, frameon=False)
hcbar(ax, hb, "hold-out cells per hexagon", box=(0.60, 0.06, 0.36, 0.028), ticks=[100, 1000, 3000])
write_values("fig_r2_validation_maps", {"ho_pred_obs_slope": fmt(slope, 2)})

# (c) R2 against distance to the nearest calibration cell
ax_c = ax = fig.add_subplot(gs[1, 0])
dedges = np.array([0, 250, 500, 750, 1000, 1500, 2000, 3000, 5000, 1e9])
dd, oo2, pp2 = dist[ho], obs[ho], pred[ho]
r2s, xs, ns = [], [], []
for a, c in zip(dedges[:-1], dedges[1:]):
    m = (dd >= a) & (dd < c)
    if m.sum() < 200:
        continue
    r2s.append(1 - ((oo2[m] - pp2[m]) ** 2).sum() / ((oo2[m] - oo2[m].mean()) ** 2).sum()); xs.append((a + min(c, 6000)) / 2 / 1000); ns.append(m.sum())
for k, lab, col, ls in [("ho_grid_holdout_OLSfull_r2", "OLS, full specification", C_OLSF, (0, (5, 3))),
                        ("ho_grid_holdout_OLS3_r2", "OLS, 3 surface predictors", C_OLS3, (0, (1.5, 2))),
                        ("ho_grid_holdout_SEMtrend_r2", "SEM trend", C_SEM, (0, (6, 2, 1.5, 2)))]:
    ax.axhline(float(V[k]), color=col, ls=ls, lw=1.5, label=f"{lab} (all hold-out cells)", zorder=2)
ax.plot(xs, r2s, "o-", color=C_MGWR, lw=2.2, ms=6.5, mec="white", mew=1.0, label="MGWR (kernel-consistent)", zorder=4)
for x_, n_, r_ in zip(xs, ns, r2s):
    ax.text(x_, r_ + 0.010, f"{100 * n_ / ho.sum():.0f}%", ha="center", fontsize=7.6, color=MUTED)
ax.set_xlabel("distance to the nearest calibration cell (km)"); ax.set_ylabel("hold-out $R^2$")
ax.set_ylim(0.52, 0.92)
clean(ax)
ax.legend(fontsize=8, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.92))
ax.text(0.99, 0.97, "labels: share of hold-out cells in each distance band", transform=ax.transAxes, ha="right", va="top", fontsize=7.6, color=MUTED)

# (d) NDBI and albedo surfaces across samples (atlas points, kernel-consistent)
sub = gs[1, 1].subgridspec(2, 4, wspace=0.0, hspace=0.04, width_ratios=[1, 1, 1, 0.07])
at = gpd.read_file(IDW_GRID)
dst = np.column_stack([at.geometry.x, at.geometry.y])
fits = [("Reported", f)] + [(f"Sample {i}", dict(np.load(CACHE / f"mgwr_rep{i}.npz", allow_pickle=True))) for i in (1, 3)]
Bs = [local_betas(ff, dst) for _, ff in fits]
xs_, ys_ = np.unique(np.round(dst[:, 0], 1)), np.unique(np.round(dst[:, 1], 1))
ix, iy = np.searchsorted(xs_, np.round(dst[:, 0], 1)), np.searchsorted(ys_, np.round(dst[:, 1], 1))
ext = [xs_.min() - 250, xs_.max() + 250, ys_.min() - 250, ys_.max() + 250]
d_axes = []
for r, (j, nm) in enumerate([(2, "NDBI"), (3, "Albedo")]):
    vmax = float(max(np.percentile(np.abs(B_[:, j]), 99) for B_ in Bs))
    for c, ((lab, _), B_) in enumerate(zip(fits, Bs)):
        a_ = fig.add_subplot(sub[r, c])
        grid_ = np.full((len(ys_), len(xs_)), np.nan); grid_[iy, ix] = B_[:, j]
        im = a_.imshow(grid_, origin="lower", extent=ext, cmap=DIV, vmin=-vmax, vmax=vmax, interpolation="nearest")
        bnd.boundary.plot(ax=a_, color=BOUNDARY, linewidth=0.4)
        a_.set_aspect("equal"); a_.set_axis_off()
        if r == 0:
            a_.set_title(lab, fontsize=9.2, color=INK, pad=2)
        if c == 0:
            a_.text(-0.04, 0.5, nm, transform=a_.transAxes, rotation=90, ha="right", va="center", fontsize=9.2, color=INK)
        d_axes.append(a_)
    cax = fig.add_subplot(sub[r, 3])
    pos = cax.get_position(); cax.set_position([pos.x0, pos.y0 + 0.15 * pos.height, pos.width, 0.7 * pos.height])
    cbb = fig.colorbar(im, cax=cax); cbb.ax.tick_params(labelsize=7.2, colors=INK, length=0); cbb.outline.set_visible(False)

# titles aligned at figure level: rows share a baseline, columns share a left edge
fig.canvas.draw()
rend = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()


def tb(*axs):
    bb = [inv.transform_bbox(a.get_tightbbox(rend)) for a in axs]
    return min(x.x0 for x in bb), max(x.y1 for x in bb)


xa, ya = tb(ax_a); xb, yb = tb(ax_b); xc, yc = tb(ax_c); xd, yd = tb(*d_axes)
xl, xr = min(xa, xc), min(xb, xd)
y1, y2 = max(ya, yb) + 0.006, max(yc, yd) + 0.006
TS = dict(fontsize=12, fontweight="bold", color=INK, ha="left", va="bottom")
fig.text(xl, y1, "(a) Hold-out error of the reported model", **TS)
fig.text(xr, y1, "(b) Observed against predicted, hold-out cells", **TS)
fig.text(xl, y2, "(c) Accuracy with distance from the calibration cells", **TS)
fig.text(xr, y2, "(d) Coefficient surfaces in independent samples", **TS)
fig.savefig(OUT, dpi=250, facecolor="white")
fig.savefig(FIGS / OUT.name, dpi=120, facecolor="white")
print("wrote", OUT, "slope", round(slope, 3))
