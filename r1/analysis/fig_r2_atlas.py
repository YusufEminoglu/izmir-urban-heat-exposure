"""Figure 8: MGWR coefficient atlas of the reported fit (500 m inverse-distance atlas, r2_29), in the palette of the
other figures: (a)-(c) local NDVI, NDBI and albedo coefficients on one symmetric diverging scale, (d) dominant local
driver (term colours of Figure 7b), (e) local R2 (1st-99th percentile), (f) local condition number clipped at its 98th
percentile. Compact 2 x 3 layout; each map carries a small colour bar in its empty north-east; titles aligned."""
import os
import sys
from pathlib import Path
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

INK, MUTED, LIGHT = "#344150", "#7A8590", "#F1F3F1"
TEAL, NAVY, ROSE, PEACH, BOUNDARY = "#86B7AA", "#667F95", "#B97872", "#E5C8BF", "#2F3D4A"
CMAP_DIV = LinearSegmentedColormap.from_list("div_soft", [NAVY, "#BFD7D0", LIGHT, PEACH, ROSE, "#8E4F49"])
CMAP_COOL = LinearSegmentedColormap.from_list("cool_soft", [LIGHT, "#BFD7D0", TEAL, NAVY])
CMAP_GRAY = LinearSegmentedColormap.from_list("gray_soft", ["#FFFFFF", "#DADDD8", "#9EA8A8", "#5F6F76", "#3B474F"])
TERM = {"ndvi": "#4F8F85", "ndbi": "#A65F58", "albedo": "#6F7D84"}  # as in Figure 7b
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

OUT = Path(r"REPO_ROOT\outputs\figures\paper1\final\fig07_mgwr_coefficient_atlas.png")
OUT = Path(os.environ.get("ATLAS_OUT", str(OUT)))
at = gpd.read_file(IDW_GRID_V1 if os.environ.get("ATLAS_V1") else IDW_GRID)
bnd = gpd.read_file(DATA / "izmir_province_boundary_filled.gpkg").to_crs(CRS)
x, y = at.geometry.x.to_numpy(), at.geometry.y.to_numpy()
xs, ys_ = np.unique(np.round(x, 3)), np.unique(np.round(y, 3))
ix = np.searchsorted(xs, np.round(x, 3)); iy = np.searchsorted(ys_, np.round(y, 3))
ext = [xs.min() - 250, xs.max() + 250, ys_.min() - 250, ys_.max() + 250]


def grid(v):
    a = np.full((len(ys_), len(xs)), np.nan)
    a[iy, ix] = v
    return a


W, H = ext[1] - ext[0], ext[3] - ext[2]
fig = plt.figure(figsize=(15.0, 15.0 * (2 * H / W) / 3 * 1.0 + 0.9))
gs = fig.add_gridspec(2, 3, left=0.005, right=0.995, top=0.95, bottom=0.005, wspace=0.01, hspace=0.07)
axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])


def cbar(ax, im, label, ticks=None):
    cax = ax.inset_axes([0.60, 0.86, 0.33, 0.032])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=ticks)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK, length=0, pad=1.5)
    cax.set_title(label, fontsize=8.4, color=MUTED, pad=3)


vmax = float(np.nanmax(np.abs(at[["mgwr_coef_ndvi", "mgwr_coef_ndbi", "mgwr_coef_albedo"]].to_numpy())))
titles = ["(a) Local NDVI coefficient", "(b) Local NDBI coefficient", "(c) Local albedo coefficient",
          "(d) Dominant local driver", "(e) Local $R^2$", "(f) Local condition number"]
for ax, col in zip(axes[0], ["mgwr_coef_ndvi", "mgwr_coef_ndbi", "mgwr_coef_albedo"]):
    im = ax.imshow(grid(at[col]), origin="lower", extent=ext, cmap=CMAP_DIV, vmin=-vmax, vmax=vmax, interpolation="nearest")
    cbar(ax, im, "standardised coefficient", ticks=[-0.6, -0.3, 0, 0.3, 0.6])
dd = at.dominant_driver.str.lower().map({"ndvi": 0, "ndbi": 1, "albedo": 2}).to_numpy(float)
cols = [TERM["ndvi"], TERM["ndbi"], TERM["albedo"]]
axes[1, 0].imshow(grid(dd), origin="lower", extent=ext, cmap=ListedColormap(cols), vmin=-0.5, vmax=2.5, interpolation="nearest", alpha=0.92)
shares = [np.mean(dd == k) * 100 for k in range(3)]
axes[1, 0].legend(handles=[Patch(color=c, label=f"{l} ({s:.1f}% of cells)" if s > 0 else f"{l} (no cells)")
                           for c, l, s in zip(cols, ["NDVI", "NDBI", "albedo"], shares)],
                  loc="upper right", bbox_to_anchor=(0.985, 0.97), fontsize=8.6, frameon=False, title="largest |coefficient|",
                  title_fontsize=8.4, handlelength=1.3)
axes[1, 0].get_legend().get_title().set_color(MUTED)
r1, r99 = np.nanpercentile(at.local_r2, [1, 99])
im = axes[1, 1].imshow(grid(at.local_r2), origin="lower", extent=ext, cmap=CMAP_COOL, interpolation="nearest", vmin=r1, vmax=r99)
cbar(axes[1, 1], im, "local $R^2$")
c98 = float(np.nanpercentile(at.local_cond, 98))
im = axes[1, 2].imshow(grid(np.minimum(at.local_cond, c98)), origin="lower", extent=ext, cmap=CMAP_GRAY, interpolation="nearest",
                       vmin=float(np.nanmin(at.local_cond)), vmax=c98)
cbar(axes[1, 2], im, "condition number")
for ax in axes.ravel():
    bnd.boundary.plot(ax=ax, color=BOUNDARY, linewidth=0.6)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    ax.set_axis_off()
# titles: one baseline per row, at the left edge of each map
fig.canvas.draw()
inv = fig.transFigure.inverted()
for r in range(2):
    pos = [ax.get_position() for ax in axes[r]]
    ytop = max(p.y1 for p in pos) + 0.006
    for c, p in enumerate(pos):
        fig.text(p.x0 + 0.004, ytop, titles[3 * r + c], fontsize=12.5, fontweight="bold", color=INK, ha="left", va="bottom")
fig.savefig(OUT, dpi=220, facecolor="white")
print("wrote", OUT, "vmax", round(vmax, 3), "cond98", round(c98, 2), "dominant shares", [round(s, 1) for s in shares])
