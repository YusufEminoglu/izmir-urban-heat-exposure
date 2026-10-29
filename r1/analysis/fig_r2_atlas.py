"""Figure 8: MGWR coefficient atlas of the reported fit (500 m inverse-distance atlas, r2_29), same six panels and
colour conventions as the earlier version: (a)-(c) local NDVI, NDBI and albedo coefficients on one symmetric
diverging scale, (d) dominant local driver, (e) local R2, (f) local condition number clipped at its 98th percentile."""
import sys
from pathlib import Path
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

INK = "#344150"

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

OUT = Path(r"REPO_ROOT\outputs\figures\paper1\final\fig07_mgwr_coefficient_atlas.png")
import os
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


fig, axes = plt.subplots(2, 3, figsize=(18, 11.5))
vmax = float(np.nanmax(np.abs(at[["mgwr_coef_ndvi", "mgwr_coef_ndbi", "mgwr_coef_albedo"]].to_numpy())))
first_im = None
for ax, col, ttl in zip(axes[0], ["mgwr_coef_ndvi", "mgwr_coef_ndbi", "mgwr_coef_albedo"],
                        ["(a) Local NDVI coefficient", "(b) Local NDBI coefficient", "(c) Local albedo coefficient"]):
    im = ax.imshow(grid(at[col]), origin="lower", extent=ext, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.set_label("coefficient")
    first_im = first_im or im
    ax.set_title(ttl, loc="left", fontsize=13, fontweight="bold")
dd = at.dominant_driver.str.lower().map({"ndvi": 0, "ndbi": 1, "albedo": 2}).to_numpy(float)
cols = ["#4a9c9e", "#c1272d", "#c9a24a"]
axes[1, 0].imshow(grid(dd), origin="lower", extent=ext, cmap=ListedColormap(cols), vmin=-0.5, vmax=2.5, interpolation="nearest")
axes[1, 0].legend(handles=[Patch(color=c, label=l) for c, l in zip(cols, ["NDVI-dominant", "NDBI-dominant", "Albedo-dominant"])],
                  loc="lower left", fontsize=9, frameon=True)
axes[1, 0].set_title("(d) Dominant local driver", loc="left", fontsize=13, fontweight="bold")
_cb = fig.colorbar(first_im, ax=axes[1, 0], fraction=0.046, pad=0.02); _cb.ax.set_visible(False)  # keeps panel widths aligned
im = axes[1, 1].imshow(grid(at.local_r2), origin="lower", extent=ext, cmap="viridis", interpolation="nearest",
                       vmin=np.nanpercentile(at.local_r2, 1), vmax=np.nanpercentile(at.local_r2, 99))
cb = fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.02); cb.set_label("local R²")
axes[1, 1].set_title("(e) Local R²", loc="left", fontsize=13, fontweight="bold")
c98 = float(np.nanpercentile(at.local_cond, 98))
im = axes[1, 2].imshow(grid(np.minimum(at.local_cond, c98)), origin="lower", extent=ext, cmap="magma_r", interpolation="nearest",
                       vmin=float(np.nanmin(at.local_cond)), vmax=c98)
cb = fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.02); cb.set_label("condition number")
axes[1, 2].set_title("(f) Local condition number", loc="left", fontsize=13, fontweight="bold")
for ax in axes.ravel():
    bnd.boundary.plot(ax=ax, color="black", linewidth=0.8)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.title.set_color(INK)
fig.tight_layout()
fig.savefig(OUT, dpi=200, facecolor="white")
print("wrote", OUT, "vmax", round(vmax, 3), "cond98", round(c98, 2))
