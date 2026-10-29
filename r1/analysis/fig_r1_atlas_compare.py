"""Supplementary Figure: reported IDW atlas (500 m) vs kernel-consistent coefficients at 100 m."""
import sys
from pathlib import Path
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

INK, MUTED, BOUNDARY = "#344150", "#7A8590", "#2F3D4A"
cmap = LinearSegmentedColormap.from_list("div_soft", ["#667F95", "#BFD7D0", "#F1F3F1", "#E5C8BF", "#B97872", "#8E4F49"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

g = load_grid()
B = np.load(CACHE / "betas_fullgrid_r0.npy")  # intercept, ndvi, ndbi, albedo (standardised)
atlas = gpd.read_file(IDW_GRID)
bnd = gpd.read_file(DATA / "izmir_province_boundary_filled.gpkg").to_crs(CRS)
nr, nc = int(g.row.max()) + 1, int(g.col.max()) + 1
X0, Y0 = 432600.0, 4359700.0
ext = [X0, X0 + nc * 100, Y0 - nr * 100, Y0]


def r100(v):
    a = np.full((nr, nc), np.nan, np.float32); a[g.row, g.col] = v; return a


def r500(v):
    n5r, n5c = int(np.ceil(nr / 5)), int(np.ceil(nc / 5))
    a = np.full((n5r, n5c), np.nan, np.float32)
    cc = np.floor((atlas.geometry.x - X0) / 500).astype(int); rr = np.floor((Y0 - atlas.geometry.y) / 500).astype(int)
    ok = (rr >= 0) & (rr < n5r) & (cc >= 0) & (cc < n5c)
    a[rr[ok], cc[ok]] = v[ok]
    return a, [X0, X0 + n5c * 500, Y0 - n5r * 500, Y0]


names = [("ndvi", 1, "mgwr_coef_ndvi", "NDVI"), ("ndbi", 2, "mgwr_coef_ndbi", "NDBI"), ("albedo", 3, "mgwr_coef_albedo", "Albedo")]
fig, axes = plt.subplots(2, 3, figsize=(15, 9.6))
for j, (k, col, acol, lab) in enumerate(names):
    v100 = B[:, col]
    lim = np.nanpercentile(np.abs(np.r_[v100, atlas[acol].to_numpy()]), 99)
    norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)
    a5, e5 = r500(atlas[acol].to_numpy())
    for i, (arr, e, ttl) in enumerate([(a5, e5, "reported atlas (IDW, 500 m)"), (r100(v100), ext, "kernel-consistent (100 m)")]):
        ax = axes[i, j]
        im = ax.imshow(arr, extent=e, cmap=cmap, norm=norm, interpolation="nearest")
        bnd.boundary.plot(ax=ax, color=BOUNDARY, linewidth=0.6)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"({'abc'[j] if i == 0 else 'def'[j]}) {lab}, {ttl}", fontsize=10.5, color=INK, loc="left", fontweight="bold")
        if i == 1:
            cb = fig.colorbar(im, ax=axes[:, j], orientation="horizontal", fraction=0.035, pad=0.03, shrink=0.8)
            cb.set_label(f"local {lab} coefficient (standardised)", color=MUTED, fontsize=9)
            cb.outline.set_visible(False); cb.ax.tick_params(colors=INK, length=0, labelsize=8.5)
out = FIGS / "figS1_atlas_idw_vs_kernel.png"
fig.savefig(out, dpi=220, bbox_inches="tight")
fig.savefig(Path(r"REPO_ROOT\outputs\figures\paper1\final") / out.name, dpi=220, bbox_inches="tight")
print(out)
