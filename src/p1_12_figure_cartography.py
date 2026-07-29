"""Elegant cartographic outputs for Study 1.

Creates a coherent grey-tinted visual package with controlled accent colours
for BBTMK slides and manuscript drafting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _config import FIGURES_DIR, DATA_INTERIM, DATA_MODELS, DATA_PROCESSED, ensure_dirs  # noqa: E402
from _config import get_logger  # noqa: E402


BOUNDARY = DATA_INTERIM / "izmir_study_boundary.gpkg"
LST_2025 = DATA_PROCESSED / "aligned_covariates" / "izmir_LST_Veri_2025_aligned_100m.tif"
NDVI_2025 = DATA_PROCESSED / "aligned_covariates" / "izmir_NDVI_aligned_100m.tif"
NDBI_2025 = DATA_PROCESSED / "aligned_covariates" / "izmir_NDBI_aligned_100m.tif"
TREE_CANOPY = DATA_PROCESSED / "aligned_covariates" / "izmir_KanopiYuksekligi_Veri_aligned_100m.tif"
DSM = DATA_INTERIM / "dsm_sym_mean_100m_aligned.tif"
LCZ = DATA_INTERIM / "lcz_filter_izmir_2018_100m.tif"
GWR = DATA_MODELS / "paper1_gwr_preliminary_sample_5000.gpkg"

STYLE_MD = FIGURES_DIR / "paper1_visual_style_guide.md"

FIG_LST_MAP = FIGURES_DIR / "paper1_map_lst_2025_elegant.png"
FIG_NDVI_NDBI = FIGURES_DIR / "paper1_map_ndvi_ndbi_dual.png"
FIG_LCZ = FIGURES_DIR / "paper1_map_lcz_greytone_accent.png"
FIG_GWR = FIGURES_DIR / "paper1_map_gwr_local_r2_points.png"
FIG_GWR_COEFS = FIGURES_DIR / "paper1_map_gwr_ndvi_ndbi_coefficients.png"
FIG_3D_LST = FIGURES_DIR / "paper1_3d_lst_dsm_surface.png"


COLORS = {
    "ink": "#1F2933",
    "muted": "#667085",
    "paper": "#F7F6F2",
    "base_gray": "#D9D6CE",
    "boundary": "#2B2B2B",
    "heat_low": "#566E8B",
    "heat_mid": "#ECE6D7",
    "heat_high": "#A53E2B",
    "veg": "#2F6B4F",
    "built": "#8C4A3C",
    "gold": "#B88A2E",
}


HEAT_CMAP = LinearSegmentedColormap.from_list(
    "paper_heat",
    [COLORS["heat_low"], "#8EA5B8", COLORS["heat_mid"], "#D79B72", COLORS["heat_high"]],
)
VEG_CMAP = LinearSegmentedColormap.from_list("paper_veg", ["#ECE6D7", "#9BAF8B", COLORS["veg"]])
BUILT_CMAP = LinearSegmentedColormap.from_list("paper_built", ["#ECE6D7", "#C7927A", COLORS["built"]])
COEF_CMAP = LinearSegmentedColormap.from_list("paper_div", ["#2F5D7C", "#F7F6F2", "#9B3D30"])


def read_raster(path: Path) -> tuple[np.ndarray, rasterio.Affine, tuple[float, float, float, float]]:
    """Read raster as masked float array plus transform and plotting extent."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        arr[arr <= -9990] = np.nan
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
        return arr, src.transform, extent


def setup_ax(ax, title: str) -> None:
    """Apply common map styling."""
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=COLORS["ink"], pad=10)
    ax.set_facecolor(COLORS["paper"])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_boundary(ax) -> None:
    """Draw study boundary."""
    boundary = gpd.read_file(BOUNDARY)
    boundary.boundary.plot(ax=ax, color=COLORS["boundary"], linewidth=0.65, alpha=0.85)


def save_raster_map(path: Path, output: Path, title: str, cmap, vmin=None, vmax=None, cbar_label: str = "") -> None:
    """Save a single raster map."""
    arr, _, extent = read_raster(path)
    fig, ax = plt.subplots(figsize=(7.4, 8.2), facecolor=COLORS["paper"])
    setup_ax(ax, title)
    im = ax.imshow(arr, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    draw_boundary(ax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
    cbar.set_label(cbar_label, color=COLORS["ink"])
    cbar.outline.set_visible(False)
    plt.tight_layout()
    fig.savefig(output, dpi=300, facecolor=COLORS["paper"], bbox_inches="tight")
    plt.close(fig)


def save_lst_map() -> None:
    """Save elegant LST map."""
    save_raster_map(LST_2025, FIG_LST_MAP, "2025 Summer Land Surface Temperature", HEAT_CMAP, 28, 56, "LST (deg C)")


def save_ndvi_ndbi_dual() -> None:
    """Save paired vegetation/built-up maps."""
    ndvi, _, extent = read_raster(NDVI_2025)
    ndbi, _, _ = read_raster(NDBI_2025)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.8), facecolor=COLORS["paper"])
    for ax, arr, cmap, title, label in [
        (axes[0], ndvi, VEG_CMAP, "Vegetation Signal", "NDVI"),
        (axes[1], ndbi, BUILT_CMAP, "Built-Up Signal", "NDBI"),
    ]:
        setup_ax(ax, title)
        im = ax.imshow(arr, extent=extent, cmap=cmap, vmin=np.nanpercentile(arr, 2), vmax=np.nanpercentile(arr, 98))
        draw_boundary(ax)
        cbar = fig.colorbar(im, ax=ax, shrink=0.62, pad=0.02)
        cbar.set_label(label)
        cbar.outline.set_visible(False)
    plt.tight_layout()
    fig.savefig(FIG_NDVI_NDBI, dpi=300, facecolor=COLORS["paper"], bbox_inches="tight")
    plt.close(fig)


def save_lcz_map() -> None:
    """Save greytone LCZ map with restrained accents."""
    arr, _, extent = read_raster(LCZ)
    classes = [2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    palette = [
        "#5B4A42", "#6B5F56", "#8D7B68", "#A79A86", "#C19C64", "#9D836D", "#6F6259",
        "#CBD0C0", "#BFC4B2", "#AEB89B", "#8A9C7A", "#B9C8CD", "#8AA0A8", "#6D7E86",
    ]
    cmap = ListedColormap(palette)
    bounds = [c - 0.5 for c in classes] + [classes[-1] + 0.5]
    norm = BoundaryNorm(bounds, cmap.N)
    fig, ax = plt.subplots(figsize=(7.4, 8.2), facecolor=COLORS["paper"])
    setup_ax(ax, "Local Climate Zones, Greytone Accent")
    masked = np.where(np.isin(arr, classes), arr, np.nan)
    im = ax.imshow(masked, extent=extent, cmap=cmap, norm=norm, interpolation="nearest")
    draw_boundary(ax)
    cbar = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02, ticks=classes)
    cbar.set_label("LCZ class")
    cbar.outline.set_visible(False)
    plt.tight_layout()
    fig.savefig(FIG_LCZ, dpi=300, facecolor=COLORS["paper"], bbox_inches="tight")
    plt.close(fig)


def save_gwr_maps() -> None:
    """Save preliminary GWR point maps."""
    gwr = gpd.read_file(GWR)
    boundary = gpd.read_file(BOUNDARY)
    fig, ax = plt.subplots(figsize=(7.4, 8.2), facecolor=COLORS["paper"])
    setup_ax(ax, "Preliminary GWR Local R2")
    boundary.plot(ax=ax, color=COLORS["base_gray"], edgecolor=COLORS["boundary"], linewidth=0.5)
    gwr.plot(ax=ax, column="gwr_local_r2", cmap=HEAT_CMAP, markersize=5, alpha=0.82, legend=True, legend_kwds={"shrink": 0.65, "label": "Local R2"})
    plt.tight_layout()
    fig.savefig(FIG_GWR, dpi=300, facecolor=COLORS["paper"], bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.8), facecolor=COLORS["paper"])
    for ax, column, title in [
        (axes[0], "gwr_coef_ndvi_2025", "Local NDVI Cooling Coefficient"),
        (axes[1], "gwr_coef_ndbi_2025", "Local NDBI Heating Coefficient"),
    ]:
        setup_ax(ax, title)
        boundary.plot(ax=ax, color=COLORS["base_gray"], edgecolor=COLORS["boundary"], linewidth=0.5)
        vmax = max(abs(gwr[column].quantile(0.05)), abs(gwr[column].quantile(0.95)))
        gwr.plot(
            ax=ax,
            column=column,
            cmap=COEF_CMAP,
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
            markersize=5,
            alpha=0.85,
            legend=True,
            legend_kwds={"shrink": 0.6},
        )
    plt.tight_layout()
    fig.savefig(FIG_GWR_COEFS, dpi=300, facecolor=COLORS["paper"], bbox_inches="tight")
    plt.close(fig)


def save_3d_lst_surface() -> None:
    """Save a 3D-style LST surface draped over DSM/SYM."""
    lst, _, _ = read_raster(LST_2025)
    dsm, _, _ = read_raster(DSM)
    step = 12
    z = dsm[::step, ::step].copy()
    c = lst[::step, ::step].copy()
    valid = np.isfinite(z) & np.isfinite(c)
    rows, cols = z.shape
    xx, yy = np.meshgrid(np.arange(cols), np.arange(rows))
    z = np.where(valid, z, np.nan)
    c = np.where(valid, c, np.nan)
    z_scaled = np.where(np.isfinite(z), z / 28.0, np.nan)

    fig = plt.figure(figsize=(9.5, 7.2), facecolor=COLORS["paper"])
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(COLORS["paper"])
    norm = plt.Normalize(28, 56)
    facecolors = HEAT_CMAP(norm(c))
    ax.plot_surface(xx, yy, z_scaled, facecolors=facecolors, linewidth=0, antialiased=False, shade=False, rstride=1, cstride=1)
    ax.view_init(elev=56, azim=-132)
    ax.set_box_aspect((1.0, 1.35, 0.34))
    ax.set_xlim(np.nanmin(xx), np.nanmax(xx))
    ax.set_ylim(np.nanmax(yy), np.nanmin(yy))
    ax.set_zlim(np.nanpercentile(z_scaled, 1), np.nanpercentile(z_scaled, 99))
    ax.set_title("2025 Summer LST Draped Over Processed DSM/SYM", loc="left", pad=10, color=COLORS["ink"], fontweight="bold", fontsize=13)
    ax.set_axis_off()
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=HEAT_CMAP)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.48, pad=0.0)
    cbar.set_label("LST (deg C)")
    cbar.outline.set_visible(False)
    fig.subplots_adjust(left=0.0, right=0.90, top=0.92, bottom=0.02)
    fig.savefig(FIG_3D_LST, dpi=300, facecolor=COLORS["paper"], bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def write_style_guide() -> None:
    """Write a compact visual style guide."""
    STYLE_MD.write_text(
        "\n".join(
            [
                "# Paper 1 Visual Style Guide",
                "",
                "Base tone: warm paper grey `#F7F6F2` with muted administrative boundaries.",
                "",
                "Core colours:",
                "",
                "- Ink: `#1F2933`",
                "- Muted text: `#667085`",
                "- Boundary: `#2B2B2B`",
                "- Heat high: `#A53E2B`",
                "- Heat low: `#566E8B`",
                "- Vegetation: `#2F6B4F`",
                "- Built-up: `#8C4A3C`",
                "- Gold accent: `#B88A2E`",
                "",
                "Usage:",
                "",
                "- LST and UHI: blue-grey to warm red.",
                "- Vegetation/canopy: muted green.",
                "- Built-up/NDBI/albedo emphasis: clay/brown-red.",
                "- Coefficients: diverging blue-grey to warm red, centred at zero.",
                "- LCZ: mostly greytone/categorical with restrained warm and green accents.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    logger = get_logger("p1_12_cartography_outputs")
    logger.info("Generating elegant cartographic outputs for Study 1.")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_lst_map()
    save_ndvi_ndbi_dual()
    save_lcz_map()
    save_gwr_maps()
    save_3d_lst_surface()
    write_style_guide()
    for path in [FIG_LST_MAP, FIG_NDVI_NDBI, FIG_LCZ, FIG_GWR, FIG_GWR_COEFS, FIG_3D_LST, STYLE_MD]:
        logger.info("Wrote cartographic output: %s", path)


if __name__ == "__main__":
    main()
