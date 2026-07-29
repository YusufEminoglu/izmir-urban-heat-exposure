"""Elite multipanel figures for Study 1.

This script upgrades the basic map package into publication-style raw panels:
hex aggregation, bivariate vulnerability mapping, district Dorling cartogram,
data-cube visualisation, and local coefficient atlas panels.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _config import EXTERNAL_MUNICIPAL, FIGURES_DIR, DATA_MODELS, DATA_PROCESSED, ensure_dirs  # noqa: E402
from _config import get_logger  # noqa: E402


MODEL_GRID = DATA_PROCESSED / "paper1_model_grid_100m_current.csv"
GWR = DATA_MODELS / "paper1_gwr_preliminary_sample_5000.gpkg"
LST_CUBE = DATA_PROCESSED / "datacubes" / "paper1_landsat_lst_jja_cube_2014_2024.nc"
ILCE = EXTERNAL_MUNICIPAL / "ilce.shp"

ELITE_DIR = FIGURES_DIR / "elite_panels"
HEX_SIDE_M = 176.0
HEX_GPKG = DATA_PROCESSED / "paper1_hex_176m_summary.gpkg"
DISTRICT_SUMMARY = DATA_PROCESSED / "paper1_district_summary.csv"

FIG_BIVARIATE_HEX = ELITE_DIR / "fig_elite_01_bivariate_hex_heat_vegetation.png"
FIG_CUBE_PANEL = ELITE_DIR / "fig_elite_02_datacube_lst_panel.png"
FIG_CARTOGRAM = ELITE_DIR / "fig_elite_03_district_dorling_cartogram.png"
FIG_GWR_ATLAS = ELITE_DIR / "fig_elite_04_gwr_coefficient_atlas.png"
FIG_MULTIPANEL = ELITE_DIR / "fig_elite_05_synthesis_multipanel.png"


BG = "#FBFBF8"
INK = "#344150"
SUBTLE = "#8A939B"
GRID = "#E8EBEA"
TEAL = "#86B7AA"
NAVY = "#667F95"
RED = "#B97872"
PEACH = "#E5C8BF"
GRAY = "#B7BDBD"
LIGHT = "#F1F3F1"
GOLD = "#C7A76E"

CMAP_SEQ = LinearSegmentedColormap.from_list("elite_seq", [LIGHT, TEAL, NAVY])
CMAP_WARM = LinearSegmentedColormap.from_list("elite_warm", [LIGHT, PEACH, RED])
CMAP_DIV = LinearSegmentedColormap.from_list("elite_div", [NAVY, TEAL, LIGHT, PEACH, RED])
BIVARIATE = np.array(
    [
        ["#8DA99A", "#93A8B9", "#8588A6"],
        ["#AFC0CA", "#D2D2CE", "#B49AB1"],
        ["#E2C8BB", "#D6A69A", "#B97872"],
    ]
)


def tag(ax, label: str) -> None:
    if hasattr(ax, "text2D"):
        ax.text2D(0.012, 0.988, label, transform=ax.transAxes, ha="left", va="top", fontsize=10.5, fontweight="bold", color=INK)
    else:
        ax.text(0.012, 0.988, label, transform=ax.transAxes, ha="left", va="top", fontsize=10.5, fontweight="bold", color=INK)


def style_map_axis(ax, title: str | None = None) -> None:
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=7)


def style_chart_axis(ax, title: str | None = None) -> None:
    ax.set_facecolor(BG)
    ax.grid(color=GRID, linewidth=0.75, alpha=0.85)
    ax.tick_params(colors=INK, labelsize=8, length=0)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    if title:
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=7)


def load_model(sample: int | None = None) -> pd.DataFrame:
    usecols = [
        "x",
        "y",
        "lst_2025",
        "ndvi_2025",
        "ndbi_2025",
        "albedo_2025",
        "tree_canopy_height_2025",
        "dsm_sym_mean_100m",
        "lcz",
        "pop_total",
        "elderly_65_plus_share",
    ]
    df = pd.read_csv(MODEL_GRID, usecols=usecols).dropna(subset=["x", "y", "lst_2025", "ndvi_2025", "ndbi_2025"])
    if sample and len(df) > sample:
        return df.sample(sample, random_state=42)
    return df


def hexagon(cx: float, cy: float, radius: float) -> Polygon:
    angles = np.deg2rad(np.arange(0, 360, 60) + 30)
    return Polygon([(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles])


def build_hex_summary(radius: float = HEX_SIDE_M) -> gpd.GeoDataFrame:
    """Build/reuse 176 m side-length hex summary from model grid cells."""
    if HEX_GPKG.exists():
        return gpd.read_file(HEX_GPKG)
    df = load_model()
    minx, maxx = df["x"].min(), df["x"].max()
    miny, maxy = df["y"].min(), df["y"].max()
    dx = math.sqrt(3) * radius
    dy = 1.5 * radius
    hexes = []
    row = 0
    y = miny - dy
    while y <= maxy + dy:
        x_offset = 0 if row % 2 == 0 else dx / 2
        x = minx - dx
        while x <= maxx + dx:
            hexes.append({"hex_id": len(hexes), "geometry": hexagon(x + x_offset, y, radius)})
            x += dx
        y += dy
        row += 1
    hex_gdf = gpd.GeoDataFrame(hexes, crs="EPSG:32635")
    points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"], crs="EPSG:32635"))
    joined = gpd.sjoin(points, hex_gdf, how="inner", predicate="within")
    agg = (
        joined.groupby("hex_id")
        .agg(
            n=("lst_2025", "size"),
            lst_mean=("lst_2025", "mean"),
            ndvi_mean=("ndvi_2025", "mean"),
            ndbi_mean=("ndbi_2025", "mean"),
            albedo_mean=("albedo_2025", "mean"),
            tree_canopy_mean=("tree_canopy_height_2025", "mean"),
            elderly_mean=("elderly_65_plus_share", "mean"),
        )
        .reset_index()
    )
    out = hex_gdf.merge(agg, on="hex_id")
    out = out[out["n"] >= 2].copy()
    out.to_file(HEX_GPKG, layer="paper1_hex_1500m_summary", driver="GPKG")
    return out


def add_bivariate_classes(hexes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = hexes.copy()
    out["heat_q"] = pd.qcut(out["lst_mean"], 3, labels=[0, 1, 2]).astype(int)
    out["veg_deficit"] = -out["ndvi_mean"]
    out["veg_q"] = pd.qcut(out["veg_deficit"], 3, labels=[0, 1, 2]).astype(int)
    out["bivar"] = out["heat_q"] * 3 + out["veg_q"]
    out["bivar_color"] = [BIVARIATE[h, v] for h, v in zip(out["heat_q"], out["veg_q"])]
    return out


def draw_bivariate_legend(ax) -> None:
    inset = ax.inset_axes([0.055, 0.055, 0.155, 0.155])
    for i in range(3):
        for j in range(3):
            inset.add_patch(Rectangle((j, i), 1, 1, facecolor=BIVARIATE[i, j], edgecolor="white", linewidth=0.7))
    inset.set_xlim(0, 3)
    inset.set_ylim(0, 3)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(False)
    inset.text(1.5, -0.34, "vegetation deficit", ha="center", va="top", fontsize=6.8, color=INK)
    inset.text(-0.34, 1.5, "heat", ha="right", va="center", rotation=90, fontsize=6.8, color=INK)


def save_bivariate_hex() -> None:
    hexes = add_bivariate_classes(build_hex_summary())
    fig, ax = plt.subplots(figsize=(8, 9), facecolor=BG)
    style_map_axis(ax, "Bivariate 176 m Hex Atlas: Heat Load x Vegetation Deficit")
    hexes.plot(ax=ax, color=hexes["bivar_color"], edgecolor="#F9F9F6", linewidth=0.03)
    draw_bivariate_legend(ax)
    tag(ax, "(a)")
    fig.savefig(FIG_BIVARIATE_HEX, dpi=300, facecolor=BG, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def save_datacube_panel() -> None:
    ds = xr.open_dataset(LST_CUBE)
    annual = ds["lst_jja"].mean(dim=("y", "x"), skipna=True).to_pandas()
    arr = ds["lst_jja"].isel(time=slice(None, None, 2)).values
    ds.close()
    fig = plt.figure(figsize=(12.2, 7.2), facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], width_ratios=[1.3, 1.0], hspace=0.28, wspace=0.18)
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])

    sample = arr[:, ::28, ::28]
    t, yy, xx = sample.shape
    xg, yg = np.meshgrid(np.arange(xx), np.arange(yy))
    norm = Normalize(np.nanpercentile(sample, 5), np.nanpercentile(sample, 95))
    for i in range(t):
        z = np.full_like(xg, i * 7, dtype=float)
        colors = CMAP_WARM(norm(sample[i]))
        ax3.plot_surface(xg, yg, z, facecolors=colors, linewidth=0, shade=False, alpha=0.86)
    ax3.view_init(elev=30, azim=-50)
    ax3.set_title("Stacked Annual LST Cube Slices", loc="left", color=INK, fontweight="bold", fontsize=10.5, pad=7)
    ax3.set_axis_off()
    tag(ax3, "(a)")

    style_chart_axis(ax1, "Annual JJA LST Mean")
    ax1.plot(annual.index.year, annual.values, color=NAVY, linewidth=2.4)
    ax1.scatter(annual.index.year, annual.values, color=RED, s=28, zorder=3)
    ax1.set_xticks(annual.index.year[::2])
    ax1.set_ylabel("deg C", color=INK)
    tag(ax1, "(b)")

    style_map_axis(ax2, "Cube Structure")
    ax2.text(0.08, 0.72, "time", color=RED, fontsize=10, weight="bold")
    ax2.text(0.52, 0.45, "y", color=TEAL, fontsize=10, weight="bold")
    ax2.text(0.35, 0.18, "x", color=NAVY, fontsize=10, weight="bold")
    for k in range(6):
        ax2.add_patch(Rectangle((0.18 + k * 0.045, 0.2 + k * 0.055), 0.42, 0.34, facecolor=LIGHT, edgecolor=NAVY, alpha=0.78, linewidth=0.8))
    ax2.text(0.5, 0.86, "2014-2024\n11 summer slices", ha="center", va="top", color=INK, fontsize=10)
    tag(ax2, "(c)")
    fig.savefig(FIG_CUBE_PANEL, dpi=300, facecolor=BG, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def district_summary() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if DISTRICT_SUMMARY.exists():
        summary = pd.read_csv(DISTRICT_SUMMARY)
    else:
        df = load_model()
        points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"], crs="EPSG:32635"))
        ilce = gpd.read_file(ILCE).to_crs("EPSG:32635").rename(columns={"ADINUMARAS": "district"})
        joined = gpd.sjoin(points, ilce[["district", "geometry"]], predicate="within", how="inner")
        summary = joined.groupby("district").agg(
            n=("lst_2025", "size"),
            lst_mean=("lst_2025", "mean"),
            ndvi_mean=("ndvi_2025", "mean"),
            pop_total=("pop_total", "sum"),
            elderly_mean=("elderly_65_plus_share", "mean"),
        ).reset_index()
        summary.to_csv(DISTRICT_SUMMARY, index=False, encoding="utf-8-sig")
    ilce = gpd.read_file(ILCE).to_crs("EPSG:32635").rename(columns={"ADINUMARAS": "district"})
    return ilce.merge(summary, on="district", how="left"), gpd.GeoDataFrame(summary)


def save_cartogram() -> None:
    districts, _ = district_summary()
    cent = districts.copy()
    cent["geometry"] = cent.geometry.representative_point()
    cent["cx"] = cent.geometry.x
    cent["cy"] = cent.geometry.y
    cent = cent.dropna(subset=["lst_mean", "n"])
    sizes = 220 + 3300 * (cent["n"] / cent["n"].max())
    fig, ax = plt.subplots(figsize=(8, 8.8), facecolor=BG)
    style_map_axis(ax, "Hexagonal District Cartogram: LST Intensity and Cell Mass")
    districts.boundary.plot(ax=ax, color=GRID, linewidth=0.4)
    sc = ax.scatter(
        cent["cx"],
        cent["cy"],
        s=sizes,
        c=cent["lst_mean"],
        cmap=CMAP_WARM,
        marker="h",
        edgecolors="#FFFFFF",
        linewidths=1.05,
        alpha=0.9,
    )
    label_threshold = cent["n"].quantile(0.45)
    for _, row in cent[cent["n"] >= label_threshold].iterrows():
        ax.text(row["cx"], row["cy"], str(row["district"]).title(), ha="center", va="center", fontsize=6.0, color=INK)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.68, pad=0.01)
    cbar.set_label("Mean LST (deg C)", fontsize=8, color=INK)
    cbar.ax.tick_params(labelsize=7, colors=INK, length=0)
    cbar.outline.set_visible(False)
    tag(ax, "(a)")
    fig.savefig(FIG_CARTOGRAM, dpi=300, facecolor=BG, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_gwr_atlas() -> None:
    gwr = gpd.read_file(GWR)
    columns = [
        ("gwr_coef_ndvi_2025", "NDVI cooling"),
        ("gwr_coef_ndbi_2025", "NDBI heating"),
        ("gwr_coef_albedo_2025", "Albedo effect"),
        ("gwr_coef_dsm_sym_mean_100m", "DSM/SYM effect"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), facecolor=BG)
    for ax, (col, title), label in zip(axes.ravel(), columns, ["(a)", "(b)", "(c)", "(d)"]):
        style_map_axis(ax, title)
        vmax = max(abs(gwr[col].quantile(0.04)), abs(gwr[col].quantile(0.96)))
        gwr.plot(ax=ax, column=col, cmap=CMAP_DIV, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax), markersize=6, alpha=0.82)
        tag(ax, label)
    fig.suptitle("Preliminary GWR Local Coefficient Atlas", x=0.02, y=0.992, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.subplots_adjust(top=0.93, hspace=0.13, wspace=0.03)
    fig.savefig(FIG_GWR_ATLAS, dpi=300, facecolor=BG, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def save_synthesis_multipanel() -> None:
    hexes = add_bivariate_classes(build_hex_summary())
    districts, _ = district_summary()
    fig = plt.figure(figsize=(12.5, 9.4), facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.08, 0.82], hspace=0.24, wspace=0.16)

    ax = fig.add_subplot(gs[0, 0])
    style_map_axis(ax, "Bivariate Heat-Vegetation Hexes")
    hexes.plot(ax=ax, color=hexes["bivar_color"], edgecolor="#F9F9F6", linewidth=0.025)
    draw_bivariate_legend(ax)
    tag(ax, "(a)")

    ax = fig.add_subplot(gs[0, 1])
    style_map_axis(ax, "District Mean LST")
    districts.plot(
        ax=ax,
        column="lst_mean",
        cmap=CMAP_WARM,
        edgecolor="white",
        linewidth=0.35,
        legend=True,
        legend_kwds={"shrink": 0.54, "pad": 0.01},
    )
    tag(ax, "(b)")

    ax = fig.add_subplot(gs[1, 0])
    style_chart_axis(ax, "Hex NDBI vs LST")
    hb = ax.hexbin(hexes["ndbi_mean"], hexes["lst_mean"], gridsize=32, cmap=CMAP_SEQ, mincnt=1, linewidths=0)
    ax.set_xlabel("NDBI", color=INK, fontsize=8.5)
    ax.set_ylabel("LST (deg C)", color=INK, fontsize=8.5)
    cb = fig.colorbar(hb, ax=ax, shrink=0.72, pad=0.012)
    cb.set_label("hex count", fontsize=8, color=INK)
    cb.ax.tick_params(labelsize=7, colors=INK, length=0)
    cb.outline.set_visible(False)
    tag(ax, "(c)")

    ax = fig.add_subplot(gs[1, 1])
    style_chart_axis(ax, "Hex NDVI vs LST")
    hb = ax.hexbin(hexes["ndvi_mean"], hexes["lst_mean"], gridsize=32, cmap=CMAP_WARM, mincnt=1, linewidths=0)
    ax.set_xlabel("NDVI", color=INK, fontsize=8.5)
    ax.set_ylabel("LST (deg C)", color=INK, fontsize=8.5)
    cb = fig.colorbar(hb, ax=ax, shrink=0.72, pad=0.012)
    cb.set_label("hex count", fontsize=8, color=INK)
    cb.ax.tick_params(labelsize=7, colors=INK, length=0)
    cb.outline.set_visible(False)
    tag(ax, "(d)")
    fig.suptitle("Heat Justice Analytical Atlas: LCZ-Aligned 2025 Snapshot", x=0.02, y=0.985, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.savefig(FIG_MULTIPANEL, dpi=300, facecolor=BG, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    logger = get_logger("p1_13_elite_figure_panels")
    ELITE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Generating elite raw-panel figure package.")
    save_bivariate_hex()
    save_datacube_panel()
    save_cartogram()
    save_gwr_atlas()
    save_synthesis_multipanel()
    for path in [FIG_BIVARIATE_HEX, FIG_CUBE_PANEL, FIG_CARTOGRAM, FIG_GWR_ATLAS, FIG_MULTIPANEL, HEX_GPKG, DISTRICT_SUMMARY]:
        logger.info("Wrote elite output: %s", path)


if __name__ == "__main__":
    main()
