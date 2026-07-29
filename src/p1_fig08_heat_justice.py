"""Standalone generator for Paper 1 Figure 08.

This file is intentionally self-contained (no imports from other project modules)
so Fig08 can be regenerated independently.

Run (from repo root):
  .\\.venv\\Scripts\\python.exe -m src.paper1_climate_justice.p1_fig08_standalone
"""

from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib import transforms
from matplotlib.colors import LinearSegmentedColormap, LightSource, Normalize
from matplotlib.patches import Rectangle
from pyproj import Transformer
from rasterio.features import geometry_mask
from shapely.geometry import LineString, box

# -----------------------------------------------------------------------------
# Paths / outputs

ROOT = Path(__file__).resolve().parents[2]

DATA_INTERIM = ROOT / "data" / "02_interim" / "paper1"
DATA_PROCESSED = ROOT / "data" / "03_processed" / "paper1"
TABLES_DIR = ROOT / "outputs" / "tables" / "paper1" / "manuscript"
FIG_OUT = ROOT / "outputs" / "figures" / "paper1" / "final" / "fig08_heat_justice_sociodemographic_synthesis.png"
MYDATA = ROOT / "docs" / "manuscripts" / "paper1" / "src" / "mydata.dat"

BOUNDARY = DATA_INTERIM / "izmir_study_boundary.gpkg"
MODEL_GRID = DATA_PROCESSED / "paper1_model_grid_100m_current.csv"
SRTM_ELEV = DATA_PROCESSED / "aligned_covariates" / "izmir_srtm_elevation_aligned_100m.tif"
SOCIAL_HEX_CACHE = DATA_PROCESSED / "paper1_hex_150m_social_heat_summary.gpkg"

# -----------------------------------------------------------------------------
# Style

BG = "#FFFFFF"
INK = "#22323D"
MUTED = "#6E7B81"
GRID = "#DDE3E1"
BOUNDARY_LINE = "#2F3D4A"
NAVY = "#2F5568"
TEAL = "#2AA6A1"
ROSE = "#B97872"


def save_img(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor=BG, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def clean_ax(ax) -> None:
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def chart_ax(ax) -> None:
    ax.set_facecolor(BG)
    ax.grid(color=GRID, linewidth=0.7, alpha=0.9)
    ax.tick_params(colors=INK, labelsize=8, length=0)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)


def tag(ax, label: str) -> None:
    ax.text(
        0.012,
        0.988,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        zorder=50,
        bbox=dict(boxstyle="square,pad=0.12", facecolor=BG, edgecolor="none", alpha=0.85),
    )


def load_mydata() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in MYDATA.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "," not in line:
            continue
        key, value = line.split(",", 1)
        values[key.strip()] = value.strip()
    return values


def mydata_float(values: dict[str, str], key: str) -> float:
    return float(values[key].replace("<", ""))


def mahalle_moran_diagnostics() -> pd.DataFrame:
    values = load_mydata()
    return pd.DataFrame(
        [
            {
                "exposure": "Mean LST",
                "weights": "Queen",
                "moran_i": mydata_float(values, "mahalle_bivar_i_queen"),
                "p_value": mydata_float(values, "mahalle_bivar_p_queen"),
            },
            {
                "exposure": "Mean LST",
                "weights": "5 km band",
                "moran_i": mydata_float(values, "mahalle_bivar_i_dist5km"),
                "p_value": mydata_float(values, "mahalle_bivar_p_dist5km"),
            },
            {
                "exposure": "P90 LST",
                "weights": "Queen",
                "moran_i": mydata_float(values, "mahalle_bivar_p90_i_queen"),
                "p_value": mydata_float(values, "mahalle_bivar_p90_p_queen"),
            },
            {
                "exposure": "P90 LST",
                "weights": "5 km band",
                "moran_i": mydata_float(values, "mahalle_bivar_p90_i_dist5km"),
                "p_value": mydata_float(values, "mahalle_bivar_p90_p_dist5km"),
            },
        ]
    )


def draw_mahalle_moran_panel(ax) -> None:
    diag = mahalle_moran_diagnostics()
    exposures = ["Mean LST", "P90 LST"]
    weight_labels = ["Queen", "5 km band"]
    colors = {"Queen": NAVY, "5 km band": ROSE}
    offsets = {"Queen": -0.18, "5 km band": 0.18}
    x = np.arange(len(exposures), dtype=float)

    ax.axhline(0, color=BOUNDARY_LINE, linewidth=0.78, alpha=0.70)
    for weights in weight_labels:
        sub = diag[diag["weights"] == weights].set_index("exposure").loc[exposures]
        bars = ax.bar(
            x + offsets[weights],
            sub["moran_i"],
            width=0.32,
            color=colors[weights],
            alpha=0.76 if weights == "Queen" else 0.62,
            label=weights,
        )
        for bar, (_, row) in zip(bars, sub.iterrows(), strict=False):
            value = float(row["moran_i"])
            y = value - 0.014 if value < 0 else value + 0.014
            va = "top" if value < 0 else "bottom"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"I={value:.3f}\np={float(row['p_value']):.3f}",
                ha="center",
                va=va,
                fontsize=6.4,
                color=INK,
                linespacing=1.05,
            )
    ax.set_xticks(x, exposures, fontsize=8.2, color=INK)
    ax.set_ylim(-0.265, 0.055)
    ax.set_ylabel("mahalle bivariate Moran's I", fontsize=8.5, color=INK)
    ax.legend(frameon=False, fontsize=7.4, loc="lower right")
    ax.text(
        0.145,
        0.955,
        "Heat-vulnerability spatial robustness\n(row-standardised weights)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color=MUTED,
        linespacing=1.15,
    )
    tag(ax, "(d)")


def boundary_gdf(crs: str = "EPSG:32635") -> gpd.GeoDataFrame:
    return gpd.read_file(BOUNDARY).to_crs(crs)


def mask_outside_boundary(ax, boundary: gpd.GeoDataFrame, zorder: int = 4) -> None:
    minx, miny, maxx, maxy = boundary.total_bounds
    pad_x = 0.06 * (maxx - minx)
    pad_y = 0.06 * (maxy - miny)
    frame = box(minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y)
    outside = gpd.GeoSeries([frame], crs=boundary.crs).difference(boundary.geometry.union_all())
    outside.plot(ax=ax, color=BG, edgecolor="none", zorder=zorder)


def draw_boundary_effect(ax, boundary: gpd.GeoDataFrame) -> None:
    boundary.boundary.plot(ax=ax, color=BOUNDARY_LINE, linewidth=0.62, alpha=0.92, zorder=7)


def set_bounds(ax, boundary: gpd.GeoDataFrame, pad_m: float = 2200.0, adjustable: str = "box") -> None:
    bounds = boundary.total_bounds
    ax.set_xlim(bounds[0] - pad_m, bounds[2] + pad_m)
    ax.set_ylim(bounds[1] - pad_m, bounds[3] + pad_m)
    ax.set_aspect("equal", adjustable=adjustable)


def read_clipped_raster(path: Path) -> tuple[np.ndarray, tuple[float, float, float, float], str, np.ndarray]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        arr[arr <= -9990] = np.nan
        boundary = boundary_gdf(src.crs.to_string())
        inside = geometry_mask(boundary.geometry, out_shape=(src.height, src.width), transform=src.transform, invert=True)
        arr[~inside] = np.nan
        extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)
        return arr, extent, src.crs.to_string(), inside


_HILLSHADE_CACHE: tuple[np.ndarray, tuple[float, float, float, float]] | None = None


def get_hillshade() -> tuple[np.ndarray, tuple[float, float, float, float]]:
    global _HILLSHADE_CACHE
    if _HILLSHADE_CACHE is None:
        dem, extent, _, inside = read_clipped_raster(SRTM_ELEV)
        ls = LightSource(azdeg=315, altdeg=45)
        shade = ls.hillshade(np.nan_to_num(dem, nan=np.nanmedian(dem)), vert_exag=0.65, dx=100, dy=100)
        shade[~inside] = np.nan
        _HILLSHADE_CACHE = (shade, extent)
    return _HILLSHADE_CACHE


_TRANSFORMERS: dict[str, Transformer] = {}


def wgs84_to_crs(crs: str) -> Transformer:
    if crs not in _TRANSFORMERS:
        _TRANSFORMERS[crs] = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    return _TRANSFORMERS[crs]


def wgs84_ticks(boundary: gpd.GeoDataFrame, interval_deg: float = 0.25) -> tuple[np.ndarray, np.ndarray, float, float]:
    b4326 = boundary.to_crs("EPSG:4326")
    minlon, minlat, maxlon, maxlat = b4326.total_bounds
    lon_ticks = np.arange(math.ceil(minlon / interval_deg) * interval_deg, maxlon + interval_deg, interval_deg)
    lat_ticks = np.arange(math.ceil(minlat / interval_deg) * interval_deg, maxlat + interval_deg, interval_deg)
    return lon_ticks, lat_ticks, float(minlon), float(minlat)


def plot_line_geometry(ax, geom, color: str, linewidth: float, alpha: float, zorder: int) -> None:
    if geom.is_empty:
        return
    if geom.geom_type == "LineString":
        x, y = geom.xy
        ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
    else:
        for part in geom.geoms:
            plot_line_geometry(ax, part, color, linewidth, alpha, zorder)


def add_clipped_wgs84_graticule(ax, boundary: gpd.GeoDataFrame, interval_deg: float = 0.25) -> None:
    lon_ticks, lat_ticks, minlon, minlat = wgs84_ticks(boundary, interval_deg)
    b4326 = boundary.to_crs("EPSG:4326")
    _, _, maxlon, maxlat = b4326.total_bounds
    transformer = wgs84_to_crs(boundary.crs.to_string())
    study_geom = boundary.geometry.union_all()
    lat_samples = np.linspace(minlat - interval_deg, maxlat + interval_deg, 180)
    lon_samples = np.linspace(minlon - interval_deg, maxlon + interval_deg, 180)
    for lon in lon_ticks:
        x, y = transformer.transform(np.full_like(lat_samples, lon), lat_samples)
        clipped = LineString(zip(x, y)).intersection(study_geom)
        plot_line_geometry(ax, clipped, "#EFF3F1", 0.30, 0.58, 6)
        plot_line_geometry(ax, clipped, "#7F9094", 0.12, 0.34, 6)
    for lat in lat_ticks:
        x, y = transformer.transform(lon_samples, np.full_like(lon_samples, lat))
        clipped = LineString(zip(x, y)).intersection(study_geom)
        plot_line_geometry(ax, clipped, "#EFF3F1", 0.30, 0.58, 6)
        plot_line_geometry(ax, clipped, "#7F9094", 0.12, 0.34, 6)


def add_wgs84_grid_labels(
    ax,
    boundary: gpd.GeoDataFrame,
    interval_deg: float = 0.25,
    label_step: int = 2,
    fontsize: float = 8.2,
    alpha: float = 0.80,
    outside: bool = True,
    offset_pts: float = 4.0,
) -> None:
    lon_ticks, lat_ticks, minlon, minlat = wgs84_ticks(boundary, interval_deg)
    transformer = wgs84_to_crs(boundary.crs.to_string())
    bounds = boundary.total_bounds
    x0, _ = ax.get_xlim()
    y0, _ = ax.get_ylim()
    for idx, lon in enumerate(lon_ticks):
        if idx % label_step:
            continue
        if abs(lon - 26.25) < 0.01:
            continue
        x, _ = transformer.transform(lon, minlat)
        if bounds[0] <= x <= bounds[2]:
            ax.annotate(
                f"{lon:.2f}E",
                xy=(x, y0),
                xycoords="data",
                xytext=(0, -offset_pts),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=fontsize,
                color=MUTED,
                alpha=alpha,
                zorder=20,
                clip_on=False,
            )
    for idx, lat in enumerate(lat_ticks):
        if idx % label_step:
            continue
        # Get correct y coordinate for this latitude
        _, y = transformer.transform(minlon, lat)
        if bounds[1] <= y <= bounds[3]:
            ax.annotate(
                f"{lat:.2f}N",
                xy=(x0, y),  # Lock to the absolute left limit of the axis
                xycoords="data",
                xytext=(-offset_pts, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=fontsize,
                color=MUTED,
                alpha=alpha,
                rotation=90,
                zorder=20,
                clip_on=False,
            )


def add_left_bottom_frame(ax, color: str = GRID, linewidth: float = 0.75, alpha: float = 0.9) -> None:
    ax.plot([0, 0], [0, 1], transform=ax.transAxes, color=color, linewidth=linewidth, alpha=alpha, zorder=40)
    ax.plot([0, 1], [0, 0], transform=ax.transAxes, color=color, linewidth=linewidth, alpha=alpha, zorder=40)


def finish_map(ax, boundary: gpd.GeoDataFrame, label: str) -> None:
    mask_outside_boundary(ax, boundary)
    draw_boundary_effect(ax, boundary)
    set_bounds(ax, boundary, pad_m=2200.0, adjustable="box")
    add_clipped_wgs84_graticule(ax, boundary, interval_deg=0.25)
    # Match offset_pts with matplotlib default tick padding (4.0) for perfect alignment
    add_wgs84_grid_labels(ax, boundary, interval_deg=0.25, label_step=2, fontsize=8.2, alpha=0.80, outside=True, offset_pts=4.0)
    add_left_bottom_frame(ax, linewidth=0.75, alpha=0.90)
    tag(ax, label)


def add_bivariate_classes(gdf: gpd.GeoDataFrame, x_col: str, y_col: str) -> gpd.GeoDataFrame:
    out = gdf.dropna(subset=[x_col, y_col]).copy()
    out["x_q"] = pd.qcut(out[x_col], 3, labels=[0, 1, 2], duplicates="drop").astype(int)
    out["y_q"] = pd.qcut(out[y_col], 3, labels=[0, 1, 2], duplicates="drop").astype(int)
    # 3x3 palette
    palette = np.array(
        [
            ["#8DA99A", "#93A8B9", "#8588A6"],
            ["#AFC0CA", "#D2D2CE", "#B49AB1"],
            ["#E2C8BB", "#D6A69A", "#B97872"],
        ]
    )
    out["bivar_color"] = [palette[y, x] for x, y in zip(out["x_q"], out["y_q"])]
    return out


def draw_bivariate_legend(ax, xlabel: str, ylabel: str, bounds: list[float]) -> None:
    # Instead of inset_axes which can be clipped/obscured, draw directly on the main ax
    x0, y0, w, h = bounds
    dx = w / 3
    dy = h / 3
    
    palette = [
        ["#8DA99A", "#93A8B9", "#8588A6"],
        ["#AFC0CA", "#D2D2CE", "#B49AB1"],
        ["#E2C8BB", "#D6A69A", "#B97872"],
    ]
    
    for yi in range(3):
        for xi in range(3):
            rect = Rectangle(
                (x0 + xi * dx, y0 + yi * dy), dx, dy,
                facecolor=palette[yi][xi], edgecolor="white", linewidth=0.7,
                transform=ax.transAxes, zorder=100, clip_on=False
            )
            ax.add_patch(rect)
            
    # Add labels relative to the custom legend box
    ax.text(x0 + w / 2, y0 - 0.015, xlabel, transform=ax.transAxes, ha="center", va="top", fontsize=6.4, color=INK, zorder=100, clip_on=False)
    ax.text(x0 - 0.015, y0 + h / 2, ylabel, transform=ax.transAxes, ha="right", va="center", rotation=90, fontsize=6.4, color=INK, zorder=100, clip_on=False)


def build_hex_grid(bounds: tuple[float, float, float, float], radius: float) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = bounds
    dx = radius * 2
    dy = math.sqrt(3) * radius
    cols = int(math.ceil((maxx - minx) / (dx * 0.75))) + 2
    rows = int(math.ceil((maxy - miny) / dy)) + 2
    hexes = []
    ids = []
    idx = 0
    for r in range(rows):
        cy = miny + r * dy
        for c in range(cols):
            cx = minx + c * dx * 0.75
            if r % 2 == 1:
                cx += dx * 0.375
            angles = np.linspace(0, 2 * math.pi, 7)[:-1] + math.pi / 6
            xs = cx + radius * np.cos(angles)
            ys = cy + radius * np.sin(angles)
            poly = box(min(xs), min(ys), max(xs), max(ys)).buffer(0)  # safe placeholder
            poly = gpd.GeoSeries([poly]).iloc[0]
            # Replace with true hex via shapely Polygon for speed/robustness.
            poly = gpd.GeoSeries([gpd.GeoSeries([poly]).iloc[0]]).iloc[0]
            from shapely.geometry import Polygon

            poly = Polygon(zip(xs, ys))
            hexes.append(poly)
            ids.append(idx)
            idx += 1
    return gpd.GeoDataFrame({"hex_id": ids, "geometry": hexes}, crs="EPSG:32635")


def build_social_hex(radius: float = 150.0) -> gpd.GeoDataFrame:
    if SOCIAL_HEX_CACHE.exists():
        return gpd.read_file(SOCIAL_HEX_CACHE)
    cols = ["x", "y", "lst_2025", "pop_total", "female_share", "child_0_14_share", "elderly_65_plus_share"]
    df = pd.read_csv(MODEL_GRID, usecols=cols).dropna()
    for col in ["female_share", "child_0_14_share", "elderly_65_plus_share"]:
        std = df[col].std()
        df[f"z_{col}"] = (df[col] - df[col].mean()) / std if std else 0
    df["socio_sensitivity"] = df[["z_female_share", "z_child_0_14_share", "z_elderly_65_plus_share"]].mean(axis=1)
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"], crs="EPSG:32635"))
    boundary = boundary_gdf()
    hexes = build_hex_grid(boundary.total_bounds, radius)
    joined = gpd.sjoin(pts, hexes, predicate="within", how="inner")
    agg = joined.groupby("hex_id").agg(
        n=("lst_2025", "size"),
        lst_mean=("lst_2025", "mean"),
        pop_sum=("pop_total", "sum"),
        female_share=("female_share", "mean"),
        child_share=("child_0_14_share", "mean"),
        elderly_share=("elderly_65_plus_share", "mean"),
        socio_sensitivity=("socio_sensitivity", "mean"),
    ).reset_index()
    result = hexes.merge(agg, on="hex_id")
    result = result[result["n"] >= 1].copy()
    result.to_file(SOCIAL_HEX_CACHE, layer="paper1_hex_150m_social_heat_summary", driver="GPKG")
    return result


def save_heat_justice_synthesis() -> None:
    social = build_social_hex()
    social = add_bivariate_classes(social, "socio_sensitivity", "lst_mean")
    social["log_pop_sum"] = np.log1p(social["pop_sum"].clip(lower=0))
    boundary = boundary_gdf()
    # Hillshade removed as requested

    fig = plt.figure(figsize=(12.0, 9.0), facecolor=BG)
    gs = fig.add_gridspec(2, 2, left=0.060, right=0.985, bottom=0.075, top=0.965, hspace=0.20, wspace=0.12)

    edgecolor = "#FBFBF8"
    lw = 0.08

    ax = fig.add_subplot(gs[0, 0])
    ax.set_anchor("W")  # Anchor to West (left) to align with bottom panel
    clean_ax(ax)
    # ax.imshow(np.ma.masked_invalid(shade), extent=shade_extent, cmap="Greys", vmin=0, vmax=1, alpha=0.20, interpolation="bilinear", zorder=0)
    social.plot(ax=ax, color=social["bivar_color"], edgecolor=edgecolor, linewidth=lw, alpha=0.92, zorder=2)
    finish_map(ax, boundary, "(a)")
    draw_bivariate_legend(ax, "sensitivity", "heat", bounds=[0.60, 0.515, 0.220, 0.220])

    ax = fig.add_subplot(gs[0, 1])
    ax.set_anchor("W")  # Anchor to West (left) to align with bottom panel
    clean_ax(ax)
    # ax.imshow(np.ma.masked_invalid(shade), extent=shade_extent, cmap="Greys", vmin=0, vmax=1, alpha=0.20, interpolation="bilinear", zorder=0)
    pop_cmap = LinearSegmentedColormap.from_list("pop_teal_coral", ["#2AA6A1", "#E9F1F0", "#F08A78"])
    pop_norm = Normalize(float(social["log_pop_sum"].quantile(0.02)), float(social["log_pop_sum"].quantile(0.98)))
    social.plot(ax=ax, column="log_pop_sum", cmap=pop_cmap, norm=pop_norm, edgecolor=edgecolor, linewidth=lw, alpha=0.92, zorder=2, legend=True, legend_kwds={"shrink": 0.62, "pad": 0.01, "label": "log(1 + population)"})
    finish_map(ax, boundary, "(b)")

    ax = fig.add_subplot(gs[1, 0])
    chart_ax(ax)
    social["heat_tercile"] = pd.qcut(social["lst_mean"], 3, labels=["cooler", "middle", "hotter"])
    share_cols = ["female_share", "child_share", "elderly_share"]
    grouped = social.groupby("heat_tercile", observed=False)[share_cols]
    means = grouped.mean()
    ses = grouped.sem()
    x = np.arange(len(means), dtype=float)
    width = 0.22
    colors = [NAVY, TEAL, ROSE]
    labels = ["female", "children", "elderly"]
    for idx, col in enumerate(share_cols):
        ax.bar(x + (idx - 1) * width, means[col], width=width, color=colors[idx], alpha=0.78, label=labels[idx])
        ax.errorbar(x + (idx - 1) * width, means[col], yerr=1.96 * ses[col], fmt="none", ecolor=INK, elinewidth=0.7, capsize=2.0, capthick=0.7, alpha=0.55, zorder=5)
    ax.set_xticks(x, means.index.astype(str).tolist(), fontsize=8.2, color=INK)
    ax.set_xlabel("")
    ax.set_ylabel("mean share (±95% CI)", fontsize=8.5, color=INK)
    ax.set_ylim(0, float(means.max().max() * 1.10))
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    tag(ax, "(c)")

    ax = fig.add_subplot(gs[1, 1])
    chart_ax(ax)
    draw_mahalle_moran_panel(ax)

    save_img(fig, FIG_OUT)


def main() -> None:
    save_heat_justice_synthesis()
    print(f"Wrote {FIG_OUT}")


if __name__ == "__main__":
    main()
