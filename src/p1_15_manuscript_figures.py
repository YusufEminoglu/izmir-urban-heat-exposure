"""Generate the final Study 1 manuscript figure set.

No draft labels such as "elite" or "basic" are used in outputs. Main figures are
`fig01` ... `fig08`; supplementary figures are `appendix_fig01` ...
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    gpd = None
try:
    import mapclassify
except ImportError:  # pragma: no cover
    mapclassify = None
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import rasterio
    from rasterio.features import geometry_mask
except ImportError:  # pragma: no cover
    rasterio = None
    geometry_mask = None
try:
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None
from matplotlib.colors import LightSource
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Rectangle
from matplotlib import transforms
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
try:
    from pyproj import Transformer
except ImportError:  # pragma: no cover
    Transformer = None
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
try:
    from shapely.geometry import LineString, Polygon, box
except ImportError:  # pragma: no cover
    LineString = Polygon = box = None
from sklearn.preprocessing import StandardScaler

try:
    from mgwr.gwr import MGWR
    from mgwr.sel_bw import Sel_BW
except ImportError:  # pragma: no cover
    MGWR = None
    Sel_BW = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _config import EXTERNAL_MUNICIPAL, FIGURES_DIR, DATA_INTERIM, DATA_MODELS, DATA_PROCESSED, TABLES_DIR, ensure_dirs  # noqa: E402
from _config import get_logger  # noqa: E402


FINAL_DIR = FIGURES_DIR / "final"
APPENDIX_DIR = FIGURES_DIR / "appendix"
FIGURE_METADATA_DIR = FIGURES_DIR / "metadata"
FIG03_CACHE_VERSION = 2

BOUNDARY = DATA_INTERIM / "izmir_study_boundary.gpkg"
ILCE = EXTERNAL_MUNICIPAL / "ilce.shp"
MAHALLE = DATA_PROCESSED / "mahalle_population_controls_2024.gpkg"
LCZ = DATA_INTERIM / "lcz_filter_izmir_2018_100m.tif"
LCZ_PROB = DATA_INTERIM / "lcz_probability_izmir_2018_100m.tif"
LST = DATA_PROCESSED / "aligned_covariates" / "izmir_LST_Veri_2025_aligned_100m.tif"
NDVI = DATA_PROCESSED / "aligned_covariates" / "izmir_NDVI_aligned_100m.tif"
NDBI = DATA_PROCESSED / "aligned_covariates" / "izmir_NDBI_aligned_100m.tif"
# NOTE: The file name is kept for continuity, but the content is EVI (not SAVI).
EVI = DATA_PROCESSED / "aligned_covariates" / "izmir_SAVI_aligned_100m.tif"
ALBEDO = DATA_PROCESSED / "aligned_covariates" / "izmir_Albedo_Veri_2025_aligned_100m.tif"
CANOPY = DATA_PROCESSED / "aligned_covariates" / "izmir_KanopiYuksekligi_Veri_aligned_100m.tif"
SRTM_ELEV = DATA_PROCESSED / "aligned_covariates" / "izmir_srtm_elevation_aligned_100m.tif"
SLOPE = DATA_PROCESSED / "aligned_covariates" / "izmir_srtm_slope_deg_aligned_100m.tif"
ASPECT_SIN = DATA_PROCESSED / "aligned_covariates" / "izmir_srtm_aspect_sin_aligned_100m.tif"
ASPECT_COS = DATA_PROCESSED / "aligned_covariates" / "izmir_srtm_aspect_cos_aligned_100m.tif"
UHI = DATA_PROCESSED / "aligned_covariates" / "izmir_UHI_Veri_2025_aligned_100m.tif"
UTFVI = DATA_PROCESSED / "aligned_covariates" / "izmir_UTFVI_Veri_2025_aligned_100m.tif"
CORINE_2018 = DATA_PROCESSED / "aligned_covariates" / "CORINE_izmir_2018_aligned_100m.tif"
GLC_2022 = DATA_PROCESSED / "aligned_covariates" / "izmir_GLC_FCS_2022_aligned_100m.tif"
MODEL_GRID = DATA_PROCESSED / "paper1_model_grid_100m_current.csv"
HEX_150 = DATA_PROCESSED / "paper1_hex_150m_summary.gpkg"
GWR = DATA_MODELS / "paper1_gwr_preliminary_sample_5000.gpkg"
MGWR_READY = DATA_PROCESSED / "paper1_mgwr_ready_sample_50k.gpkg"
MGWR_SAMPLE_N = 2500
MGWR_SAMPLE = DATA_MODELS / f"paper1_mgwr_sample_{MGWR_SAMPLE_N}.gpkg"
LST_CUBE = DATA_PROCESSED / "datacubes" / "paper1_landsat_lst_jja_cube_2014_2024.nc"
MYDATA = ROOT / "docs" / "manuscripts" / "paper1" / "src" / "mydata.dat"

FIGURES = {
    "fig01": FINAL_DIR / "fig01_conceptual_framework.png",
    "fig02": FINAL_DIR / "fig02_study_area_data_stack.png",
    "fig03": FINAL_DIR / "fig03_cloud_local_lst_datacube.png",
    "fig04": FINAL_DIR / "fig04_lcz_thermal_differentiation.png",
    "fig05": FINAL_DIR / "fig05_bivariate_hex_heat_vegetation.png",
    "fig06": FINAL_DIR / "fig06_baseline_model_diagnostics.png",
    "fig07": FINAL_DIR / "fig07_hex_spatial_coefficient_atlas.png",
    "fig08": FINAL_DIR / "fig08_heat_justice_sociodemographic_synthesis.png",
}

APPENDIX = {
    "appendix_fig01": APPENDIX_DIR / "appendix_fig01_neighborhood_population_cartogram.png",
    "appendix_fig02": APPENDIX_DIR / "appendix_fig02_srtm_terrain_controls.png",
    "appendix_fig03": APPENDIX_DIR / "appendix_fig03_hex_local_r2.png",
    "appendix_fig04": APPENDIX_DIR / "appendix_fig04_srtm_lst_3d_drape.png",
    "appendix_fig05": APPENDIX_DIR / "appendix_fig05_correlation_matrix.png",
}

BG = "#FFFFFF"
INK = "#344150"
MUTED = "#7A8590"
GRID = "#E7EBEA"
LIGHT = "#F1F3F1"
TEAL = "#86B7AA"
NAVY = "#667F95"
ROSE = "#B97872"
PEACH = "#E5C8BF"
BOUNDARY_LINE = "#2F3D4A"

CMAP_HEAT = LinearSegmentedColormap.from_list("heat_soft", [LIGHT, "#EAD8D1", "#D8AAA0", ROSE])
CMAP_COOL = LinearSegmentedColormap.from_list("cool_soft", [LIGHT, "#BFD7D0", TEAL, NAVY])
CMAP_DIV = LinearSegmentedColormap.from_list("div_soft", [NAVY, "#BFD7D0", LIGHT, PEACH, ROSE])
CMAP_GRAY = LinearSegmentedColormap.from_list("gray_soft", ["#FFFFFF", "#DADDD8", "#9EA8A8", "#5F6F76"])

LCZ_LABELS = {
    2: "Compact mid-rise",
    3: "Compact low-rise",
    5: "Open mid-rise",
    6: "Open low-rise",
    8: "Large low-rise",
    9: "Sparsely built",
    10: "Heavy industry",
    11: "Dense trees",
    12: "Scattered trees",
    13: "Bush/scrub",
    14: "Low plants",
    15: "Bare rock/paved",
    16: "Bare soil/sand",
    17: "Water",
}
LCZ_CLASSES = list(LCZ_LABELS)
LCZ_CMAP = ListedColormap(
    [
        "#6A625C",
        "#817870",
        "#9A8E80",
        "#B4AA9A",
        "#C5AA78",
        "#AA9279",
        "#786F67",
        "#D3D7CD",
        "#C6CCBB",
        "#B8C1A7",
        "#9FAC8F",
        "#C4D2D5",
        "#9DB2B8",
        "#7D9198",
    ]
)
LCZ_NORM = BoundaryNorm([value - 0.5 for value in LCZ_CLASSES] + [LCZ_CLASSES[-1] + 0.5], LCZ_CMAP.N)
BIVARIATE = np.array(
    [
        ["#8DA99A", "#93A8B9", "#8588A6"],
        ["#AFC0CA", "#D2D2CE", "#B49AB1"],
        ["#E2C8BB", "#D6A69A", "#B97872"],
    ]
)


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


def clean_ax(ax, title: str | None = None) -> None:
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=7)


def chart_ax(ax, title: str | None = None) -> None:
    ax.set_facecolor(BG)
    ax.grid(color=GRID, linewidth=0.7, alpha=0.9)
    ax.tick_params(colors=INK, labelsize=8, length=0)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    if title:
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=7)


def save_img(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, facecolor=BG, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def boundary_gdf(crs: str = "EPSG:32635") -> gpd.GeoDataFrame:
    return gpd.read_file(BOUNDARY).to_crs(crs)


def read_clipped_raster(path: Path) -> tuple[np.ndarray, tuple[float, float, float, float], rasterio.Affine, str, np.ndarray]:
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
        return arr, extent, src.transform, src.crs.to_string(), inside


def hillshade_array() -> tuple[np.ndarray, tuple[float, float, float, float], str]:
    dem, extent, _, crs, inside = read_clipped_raster(SRTM_ELEV)
    ls = LightSource(azdeg=315, altdeg=45)
    shade = ls.hillshade(np.nan_to_num(dem, nan=np.nanmedian(dem)), vert_exag=0.65, dx=100, dy=100)
    shade[~inside] = np.nan
    return shade, extent, crs


HILLSHADE_CACHE: tuple[np.ndarray, tuple[float, float, float, float], str] | None = None
TRANSFORMER_CACHE: dict[str, Transformer] = {}


def get_hillshade() -> tuple[np.ndarray, tuple[float, float, float, float], str]:
    global HILLSHADE_CACHE
    if HILLSHADE_CACHE is None:
        HILLSHADE_CACHE = hillshade_array()
    return HILLSHADE_CACHE


def wgs84_to_crs(crs) -> Transformer:
    key = str(crs)
    if key not in TRANSFORMER_CACHE:
        TRANSFORMER_CACHE[key] = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    return TRANSFORMER_CACHE[key]


def fill_internal_nodata_for_display(arr: np.ndarray, inside: np.ndarray, path: Path) -> np.ndarray:
    out = arr.copy()
    if path == CANOPY:
        out[np.isnan(out) & inside] = 0
        out[~inside] = np.nan
        return out
    out[~inside] = np.nan
    return out


def draw_boundary_effect(ax, boundary: gpd.GeoDataFrame, shadow_scale: float = 1.0) -> None:
    geom = boundary.geometry.union_all()
    rings = [
        (2000, 1500, "#EEF1EF", 0.58),
        (1500, 1000, "#E2E8E5", 0.50),
        (1000, 0, "#D3DEDA", 0.40),
    ]
    for outer_m, inner_m, color, alpha in rings:
        outer = geom.buffer(outer_m)
        inner = geom.buffer(inner_m) if inner_m else geom
        ring = gpd.GeoSeries([outer.difference(inner)], crs=boundary.crs)
        ring.plot(ax=ax, color=color, edgecolor="none", alpha=alpha, zorder=5)
    boundary.boundary.plot(ax=ax, color=BOUNDARY_LINE, linewidth=0.62, alpha=0.92, zorder=7)
    ax.collections[-1].set_path_effects(
        [
            pe.SimpleLineShadow(
                offset=(0.4 * shadow_scale, -0.4 * shadow_scale),
                alpha=0.055 * shadow_scale,
            ),
            pe.Normal(),
        ]
    )


def mask_outside_boundary(ax, boundary: gpd.GeoDataFrame, zorder: int = 4) -> None:
    minx, miny, maxx, maxy = boundary.total_bounds
    pad_x = 0.06 * (maxx - minx)
    pad_y = 0.06 * (maxy - miny)
    frame = box(minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y)
    outside = gpd.GeoSeries([frame], crs=boundary.crs).difference(boundary.geometry.union_all())
    outside.plot(ax=ax, color=BG, edgecolor="none", zorder=zorder)


def set_bounds(
    ax,
    boundary: gpd.GeoDataFrame,
    pad_m: float = 0.0,
    adjustable: str = "box",
    fill_to_axes: bool = False,
) -> None:
    bounds = boundary.total_bounds
    xmin = float(bounds[0] - pad_m)
    xmax = float(bounds[2] + pad_m)
    ymin = float(bounds[1] - pad_m)
    ymax = float(bounds[3] + pad_m)

    if fill_to_axes:
        # Expand the shorter dimension so the map fills the axes box while keeping equal aspect.
        box = ax.get_position()
        desired = float(box.width / max(box.height, 1e-9))
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        dx = xmax - xmin
        dy = ymax - ymin
        current = dx / max(dy, 1e-9)
        if current < desired:
            dx_new = desired * dy
            xmin = cx - 0.5 * dx_new
            xmax = cx + 0.5 * dx_new
        else:
            dy_new = dx / desired
            ymin = cy - 0.5 * dy_new
            ymax = cy + 0.5 * dy_new

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable=adjustable)


def wgs84_ticks(boundary: gpd.GeoDataFrame, interval_deg: float = 0.25) -> tuple[np.ndarray, np.ndarray, float, float]:
    b4326 = boundary.to_crs("EPSG:4326")
    minlon, minlat, maxlon, maxlat = b4326.total_bounds
    lon_ticks = np.arange(math.ceil(minlon / interval_deg) * interval_deg, maxlon + interval_deg, interval_deg)
    lat_ticks = np.arange(math.ceil(minlat / interval_deg) * interval_deg, maxlat + interval_deg, interval_deg)
    return lon_ticks, lat_ticks, minlon, minlat


def add_wgs84_grid_labels(
    ax,
    boundary: gpd.GeoDataFrame,
    interval_deg: float = 0.25,
    label_step: int = 2,
    fontsize: float = 5.0,
    alpha: float = 0.86,
    x_frac: float = 0.018,
    y_frac: float = 0.018,
    outside: bool = False,
    offset_pts: float = 3.0,
    outside_y_use_bounds: bool = False,
    outside_y_anchor_axes: bool = False,
    outside_y_axes_x: float = 0.0,
    outside_y_axes_ha: str = "right",
) -> None:
    lon_ticks, lat_ticks, minlon, minlat = wgs84_ticks(boundary, interval_deg)
    transformer = wgs84_to_crs(boundary.crs)
    bounds = boundary.total_bounds
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    label_y = y0 + y_frac * (y1 - y0)
    label_x = x0 + x_frac * (x1 - x0)
    for idx, lon in enumerate(lon_ticks):
        if idx % label_step:
            continue
        if math.isclose(lon, 26.25, abs_tol=1e-9):
            continue
        x, _ = transformer.transform(lon, minlat)
        if bounds[0] <= x <= bounds[2]:
            if outside:
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
            else:
                ax.text(x, label_y, f"{lon:.2f}E", ha="center", va="bottom", fontsize=fontsize, color=MUTED, alpha=alpha, zorder=20)
    for idx, lat in enumerate(lat_ticks):
        if idx % label_step:
            continue
        # Place latitude labels beside the left-most graticule (not the boundary bbox).
        lon_for_label = float(lon_ticks[0]) if len(lon_ticks) else float(minlon)
        x_line, y = transformer.transform(lon_for_label, lat)
        if bounds[1] <= y <= bounds[3]:
            if outside:
                x_anchor = x_line if outside_y_use_bounds else x0
                if outside_y_anchor_axes:
                    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
                    ax.text(
                        outside_y_axes_x,
                        y,
                        f"{lat:.2f}N",
                        transform=trans,
                        ha=outside_y_axes_ha,
                        va="center",
                        fontsize=fontsize,
                        color=MUTED,
                        alpha=alpha,
                        rotation=90,
                        zorder=20,
                        clip_on=False,
                    )
                else:
                    ax.annotate(
                        f"{lat:.2f}N",
                        xy=(x_anchor, y),
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
            else:
                ax.text(label_x, y, f"{lat:.2f}N", ha="left", va="center", fontsize=fontsize, color=MUTED, alpha=alpha, rotation=90, zorder=20)


def add_map_grid(
    ax,
    boundary: gpd.GeoDataFrame,
    interval_deg: float = 0.25,
    label_step: int = 2,
    fontsize: float = 10.0,
    alpha: float = 0.86,
    x_frac: float = 0.018,
    y_frac: float = 0.018,
    label_outside: bool = False,
    label_offset_pts: float = 3.0,
    outside_y_use_bounds: bool = False,
    outside_y_anchor_axes: bool = False,
    outside_y_axes_x: float = 0.0,
    outside_y_axes_ha: str = "right",
) -> None:
    add_clipped_wgs84_graticule(ax, boundary, interval_deg=interval_deg)
    add_wgs84_grid_labels(
        ax,
        boundary,
        interval_deg=interval_deg,
        label_step=label_step,
        fontsize=fontsize,
        alpha=alpha,
        x_frac=x_frac,
        y_frac=y_frac,
        outside=label_outside,
        offset_pts=label_offset_pts,
        outside_y_use_bounds=outside_y_use_bounds,
        outside_y_anchor_axes=outside_y_anchor_axes,
        outside_y_axes_x=outside_y_axes_x,
        outside_y_axes_ha=outside_y_axes_ha,
    )


def add_left_bottom_frame(ax, color: str = GRID, linewidth: float = 0.75, alpha: float = 0.9) -> None:
    ax.plot([0, 0], [0, 1], transform=ax.transAxes, color=color, linewidth=linewidth, alpha=alpha, zorder=40)
    ax.plot([0, 1], [0, 0], transform=ax.transAxes, color=color, linewidth=linewidth, alpha=alpha, zorder=40)


def plot_line_geometry(ax, geom, color: str, linewidth: float, alpha: float, zorder: int) -> None:
    if geom.is_empty:
        return
    if geom.geom_type == "LineString":
        x, y = geom.xy
        ax.plot(x, y, color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)
    elif geom.geom_type in {"MultiLineString", "GeometryCollection"}:
        for part in geom.geoms:
            plot_line_geometry(ax, part, color, linewidth, alpha, zorder)


def add_clipped_wgs84_graticule(ax, boundary: gpd.GeoDataFrame, interval_deg: float = 0.25) -> None:
    lon_ticks, lat_ticks, minlon, minlat = wgs84_ticks(boundary, interval_deg)
    b4326 = boundary.to_crs("EPSG:4326")
    _, _, maxlon, maxlat = b4326.total_bounds
    transformer = wgs84_to_crs(boundary.crs)
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


def add_north_arrow(
    ax,
    x_frac: float = 0.94,
    y_frac: float = 0.17,
    dy_frac: float = 0.10,
    color: str = INK,
    fontsize: float = 7.5,
    lw: float = 0.85,
) -> None:
    ax.annotate(
        "N",
        xy=(x_frac, y_frac),
        xytext=(x_frac, y_frac - dy_frac),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0),
        zorder=20,
    )


def add_scale_bar(ax, length_km: int = 20, x_frac: float = 0.055, y_frac: float = 0.055) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    length = length_km * 1000
    sx = x0 + x_frac * (x1 - x0)
    sy = y0 + y_frac * (y1 - y0)
    ax.plot([sx, sx + length], [sy, sy], color=INK, linewidth=1.7, solid_capstyle="butt", zorder=20)
    ax.plot([sx, sx], [sy - 900, sy + 900], color=INK, linewidth=0.9, zorder=20)
    ax.plot([sx + length, sx + length], [sy - 900, sy + 900], color=INK, linewidth=0.9, zorder=20)
    ax.text(sx + length / 2, sy + 1800, f"{length_km} km", ha="center", va="bottom", fontsize=6.2, color=INK, zorder=20)


def add_soft_segmented_scale_bar(
    ax,
    x_frac: float = 0.85,
    y_frac: float = 0.62,
    width_frac: float = 0.22,
    segments: int = 4,
    total_km: int = 20,
    height_frac: float = 0.013,
    color_a: str = "#C7D3D8",
    color_b: str = "#9FB2BA",
    text_color: str = MUTED,
) -> None:
    x0 = x_frac - width_frac / 2
    y0 = y_frac
    seg_w = width_frac / segments
    for idx in range(segments):
        face = color_a if idx % 2 == 0 else color_b
        ax.add_patch(
            Rectangle(
                (x0 + idx * seg_w, y0),
                seg_w,
                height_frac,
                transform=ax.transAxes,
                facecolor=face,
                edgecolor="white",
                linewidth=0.45,
                zorder=22,
                clip_on=False,
            )
        )
    for idx in range(segments + 1):
        xi = x0 + idx * seg_w
        ax.plot(
            [xi, xi],
            [y0, y0 - 0.010],
            transform=ax.transAxes,
            color=MUTED,
            linewidth=0.45,
            zorder=23,
            clip_on=False,
        )
        ax.text(
            xi,
            y0 - 0.018,
            f"{int(total_km * idx / segments)}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=4.6,
            color=text_color,
            zorder=23,
            clip_on=False,
        )
    ax.text(
        x0 + width_frac + 0.012,
        y0 - 0.018,
        "km",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=4.6,
        color=text_color,
        zorder=23,
        clip_on=False,
    )


def add_mini_panel_furniture(ax, x_frac: float = 0.85, y_frac: float = 0.70) -> None:
    soft_color = "#91A4AD"
    add_north_arrow(
        ax,
        x_frac=x_frac,
        y_frac=y_frac + 0.12,
        dy_frac=0.065,
        color=soft_color,
        fontsize=5.8,
        lw=0.65,
    )
    add_soft_segmented_scale_bar(
        ax,
        x_frac=x_frac,
        y_frac=y_frac - 0.005,
        width_frac=0.20,
        segments=4,
        total_km=20,
        height_frac=0.012,
        color_a="#D5E0E3",
        color_b="#A9BCC3",
        text_color=soft_color,
    )


def add_mini_horizontal_legend(ax, mappable, label: str, x_center: float = 0.85, y_center: float = 0.50) -> None:
    width = 0.30
    height = 0.020
    x0 = x_center - width / 2
    y0 = y_center - height / 2
    cax = ax.inset_axes([x0, y0, width, height])
    cb = ax.figure.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=5.6, colors=INK, length=0, pad=0.2)
    cb.outline.set_visible(False)
    cax.set_xlabel(label, fontsize=5.8, color=MUTED, labelpad=0.3)
    return cb


def mini_panel_header(ax, label: str, title: str, y: float = 1.018) -> None:
    ax.text(
        0.012,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        zorder=50,
        clip_on=False,
        bbox=dict(boxstyle="square,pad=0.12", facecolor=BG, edgecolor="none", alpha=0.85),
    )
    ax.text(
        0.105,
        y,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        fontweight="bold",
        color=INK,
        zorder=30,
        clip_on=False,
        bbox=dict(boxstyle="square,pad=0.12", facecolor=BG, edgecolor="none", alpha=0.78),
    )


def draw_figure_level_panel_header(fig: plt.Figure, ax, label: str, title: str) -> None:
    bbox = ax.get_position()
    y = bbox.y1 + 0.0065
    fig.text(
        bbox.x0 + 0.012 * bbox.width,
        y,
        label,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        bbox=dict(boxstyle="square,pad=0.12", facecolor=BG, edgecolor="none", alpha=0.85),
    )
    fig.text(
        bbox.x0 + 0.105 * bbox.width,
        y,
        title,
        ha="left",
        va="bottom",
        fontsize=8.6,
        fontweight="bold",
        color=INK,
        bbox=dict(boxstyle="square,pad=0.12", facecolor=BG, edgecolor="none", alpha=0.78),
    )


def finish_map(
    ax,
    boundary: gpd.GeoDataFrame,
    label: str | None = None,
    grid: bool = True,
    furniture: bool = True,
    pad_m: float = 2200.0,
    grid_label_fontsize: float = 10.0,
    grid_label_alpha: float = 0.86,
    grid_x_frac: float = 0.018,
    grid_y_frac: float = 0.018,
    grid_label_outside: bool = False,
    grid_label_offset_pts: float = 3.0,
    grid_outside_y_use_bounds: bool = False,
    grid_outside_y_anchor_axes: bool = False,
    grid_outside_y_axes_x: float = 0.0,
    grid_outside_y_axes_ha: str = "right",
    frame_left_bottom: bool = False,
    aspect_adjustable: str = "box",
    fill_to_axes: bool = False,
    shadow_scale: float = 1.0,
) -> None:
    mask_outside_boundary(ax, boundary)
    draw_boundary_effect(ax, boundary, shadow_scale=shadow_scale)
    set_bounds(ax, boundary, pad_m=pad_m, adjustable=aspect_adjustable, fill_to_axes=fill_to_axes)
    if grid:
        add_map_grid(
            ax,
            boundary,
            fontsize=grid_label_fontsize,
            alpha=grid_label_alpha,
            x_frac=grid_x_frac,
            y_frac=grid_y_frac,
            label_outside=grid_label_outside,
            label_offset_pts=grid_label_offset_pts,
            outside_y_use_bounds=grid_outside_y_use_bounds,
            outside_y_anchor_axes=grid_outside_y_anchor_axes,
            outside_y_axes_x=grid_outside_y_axes_x,
            outside_y_axes_ha=grid_outside_y_axes_ha,
        )
    if furniture:
        add_north_arrow(ax)
        add_scale_bar(ax)
    if frame_left_bottom:
        add_left_bottom_frame(ax)
    if label:
        tag(ax, label)


def raster_panel(
    ax,
    path: Path,
    title: str,
    cmap,
    label: str,
    vmin=None,
    vmax=None,
    norm=None,
    cbar_label: str | None = None,
    *,
    grid_label_fontsize: float = 10.0,
    grid_label_alpha: float = 0.86,
    grid_x_frac: float = 0.018,
    grid_y_frac: float = 0.018,
    grid_label_outside: bool = False,
    grid_label_offset_pts: float = 3.0,
    frame_left_bottom: bool = False,
    furniture: bool = True,
):
    arr, extent, _, crs, inside = read_clipped_raster(path)
    boundary = boundary_gdf(crs)
    if path == LCZ:
        arr = np.where(np.isin(arr, LCZ_CLASSES), arr, np.nan)
    else:
        arr = fill_internal_nodata_for_display(arr, inside, path)
    if vmin is None and vmax is None and norm is None:
        vmin, vmax = np.nanpercentile(arr, [2, 98])
    clean_ax(ax, title)
    # Hillshade removed as requested
    im = ax.imshow(np.ma.masked_invalid(arr), extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, interpolation="nearest", alpha=0.82, zorder=2)
    finish_map(
        ax,
        boundary,
        label,
        furniture=furniture,
        grid_label_fontsize=grid_label_fontsize,
        grid_label_alpha=grid_label_alpha,
        grid_x_frac=grid_x_frac,
        grid_y_frac=grid_y_frac,
        grid_label_outside=grid_label_outside,
        grid_label_offset_pts=grid_label_offset_pts,
        frame_left_bottom=frame_left_bottom,
    )
    if cbar_label:
        cb = ax.figure.colorbar(im, ax=ax, shrink=0.68, pad=0.012)
        cb.set_label(cbar_label, fontsize=8, color=INK)
        cb.ax.tick_params(labelsize=7, colors=INK, length=0)
        cb.outline.set_visible(False)
    return im


def mini_raster_panel(
    ax,
    path: Path,
    title: str,
    cmap,
    label: str,
    vmin=None,
    vmax=None,
    norm=None,
    cbar_label: str | None = None,
    pad_m: float = 2200.0,
    grid_label_outside: bool = False,
    grid_label_offset_pts: float = 2.0,
    draw_header: bool = True,
    shadow_scale: float = 1.0,
):
    arr, extent, _, crs, inside = read_clipped_raster(path)
    boundary = boundary_gdf(crs)
    if path == LCZ:
        arr = np.where(np.isin(arr, LCZ_CLASSES), arr, np.nan)
    else:
        arr = fill_internal_nodata_for_display(arr, inside, path)
    if vmin is None and vmax is None and norm is None:
        vmin, vmax = np.nanpercentile(arr, [2, 98])
    clean_ax(ax)
    # Hillshade removed as requested
    im = ax.imshow(np.ma.masked_invalid(arr), extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, interpolation="nearest", alpha=0.84, zorder=2)
    mask_outside_boundary(ax, boundary)
    draw_boundary_effect(ax, boundary, shadow_scale=shadow_scale)
    set_bounds(ax, boundary, pad_m=pad_m)
    add_map_grid(
        ax,
        boundary,
        fontsize=6.1,
        alpha=0.70,
        x_frac=0.010,
        y_frac=0.008,
        label_outside=grid_label_outside,
        label_offset_pts=grid_label_offset_pts,
    )
    add_left_bottom_frame(ax, linewidth=0.65, alpha=0.70)
    if draw_header:
        mini_panel_header(ax, label, title)
    add_mini_panel_furniture(ax, x_frac=0.80, y_frac=0.70)
    if cbar_label and path != LCZ:
        add_mini_horizontal_legend(ax, im, cbar_label, x_center=0.80, y_center=0.50)
    return im


def mini_lcz_legend_panel(ax, draw_header: bool = True, header_y: float = 1.018) -> None:
    clean_ax(ax)
    ax.set_aspect("auto")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if draw_header:
        mini_panel_header(ax, "(p)", "LCZ Classes", y=header_y)
    # Fill the panel more like a map panel so the visual weight matches neighboring cells.
    y_values = np.linspace(0.82, 0.16, 7)
    for idx, value in enumerate(LCZ_CLASSES):
        col = idx // 7
        row = idx % 7
        x = 0.07 + col * 0.45
        y = y_values[row]
        ax.add_patch(Rectangle((x, y - 0.022), 0.050, 0.036, facecolor=LCZ_CMAP(LCZ_NORM([value])[0]), edgecolor="#FFFFFF", linewidth=0.35))
        ax.text(x + 0.066, y, f"{value} {LCZ_LABELS[value]}", ha="left", va="center", fontsize=6.4, color=INK)


def mini_population_panel(
    ax,
    label: str = "(d)",
    pad_m: float = 2200.0,
    grid_label_outside: bool = False,
    grid_label_offset_pts: float = 2.0,
    draw_header: bool = True,
    shadow_scale: float = 1.0,
) -> None:
    clean_ax(ax)
    boundary = boundary_gdf()
    mahalle = gpd.read_file(MAHALLE).to_crs("EPSG:32635")
    # Hillshade removed as requested
    mahalle.plot(ax=ax, column="pop_total", cmap=CMAP_GRAY, linewidth=0.02, edgecolor="#FFFFFF", alpha=0.84, legend=False, zorder=2)
    mask_outside_boundary(ax, boundary)
    draw_boundary_effect(ax, boundary, shadow_scale=shadow_scale)
    set_bounds(ax, boundary, pad_m=pad_m)
    add_map_grid(ax, boundary, fontsize=6.1, alpha=0.70, x_frac=0.010, y_frac=0.008, label_outside=grid_label_outside, label_offset_pts=grid_label_offset_pts)
    add_left_bottom_frame(ax, linewidth=0.65, alpha=0.70)
    if draw_header:
        mini_panel_header(ax, label, "Population")
    norm = Normalize(vmin=float(mahalle["pop_total"].quantile(0.02)), vmax=float(mahalle["pop_total"].quantile(0.98)))
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=CMAP_GRAY)
    add_mini_panel_furniture(ax, x_frac=0.80, y_frac=0.70)
    add_mini_horizontal_legend(ax, mappable, "population", x_center=0.80, y_center=0.50)


def lcz_axis_label(lcz: int) -> str:
    label = LCZ_LABELS[lcz].replace("/", "/ ")
    words = label.split()
    if len(words) <= 1:
        return f"{lcz}\n{label}"
    if len(words) == 2:
        return f"{lcz} {words[0]}\n{words[1]}"
    split = math.ceil(len(words) / 2)
    return f"{lcz} {' '.join(words[:split])}\n{' '.join(words[split:])}"


def add_lcz_legend(ax, outside: bool = True) -> None:
    handles = [Patch(facecolor=LCZ_CMAP(LCZ_NORM([value])[0]), edgecolor="none", label=f"{value}: {LCZ_LABELS[value]}") for value in LCZ_CLASSES]
    if outside:
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.015),
            frameon=True,
            framealpha=0.94,
            fontsize=5.8,
            ncol=2,
            title="LCZ",
            title_fontsize=6.6,
            borderpad=0.35,
            handlelength=1.0,
            handletextpad=0.35,
            columnspacing=0.65,
        )
    else:
        ax.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0.015, 0.015),
            frameon=True,
            framealpha=0.90,
            fontsize=4.4,
            ncol=2,
            title="LCZ",
            title_fontsize=5.2,
            borderpad=0.25,
            handlelength=0.8,
            handletextpad=0.25,
            columnspacing=0.42,
        )


def hexagon(cx: float, cy: float, radius: float) -> Polygon:
    angles = np.deg2rad(np.arange(0, 360, 60) + 30)
    return Polygon([(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles])


def build_hex_grid(bounds: np.ndarray, radius: float, crs: str = "EPSG:32635") -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = bounds
    dx = math.sqrt(3) * radius
    dy = 1.5 * radius
    rows = []
    y = miny - dy
    row = 0
    while y <= maxy + dy:
        x_offset = 0 if row % 2 == 0 else dx / 2
        x = minx - dx
        while x <= maxx + dx:
            rows.append({"hex_id": len(rows), "geometry": hexagon(x + x_offset, y, radius)})
            x += dx
        y += dy
        row += 1
    return gpd.GeoDataFrame(rows, crs=crs)


def clip_to_boundary(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    boundary = boundary_gdf(gdf.crs.to_string() if gdf.crs else "EPSG:32635")
    return gpd.clip(gdf, boundary)


def add_bivariate_classes(gdf: gpd.GeoDataFrame, x_col: str, y_col: str, out_col: str = "bivar_color") -> gpd.GeoDataFrame:
    out = gdf.dropna(subset=[x_col, y_col]).copy()
    out["x_q"] = pd.qcut(out[x_col], 3, labels=[0, 1, 2], duplicates="drop").astype(int)
    out["y_q"] = pd.qcut(out[y_col], 3, labels=[0, 1, 2], duplicates="drop").astype(int)
    out[out_col] = [BIVARIATE[y, x] for x, y in zip(out["x_q"], out["y_q"])]
    return out


def draw_bivariate_legend(
    ax,
    xlabel: str,
    ylabel: str,
    counts: np.ndarray | None = None,
    bounds: list[float] | None = None,
) -> None:
    if bounds is None:
        bounds = [0.035, 0.035, 0.168, 0.168]
    inset = ax.inset_axes(bounds)
    total = counts.sum() if counts is not None else 0
    for yi in range(3):
        for xi in range(3):
            inset.add_patch(Rectangle((xi, yi), 1, 1, facecolor=BIVARIATE[yi, xi], edgecolor="white", linewidth=0.7))
            if counts is not None and total > 0:
                share = 100.0 * counts[yi, xi] / total
                text_color = INK if share < 18 else "#FFFFFF"
                inset.text(xi + 0.5, yi + 0.50, f"{share:.1f}", ha="center", va="center", fontsize=5.2, color=text_color, fontweight="bold")
    inset.set_xlim(0, 3)
    inset.set_ylim(0, 3)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(False)
    inset.text(1.5, -0.35, xlabel, ha="center", va="top", fontsize=6.4, color=INK)
    inset.text(-0.35, 1.5, ylabel, ha="right", va="center", rotation=90, fontsize=6.4, color=INK)
    if counts is not None and total > 0:
        inset.text(1.5, 3.18, "% of 150 m hexes", ha="center", va="bottom", fontsize=5.3, color=MUTED)


def model_points(columns: list[str]) -> gpd.GeoDataFrame:
    df = pd.read_csv(MODEL_GRID, usecols=["x", "y", *columns]).dropna(subset=["x", "y"])
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"], crs="EPSG:32635"))


def build_environment_hex(radius: float = 150.0) -> gpd.GeoDataFrame:
    out = HEX_150
    if out.exists() and out.stat().st_mtime >= MODEL_GRID.stat().st_mtime:
        return gpd.read_file(out)
    cols = ["lst_2025", "ndvi_2025", "ndbi_2025", "albedo_2025", "evi_2025", "tree_canopy_height_2025"]
    pts = model_points(cols).dropna(subset=["lst_2025", "ndvi_2025"])
    boundary = boundary_gdf()
    hexes = build_hex_grid(boundary.total_bounds, radius)
    joined = gpd.sjoin(pts, hexes, predicate="within", how="inner")
    agg = joined.groupby("hex_id").agg(
        n=("lst_2025", "size"),
        lst_mean=("lst_2025", "mean"),
        ndvi_mean=("ndvi_2025", "mean"),
        ndbi_mean=("ndbi_2025", "mean"),
        albedo_mean=("albedo_2025", "mean"),
        evi_mean=("evi_2025", "mean"),
        canopy_mean=("tree_canopy_height_2025", "mean"),
    ).reset_index()
    result = hexes.merge(agg, on="hex_id")
    result = result[result["n"] >= 1].copy()
    result.to_file(out, layer="paper1_hex_150m_summary", driver="GPKG")
    return result


def build_social_hex(radius: float = 150.0) -> gpd.GeoDataFrame:
    out = DATA_PROCESSED / "paper1_hex_150m_social_heat_summary.gpkg"
    if out.exists():
        return gpd.read_file(out)
    cols = ["x", "y", "lst_2025", "pop_total", "female_share", "child_0_14_share", "elderly_65_plus_share"]
    df = pd.read_csv(MODEL_GRID, usecols=cols).dropna()
    for col in ["female_share", "child_0_14_share", "elderly_65_plus_share"]:
        std = df[col].std()
        df[f"z_{col}"] = (df[col] - df[col].mean()) / std if std else 0
    df["socio_sensitivity"] = df[["z_female_share", "z_child_0_14_share", "z_elderly_65_plus_share"]].mean(axis=1)
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"], crs="EPSG:32635"))
    boundary = boundary_gdf()
    # Keep *full* hex geometries (no geometric clipping). We only filter to hexes that
    # intersect the boundary; masking is handled at render time. This avoids jagged
    # edge polygons that can look like Voronoi cells.
    hexes = build_hex_grid(boundary.total_bounds, radius)
    boundary_union = boundary.geometry.union_all()
    hexes = hexes[hexes.intersects(boundary_union)].copy()
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
    # Keep only hexes with population-matched samples. The full hex lattice is still
    # used in plotting as a muted base layer to show the study-area footprint.
    result = hexes.merge(agg, on="hex_id", how="inner")
    result = result[result["n"] >= 1].copy()
    result.to_file(out, layer="paper1_hex_150m_social_heat_summary", driver="GPKG")
    return result


def aggregate_gwr_to_hex(radius: float = 150.0) -> gpd.GeoDataFrame:
    out = DATA_MODELS / "paper1_gwr_preliminary_hex_150m_idw.gpkg"
    if out.exists():
        return gpd.read_file(out)
    gwr = gpd.read_file(GWR).to_crs("EPSG:32635")
    boundary = boundary_gdf()
    grid_pts = model_points(["lst_2025"]).dropna(subset=["lst_2025"])
    hexes = build_hex_grid(boundary.total_bounds, radius)
    joined_grid = gpd.sjoin(grid_pts, hexes, predicate="within", how="inner")
    keep = joined_grid.groupby("hex_id").agg(n_grid=("lst_2025", "size")).reset_index()
    result = hexes.merge(keep, on="hex_id")
    centroids = result.geometry.representative_point()
    xy = np.column_stack([centroids.x, centroids.y])
    sample_xy = np.column_stack([gwr.geometry.x, gwr.geometry.y])
    tree = cKDTree(sample_xy)
    coef_cols = [col for col in gwr.columns if col.startswith("gwr_coef_")] + ["gwr_local_r2"]
    dist, idx = tree.query(xy, k=min(8, len(gwr)))
    dist = np.atleast_2d(dist)
    idx = np.atleast_2d(idx)
    weights = 1 / np.maximum(dist, 1.0) ** 2
    weights = weights / weights.sum(axis=1, keepdims=True)
    for col in coef_cols:
        vals = gwr[col].to_numpy()[idx]
        result[col] = np.sum(vals * weights, axis=1)
    result["n"] = result["n_grid"]
    result.to_file(out, layer="paper1_gwr_preliminary_hex_150m_idw", driver="GPKG")
    return result


def ensure_mgwr_sample_results() -> gpd.GeoDataFrame:
    if MGWR_SAMPLE.exists():
        return gpd.read_file(MGWR_SAMPLE).to_crs("EPSG:32635")
    if MGWR is None or Sel_BW is None:
        raise ModuleNotFoundError("`mgwr` is required to build MGWR results for fig07.")
    sample = gpd.read_file(MGWR_READY).to_crs("EPSG:32635")
    predictors = ["ndvi_2025", "ndbi_2025", "albedo_2025"]
    use = ["x", "y", "lst_2025", *predictors]
    sample = sample[use + ["geometry"]].dropna().reset_index(drop=True).copy()
    if len(sample) > MGWR_SAMPLE_N:
        sample = sample.sample(MGWR_SAMPLE_N, random_state=42).reset_index(drop=True)
    coords = sample[["x", "y"]].to_numpy()
    y = sample[["lst_2025"]].to_numpy()
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X = x_scaler.fit_transform(sample[predictors].to_numpy())
    y_scaled = y_scaler.fit_transform(y)
    selector = Sel_BW(coords, y_scaled, X, kernel="bisquare", fixed=False, multi=True, constant=True, n_jobs=1)
    selector.search(criterion="AICc", max_iter_multi=3, tol_multi=2e-3)
    model = MGWR(coords, y_scaled, X, selector=selector, kernel="bisquare", fixed=False, constant=True, n_jobs=1)
    results = model.fit()
    params = results.params
    out = sample.copy()
    out["mgwr_intercept"] = params[:, 0]
    for idx, name in enumerate(predictors, start=1):
        out[f"mgwr_coef_{name}"] = params[:, idx]
    out["mgwr_predicted_scaled"] = results.predy.flatten()
    out["mgwr_residual_scaled"] = results.resid_response.flatten()
    # MGWR does not currently expose localR2; compute a lightweight neighbourhood fit proxy
    # using bisquare weights within a fixed metric bandwidth.
    fit_bw_m = 20000.0
    xy = coords.astype("float64")
    tree = cKDTree(xy)
    idxs = tree.query_ball_point(xy, r=fit_bw_m)
    y_true = y_scaled.flatten().astype("float64")
    y_pred = out["mgwr_predicted_scaled"].to_numpy(dtype="float64")
    local_r2 = np.full(len(out), np.nan, dtype="float32")
    for i, neigh in enumerate(idxs):
        if len(neigh) < 25:
            continue
        d = np.linalg.norm(xy[neigh] - xy[i], axis=1)
        w = (1 - (d / fit_bw_m) ** 2) ** 2
        w[d >= fit_bw_m] = 0.0
        if w.sum() <= 0:
            continue
        yt = y_true[neigh]
        yp = y_pred[neigh]
        ybar = np.sum(w * yt) / np.sum(w)
        sst = np.sum(w * (yt - ybar) ** 2)
        sse = np.sum(w * (yt - yp) ** 2)
        if sst > 0:
            local_r2[i] = float(max(0.0, 1.0 - sse / sst))
    out["mgwr_local_r2"] = local_r2
    out_gdf = gpd.GeoDataFrame(out, geometry=gpd.points_from_xy(out["x"], out["y"], crs="EPSG:32635"), crs="EPSG:32635")
    MGWR_SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    out_gdf.to_file(MGWR_SAMPLE, layer="paper1_mgwr_sample_50k", driver="GPKG")
    return out_gdf


def aggregate_mgwr_to_hex(radius: float = 150.0) -> gpd.GeoDataFrame:
    out = DATA_MODELS / "paper1_mgwr_hex_150m_idw.gpkg"
    if out.exists():
        return gpd.read_file(out)
    mgwr = ensure_mgwr_sample_results()
    boundary = boundary_gdf()
    grid_pts = model_points(["lst_2025"]).dropna(subset=["lst_2025"])
    # Keep full hex geometries; filter to boundary intersection (do not clip).
    hexes = build_hex_grid(boundary.total_bounds, radius)
    boundary_union = boundary.geometry.union_all()
    hexes = hexes[hexes.intersects(boundary_union)].copy()
    joined_grid = gpd.sjoin(grid_pts, hexes, predicate="within", how="inner")
    keep = joined_grid.groupby("hex_id").agg(n_grid=("lst_2025", "size")).reset_index()
    result = hexes.merge(keep, on="hex_id", how="left")
    result["n_grid"] = result["n_grid"].fillna(0).astype(int)
    centroids = result.geometry.representative_point()
    xy = np.column_stack([centroids.x, centroids.y])
    sample_xy = np.column_stack([mgwr.geometry.x, mgwr.geometry.y])
    tree = cKDTree(sample_xy)
    coef_cols = [col for col in mgwr.columns if col.startswith("mgwr_coef_")] + ["mgwr_local_r2"]
    dist, idx = tree.query(xy, k=min(12, len(mgwr)))
    dist = np.atleast_2d(dist)
    idx = np.atleast_2d(idx)
    weights = 1 / np.maximum(dist, 1.0) ** 2
    weights = weights / weights.sum(axis=1, keepdims=True)
    for col in coef_cols:
        vals = mgwr[col].to_numpy()[idx]
        result[col] = np.sum(vals * weights, axis=1)
    result["n"] = result["n_grid"]
    # MGWR local fit proxy can be undefined near sparse-sample edges; keep as NaN and
    # render transparently as no-data rather than imputing values.
    result.to_file(out, layer="paper1_mgwr_hex_150m_idw", driver="GPKG")
    return result


def save_conceptual_framework() -> None:
    fig = plt.figure(figsize=(12.4, 6.8), facecolor=BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.45], wspace=0.045)
    ax_map = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])
    clean_ax(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if gpd is not None and MAHALLE.exists():
        mahalle = gpd.read_file(MAHALLE).to_crs(32635)
        boundary = gpd.read_file(BOUNDARY).to_crs(32635) if BOUNDARY.exists() else mahalle.dissolve()
        matched = mahalle[mahalle["pop_total"].notna()].copy()
        not_listed = mahalle[mahalle["pop_total"].isna()].copy()
        ax_map.set_facecolor(BG)
        if not not_listed.empty:
            not_listed.plot(ax=ax_map, facecolor="#F8F2E8", edgecolor="#D5CEC2", linewidth=0.18, hatch="///", zorder=1)
        if not matched.empty:
            matched.plot(ax=ax_map, facecolor="#BFD9D1", edgecolor="#FFFFFF", linewidth=0.13, alpha=0.92, zorder=2)
        boundary.boundary.plot(ax=ax_map, color=INK, linewidth=1.05, zorder=4)
        minx, miny, maxx, maxy = boundary.total_bounds
        pad_x = (maxx - minx) * 0.055
        pad_y = (maxy - miny) * 0.055
        ax_map.set_xlim(minx - pad_x, maxx + pad_x)
        ax_map.set_ylim(miny - pad_y, maxy + pad_y)
        ax_map.set_aspect("equal")
        ax_map.axis("off")
        ax_map.text(0.02, 0.98, "(a) Study area and analysis units", transform=ax_map.transAxes, ha="left", va="top", fontsize=11.2, fontweight="bold", color=INK)
        ax_map.text(0.02, 0.925, "Izmir Province boundary with mahalle reporting support", transform=ax_map.transAxes, ha="left", va="top", fontsize=8.3, color=MUTED)
        handles = [
            Patch(facecolor="#BFD9D1", edgecolor="#FFFFFF", label="Matched ADNKS mahalle"),
            Patch(facecolor="#F8F2E8", edgecolor="#D5CEC2", hatch="///", label="<500 / OSB / not listed"),
            Line2D([0], [0], color=INK, linewidth=1.05, label="Province boundary"),
        ]
        ax_map.legend(handles=handles, loc="lower left", frameon=False, fontsize=7.2, handlelength=1.6, borderaxespad=0.15)
    else:
        clean_ax(ax_map)
        ax_map.text(0.5, 0.5, "Study area\nand mahalle units", ha="center", va="center", fontsize=12, color=INK)

    ax.text(0.02, 0.98, "(b) Cloud-local data-cube and modelling chain", fontsize=11.2, fontweight="bold", color=INK, ha="left", va="top")
    ax.text(0.02, 0.93, "Aligned x-y layers, annual context, and scale-aware translation to planning units", fontsize=8.5, color=MUTED, ha="left", va="top")

    cube_x, cube_y = 0.05, 0.56
    layer_colors = [PEACH, "#DFC2B7", "#D6B5AE", "#C8AAA9", "#BFA5A7"]
    layer_labels = ["LST 2014-2025", "LCZ + probability", "NDVI / NDBI / EVI", "Albedo + canopy", "Terrain + population"]
    for i, (color, label) in enumerate(zip(layer_colors, layer_labels)):
        dx, dy = i * 0.026, i * 0.026
        rect = Rectangle((cube_x + dx, cube_y + dy), 0.23, 0.13, facecolor=color, edgecolor="white", linewidth=1.0, alpha=0.88, zorder=2 + i)
        rect.set_path_effects([pe.SimplePatchShadow(offset=(1.2, -1.2), alpha=0.10), pe.Normal()])
        ax.add_patch(rect)
    ax.text(cube_x, cube_y + 0.27, "Data-cube logic", fontsize=9.2, fontweight="bold", color=INK, ha="left")
    ax.text(cube_x + 0.35, cube_y + 0.23, "Variables:", fontsize=7.9, fontweight="bold", color=INK, ha="left")
    for j, label in enumerate(layer_labels):
        ax.text(cube_x + 0.35, cube_y + 0.19 - j * 0.035, label, fontsize=7.2, color=MUTED, ha="left", va="top")
    ax.text(cube_x, cube_y - 0.035, "x-y grid | annual slices | variable stack", fontsize=7.4, color=MUTED, ha="left")

    boxes = [
        ("100 m LCZ-aligned\nmodelling grid", 0.54, 0.66, TEAL),
        ("OLS + Moran\nspatial diagnostics", 0.78, 0.66, NAVY),
        ("Sampled MGWR\nlocal coefficients", 0.54, 0.37, PEACH),
        ("Mahalle planning\npriority translation", 0.78, 0.37, ROSE),
    ]
    for text, x, y, color in boxes:
        rect = Rectangle((x, y), 0.18, 0.17, facecolor=color, edgecolor="white", linewidth=1.1, alpha=0.72)
        rect.set_path_effects([pe.SimplePatchShadow(offset=(1.4, -1.4), alpha=0.11), pe.Normal()])
        ax.add_patch(rect)
        ax.text(x + 0.09, y + 0.085, text, ha="center", va="center", fontsize=7.9, color=INK, fontweight="bold")
    for start, end in [
        ((0.46, 0.745), (0.54, 0.745)),
        ((0.72, 0.745), (0.78, 0.745)),
        ((0.63, 0.66), (0.63, 0.53)),
        ((0.72, 0.455), (0.78, 0.455)),
        ((0.86, 0.66), (0.86, 0.53)),
    ]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12, linewidth=0.95, color=MUTED))

    unit_y = 0.16
    unit_steps = [("100 m cells", 0.09, "#BFD9D1"), ("150 m hex\ncommunication", 0.29, "#D8C8B6"), ("mahalle\nstatistics", 0.50, "#D5A19A"), ("district\nsummary", 0.70, "#92A6B7")]
    for label, x, color in unit_steps:
        ax.add_patch(Rectangle((x, unit_y), 0.14, 0.10, facecolor=color, edgecolor="white", linewidth=1.0, alpha=0.86))
        ax.text(x + 0.07, unit_y + 0.05, label, ha="center", va="center", fontsize=7.7, color=INK, fontweight="bold")
    for x in [0.23, 0.44, 0.64]:
        ax.add_patch(FancyArrowPatch((x, unit_y + 0.05), (x + 0.055, unit_y + 0.05), arrowstyle="-|>", mutation_scale=10, linewidth=0.85, color=MUTED))
    ax.text(0.06, 0.30, "Analysis-unit ladder", fontsize=9.5, fontweight="bold", color=INK, ha="left")
    ax.text(0.06, 0.075, "The cube is analogous in logic to space-time cube workflows, but implemented as reproducible GEE-Python rasters rather than an Esri STC object.", fontsize=7.0, color=MUTED, ha="left")
    save_img(fig, FIGURES["fig01"])


def save_study_area_stack() -> None:
    # Slightly taller canvas gives each mini-map more vertical breathing room
    # while preserving a very tight, journal-style multipanel layout.
    fig = plt.figure(figsize=(12.4, 13.0), facecolor=BG)
    fig.subplots_adjust(left=0.004, right=0.996, bottom=0.004, top=0.996)
    gs = fig.add_gridspec(4, 4, hspace=-0.040, wspace=0.004)
    panels = [
        (0, LST, "LST", CMAP_HEAT, "(a)", 28, 56, None, "deg C"),
        (1, UHI, "UHI", CMAP_HEAT, "(b)", None, None, None, "deg C"),
        (2, UTFVI, "UTFVI", CMAP_HEAT, "(c)", None, None, None, "index"),
        (4, LCZ_PROB, "LCZ Probability", CMAP_COOL, "(e)", 0, 100, None, "%"),
        (5, NDVI, "NDVI", CMAP_COOL, "(f)", None, None, None, "index"),
        (6, EVI, "EVI", CMAP_COOL, "(g)", None, None, None, "index"),
        (7, NDBI, "NDBI", CMAP_HEAT, "(h)", None, None, None, "index"),
        (8, ALBEDO, "Albedo", CMAP_GRAY, "(i)", None, None, None, "index"),
        (9, CANOPY, "Tree Canopy Height", CMAP_COOL, "(j)", 0, None, None, "m"),
        (10, SRTM_ELEV, "SRTM Elevation", CMAP_GRAY, "(k)", None, None, None, "m"),
        (11, SLOPE, "SRTM Slope", CMAP_COOL, "(l)", 0, None, None, "degrees"),
        (12, CORINE_2018, "CORINE 2018", CMAP_GRAY, "(m)", None, None, None, "class"),
        (13, GLC_2022, "GLC-FCS 2022", CMAP_GRAY, "(n)", None, None, None, "class"),
        (14, LCZ, "LCZ Classes", LCZ_CMAP, "(o)", None, None, LCZ_NORM, None),
    ]
    axes = [fig.add_subplot(gs[i, j]) for i in range(4) for j in range(4)]
    for idx, path, title, cmap, label, vmin, vmax, norm, cbar_label in panels:
        mini_raster_panel(
            axes[idx],
            path,
            title,
            cmap,
            label,
            vmin,
            vmax,
            norm=norm,
            cbar_label=cbar_label,
            pad_m=6200.0,
            grid_label_outside=False,
            draw_header=False,
        )
    mini_population_panel(axes[3], "(d)", pad_m=6200.0, grid_label_outside=False, draw_header=False)
    mini_lcz_legend_panel(axes[-1], draw_header=True, header_y=0.96)
    panel_titles = {
        0: ("(a)", "LST"),
        1: ("(b)", "UHI"),
        2: ("(c)", "UTFVI"),
        3: ("(d)", "Population"),
        4: ("(e)", "LCZ Probability"),
        5: ("(f)", "NDVI"),
        6: ("(g)", "EVI"),
        7: ("(h)", "NDBI"),
        8: ("(i)", "Albedo"),
        9: ("(j)", "Tree Canopy Height"),
        10: ("(k)", "SRTM Elevation"),
        11: ("(l)", "SRTM Slope"),
        12: ("(m)", "CORINE 2018"),
        13: ("(n)", "GLC-FCS 2022"),
        14: ("(o)", "LCZ Classes"),
    }
    for idx, (label, title) in panel_titles.items():
        draw_figure_level_panel_header(fig, axes[idx], label, title)
    save_img(fig, FIGURES["fig02"])


def fig03_cache_paths() -> tuple[Path, Path]:
    FIGURE_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURE_METADATA_DIR / "fig03_datacube_cache.npz", FIGURE_METADATA_DIR / "fig03_datacube_metadata.json"


def fig03_cache_is_current(cache_path: Path, meta_path: Path) -> bool:
    if not cache_path.exists() or not meta_path.exists():
        return False
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if metadata.get("cache_version") != FIG03_CACHE_VERSION:
        return False
    cache_time = min(cache_path.stat().st_mtime, meta_path.stat().st_mtime)
    source_time = max(LST_CUBE.stat().st_mtime, SRTM_ELEV.stat().st_mtime)
    return cache_time >= source_time


def build_fig03_cache() -> dict[str, np.ndarray]:
    ds = xr.open_dataset(LST_CUBE)
    cube = ds["lst_jja"]
    annual = cube.mean(dim=("y", "x"), skipna=True).to_pandas()
    annual_std = cube.std(dim=("y", "x"), skipna=True).to_pandas()
    slice_years = np.array([2014, 2016, 2018, 2020, 2022, 2024], dtype="int16")
    slice_cube = cube.sel(time=[np.datetime64(f"{year}-07-15") for year in slice_years]).load()
    x_vals = ds["x"].values.astype("float32")
    y_vals = ds["y"].values.astype("float32")
    xx, yy = np.meshgrid(x_vals, y_vals)
    slice_stack = slice_cube.values.astype("float32")
    valid_any = np.isfinite(slice_stack).any(axis=0)
    side = 200.0
    dx = 1.5 * side
    dy = math.sqrt(3.0) * side
    x_centers = np.arange(float(x_vals.min()) - dx, float(x_vals.max()) + dx, dx, dtype="float32")
    center_x_parts = []
    center_y_parts = []
    for col, x_center in enumerate(x_centers):
        y_start = float(y_vals.min()) - dy + (0.5 * dy if col % 2 else 0.0)
        ys = np.arange(y_start, float(y_vals.max()) + dy, dy, dtype="float32")
        center_x_parts.append(np.full(len(ys), x_center, dtype="float32"))
        center_y_parts.append(ys)
    hex_x_all = np.concatenate(center_x_parts)
    hex_y_all = np.concatenate(center_y_parts)
    tree = cKDTree(np.column_stack([hex_x_all, hex_y_all]))
    _, nearest_hex = tree.query(np.column_stack([xx[valid_any], yy[valid_any]]), k=1, workers=-1)
    n_hex_all = len(hex_x_all)
    hex_values_all = np.full((len(slice_years), n_hex_all), np.nan, dtype="float32")
    flat_nearest = nearest_hex.astype("int32")
    for idx, arr in enumerate(slice_stack):
        vals = arr[valid_any]
        finite = np.isfinite(vals)
        counts = np.bincount(flat_nearest[finite], minlength=n_hex_all)
        sums = np.bincount(flat_nearest[finite], weights=vals[finite].astype("float64"), minlength=n_hex_all)
        keep = counts > 0
        hex_values_all[idx, keep] = (sums[keep] / counts[keep]).astype("float32")
    keep_hex = np.isfinite(hex_values_all).any(axis=0)
    hex_x = hex_x_all[keep_hex]
    hex_y = hex_y_all[keep_hex]
    hex_values = hex_values_all[:, keep_hex]
    dem, _, _, _, dem_inside = read_clipped_raster(SRTM_ELEV)
    dem_valid = np.isfinite(dem) & dem_inside
    rng = np.random.default_rng(42)
    dem_sample_pool = dem[dem_valid]
    dem_sample = rng.choice(dem_sample_pool, size=min(60000, len(dem_sample_pool)), replace=False)
    if mapclassify is None:
        raise ModuleNotFoundError("`mapclassify` is required for fig03. Install it or run with `--only` for other figures.")
    dem_bins = mapclassify.NaturalBreaks(dem_sample, k=5).bins
    elev_palette = ["#5F7180", "#8DA1A9", "#C4C9C5", "#D7B8AE", "#A86462"]
    box_data = []
    box_years = []
    scatter_rows = []
    for year in annual.index.year:
        arr = cube.sel(time=np.datetime64(f"{year}-07-15")).values
        valid = np.isfinite(arr) & dem_valid
        vals = arr[valid]
        if len(vals) > 0:
            rng = np.random.default_rng(year)
            flat_idx = np.flatnonzero(valid)
            chosen = rng.choice(flat_idx, size=min(1300, len(flat_idx)), replace=False)
            lst_sample = arr.ravel()[chosen]
            elev_sample = dem.ravel()[chosen]
            elev_class = np.digitize(elev_sample, dem_bins, right=True)
            jitter_sample = rng.normal(0, 0.060, size=len(lst_sample))
            box_data.append(vals[rng.choice(len(vals), size=min(4000, len(vals)), replace=False)])
            box_years.append(year)
            for value, cls, jitter in zip(lst_sample, elev_class, jitter_sample):
                scatter_rows.append((year, value, int(np.clip(cls, 0, 4)), float(jitter)))
    ds.close()
    box_matrix = np.full((len(box_data), max(len(values) for values in box_data)), np.nan, dtype="float32")
    for idx, values in enumerate(box_data):
        box_matrix[idx, : len(values)] = values.astype("float32")
    scatter = np.array(scatter_rows, dtype="float32")
    cache = {
        "annual_years": annual.index.year.to_numpy(dtype="int16"),
        "annual_mean": annual.values.astype("float32"),
        "annual_std": annual_std.values.astype("float32"),
        "slice_years": slice_years,
        "hex_x": hex_x.astype("float32"),
        "hex_y": hex_y.astype("float32"),
        "hex_values": hex_values.astype("float32"),
        "dem_bins": dem_bins.astype("float32"),
        "box_years": np.array(box_years, dtype="int16"),
        "box_matrix": box_matrix,
        "scatter": scatter,
    }
    cache_path, meta_path = fig03_cache_paths()
    np.savez_compressed(cache_path, **cache)
    metadata = {
        "figure": "fig03_cloud_local_lst_datacube",
        "cache_version": FIG03_CACHE_VERSION,
        "created_from": {
            "lst_cube": str(LST_CUBE),
            "srtm_elevation": str(SRTM_ELEV),
        },
        "panel_coding": {
            "a": "Representative annual JJA LST cube slices aggregated to true EPSG:32635 hexagons with 200 m side length in two rows: 2014, 2016, 2018; 2020, 2022, 2024.",
            "b": "Annual LST distributions with jittered points coloured by five Jenks natural-break SRTM elevation classes; spatial mean JJA LST and +/-1 standard deviation are overlaid.",
        },
        "panel_a_hex_side_m": 200.0,
        "dem_jenks_bins_m": [float(value) for value in dem_bins],
        "cache_arrays": list(cache.keys()),
        "purpose": "Allows style-only figure regeneration without re-reading and resampling the full netCDF cube and DEM.",
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return cache


def load_fig03_cache() -> dict[str, np.ndarray]:
    cache_path, meta_path = fig03_cache_paths()
    if not fig03_cache_is_current(cache_path, meta_path):
        return build_fig03_cache()
    with np.load(cache_path) as npz:
        return {key: npz[key] for key in npz.files}


def save_datacube_panel() -> None:
    cache = load_fig03_cache()
    annual_years = cache["annual_years"].astype(int)
    annual_mean = cache["annual_mean"]
    annual_std = cache["annual_std"]
    slice_years = cache["slice_years"].astype(int)
    hex_x = cache["hex_x"]
    hex_y = cache["hex_y"]
    hex_values = cache["hex_values"]
    dem_bins = cache["dem_bins"]
    box_years = cache["box_years"].astype(int).tolist()
    box_matrix = cache["box_matrix"]
    box_data = [row[np.isfinite(row)] for row in box_matrix]
    scatter = cache["scatter"]
    cube_cmap = CMAP_DIV
    elev_palette = ["#5F7180", "#8DA1A9", "#C4C9C5", "#D7B8AE", "#A86462"]
    fig = plt.figure(figsize=(13.4, 11.5), facecolor=BG)
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[7.5, 0.26, 2.9],
        hspace=0.10,
        left=0.055,
        right=0.945,
        top=0.965,
        bottom=0.075,
    )
    gs_maps = gs[0].subgridspec(2, 3, wspace=0.03, hspace=0.06)
    x_pad = 2500.0
    y_pad = 2500.0
    x_limits = (float(np.nanmin(hex_x)) - x_pad, float(np.nanmax(hex_x)) + x_pad)
    y_limits = (float(np.nanmin(hex_y)) - y_pad, float(np.nanmax(hex_y)) + y_pad)
    boundary = boundary_gdf("EPSG:32635")
    map_axes: list[plt.Axes] = []
    for slice_idx, year in enumerate(slice_years):
        r = slice_idx // 3
        c = slice_idx % 3
        iax = fig.add_subplot(gs_maps[r, c])
        iax.set_facecolor(BG)
        values = hex_values[slice_idx]
        keep = np.isfinite(values)
        iax.scatter(
            hex_x[keep],
            hex_y[keep],
            c=values[keep],
            cmap=cube_cmap,
            vmin=25,
            vmax=55,
            marker="h",
            s=7.2,
            linewidth=0,
            edgecolors="none",
            alpha=1.0,
            rasterized=True,
        )
        iax.set_xlim(*x_limits)
        iax.set_ylim(*y_limits)
        iax.set_aspect("equal", adjustable="box")
        iax.set_xticks([])
        iax.set_yticks([])
        for spine in iax.spines.values():
            spine.set_visible(False)
        mask_outside_boundary(iax, boundary)
        draw_boundary_effect(iax, boundary)
        add_map_grid(iax, boundary, interval_deg=0.50, label_step=1, fontsize=9.0, alpha=0.55, x_frac=0.010, y_frac=0.010, label_outside=True, label_offset_pts=2.0)
        add_left_bottom_frame(iax, linewidth=0.55, alpha=0.75)
        iax.text(0.04, 0.92, str(year), transform=iax.transAxes, ha="left", va="top", fontsize=6.6, color=INK, fontweight="bold", bbox=dict(boxstyle="square,pad=0.10", facecolor=BG, edgecolor="none", alpha=0.78))
        map_axes.append(iax)
    fig.text(0.055, 0.985, "(a)", ha="left", va="top", fontsize=13.0, color=INK, fontweight="bold")
    norm = Normalize(25, 55)
    cax = fig.add_subplot(gs[1])
    clean_ax(cax)
    cax.set_axis_off()
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cube_cmap), ax=map_axes, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([])
    cb.ax.tick_params(length=0)
    cax.set_xlabel("LST (deg C)", fontsize=7.0, color=MUTED, labelpad=0.5)
    cb.ax.text(
        0.01,
        0.50,
        f"{norm.vmin:.0f}",
        transform=cb.ax.transAxes,
        ha="left",
        va="center",
        fontsize=6.6,
        color=INK,
        fontweight="bold",
    )
    cb.ax.text(
        0.99,
        0.50,
        f"{norm.vmax:.0f}",
        transform=cb.ax.transAxes,
        ha="right",
        va="center",
        fontsize=6.6,
        color=INK,
        fontweight="bold",
    )
    ax = fig.add_subplot(gs[2])
    chart_ax(ax)
    positions = np.arange(len(box_data))
    bp = ax.boxplot(box_data, positions=positions, widths=0.62, patch_artist=True, showfliers=False, medianprops={"color": BOUNDARY_LINE, "linewidth": 1.0})
    for patch in bp["boxes"]:
        patch.set_facecolor("#D7E3DF")
        patch.set_edgecolor("#7E9295")
        patch.set_alpha(0.86)
    for whisker in bp["whiskers"]:
        whisker.set_color("#7E9295")
    for cap in bp["caps"]:
        cap.set_color("#7E9295")
    year_to_x = {year: idx for idx, year in enumerate(box_years)}
    scatter_years = scatter[:, 0].astype(int)
    scatter_x = np.array([year_to_x[year] for year in scatter_years], dtype="float32") + scatter[:, 3]
    scatter_y = scatter[:, 1]
    scatter_cls = scatter[:, 2].astype(int)
    for cls, color in enumerate(elev_palette):
        keep = scatter_cls == cls
        ax.scatter(scatter_x[keep], scatter_y[keep], s=2.38, color=color, alpha=0.25, linewidth=0, edgecolors="none", zorder=3)
    year_to_pos = {year: idx for idx, year in enumerate(box_years)}
    mean_x = np.array([year_to_pos[year] for year in annual_years if year in year_to_pos], dtype="float32")
    mean_keep = np.array([idx for idx, year in enumerate(annual_years) if year in year_to_pos])
    mean_y = annual_mean[mean_keep]
    mean_sd = annual_std[mean_keep]
    ax.fill_between(mean_x, mean_y - mean_sd, mean_y + mean_sd, color="#D8DDDA", alpha=0.18, linewidth=0, zorder=1)
    ax.plot(mean_x, mean_y, color=NAVY, linewidth=1.85, zorder=5)
    ax.scatter(mean_x, mean_y, color=ROSE, s=18, zorder=6, edgecolor="none", linewidth=0)
    turning_idx = int(np.argmax(np.abs(np.diff(mean_y)))) + 1
    ax.scatter([mean_x[turning_idx]], [mean_y[turning_idx]], s=72, facecolor="none", edgecolor=BOUNDARY_LINE, linewidth=1.05, zorder=7)
    ax.annotate("turning point", xy=(mean_x[turning_idx], mean_y[turning_idx]), xytext=(mean_x[turning_idx] - 1.2, mean_y[turning_idx] + 2.4), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.7), fontsize=6.9, color=MUTED)
    ax.set_ylabel("deg C", fontsize=8.5, color=INK)
    ax.set_xlabel("summer composite year", fontsize=8.0, color=INK)
    tick_idx = np.arange(0, len(box_years), 2)
    ax.set_xticks(tick_idx, [str(box_years[i]) for i in tick_idx])
    handles = []
    lower = 0.0
    for idx, upper in enumerate(dem_bins):
        label = f"{lower:.0f}-{upper:.0f} m" if idx else f"<= {upper:.0f} m"
        handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=elev_palette[idx], markeredgecolor="none", markersize=4.8, label=label, alpha=0.85))
        lower = upper
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.50, -0.002), frameon=False, fontsize=6.4, ncol=5, handletextpad=0.24, columnspacing=0.85, borderpad=0.0, labelcolor=MUTED)
    tag(ax, "(b)")
    save_img(fig, FIGURES["fig03"])


def fig04_cache_paths() -> tuple[Path, Path]:
    FIGURE_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURE_METADATA_DIR / "fig04_lcz_distribution_cache.npz", FIGURE_METADATA_DIR / "fig04_lcz_distribution_metadata.json"


def fig04_cache_is_current(cache_path: Path, meta_path: Path) -> bool:
    return cache_path.exists() and meta_path.exists() and cache_path.stat().st_mtime >= MODEL_GRID.stat().st_mtime


def build_fig04_cache() -> dict[str, np.ndarray]:
    df = pd.read_csv(MODEL_GRID, usecols=["lst_2025", "lcz"]).dropna()
    df = df[df["lcz"].isin(LCZ_CLASSES)].copy()
    grouped = []
    point_rows = []
    violin_rows = []
    summary_rows = []
    for lcz, group in df.groupby("lcz"):
        values = group["lst_2025"].to_numpy(dtype="float32")
        if len(values) < 10:
            continue
        grouped.append(int(lcz))
        summary_rows.append((int(lcz), len(values), float(np.mean(values)), float(np.median(values)), float(np.std(values))))
        rng = np.random.default_rng(int(lcz) * 100)
        point_sample = rng.choice(values, size=min(450, len(values)), replace=False)
        violin_sample = rng.choice(values, size=min(4500, len(values)), replace=False)
        for value in point_sample:
            point_rows.append((int(lcz), float(value), float(rng.normal(0, 0.055))))
        for value in violin_sample:
            violin_rows.append((int(lcz), float(value)))
    summary = pd.DataFrame(summary_rows, columns=["lcz", "count", "mean", "median", "std"]).sort_values("median", ascending=False)
    order = summary["lcz"].to_numpy(dtype="int16")
    cache = {
        "order": order,
        "summary": summary.to_numpy(dtype="float32"),
        "point_rows": np.array(point_rows, dtype="float32"),
        "violin_rows": np.array(violin_rows, dtype="float32"),
    }
    cache_path, meta_path = fig04_cache_paths()
    np.savez_compressed(cache_path, **cache)
    metadata = {
        "figure": "fig04_lcz_thermal_differentiation",
        "created_from": {"model_grid": str(MODEL_GRID)},
        "panel_coding": {
            "a": "2025 summer LST map styled with the Fig02 mini-map cartographic language.",
            "b": "LCZ map styled with the Fig02 mini-map cartographic language.",
            "c": "LCZ classes ordered by median LST; violin distributions, boxplots, and jittered cell samples replace the previous mean bar chart.",
        },
        "lcz_order_by_median_lst": [int(value) for value in order],
        "purpose": "Allows style-only Fig04 regeneration without re-reading the full model grid.",
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return cache


def load_fig04_cache() -> dict[str, np.ndarray]:
    cache_path, meta_path = fig04_cache_paths()
    if not fig04_cache_is_current(cache_path, meta_path):
        return build_fig04_cache()
    with np.load(cache_path) as npz:
        return {key: npz[key] for key in npz.files}


def save_lcz_thermal_differentiation() -> None:
    tests = pd.read_csv(TABLES_DIR / "paper1_lcz_lst_stat_tests.csv")
    cache = load_fig04_cache()
    order = cache["order"].astype(int).tolist()
    point_rows = cache["point_rows"]
    violin_rows = cache["violin_rows"]
    fig = plt.figure(figsize=(12.8, 9.4), facecolor=BG)
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[4.5, 0.45, 4.55],
        left=0.095,
        right=0.890,
        top=0.965,
        bottom=0.075,
        hspace=0.0,
    )
    gs_top = gs[0].subgridspec(1, 3, width_ratios=[4.5, 1.0, 4.5], wspace=0.0)
    ax_l = fig.add_subplot(gs_top[0, 0])
    ax_r = fig.add_subplot(gs_top[0, 2])
    ax_c = fig.add_subplot(gs[2])
    mini_raster_panel(ax_l, LST, "2025 Summer LST", CMAP_HEAT, "(a)", 28, 56, cbar_label="deg C", grid_label_outside=True, grid_label_offset_pts=2.0, shadow_scale=(1 / 2.3))
    ax_lcz = ax_r
    mini_raster_panel(ax_lcz, LCZ, "LCZ Classes", LCZ_CMAP, "(b)", norm=LCZ_NORM, grid_label_outside=True, grid_label_offset_pts=2.0, shadow_scale=(1 / 2.3))
    ax = ax_c
    chart_ax(ax)
    ax.margins(y=0.02)
    data = [violin_rows[violin_rows[:, 0] == lcz, 1] for lcz in order]
    positions = np.arange(len(order))
    parts = ax.violinplot(data, positions=positions, widths=0.86, showmeans=False, showmedians=False, showextrema=False)
    for idx, body in enumerate(parts["bodies"]):
        body.set_facecolor(LCZ_CMAP(LCZ_NORM([order[idx]])[0]))
        body.set_edgecolor("none")
        body.set_alpha(0.34)
    bp = ax.boxplot(data, positions=positions, widths=0.18, patch_artist=True, showfliers=False, medianprops={"color": BOUNDARY_LINE, "linewidth": 1.05})
    for patch in bp["boxes"]:
        patch.set_facecolor("#F4F5F1")
        patch.set_edgecolor("#7E9295")
        patch.set_alpha(0.72)
    for item in [*bp["whiskers"], *bp["caps"]]:
        item.set_color("#8FA0A3")
        item.set_linewidth(0.85)
    lcz_to_x = {lcz: idx for idx, lcz in enumerate(order)}
    for lcz in order:
        keep = point_rows[:, 0].astype(int) == lcz
        xj = lcz_to_x[lcz] + point_rows[keep, 2]
        yj = point_rows[keep, 1]
        ax.scatter(xj, yj, s=2.2, color=LCZ_CMAP(LCZ_NORM([lcz])[0]), alpha=0.18, linewidth=0, edgecolors="none", zorder=3)
    ax.set_xticks(positions, [lcz_axis_label(lcz) for lcz in order], fontsize=6.4, color=INK)
    ax.set_xlim(-0.65, len(order) - 0.35)
    ax.set_ylabel("LST (deg C)", fontsize=8.5, color=INK)
    omnibus = tests[tests["test"] == "kruskal_wallis"].iloc[0]
    ax.text(0.99, 0.96, f"Kruskal-Wallis p < 0.001; eta2 approx. {omnibus['eta_squared_approx']:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=8.1, color=MUTED)
    tag(ax, "(c)")
    save_img(fig, FIGURES["fig04"])


def save_bivariate_heat_vegetation() -> None:
    hexes = build_environment_hex()
    hexes["veg_deficit"] = -hexes["ndvi_mean"]
    bivar = add_bivariate_classes(hexes, "veg_deficit", "lst_mean")
    boundary = boundary_gdf()
    # Note: build_environment_hex() keeps only hexes with >=1 sampled point.
    # Large "white" areas in the map are therefore genuinely "no data" (no sampled cells),
    # not an extra bivariate colour. Draw a faint full hex outline to make this explicit.
    full_hex = gpd.clip(build_hex_grid(boundary.total_bounds, 150.0), boundary)
    counts = pd.crosstab(bivar["y_q"], bivar["x_q"]).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0).to_numpy()
    rho = bivar[["lst_mean", "ndvi_mean"]].corr(method="spearman").iloc[0, 1]
    pearson = bivar[["lst_mean", "ndvi_mean"]].corr(method="pearson").iloc[0, 1]
    slope, intercept = np.polyfit(bivar["ndvi_mean"], bivar["lst_mean"], 1)
    fitted = intercept + slope * bivar["ndvi_mean"]
    r2 = 1.0 - float(np.sum((bivar["lst_mean"] - fitted) ** 2) / np.sum((bivar["lst_mean"] - bivar["lst_mean"].mean()) ** 2))
    low_veg_hot = bivar[(bivar["x_q"] == 2) & (bivar["y_q"] == 2)]
    high_veg_cool = bivar[(bivar["x_q"] == 0) & (bivar["y_q"] == 0)]
    thermal_gap = float(low_veg_hot["lst_mean"].median() - high_veg_cool["lst_mean"].median())
    high_risk_share = 100.0 * len(low_veg_hot) / len(bivar)
    lst_breaks = bivar.groupby("y_q")["lst_mean"].agg(["min", "max"]).round(1)
    ndvi_breaks = bivar.groupby("x_q")["ndvi_mean"].agg(["min", "max"]).round(3)
    fig, ax = plt.subplots(figsize=(9.2, 9.4), facecolor=BG)
    fig.subplots_adjust(left=0.015, right=0.995, bottom=0.018, top=0.992)
    clean_ax(ax)
    # Hillshade removed as requested
    full_hex.boundary.plot(ax=ax, color="#E9EFEE", linewidth=0.10, alpha=0.80, zorder=1.5)
    bivar.plot(ax=ax, color=bivar["bivar_color"], edgecolor="#FBFBF8", linewidth=0.014, alpha=0.88, zorder=2)
    low_veg_hot.plot(ax=ax, facecolor="none", edgecolor=BOUNDARY_LINE, linewidth=0.030, alpha=0.38, zorder=3)
    finish_map(
        ax,
        boundary,
        None,
        furniture=False,
        grid_label_fontsize=8.2,
        grid_label_alpha=0.80,
        grid_label_outside=True,
        grid_label_offset_pts=2.0,
        frame_left_bottom=True,
        shadow_scale=(1 / 3),
    )
    draw_bivariate_legend(ax, "vegetation deficit", "heat", counts=counts, bounds=[0.720, 0.748, 0.190, 0.190])
    inset = ax.inset_axes([0.705, 0.485, 0.270, 0.205])
    inset.set_facecolor(BG)
    inset.grid(color=GRID, linewidth=0.65, alpha=0.85)
    inset.tick_params(colors=INK, labelsize=5.8, length=0, pad=1.0)
    for side in ["top", "right"]:
        inset.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        inset.spines[side].set_color(GRID)
    sample = bivar[["ndvi_mean", "lst_mean"]].dropna()
    if len(sample) > 4000:
        sample = sample.sample(4000, random_state=12)
    inset.scatter(sample["ndvi_mean"], sample["lst_mean"], s=3.2, color=INK, alpha=0.09, linewidth=0, zorder=2)
    x_line = np.linspace(float(bivar["ndvi_mean"].quantile(0.02)), float(bivar["ndvi_mean"].quantile(0.98)), 50)
    fit_color = CMAP_COOL(0.80)
    inset.plot(x_line, intercept + slope * x_line, color=fit_color, linewidth=1.45, alpha=0.95, zorder=4)
    inset.set_xlabel("NDVI", fontsize=6.2, color=INK, labelpad=1.0)
    inset.set_ylabel("LST", fontsize=6.2, color=INK, labelpad=1.0)
    inset.text(
        0.02,
        0.98,
        f"n={len(bivar):,}  rho={rho:.2f}\n"
        f"slope={slope/10:.2f} °C per +0.10 NDVI\n"
        f"high-risk={high_risk_share:.1f}%  gap={thermal_gap:.1f} °C",
        transform=inset.transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
        color=INK,
        linespacing=1.18,
        bbox=dict(boxstyle="square,pad=0.20", facecolor=BG, edgecolor="none", alpha=0.82),
        zorder=10,
    )
    class_text = (
        "tercile ranges, low deficit -> high deficit\n"
        f"LST: {lst_breaks.loc[0, 'min']:.1f}-{lst_breaks.loc[0, 'max']:.1f} | "
        f"{lst_breaks.loc[1, 'min']:.1f}-{lst_breaks.loc[1, 'max']:.1f} | "
        f"{lst_breaks.loc[2, 'min']:.1f}-{lst_breaks.loc[2, 'max']:.1f} deg C\n"
        f"NDVI: {ndvi_breaks.loc[0, 'min']:.2f}-{ndvi_breaks.loc[0, 'max']:.2f} | "
        f"{ndvi_breaks.loc[1, 'min']:.2f}-{ndvi_breaks.loc[1, 'max']:.2f} | "
        f"{ndvi_breaks.loc[2, 'min']:.2f}-{ndvi_breaks.loc[2, 'max']:.2f}"
    )
    ax.text(
        0.035,
        0.038,
        class_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.7,
        color=MUTED,
        linespacing=1.24,
        zorder=30,
        bbox=dict(boxstyle="square,pad=0.22", facecolor=BG, edgecolor="none", alpha=0.78),
    )
    save_img(fig, FIGURES["fig05"])


def save_baseline_diagnostics() -> None:
    df = pd.read_csv(
        MODEL_GRID,
        usecols=[
            "lst_2025",
            "ndvi_2025",
            "ndbi_2025",
            "albedo_2025",
            "evi_2025",
            "tree_canopy_height_2025",
            "srtm_elevation_m",
            "srtm_slope_deg",
        ],
    ).dropna()
    sample = df.sample(min(50000, len(df)), random_state=42)
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.8), facecolor=BG)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.070, top=0.965, hspace=0.123, wspace=0.113)
    ax = axes[0, 0]
    chart_ax(ax)
    response_cmap = LinearSegmentedColormap.from_list("response_soft", ["#F6F7F3", "#DCE7E4", "#AABBC1", "#D8B7AD", "#B97872"])
    hb = ax.hexbin(
        sample["ndvi_2025"],
        sample["ndbi_2025"],
        C=sample["lst_2025"],
        reduce_C_function=np.nanmedian,
        gridsize=54,
        mincnt=8,
        cmap=response_cmap,
        linewidths=0,
        alpha=0.96,
    )
    cb = fig.colorbar(hb, ax=ax, shrink=0.78, pad=0.012)
    cb.set_label("median LST (deg C)", fontsize=7.4, color=INK)
    cb.ax.tick_params(labelsize=6.7, colors=INK, length=0)
    cb.outline.set_visible(False)
    ndvi_rho = sample[["lst_2025", "ndvi_2025"]].corr(method="spearman").iloc[0, 1]
    ndbi_rho = sample[["lst_2025", "ndbi_2025"]].corr(method="spearman").iloc[0, 1]
    ndvi_q = np.quantile(sample["ndvi_2025"], [0.10, 0.50, 0.90])
    ndbi_q = np.quantile(sample["ndbi_2025"], [0.10, 0.50, 0.90])
    ax.axvline(ndvi_q[1], color="#FFFFFF", linewidth=0.85, alpha=0.75, zorder=4)
    ax.axhline(ndbi_q[1], color="#FFFFFF", linewidth=0.85, alpha=0.75, zorder=4)
    ndvi_bins = np.linspace(sample["ndvi_2025"].quantile(0.02), sample["ndvi_2025"].quantile(0.98), 28)
    line_rows = []
    for lo, hi in zip(ndvi_bins[:-1], ndvi_bins[1:]):
        part = sample[(sample["ndvi_2025"] >= lo) & (sample["ndvi_2025"] < hi)]
        if len(part) >= 60:
            line_rows.append((part["ndvi_2025"].median(), part["ndbi_2025"].median()))
    line = np.array(line_rows)
    if len(line) > 1:
        ax.plot(line[:, 0], line[:, 1], color=INK, linewidth=1.15, alpha=0.88, zorder=6)
        ax.scatter(line[:, 0], line[:, 1], s=8, color=INK, alpha=0.55, linewidth=0, zorder=7)
    ax.set_xlabel("NDVI, greenness", fontsize=8.5, color=INK)
    ax.set_ylabel("NDBI, built-up intensity", fontsize=8.5, color=INK)
    ax.text(
        0.025,
        0.040,
        f"Spearman rho: NDVI-LST {ndvi_rho:.2f}; NDBI-LST {ndbi_rho:.2f}\nwhite lines: medians; dark line: median NDBI by NDVI bin",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.5,
        color=INK,
        bbox=dict(boxstyle="square,pad=0.25", facecolor=BG, edgecolor="none", alpha=0.78),
    )
    tag(ax, "(a)")
    ax = axes[0, 1]
    chart_ax(ax)
    coefs = pd.read_csv(TABLES_DIR / "paper1_ols_coefficients.csv")
    terms = ["ndvi_2025", "ndbi_2025", "albedo_2025", "tree_canopy_height_2025", "srtm_elevation_m", "srtm_slope_deg"]
    physical = coefs[(coefs["model"] == "physical_environmental") & (coefs["term"].isin(terms))].copy()
    label_map = {
        "ndvi_2025": "NDVI",
        "ndbi_2025": "NDBI",
        "albedo_2025": "Albedo",
        "tree_canopy_height_2025": "Canopy height",
        "srtm_elevation_m": "Elevation",
        "srtm_slope_deg": "Slope",
    }
    physical["label"] = physical["term"].map(label_map)
    physical = physical.sort_values("coef")
    y = np.arange(len(physical))
    colors = [NAVY if value < 0 else ROSE for value in physical["coef"]]
    ax.barh(y, physical["coef"], color=colors, alpha=0.70, height=0.58, edgecolor=BG, linewidth=0.45)
    xerr = np.vstack([physical["coef"] - physical["ci_low"], physical["ci_high"] - physical["coef"]])
    ax.errorbar(physical["coef"], y, xerr=xerr, fmt="none", ecolor=INK, elinewidth=0.75, capsize=2.2, capthick=0.75, alpha=0.78, zorder=5)
    rng = np.random.default_rng(106)
    for row_idx, (_, row) in enumerate(physical.iterrows()):
        ci_half = 0.5 * (float(row["ci_high"]) - float(row["ci_low"]))
        sigma = ci_half / 1.96 if ci_half > 0 else 0.0
        if sigma <= 0:
            continue
        draws = rng.normal(loc=float(row["coef"]), scale=sigma, size=140)
        yj = row_idx + rng.uniform(-0.18, 0.18, size=draws.size)
        ax.scatter(draws, yj, s=7.5, color="#0E2E33", alpha=0.10, linewidth=0, zorder=4)
    ax.axvline(0, color=INK, linewidth=0.85)
    ax.set_yticks(y, physical["label"], fontsize=7.6, color=INK)
    ax.set_xlabel("standardized coefficient", fontsize=8.5, color=INK)
    x_left = float(min(physical["ci_low"].min() - 0.18, -2.45))
    x_right = float(max(physical["ci_high"].max() + 0.26, 1.55))
    ax.set_xlim(x_left, x_right)
    for row_idx, (_, row) in enumerate(physical.iterrows()):
        coef = float(row["coef"])
        pad = 0.06
        if coef >= 0:
            x_text = min(coef + pad, x_right - 0.02)
            ha = "left"
        else:
            x_text = max(coef - pad, x_left + 0.02)
            ha = "right"
        ax.text(x_text, row_idx, f"{coef:.2f}", ha=ha, va="center", fontsize=6.6, color=INK)
    tag(ax, "(b)")
    ax = axes[1, 0]
    clean_ax(ax)
    corr_cols = ["lst_2025", "ndvi_2025", "ndbi_2025", "albedo_2025", "evi_2025", "tree_canopy_height_2025", "srtm_elevation_m", "srtm_slope_deg"]
    corr = df[corr_cols].corr(method="spearman")
    short = ["LST", "NDVI", "NDBI", "Alb.", "EVI", "Can.", "Elev.", "Slope"]
    n_vars = len(corr_cols)
    ax.set_xlim(-0.5, n_vars - 0.5)
    ax.set_ylim(n_vars - 0.5, -0.5)
    ax.set_xticks(range(n_vars), short, rotation=45, ha="right", fontsize=6.8, color=INK)
    ax.set_yticks(range(n_vars), short, fontsize=6.8, color=INK)
    for i in range(n_vars):
        for j in range(n_vars):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=BG, edgecolor="#FFFFFF", linewidth=0.9, zorder=0))
    norm = Normalize(-1, 1)
    for i in range(n_vars):
        for j in range(i):
            value = corr.iloc[i, j]
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=CMAP_DIV(norm(value)), edgecolor="#FFFFFF", linewidth=0.9, alpha=0.92, zorder=1))
            text_color = "#FFFFFF" if abs(value) > 0.62 else INK
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.4, color=text_color, zorder=2)
    for i in range(n_vars):
        values = df[corr_cols[i]].sample(min(25000, len(df)), random_state=100 + i).to_numpy()
        lo, hi = np.nanpercentile(values, [1, 99])
        values = values[(values >= lo) & (values <= hi)]
        xs = np.linspace(lo, hi, 120)
        kde = gaussian_kde(values)(xs)
        kde = kde / kde.max() * 0.34
        x_scaled = (xs - lo) / (hi - lo) * 0.76 + i - 0.38
        y_base = np.full_like(x_scaled, i + 0.26)
        ax.fill_between(x_scaled, y_base, y_base - kde, color="#D9E3DF", alpha=0.92, linewidth=0, zorder=2)
        ax.plot(x_scaled, y_base - kde, color="#7E9295", linewidth=0.70, alpha=0.85, zorder=3)
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            value = corr.iloc[i, j]
            size = 860 * abs(value) + 38
            ax.scatter(j, i, s=size, color=CMAP_DIV(norm(value)), edgecolor=BG, linewidth=0.65, alpha=0.93, zorder=4)
            if abs(value) >= 0.45:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.4, color="#FFFFFF" if abs(value) > 0.68 else INK, zorder=5)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP_DIV)
    cb = ax.figure.colorbar(sm, ax=ax, shrink=0.62, pad=0.012)
    cb.set_label("Spearman rho", fontsize=7.2, color=INK)
    cb.ax.tick_params(labelsize=6.5, colors=INK, length=0)
    cb.outline.set_visible(False)
    ax.grid(False)
    tag(ax, "(c)")
    ax = axes[1, 1]
    chart_ax(ax)
    ols = pd.read_csv(TABLES_DIR / "manuscript" / "table5_ols_summary_vif.csv")
    model_labels = ["Physical", "Heat justice"]
    x = np.arange(len(ols), dtype=float)
    ax.bar(x - 0.16, ols["adj_r2"], width=0.30, color=NAVY, alpha=0.68, label="Adjusted R2")
    ax.bar(x + 0.16, ols["max_vif"] / 10, width=0.30, color=ROSE, alpha=0.58, label="Max VIF / 10")
    ax.axhline(1.0, color=BOUNDARY_LINE, linewidth=0.65, linestyle=(0, (3, 3)), alpha=0.65)
    ax.set_xticks(x, model_labels, fontsize=8.2, color=INK)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("diagnostic value (VIF scaled /10)", fontsize=8.5, color=INK)
    for idx, row in ols.iterrows():
        ax.text(idx - 0.16, row["adj_r2"] + 0.025, f"{row['adj_r2']:.3f}", ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.text(idx + 0.16, row["max_vif"] / 10 + 0.025, f"VIF {row['max_vif']:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.text(idx, 0.055, f"n={int(row['n']):,}\np={int(row['predictor_count'])}\nVIF>=10: {int(row['terms_vif_ge_10'])}", ha="center", va="bottom", fontsize=6.2, color=MUTED, linespacing=1.15)
    ax.legend(frameon=False, fontsize=7.2, loc="upper left", bbox_to_anchor=(0.12, 0.90), ncol=1, handlelength=1.4)
    tag(ax, "(d)")
    save_img(fig, FIGURES["fig06"])


def parse_only_args(argv: list[str]) -> set[str] | None:
    if "--only" not in argv:
        return None
    idx = argv.index("--only")
    values: list[str] = []
    for item in argv[idx + 1 :]:
        if item.startswith("--"):
            break
        values.append(item)
    if not values:
        raise SystemExit("`--only` requires at least one figure key, e.g. `--only fig06`.")
    return {v.strip().lower() for v in values if v.strip()}


def save_hex_coefficient_atlas() -> None:
    hexes = aggregate_mgwr_to_hex()
    boundary = boundary_gdf()
    dominant_map = {
        "mgwr_coef_ndvi_2025": "NDVI-dominant",
        "mgwr_coef_ndbi_2025": "NDBI-dominant",
        "mgwr_coef_albedo_2025": "Albedo-dominant",
    }
    coef_stack = hexes[list(dominant_map)].abs()
    dominant_col = coef_stack.idxmax(axis=1)
    hexes["dominant_driver"] = dominant_col.map(dominant_map)
    cols = [
        ("mgwr_coef_ndvi_2025", "coef"),
        ("mgwr_coef_ndbi_2025", "coef"),
        ("mgwr_coef_albedo_2025", "coef"),
        ("dominant_driver", "category"),
    ]
    coef_cols = [c for c, kind in cols if kind == "coef"]
    shared_vmax = float(
        max(
            max(abs(hexes[col].quantile(0.05)), abs(hexes[col].quantile(0.95)))
            for col in coef_cols
            if col in hexes.columns
        )
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.7, 9.4), facecolor=BG)
    fig.subplots_adjust(left=0.030, right=0.992, bottom=0.070, top=0.985, hspace=0.035, wspace=0.055)
    words = [
        "Local NDVI coefficient",
        "Local NDBI coefficient",
        "Local albedo coefficient",
        "Dominant local driver",
    ]
    category_order = ["NDVI-dominant", "NDBI-dominant", "Albedo-dominant"]
    category_cmap = ListedColormap([TEAL, ROSE, "#B89A6A"])
    category_norm = BoundaryNorm(np.arange(-0.5, len(category_order) + 0.5, 1), category_cmap.N)
    hexes["dominant_driver_code"] = pd.Categorical(hexes["dominant_driver"], categories=category_order, ordered=True).codes
    for idx, (ax, (col, kind), label, word) in enumerate(zip(np.atleast_1d(axes).ravel(), cols, ["(a)", "(b)", "(c)", "(d)"], words, strict=False)):
        clean_ax(ax)
        if kind == "coef":
            hexes.plot(
                ax=ax,
                column=col,
                cmap=CMAP_DIV,
                norm=TwoSlopeNorm(vmin=-shared_vmax, vcenter=0, vmax=shared_vmax),
                edgecolor="none",
                linewidth=0,
                legend=True,
                legend_kwds={"shrink": 0.62, "pad": 0.01, "label": "coefficient"},
            )
        else:
            hexes.plot(
                ax=ax,
                column="dominant_driver_code",
                cmap=category_cmap,
                norm=category_norm,
                edgecolor="none",
                linewidth=0,
                legend=False,
            )
        # Hillshade removed as requested
        finish_map(
            ax,
            boundary,
            label,
            grid=True,
            furniture=False,
            grid_label_fontsize=8.2,
            grid_label_alpha=0.80,
            grid_x_frac=0.030,
            grid_y_frac=0.024,
            grid_label_outside=True,
            grid_label_offset_pts=2.0,
            frame_left_bottom=True,
        )
        if kind == "category":
            handles = [
                Patch(facecolor=category_cmap(category_norm([code])[0]), edgecolor="none", label=cat)
                for code, cat in enumerate(category_order)
            ]
            ax.legend(
                handles=handles,
                loc="lower left",
                bbox_to_anchor=(0.015, 0.015),
                frameon=True,
                framealpha=0.94,
                facecolor=BG,
                edgecolor="none",
                fontsize=7.0,
                labelcolor=INK,
                borderpad=0.35,
                handlelength=1.0,
                handletextpad=0.40,
            )
        ax.text(
            0.985,
            0.032,
            word,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.2,
            fontweight="bold",
            color=MUTED,
            zorder=60,
            bbox=dict(boxstyle="square,pad=0.12", facecolor=BG, edgecolor="none", alpha=0.78),
        )
    save_img(fig, FIGURES["fig07"])


def save_heat_justice_synthesis() -> None:
    boundary = boundary_gdf()
    base_hex = build_hex_grid(boundary.total_bounds, 150.0)
    base_hex = base_hex[base_hex.intersects(boundary.geometry.union_all())].copy()

    social = build_social_hex()
    social = add_bivariate_classes(social, "socio_sensitivity", "lst_mean")
    social["log_pop_sum"] = np.log1p(social["pop_sum"].clip(lower=0))
    fig = plt.figure(figsize=(12.0, 9.0), facecolor=BG)
    gs = fig.add_gridspec(2, 2, left=0.060, right=0.985, bottom=0.075, top=0.965, hspace=0.20, wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    clean_ax(ax)
    # Hillshade removed as requested
    social_plot = social
    base_hex.plot(ax=ax, color="#E9EEEC", edgecolor="none", linewidth=0, alpha=0.95, zorder=1)
    social_plot.plot(
        ax=ax,
        color=social_plot["bivar_color"],
        edgecolor="none",
        linewidth=0,
        alpha=1.0,
        zorder=2,
    )
    finish_map(
        ax,
        boundary,
        "(a)",
        furniture=False,
        grid_label_fontsize=8.2,
        grid_label_alpha=0.80,
        grid_x_frac=0.030,
        grid_y_frac=0.024,
        grid_label_outside=True,
        grid_label_offset_pts=2.0,
        frame_left_bottom=True,
        aspect_adjustable="datalim",
    )
    # Panel (a): shift x-limits so the map sits left within the fixed CRS grid box (remove extra right-side whitespace).
    x0, x1 = ax.get_xlim()
    b0, _, b2, _ = boundary.total_bounds
    right_margin = x1 - b2
    left_margin = b0 - x0
    if right_margin > left_margin + 1:
        dx = right_margin - left_margin
        ax.set_xlim(x0 - dx, x1 - dx)
    draw_bivariate_legend(ax, "sensitivity", "heat", bounds=[0.60, 0.515, 0.220, 0.220])
    ax = fig.add_subplot(gs[0, 1])
    clean_ax(ax)
    # Hillshade removed as requested
    pop_cmap = LinearSegmentedColormap.from_list("pop_teal_coral", ["#2AA6A1", "#E9F1F0", "#F08A78"])
    pop_norm = Normalize(float(social_plot["log_pop_sum"].quantile(0.02)), float(social_plot["log_pop_sum"].quantile(0.98)))
    base_hex.plot(ax=ax, color="#E9EEEC", edgecolor="none", linewidth=0, alpha=0.95, zorder=1)
    social_plot.plot(
        ax=ax,
        column="log_pop_sum",
        cmap=pop_cmap,
        norm=pop_norm,
        edgecolor="none",
        linewidth=0,
        alpha=1.0,
        zorder=2,
        legend=True,
        legend_kwds={"shrink": 0.62, "pad": 0.01, "label": "log(1 + population)"},
    )
    finish_map(
        ax,
        boundary,
        "(b)",
        furniture=False,
        grid_label_fontsize=8.2,
        grid_label_alpha=0.80,
        grid_x_frac=0.030,
        grid_y_frac=0.024,
        grid_label_outside=True,
        grid_label_offset_pts=0.0,
        grid_outside_y_use_bounds=True,
        grid_outside_y_anchor_axes=True,
        grid_outside_y_axes_x=0.006,
        grid_outside_y_axes_ha="left",
        frame_left_bottom=True,
        aspect_adjustable="datalim",
    )
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
    ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.015, 1.0))
    tag(ax, "(c)")
    ax = fig.add_subplot(gs[1, 1])
    chart_ax(ax)
    draw_mahalle_moran_panel(ax)
    save_img(fig, FIGURES["fig08"])


def save_hexagonal_district_cartogram() -> None:
    pts = model_points(["lst_2025"]).dropna(subset=["lst_2025"])
    mahalle = gpd.read_file(MAHALLE).to_crs("EPSG:32635")
    mahalle = mahalle[mahalle["pop_total"].fillna(0) > 0].copy()
    joined = gpd.sjoin(pts, mahalle[["mahalle_name", "district_name", "pop_total", "geometry"]], predicate="within", how="inner")
    summary = joined.groupby(["mahalle_name", "district_name"]).agg(lst_mean=("lst_2025", "mean"), n=("lst_2025", "size")).reset_index()
    carto = mahalle.merge(summary, on=["mahalle_name", "district_name"], how="inner")
    rep = carto.geometry.representative_point()
    carto["cx"] = rep.x
    carto["cy"] = rep.y
    carto["radius"] = 260 + 1900 * np.sqrt(carto["pop_total"] / carto["pop_total"].max())
    fig, ax = plt.subplots(figsize=(9.0, 9.4), facecolor=BG)
    clean_ax(ax, "Neighborhood Population Cartogram")
    boundary = boundary_gdf()
    boundary.boundary.plot(ax=ax, color=GRID, linewidth=0.55, zorder=1)
    norm = Normalize(carto["lst_mean"].quantile(0.03), carto["lst_mean"].quantile(0.97))
    for _, row in carto.sort_values("radius", ascending=False).iterrows():
        circ = Circle((row["cx"], row["cy"]), row["radius"], facecolor=CMAP_HEAT(norm(row["lst_mean"])), edgecolor="#FFFFFF", linewidth=0.18, alpha=0.88, zorder=2)
        ax.add_patch(circ)
    finish_map(
        ax,
        boundary,
        grid=True,
        furniture=False,
        grid_label_fontsize=8.2,
        grid_label_alpha=0.80,
        grid_x_frac=0.030,
        grid_y_frac=0.024,
        grid_label_outside=True,
        grid_label_offset_pts=2.0,
        frame_left_bottom=True,
    )
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=CMAP_HEAT)
    cb = fig.colorbar(mappable, ax=ax, shrink=0.62, pad=0.01)
    cb.set_label("Mean LST (deg C)", fontsize=8, color=INK)
    cb.outline.set_visible(False)
    ax.text(0.02, 0.035, "Circle area is proportional to neighborhood population; geography is intentionally distorted.", transform=ax.transAxes, fontsize=7.2, color=MUTED, ha="left", va="bottom")
    tag(ax, "(a)")
    save_img(fig, APPENDIX["appendix_fig01"])


def save_terrain_appendix() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.6), facecolor=BG)
    raster_panel(
        axes[0],
        SRTM_ELEV,
        "SRTM Elevation",
        CMAP_GRAY,
        "(a)",
        cbar_label="m",
        furniture=False,
        grid_label_fontsize=8.2,
        grid_label_alpha=0.80,
        grid_x_frac=0.030,
        grid_y_frac=0.024,
        grid_label_outside=True,
        grid_label_offset_pts=2.0,
        frame_left_bottom=True,
    )
    raster_panel(
        axes[1],
        SLOPE,
        "SRTM Slope",
        CMAP_COOL,
        "(b)",
        0,
        None,
        cbar_label="degrees",
        furniture=False,
        grid_label_fontsize=8.2,
        grid_label_alpha=0.80,
        grid_x_frac=0.030,
        grid_y_frac=0.024,
        grid_label_outside=True,
        grid_label_offset_pts=2.0,
        frame_left_bottom=True,
    )
    sin_arr, extent, _, crs, inside = read_clipped_raster(ASPECT_SIN)
    cos_arr, _, _, _, _ = read_clipped_raster(ASPECT_COS)
    aspect = (np.degrees(np.arctan2(sin_arr, cos_arr)) + 360) % 360
    aspect[~inside] = np.nan
    boundary = boundary_gdf(crs)
    clean_ax(axes[2], "SRTM Aspect")
    shade, shade_extent, _ = get_hillshade()
    axes[2].imshow(np.ma.masked_invalid(shade), extent=shade_extent, cmap="Greys", vmin=0, vmax=1, alpha=0.34, interpolation="bilinear", zorder=1)
    im = axes[2].imshow(np.ma.masked_invalid(aspect), extent=extent, cmap="twilight_shifted", vmin=0, vmax=360, alpha=0.78, interpolation="nearest", zorder=2)
    finish_map(
        axes[2],
        boundary,
        "(c)",
        furniture=False,
        grid_label_fontsize=8.2,
        grid_label_alpha=0.80,
        grid_x_frac=0.030,
        grid_y_frac=0.024,
        grid_label_outside=True,
        grid_label_offset_pts=2.0,
        frame_left_bottom=True,
    )
    cb = fig.colorbar(im, ax=axes[2], shrink=0.68, pad=0.012)
    cb.set_label("azimuth degrees", fontsize=8, color=INK)
    cb.outline.set_visible(False)
    save_img(fig, APPENDIX["appendix_fig02"])


def save_hex_local_r2() -> None:
    hexes = aggregate_gwr_to_hex()
    boundary = boundary_gdf()
    fig, ax = plt.subplots(figsize=(8.2, 9.0), facecolor=BG)
    clean_ax(ax, "Exploratory Local Fit Proxy (R²) on 150 m Hexagons")
    shade, shade_extent, _ = get_hillshade()
    ax.imshow(np.ma.masked_invalid(shade), extent=shade_extent, cmap="Greys", vmin=0, vmax=1, alpha=0.22, interpolation="bilinear", zorder=1)
    hexes.plot(ax=ax, column="gwr_local_r2", cmap=CMAP_COOL, edgecolor="#FBFBF8", linewidth=0.02, legend=True, legend_kwds={"shrink": 0.66, "pad": 0.01, "label": "local fit proxy (R²)"}, zorder=2)
    finish_map(
        ax,
        boundary,
        "(a)",
        furniture=False,
        grid_label_fontsize=8.2,
        grid_label_alpha=0.80,
        grid_x_frac=0.030,
        grid_y_frac=0.024,
        grid_label_outside=True,
        grid_label_offset_pts=2.0,
        frame_left_bottom=True,
    )
    save_img(fig, APPENDIX["appendix_fig03"])


def save_srtm_3d_drape() -> None:
    lst, extent, _, crs, inside = read_clipped_raster(LST)
    dem, _, _, _, _ = read_clipped_raster(SRTM_ELEV)
    shade, shade_extent, _ = get_hillshade()
    boundary = boundary_gdf(crs)
    norm = Normalize(28, 56)
    rgb = CMAP_HEAT(norm(lst))[..., :3]
    relief = LightSource(azdeg=315, altdeg=45).shade_rgb(rgb, np.nan_to_num(dem, nan=np.nanmedian(dem)), vert_exag=0.8, dx=100, dy=100)
    relief[~inside] = np.nan
    fig, ax = plt.subplots(figsize=(8.6, 9.2), facecolor=BG)
    clean_ax(ax, "Relief-Shaded LST Drape on SRTM Terrain")
    ax.imshow(np.ma.masked_invalid(shade), extent=shade_extent, cmap="Greys", vmin=0, vmax=1, alpha=0.22, interpolation="bilinear", zorder=1)
    ax.imshow(np.ma.masked_invalid(relief), extent=extent, interpolation="bilinear", alpha=0.96, zorder=2)
    levels = np.nanpercentile(dem, [20, 40, 60, 80])
    rows, cols = dem.shape
    xs = np.linspace(extent[0], extent[1], cols)
    ys = np.linspace(extent[3], extent[2], rows)
    ax.contour(xs, ys, dem, levels=levels, colors="#FFFFFF", linewidths=0.28, alpha=0.45, zorder=3)
    finish_map(
        ax,
        boundary,
        "(a)",
        furniture=False,
        grid_label_fontsize=8.2,
        grid_label_alpha=0.80,
        grid_x_frac=0.030,
        grid_y_frac=0.024,
        grid_label_outside=True,
        grid_label_offset_pts=2.0,
        frame_left_bottom=True,
    )
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=CMAP_HEAT)
    cb = fig.colorbar(mappable, ax=ax, shrink=0.66, pad=0.01)
    cb.set_label("LST (deg C)", fontsize=8, color=INK)
    cb.outline.set_visible(False)
    save_img(fig, APPENDIX["appendix_fig04"])


def save_correlation_appendix() -> None:
    df = pd.read_csv(
        MODEL_GRID,
        usecols=[
            "lst_2025",
            "ndvi_2025",
            "ndbi_2025",
            "albedo_2025",
            "evi_2025",
            "tree_canopy_height_2025",
            "srtm_elevation_m",
            "srtm_slope_deg",
            "female_share",
            "child_0_14_share",
            "elderly_65_plus_share",
        ],
    ).dropna()
    corr = df.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8.4, 7.5), facecolor=BG)
    clean_ax(ax, "Full Spearman Correlation Matrix")
    im = ax.imshow(corr, cmap=CMAP_DIV, vmin=-1, vmax=1)
    labels = ["LST", "NDVI", "NDBI", "Albedo", "EVI", "Canopy", "Elev.", "Slope", "Female", "Child", "Elderly"]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7.4, color=INK)
    ax.set_yticks(range(len(labels)), labels, fontsize=7.4, color=INK)
    fig.colorbar(im, ax=ax, shrink=0.72, pad=0.012, label="Spearman rho")
    tag(ax, "(a)")
    save_img(fig, APPENDIX["appendix_fig05"])


def write_null_qa_report() -> None:
    rows = []
    raster_layers = [
        ("LST", LST),
        ("UHI", UHI),
        ("UTFVI", UTFVI),
        ("LCZ", LCZ),
        ("LCZ probability", LCZ_PROB),
        ("NDVI", NDVI),
        ("EVI", EVI),
        ("NDBI", NDBI),
        ("Albedo", ALBEDO),
        ("Canopy height", CANOPY),
        ("SRTM elevation", SRTM_ELEV),
        ("SRTM slope", SLOPE),
        ("CORINE 2018", CORINE_2018),
        ("GLC-FCS 2022", GLC_2022),
    ]
    for name, path in raster_layers:
        with rasterio.open(path) as src:
            arr = src.read(1).astype("float32")
            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan
            arr[arr <= -9990] = np.nan
            boundary = boundary_gdf(src.crs.to_string())
            inside = geometry_mask(boundary.geometry, out_shape=(src.height, src.width), transform=src.transform, invert=True)
            total = int(inside.sum())
            missing = int(np.isnan(arr[inside]).sum())
            rows.append({"layer": name, "source": str(path), "inside_cells": total, "missing_inside": missing, "missing_pct": 100 * missing / total if total else np.nan})
    df = pd.DataFrame(rows)
    out = TABLES_DIR / "paper1_map_null_qa_report.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")


def save_final_register() -> None:
    rows = [
        ["Figure 1", FIGURES["fig01"].name, "Introduction", "Conceptual and parameter framework"],
        ["Figure 2", FIGURES["fig02"].name, "Study Area and Data", "Clipped data stack with legends"],
        ["Figure 3", FIGURES["fig03"].name, "Cloud-Local Data Cube", "LST cube slices and elevation-coloured distributions with overlaid mean +/-1 SD"],
        ["Figure 4", FIGURES["fig04"].name, "Results 5.1", "LCZ thermal differentiation with raincloud distributions"],
        ["Figure 5", FIGURES["fig05"].name, "Results 5.2", "Bivariate 150 m heat-vegetation hex atlas with class shares and greenness-cooling statistics"],
        ["Figure 6", FIGURES["fig06"].name, "Results 5.3", "Enhanced baseline diagnostics: binned thermal response surface, OLS coefficient intervals, annotated Spearman structure, and model-fit/collinearity summary"],
        ["Figure 7", FIGURES["fig07"].name, "Results 5.4", "Hex-aggregated spatial coefficient atlas for NDVI, NDBI, albedo, and dominant local driver"],
        ["Figure 8", FIGURES["fig08"].name, "Results 5.5", "Heat-justice sociodemographic synthesis with mahalle Moran robustness diagnostics"],
        ["Appendix Figure 1", APPENDIX["appendix_fig01"].name, "Supplementary", "Neighborhood population cartogram"],
        ["Appendix Figure 2", APPENDIX["appendix_fig02"].name, "Supplementary", "SRTM terrain controls"],
        ["Appendix Figure 3", APPENDIX["appendix_fig03"].name, "Supplementary", "Hex local fit proxy (R2)"],
        ["Appendix Figure 4", APPENDIX["appendix_fig04"].name, "Supplementary", "Relief-shaded SRTM LST drape"],
        ["Appendix Figure 5", APPENDIX["appendix_fig05"].name, "Supplementary", "Full correlation matrix"],
    ]
    df = pd.DataFrame(rows, columns=["figure", "file", "manuscript_location", "purpose"])
    df.to_csv(FINAL_DIR / "paper1_final_figure_register.csv", index=False, encoding="utf-8-sig")
    lines = ["# Paper 1 Final Figure Register", "", "| Figure | File | Manuscript location | Purpose |", "|---|---|---|---|"]
    lines.extend([f"| {a} | `{b}` | {c} | {d} |" for a, b, c, d in rows])
    (FINAL_DIR / "paper1_final_figure_register.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    logger = get_logger("p1_15_final_manuscript_figures")
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    APPENDIX_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Generating corrected final manuscript figure package.")

    only = parse_only_args(sys.argv[1:])

    jobs: list[tuple[str, callable[[], None]]] = [
        ("fig01", save_conceptual_framework),
        ("fig02", save_study_area_stack),
        ("fig03", save_datacube_panel),
        ("fig04", save_lcz_thermal_differentiation),
        ("fig05", save_bivariate_heat_vegetation),
        ("fig06", save_baseline_diagnostics),
        ("fig07", save_hex_coefficient_atlas),
        ("fig08", save_heat_justice_synthesis),
        ("appendix_fig01", save_hexagonal_district_cartogram),
        ("appendix_fig02", save_terrain_appendix),
        ("appendix_fig03", save_hex_local_r2),
        ("appendix_fig04", save_srtm_3d_drape),
        ("appendix_fig05", save_correlation_appendix),
    ]

    for key, job in jobs:
        if only is not None and key not in only:
            continue
        logger.info("Generating %s", key)
        job()
        if key in FIGURES:
            logger.info("Wrote final figure: %s", FIGURES[key])
        elif key in APPENDIX:
            logger.info("Wrote final figure: %s", APPENDIX[key])

    if only is None:
        write_null_qa_report()
        save_final_register()
        logger.info("Wrote final register: %s", FINAL_DIR / "paper1_final_figure_register.md")


if __name__ == "__main__":
    main()
