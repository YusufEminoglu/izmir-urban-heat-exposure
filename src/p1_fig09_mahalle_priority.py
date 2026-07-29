"""Generate Paper 1 Figure 09 with a 3-panel priority layout.

Panel (a): Izmir mahalle-scale planning-priority map (Jenks classes).
Panel (b): Radial stacked district profile (radialGraph-style logic).
Panel (c): Diverging district chart (social vulnerability vs physical exposure).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import mapclassify
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[2]
PRIORITY_GPKG = ROOT / "data" / "04_models" / "paper1" / "paper1_mahalle_heat_justice_priority.gpkg"
MAHALLE_BASE_GPKG = ROOT / "data" / "03_processed" / "paper1" / "mahalle_population_controls_2024.gpkg"
BOUNDARY_GPKG = ROOT / "data" / "02_interim" / "paper1" / "izmir_study_boundary.gpkg"
OUT_FIG = ROOT / "outputs" / "figures" / "paper1" / "final" / "fig09_mahalle_heat_justice_priority.png"

# Base style
BG = "#FFFFFF"
INK = "#344150"
MUTED = "#7A8590"
GRID = "#E7EBEA"
BOUNDARY_LINE = "#2F3D4A"
LIGHT = "#F1F3F1"
CMAP_PRIORITY = LinearSegmentedColormap.from_list(
    "priority_soft", [LIGHT, "#BFD7D0", "#86B7AA", "#D8AAA0", "#B97872"]
)

# radialGraph.py style colors
RG_COLORS = {
    "LST": "#99CCE5",
    "IMP": "#808080",
    "TCH": "#d8b365",
    "PDN": "#5ab4ac",
    "P65": "#BFBFBF",
    "P5": "#E68080",
}
RG_TEXT = "#37474f"


def _clean_ax(ax) -> None:
    ax.set_facecolor(BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _tag(ax, label: str) -> None:
    ax.text(
        0.012,
        0.988,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.8,
        fontweight="bold",
        color=INK,
        zorder=30,
        bbox=dict(boxstyle="square,pad=0.12", facecolor=BG, edgecolor="none", alpha=0.85),
    )


def _boundary_gdf() -> gpd.GeoDataFrame:
    if BOUNDARY_GPKG.exists():
        bnd = gpd.read_file(BOUNDARY_GPKG).to_crs("EPSG:32635")
        bnd = bnd[bnd.geometry.notna()].copy()
        bnd["geometry"] = bnd.geometry.buffer(0)
        return bnd
    return gpd.read_file(MAHALLE_BASE_GPKG).to_crs("EPSG:32635").dissolve().reset_index(drop=True)


def _add_graticule(ax, boundary: gpd.GeoDataFrame, interval_deg: float = 0.25) -> None:
    b4326 = boundary.to_crs("EPSG:4326")
    minlon, minlat, maxlon, maxlat = b4326.total_bounds
    lon_ticks = np.arange(np.ceil(minlon / interval_deg) * interval_deg, maxlon + interval_deg, interval_deg)
    lat_ticks = np.arange(np.ceil(minlat / interval_deg) * interval_deg, maxlat + interval_deg, interval_deg)
    for lon in lon_ticks:
        line = gpd.GeoSeries.from_wkt(
            [f"LINESTRING ({lon} {minlat-interval_deg}, {lon} {maxlat+interval_deg})"], crs="EPSG:4326"
        ).to_crs(boundary.crs).iloc[0]
        clipped = gpd.GeoSeries([line], crs=boundary.crs).intersection(boundary.geometry.union_all()).iloc[0]
        if not clipped.is_empty:
            if clipped.geom_type == "LineString":
                x, y = clipped.xy
                ax.plot(x, y, color="#7F9094", linewidth=0.24, alpha=0.42, zorder=3)
            else:
                for seg in clipped.geoms:
                    x, y = seg.xy
                    ax.plot(x, y, color="#7F9094", linewidth=0.24, alpha=0.42, zorder=3)
    for lat in lat_ticks:
        line = gpd.GeoSeries.from_wkt(
            [f"LINESTRING ({minlon-interval_deg} {lat}, {maxlon+interval_deg} {lat})"], crs="EPSG:4326"
        ).to_crs(boundary.crs).iloc[0]
        clipped = gpd.GeoSeries([line], crs=boundary.crs).intersection(boundary.geometry.union_all()).iloc[0]
        if not clipped.is_empty:
            if clipped.geom_type == "LineString":
                x, y = clipped.xy
                ax.plot(x, y, color="#7F9094", linewidth=0.24, alpha=0.42, zorder=3)
            else:
                for seg in clipped.geoms:
                    x, y = seg.xy
                    ax.plot(x, y, color="#7F9094", linewidth=0.24, alpha=0.42, zorder=3)


def _add_graticule_labels(ax, boundary: gpd.GeoDataFrame, interval_deg: float = 0.25) -> None:
    b4326 = boundary.to_crs("EPSG:4326")
    minlon, minlat, maxlon, maxlat = b4326.total_bounds
    lon_ticks = np.arange(np.ceil(minlon / interval_deg) * interval_deg, maxlon + interval_deg, interval_deg)[::2]
    lat_ticks = np.arange(np.ceil(minlat / interval_deg) * interval_deg, maxlat + interval_deg, interval_deg)[::2]
    x0, y0, x1, y1 = boundary.total_bounds
    pad_x = 0.012 * (x1 - x0)
    pad_y = 0.012 * (y1 - y0)
    # Remove edge labels that fall outside desired display frame.
    lon_ticks = np.array([v for v in lon_ticks if not np.isclose(v, 26.25)])
    lat_ticks = np.array([v for v in lat_ticks if not np.isclose(v, 39.50)])
    for lon in lon_ticks:
        p = gpd.GeoSeries.from_wkt([f"POINT ({lon} {minlat})"], crs="EPSG:4326").to_crs(boundary.crs).iloc[0]
        ax.text(p.x, y0 - pad_y, f"{lon:.2f}E", fontsize=7.0, color=MUTED, ha="center", va="top", zorder=9)
    for lat in lat_ticks:
        p = gpd.GeoSeries.from_wkt([f"POINT ({minlon} {lat})"], crs="EPSG:4326").to_crs(boundary.crs).iloc[0]
        ax.text(x0 - pad_x, p.y, f"{lat:.2f}N", fontsize=7.0, color=MUTED, ha="right", va="center", rotation=90, zorder=9)


def _add_north_arrow(
    ax,
    x_frac: float = 0.08,
    y_frac: float = 0.16,
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
        fontweight="bold",
        color=color,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0),
        zorder=20,
    )


def _add_scale_bar(ax, length_km: int = 20, x_frac: float = 0.055, y_frac: float = 0.055) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    length = length_km * 1000
    x_start = x0 + x_frac * (x1 - x0)
    y = y0 + y_frac * (y1 - y0)
    x_end = x_start + length
    ax.plot([x_start, x_end], [y, y], color=INK, linewidth=1.7, solid_capstyle="butt", zorder=20)
    ax.plot([x_start, x_start], [y - 900, y + 900], color=INK, linewidth=0.9, zorder=20)
    ax.plot([x_end, x_end], [y - 900, y + 900], color=INK, linewidth=0.9, zorder=20)
    ax.text(
        (x_start + x_end) / 2,
        y + 1800,
        f"{length_km} km",
        fontsize=6.2,
        color=INK,
        ha="center",
        va="bottom",
        zorder=20,
    )


def _add_soft_segmented_scale_bar(
    ax,
    x_frac: float = 0.08,
    y_frac: float = 0.055,
    width_frac: float = 0.14,
    segments: int = 4,
    total_km: int = 20,
    height_frac: float = 0.010,
    color_a: str = "#D5E0E3",
    color_b: str = "#A9BCC3",
    text_color: str = "#91A4AD",
) -> None:
    x0 = x_frac - width_frac / 2
    y0 = y_frac
    seg_w = width_frac / segments
    for idx in range(segments):
        face = color_a if idx % 2 == 0 else color_b
        ax.add_patch(
            mpatches.Rectangle(
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
            color="#8FA3AB",
            linewidth=0.45,
            zorder=22,
            clip_on=False,
        )
    ax.text(
        x0 + width_frac / 2,
        y0 - 0.014,
        f"{total_km} km",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.0,
        color=text_color,
        zorder=23,
        clip_on=False,
    )


def _add_panel_a_furniture(ax, x_frac: float = 0.08, y_frac: float = 0.11) -> None:
    soft_color = "#91A4AD"
    _add_north_arrow(
        ax,
        x_frac=x_frac,
        y_frac=y_frac + 0.08,
        dy_frac=0.048,
        color=soft_color,
        fontsize=6.8,
        lw=0.65,
    )
    _add_soft_segmented_scale_bar(
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


def _add_top10_component_panel(ax, mahalle: gpd.GeoDataFrame) -> None:
    top = mahalle.sort_values("priority_score_0_100", ascending=False).head(10).copy()
    if top.empty:
        return
    # Convert to 0..100 per component for compact equal-width cell bars.
    comp = pd.DataFrame(
        {
            "name": top["mahalle_name_final"].astype(str).str.title(),
            "MGWR": top["morph_pressure"].astype(float),
            "Vulnerable": top["vulnerable_share"].astype(float),
            "Heat": top["lst_mean"].astype(float),
            "Population": np.log1p(top["pop_total"].astype(float)),
        }
    )
    for col in ["MGWR", "Vulnerable", "Heat", "Population"]:
        v = comp[col].to_numpy(dtype=float)
        vmin, vmax = np.nanmin(v), np.nanmax(v)
        if np.isclose(vmax, vmin):
            comp[col] = 100.0
        else:
            comp[col] = 100.0 * (v - vmin) / (vmax - vmin)

    # Layout in axes-fraction coordinates (panel A top-right, slightly shifted left).
    x0, y0, w, h = 0.64, 0.58, 0.315, 0.36
    header_y = y0 + h - 0.020
    ax.text(
        x0 + w * 0.50,
        header_y,
        "Top 10 vulnerable neighbourhood",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.6,
        color=INK,
        fontweight="bold",
        zorder=26,
    )

    cols = ["MGWR", "Vulnerable", "Heat", "Population"]
    name_col_w = 0.24 * w
    col_x_start = x0 + name_col_w
    col_total_w = w - name_col_w
    cw = col_total_w / 4.0
    row_top = y0 + h - 0.06
    rh = (h - 0.08) / 10.0

    for i, c in enumerate(cols):
        cx = col_x_start + i * cw + cw / 2
        ax.text(
            cx,
            row_top + 0.011,
            c,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=6.2,
            color=MUTED,
            zorder=26,
        )

    fill_colors = {"MGWR": "#808080", "Vulnerable": "#E68080", "Heat": "#99CCE5", "Population": "#5ab4ac"}
    for r_idx, row in comp.reset_index(drop=True).iterrows():
        y = row_top - (r_idx + 1) * rh
        ax.text(
            x0 + 0.010,
            y + rh * 0.52,
            str(row["name"]),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=6.1,
            color=INK,
            zorder=26,
        )
        for i, c in enumerate(cols):
            cell_x = col_x_start + i * cw
            # cell outline
            ax.add_patch(
                mpatches.Rectangle(
                    (cell_x, y + rh * 0.12),
                    cw * 0.96,
                    rh * 0.72,
                    transform=ax.transAxes,
                    facecolor="none",
                    edgecolor="#D0D7D6",
                    linewidth=0.35,
                    zorder=26,
                )
            )
            frac = float(np.clip(row[c] / 100.0, 0.0, 1.0))
            ax.add_patch(
                mpatches.Rectangle(
                    (cell_x, y + rh * 0.12),
                    cw * 0.96 * frac,
                    rh * 0.72,
                    transform=ax.transAxes,
                    facecolor=fill_colors[c],
                    edgecolor="none",
                    alpha=0.9,
                    zorder=27,
                )
            )


def _norm_1_10(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    scaler = MinMaxScaler(feature_range=(1, 10))
    out = df.copy()
    out[cols] = scaler.fit_transform(out[cols].astype(float))
    return out


def build_fig09() -> None:
    mahalle_score = gpd.read_file(PRIORITY_GPKG).to_crs("EPSG:32635")
    mahalle_all = gpd.read_file(MAHALLE_BASE_GPKG).to_crs("EPSG:32635")
    mahalle = mahalle_score[mahalle_score["priority_score_0_100"].notna()].copy()
    excluded = mahalle_all[
        mahalle_all["pop_total"].isna()
        | (pd.to_numeric(mahalle_all["pop_total"], errors="coerce") < 500)
    ].copy()

    # District features derived from mahalle table.
    d_agg = (
        mahalle.groupby("district_name_final", dropna=True)
        .agg(
            lst=("lst_mean", "mean"),
            morph=("morph_pressure", "mean"),
            hot=("hot_cell_share", "mean"),
            child=("child_0_14_share", "mean"),
            elderly=("elderly_65_plus_share", "mean"),
            pop=("pop_total", "sum"),
            priority=("priority_score_0_100", "mean"),
        )
        .reset_index()
        .rename(columns={"district_name_final": "district"})
    )
    d_agg = d_agg[d_agg["district"].notna()].copy()

    d_agg["pop_scale"] = np.log1p(d_agg["pop"].astype(float))
    d_scaled = _norm_1_10(d_agg, ["lst", "morph", "hot", "child", "elderly", "pop_scale"])
    d_scaled["score_physical"] = 0.50 * d_scaled["lst"] + 0.30 * d_scaled["morph"] + 0.20 * d_scaled["hot"]
    d_scaled["score_social"] = 0.45 * d_scaled["child"] + 0.35 * d_scaled["elderly"] + 0.20 * d_scaled["pop_scale"]
    d_scaled["total"] = d_scaled["score_physical"] + d_scaled["score_social"]
    d_scaled = d_scaled.sort_values("total", ascending=True).reset_index(drop=True)

    fig = plt.figure(figsize=(13.5, 10.0), facecolor=BG)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.9, 1.0],
        left=0.012,
        right=0.995,
        bottom=0.018,
        top=0.985,
        hspace=0.045,
        wspace=0.02,
    )

    # Panel A: left complete (spans both rows)
    ax_map = fig.add_subplot(gs[:, 0])
    _clean_ax(ax_map)
    boundary = _boundary_gdf()
    boundary.plot(ax=ax_map, color="#EEF2F0", edgecolor="none", zorder=0)
    if not excluded.empty:
        excluded = gpd.overlay(excluded, boundary[["geometry"]], how="intersection")
        excluded = excluded[excluded.geometry.notna()].copy()
        excluded["geometry"] = excluded.geometry.buffer(0)
        excluded.plot(ax=ax_map, color="#D9DEDD", edgecolor="none", linewidth=0, alpha=0.92, zorder=1)
        excluded.plot(
            ax=ax_map,
            facecolor="none",
            edgecolor="#B8C1BF",
            linewidth=0.18,
            hatch="///",
            alpha=0.9,
            zorder=2.1,
        )

    k = 6
    classifier = mapclassify.NaturalBreaks(mahalle["priority_score_0_100"].to_numpy(), k=k)
    bins = classifier.bins
    mahalle["jenks_class"] = classifier.yb
    class_colors = [CMAP_PRIORITY(v) for v in np.linspace(0.16, 0.96, k)]
    class_cmap = ListedColormap(class_colors)
    mahalle.plot(
        ax=ax_map,
        column="jenks_class",
        cmap=class_cmap,
        linewidth=0.0,
        edgecolor="none",
        alpha=0.98,
        zorder=2,
    )

    _add_graticule(ax_map, boundary, interval_deg=0.25)
    _add_graticule_labels(ax_map, boundary, interval_deg=0.25)
    boundary.boundary.plot(ax=ax_map, color="#202B33", linewidth=1.05, zorder=6, alpha=0.92)
    minx, miny, maxx, maxy = boundary.total_bounds
    pad = 2200.0
    ax_map.set_xlim(minx - pad, maxx + pad)
    ax_map.set_ylim(miny - pad, maxy + pad)
    ax_map.set_aspect("equal")
    _add_panel_a_furniture(ax_map, x_frac=0.125, y_frac=0.095)

    low = float(mahalle["priority_score_0_100"].min())
    breaks = [low] + [float(v) for v in bins]
    patches_map: list[mpatches.Patch] = []
    for i in range(k):
        patches_map.append(
            mpatches.Patch(color=class_colors[i], label=f"{breaks[i]:.1f} - {breaks[i+1]:.1f}")
        )
    patches_map.append(
        mpatches.Patch(facecolor="#D9DEDD", edgecolor="#B8C1BF", hatch="///", label="<500 pop")
    )
    patches_map.append(
        mpatches.Patch(facecolor="none", edgecolor="#202B33", linewidth=1.0, label="Study area boundary")
    )
    patches_map.append(
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="None",
            markersize=8.5,
            markerfacecolor="#202B33",
            markeredgecolor="white",
            markeredgewidth=0.4,
            label="Top 10 vulnerable mahalle",
        )
    )
    ax_map.legend(
        handles=patches_map,
        title="Planning priority (Jenks)",
        title_fontsize=8.6,
        fontsize=7.6,
        loc="upper left",
        bbox_to_anchor=(0.012, 0.955),
        frameon=True,
        facecolor="white",
        edgecolor="none",
        ncol=1,
    )
    # Align legend title and Top-10 title horizontally.
    leg = ax_map.get_legend()
    if leg is not None:
        leg.set_title("Planning priority (Jenks)")
        leg._legend_title_box._text.set_position((0, 0))
    # Mark top 10 vulnerable mahalle locations with star symbols.
    top_star = mahalle.sort_values("priority_score_0_100", ascending=False).head(10).copy()
    if not top_star.empty:
        top_star["rp"] = top_star.geometry.representative_point()
        ax_map.scatter(
            top_star["rp"].x,
            top_star["rp"].y,
            marker="*",
            s=52,
            c="#202B33",
            edgecolors="white",
            linewidths=0.4,
            zorder=8,
            alpha=0.95,
        )
    _add_top10_component_panel(ax_map, mahalle)
    _tag(ax_map, "(a)")

    # Panel B: right top
    ax_rad = fig.add_subplot(gs[0, 1], projection="polar")
    n = len(d_scaled)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    width = 2 * np.pi / n * 0.90
    stack_order = ["morph", "lst", "hot", "child", "elderly", "pop_scale"]
    # Panel B palette: discrete and publication-friendly.
    color_map = {
        "morph": "#7E9295",      # MGWR morphology pressure
        "lst": "#8EC1DA",        # mean LST
        "hot": "#D8AAA0",        # hot-cell share
        "child": "#E58A8A",      # child share
        "elderly": "#BDBDBD",    # elderly share
        "pop_scale": "#86B7AA",  # population scale (log)
    }
    bottom = np.zeros(n, dtype=float)
    for key in stack_order:
        values = d_scaled[key].to_numpy(dtype=float)
        ax_rad.bar(
            angles,
            values,
            width=width,
            bottom=bottom,
            color=color_map[key],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.95,
        )
        bottom += values

    ax_rad.set_theta_offset(np.pi / 2)
    ax_rad.set_theta_direction(-1)
    ax_rad.set_ylim(-14.0, float(bottom.max() + 7.0))
    ax_rad.set_frame_on(False)
    ax_rad.xaxis.grid(False)
    ax_rad.yaxis.grid(False)
    ax_rad.set_xticks([])
    ax_rad.set_yticks([])

    for angle, height, label in zip(angles, bottom, d_scaled["district"].tolist()):
        rot = np.rad2deg(angle)
        if 0 <= angle < np.pi:
            ha, txt_rot = "left", -rot + 90
        else:
            ha, txt_rot = "right", -rot + 270
        ax_rad.text(
            angle,
            height + 1.5,
            str(label).title(),
            ha=ha,
            va="center",
            rotation=txt_rot,
            rotation_mode="anchor",
            fontsize=7.3,
            color=RG_TEXT,
            fontweight="normal",
        )

    top_row = d_scaled.iloc[-1]
    ax_rad.text(
        0.5,
        0.5,
        (
            f"HIGHEST RISK:\n{str(top_row['district']).upper()}\n"
            f"(score: {top_row['total']:.1f})\n\n"
            f"Avg Phys: {d_scaled['score_physical'].mean():.1f}\n"
            f"Avg Soc: {d_scaled['score_social'].mean():.1f}"
        ),
        transform=ax_rad.transAxes,
        ha="center",
        va="center",
        fontsize=6.5,
        color=INK,
        fontweight="bold",
    )

    patches = [
        mpatches.Patch(color=color_map["morph"], label="MGWR morph. pressure"),
        mpatches.Patch(color=color_map["lst"], label="Mean LST"),
        mpatches.Patch(color=color_map["hot"], label="Hot-cell share"),
        mpatches.Patch(color=color_map["child"], label="Child share (0-14)"),
        mpatches.Patch(color=color_map["elderly"], label="Elderly share (65+)"),
        mpatches.Patch(color=color_map["pop_scale"], label="Population scale (log)"),
    ]
    ax_rad.legend(
        handles=patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=3,
        frameon=False,
        fontsize=7.0,
    )

    # Panel C: right bottom
    ax_div = fig.add_subplot(gs[1, 1])
    ax_div.set_facecolor(BG)
    ax_div.set_box_aspect(1.22)
    y_pos = np.arange(len(d_scaled))
    social = d_scaled["score_social"].to_numpy(dtype=float)
    physical = d_scaled["score_physical"].to_numpy(dtype=float)

    ax_div.barh(y_pos, -social, color="#999999", alpha=0.30, height=0.65)
    ax_div.barh(y_pos, -d_scaled["child"].to_numpy(), color=RG_COLORS["P5"], alpha=0.62, height=0.65, label="High child pop.")
    ax_div.barh(y_pos, physical, color="#999999", alpha=0.30, height=0.65)
    ax_div.barh(y_pos, d_scaled["lst"].to_numpy(), color=RG_COLORS["LST"], alpha=0.62, height=0.65, label="High LST")
    ax_div.axvline(0, color="#AAAAAA", linewidth=1.0)

    ax_div.set_yticks(y_pos)
    ax_div.set_yticklabels([str(v).title() for v in d_scaled["district"].tolist()], fontsize=7.3, color=RG_TEXT)
    ax_div.tick_params(axis="y", length=0, pad=0)
    ax_div.set_xticks([])
    for side in ["top", "right", "left", "bottom"]:
        ax_div.spines[side].set_visible(False)

    ax_div.text(0.01, 1.01, "High Social\nVulnerability", transform=ax_div.transAxes, ha="left", va="top", fontsize=8.0, color=RG_COLORS["P5"])
    ax_div.text(0.99, 1.01, "High Physical\nExposure", transform=ax_div.transAxes, ha="right", va="top", fontsize=8.0, color=RG_COLORS["LST"])
    ax_div.set_xlabel(
        "<- Social vulnerability dominant | Physical exposure dominant ->",
        fontsize=8.3,
        color="#666666",
        labelpad=10,
    )
    ax_div.legend(frameon=False, fontsize=7.8, loc="lower right")
    # Align panel labels (b) and (c) on the same vertical line in figure coordinates.
    pos_b = ax_rad.get_position()
    pos_c = ax_div.get_position()
    label_x = min(pos_b.x0, pos_c.x0) + 0.002
    fig.text(
        label_x,
        pos_b.y1 - 0.006,
        "(b)",
        ha="left",
        va="top",
        fontsize=10.8,
        fontweight="bold",
        color=INK,
        zorder=40,
        bbox=dict(boxstyle="square,pad=0.12", facecolor=BG, edgecolor="none", alpha=0.85),
    )
    fig.text(
        label_x,
        pos_c.y1 - 0.006,
        "(c)",
        ha="left",
        va="top",
        fontsize=10.8,
        fontweight="bold",
        color=INK,
        zorder=40,
        bbox=dict(boxstyle="square,pad=0.12", facecolor=BG, edgecolor="none", alpha=0.85),
    )

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=320, facecolor=BG, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main() -> None:
    build_fig09()
    print(f"Wrote {OUT_FIG}")


if __name__ == "__main__":
    main()
