"""
Paper 1 – Raster coverage sanity check.

Goal: identify rasters that do not fully cover the study boundary, or that contain
significant NoData/NaN inside the boundary (i.e., "holes" that will propagate to
downstream grids / figures).

Output:
  outputs/tables/paper1/paper1_raster_coverage_report.csv
"""

from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import box, mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _config import DATA_INTERIM, INTERIM_ANNUAL_LST, PROCESSED_ALIGNED, BOUNDARY_GPKG, TABLES_DIR  # noqa: E402


@dataclass(frozen=True)
class RasterCoverageRow:
    path: str
    basename: str
    status: str
    crs: str | None = None
    nodata: float | None = None
    bbox_outside: bool | None = None
    area_in_bounds_frac: float | None = None
    boundary_pixels: int | None = None
    valid_pixels: int | None = None
    valid_frac: float | None = None
    width: int | None = None
    height: int | None = None
    pixel_x: float | None = None
    pixel_y: float | None = None
    error: str | None = None


def _scan_rasters() -> list[str]:
    patterns = [
        str(DATA_INTERIM / "*.tif"),
        str(INTERIM_ANNUAL_LST / "*.tif"),
        str(PROCESSED_ALIGNED / "*.tif"),
    ]
    paths: list[str] = []
    for pat in patterns:
        paths.extend(glob.glob(pat))
    return sorted(set(paths))


def build_report(
    boundary_path: str = str(BOUNDARY_GPKG),
    out_path: str = str(TABLES_DIR / "paper1_raster_coverage_report.csv"),
    coverage_area_threshold: float = 0.999,
    valid_pixel_threshold: float = 0.98,
) -> pd.DataFrame:
    gdf = gpd.read_file(boundary_path)
    geom = gdf.union_all()

    rows: list[RasterCoverageRow] = []
    for path in _scan_rasters():
        basename = os.path.basename(path)
        try:
            with rasterio.open(path) as ds:
                crs = ds.crs
                if crs is None:
                    rows.append(RasterCoverageRow(path=path, basename=basename, status="no_crs"))
                    continue

                geom2 = gpd.GeoSeries([geom], crs=gdf.crs).to_crs(crs).iloc[0]
                raster_bounds = ds.bounds
                raster_bounds_poly = box(
                    raster_bounds.left,
                    raster_bounds.bottom,
                    raster_bounds.right,
                    raster_bounds.top,
                )

                area_total = float(geom2.area) if geom2.area else 0.0
                area_in = float(geom2.intersection(raster_bounds_poly).area) if area_total > 0 else 0.0
                area_in_bounds_frac = (area_in / area_total) if area_total > 0 else np.nan

                bb = geom2.bounds
                bbox_outside = (
                    (bb[0] < raster_bounds.left)
                    or (bb[1] < raster_bounds.bottom)
                    or (bb[2] > raster_bounds.right)
                    or (bb[3] > raster_bounds.top)
                )

                inter = geom2.intersection(raster_bounds_poly)
                if inter.is_empty:
                    rows.append(
                        RasterCoverageRow(
                            path=path,
                            basename=basename,
                            status="no_intersection",
                            crs=str(crs),
                            bbox_outside=bbox_outside,
                            area_in_bounds_frac=float(area_in_bounds_frac),
                        )
                    )
                    continue

                # Pixel-validity only computed where geometry is within raster bounds.
                inter_bb = inter.bounds
                win = (
                    rasterio.windows.from_bounds(*inter_bb, transform=ds.transform)
                    .round_offsets()
                    .round_lengths()
                )
                data = ds.read(1, window=win, masked=False)
                transform = ds.window_transform(win)
                mask = geometry_mask(
                    [mapping(inter)],
                    out_shape=data.shape,
                    transform=transform,
                    invert=True,
                    all_touched=False,
                )
                inside = data[mask]
                boundary_pixels = int(mask.sum())
                if boundary_pixels == 0:
                    rows.append(
                        RasterCoverageRow(
                            path=path,
                            basename=basename,
                            status="empty_mask",
                            crs=str(crs),
                            bbox_outside=bbox_outside,
                            area_in_bounds_frac=float(area_in_bounds_frac),
                        )
                    )
                    continue

                nodata = ds.nodata
                if np.issubdtype(inside.dtype, np.floating):
                    finite = np.isfinite(inside)
                else:
                    finite = np.ones_like(inside, dtype=bool)
                if nodata is not None:
                    finite &= inside != nodata

                valid_pixels = int(finite.sum())
                valid_frac = valid_pixels / boundary_pixels

                rows.append(
                    RasterCoverageRow(
                        path=path,
                        basename=basename,
                        status="ok",
                        crs=str(crs),
                        nodata=None if nodata is None else float(nodata),
                        bbox_outside=bool(bbox_outside),
                        area_in_bounds_frac=float(area_in_bounds_frac),
                        boundary_pixels=int(boundary_pixels),
                        valid_pixels=int(valid_pixels),
                        valid_frac=float(valid_frac),
                        width=int(ds.width),
                        height=int(ds.height),
                        pixel_x=float(ds.res[0]),
                        pixel_y=float(ds.res[1]),
                    )
                )
        except Exception as e:
            rows.append(RasterCoverageRow(path=path, basename=basename, status="error", error=str(e)))

    df = pd.DataFrame([r.__dict__ for r in rows])

    df["problem"] = False
    ok = df["status"] == "ok"
    df.loc[~ok, "problem"] = True
    df.loc[ok & (df["area_in_bounds_frac"] < coverage_area_threshold), "problem"] = True
    df.loc[ok & (df["valid_frac"] < valid_pixel_threshold), "problem"] = True

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.sort_values(
        ["problem", "area_in_bounds_frac", "valid_frac", "basename"],
        ascending=[False, True, True, True],
    ).to_csv(out_path, index=False)
    return df


def main() -> None:
    df = build_report()
    prob = df[df["problem"]].copy()
    print(str(TABLES_DIR / "paper1_raster_coverage_report.csv"))
    if len(prob) == 0:
        print("No problems detected.")
        return
    show = prob[
        [
            "basename",
            "area_in_bounds_frac",
            "valid_frac",
            "bbox_outside",
            "nodata",
            "crs",
            "path",
        ]
    ].sort_values(["area_in_bounds_frac", "valid_frac"])
    with pd.option_context("display.max_rows", 500, "display.max_colwidth", 180):
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()

