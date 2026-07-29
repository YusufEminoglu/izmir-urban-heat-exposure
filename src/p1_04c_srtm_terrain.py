"""Create SRTM-derived terrain controls aligned to the Study 1 100 m grid."""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:  # pragma: no cover
    distance_transform_edt = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _config import DATA_INTERIM, DATA_PROCESSED, TABLES_DIR, ensure_dirs  # noqa: E402
from _config import get_logger  # noqa: E402


BOUNDARY = DATA_INTERIM / "izmir_study_boundary.gpkg"
REFERENCE_GRID = DATA_INTERIM / "lcz_filter_izmir_2018_100m.tif"
SRTM = DATA_INTERIM / "izmir_srtm.tif"
OUTPUT_DIR = DATA_PROCESSED / "aligned_covariates"
REPORT_CSV = TABLES_DIR / "paper1_srtm_terrain_report.csv"

NODATA = -9999.0

OUTPUTS = {
    "elevation": OUTPUT_DIR / "izmir_srtm_elevation_aligned_100m.tif",
    "slope": OUTPUT_DIR / "izmir_srtm_slope_deg_aligned_100m.tif",
    "aspect_sin": OUTPUT_DIR / "izmir_srtm_aspect_sin_aligned_100m.tif",
    "aspect_cos": OUTPUT_DIR / "izmir_srtm_aspect_cos_aligned_100m.tif",
}


def fill_missing_inside_boundary_nearest(array: np.ndarray, inside_mask: np.ndarray) -> np.ndarray:
    """(Deprecated) Fill NaN holes inside the boundary using nearest-neighbour values.

    For academic correctness, we avoid implicit spatial imputation at the terrain
    control creation step. If a DEM export has gaps, the export should be fixed
    upstream or gaps should be documented and masked explicitly.
    """
    if distance_transform_edt is None:  # pragma: no cover
        return array

    missing = inside_mask & ~np.isfinite(array)
    if not missing.any():
        return array

    valid = inside_mask & np.isfinite(array)
    if not valid.any():
        return array

    inv_valid = ~valid
    _, (iy, ix) = distance_transform_edt(inv_valid, return_indices=True)
    array[missing] = array[iy[missing], ix[missing]]
    return array


def write_raster(path: Path, array: np.ndarray, ref: rasterio.DatasetReader, description: str) -> None:
    """Write one float32 aligned terrain raster."""
    profile = ref.profile.copy()
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=NODATA,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        predictor=3,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.where(np.isfinite(array), array, NODATA).astype("float32")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)
        dst.update_tags(
            source=str(SRTM),
            processing="SRTM reprojected to LCZ 100 m grid; outside Izmir boundary set to nodata",
            description=description,
        )


def terrain_from_dem(dem: np.ndarray, xres: float, yres: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return slope degrees, aspect sine, and aspect cosine from a projected DEM."""
    dem_work = dem.copy()
    finite = np.isfinite(dem_work)
    if not finite.any():
        raise ValueError("Aligned DEM contains no finite cells.")
    dem_work[~finite] = np.nanmedian(dem_work[finite])

    dz_dy, dz_dx = np.gradient(dem_work, abs(yres), abs(xres))
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    aspect = np.arctan2(-dz_dx, dz_dy)
    aspect = np.mod(aspect, 2 * np.pi)

    slope[~finite] = np.nan
    aspect[~finite] = np.nan
    aspect_sin = np.sin(aspect)
    aspect_cos = np.cos(aspect)
    return slope, aspect_sin, aspect_cos


def main() -> None:
    ensure_dirs()
    logger = get_logger("p1_04c_srtm_terrain")
    logger.info("Building SRTM terrain controls on Study 1 grid.")

    with rasterio.open(REFERENCE_GRID) as ref:
        boundary = gpd.read_file(BOUNDARY).to_crs(ref.crs)
        inside_mask = geometry_mask(boundary.geometry, out_shape=(ref.height, ref.width), transform=ref.transform, invert=True)

        elevation = np.full((ref.height, ref.width), NODATA, dtype="float32")
        with rasterio.open(SRTM) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=elevation,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                dst_nodata=NODATA,
                resampling=Resampling.bilinear,
            )
            source_meta = {
                "source_crs": src.crs.to_string() if src.crs else None,
                "source_width": src.width,
                "source_height": src.height,
                "source_res_x": src.res[0],
                "source_res_y": src.res[1],
            }

        elevation = elevation.astype("float64")
        elevation[elevation == NODATA] = np.nan
        elevation[~inside_mask] = np.nan
        slope, aspect_sin, aspect_cos = terrain_from_dem(elevation, ref.res[0], ref.res[1])
        for arr in [slope, aspect_sin, aspect_cos]:
            arr[~inside_mask] = np.nan

        write_raster(OUTPUTS["elevation"], elevation, ref, "SRTM elevation, metres")
        write_raster(OUTPUTS["slope"], slope, ref, "SRTM-derived slope, degrees")
        write_raster(OUTPUTS["aspect_sin"], aspect_sin, ref, "SRTM-derived aspect sine; east-west exposure component")
        write_raster(OUTPUTS["aspect_cos"], aspect_cos, ref, "SRTM-derived aspect cosine; north-south exposure component")

    rows = []
    arrays = {
        "elevation": elevation,
        "slope": slope,
        "aspect_sin": aspect_sin,
        "aspect_cos": aspect_cos,
    }
    for name, arr in arrays.items():
        valid = np.isfinite(arr)
        rows.append(
            {
                "variable": name,
                "output": str(OUTPUTS[name]),
                "valid_cells": int(valid.sum()),
                "min": float(np.nanmin(arr)) if valid.any() else np.nan,
                "mean": float(np.nanmean(arr)) if valid.any() else np.nan,
                "max": float(np.nanmax(arr)) if valid.any() else np.nan,
            }
        )
    report = pd.DataFrame(rows)
    for key, value in source_meta.items():
        report[key] = value
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")
    logger.info("Wrote SRTM terrain controls: %s", OUTPUT_DIR)
    logger.info("Wrote SRTM report: %s", REPORT_CSV)


if __name__ == "__main__":
    main()
