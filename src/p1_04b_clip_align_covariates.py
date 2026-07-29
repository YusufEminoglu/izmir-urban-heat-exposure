"""Clip and align user-downloaded Study 1 covariate rasters to the LCZ grid."""

from __future__ import annotations

import re
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
OUTPUT_DIR = DATA_PROCESSED / "aligned_covariates"
REPORT_CSV = TABLES_DIR / "paper1_aligned_covariates_report.csv"

CONTINUOUS_NODATA = -9999.0
CATEGORICAL_NODATA = -32768


SKIP_PATTERNS = [
    "lcz_filter",
    "lcz_probability",
    "dsm_sym_mean_100m_aligned",
]


def is_categorical(path: Path) -> bool:
    """Return True for categorical land-cover products."""
    name = path.name.lower()
    return "corine" in name or "glc_fcs" in name


def output_name(path: Path) -> str:
    """Create a clear aligned output name."""
    if path.name.lower() == "tch_filled_nodata.tif":
        return "izmir_KanopiYuksekligi_Veri_aligned_100m.tif"
    stem = path.stem
    stem = re.sub(r"_raw$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_ANALYTIC$", "", stem, flags=re.IGNORECASE)
    return f"{stem}_aligned_100m.tif"


def should_process(path: Path) -> bool:
    """Decide whether a GeoTIFF is a downloaded covariate needing alignment."""
    name = path.name.lower()
    if name == "izmir_kanopiyuksekligi_veri.tif" and (path.parent / "TCH_filled_nodata.tif").exists():
        return False
    return not any(pattern in name for pattern in SKIP_PATTERNS)


def fill_missing_inside_boundary_nearest(
    destination: np.ndarray,
    inside_mask: np.ndarray,
    nodata: float,
) -> np.ndarray:
    """(Deprecated) Fill nodata holes inside the study boundary using nearest-neighbour values.

    This function is intentionally *not used* in the main workflow anymore.
    For academic correctness, missing values inside the boundary should remain
    missing and be handled transparently (masking, explicit "no-data" symbology,
    or model-specific imputation with clear reporting).
    """
    if np.issubdtype(destination.dtype, np.floating):
        missing = (inside_mask) & ((destination == nodata) | ~np.isfinite(destination))
    else:
        missing = (destination == nodata) & inside_mask
    if not missing.any():
        return destination

    if np.issubdtype(destination.dtype, np.floating):
        valid = inside_mask & (destination != nodata) & np.isfinite(destination)
    else:
        valid = (destination != nodata) & inside_mask
    if not valid.any():
        # Degenerate case: nothing to copy from; leave as-is.
        return destination

    if distance_transform_edt is None:  # pragma: no cover
        return destination

    inv_valid = ~valid
    _, (iy, ix) = distance_transform_edt(inv_valid, return_indices=True)
    destination[missing] = destination[iy[missing], ix[missing]]
    return destination


def align_one(path: Path, ref: rasterio.DatasetReader, inside_mask: np.ndarray) -> dict[str, object]:
    """Align one raster to the reference grid and mask outside-boundary cells."""
    categorical = is_categorical(path)
    nodata = CATEGORICAL_NODATA if categorical else CONTINUOUS_NODATA
    dtype = "int16" if categorical else "float32"
    resampling = Resampling.nearest if categorical else Resampling.bilinear

    with rasterio.open(path) as src:
        destination = np.full((ref.height, ref.width), nodata, dtype=dtype)
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            dst_nodata=nodata,
            resampling=resampling,
        )

        zeros_before_mask = int((destination == 0).sum())
        zeros_inside = int(((destination == 0) & inside_mask).sum())
        destination[~inside_mask] = nodata

        holes_before = int(((destination == nodata) & inside_mask).sum())
        holes_after = holes_before

        profile = ref.profile.copy()
        profile.update(
            driver="GTiff",
            dtype=dtype,
            count=1,
            nodata=nodata,
            compress="deflate",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        if not categorical:
            profile["predictor"] = 3

        output = OUTPUT_DIR / output_name(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output, "w", **profile) as dst:
            dst.write(destination, 1)
            dst.update_tags(
                source=str(path),
                processing="aligned to LCZ 100 m grid; outside Izmir boundary set to nodata",
            )

        valid = destination != nodata
        return {
            "source": str(path),
            "output": str(output),
            "kind": "categorical" if categorical else "continuous",
            "source_crs": src.crs.to_string() if src.crs else None,
            "source_width": src.width,
            "source_height": src.height,
            "source_res_x": src.res[0],
            "source_res_y": src.res[1],
            "output_valid_cells": int(valid.sum()),
            "zeros_before_boundary_mask": zeros_before_mask,
            "zeros_inside_boundary_after_alignment": zeros_inside,
            "holes_inside_boundary_before_fill": holes_before,
            "holes_inside_boundary_after_fill": holes_after,
            "output_min": float(np.nanmin(destination[valid])) if valid.any() else np.nan,
            "output_max": float(np.nanmax(destination[valid])) if valid.any() else np.nan,
        }


def main() -> None:
    ensure_dirs()
    logger = get_logger("p1_04b_clip_align_covariates")
    logger.info("Clipping and aligning downloaded Study 1 covariates.")

    candidates = [path for path in sorted(DATA_INTERIM.glob("*.tif")) if should_process(path)]
    with rasterio.open(REFERENCE_GRID) as ref:
        boundary = gpd.read_file(BOUNDARY).to_crs(ref.crs)
        inside_mask = geometry_mask(
            boundary.geometry,
            out_shape=(ref.height, ref.width),
            transform=ref.transform,
            invert=True,
        )
        rows = [align_one(path, ref, inside_mask) for path in candidates]

    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")
    logger.info("Aligned %s covariate rasters into %s", len(rows), OUTPUT_DIR)
    logger.info("Wrote covariate report: %s", REPORT_CSV)


if __name__ == "__main__":
    main()
