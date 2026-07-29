"""Align the processed DSM/SYM-derived surface to the Study 1 100 m grid."""

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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _config import (  # noqa: E402
    EXTERNAL_ELEVATION,
    MAIN_CRS,
    DATA_INTERIM,
    TABLES_DIR,
    ensure_dirs,
    require_file,
)
from _config import get_logger  # noqa: E402
from _raster_ops import raster_metadata  # noqa: E402


DSM_SOURCE = EXTERNAL_ELEVATION / "izmir_sym5.tif"
REFERENCE_GRID = DATA_INTERIM / "lcz_filter_izmir_2018_100m.tif"
BOUNDARY = DATA_INTERIM / "izmir_study_boundary.gpkg"
DSM_MEAN_100M = DATA_INTERIM / "dsm_sym_mean_100m_aligned.tif"
METADATA_CSV = TABLES_DIR / "dsm_sym_100m_metadata.csv"
NODATA_VALUE = -9999.0


def align_dsm_to_reference() -> None:
    """Reproject and average the processed DSM/SYM surface to the LCZ grid."""
    require_file(DSM_SOURCE, "processed DSM/SYM surface")
    require_file(REFERENCE_GRID, "LCZ 100 m reference grid")
    require_file(BOUNDARY, "Study 1 Izmir boundary")

    with rasterio.open(REFERENCE_GRID) as ref, rasterio.open(DSM_SOURCE) as src:
        destination = np.full((ref.height, ref.width), NODATA_VALUE, dtype="float32")
        src_nodata = src.nodata

        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            dst_nodata=NODATA_VALUE,
            resampling=Resampling.average,
        )

        boundary = gpd.read_file(BOUNDARY).to_crs(ref.crs)
        inside_mask = geometry_mask(
            boundary.geometry,
            out_shape=(ref.height, ref.width),
            transform=ref.transform,
            invert=True,
        )
        destination[~inside_mask] = NODATA_VALUE
        destination[destination == 0] = NODATA_VALUE

        profile = ref.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            nodata=NODATA_VALUE,
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        DSM_MEAN_100M.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(DSM_MEAN_100M, "w", **profile) as dst:
            dst.write(destination, 1)
            dst.update_tags(
                description=(
                    "Processed DSM/SYM-derived surface averaged to the Study 1 100 m LCZ grid; "
                    "not direct raw IMM municipal data."
                )
            )


def write_metadata() -> None:
    """Write source/reference/output metadata for auditing."""
    rows = []
    for label, path in [
        ("source_processed_dsm_sym", DSM_SOURCE),
        ("reference_lcz_grid", REFERENCE_GRID),
        ("study_boundary", BOUNDARY),
        ("output_dsm_sym_mean_100m", DSM_MEAN_100M),
    ]:
        if path.suffix.lower() in {".tif", ".tiff"}:
            meta = raster_metadata(path)
        else:
            meta = {"exists": path.exists()}
        meta["label"] = label
        meta["path"] = str(path)
        rows.append(meta)
    METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(METADATA_CSV, index=False, encoding="utf-8-sig")


def main() -> None:
    ensure_dirs()
    logger = get_logger("p1_03_dsm_morphology")
    logger.info("Aligning processed DSM/SYM-derived surface to LCZ 100 m grid.")
    align_dsm_to_reference()
    write_metadata()
    logger.info("Wrote aligned DSM/SYM mean raster: %s", DSM_MEAN_100M)
    logger.info("Wrote DSM/SYM metadata: %s", METADATA_CSV)


if __name__ == "__main__":
    main()
