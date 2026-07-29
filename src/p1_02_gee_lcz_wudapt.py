"""Export the global WUDAPT-compatible LCZ layer for Izmir from Earth Engine."""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _config import (  # noqa: E402
    GEE_PROJECT,
    DATA_INTERIM,
    TABLES_DIR,
    ensure_dirs,
    require_file,
)
from _config import get_logger  # noqa: E402


DATASET_ID = "RUB/RUBCLIM/LCZ/global_lcz_map/latest"
BOUNDARY = DATA_INTERIM / "izmir_study_boundary.gpkg"
LCZ_FILTER_TIF = DATA_INTERIM / "lcz_filter_izmir_2018_100m.tif"
LCZ_PROBABILITY_TIF = DATA_INTERIM / "lcz_probability_izmir_2018_100m.tif"
METADATA_MD = TABLES_DIR / "lcz_source_metadata.md"


def initialise_earth_engine() -> None:
    """Initialise Earth Engine, raising a clear message if auth is missing."""
    import ee

    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception as exc:
        raise RuntimeError(
            f"Earth Engine is not initialised for project `{GEE_PROJECT}`. Run `earthengine authenticate` or "
            "`python -c \"import ee; ee.Authenticate(); ee.Initialize()\"` first."
        ) from exc


def export_lcz_layers() -> None:
    """Clip and export LCZ_Filter and LCZ_Probability to local GeoTIFFs."""
    import ee
    import geemap

    boundary_path = require_file(BOUNDARY, "Study 1 Izmir boundary")
    boundary = gpd.read_file(boundary_path).to_crs("EPSG:4326")
    region = geemap.geopandas_to_ee(boundary)

    image = ee.ImageCollection(DATASET_ID).mosaic().clip(region)
    region_geometry = region.geometry()

    exports = [
        ("LCZ_Filter", LCZ_FILTER_TIF),
        ("LCZ_Probability", LCZ_PROBABILITY_TIF),
    ]
    for band, output in exports:
        output.parent.mkdir(parents=True, exist_ok=True)
        geemap.ee_export_image(
            image.select(band),
            filename=str(output),
            scale=100,
            region=region_geometry,
            crs="EPSG:32635",
            file_per_band=False,
        )


def write_metadata() -> None:
    """Write a compact source note for later methods sections."""
    METADATA_MD.parent.mkdir(parents=True, exist_ok=True)
    METADATA_MD.write_text(
        "\n".join(
            [
                "# LCZ Source Metadata",
                "",
                f"- Earth Engine dataset: `{DATASET_ID}`",
                f"- Earth Engine project used for export: `{GEE_PROJECT}`",
                "- Product: Global map of Local Climate Zones, latest version.",
                "- Nominal year: 2018.",
                "- Pixel size: 100 m.",
                "- Default band for modelling: `LCZ_Filter`.",
                "- Confidence/support band: `LCZ_Probability`.",
                "- Citation to include: Demuzere et al. (2022), Earth System Science Data, 14, 3835-3873.",
                "- Licence: CC-BY-4.0.",
                "",
                "Method note: use the global LCZ product as the baseline LCZ layer for whole-Izmir modelling. "
                "If visual checks show obvious local errors, add a local validation/refinement step before MGWR.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    logger = get_logger("p1_02_gee_lcz_wudapt")
    logger.info("Preparing WUDAPT-compatible LCZ export for Izmir.")
    initialise_earth_engine()
    export_lcz_layers()
    write_metadata()
    logger.info("Wrote LCZ layer: %s", LCZ_FILTER_TIF)
    logger.info("Wrote LCZ probability layer: %s", LCZ_PROBABILITY_TIF)
    logger.info("Wrote source metadata: %s", METADATA_MD)


if __name__ == "__main__":
    main()
