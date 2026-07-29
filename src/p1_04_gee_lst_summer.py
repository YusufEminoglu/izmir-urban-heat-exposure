"""Export whole-Izmir summer Landsat LST and spectral indices from Earth Engine."""

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


BOUNDARY = DATA_INTERIM / "izmir_study_boundary.gpkg"
REFERENCE_GRID = DATA_INTERIM / "lcz_filter_izmir_2018_100m.tif"
LST_MEAN_TIF = DATA_INTERIM / "lst_mean_izmir_2014_2024_jja_100m.tif"
NDVI_MEAN_TIF = DATA_INTERIM / "ndvi_mean_izmir_2014_2024_jja_100m.tif"
NDBI_MEAN_TIF = DATA_INTERIM / "ndbi_mean_izmir_2014_2024_jja_100m.tif"
ALBEDO_MEAN_TIF = DATA_INTERIM / "albedo_mean_izmir_2014_2024_jja_100m.tif"
METADATA_MD = TABLES_DIR / "landsat_lst_source_metadata.md"


def initialise_earth_engine() -> None:
    """Initialise Earth Engine with the project configured for this study."""
    import ee

    ee.Initialize(project=GEE_PROJECT)


def qa_mask_landsat_l2(image):
    """Mask Landsat Collection 2 Level 2 cloud, shadow, snow, cirrus, and fill pixels."""
    import ee

    qa = image.select("QA_PIXEL")
    mask = (
        qa.bitwiseAnd(1 << 0)
        .eq(0)
        .And(qa.bitwiseAnd(1 << 1).eq(0))
        .And(qa.bitwiseAnd(1 << 2).eq(0))
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
        .And(qa.bitwiseAnd(1 << 5).eq(0))
    )
    saturation_mask = image.select("QA_RADSAT").eq(0)
    return image.updateMask(mask).updateMask(saturation_mask)


def add_landsat_variables(image):
    """Add LST Celsius, NDVI, NDBI, and broad albedo proxy bands."""
    import ee

    optical = image.select("SR_B.*").multiply(0.0000275).add(-0.2)
    thermal_c = image.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15)

    ndvi = optical.normalizedDifference(["SR_B5", "SR_B4"]).rename("ndvi")
    ndbi = optical.normalizedDifference(["SR_B6", "SR_B5"]).rename("ndbi")
    albedo = (
        optical.select("SR_B2").multiply(0.356)
        .add(optical.select("SR_B4").multiply(0.130))
        .add(optical.select("SR_B5").multiply(0.373))
        .add(optical.select("SR_B6").multiply(0.085))
        .add(optical.select("SR_B7").multiply(0.072))
        .subtract(0.0018)
        .rename("albedo")
    )

    return image.addBands(
        [
            thermal_c.rename("lst_c"),
            ndvi,
            ndbi,
            albedo,
        ],
        overwrite=True,
    )


def summer_collection():
    """Build the 2014-2024 June-July-August Landsat 8/9 collection."""
    import ee

    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    collection = (
        l8.merge(l9)
        .filterDate("2014-06-01", "2024-09-01")
        .filter(ee.Filter.calendarRange(6, 8, "month"))
        .filter(ee.Filter.lt("CLOUD_COVER", 30))
        .map(qa_mask_landsat_l2)
        .map(add_landsat_variables)
    )
    return collection


def export_image_band(image, band: str, output: Path, region) -> None:
    """Export one image band to a local GeoTIFF."""
    import geemap

    output.parent.mkdir(parents=True, exist_ok=True)
    geemap.ee_export_image(
        image.select(band).float(),
        filename=str(output),
        scale=100,
        region=region,
        crs="EPSG:32635",
        file_per_band=False,
        unmask_value=-9999,
        timeout=900,
    )
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"Earth Engine export did not create a readable output for band `{band}`: {output}")


def export_summer_means() -> None:
    """Compute and export summer mean LST and index rasters."""
    import geemap

    require_file(BOUNDARY, "Study 1 Izmir boundary")
    require_file(REFERENCE_GRID, "LCZ reference grid")

    boundary = gpd.read_file(BOUNDARY).to_crs("EPSG:4326")
    region_fc = geemap.geopandas_to_ee(boundary)
    region = region_fc.geometry()

    mean_image = summer_collection().select(["lst_c", "ndvi", "ndbi", "albedo"]).mean().clip(region_fc)
    for band, path in [
        ("lst_c", LST_MEAN_TIF),
        ("ndvi", NDVI_MEAN_TIF),
        ("ndbi", NDBI_MEAN_TIF),
        ("albedo", ALBEDO_MEAN_TIF),
    ]:
        if path.exists() and path.stat().st_size > 0:
            continue
        export_image_band(mean_image, band, path, region)


def write_metadata() -> None:
    """Write a compact source note for the methods section."""
    METADATA_MD.parent.mkdir(parents=True, exist_ok=True)
    METADATA_MD.write_text(
        "\n".join(
            [
                "# Landsat Summer LST Source Metadata",
                "",
                f"- Earth Engine project: `{GEE_PROJECT}`",
                "- Collections: `LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2`.",
                "- Time window: 2014-06-01 to 2024-09-01.",
                "- Season: June, July, August.",
                "- Image-level cloud filter: `CLOUD_COVER < 30`.",
                "- Pixel QA mask: fill, dilated cloud, cirrus, cloud, cloud shadow, snow, and radiometric saturation removed.",
                "- LST conversion: `ST_B10 * 0.00341802 + 149.0 - 273.15`.",
                "- Optical reflectance conversion: `SR_B* * 0.0000275 - 0.2`.",
                "- Export CRS/resolution: `EPSG:32635`, 100 m.",
                "- Null-data rule: outside-boundary and zero-coded cells are treated as nodata in later alignment/model-table steps.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    logger = get_logger("p1_04_gee_lst_summer")
    logger.info("Exporting whole-Izmir Landsat summer LST and spectral index means.")
    initialise_earth_engine()
    export_summer_means()
    write_metadata()
    for label, path in [
        ("LST mean raster", LST_MEAN_TIF),
        ("NDVI mean raster", NDVI_MEAN_TIF),
        ("NDBI mean raster", NDBI_MEAN_TIF),
        ("albedo mean raster", ALBEDO_MEAN_TIF),
    ]:
        logger.info("%s status: exists=%s path=%s", label, path.exists(), path)
    logger.info("Wrote source metadata: %s", METADATA_MD)


if __name__ == "__main__":
    main()
