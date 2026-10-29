"""Generate annual 2014-2024 summer Landsat LST rasters and a NetCDF cube.

This script exists to support the stronger abstract/manuscript claim that
multi-year summer Landsat observations were organised as a spatial data cube.
It exports one JJA mean LST GeoTIFF per year from Earth Engine, then stacks the
available rasters into an xarray/NetCDF cube.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p1_04_gee_lst_summer import add_landsat_variables, qa_mask_landsat_l2  # noqa: E402
from _config import GEE_PROJECT, DATA_INTERIM, DATA_PROCESSED, TABLES_DIR, ensure_dirs  # noqa: E402
from _config import get_logger  # noqa: E402


BOUNDARY = DATA_INTERIM / "izmir_study_boundary.gpkg"
ANNUAL_DIR = DATA_INTERIM / "annual_lst_2014_2024"
CUBE = DATA_PROCESSED / "datacubes" / "paper1_landsat_lst_jja_cube_2014_2024.nc"
REPORT = TABLES_DIR / "paper1_landsat_lst_jja_cube_inventory.csv"
YEARS = list(range(2014, 2025))


def annual_collection(year: int):
    """Build a JJA Landsat 8/9 collection for one year."""
    import ee

    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
    return (
        l8.merge(l9)
        .filterDate(f"{year}-06-01", f"{year}-09-01")
        .filter(ee.Filter.lt("CLOUD_COVER", 30))
        .map(qa_mask_landsat_l2)
        .map(add_landsat_variables)
    )


def export_annual_rasters() -> None:
    """Export missing annual JJA mean LST rasters from Earth Engine."""
    import ee
    import geemap

    ee.Initialize(project=GEE_PROJECT)
    boundary = gpd.read_file(BOUNDARY).to_crs("EPSG:4326")
    region_fc = geemap.geopandas_to_ee(boundary)
    region = region_fc.geometry()
    ANNUAL_DIR.mkdir(parents=True, exist_ok=True)

    for year in YEARS:
        output = ANNUAL_DIR / f"lst_jja_mean_{year}_100m.tif"
        if output.exists() and output.stat().st_size > 0:
            continue
        image = annual_collection(year).select("lst_c").mean().clip(region_fc).float()
        geemap.ee_export_image(
            image,
            filename=str(output),
            scale=100,
            region=region,
            crs="EPSG:32635",
            file_per_band=False,
            unmask_value=-9999,
            timeout=900,
        )
        if not output.exists() or output.stat().st_size == 0:
            raise RuntimeError(f"Annual LST export failed for {year}: {output}")


def coords_from_raster(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Return x/y coordinate arrays and metadata from a raster."""
    with rasterio.open(path) as src:
        xs = src.transform.c + (np.arange(src.width) + 0.5) * src.transform.a
        ys = src.transform.f + (np.arange(src.height) + 0.5) * src.transform.e
        meta = {"crs": src.crs.to_string() if src.crs else None, "width": src.width, "height": src.height}
    return xs, ys, meta


def build_cube_from_available_rasters() -> None:
    """Stack available annual rasters into a NetCDF cube."""
    paths = [ANNUAL_DIR / f"lst_jja_mean_{year}_100m.tif" for year in YEARS]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing annual LST rasters: {missing}")

    x, y, meta = coords_from_raster(paths[0])
    arrays = []
    for path in paths:
        with rasterio.open(path) as src:
            arr = src.read(1).astype("float32")
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            arr[arr <= -9990] = np.nan
            arr[(arr < -50) | (arr > 90)] = np.nan
            arr[arr == 0] = np.nan
            arrays.append(arr)
    stack = np.stack(arrays)
    ds = xr.Dataset(
        data_vars={"lst_jja": (("time", "y", "x"), stack, {"units": "degC", "nodata": "NaN"})},
        coords={"time": pd.to_datetime([f"{year}-07-15" for year in YEARS]), "y": y, "x": x},
    )
    ds.attrs.update(
        {
            "title": "Study 1 annual summer Landsat LST data cube",
            "crs": meta["crs"],
            "workflow": "Earth Engine annual JJA LST exports stacked with xarray",
            "years": ", ".join(map(str, YEARS)),
            "study_area": "whole Izmir province",
        }
    )
    CUBE.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(CUBE, engine="netcdf4", encoding={"lst_jja": {"zlib": True, "complevel": 4}})

    pd.DataFrame(
        [
            {
                "cube": str(CUBE),
                "years": ",".join(map(str, YEARS)),
                "time_steps": len(YEARS),
                "width": meta["width"],
                "height": meta["height"],
                "crs": meta["crs"],
            }
        ]
    ).to_csv(REPORT, index=False, encoding="utf-8-sig")


def main() -> None:
    ensure_dirs()
    logger = get_logger("p1_11_gee_annual_lst_cube")
    logger.info("Exporting annual Landsat JJA LST rasters and building NetCDF cube.")
    export_annual_rasters()
    build_cube_from_available_rasters()
    logger.info("Wrote annual LST cube: %s", CUBE)
    logger.info("Wrote annual LST cube report: %s", REPORT)


if __name__ == "__main__":
    main()
