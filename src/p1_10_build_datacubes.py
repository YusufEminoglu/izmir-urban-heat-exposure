"""Build xarray/NetCDF data cubes for the Study 1 cloud-local pipeline."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _config import DATA_PROCESSED, TABLES_DIR, ensure_dirs  # noqa: E402
from _config import get_logger  # noqa: E402


ALIGNED = DATA_PROCESSED / "aligned_covariates"
DATACUBE_DIR = DATA_PROCESSED / "datacubes"
THERMAL_CUBE = DATACUBE_DIR / "paper1_thermal_covariate_cube_2025.nc"
GLC_CUBE = DATACUBE_DIR / "paper1_glc_fcs_time_cube_1985_2022.nc"
CORINE_CUBE = DATACUBE_DIR / "paper1_corine_time_cube_1990_2018.nc"
REPORT_CSV = TABLES_DIR / "paper1_datacube_inventory.csv"


THERMAL_VARIABLES = {
    "lst_2025": ALIGNED / "izmir_LST_Veri_2025_aligned_100m.tif",
    "ndvi_2025": ALIGNED / "izmir_NDVI_aligned_100m.tif",
    "ndbi_2025": ALIGNED / "izmir_NDBI_aligned_100m.tif",
    # NOTE: file name kept for continuity; content is EVI.
    "evi_2025": ALIGNED / "izmir_SAVI_aligned_100m.tif",
    "albedo_2025": ALIGNED / "izmir_Albedo_Veri_2025_aligned_100m.tif",
    "uhi_2025": ALIGNED / "izmir_UHI_Veri_2025_aligned_100m.tif",
    "utfvi_2025": ALIGNED / "izmir_UTFVI_Veri_2025_aligned_100m.tif",
    "tree_canopy_height_2025": ALIGNED / "izmir_KanopiYuksekligi_Veri_aligned_100m.tif",
}


def coords_from_raster(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Return x/y coordinate arrays and grid metadata from a raster."""
    with rasterio.open(path) as src:
        xs = src.transform.c + (np.arange(src.width) + 0.5) * src.transform.a
        ys = src.transform.f + (np.arange(src.height) + 0.5) * src.transform.e
        meta = {
            "crs": src.crs.to_string() if src.crs else None,
            "transform": tuple(src.transform),
            "resolution_x": src.res[0],
            "resolution_y": src.res[1],
            "width": src.width,
            "height": src.height,
            "bounds": tuple(src.bounds),
        }
    return xs, ys, meta


def read_array(path: Path) -> np.ndarray:
    """Read raster band 1 as float32 with nodata converted to NaN."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan
    return arr


def write_thermal_cube() -> dict[str, object]:
    """Write the 2025 thermal/environmental covariate cube."""
    ref = next(iter(THERMAL_VARIABLES.values()))
    x, y, meta = coords_from_raster(ref)
    data_vars = {}
    for name, path in THERMAL_VARIABLES.items():
        arr = read_array(path)
        data_vars[name] = (
            ("y", "x"),
            arr,
            {
                "source": str(path),
                "grid": "100 m LCZ-aligned EPSG:32635",
                "nodata": "NaN",
            },
        )

    ds = xr.Dataset(data_vars=data_vars, coords={"y": y, "x": x})
    ds.attrs.update(
        {
            "title": "Study 1 2025 thermal and environmental covariate data cube",
            "crs": meta["crs"],
            "workflow": "cloud-local LCZ-aligned xarray/NetCDF cube",
            "study_area": "whole Izmir province",
            "null_data_rule": "outside-boundary and zero-coded outside cells treated as nodata during alignment",
        }
    )
    THERMAL_CUBE.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(THERMAL_CUBE, engine="netcdf4", encoding={name: {"zlib": True, "complevel": 4} for name in ds.data_vars})
    return {"cube": str(THERMAL_CUBE), "variables": len(ds.data_vars), **meta}


def year_from_name(path: Path) -> int:
    """Extract a four-digit year from a filename."""
    match = re.search(r"(19|20)\d{2}", path.name)
    if not match:
        raise ValueError(f"No year found in {path.name}")
    return int(match.group(0))


def write_time_cube(paths: list[Path], variable: str, output: Path, title: str) -> dict[str, object]:
    """Write a categorical time-series raster cube."""
    paths = sorted(paths, key=year_from_name)
    x, y, meta = coords_from_raster(paths[0])
    years = [year_from_name(path) for path in paths]
    stack = []
    for path in paths:
        arr = read_array(path)
        stack.append(arr)
    values = np.stack(stack).astype("float32")

    ds = xr.Dataset(
        data_vars={
            variable: (
                ("time", "y", "x"),
                values,
                {"source_files": "; ".join(str(path) for path in paths), "nodata": "NaN"},
            )
        },
        coords={"time": pd.to_datetime([f"{year}-01-01" for year in years]), "y": y, "x": x},
    )
    ds.attrs.update(
        {
            "title": title,
            "crs": meta["crs"],
            "workflow": "LCZ-aligned xarray/NetCDF categorical time cube",
            "study_area": "whole Izmir province",
            "years": ", ".join(map(str, years)),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(output, engine="netcdf4", encoding={variable: {"zlib": True, "complevel": 4}})
    return {"cube": str(output), "variables": 1, "time_steps": len(years), "years": ",".join(map(str, years)), **meta}


def main() -> None:
    ensure_dirs()
    logger = get_logger("p1_10_build_datacubes")
    logger.info("Building Study 1 xarray/NetCDF data cubes.")

    rows = [write_thermal_cube()]
    rows.append(
        write_time_cube(
            sorted(ALIGNED.glob("izmir_GLC_FCS_*_aligned_100m.tif")),
            "glc_fcs_class",
            GLC_CUBE,
            "Study 1 GLC_FCS categorical land-cover time cube",
        )
    )
    rows.append(
        write_time_cube(
            sorted(ALIGNED.glob("CORINE_izmir_*_aligned_100m.tif")),
            "corine_class",
            CORINE_CUBE,
            "Study 1 CORINE categorical land-cover time cube",
        )
    )

    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")
    logger.info("Wrote thermal cube: %s", THERMAL_CUBE)
    logger.info("Wrote GLC_FCS time cube: %s", GLC_CUBE)
    logger.info("Wrote CORINE time cube: %s", CORINE_CUBE)
    logger.info("Wrote cube inventory: %s", REPORT_CSV)


if __name__ == "__main__":
    main()
