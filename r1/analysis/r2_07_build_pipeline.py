"""Rebuild the paper-1 data tree on the complete province grid, so the original pipeline scripts
(p1_06, p1_07, p1_15) can be rerun unchanged against it.

pipeline/ mirrors the project layout (src/, data/, outputs/, docs/manuscripts/paper1/src/mydata.dat).
"""
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

ROOTB = Path(r"REPO_ROOT")
P = ROOT / "pipeline"
NR, NC = 1744, 1981
T = from_origin(432600.0, 4359700.0, 100.0, 100.0)
interim = P / "data" / "02_interim" / "paper1"
proc = P / "data" / "03_processed" / "paper1"
al = proc / "aligned_covariates"
for d in [interim, al, proc / "datacubes", P / "data" / "04_models" / "paper1", P / "data" / "00_external" / "akilli_sehir_db",
          P / "outputs" / "tables" / "paper1", P / "outputs" / "figures" / "paper1" / "final", P / "outputs" / "figures" / "paper1" / "appendix",
          P / "docs" / "manuscripts" / "paper1" / "src", P / "logs"]:
    d.mkdir(parents=True, exist_ok=True)
if (P / "src").exists():
    shutil.rmtree(P / "src")
shutil.copytree(ROOTB / "src", P / "src", ignore=shutil.ignore_patterns("__pycache__"))

mask = np.load(CACHE / "province_mask.npy")
full = pd.read_csv(GRID_CSV)  # v2 grid (r2_27)
gee = {p.stem: np.load(p) for p in (CACHE / "gee").glob("*.npy")}


def to_raster(col):
    a = np.full((NR, NC), np.nan, np.float32)
    a[full.row.to_numpy(), full.col.to_numpy()] = full[col].to_numpy(np.float32)
    return a


def write(path, arr, nodata, dtype="float32"):
    a = np.where(mask & np.isfinite(arr), arr, nodata).astype(dtype)
    with rasterio.open(path, "w", driver="GTiff", height=NR, width=NC, count=1, dtype=dtype, crs=CRS, transform=T,
                       nodata=nodata, compress="deflate") as dst:
        dst.write(a, 1)


cont = {"izmir_LST_Veri_2025_aligned_100m.tif": "lst_2025", "izmir_NDVI_aligned_100m.tif": "ndvi_2025",
        "izmir_NDBI_aligned_100m.tif": "ndbi_2025", "izmir_SAVI_aligned_100m.tif": "evi_2025",
        "izmir_Albedo_Veri_2025_aligned_100m.tif": "albedo_2025", "izmir_UHI_Veri_2025_aligned_100m.tif": "uhi_2025_diagnostic",
        "izmir_UTFVI_Veri_2025_aligned_100m.tif": "utfvi_2025_diagnostic", "izmir_KanopiYuksekligi_Veri_aligned_100m.tif": "tree_canopy_height_2025",
        "izmir_srtm_elevation_aligned_100m.tif": "srtm_elevation_m", "izmir_srtm_aligned_100m.tif": "srtm_elevation_m",
        "izmir_srtm_slope_deg_aligned_100m.tif": "srtm_slope_deg", "izmir_srtm_aspect_sin_aligned_100m.tif": "srtm_aspect_sin",
        "izmir_srtm_aspect_cos_aligned_100m.tif": "srtm_aspect_cos"}
for fn, col in cont.items():
    write(al / fn, to_raster(col), -9999.0)
write(al / "CORINE_izmir_2018_aligned_100m.tif", to_raster("corine_2018"), -32768, "int16")
write(al / "izmir_GLC_FCS_2022_aligned_100m.tif", to_raster("glc_fcs_2022"), -32768, "int16")
write(interim / "lcz_filter_izmir_2018_100m.tif", to_raster("lcz"), -2147483648, "int32")
write(interim / "lcz_probability_izmir_2018_100m.tif", to_raster("lcz_probability"), -2147483648, "int32")

# boundary: dissolved districts, holes removed (identical to the map boundary of Figure 4)
b = gpd.read_file(DATA / "izmir_province_boundary_filled.gpkg")
b.to_file(interim / "izmir_study_boundary.gpkg", driver="GPKG")
ilce_src = Path(r"EXTERNAL_DATA_DIR\building_inventory_project\data\00_external\boundaries")
for f in ilce_src.glob("ilce.*"):
    shutil.copy2(f, P / "data" / "00_external" / "akilli_sehir_db" / f.name)
shutil.copy2(MAHALLE, proc / "mahalle_population_controls_2024.gpkg")
full.drop(columns=["gapfilled", "nearest_fill"], errors="ignore").to_csv(proc / "paper1_model_grid_100m_current.csv", index=False)

# datacubes: complete 2025 cube, annual cube unchanged
with xr.open_dataset(CUBE_2025) as c25:
    ds = c25.load().copy()
src_col = {"lst_2025": "lst_2025", "ndvi_2025": "ndvi_2025", "ndbi_2025": "ndbi_2025", "evi_2025": "evi_2025", "albedo_2025": "albedo_2025",
           "uhi_2025": "uhi_2025_diagnostic", "utfvi_2025": "utfvi_2025_diagnostic", "tree_canopy_height_2025": "tree_canopy_height_2025"}
for k, col in src_col.items():
    ds[k].values[:] = np.where(mask, to_raster(col), np.nan)
    if k != "tree_canopy_height_2025":
        ds[k].attrs["source"] = ("Earth Engine export of the documented June-August 2025 Landsat composite (r2_05, r2_27)"
                                 if k not in ("uhi_2025", "utfvi_2025") else "derived from lst_2025 (r2_27)")
ds.attrs["coverage"] = "complete province grid"
ds.attrs["source_2025_landsat"] = ("June-August 2025 Landsat 8/9 Collection 2 Level-2 per-pixel mean (scene cloud cover < 10%, QA_PIXEL bits 0-5 "
                                   "and QA_RADSAT masked; LST from ST_B10; NDVI, NDBI, EVI and Liang albedo from surface reflectance), "
                                   "Earth Engine export on the analysis grid (scripts r2_05, r2_27)")
ds.to_netcdf(proc / "datacubes" / "paper1_thermal_covariate_cube_2025.nc", engine="netcdf4",
             encoding={k: {"zlib": True, "complevel": 4} for k in ds.data_vars})
shutil.copy2(CUBE_ANNUAL, proc / "datacubes" / CUBE_ANNUAL.name)
for f in (DATA / "datacubes").glob("*glc*") :
    shutil.copy2(f, proc / "datacubes" / f.name)
for f in (DATA / "datacubes").glob("*corine*"):
    shutil.copy2(f, proc / "datacubes" / f.name)
shutil.copy2(ROOTB / "docs" / "manuscripts" / "paper1" / "src" / "mydata.dat", P / "docs" / "manuscripts" / "paper1" / "src" / "mydata.dat")
print("pipeline tree at", P, "grid rows", len(full))
