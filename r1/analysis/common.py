"""Shared paths and loaders for the Cities R1 revision analyses (paper 1, Izmir).

Every analysis script writes its numbers to outputs/values/<script>.csv as
key,value rows; build_mydata.py merges them into the manuscript's mydata.dat.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
VALUES = OUT / "values"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
CACHE = OUT / "cache"
for _p in (OUT, VALUES, TABLES, FIGS, CACHE):
    _p.mkdir(parents=True, exist_ok=True)

GRID_CSV = DATA / "paper1_model_grid_100m_v2.csv"  # v2: 2025 Landsat layers from the documented export (r2_27)
GRID_CSV_V1 = DATA / "paper1_model_grid_100m_complete.csv"  # v1: alternative 2025 layers, completed grid
GRID_CSV_R0 = DATA / "paper1_model_grid_100m_current.csv"
MGWR_SAMPLE = DATA / "paper1_mgwr_sample_v2.gpkg"  # same 4,000 calibration cells, v2 layers and the v2 reported fit (r2_28)
MGWR_SAMPLE_V1 = DATA / "mgwr_run" / "paper1_mgwr_sample_true.gpkg"
IDW_GRID = DATA / "paper1_mgwr_coefficient_grid_idw_v2.gpkg"  # v2 atlas (r2_29)
IDW_GRID_V1 = DATA / "mgwr_run" / "paper1_mgwr_coefficient_grid_idw.gpkg"
CUBE_ANNUAL = DATA / "datacubes" / "paper1_landsat_lst_jja_cube_2014_2024.nc"  # 2014-2024 cube; analyses use load_summer_series()
CUBE_2025 = ROOT / "pipeline" / "data" / "03_processed" / "paper1" / "datacubes" / "paper1_thermal_covariate_cube_2025.nc"
CUBE_2025_R0 = DATA / "datacubes" / "paper1_thermal_covariate_cube_2025.nc"

PYPROJ = Path(r"EXTERNAL_DATA_DIR")
BUILDINGS = PYPROJ / "building_inventory_project" / "data" / "01_raw" / "buildings" / "izmir_buildings_bbb.gpkg"
MAHALLE = PYPROJ / "DM" / "data" / "01_raw" / "nufus" / "mahalle_population_controls_2024.gpkg"
PRIORITY_CSV_R0 = Path(r"REPO_ROOT\outputs\tables\paper1\paper1_mahalle_heat_justice_priority.csv")
PRIORITY_CSV = ROOT / "outputs" / "tables" / "mahalle_components_complete.csv"  # recomputed on the complete grid (r2_09)

CRS = "EPSG:32635"
DEP = "lst_2025"
PREDICTORS = ["ndvi_2025", "ndbi_2025", "albedo_2025"]
OLS_CONT = ["ndvi_2025", "ndbi_2025", "albedo_2025", "tree_canopy_height_2025",
            "srtm_elevation_m", "srtm_slope_deg", "srtm_aspect_sin", "srtm_aspect_cos",
            "lcz_probability"]
MODEL_COLS = [DEP, "lcz"] + OLS_CONT

# WUDAPT/RUBCLIM integer codes -> standard LCZ labels
LCZ_LABEL = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
             10: "10", 11: "A", 12: "B", 13: "C", 14: "D", 15: "E", 16: "F", 17: "G"}
LCZ_NAME = {0: "Unclassified", 1: "Compact high-rise", 2: "Compact mid-rise", 3: "Compact low-rise",
            4: "Open high-rise", 5: "Open mid-rise", 6: "Open low-rise", 7: "Lightweight low-rise",
            8: "Large low-rise", 9: "Sparsely built", 10: "Heavy industry", 11: "Dense trees",
            12: "Scattered trees", 13: "Bush, scrub", 14: "Low plants", 15: "Bare rock or paved",
            16: "Bare soil or sand", 17: "Water"}
BUILT = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
NATURAL = [11, 12, 13, 14, 15, 16, 17]


def load_grid(valid_only: bool = True) -> pd.DataFrame:
    cache = CACHE / "grid.pkl"
    if cache.exists():
        df = pd.read_pickle(cache)
    else:
        df = pd.read_csv(GRID_CSV, encoding="utf-8-sig")
        df.to_pickle(cache)
    if valid_only:
        df = df.dropna(subset=MODEL_COLS).reset_index(drop=True)
        df["lcz"] = df["lcz"].astype(int)
    return df


def write_values(name: str, values: dict) -> None:
    """Write key,value rows (strings kept verbatim) for mydata.dat."""
    path = VALUES / f"{name}.csv"
    with open(path, "w", encoding="utf-8") as f:
        for k, v in values.items():
            f.write(f"{k},{v}\n")
    print(f"[values] {len(values)} keys -> {path.name}")


def fmt(x: float, nd: int = 3) -> str:
    """Format with fixed decimals and a real minus sign handled by LaTeX math."""
    return f"{x:.{nd}f}"


def fmt_int(x: float) -> str:
    return f"{int(round(x)):,}".replace(",", "{,}")


def limit_blas(n: int) -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = str(n)


def residential_mask(df: pd.DataFrame) -> pd.Series:
    """Residential cells: cells with at least one dwelling (inventory use MESKEN); a mahalle without any
    dwelling cell falls back to its cells with any building. Needs columns join_key, bsf, bsf_res."""
    m = df.bsf_res.fillna(0) > 0
    has = set(df.loc[m, "join_key"].dropna())
    return m | (df.join_key.notna() & ~df.join_key.isin(has) & (df.bsf.fillna(0) > 0))


CUBE_ANNUAL_2026 = DATA / "datacubes" / "paper1_landsat_lst_jja_cube_2014_2026.nc"  # r2_40: cube extended with 2025, 2026


def load_summer_series():
    """The 13 summers 2014-2026 on the analysis grid: slices of the annual cube (p1_11 method) with the 2025 slice
    replaced by the analysed 2025 composite. Returns (years, float array [T, rows, cols])."""
    import xarray as xr
    ds = xr.open_dataset(CUBE_ANNUAL_2026)
    years = pd.to_datetime(ds.time.values).year.tolist()
    a = ds["lst_jja"].values.astype(float)
    ds.close()
    g = load_grid(valid_only=False)
    l25 = np.full(a.shape[1:], np.nan)
    l25[g.row.to_numpy(int), g.col.to_numpy(int)] = g[DEP].to_numpy(float)
    a[years.index(2025)] = l25
    return years, a
