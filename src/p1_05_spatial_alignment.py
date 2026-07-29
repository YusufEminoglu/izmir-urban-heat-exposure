"""Build the first Study 1 100 m model table from aligned local layers."""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import xy

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _config import DATA_INTERIM, DATA_PROCESSED, TABLES_DIR, ensure_dirs  # noqa: E402
from _config import get_logger  # noqa: E402


BOUNDARY = DATA_INTERIM / "izmir_study_boundary.gpkg"
LCZ = DATA_INTERIM / "lcz_filter_izmir_2018_100m.tif"
LCZ_PROBABILITY = DATA_INTERIM / "lcz_probability_izmir_2018_100m.tif"
POPULATION = DATA_PROCESSED / "mahalle_population_controls_2024.gpkg"
ALIGNED_COVARIATES_DIR = DATA_PROCESSED / "aligned_covariates"
LST_2025 = ALIGNED_COVARIATES_DIR / "izmir_LST_Veri_2025_aligned_100m.tif"
SRTM_ELEVATION = ALIGNED_COVARIATES_DIR / "izmir_srtm_elevation_aligned_100m.tif"

MODEL_TABLE_CSV = DATA_PROCESSED / "paper1_model_grid_100m_current.csv"
MGWR_SAMPLE_GPKG = DATA_PROCESSED / "paper1_mgwr_ready_sample_50k.gpkg"
REPORT_CSV = TABLES_DIR / "paper1_model_grid_report.csv"

POPULATION_COLUMNS = [
    "pop_total",
    "female_share",
    "child_0_14_share",
    "elderly_65_plus_share",
    "age_dependency_proxy",
]

CONTINUOUS_COVARIATES = {
    "albedo_2025": ALIGNED_COVARIATES_DIR / "izmir_Albedo_Veri_2025_aligned_100m.tif",
    "ndvi_2025": ALIGNED_COVARIATES_DIR / "izmir_NDVI_aligned_100m.tif",
    "ndbi_2025": ALIGNED_COVARIATES_DIR / "izmir_NDBI_aligned_100m.tif",
    # NOTE: The file name is kept as-is for continuity, but the content is EVI.
    "evi_2025": ALIGNED_COVARIATES_DIR / "izmir_SAVI_aligned_100m.tif",
    "tree_canopy_height_2025": ALIGNED_COVARIATES_DIR / "izmir_KanopiYuksekligi_Veri_aligned_100m.tif",
    "srtm_elevation_m": ALIGNED_COVARIATES_DIR / "izmir_srtm_elevation_aligned_100m.tif",
    "srtm_slope_deg": ALIGNED_COVARIATES_DIR / "izmir_srtm_slope_deg_aligned_100m.tif",
    "srtm_aspect_sin": ALIGNED_COVARIATES_DIR / "izmir_srtm_aspect_sin_aligned_100m.tif",
    "srtm_aspect_cos": ALIGNED_COVARIATES_DIR / "izmir_srtm_aspect_cos_aligned_100m.tif",
    "uhi_2025_diagnostic": ALIGNED_COVARIATES_DIR / "izmir_UHI_Veri_2025_aligned_100m.tif",
    "utfvi_2025_diagnostic": ALIGNED_COVARIATES_DIR / "izmir_UTFVI_Veri_2025_aligned_100m.tif",
}

CATEGORICAL_COVARIATES = {
    "corine_2018": ALIGNED_COVARIATES_DIR / "CORINE_izmir_2018_aligned_100m.tif",
    "glc_fcs_2022": ALIGNED_COVARIATES_DIR / "izmir_GLC_FCS_2022_aligned_100m.tif",
}


def read_band(path: Path) -> tuple[np.ndarray, rasterio.DatasetReader]:
    """Read a single-band raster as float64 plus its open dataset."""
    src = rasterio.open(path)
    array = src.read(1).astype("float64")
    nodata = src.nodata
    if nodata is not None:
        array[array == nodata] = np.nan
    return array, src


def rasterize_population(ref: rasterio.DatasetReader) -> dict[str, np.ndarray]:
    """Rasterize mahalle-level population controls to the reference grid."""
    pop = gpd.read_file(POPULATION).to_crs(ref.crs)
    rasters: dict[str, np.ndarray] = {}
    for column in POPULATION_COLUMNS:
        valid = pop[pop[column].notna()].copy()
        shapes = ((geom, float(value)) for geom, value in zip(valid.geometry, valid[column]))
        rasters[column] = rasterize(
            shapes=shapes,
            out_shape=(ref.height, ref.width),
            transform=ref.transform,
            fill=np.nan,
            dtype="float64",
        )
    return rasters


def read_optional_covariates() -> dict[str, np.ndarray]:
    """Read aligned covariate rasters if they exist."""
    covariates: dict[str, np.ndarray] = {}
    for name, path in {**CONTINUOUS_COVARIATES, **CATEGORICAL_COVARIATES}.items():
        if path.exists():
            array, src = read_band(path)
            src.close()
            covariates[name] = array
    return covariates


def build_model_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the current model table and a compact report."""
    lcz, ref = read_band(LCZ)
    lcz_probability, lcz_probability_src = read_band(LCZ_PROBABILITY)
    lst, lst_src = read_band(LST_2025)
    elevation, elevation_src = read_band(SRTM_ELEVATION)

    try:
        boundary = gpd.read_file(BOUNDARY).to_crs(ref.crs)
        inside = geometry_mask(
            boundary.geometry,
            out_shape=(ref.height, ref.width),
            transform=ref.transform,
            invert=True,
        )

        population = rasterize_population(ref)
        covariates = read_optional_covariates()

        # Valid rows are defined by the *core* layers only.
        #
        # Important: optional covariates can still contain export-edge gaps or dataset-specific
        # missingness within the study boundary. Requiring every optional covariate to be finite
        # here would drop large contiguous areas from the model grid and downstream hex maps,
        # making figures appear blank even when the core analytical layers are valid.
        #
        # We therefore keep cells where the core layers are available and carry optional
        # covariates as NaN when missing. Each downstream model can then decide which columns
        # are required via its own subset logic without silently fabricating values.
        valid = (
            inside
            & np.isfinite(lcz)
            & np.isfinite(lcz_probability)
            & np.isfinite(lst)
            & np.isfinite(elevation)
            & (lst != 0)
            & (elevation != 0)
        )

        rows, cols = np.where(valid)
        xs, ys = xy(ref.transform, rows, cols, offset="center")
        data = {
            "row": rows,
            "col": cols,
            "x": np.asarray(xs),
            "y": np.asarray(ys),
            "lst_2025": lst[rows, cols],
            "lcz": lcz[rows, cols].astype("int16"),
            "lcz_probability": lcz_probability[rows, cols],
            "srtm_elevation_m": elevation[rows, cols],
        }
        for column, array in population.items():
            data[column] = array[rows, cols]
        for column, array in covariates.items():
            if column == "srtm_elevation_m":
                # SRTM elevation is already included as a core column.
                continue
            values = array[rows, cols]
            if column in CATEGORICAL_COVARIATES:
                # Preserve missing category values; downstream models may impute or drop.
                values = values.astype("float64")
            data[column] = values
        df = pd.DataFrame(data)

        report = pd.DataFrame(
            [
                {"metric": "total_grid_cells", "value": int(ref.width * ref.height)},
                {"metric": "inside_boundary_cells", "value": int(inside.sum())},
                {"metric": "valid_model_cells", "value": int(len(df))},
                {"metric": "lcz_zero_cells_total", "value": int((lcz == 0).sum())},
                {"metric": "lst_zero_cells_total", "value": int((lst == 0).sum())},
                {"metric": "srtm_elevation_zero_cells_total", "value": int((elevation == 0).sum())},
                {"metric": "population_matched_valid_cells", "value": int(df["pop_total"].notna().sum())},
                {"metric": "optional_covariates_loaded", "value": len(covariates)},
            ]
        )
        return df, report
    finally:
        ref.close()
        lcz_probability_src.close()
        lst_src.close()
        elevation_src.close()


def write_outputs(df: pd.DataFrame, report: pd.DataFrame) -> None:
    """Write full CSV table and a stratified spatial MGWR sample."""
    MODEL_TABLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MODEL_TABLE_CSV, index=False, encoding="utf-8-sig")

    sample_n = min(50_000, len(df))
    sample = (
        df.groupby("lcz", group_keys=False)
        .apply(
            lambda group: group.sample(max(1, round(sample_n * len(group) / len(df))), random_state=42),
            include_groups=False,
        )
        .head(sample_n)
        .copy()
    )
    sample["lcz"] = df.loc[sample.index, "lcz"].to_numpy()
    gdf = gpd.GeoDataFrame(
        sample,
        geometry=gpd.points_from_xy(sample["x"], sample["y"], crs="EPSG:32635"),
        crs="EPSG:32635",
    )
    MGWR_SAMPLE_GPKG.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(MGWR_SAMPLE_GPKG, layer="paper1_mgwr_ready_sample_50k", driver="GPKG")

    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")


def main() -> None:
    ensure_dirs()
    logger = get_logger("p1_05_spatial_alignment")
    logger.info("Building current Study 1 100 m model table.")
    df, report = build_model_table()
    write_outputs(df, report)
    logger.info("Wrote current full model table CSV: %s", MODEL_TABLE_CSV)
    logger.info("Wrote stratified MGWR sample: %s", MGWR_SAMPLE_GPKG)
    logger.info("Wrote model-grid report: %s", REPORT_CSV)


if __name__ == "__main__":
    main()
