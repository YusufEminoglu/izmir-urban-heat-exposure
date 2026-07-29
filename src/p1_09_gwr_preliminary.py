"""Preliminary spatially varying coefficient model for Study 1.

This script runs a tractable GWR on a spatially stratified sample. It is meant
as a high-quality preliminary spatial model for BBTMK and as a staging point
before any full MGWR run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _config import DATA_MODELS, DATA_PROCESSED, TABLES_DIR, ensure_dirs  # noqa: E402
from _config import get_logger  # noqa: E402


MODEL_GRID = DATA_PROCESSED / "paper1_model_grid_100m_current.csv"
OUTPUT_GPKG = DATA_MODELS / "paper1_gwr_preliminary_sample_5000.gpkg"
SUMMARY_CSV = TABLES_DIR / "paper1_gwr_preliminary_summary.csv"
COEFS_CSV = TABLES_DIR / "paper1_gwr_preliminary_coefficients_summary.csv"

DEPENDENT = "lst_2025"
PREDICTORS = [
    "ndvi_2025",
    "ndbi_2025",
    "albedo_2025",
]
SAMPLE_N = 5_000


def load_spatial_sample() -> pd.DataFrame:
    """Load a spatially stratified and LCZ-balanced sample."""
    usecols = ["x", "y", "lcz", DEPENDENT] + PREDICTORS
    df = pd.read_csv(MODEL_GRID, usecols=usecols).dropna().copy()
    df["tile_x"] = (df["x"] // 5000).astype(int)
    df["tile_y"] = (df["y"] // 5000).astype(int)
    df["stratum"] = df["lcz"].astype(str) + "_" + df["tile_x"].astype(str) + "_" + df["tile_y"].astype(str)
    sample = (
        df.groupby("stratum", group_keys=False)
        .apply(lambda group: group.sample(min(len(group), max(1, round(SAMPLE_N * len(group) / len(df)))), random_state=42), include_groups=False)
        .sample(n=min(SAMPLE_N, len(df)), random_state=42)
        .copy()
    )
    return sample


def main() -> None:
    ensure_dirs()
    logger = get_logger("p1_09_gwr_preliminary")
    logger.info("Running preliminary GWR on sample of up to %s cells.", SAMPLE_N)

    sample = load_spatial_sample()
    coords = sample[["x", "y"]].to_numpy()
    y = sample[[DEPENDENT]].to_numpy()

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x = x_scaler.fit_transform(sample[PREDICTORS].to_numpy())
    y_scaled = y_scaler.fit_transform(y)

    selector = Sel_BW(coords, y_scaled, x, kernel="bisquare", fixed=False, n_jobs=1)
    bandwidth = selector.search(criterion="AICc")
    model = GWR(coords, y_scaled, x, bw=bandwidth, kernel="bisquare", fixed=False, n_jobs=1)
    results = model.fit()

    params = results.params
    local = sample.reset_index(drop=True).copy()
    local["gwr_intercept"] = params[:, 0]
    for i, predictor in enumerate(PREDICTORS, start=1):
        local[f"gwr_coef_{predictor}"] = params[:, i]
    local["gwr_local_r2"] = results.localR2
    local["gwr_predicted_scaled"] = results.predy.flatten()
    local["gwr_residual_scaled"] = results.resid_response.flatten()

    gdf = gpd.GeoDataFrame(
        local,
        geometry=gpd.points_from_xy(local["x"], local["y"], crs="EPSG:32635"),
        crs="EPSG:32635",
    )
    OUTPUT_GPKG.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(OUTPUT_GPKG, layer="paper1_gwr_preliminary_sample_5000", driver="GPKG")

    summary = pd.DataFrame(
        [
            {
                "model": "preliminary_gwr_sample_5000",
                "n": len(sample),
                "bandwidth": bandwidth,
                "aicc": results.aicc,
                "bic": results.bic,
                "r2": results.R2,
                "adj_r2": results.adj_R2,
                "sigma2": results.sigma2,
                "predictors": ", ".join(PREDICTORS),
                "note": "Dependent and predictors standardized; adaptive bisquare GWR on spatially stratified sample.",
            }
        ]
    )
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    coef_rows = []
    for column in ["gwr_intercept"] + [f"gwr_coef_{predictor}" for predictor in PREDICTORS] + ["gwr_local_r2"]:
        values = local[column]
        coef_rows.append(
            {
                "column": column,
                "mean": values.mean(),
                "std": values.std(),
                "min": values.min(),
                "p05": values.quantile(0.05),
                "median": values.median(),
                "p95": values.quantile(0.95),
                "max": values.max(),
            }
        )
    pd.DataFrame(coef_rows).to_csv(COEFS_CSV, index=False, encoding="utf-8-sig")

    logger.info("Wrote preliminary GWR layer: %s", OUTPUT_GPKG)
    logger.info("Wrote GWR summary: %s", SUMMARY_CSV)
    logger.info("Wrote coefficient summary: %s", COEFS_CSV)


if __name__ == "__main__":
    main()
