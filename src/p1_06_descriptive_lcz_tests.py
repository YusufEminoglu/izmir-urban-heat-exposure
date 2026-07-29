"""Descriptive statistics and LCZ-LST tests for Study 1."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import false_discovery_control

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _config import DATA_PROCESSED, TABLES_DIR, ensure_dirs  # noqa: E402
from _config import get_logger  # noqa: E402


MODEL_GRID = DATA_PROCESSED / "paper1_model_grid_100m_current.csv"
SUMMARY_CSV = TABLES_DIR / "paper1_variable_summary.csv"
LCZ_SUMMARY_CSV = TABLES_DIR / "paper1_lcz_lst_summary.csv"
CORRELATION_CSV = TABLES_DIR / "paper1_predictor_correlations.csv"
LCZ_TEST_CSV = TABLES_DIR / "paper1_lcz_lst_stat_tests.csv"

DEPENDENT = "lst_2025"
NUMERIC_COLUMNS = [
    "lst_2025",
    "lcz_probability",
    "pop_total",
    "female_share",
    "child_0_14_share",
    "elderly_65_plus_share",
    "age_dependency_proxy",
    "albedo_2025",
    "ndvi_2025",
    "ndbi_2025",
    "evi_2025",
    "tree_canopy_height_2025",
    "srtm_elevation_m",
    "srtm_slope_deg",
    "srtm_aspect_sin",
    "srtm_aspect_cos",
    "uhi_2025_diagnostic",
    "utfvi_2025_diagnostic",
]

PREDICTOR_COLUMNS = [
    "lcz_probability",
    "pop_total",
    "female_share",
    "child_0_14_share",
    "elderly_65_plus_share",
    "age_dependency_proxy",
    "albedo_2025",
    "ndvi_2025",
    "ndbi_2025",
    "evi_2025",
    "tree_canopy_height_2025",
    "srtm_elevation_m",
    "srtm_slope_deg",
    "srtm_aspect_sin",
    "srtm_aspect_cos",
]


def eta_squared_kruskal(groups: list[np.ndarray], h_statistic: float) -> float:
    """Approximate eta-squared for Kruskal-Wallis."""
    n = sum(len(group) for group in groups)
    k = len(groups)
    if n <= k:
        return np.nan
    return (h_statistic - k + 1) / (n - k)


def pairwise_mannwhitney(groups: dict[int, np.ndarray]) -> pd.DataFrame:
    """Run pairwise Mann-Whitney tests between LCZ groups with FDR correction."""
    rows = []
    keys = sorted(groups)
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            left_values = groups[left]
            right_values = groups[right]
            stat, p_value = stats.mannwhitneyu(left_values, right_values, alternative="two-sided")
            rows.append(
                {
                    "test": "pairwise_mannwhitney",
                    "lcz_a": left,
                    "lcz_b": right,
                    "statistic": stat,
                    "p_value": p_value,
                    "median_diff_a_minus_b": np.median(left_values) - np.median(right_values),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_value_fdr"] = false_discovery_control(result["p_value"].to_numpy(), method="bh")
    return result


def main() -> None:
    ensure_dirs()
    logger = get_logger("p1_06_descriptive_lcz_tests")
    logger.info("Running Study 1 descriptive statistics and LCZ-LST tests.")

    df = pd.read_csv(MODEL_GRID)

    summary = df[NUMERIC_COLUMNS].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    summary["missing"] = df[NUMERIC_COLUMNS].isna().sum()
    summary["missing_share"] = summary["missing"] / len(df)
    summary.to_csv(SUMMARY_CSV, encoding="utf-8-sig")

    lcz_summary = (
        df.groupby("lcz")[DEPENDENT]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
        .sort_values("lcz")
    )
    lcz_summary.to_csv(LCZ_SUMMARY_CSV, index=False, encoding="utf-8-sig")

    correlation = df[[DEPENDENT] + PREDICTOR_COLUMNS].corr(method="spearman")
    correlation.to_csv(CORRELATION_CSV, encoding="utf-8-sig")

    grouped = {
        int(lcz): group[DEPENDENT].dropna().to_numpy()
        for lcz, group in df.groupby("lcz")
        if len(group[DEPENDENT].dropna()) >= 30
    }
    group_arrays = list(grouped.values())
    h_stat, p_value = stats.kruskal(*group_arrays)
    omnibus = pd.DataFrame(
        [
            {
                "test": "kruskal_wallis",
                "lcz_a": np.nan,
                "lcz_b": np.nan,
                "statistic": h_stat,
                "p_value": p_value,
                "p_value_fdr": p_value,
                "median_diff_a_minus_b": np.nan,
                "eta_squared_approx": eta_squared_kruskal(group_arrays, h_stat),
                "groups": len(group_arrays),
                "n": sum(len(group) for group in group_arrays),
            }
        ]
    )
    pairwise = pairwise_mannwhitney(grouped)
    tests = pd.concat([omnibus, pairwise], ignore_index=True, sort=False)
    tests.to_csv(LCZ_TEST_CSV, index=False, encoding="utf-8-sig")

    logger.info("Wrote variable summary: %s", SUMMARY_CSV)
    logger.info("Wrote LCZ-LST summary: %s", LCZ_SUMMARY_CSV)
    logger.info("Wrote Spearman correlations: %s", CORRELATION_CSV)
    logger.info("Wrote LCZ-LST tests: %s", LCZ_TEST_CSV)


if __name__ == "__main__":
    main()
