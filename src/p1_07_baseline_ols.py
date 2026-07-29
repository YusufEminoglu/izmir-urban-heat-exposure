"""Baseline OLS and VIF diagnostics for Study 1."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _config import DATA_PROCESSED, TABLES_DIR, ensure_dirs  # noqa: E402
from _config import get_logger  # noqa: E402


MODEL_GRID = DATA_PROCESSED / "paper1_model_grid_100m_current.csv"
OLS_SUMMARY_CSV = TABLES_DIR / "paper1_ols_model_summary.csv"
OLS_COEFS_CSV = TABLES_DIR / "paper1_ols_coefficients.csv"
VIF_CSV = TABLES_DIR / "paper1_vif_diagnostics.csv"
PREDICTOR_DECISIONS_MD = TABLES_DIR / "paper1_predictor_decisions.md"

DEPENDENT = "lst_2025"

PHYSICAL_PREDICTORS = [
    "ndvi_2025",
    "ndbi_2025",
    "albedo_2025",
    "tree_canopy_height_2025",
    "srtm_elevation_m",
    "srtm_slope_deg",
    "srtm_aspect_sin",
    "srtm_aspect_cos",
    "lcz_probability",
]

SOCIAL_PREDICTORS = [
    "pop_total",
    "female_share",
    "child_0_14_share",
    "elderly_65_plus_share",
]


def add_lcz_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Add LCZ fixed-effect dummies, dropping the most common class."""
    dummies = pd.get_dummies(df["lcz"].astype("int16"), prefix="lcz", dtype=float)
    if not dummies.empty:
        reference = dummies.sum().idxmax()
        dummies = dummies.drop(columns=reference)
    return pd.concat([df, dummies], axis=1)


def standardize_design(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Standardize continuous predictors while preserving dummy variables."""
    out = df.copy()
    continuous = [col for col in columns if not re.fullmatch(r"lcz_\d+", col)]
    scaler = StandardScaler()
    out[continuous] = scaler.fit_transform(out[continuous])
    return out


def fit_ols(df: pd.DataFrame, predictors: list[str], model_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit robust OLS and return summary, coefficients, and VIF tables."""
    needed = [DEPENDENT, "lcz"] + predictors
    working = df[needed].dropna().copy()
    working = add_lcz_dummies(working)
    lcz_columns = [col for col in working.columns if re.fullmatch(r"lcz_\d+", col)]
    x_columns = predictors + lcz_columns
    working = standardize_design(working, predictors)

    y = working[DEPENDENT].astype(float)
    x = sm.add_constant(working[x_columns].astype(float), has_constant="add")
    model = sm.OLS(y, x).fit(cov_type="HC3")

    coefs = pd.DataFrame(
        {
            "model": model_name,
            "term": model.params.index,
            "coef": model.params.values,
            "std_error_hc3": model.bse.values,
            "t": model.tvalues.values,
            "p_value": model.pvalues.values,
            "ci_low": model.conf_int()[0].values,
            "ci_high": model.conf_int()[1].values,
        }
    )
    summary = pd.DataFrame(
        [
            {
                "model": model_name,
                "n": int(model.nobs),
                "r2": model.rsquared,
                "adj_r2": model.rsquared_adj,
                "aic": model.aic,
                "bic": model.bic,
                "dependent": DEPENDENT,
                "predictors": json.dumps(x_columns, ensure_ascii=False),
            }
        ]
    )

    vif_x = working[x_columns].astype(float)
    vif = pd.DataFrame(
        {
            "model": model_name,
            "term": x_columns,
            "vif": [variance_inflation_factor(vif_x.values, i) for i in range(vif_x.shape[1])],
        }
    )
    return summary, coefs, vif


def write_predictor_decisions(vif: pd.DataFrame) -> None:
    """Document why the first model excludes some available variables."""
    high_vif = vif[vif["vif"] >= 10].sort_values(["model", "vif"], ascending=[True, False])
    lines = [
        "# Predictor Decisions",
        "",
        "- Active dependent variable: `lst_2025`.",
        "- `evi_2025` is available (stored under the prior SAVI filename), but is excluded from baseline OLS because it is highly collinear with `ndvi_2025` (VIF >> 10 when both are included).",
        "- `uhi_2025_diagnostic` and `utfvi_2025_diagnostic` are excluded from causal/predictive models because they are heat/LST-derived diagnostic layers, not independent predictors.",
        "- `corine_2018` and `glc_fcs_2022` are retained for descriptive/categorical interpretation; LCZ dummies are used as the primary categorical urban-form controls in OLS.",
        "- SRTM-derived elevation, slope, and aspect sine/cosine are included as terrain controls. They come from `izmir_srtm.tif`, not from DSM/SYM, so building height does not distort terrain.",
        "- `tree_canopy_height_2025` is included because the final TCH product now provides full interior coverage, with non-tree cells coded as `0 m` and export gaps removed from the study-area interior.",
        "- Population controls are included only in the heat-justice model because official ADNKS values are available only for matched mahalle cells.",
        "",
        "## VIF Watchlist",
        "",
    ]
    if high_vif.empty:
        lines.append("No model terms exceeded VIF >= 10.")
    else:
        for _, row in high_vif.iterrows():
            lines.append(f"- `{row['model']}` / `{row['term']}`: VIF {row['vif']:.2f}")
    PREDICTOR_DECISIONS_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    logger = get_logger("p1_07_baseline_ols")
    logger.info("Running baseline OLS and VIF diagnostics.")

    df = pd.read_csv(MODEL_GRID)
    physical_summary, physical_coefs, physical_vif = fit_ols(df, PHYSICAL_PREDICTORS, "physical_environmental")
    justice_summary, justice_coefs, justice_vif = fit_ols(
        df,
        PHYSICAL_PREDICTORS + SOCIAL_PREDICTORS,
        "heat_justice_with_population",
    )

    summary = pd.concat([physical_summary, justice_summary], ignore_index=True)
    coefs = pd.concat([physical_coefs, justice_coefs], ignore_index=True)
    vif = pd.concat([physical_vif, justice_vif], ignore_index=True)

    summary.to_csv(OLS_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    coefs.to_csv(OLS_COEFS_CSV, index=False, encoding="utf-8-sig")
    vif.to_csv(VIF_CSV, index=False, encoding="utf-8-sig")
    write_predictor_decisions(vif)

    logger.info("Wrote OLS summary: %s", OLS_SUMMARY_CSV)
    logger.info("Wrote OLS coefficients: %s", OLS_COEFS_CSV)
    logger.info("Wrote VIF diagnostics: %s", VIF_CSV)
    logger.info("Wrote predictor decision note: %s", PREDICTOR_DECISIONS_MD)


if __name__ == "__main__":
    main()
