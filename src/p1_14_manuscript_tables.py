"""Curate Study 1 manuscript-ready table drafts from analysis outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _config import TABLES_DIR, ensure_dirs  # noqa: E402
from _config import get_logger  # noqa: E402


MANUSCRIPT_DIR = TABLES_DIR / "manuscript"

TABLE1 = MANUSCRIPT_DIR / "table1_data_source_matrix.csv"
TABLE2 = MANUSCRIPT_DIR / "table2_variable_definitions.csv"
TABLE3 = MANUSCRIPT_DIR / "table3_model_specifications.csv"
TABLE4 = MANUSCRIPT_DIR / "table4_lcz_lst_summary_and_tests.csv"
TABLE5 = MANUSCRIPT_DIR / "table5_ols_summary_vif.csv"
TABLE6 = MANUSCRIPT_DIR / "table6_spatial_model_summary_preliminary.csv"
REGISTER = MANUSCRIPT_DIR / "paper1_main_table_register.md"


def table1_data_sources() -> pd.DataFrame:
    """Return curated source matrix."""
    rows = [
        ["Study boundary", "Izmir province dissolved district boundary", "local municipal/open vector", "current", "vector", "analysis mask and reporting extent", "Interior holes below threshold removed; final boundary has no unwanted holes"],
        ["LST", "Summer 2025 land surface temperature", "user-downloaded raster", "2025", "aligned 100 m", "dependent variable", "Zero-coded outside-boundary cells treated as nodata"],
        ["LCZ", "WUDAPT/RUBCLIM global LCZ", "RUB/RUBCLIM/LCZ/global_lcz_map/latest via GEE", "nominal 2018", "100 m", "urban-form classification and fixed effects", "Temporal mismatch must be acknowledged"],
        ["LCZ probability", "LCZ class probability", "same LCZ source", "nominal 2018", "100 m", "classification confidence/control", "Used as support/control layer"],
        ["NDVI", "Vegetation greenness", "user-downloaded raster", "2025", "aligned 100 m", "biophysical predictor", "Vegetation indices are screened for collinearity"],
        ["EVI", "Enhanced Vegetation Index", "user-downloaded raster (stored as izmir_SAVI for continuity)", "2025", "aligned 100 m", "optional biophysical predictor", "Highly collinear with NDVI; not used together in baseline models"],
        ["NDBI", "Built-up intensity proxy", "user-downloaded raster", "2025", "aligned 100 m", "built-environment predictor", "Expected positive relationship with LST"],
        ["Albedo", "Surface reflectance/albedo", "user-downloaded raster", "2025", "aligned 100 m", "surface-energy predictor", "Interpret with local surface-material context"],
        ["Tree canopy height", "Canopy height", "TCH_filled_nodata.tif (final user-prepared raster)", "current/source year", "aligned 100 m", "vegetation-structure predictor", "Non-tree cells are coded as 0 m; no implicit spatial interpolation is used"],
        ["SRTM", "Elevation, slope, aspect sine/cosine", "izmir_srtm.tif", "SRTM reference period", "aligned 100 m", "terrain controls", "Used instead of DSM/SYM for slope/aspect"],
        ["Population", "ADNKS mahalle population controls", "izmir_population_data.xlsx + mahalle geometries", "2024", "mahalle rasterized to 100 m", "heat-justice controls", "Workbook excludes low-population neighbourhoods"],
        ["CORINE", "Land-cover time slices", "user-downloaded CORINE rasters", "1990-2018", "aligned 100 m", "descriptive temporal context", "Categorical; nearest-neighbour alignment"],
        ["GLC_FCS", "Land-cover time cube", "user-downloaded GLC_FCS rasters", "1985-2022", "aligned 100 m", "descriptive temporal context", "Categorical; nearest-neighbour alignment"],
    ]
    return pd.DataFrame(rows, columns=["data_group", "dataset", "source", "year", "working_resolution", "study_role", "key_limitation_or_rule"])


def table2_variables() -> pd.DataFrame:
    """Return curated variable definition matrix."""
    rows = [
        ["lst_2025", "Summer 2025 LST", "dependent", "deg C", "Thermal outcome variable"],
        ["ndvi_2025", "NDVI", "core predictor", "index", "Vegetation greenness/cooling proxy"],
        ["ndbi_2025", "NDBI", "core predictor", "index", "Built-up intensity/heating proxy"],
        ["albedo_2025", "Albedo", "core predictor", "index", "Surface reflectance/energy balance"],
        ["evi_2025", "EVI", "optional predictor", "index", "Enhanced vegetation index (replaces prior SAVI inputs); excluded from baseline when NDVI is retained"],
        ["tree_canopy_height_2025", "Tree canopy height", "core predictor", "m", "Vertical vegetation structure; non-tree cells coded as 0 m in the final canopy product"],
        ["lcz", "LCZ class", "categorical control", "class", "Local climate zone fixed effects"],
        ["lcz_probability", "LCZ probability", "quality/control", "probability", "LCZ classification support"],
        ["srtm_elevation_m", "SRTM elevation", "terrain control", "m", "Topographic cooling/heating context"],
        ["srtm_slope_deg", "SRTM slope", "terrain control", "degrees", "Topographic gradient"],
        ["srtm_aspect_sin", "Aspect sine", "terrain control", "unitless", "East-west exposure component"],
        ["srtm_aspect_cos", "Aspect cosine", "terrain control", "unitless", "North-south exposure component"],
        ["pop_total", "Total population", "justice control", "persons", "Population exposure proxy"],
        ["female_share", "Female population share", "justice control", "share", "Demographic vulnerability proxy"],
        ["child_0_14_share", "Child share", "justice control", "share", "Demographic vulnerability proxy"],
        ["elderly_65_plus_share", "Elderly share", "justice control", "share", "Demographic vulnerability proxy"],
        ["uhi_2025_diagnostic", "UHI diagnostic", "diagnostic only", "index/deg C", "Heat-derived layer; not independent predictor"],
        ["utfvi_2025_diagnostic", "UTFVI diagnostic", "diagnostic only", "index", "Heat-derived layer; not independent predictor"],
    ]
    return pd.DataFrame(rows, columns=["variable", "definition", "model_role", "unit", "interpretation"])


def table3_models() -> pd.DataFrame:
    """Return planned model specification table."""
    rows = [
        ["LCZ difference test", "lst_2025", "lcz", "Kruskal-Wallis + pairwise Mann-Whitney with FDR", "Tests whether LCZ classes have distinct LST distributions"],
        ["Physical/environmental OLS", "lst_2025", "NDVI, NDBI, albedo, tree canopy, SRTM terrain, LCZ probability, LCZ dummies", "HC3 robust OLS + VIF", "Global baseline and collinearity check using coverage-verified predictors"],
        ["Heat-justice OLS", "lst_2025", "Physical predictors + population controls", "HC3 robust OLS + VIF", "Tests whether demographic controls alter baseline heat model"],
        ["Exploratory sampled spatial model", "lst_2025", "NDVI, NDBI, albedo", "adaptive bisquare GWR on spatial sample", "Current spatial non-stationarity evidence; interpret as exploratory rather than full-province final MGWR"],
    ]
    return pd.DataFrame(rows, columns=["model", "dependent_variable", "predictor_set", "estimator", "purpose"])


def table4_lcz() -> pd.DataFrame:
    lcz = pd.read_csv(TABLES_DIR / "paper1_lcz_lst_summary.csv")
    tests = pd.read_csv(TABLES_DIR / "paper1_lcz_lst_stat_tests.csv")
    omnibus = tests[tests["test"] == "kruskal_wallis"].iloc[0].to_dict()
    lcz["omnibus_test"] = "Kruskal-Wallis"
    lcz["omnibus_h"] = omnibus["statistic"]
    lcz["omnibus_p"] = omnibus["p_value"]
    lcz["eta_squared_approx"] = omnibus.get("eta_squared_approx")
    return lcz


def table5_ols() -> pd.DataFrame:
    summary = pd.read_csv(TABLES_DIR / "paper1_ols_model_summary.csv")
    vif = pd.read_csv(TABLES_DIR / "paper1_vif_diagnostics.csv")
    rows = []
    for _, row in summary.iterrows():
        model_vif = vif[vif["model"] == row["model"]]
        predictors = json.loads(row["predictors"])
        rows.append(
            {
                "model": row["model"],
                "n": int(row["n"]),
                "r2": row["r2"],
                "adj_r2": row["adj_r2"],
                "aic": row["aic"],
                "bic": row["bic"],
                "predictor_count": len(predictors),
                "max_vif": model_vif["vif"].max(),
                "terms_vif_ge_10": int((model_vif["vif"] >= 10).sum()),
            }
        )
    return pd.DataFrame(rows)


def table6_spatial() -> pd.DataFrame:
    summary_path = TABLES_DIR / "paper1_gwr_preliminary_summary.csv"
    coef_path = TABLES_DIR / "paper1_gwr_preliminary_coefficients_summary.csv"
    if not summary_path.exists():
        return pd.DataFrame([{"model": "exploratory_sampled_spatial_model", "status": "not_run"}])
    summary = pd.read_csv(summary_path)
    coefs = pd.read_csv(coef_path) if coef_path.exists() else pd.DataFrame()
    out = summary.copy()
    out["status"] = "exploratory_sampled_spatial_model"
    out["coefficient_summary_file"] = str(coef_path) if coef_path.exists() else ""
    out["coefficient_terms_summarised"] = len(coefs) if not coefs.empty else 0
    out["interpretation_rule"] = "Do not describe as final MGWR unless a true MGWR is run and reported."
    return out


def write_register() -> None:
    lines = [
        "# Paper 1 Main Manuscript Table Register",
        "",
        "| Table | File | Purpose |",
        "|---|---|---|",
        f"| Table 1 | `{TABLE1.name}` | Data source, year, resolution, role, limitation matrix |",
        f"| Table 2 | `{TABLE2.name}` | Variable definitions and modelling roles |",
        f"| Table 3 | `{TABLE3.name}` | Statistical model specifications |",
        f"| Table 4 | `{TABLE4.name}` | LCZ-LST summaries and omnibus tests |",
        f"| Table 5 | `{TABLE5.name}` | OLS model summary and VIF diagnostics |",
        f"| Table 6 | `{TABLE6.name}` | Exploratory sampled spatial model summary with explicit non-MGWR wording |",
    ]
    REGISTER.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    logger = get_logger("p1_14_manuscript_tables")
    MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    table1_data_sources().to_csv(TABLE1, index=False, encoding="utf-8-sig")
    table2_variables().to_csv(TABLE2, index=False, encoding="utf-8-sig")
    table3_models().to_csv(TABLE3, index=False, encoding="utf-8-sig")
    table4_lcz().to_csv(TABLE4, index=False, encoding="utf-8-sig")
    table5_ols().to_csv(TABLE5, index=False, encoding="utf-8-sig")
    table6_spatial().to_csv(TABLE6, index=False, encoding="utf-8-sig")
    write_register()

    for path in [TABLE1, TABLE2, TABLE3, TABLE4, TABLE5, TABLE6, REGISTER]:
        logger.info("Wrote manuscript table draft: %s", path)


if __name__ == "__main__":
    main()
