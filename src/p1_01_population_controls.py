"""Prepare Study 1 mahalle-level population and vulnerability controls."""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _config import (  # noqa: E402
    EXTERNAL_MUNICIPAL,
    MAIN_CRS,
    DATA_INTERIM,
    DATA_PROCESSED,
    TABLES_DIR,
    ensure_dirs,
    require_file,
)
from _config import get_logger  # noqa: E402
from _vector_ops import read_vector, write_vector  # noqa: E402


POPULATION_XLSX = DATA_INTERIM / "izmir_population_data.xlsx"
OUTPUT_GPKG = DATA_PROCESSED / "mahalle_population_controls_2024.gpkg"
OUTPUT_CSV = TABLES_DIR / "mahalle_population_controls_2024.csv"
JOIN_REPORT_CSV = TABLES_DIR / "mahalle_population_join_report.csv"


def normalize_name(value: object) -> str:
    """Normalize Turkish administrative names for joining."""
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    translation = str.maketrans("ÇĞİÖŞÜÂÎÛ", "CGIOSUAIU")
    text = text.translate(translation)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_population_workbook(path: Path) -> pd.DataFrame:
    """Read and flatten the ADNKS workbook."""
    require_file(path, "2024 Izmir population workbook")
    raw = pd.read_excel(path, sheet_name=0, header=[7, 8])
    raw.columns = [
        "_".join(str(part) for part in col if not str(part).startswith("Unnamed")).strip("_")
        for col in raw.columns
    ]
    raw = raw[raw["İL ADI"].eq("İZMİR")].copy()

    rename = {
        "İLÇE ADI": "district_name",
        "MAHALLE ADI": "mahalle_name",
        "MAHALLE KAYIT NO": "mahalle_kayit_no",
        "GENEL TOPLAM_TOPLAM": "pop_total",
        "GENEL TOPLAM_ERKEK": "pop_male",
        "GENEL TOPLAM_KADIN": "pop_female",
        "0-14 YAŞ GRUBU_TOPLAM": "pop_0_14",
        "0-14 YAŞ GRUBU_ERKEK": "male_0_14",
        "0-14 YAŞ GRUBU_KADIN": "female_0_14",
        "15-64 YAŞ GRUBU_TOPLAM": "pop_15_64",
        "15-64 YAŞ GRUBU_ERKEK": "male_15_64",
        "15-64 YAŞ GRUBU_KADIN": "female_15_64",
        "65 VE ÜZERİ YAŞ GRUBU_TOPLAM": "pop_65_plus",
        "65 VE ÜZERİ YAŞ GRUBU_ERKEK": "male_65_plus",
        "65 VE ÜZERİ YAŞ GRUBU_KADIN": "female_65_plus",
    }
    keep = list(rename)
    df = raw[keep].rename(columns=rename).copy()

    numeric_cols = [col for col in df.columns if col.startswith(("pop_", "male_", "female_"))]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].replace("C", pd.NA), errors="coerce")

    df["district_norm"] = df["district_name"].map(normalize_name)
    df["mahalle_norm"] = df["mahalle_name"].map(normalize_name)
    df["join_key"] = df["district_norm"] + "|" + df["mahalle_norm"]
    df["female_share"] = df["pop_female"] / df["pop_total"]
    df["child_0_14_share"] = df["pop_0_14"] / df["pop_total"]
    df["elderly_65_plus_share"] = df["pop_65_plus"] / df["pop_total"]
    df["age_dependency_proxy"] = (df["pop_0_14"] + df["pop_65_plus"]) / df["pop_15_64"]
    return df


def mahalle_with_districts() -> gpd.GeoDataFrame:
    """Attach district names to mahalle polygons using representative points."""
    mahalle = read_vector(EXTERNAL_MUNICIPAL / "mahalle.shp")
    ilce = read_vector(EXTERNAL_MUNICIPAL / "ilce.shp")
    mahalle = mahalle.rename(columns={"ADINUMARAS": "mahalle_name_geom"})
    ilce = ilce.rename(columns={"ADINUMARAS": "district_name_geom"})

    points = mahalle[["mahalle_name_geom", "geometry"]].copy()
    points["geometry"] = points.geometry.representative_point()
    joined = gpd.sjoin(
        points,
        ilce[["district_name_geom", "geometry"]],
        how="left",
        predicate="within",
    )
    mahalle["district_name_geom"] = joined["district_name_geom"].to_numpy()
    mahalle["district_norm"] = mahalle["district_name_geom"].map(normalize_name)
    mahalle["mahalle_norm"] = mahalle["mahalle_name_geom"].map(normalize_name)
    mahalle["join_key"] = mahalle["district_norm"] + "|" + mahalle["mahalle_norm"]
    return mahalle


def prepare_population_controls() -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Join cleaned census data to mahalle geometries and return report rows."""
    pop = clean_population_workbook(POPULATION_XLSX)
    mahalle = mahalle_with_districts()

    joined = mahalle.merge(pop, on="join_key", how="left", suffixes=("", "_pop"))
    joined = joined.set_crs(MAIN_CRS, allow_override=True)

    geom_keys = set(mahalle["join_key"])
    pop_keys = set(pop["join_key"])
    report_rows = [
        {
            "category": "geometry_without_population_match",
            "count": len(geom_keys - pop_keys),
            "keys": "; ".join(sorted(geom_keys - pop_keys)[:50]),
        },
        {
            "category": "population_without_geometry_match",
            "count": len(pop_keys - geom_keys),
            "keys": "; ".join(sorted(pop_keys - geom_keys)[:50]),
        },
        {
            "category": "matched_geometry_rows",
            "count": int(joined["pop_total"].notna().sum()),
            "keys": "",
        },
        {
            "category": "total_geometry_rows",
            "count": len(joined),
            "keys": "",
        },
        {
            "category": "total_population_rows",
            "count": len(pop),
            "keys": "",
        },
    ]
    return joined, pd.DataFrame(report_rows)


def main() -> None:
    ensure_dirs()
    logger = get_logger("p1_01_population_controls")
    logger.info("Preparing Study 1 population controls.")

    joined, report = prepare_population_controls()
    write_vector(joined, OUTPUT_GPKG, layer="mahalle_population_controls_2024")
    joined.drop(columns="geometry").to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    report.to_csv(JOIN_REPORT_CSV, index=False, encoding="utf-8-sig")

    matched = int(joined["pop_total"].notna().sum())
    logger.info("Wrote population controls: %s", OUTPUT_GPKG)
    logger.info("Matched %s/%s mahalle geometries to population rows.", matched, len(joined))
    logger.info("Wrote join report: %s", JOIN_REPORT_CSV)


if __name__ == "__main__":
    main()
