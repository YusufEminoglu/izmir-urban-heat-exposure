"""Study 1 local data inventory and study-boundary preparation."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _config import (  # noqa: E402
    EXTERNAL_MUNICIPAL,
    EXTERNAL_ELEVATION,
    DATA_INTERIM,
    TABLES_DIR,
    ensure_dirs,
    require_file,
)
from _config import get_logger  # noqa: E402
from _raster_ops import raster_metadata  # noqa: E402
from _vector_ops import dissolve_all, read_vector, vector_metadata, write_vector  # noqa: E402
from _vector_ops import count_polygon_holes, polygon_hole_areas, remove_polygon_holes_smaller_than  # noqa: E402


HOLE_REMOVAL_THRESHOLD_M2 = 1_973_980_279.15


VECTOR_LAYERS = {
    "ilce": EXTERNAL_MUNICIPAL / "ilce.shp",
    "mahalle": EXTERNAL_MUNICIPAL / "mahalle.shp",
    "yapi": EXTERNAL_MUNICIPAL / "yapi.shp",
    "yolorta": EXTERNAL_MUNICIPAL / "yolorta.shp",
    "numarataj": EXTERNAL_MUNICIPAL / "numarataj.shp",
}

RASTER_LAYERS = {
    "izmir_sym5": EXTERNAL_ELEVATION / "izmir_sym5.tif",
}


def build_inventory() -> pd.DataFrame:
    """Return a table describing local data readiness for Study 1."""
    rows: list[dict[str, object]] = []

    for name, path in VECTOR_LAYERS.items():
        row: dict[str, object] = {"name": name, "kind": "vector", "path": str(path)}
        if path.exists():
            try:
                row.update(vector_metadata(path))
                row["status"] = "ok"
            except Exception as exc:  # pragma: no cover
                row["status"] = "read_error"
                row["error"] = str(exc)
        else:
            row["status"] = "missing"
        rows.append(row)

    for name, path in RASTER_LAYERS.items():
        row = {"name": name, "kind": "raster", "path": str(path)}
        if path.exists():
            try:
                row.update(raster_metadata(path))
                row["status"] = "ok"
            except Exception as exc:  # pragma: no cover
                row["status"] = "read_error"
                row["error"] = str(exc)
        else:
            row["status"] = "missing"
        rows.append(row)

    return pd.DataFrame(rows)


def create_study_boundary() -> Path:
    """Create a dissolved Izmir boundary from the ilce layer."""
    ilce_path = require_file(VECTOR_LAYERS["ilce"], "Izmir ilce boundary shapefile")
    ilce = read_vector(ilce_path)
    boundary = dissolve_all(ilce)
    before_holes = count_polygon_holes(boundary.geometry.iloc[0])
    before_hole_areas = polygon_hole_areas(boundary.geometry.iloc[0])
    boundary["geometry"] = boundary.geometry.apply(
        lambda geom: remove_polygon_holes_smaller_than(geom, HOLE_REMOVAL_THRESHOLD_M2)
    )
    after_holes = count_polygon_holes(boundary.geometry.iloc[0])
    output = DATA_INTERIM / "izmir_study_boundary.gpkg"
    write_vector(boundary, output, layer="izmir_study_boundary")
    removed = before_holes - after_holes
    removed_area = sum(area for area in before_hole_areas if area < HOLE_REMOVAL_THRESHOLD_M2)
    logger = get_logger("p1_00_inventory_and_boundary")
    logger.info(
        "Boundary hole cleanup: threshold=%s m2, before=%s, after=%s, removed=%s, removed_area_m2=%.2f",
        HOLE_REMOVAL_THRESHOLD_M2,
        before_holes,
        after_holes,
        removed,
        removed_area,
    )
    return output


def main() -> None:
    ensure_dirs()
    logger = get_logger("p1_00_inventory_and_boundary")
    logger.info("Starting Study 1 local inventory and boundary preparation.")

    inventory = build_inventory()
    inventory_path = TABLES_DIR / "paper1_local_data_inventory.csv"
    inventory.to_csv(inventory_path, index=False, encoding="utf-8-sig")
    logger.info("Wrote data inventory: %s", inventory_path)

    boundary_path = create_study_boundary()
    logger.info("Wrote Study 1 boundary: %s", boundary_path)

    ok_count = int((inventory["status"] == "ok").sum())
    logger.info("Inventory complete: %s/%s local layers readable.", ok_count, len(inventory))


if __name__ == "__main__":
    main()
