"""Raster helper functions used across the studies."""

from __future__ import annotations

from pathlib import Path

import rasterio

from .config import MAIN_CRS


def raster_metadata(path: Path) -> dict[str, object]:
    """Read lightweight metadata for a raster file without loading pixels."""
    with rasterio.open(path) as src:
        crs = src.crs.to_string() if src.crs else None
        return {
            "exists": path.exists(),
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "crs": crs,
            "resolution_x": src.res[0],
            "resolution_y": src.res[1],
            "bounds": tuple(src.bounds),
            "dtype": ", ".join(sorted(set(src.dtypes))),
            "matches_main_crs": crs == MAIN_CRS,
        }


def assert_raster_crs(path: Path, expected_crs: str = MAIN_CRS) -> None:
    """Raise if a raster lacks the expected CRS."""
    with rasterio.open(path) as src:
        actual = src.crs.to_string() if src.crs else None
    if actual != expected_crs:
        raise ValueError(f"Unexpected CRS for {path}: {actual}; expected {expected_crs}")
