"""Vector helper functions used across the studies."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon

from .config import MAIN_CRS


def read_vector(path: Path, target_crs: str = MAIN_CRS) -> gpd.GeoDataFrame:
    """Read a vector layer and reproject it to the project CRS if needed."""
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"Vector layer has no CRS: {path}")
    if gdf.crs.to_string() != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf


def dissolve_all(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a single dissolved geometry preserving the input CRS."""
    dissolved = gdf.dissolve().reset_index(drop=True)
    dissolved["area_m2"] = dissolved.geometry.area
    return dissolved[["area_m2", "geometry"]]


def remove_polygon_holes_smaller_than(geometry, area_threshold_m2: float):
    """Remove interior polygon rings whose area is below a threshold."""
    if geometry.is_empty:
        return geometry
    if isinstance(geometry, Polygon):
        return _remove_polygon_holes(geometry, area_threshold_m2)
    if isinstance(geometry, MultiPolygon):
        return MultiPolygon([_remove_polygon_holes(poly, area_threshold_m2) for poly in geometry.geoms])
    raise TypeError(f"Unsupported geometry type for hole removal: {geometry.geom_type}")


def _remove_polygon_holes(polygon: Polygon, area_threshold_m2: float) -> Polygon:
    kept_interiors = []
    for interior in polygon.interiors:
        hole = Polygon(interior)
        if hole.area >= area_threshold_m2:
            kept_interiors.append(interior)
    return Polygon(polygon.exterior, kept_interiors)


def count_polygon_holes(geometry) -> int:
    """Count interior rings in Polygon or MultiPolygon geometry."""
    if geometry.is_empty:
        return 0
    if isinstance(geometry, Polygon):
        return len(geometry.interiors)
    if isinstance(geometry, MultiPolygon):
        return sum(len(poly.interiors) for poly in geometry.geoms)
    return 0


def polygon_hole_areas(geometry) -> list[float]:
    """Return interior-ring areas for Polygon or MultiPolygon geometry."""
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [Polygon(interior).area for interior in geometry.interiors]
    if isinstance(geometry, MultiPolygon):
        areas: list[float] = []
        for poly in geometry.geoms:
            areas.extend(Polygon(interior).area for interior in poly.interiors)
        return areas
    return []


def write_vector(gdf: gpd.GeoDataFrame, path: Path, layer: str | None = None) -> None:
    """Write a vector layer, creating parent folders as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {}
    if layer:
        kwargs["layer"] = layer
    gdf.to_file(path, **kwargs)


def vector_metadata(path: Path) -> dict[str, object]:
    """Read lightweight metadata for a vector file."""
    gdf = gpd.read_file(path)
    bounds = gdf.total_bounds.tolist() if len(gdf) else None
    return {
        "exists": path.exists(),
        "feature_count": int(len(gdf)),
        "crs": gdf.crs.to_string() if gdf.crs else None,
        "geometry_types": ", ".join(sorted(gdf.geometry.geom_type.dropna().unique())),
        "columns": ", ".join(map(str, gdf.columns)),
        "bounds": bounds,
    }
