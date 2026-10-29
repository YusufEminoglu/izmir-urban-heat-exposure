"""Where are the 2025 layers missing inside the Izmir province boundary?

Province boundary = dissolved IMM district (ilce) polygons with interior holes removed.
Rasterised to the LCZ-aligned 100 m grid (x0 432600, y0 4359700, 1981 x 1744).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import shapely
from rasterio import features
from rasterio.transform import from_origin
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

ILCE = Path(r"EXTERNAL_DATA_DIR\building_inventory_project\data\00_external\boundaries\ilce.shp")
SB = Path(r"EXTERNAL_DATA_DIR\building_inventory_project\data\00_external\boundaries\izmir_study_boundary.gpkg")
NC, NR = 1981, 1744
T = from_origin(432600.0, 4359700.0, 100.0, 100.0)

ilce = gpd.read_file(ILCE).to_crs(CRS)
prov = ilce.dissolve().geometry.iloc[0]
prov = shapely.make_valid(prov)
polys = list(prov.geoms) if prov.geom_type == "MultiPolygon" else [prov]
filled = shapely.MultiPolygon([shapely.Polygon(p.exterior) for p in polys if p.geom_type == "Polygon"])
print("ilce:", len(ilce), "parts:", len(polys), "area km2 dissolved", prov.area / 1e6, "filled", filled.area / 1e6)
sb = gpd.read_file(SB).to_crs(CRS)
print("study boundary gpkg area km2", sb.area.sum() / 1e6, "symdiff km2", sb.geometry.union_all().symmetric_difference(filled).area / 1e6)
gpd.GeoDataFrame(geometry=[filled], crs=CRS).to_file(DATA / "izmir_province_boundary_filled.gpkg", driver="GPKG")

mask = features.rasterize([(filled, 1)], out_shape=(NR, NC), transform=T, fill=0, dtype="uint8").astype(bool)
print("boundary cells (centre inside):", mask.sum())

c25 = xr.open_dataset(CUBE_2025)
ann = xr.open_dataset(CUBE_ANNUAL)
rows = []
lay = {k: c25[k].values for k in c25.data_vars}
lay["annual_2024"] = ann["lst_jja"].values[-1]
g = load_grid(valid_only=False)
for name in ["lcz", "lcz_probability", "tree_canopy_height_2025", "srtm_elevation_m", "corine_2018", "glc_fcs_2022"]:
    a = np.full((NR, NC), np.nan); v = g[name].to_numpy(float)
    a[g.row, g.col] = v
    lay["grid_" + name] = a
miss_union = np.zeros((NR, NC), bool)
for k, a in lay.items():
    valid = np.isfinite(a) & (a != -9999)
    if k.startswith("grid_corine") or k.startswith("grid_glc"):
        valid &= a > 0
    miss = mask & ~valid
    if k in ("lst_2025", "ndvi_2025", "ndbi_2025", "albedo_2025", "evi_2025", "uhi_2025", "utfvi_2025"):
        miss_union |= miss
    # distance of missing cells to the boundary edge (cells)
    edge = ndimage.distance_transform_edt(mask)
    d = edge[miss]
    rows.append(dict(layer=k, valid_in_boundary=int((mask & valid).sum()), missing=int(miss.sum()), missing_ha=int(miss.sum()),
                     missing_within_1km_of_edge=float(np.mean(d <= 10)) if len(d) else np.nan,
                     outside_boundary_valid=int((~mask & valid).sum())))
R = pd.DataFrame(rows)
print(R.to_string())
R.to_csv(TABLES / "coverage_by_layer.csv", index=False)
lab, n = ndimage.label(miss_union, structure=np.ones((3, 3)))
sizes = np.bincount(lab.ravel())[1:]
print("missing 2025 union cells", miss_union.sum(), "patches", n, "largest patches", np.sort(sizes)[-10:])
np.save(CACHE / "province_mask.npy", mask)
np.save(CACHE / "missing_2025_union.npy", miss_union)
