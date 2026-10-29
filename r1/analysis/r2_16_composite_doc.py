"""Documentation of the June-August 2025 Landsat composite (requested in the internal review):
scene count by mission, WRS-2 path/row, acquisition dates and local overpass time, and the number of
clear observations per 100 m cell inside the province (same collection, cloud and QA filters as the
re-export of r2_05). Also records the version metadata of the global LCZ asset.
"""
import sys
from pathlib import Path
import numpy as np
import geopandas as gpd
import ee

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
from r2_01_gee_probe import coll  # noqa

ee.Initialize(project=__import__("os").environ["EE_PROJECT"])
b = gpd.read_file(DATA / "izmir_province_boundary_filled.gpkg").to_crs(4326)
x0, y0, x1, y1 = b.total_bounds
geom = ee.Geometry(b.geometry.union_all().buffer(0).simplify(0.005).buffer(0).__geo_interface__, None, False)
rect = ee.Geometry.Rectangle([x0, y0, x1, y1])
c = coll(10).filterBounds(rect)
# scene metadata from the unprocessed collection (same filters); coll() rebuilds images without properties
raw = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
       .filterDate("2025-06-01", "2025-09-01").filter(ee.Filter.lt("CLOUD_COVER", 10)).filterBounds(rect))
info = raw.aggregate_array("SPACECRAFT_ID").getInfo()
pr = raw.aggregate_array("WRS_PATH").getInfo(), raw.aggregate_array("WRS_ROW").getInfo()
dates = raw.aggregate_array("DATE_ACQUIRED").getInfo()
times = raw.aggregate_array("SCENE_CENTER_TIME").getInfo()
print("scenes", len(info), {k: info.count(k) for k in set(info)})
print("path/rows", sorted(set(zip(*pr))))
print("dates", min(dates), max(dates), "n unique", len(set(dates)))
print("times UTC", sorted(t[:5] for t in times)[:3], sorted(t[:5] for t in times)[-3:])
cnt = c.select("lst").count().rename("n")
stats = cnt.reduceRegion(ee.Reducer.percentile([0, 5, 25, 50, 75, 95, 100]).combine(ee.Reducer.mean(), sharedInputs=True),
                         geometry=geom, scale=100, maxPixels=1e10, tileScale=4).getInfo()
print("clear obs per cell", stats)
lcz = ee.ImageCollection("RUB/RUBCLIM/LCZ/global_lcz_map/latest")
props = lcz.first().toDictionary().getInfo()
print("lcz props", {k: v for k, v in props.items() if "version" in k.lower() or "year" in k.lower() or "date" in k.lower()})
print("lcz asset id", lcz.first().get("system:id").getInfo(), lcz.get("system:version").getInfo() if True else "")
utc = sorted(int(t[:2]) * 60 + int(t[3:5]) for t in times)
v = {"ls_scenes": len(info), "ls_scenes_l8": info.count("LANDSAT_8"), "ls_scenes_l9": info.count("LANDSAT_9"),
     "ls_pathrows": "; ".join(f"{p}/{r}" for p, r in sorted(set(zip(*pr)))), "ls_date_first": min(dates), "ls_date_last": max(dates),
     "ls_utc_med": f"{utc[len(utc) // 2] // 60:02d}:{utc[len(utc) // 2] % 60:02d}",
     "ls_clear_med": int(stats["n_p50"]), "ls_clear_p5": int(stats["n_p5"]), "ls_clear_min": int(stats["n_p0"]), "ls_clear_max": int(stats["n_p100"])}
print(v)
write_values("r2_16_composite_doc", v)
