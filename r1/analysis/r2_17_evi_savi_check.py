"""Is the stored 'evi_2025' layer EVI or SAVI? (its cube provenance names an izmir_SAVI file)

Samples 2,000 random grid cells, computes June-August 2025 Landsat 8/9 EVI and SAVI (L = 0.5) from the
same masked collection as the re-export, and correlates both with the stored layer; also NDVI as reference.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import ee

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
from r2_01_gee_probe import qa  # noqa

ee.Initialize(project=__import__("os").environ["EE_PROJECT"])
g = load_grid(valid_only=False).dropna(subset=["evi_2025", "ndvi_2025"])
s = g[g.gapfilled.fillna(0) == 0].sample(2000, random_state=3)


def idx(img):
    o = img.select("SR_B.*").multiply(0.0000275).add(-0.2)
    n, r, b = o.select("SR_B5"), o.select("SR_B4"), o.select("SR_B2")
    evi = o.expression("2.5*(n-r)/(n+6*r-7.5*b+1)", {"n": n, "r": r, "b": b}).rename("evi")
    savi = o.expression("1.5*(n-r)/(n+r+0.5)", {"n": n, "r": r}).rename("savi")
    return ee.Image.cat([evi, savi, o.normalizedDifference(["SR_B5", "SR_B4"]).rename("ndvi")])


col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
       .filterDate("2025-06-01", "2025-09-01").filter(ee.Filter.lt("CLOUD_COVER", 10)).map(qa).map(idx)).mean()
fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([x, y], "EPSG:32635"), {"i": int(i)}) for i, x, y in zip(s.index, s.x, s.y)])
res = col.reduceRegions(fc, ee.Reducer.mean(), scale=100, crs="EPSG:32635").getInfo()
d = pd.DataFrame([dict(i=f["properties"]["i"], evi=f["properties"].get("evi"), savi=f["properties"].get("savi"), ndvi=f["properties"].get("ndvi"))
                  for f in res["features"]]).set_index("i").dropna()
d = d.join(s[["evi_2025", "ndvi_2025"]])
print(d.describe().round(3))
print(d.corr().round(4))
for k in ["evi", "savi"]:
    b = np.polyfit(d[k], d.evi_2025, 1)
    print(k, "slope/intercept vs stored", b.round(3), "mean abs diff", round((d[k] - d.evi_2025).abs().mean(), 4))
