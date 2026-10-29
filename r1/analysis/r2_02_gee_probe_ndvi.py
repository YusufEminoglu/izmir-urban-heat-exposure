"""Identify the NDVI source of the stored 2025 stack (Landsat variants and Sentinel-2)."""
import sys
from pathlib import Path
import numpy as np
import ee

sys.path.insert(0, str(Path(__file__).parent))
from r2_01_gee_probe import fetch, ref, qa, vars_, coll  # noqa


def cmp(name, img, band, key):
    a = fetch(img, band); r = ref[key]
    ok = np.isfinite(a) & np.isfinite(r)
    d = a[ok] - r[ok]
    print(f"{name:40s} {band}: r={np.corrcoef(a[ok], r[ok])[0, 1]:.4f} mad={np.median(np.abs(d)):.4f} meanA={a[ok].mean():.3f} meanRef={r[ok].mean():.3f}", flush=True)


def s2(cloud, red="median"):
    def m(img):
        scl = img.select("SCL")
        ok = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        o = img.updateMask(ok).divide(10000)
        nd = o.normalizedDifference(["B8", "B4"]).rename("ndvi")
        nb = o.normalizedDifference(["B11", "B8"]).rename("ndbi")
        ev = o.expression("2.5*(n-r)/(n+6*r-7.5*b+1)", {"n": o.select("B8"), "r": o.select("B4"), "b": o.select("B2")}).rename("evi")
        return ee.Image.cat([nd, nb, ev])
    c = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate("2025-06-01", "2025-09-01").filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud)).map(m)
    return getattr(c, red)()


L = coll(10)
cmp("Landsat mean cc<10", L.mean(), "ndvi", "ndvi_2025")
cmp("Landsat median cc<10", L.median(), "ndvi", "ndvi_2025")
cmp("Landsat max cc<10", L.max(), "ndvi", "ndvi_2025")
for cc in (10, 20):
    for red in ("median", "mean"):
        cmp(f"S2 {red} cc<{cc}", s2(cc, red), "ndvi", "ndvi_2025")
img = s2(10, "median").setDefaultProjection(crs="EPSG:32635", scale=10).reduceResolution(ee.Reducer.mean(), maxPixels=200)
cmp("S2 median cc<10 reduceRes", img, "ndvi", "ndvi_2025")
cmp("S2 median cc<10", s2(10), "ndbi", "ndbi_2025")
cmp("S2 median cc<10", s2(10), "evi", "evi_2025")
cmp("Landsat mean cc<10", L.mean(), "ndbi", "ndbi_2025")
cmp("Landsat mean cc<10", L.mean(), "evi", "evi_2025")
