"""Were the stored 2025 NDVI/NDBI/'EVI' computed from Sentinel-2 L2A digital numbers that still carry the
+1000 radiometric offset of processing baseline 04.00 (2022 onwards)?

On the r2_01 test window, June-August 2025 Sentinel-2 indices are computed from DN + 1000 (the values of the
non-harmonised L2A product) and compared with the stored layers, together with the offset-free indices.
"""
import sys
from pathlib import Path
import numpy as np
import ee

sys.path.insert(0, str(Path(__file__).parent))
from r2_01_gee_probe import fetch, ref  # noqa  (initialises Earth Engine and the test window)


def s2(cloud=20, offset=1000.0):
    def m(img):
        scl = img.select("SCL")
        ok = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        b = img.select(["B2", "B4", "B8", "B11"]).add(offset)
        n, r, bl, sw = b.select("B8"), b.select("B4"), b.select("B2"), b.select("B11")
        out = ee.Image.cat([n.subtract(r).divide(n.add(r)).rename("ndvi"), sw.subtract(n).divide(sw.add(n)).rename("ndbi"),
                            n.subtract(r).multiply(1.5).divide(n.add(r).add(5000)).rename("savi_dn"),
                            n.divide(1e4).subtract(r.divide(1e4)).multiply(1.5).divide(n.divide(1e4).add(r.divide(1e4)).add(0.5)).rename("savi")])
        return out.updateMask(ok)
    return (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate("2025-06-01", "2025-09-01")
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud)).map(m))


for name, img in [("S2 DN+1000 median cc<20", s2().median()), ("S2 DN+1000 mean cc<20", s2().mean()),
                  ("S2 DN+1000 median cc<100", s2(100).median()), ("S2 offset-free median cc<20", s2(offset=0).median())]:
    out = []
    for b, k in [("ndvi", "ndvi_2025"), ("ndbi", "ndbi_2025"), ("savi", "evi_2025"), ("savi_dn", "evi_2025")]:
        a = fetch(img, b); r = ref[k]
        ok = np.isfinite(a) & np.isfinite(r)
        out.append(f"{b}: r={np.corrcoef(a[ok], r[ok])[0, 1]:.4f} mad={np.median(np.abs(a[ok] - r[ok])):.4f} meanA={a[ok].mean():.3f} meanRef={r[ok].mean():.3f}")
    print(name, "|", " | ".join(out), flush=True)

# areal mean of the 10 m pixels to the 100 m cell (an export at native resolution aggregated locally)
for name, img in [("S2 DN+1000 mean cc<20 areal-mean", s2().mean()), ("S2 DN+1000 median cc<20 areal-mean", s2().median())]:
    img = img.setDefaultProjection(crs="EPSG:32635", scale=10).reduceResolution(ee.Reducer.mean(), maxPixels=1024)
    out = []
    for b, k in [("ndvi", "ndvi_2025"), ("ndbi", "ndbi_2025"), ("savi", "evi_2025")]:
        a = fetch(img, b); r = ref[k]
        ok = np.isfinite(a) & np.isfinite(r)
        out.append(f"{b}: r={np.corrcoef(a[ok], r[ok])[0, 1]:.4f} mad={np.median(np.abs(a[ok] - r[ok])):.4f} meanA={a[ok].mean():.3f} meanRef={r[ok].mean():.3f}")
    print(name, "|", " | ".join(out), flush=True)
