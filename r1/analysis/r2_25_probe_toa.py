"""Were the stored 2025 NDVI/NDBI/EVI computed from Landsat top-of-atmosphere (TOA) reflectance?

Same test window, season, scene cloud threshold and QA mask as r2_01; indices from the Collection 2 Tier 1 TOA
collections of Landsat 8 and 9, compared with the stored layers.
"""
import sys
from pathlib import Path
import numpy as np
import ee

sys.path.insert(0, str(Path(__file__).parent))
from r2_01_gee_probe import fetch, ref  # noqa  (initialises Earth Engine and the test window)


def qa_toa(img):
    return img.updateMask(img.select("QA_PIXEL").bitwiseAnd(0b111111).eq(0))


def idx(img):
    return ee.Image.cat([img.normalizedDifference(["B5", "B4"]).rename("ndvi"), img.normalizedDifference(["B6", "B5"]).rename("ndbi"),
                         img.expression("2.5*(n-r)/(n+6*r-7.5*b+1)", {"n": img.select("B5"), "r": img.select("B4"), "b": img.select("B2")}).rename("evi")])


def toa(cloud, reducer="mean"):
    c = (ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA").merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_TOA"))
         .filterDate("2025-06-01", "2025-09-01").filter(ee.Filter.lt("CLOUD_COVER", cloud)).map(qa_toa).map(idx))
    return c.mean() if reducer == "mean" else c.median()


for name, img in [("TOA mean cc<10", toa(10)), ("TOA median cc<10", toa(10, "median")), ("TOA mean cc<30", toa(30))]:
    out = []
    for b, k in [("ndvi", "ndvi_2025"), ("ndbi", "ndbi_2025"), ("evi", "evi_2025")]:
        a = fetch(img, b); r = ref[k]
        ok = np.isfinite(a) & np.isfinite(r)
        out.append(f"{b}: r={np.corrcoef(a[ok], r[ok])[0, 1]:.4f} mad={np.median(np.abs(a[ok] - r[ok])):.4f} meanA={a[ok].mean():.3f} meanRef={r[ok].mean():.3f}")
    print(name, "|", " | ".join(out), flush=True)
