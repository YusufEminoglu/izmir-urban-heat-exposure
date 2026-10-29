"""Test whether the stored NDVI/NDBI/EVI were computed from unscaled Landsat Collection 2 DNs."""
import sys
from pathlib import Path
import numpy as np
import ee

sys.path.insert(0, str(Path(__file__).parent))
from r2_01_gee_probe import fetch, ref, qa  # noqa


def cmp(name, img, band, key):
    a = fetch(img, band); r = ref[key]
    ok = np.isfinite(a) & np.isfinite(r)
    d = a[ok] - r[ok]
    print(f"{name:42s} {band}: r={np.corrcoef(a[ok], r[ok])[0, 1]:.5f} mad={np.median(np.abs(d)):.5f} max={np.abs(d).max():.4f} meanA={a[ok].mean():.4f} meanRef={r[ok].mean():.4f}", flush=True)


def raw_vars(img):
    o = img.select("SR_B.*")
    ndvi = o.normalizedDifference(["SR_B5", "SR_B4"]).rename("ndvi")
    ndbi = o.normalizedDifference(["SR_B6", "SR_B5"]).rename("ndbi")
    evi = o.expression("2.5*(n-r)/(n+6*r-7.5*b+1)", {"n": o.select("SR_B5"), "r": o.select("SR_B4"), "b": o.select("SR_B2")}).rename("evi")
    savi = o.expression("1.5*(n-r)/(n+r+0.5)", {"n": o.select("SR_B5"), "r": o.select("SR_B4")}).rename("savi")
    return ee.Image.cat([ndvi, ndbi, evi, savi])


def c(cloud, mask=True, d0="2025-06-01", d1="2025-09-01"):
    col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
           .filterDate(d0, d1).filter(ee.Filter.lt("CLOUD_COVER", cloud)))
    if mask:
        col = col.map(qa)
    return col.map(raw_vars)


for cc in (10, 30):
    for red in ("mean", "median"):
        img = getattr(c(cc), red)()
        cmp(f"raw DN {red} cc<{cc}", img, "ndvi", "ndvi_2025")
img = c(10).mean()
cmp("raw DN mean cc<10", img, "ndbi", "ndbi_2025")
cmp("raw DN mean cc<10", img, "evi", "evi_2025")
cmp("raw DN mean cc<10 (savi vs evi file)", img, "savi", "evi_2025")
img = c(10, mask=False).mean()
cmp("raw DN mean cc<10 nomask", img, "ndvi", "ndvi_2025")
