"""Find the GEE recipe that reproduces the stored 2025 layers, on a test window of the analysis grid."""
import io
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import requests
import rasterio
import ee

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

ee.Initialize(project=__import__("os").environ["EE_PROJECT"])
X0, Y0 = 432600.0, 4359700.0
# test window: rows/cols of the analysis grid (central Izmir, includes coast)
R0_, C0_, H, W = 800, 900, 200, 200
tf = [100, 0, X0 + C0_ * 100, 0, -100, Y0 - R0_ * 100]

c25 = xr.open_dataset(CUBE_2025)
ref = {k: c25[k].values[R0_:R0_ + H, C0_:C0_ + W] for k in ["lst_2025", "ndvi_2025", "albedo_2025", "ndbi_2025", "evi_2025"]}


def qa(img):
    q = img.select("QA_PIXEL")
    m = q.bitwiseAnd(0b111111).eq(0)
    return img.updateMask(m).updateMask(img.select("QA_RADSAT").eq(0))


def qa_simple(img):
    q = img.select("QA_PIXEL")
    m = q.bitwiseAnd(1 << 3).eq(0).And(q.bitwiseAnd(1 << 4).eq(0))
    return img.updateMask(m)


def vars_(img):
    o = img.select("SR_B.*").multiply(0.0000275).add(-0.2)
    lst = img.select("ST_B10").multiply(0.00341802).add(149.0).subtract(273.15).rename("lst")
    ndvi = o.normalizedDifference(["SR_B5", "SR_B4"]).rename("ndvi")
    ndbi = o.normalizedDifference(["SR_B6", "SR_B5"]).rename("ndbi")
    alb = o.expression("0.356*b2+0.130*b4+0.373*b5+0.085*b6+0.072*b7-0.0018",
                       {"b2": o.select("SR_B2"), "b4": o.select("SR_B4"), "b5": o.select("SR_B5"), "b6": o.select("SR_B6"), "b7": o.select("SR_B7")}).rename("albedo")
    evi = o.expression("2.5*(n-r)/(n+6*r-7.5*b+1)", {"n": o.select("SR_B5"), "r": o.select("SR_B4"), "b": o.select("SR_B2")}).rename("evi")
    return ee.Image.cat([lst, ndvi, ndbi, alb, evi])


def coll(cloud, d0="2025-06-01", d1="2025-09-01", mask=qa):
    return (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
            .filterDate(d0, d1).filter(ee.Filter.lt("CLOUD_COVER", cloud)).map(mask).map(vars_))


def fetch(img, band):
    url = img.select(band).getDownloadURL({"crs": "EPSG:32635", "crs_transform": tf, "dimensions": [W, H], "format": "GEO_TIFF"})
    with rasterio.open(io.BytesIO(requests.get(url, timeout=300).content)) as src:
        a = src.read(1).astype(float)
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
    return a


def compare(name, img, rr=False):
    if rr:
        img = img.setDefaultProjection(crs="EPSG:32635", scale=30).reduceResolution(ee.Reducer.mean(), maxPixels=64)
    out = []
    for b, k in [("lst", "lst_2025"), ("ndvi", "ndvi_2025"), ("albedo", "albedo_2025")]:
        a = fetch(img, b); r = ref[k]
        ok = np.isfinite(a) & np.isfinite(r)
        d = a[ok] - r[ok]
        out.append(f"{b}: r={np.corrcoef(a[ok], r[ok])[0, 1]:.5f} mad={np.median(np.abs(d)):.4f} max={np.abs(d).max():.3f} n={ok.sum()}")
    print(name, "|", " | ".join(out), flush=True)


if __name__ == "__main__":
    for cloud in (10, 20, 30):
        compare(f"mean cc<{cloud}", coll(cloud).mean())
    compare("median cc<10", coll(10).median())
    compare("mean cc<10 reduceRes", coll(10).mean(), rr=True)
    compare("mean cc<10 simpleQA", coll(10, mask=qa_simple).mean())
    compare("mean cc<20 05-01..10-01", coll(20, "2025-05-01", "2025-10-01").mean())
