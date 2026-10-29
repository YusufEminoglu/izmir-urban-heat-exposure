"""Download full-province layers from Earth Engine on the exact analysis grid (EPSG:32635, 100 m,
origin 432600/4359700, 1981 x 1744), in 2 x 2 tiles, to outputs/cache/gee/<name>.npy.

Same-source layers (to complete the grid exactly): LCZ filter and probability (WUDAPT/RUBCLIM),
CORINE 2012/2018, GLC_FCS30D 2015/2022, SRTM elevation.
Calibration layers for the 2025 spectral stack: Landsat 8/9 C2 L2 JJA 2025 (CLOUD_COVER < 10,
QA mask) mean LST, albedo, NDVI, NDBI, EVI; Sentinel-2 SR JJA 2025 mean NDVI, NDBI, EVI.
"""
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import requests
import rasterio
import ee

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
from r2_01_gee_probe import coll  # noqa

ee.Initialize(project=__import__("os").environ["EE_PROJECT"])
X0, Y0, NC, NR = 432600.0, 4359700.0, 1981, 1744
OUTD = CACHE / "gee"; OUTD.mkdir(exist_ok=True)
TILES = [(r, c, min(872, NR - r), min(991, NC - c)) for r in (0, 872) for c in (0, 991)]


def s2():
    def m(img):
        scl = img.select("SCL")
        ok = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        o = img.updateMask(ok).divide(10000)
        return ee.Image.cat([o.normalizedDifference(["B8", "B4"]).rename("s2_ndvi"), o.normalizedDifference(["B11", "B8"]).rename("s2_ndbi"),
                             o.expression("2.5*(n-r)/(n+6*r-7.5*b+1)", {"n": o.select("B8"), "r": o.select("B4"), "b": o.select("B2")}).rename("s2_evi")])
    return ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate("2025-06-01", "2025-09-01").filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10)).map(m).mean()


# "latest" is the release of 13 November 2023 (asset update time); Earth Engine has no versioned ID for it (v1 = 2022 release)
lcz = ee.ImageCollection("RUB/RUBCLIM/LCZ/global_lcz_map/latest").mosaic()
glc_annual = ee.ImageCollection("projects/sat-io/open-datasets/GLC-FCS30D/annual").mosaic()
L = coll(10).mean()
S = s2()
LAYERS = {
    "lcz": lcz.select("LCZ_Filter"), "lcz_probability": lcz.select("LCZ_Probability"),
    "corine_2018": ee.Image("COPERNICUS/CORINE/V20/100m/2018").select("landcover"),
    "corine_2012": ee.Image("COPERNICUS/CORINE/V20/100m/2012").select("landcover"),
    "glc_2022": glc_annual.select("b23"), "glc_2015": glc_annual.select("b16"),
    "srtm": ee.Image("USGS/SRTMGL1_003").select("elevation"),
    "l_lst": L.select("lst"), "l_albedo": L.select("albedo"), "l_ndvi": L.select("ndvi"), "l_ndbi": L.select("ndbi"), "l_evi": L.select("evi"),
    "s2_ndvi": S.select("s2_ndvi"), "s2_ndbi": S.select("s2_ndbi"), "s2_evi": S.select("s2_evi"),
}
CATEG = {"lcz", "corine_2018", "corine_2012", "glc_2022", "glc_2015"}


def fetch_tile(img, r, c, h, w, categorical):
    tf = [100, 0, X0 + c * 100, 0, -100, Y0 - r * 100]
    im = img.toFloat()
    for attempt in range(4):
        try:
            url = im.getDownloadURL({"crs": "EPSG:32635", "crs_transform": tf, "dimensions": [w, h], "format": "GEO_TIFF"})
            resp = requests.get(url, timeout=600)
            resp.raise_for_status()
            with rasterio.open(io.BytesIO(resp.content)) as src:
                a = src.read(1).astype(np.float32)
                if src.nodata is not None:
                    a[a == src.nodata] = np.nan
            return a
        except Exception as e:  # noqa
            print("retry", r, c, e, flush=True); time.sleep(10)
    raise RuntimeError("tile failed")


def fetch(name):
    out = OUTD / f"{name}.npy"
    if out.exists():
        return name, "cached"
    full = np.full((NR, NC), np.nan, np.float32)
    for r, c, h, w in TILES:
        full[r:r + h, c:c + w] = fetch_tile(LAYERS[name], r, c, h, w, name in CATEG)
    np.save(out, full)
    return name, f"ok valid={np.isfinite(full).sum()}"


if __name__ == "__main__":
    names = sys.argv[1:] or list(LAYERS)
    with ThreadPoolExecutor(5) as ex:
        for n, msg in ex.map(fetch, names):
            print(n, msg, flush=True)
