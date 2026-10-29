"""Is the stored NDVI/NDBI/EVI spatially shifted relative to the analysis grid?"""
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
from r2_01_gee_probe import coll  # noqa

ee.Initialize(project=__import__("os").environ["EE_PROJECT"])
X0, Y0 = 432600.0, 4359700.0
R0_, C0_, H, W = 795, 895, 210, 210
c25 = xr.open_dataset(CUBE_2025)


def fetch(img, band, step=100.0, off=(0.0, 0.0), n=(W, H)):
    tf = [step, 0, X0 + C0_ * 100 + off[0], 0, -step, Y0 - R0_ * 100 + off[1]]
    url = img.select(band).getDownloadURL({"crs": "EPSG:32635", "crs_transform": tf, "dimensions": list(n), "format": "GEO_TIFF"})
    with rasterio.open(io.BytesIO(requests.get(url, timeout=300).content)) as src:
        a = src.read(1).astype(float)
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
    return a


L = coll(10).mean()
for band, key in [("ndvi", "ndvi_2025"), ("lst", "lst_2025"), ("ndbi", "ndbi_2025"), ("albedo", "albedo_2025")]:
    a = fetch(L, band)
    ref = c25[key].values[R0_ - 0:R0_ + H, C0_:C0_ + W]
    best = []
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            A = a[5:H - 5, 5:W - 5]
            R = c25[key].values[R0_ + 5 + dy:R0_ + H - 5 + dy, C0_ + 5 + dx:C0_ + W - 5 + dx]
            ok = np.isfinite(A) & np.isfinite(R)
            best.append((np.corrcoef(A[ok], R[ok])[0, 1], dy, dx))
    best.sort(reverse=True)
    print(band, "best shifts (r, drow, dcol):", [(round(r, 4), dy, dx) for r, dy, dx in best[:3]], "zero-shift r:",
          round([b for b in best if b[1] == 0 and b[2] == 0][0][0], 4), flush=True)
# 30 m native: does the stored NDVI equal a 30 m sample at the cell corner or a mean of the cell?
a30 = fetch(L, "ndvi", step=30.0, n=(700, 700))
