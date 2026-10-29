"""Extend the annual summer LST cube to 2014-2026 with the cube's own method (p1_11: Landsat 8/9 C2 L2, 1 Jun-31 Aug,
CLOUD_COVER < 30, QA_PIXEL bits 0-5 + QA_RADSAT mask, ST_B10 -> deg C, per-cell mean), downloaded on the exact analysis
grid (EPSG:32635, 100 m, origin 432600/4359700, 1981 x 1744).

2024 is re-exported as a check against the stored cube; 2025 and 2026 are new time steps.
Writes outputs/cache/gee/annual_lst_<year>.npy, outputs/values/r2_40_annual_lst.csv (scene counts, 2024 agreement)
and data/datacubes/paper1_landsat_lst_jja_cube_2014_2026.nc.
"""
import io
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import rasterio
import xarray as xr
import ee

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
sys.path.insert(0, r"REPO_ROOT")
from src.paper1_climate_justice.p1_04_gee_lst_summer import add_landsat_variables, qa_mask_landsat_l2  # noqa: E402

ee.Initialize(project=__import__("os").environ["EE_PROJECT"])
X0, Y0, NC, NR = 432600.0, 4359700.0, 1981, 1744
TILES = [(r, c, min(872, NR - r), min(991, NC - c)) for r in (0, 872) for c in (0, 991)]
OUTD = CACHE / "gee"; OUTD.mkdir(exist_ok=True)
OLD = DATA / "datacubes" / "paper1_landsat_lst_jja_cube_2014_2024.nc"
NEW = DATA / "datacubes" / "paper1_landsat_lst_jja_cube_2014_2026.nc"
RECT = ee.Geometry.Rectangle([26.2, 37.8, 28.5, 39.4])


def raw(year):
    return (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"))
            .filterDate(f"{year}-06-01", f"{year}-09-01").filter(ee.Filter.lt("CLOUD_COVER", 30)))


def annual_image(year):
    return raw(year).map(qa_mask_landsat_l2).map(add_landsat_variables).select("lst_c").mean()


def fetch_tile(img, r, c, h, w):
    tf = [100, 0, X0 + c * 100, 0, -100, Y0 - r * 100]
    for attempt in range(5):
        try:
            url = img.toFloat().getDownloadURL({"crs": "EPSG:32635", "crs_transform": tf, "dimensions": [w, h], "format": "GEO_TIFF"})
            resp = requests.get(url, timeout=600)
            resp.raise_for_status()
            with rasterio.open(io.BytesIO(resp.content)) as src:
                a = src.read(1).astype(np.float32)
                if src.nodata is not None:
                    a[a == src.nodata] = np.nan
            return a
        except Exception as e:  # noqa
            print("retry", r, c, e, flush=True); time.sleep(15)
    raise RuntimeError("tile failed")


def fetch(year):
    out = OUTD / f"annual_lst_{year}.npy"
    if out.exists():
        return np.load(out)
    full = np.full((NR, NC), np.nan, np.float32)
    img = annual_image(year)
    for r, c, h, w in TILES:
        full[r:r + h, c:c + w] = fetch_tile(img, r, c, h, w)
    full[(full < -50) | (full > 90) | (full == 0)] = np.nan  # same screening as p1_11
    np.save(out, full)
    return full


vals = {}
for y in range(2014, 2027):
    rc = raw(y).filterBounds(RECT)
    vals[f"scenes_{y}"] = rc.size().getInfo()
    sp = rc.aggregate_array("SPACECRAFT_ID").getInfo()
    vals[f"scenes_l9_{y}"] = sum(s == "LANDSAT_9" for s in sp)
    print(y, vals[f"scenes_{y}"], "L9:", vals[f"scenes_l9_{y}"], flush=True)

old = xr.open_dataset(OLD)
assert old.sizes["y"] == NR and old.sizes["x"] == NC
assert abs(float(old.x[0]) - (X0 + 50)) < 1e-6 and abs(float(old.y[0]) - (Y0 - 50)) < 1e-6, (float(old.x[0]), float(old.y[0]))
mask = np.load(CACHE / "province_mask.npy")

new = {}
for y in (2024, 2025, 2026):
    new[y] = fetch(y)
    v = np.isfinite(new[y]) & mask
    vals[f"cells_{y}"] = int(v.sum())
    vals[f"mean_{y}"] = float(np.nanmean(new[y][mask]))
    print(y, "cells", vals[f"cells_{y}"], "mean", round(vals[f"mean_{y}"], 3), flush=True)

s24 = old["lst_jja"].sel(time="2024").values[0]
both = np.isfinite(s24) & np.isfinite(new[2024]) & mask
d = new[2024][both] - s24[both]
vals.update(check2024_n=int(both.sum()), check2024_r=float(np.corrcoef(new[2024][both], s24[both])[0, 1]),
            check2024_mean_diff=float(d.mean()), check2024_mad=float(np.abs(d).mean()), check2024_p99_absdiff=float(np.percentile(np.abs(d), 99)),
            stored2024_cells=int((np.isfinite(s24) & mask).sum()))
print({k: v for k, v in vals.items() if k.startswith(("check", "stored"))}, flush=True)

# analysed 2025 composite vs the cube-method 2025 step
g = pd.read_csv(GRID_CSV, usecols=["row", "col", "lst_2025"]).rename(columns={"lst_2025": "lst"})
if {"row", "col", "lst"} <= set(g.columns):
    a25 = np.full((NR, NC), np.nan, np.float32); a25[g["row"].values, g["col"].values] = g["lst"].values
    m = np.isfinite(a25) & np.isfinite(new[2025]) & mask
    vals.update(c25_vs_analysed_r=float(np.corrcoef(a25[m], new[2025][m])[0, 1]), c25_vs_analysed_mean_diff=float((new[2025][m] - a25[m]).mean()))
    print("2025 cube-method vs analysed composite:", vals["c25_vs_analysed_r"], vals["c25_vs_analysed_mean_diff"], flush=True)

write_values("r2_40_annual_lst", {k: (fmt(v, 4) if isinstance(v, float) else str(v)) for k, v in vals.items()})

# extended cube: stored 2014-2024 unchanged + new 2025, 2026
years = list(range(2014, 2027))
# the stored slices were clipped to the province boundary in Earth Engine; the downloads cover the whole grid rectangle,
# so the new slices take the stored cube's footprint (cells finite in any stored year; contains every province cell)
oldv = old["lst_jja"].values.astype(np.float32)
fp = np.isfinite(oldv).any(0)
assert fp[mask].all()
add = np.stack([np.where(fp, new[2025], np.nan), np.where(fp, new[2026], np.nan)]).astype(np.float32)
vals.update(footprint_cells=int(fp.sum()))
write_values("r2_40_annual_lst", {k: (fmt(v, 4) if isinstance(v, float) else str(v)) for k, v in vals.items()})
stack = np.concatenate([oldv, add])
ds = xr.Dataset({"lst_jja": (("time", "y", "x"), stack, {"units": "degC", "nodata": "NaN"})},
                coords={"time": pd.to_datetime([f"{y}-07-15" for y in years]), "y": old.y.values, "x": old.x.values})
ds.attrs.update(old.attrs)
ds.attrs.update(years=", ".join(map(str, years)),
                workflow="Earth Engine annual JJA LST (Landsat 8/9 C2 L2, CLOUD_COVER<30, QA mask, mean) stacked with xarray; "
                         "2025-2026 added 2026-10-29 with the same method on the analysis grid")
old.close()
ds.to_netcdf(NEW, engine="netcdf4", encoding={"lst_jja": {"zlib": True, "complevel": 4}})
print("wrote", NEW, flush=True)
