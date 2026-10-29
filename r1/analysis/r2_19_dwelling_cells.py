"""Dwelling-based residential cells: the municipal inventory records building use (YAPITIP: MESKEN dwelling,
ISYERI workplace, KAMU public). Adds to grid_inventory.pkl, per 100 m cell:
  bsf_res  footprint fraction of dwelling buildings
  fa_res   dwelling floor area (footprint x above-ground floors, m2)
so that the mahalle heat terms can be computed over cells with dwellings (and weighted by dwelling floor area).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

X0, Y0, CELL = 432600.0, 4359700.0, 100.0
z = np.load(CACHE / "buildings_cells.npz", allow_pickle=True)
bx, by, area, fl, typ = z["x"], z["y"], z["area"], z["floors"], z["typ"]
ok = np.isfinite(area) & (area > 5) & (area < 200_000)
bx, by, area, fl, typ = bx[ok], by[ok], area[ok], fl[ok], typ[ok]
res = typ == "MESKEN"
print("buildings", ok.sum(), "dwellings", res.sum(), {t: int((typ == t).sum()) for t in np.unique(typ)})
col = np.floor((bx - X0) / CELL).astype(np.int64); row = np.floor((Y0 - by) / CELL).astype(np.int64)
inv = pd.read_pickle(CACHE / "grid_inventory.pkl")
key = inv.row.astype(np.int64) * 10000 + inv.col
d = pd.DataFrame({"key": row[res] * 10000 + col[res], "a": area[res], "fa": area[res] * fl[res]}).groupby("key").sum()
inv["bsf_res"] = (key.map(d.a).fillna(0) / (CELL * CELL)).clip(upper=1).to_numpy()
inv["fa_res"] = key.map(d.fa).fillna(0).to_numpy()
inv.to_pickle(CACHE / "grid_inventory.pkl")
v = {"n_dwellings": fmt_int(res.sum()), "n_buildings_used": fmt_int(ok.sum()), "n_cells_dwelling": fmt_int((inv.bsf_res > 0).sum()),
     "n_cells_building": fmt_int((inv.bsf > 0).sum())}
print(v)
write_values("r2_19_dwelling_cells", v)
