"""Figure 4b values: median LST of compact low-rise (LCZ 3) and dense trees (LCZ A) within the same elevation band,
for the bands in which both classes have at least 30 cells below 200 m."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

g = load_grid(valid_only=False).dropna(subset=[DEP, "lcz", "srtm_elevation_m"])
bands = [0, 50, 100, 200]
g["band"] = pd.cut(g.srtm_elevation_m, bands, right=False)
m = g[g.lcz.isin([3, 11])].groupby(["band", "lcz"], observed=False)[DEP].agg(["median", "size"]).unstack()
d = (m["median"][3] - m["median"][11])[(m["size"][3] >= 30) & (m["size"][11] >= 30)]
v = {"elev_c3A_min": fmt(d.min(), 1), "elev_c3A_max": fmt(d.max(), 1)}
Y = pd.read_csv(TABLES / "lcz_mean_lst_by_year.csv", index_col=0)
Y.index = Y.index.astype(int)
yr = Y.loc[3] - Y.loc[11]  # LCZ 3 minus LCZ A, every summer of the cube and 2025
v["c3A_year_min"] = fmt(yr.min(), 1); v["c3A_year_max"] = fmt(yr.max(), 1)
print(m, d, v)
write_values("r2_32_lcz_elevation_bands", v)
