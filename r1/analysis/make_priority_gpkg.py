"""Revised mahalle priority layer in the schema Figure 11 expects (morph_pressure := revised M_m)."""
import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
p = pd.read_csv(TABLES / "priority_revised_by_mahalle.csv", encoding="utf-8-sig")
mah = gpd.read_file(MAHALLE).to_crs(CRS)[["join_key", "geometry"]]
g = mah.merge(p, on="join_key", how="inner")
g["morph_pressure"] = g["C"]
g["priority_score_0_100"] = g["P0"]
g["priority_rank"] = g["rank"]
keep = ["join_key", "district_name_final", "mahalle_name_final", "pop_total", "lst_mean", "lst_p90", "hot_cell_share", "morph_pressure",
        "vulnerable_share", "child_0_14_share", "elderly_65_plus_share", "hot_exposed_pop_proxy", "priority_score_0_100", "priority_rank", "consensus_top", "n_res", "geometry"]
g[keep].to_file(CACHE / "priority_revised.gpkg", driver="GPKG")
print(len(g))
