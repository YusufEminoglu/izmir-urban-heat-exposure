"""Physical OLS (Eq. 5 specification: NDVI, NDBI, albedo, canopy height, terrain, LCZ probability and LCZ dummies,
standardised continuous predictors) with the 2025 LST, NDVI, NDBI and albedo taken from the documented Earth Engine
re-export instead of the stored layers. Compares adjusted R2 and the surface coefficients with the reported model,
and records how well the re-export reproduces each stored layer."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

G = {k: np.load(CACHE / "gee" / f"{k}.npy") for k in ["l_lst", "l_ndvi", "l_ndbi", "l_albedo"]}
g = load_grid()
r, c = g.row.to_numpy(int), g.col.to_numpy(int)
alt = g.copy()
for src, dst in [("l_lst", DEP), ("l_ndvi", "ndvi_2025"), ("l_ndbi", "ndbi_2025"), ("l_albedo", "albedo_2025")]:
    alt[dst] = G[src][r, c]
v = {}
for col in [DEP, "ndvi_2025", "ndbi_2025", "albedo_2025"]:
    ok = np.isfinite(alt[col]) & np.isfinite(g[col])
    v[f"scr_r_{col.split('_')[0]}"] = fmt(np.corrcoef(alt.loc[ok, col], g.loc[ok, col])[0, 1], 3)


def fit(df):
    d = df[[DEP, "lcz"] + OLS_CONT].dropna().copy()
    dm = pd.get_dummies(d.lcz.astype(int), prefix="lcz", dtype=float)
    dm = dm.drop(columns=dm.sum().idxmax())
    Xc = (d[OLS_CONT] - d[OLS_CONT].mean()) / d[OLS_CONT].std()
    X = sm.add_constant(pd.concat([Xc, dm], axis=1))
    m = sm.OLS(d[DEP].astype(float), X).fit()
    return m, len(d)


m0, n0 = fit(g)
m1, n1 = fit(alt)
v["scr_ols_n"] = fmt_int(n1)
v["scr_ols_r2_adj"] = fmt(m1.rsquared_adj, 3); v["ref_ols_r2_adj"] = fmt(m0.rsquared_adj, 3)
for k in ["ndvi_2025", "ndbi_2025", "albedo_2025"]:
    nm = k.split("_")[0]
    v[f"scr_ols_b_{nm}"] = fmt(m1.params[k], 2); v[f"ref_ols_b_{nm}"] = fmt(m0.params[k], 2)
print(v)
write_values("r2_21_scripted_ols", {k: x for k, x in v.items() if k.startswith("ref_")})  # v2: analysed = scripted layers
