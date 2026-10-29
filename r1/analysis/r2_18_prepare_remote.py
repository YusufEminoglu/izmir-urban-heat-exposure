"""Bundle the inputs of the intercept/slope decomposition (r2_18) for a remote machine without geodata libraries."""
import sys
import shutil
from pathlib import Path
import numpy as np
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa
from mgwr_core import load_r0_fit

OUT = CACHE / "remote_dec"
OUT.mkdir(exist_ok=True)
s = gpd.read_file(MGWR_SAMPLE)
coords = s[["x", "y"]].to_numpy(float)
X = s[PREDICTORS].to_numpy(float); y = s[DEP].to_numpy(float)
g = load_grid()
key = g.row.astype(np.int64) * 10000 + g.col
skey = set((s.row.astype(np.int64) * 10000 + s.col).tolist())
ho = np.where(~key.isin(skey).to_numpy())[0]
sub = np.random.default_rng(4).choice(ho, 200_000, replace=False)
gs = g.iloc[sub]
np.savez(OUT / "dec_inputs.npz", coords=coords, X=X, y=y, hxy=gs[["x", "y"]].to_numpy(float), hX=gs[PREDICTORS].to_numpy(float),
         hy=gs[DEP].to_numpy(float))
r0 = load_r0_fit()
np.savez(OUT / "r0_fit.npz", **{k: v for k, v in r0.items() if isinstance(v, (np.ndarray, float, int, np.floating, np.integer))})
for f in ["mgwr_dec_int_only.npz", "mgwr_dec_glob_int.npz"]:
    if (CACHE / f).exists():
        shutil.copy2(CACHE / f, OUT / f)
shutil.copy2(Path(__file__).parent / "mgwr_core.py", OUT / "mgwr_core.py")
print(sorted(p.name for p in OUT.iterdir()))
