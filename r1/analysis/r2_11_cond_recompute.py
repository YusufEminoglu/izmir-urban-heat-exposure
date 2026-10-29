"""Local condition numbers (Belsley index with kernel-weighted columns, as in mgwr.MGWRResults.local_collinearity)
recomputed directly for MGWR fits whose stored values are missing."""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa


def local_cond(coords, X, bws):
    n, k = X.shape
    D = cdist(coords, coords)
    Ds = np.sort(D, 1)
    out = np.empty(n)
    for i in range(n):
        cols = []
        for j in range(k):
            h = Ds[i, int(bws[j]) - 1] * 1.0000001
            z = D[i] / h
            w = np.where(z < 1, (1 - z ** 2) ** 2, 0.0)
            cols.append(X[:, j] * w)
        xw = np.column_stack(cols)
        sxw = xw / np.sqrt((xw ** 2).sum(0))
        s = np.linalg.svd(sxw, compute_uv=False)
        out[i] = s[0] / s[-1]
    return out


if __name__ == "__main__":
    for name in sys.argv[1:]:
        p = CACHE / f"mgwr_{name}.npz"
        f = dict(np.load(p))
        X = f["Z"] if f["Z"].shape[1] == len(f["bws"]) else np.column_stack([np.ones(len(f["Z"])), f["Z"]])
        c = local_cond(f["coords"], X, f["bws"])
        print(name, "stored max", np.nanmax(f["cond"]) if np.isfinite(f["cond"]).any() else None, "recomputed median/max", np.median(c), c.max())
        if not np.isfinite(f["cond"]).any():
            f["cond"] = c
            np.savez(p, **f)
            print("  stored recomputed values")
