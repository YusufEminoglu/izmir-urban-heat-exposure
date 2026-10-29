"""True multi-scale GWR (MGWR) on a 10k stratified sample — Study 1 (Izmir heat justice).

Replaces the single-bandwidth GWR of p1_09 with a genuine multi-scale MGWR:
NDVI, NDBI and albedo each receive their own adaptive bisquare bandwidth via the
back-fitting algorithm (Fotheringham et al. 2017), selected by AICc. Reports
per-predictor bandwidths, back-fitting convergence, global/local fit, local
condition numbers, and a manually computed local R2 (reference-bandwidth method,
since mgwr does not implement localR2 for multiple bandwidths).
"""
import json, time, sys
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW

SAMPLE_GPKG = "REPO_ROOT/data/03_processed/paper1/paper1_mgwr_ready_sample_50k.gpkg"
N = 10_000
SEED = 42
PREDICTORS = ["ndvi_2025", "ndbi_2025", "albedo_2025"]
DEP = "lst_2025"
OUT_GPKG = "outputs/paper1_mgwr_sample_true.gpkg"
OUT_SUMMARY = "outputs/tables/paper1_mgwr_true_summary.csv"
OUT_COEFS = "outputs/tables/paper1_mgwr_true_coefficients_summary.csv"
OUT_CONV = "outputs/mgwr_convergence_log.json"


def main():
    t_all = time.time()
    g = gpd.read_file(SAMPLE_GPKG)
    g = g.dropna(subset=[DEP] + PREDICTORS + ["lcz", "x", "y"]).reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(g), size=min(N, len(g)), replace=False)
    s = g.iloc[idx].reset_index(drop=True).copy()

    coords = s[["x", "y"]].to_numpy()
    y = s[[DEP]].to_numpy()
    X = s[PREDICTORS].to_numpy()
    xsc, ysc = StandardScaler(), StandardScaler()
    Xs = xsc.fit_transform(X)
    ys = ysc.fit_transform(y)

    # --- capture back-fitting convergence by tee-ing stdout during verbose search ---
    class Tee:
        def __init__(self, stream, buf): self.stream, self.buf = stream, buf
        def write(self, m): self.stream.write(m); self.buf.append(m)
        def flush(self): self.stream.flush()
    buf = []
    old = sys.stdout
    sys.stdout = Tee(old, buf)
    t0 = time.time()
    sel = Sel_BW(coords, ys, Xs, kernel="bisquare", fixed=False, multi=True, n_jobs=1)
    bws = sel.search(criterion="AICc", multi_bw_min=[2],
                     search_method="golden_section", tol_multi=1e-4,
                     max_iter_multi=200, verbose=True)
    t_search = time.time() - t0
    sys.stdout = old

    # parse convergence lines
    conv = []
    for line in "".join(buf).splitlines():
        if line.startswith("Current iteration"):
            parts = line.split(",")
            it = int(parts[0].split(":")[1])
            soc = float(parts[1].split(":")[1])
            conv.append({"iteration": it, "SOC": soc})

    t1 = time.time()
    model = MGWR(coords, ys, Xs, sel, kernel="bisquare", fixed=False,
                 sigma2_v1=True, n_jobs=1)
    res = model.fit()
    t_fit = time.time() - t1

    # per-predictor bandwidths (adaptive = number of nearest neighbours). order = [intercept, preds...]
    bw_names = ["intercept"] + PREDICTORS
    bw_map = {n: float(b) for n, b in zip(bw_names, bws)}

    # local condition numbers
    lc = res.local_collinearity()
    cond = np.asarray(lc[0]).flatten()  # local condition number

    # --- manual local R2 (reference-bandwidth adaptive bisquare neighbourhood) ---
    # mgwr does not implement localR2 for multiple bandwidths. Use the median of the
    # per-predictor (non-intercept) bandwidths as the reference neighbourhood size k,
    # and compute weighted local R2 = 1 - (sum w*e^2)/(sum w*(y-ybar_local)^2).
    pred_bws = [bw_map[p] for p in PREDICTORS]
    k_ref = int(round(np.median(pred_bws)))
    k_ref = max(k_ref, 5)
    tree = cKDTree(coords)
    dnn, inn = tree.query(coords, k=k_ref + 1)  # includes self
    e = np.asarray(res.resid_response).flatten()
    yv = ys.flatten()
    local_r2 = np.empty(len(coords))
    for i in range(len(coords)):
        nbr = inn[i]
        dd = dnn[i]
        dmax = dd[-1] if dd[-1] > 0 else 1.0
        w = (1 - (dd / dmax) ** 2) ** 2  # bisquare
        ybar = np.average(yv[nbr], weights=w)
        rss = np.sum(w * e[nbr] ** 2)
        tss = np.sum(w * (yv[nbr] - ybar) ** 2)
        local_r2[i] = 1 - rss / tss if tss > 0 else np.nan

    # write outputs
    params = np.asarray(res.params)
    tvals = np.asarray(res.tvalues)
    s["mgwr_intercept"] = params[:, 0]
    for j, p in enumerate(PREDICTORS, start=1):
        s[f"mgwr_coef_{p}"] = params[:, j]
        s[f"mgwr_t_{p}"] = tvals[:, j]
    s["mgwr_local_r2"] = local_r2
    s["mgwr_cond"] = cond
    s["mgwr_predicted_scaled"] = np.asarray(res.predy).flatten()
    s["mgwr_residual_scaled"] = e
    # back-transform residual to LST units for spatial diagnostics
    s["mgwr_residual_lst"] = e * ysc.scale_[0]

    gout = gpd.GeoDataFrame(s, geometry=gpd.points_from_xy(s["x"], s["y"], crs="EPSG:32635"),
                            crs="EPSG:32635")
    import os
    os.makedirs("outputs/tables", exist_ok=True)
    gout.to_file(OUT_GPKG, layer="paper1_mgwr_sample_true", driver="GPKG")

    summary = {
        "model": "true_mgwr_sample_10000",
        "n": int(len(s)),
        "bw_intercept": bw_map["intercept"],
        "bw_ndvi": bw_map["ndvi_2025"],
        "bw_ndbi": bw_map["ndbi_2025"],
        "bw_albedo": bw_map["albedo_2025"],
        "R2": float(res.R2),
        "adj_R2": float(res.adj_R2),
        "aicc": float(res.aicc),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "ENP": float(res.ENP),
        "local_r2_ref_k": k_ref,
        "local_r2_median": float(np.nanmedian(local_r2)),
        "local_r2_min": float(np.nanmin(local_r2)),
        "local_r2_max": float(np.nanmax(local_r2)),
        "local_r2_sd": float(np.nanstd(local_r2)),
        "cond_median": float(np.median(cond)),
        "cond_mean": float(np.mean(cond)),
        "cond_max": float(np.max(cond)),
        "cond_share_gt30_pct": float(100 * np.mean(cond > 30)),
        "n_backfit_iterations": len(conv),
        "final_SOC": conv[-1]["SOC"] if conv else None,
        "t_search_s": t_search,
        "t_fit_s": t_fit,
        "note": ("True multi-scale MGWR; per-predictor adaptive bisquare bandwidths via "
                 "AICc golden-section back-fitting; y and predictors standardized; n_jobs=1."),
    }
    pd.DataFrame([summary]).to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    # coefficient distribution table
    rows = []
    for col in ["mgwr_intercept"] + [f"mgwr_coef_{p}" for p in PREDICTORS] + ["mgwr_local_r2", "mgwr_cond"]:
        v = s[col]
        rows.append({"column": col, "mean": v.mean(), "std": v.std(), "min": v.min(),
                     "p05": v.quantile(0.05), "median": v.median(),
                     "p95": v.quantile(0.95), "max": v.max()})
    pd.DataFrame(rows).to_csv(OUT_COEFS, index=False, encoding="utf-8-sig")

    with open(OUT_CONV, "w") as f:
        json.dump({"bandwidths": bw_map, "convergence": conv,
                   "summary": summary}, f, indent=2)

    print("DONE total_s=%.1f" % (time.time() - t_all))
    print("bandwidths:", bw_map)
    print("R2=%.4f adjR2=%.4f ENP=%.1f iters=%d" % (res.R2, res.adj_R2, res.ENP, len(conv)))
    print("local_r2 median=%.4f  cond median=%.2f  cond>30=%.2f%%" % (
        np.nanmedian(local_r2), np.median(cond), 100 * np.mean(cond > 30)))


if __name__ == "__main__":
    main()
