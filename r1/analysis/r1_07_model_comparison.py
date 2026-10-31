"""Model comparison on a common footing and the residual definitions behind Moran's I.

On the reported 4,000-cell sample (standardised LST and predictors, KNN k=8
row-standardised W, the matrix used for both estimation and diagnostics):
  * OLS, ML spatial error (SEM) and ML spatial lag (SAR) refit with spreg
  * Moran's I of each residual definition:
      SEM  u = y - X b (trend residual, contains lambda W u)
           e = u - lambda W u (spatially filtered innovation)
      SAR  e = y - rho W y - X b (innovation)
           u = y - (I - rho W)^-1 X b (reduced-form prediction residual)
  * Gaussian log-likelihood, AIC, AICc on the same y for all four models
  * MGWR residual Moran's I under k = 4, 8, 16 and a 1.5 km distance band
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from libpysal.weights import KNN, DistanceBand
from esda.moran import Moran
import spreg

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

s = gpd.read_file(MGWR_SAMPLE)
coords = s[["x", "y"]].to_numpy(float)
y = s[DEP].to_numpy(float)
X = s[PREDICTORS].to_numpy(float)
ys = ((y - y.mean()) / y.std()).reshape(-1, 1)
Xs = (X - X.mean(0)) / X.std(0)
n = len(ys)
w = KNN.from_array(coords, k=8); w.transform = "r"
Wd = w.full()[0]


def moran(v, ww=w):
    m = Moran(v.ravel(), ww, permutations=999)
    return m.I, m.p_sim, m.z_sim


ols = spreg.OLS(ys, Xs)
sem = spreg.ML_Error(ys, Xs, w=w, method="full")
sar = spreg.ML_Lag(ys, Xs, w=w, method="full")

lam = float(sem.lam)
u_sem = sem.u.ravel()
e_sem = sem.e_filtered.ravel()
rho = float(sar.rho)
b_sar = sar.betas[:-1].ravel()
Z = np.column_stack([np.ones(n), Xs])
e_sar = sar.u.ravel()  # y - rho Wy - Xb
u_sar_rf = ys.ravel() - np.linalg.solve(np.eye(n) - rho * Wd, Z @ b_sar)
u_sar_pr = (ys.ravel() - sar.predy_e.ravel()) if hasattr(sar, "predy_e") else u_sar_rf

R = {}
R["ols"] = moran(ols.u)
R["sem_u"] = moran(u_sem)
R["sem_e"] = moran(e_sem)
R["sar_e"] = moran(e_sar)
R["sar_rf"] = moran(u_sar_rf)
for k, v in R.items():
    print(k, np.round(v, 4))


def gauss_ll(res, k):
    s2 = (res ** 2).mean()
    ll = -n / 2 * (np.log(2 * np.pi * s2) + 1)
    return ll


ll_ols = gauss_ll(ols.u.ravel(), 4)
k_ols, k_sem, k_sar = 5, 6, 6  # betas + sigma2 (+ lambda/rho)
aic = lambda ll, k: -2 * ll + 2 * k
aicc = lambda ll, k: aic(ll, k) + 2 * k * (k + 1) / (n - k - 1)
print("OLS R2", ols.r2, "ll", ll_ols, float(ols.logll))
print("SEM pr2", sem.pr2, "ll", float(sem.logll), "aic", float(sem.aic), "lam", lam)
print("SAR pr2", sar.pr2, "pr2_e", getattr(sar, "pr2_e", None), "ll", float(sar.logll), "aic", float(sar.aic), "rho", rho)

# MGWR (refit 'orig' if available, else stored residuals)
fz = CACHE / "mgwr_orig.npz"
if fz.exists():  # v2: the reported fit itself (r2_28)
    f = np.load(fz)
    e_m = f["resid"]; enp = float(f["ENP"]); ll_m = gauss_ll(e_m, 0); aicc_m = float(f["aicc"]); aic_m = float(f["aic"]); r2_m = float(f["R2"])
else:
    e_m = s["mgwr_residual_scaled"].to_numpy(); enp = 885.12; r2_m = 0.8802
    ll_m = gauss_ll(e_m, 0); aic_m = 4636.73; aicc_m = 5141.79
R["mgwr_k8"] = moran(e_m)
for kk in (4, 16):
    wk = KNN.from_array(coords, k=kk); wk.transform = "r"
    R[f"mgwr_k{kk}"] = moran(e_m, wk)
    R[f"ols_k{kk}"] = moran(ols.u, wk)
wdb = DistanceBand.from_array(coords, threshold=1500, binary=True, silence_warnings=True); wdb.transform = "r"
R["mgwr_db"] = moran(e_m, wdb); R["ols_db"] = moran(ols.u, wdb)
for k, v in R.items():
    print(k, np.round(v, 4))

tab = pd.DataFrame([
    dict(model="OLS", r2=float(ols.r2), ll=float(ols.logll), k=k_ols, aic=aic(float(ols.logll), k_ols), aicc=aicc(float(ols.logll), k_ols),
         moran=R["ols"][0]),
    dict(model="SEM", r2=float(sem.pr2), ll=float(sem.logll), k=k_sem, aic=aic(float(sem.logll), k_sem), aicc=aicc(float(sem.logll), k_sem), moran=R["sem_u"][0], moran_alt=R["sem_e"][0]),
    dict(model="SAR", r2=float(sar.pr2), ll=float(sar.logll), k=k_sar, aic=aic(float(sar.logll), k_sar), aicc=aicc(float(sar.logll), k_sar), moran=R["sar_e"][0], moran_alt=R["sar_rf"][0]),
    dict(model="MGWR", r2=r2_m, ll=ll_m, k=enp, aic=aic_m, aicc=aicc_m, moran=R["mgwr_k8"][0]),
])
tab.to_csv(TABLES / "model_comparison_common.csv", index=False)
print(tab)

vals = {
    "ols_sample_r2": fmt(float(ols.r2)), "ols_sample_ll": fmt(float(ols.logll), 1), "ols_sample_aic": fmt(aic(float(ols.logll), k_ols), 1),
    "ols_sample_resid_morans_i": fmt(R["ols"][0]),
    "sem_pseudo_r2": fmt(float(sem.pr2)), "sem_ll": fmt(float(sem.logll), 1), "sem_aic": fmt(aic(float(sem.logll), k_sem), 1), "sem_lambda": fmt(lam),
    "sem_resid_morans_i": fmt(R["sem_u"][0]), "sem_filtered_morans_i": fmt(R["sem_e"][0]), "sem_filtered_morans_p": fmt(R["sem_e"][1]),
    "sar_pseudo_r2": fmt(float(sar.pr2)), "sar_ll": fmt(float(sar.logll), 1), "sar_aic": fmt(-2 * float(sar.logll) + 2 * 6, 1), "sar_rho": fmt(rho),
    "sar_resid_morans_i": fmt(R["sar_e"][0]), "sar_rf_morans_i": fmt(R["sar_rf"][0]),
    "mgwr_ll": fmt(ll_m, 1), "mgwr_aic": fmt(aic_m, 1),
    "mgwr_resid_morans_i_k4": fmt(R["mgwr_k4"][0]), "mgwr_resid_morans_i_k16": fmt(R["mgwr_k16"][0]),
    "mgwr_resid_morans_i_db": fmt(R["mgwr_db"][0]), "mgwr_resid_morans_z": fmt(R["mgwr_k8"][2], 1),
    "ols_resid_morans_i_k4": fmt(R["ols_k4"][0]), "ols_resid_morans_i_k16": fmt(R["ols_k16"][0]), "ols_resid_morans_i_db": fmt(R["ols_db"][0]),
    "mgwr_resid_morans_i_rc": fmt(R["mgwr_k8"][0]),
}
write_values("r1_07_model_comparison", vals)
