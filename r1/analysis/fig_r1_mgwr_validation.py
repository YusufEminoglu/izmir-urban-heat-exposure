"""New Figure: out-of-sample performance and stability of the MGWR solution."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

INK, MUTED, GRID = "#1F2933", "#667085", "#E6E8EB"
MODEL_COL = {"MGWR": "#B5432C", "OLS3": "#2F6FB0", "OLSfull": "#B98A1E", "SEMtrend": "#1E8A5A"}
MODEL_LAB = {"MGWR": "MGWR (kernel-consistent)", "OLS3": "OLS, 3 surface predictors", "OLSfull": "OLS, full specification",
             "SEMtrend": "SEM trend $X\\hat\\beta$"}
COEF_COL = {"ndvi": "#1E8A5A", "ndbi": "#B5432C", "albedo": "#B98A1E"}
COEF_LAB = {"ndvi": "NDVI", "ndbi": "NDBI", "albedo": "Albedo"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#B9BEC5", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED})

CV = pd.read_csv(TABLES / "validation_summary.csv")
ST = pd.read_csv(TABLES / "mgwr_stability.csv")
IS = pd.read_csv(TABLES / "atlas_interpolation_sensitivity.csv")
MED = pd.read_csv(TABLES / "mgwr_stability_lcz_medians.csv")

fig = plt.figure(figsize=(13, 9.8))
gs = fig.add_gridspec(2, 2, hspace=0.36, wspace=0.24, left=0.07, right=0.97, top=0.95, bottom=0.08)


def tag(ax, s, title):
    ax.text(-0.02, 1.06, f"{s} {title}", transform=ax.transAxes, fontsize=13, fontweight="bold", color=INK, ha="left", va="bottom")


def clean(ax):
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# (a) RMSE by validation setting and model
ax = fig.add_subplot(gs[0, 0])
settings = [("rnd", "sample", "Random 5-fold\n(sample cells)"), ("blk", "sample", "Spatial blocks\n(sample cells)"),
            ("blk", "grid", "Spatial blocks\n(all 100 m cells)"), (None, "grid_holdout", "Full-grid hold-out\n(1.18 M cells)")]
models = ["MGWR", "OLSfull", "SEMtrend", "OLS3"]
for i, (sc, st, lab) in enumerate(settings):
    for j, m in enumerate(models):
        if sc is None:
            mm = {"MGWR": "MGWR_kernel"}.get(m, m)
            r = CV[(CV["set"] == st) & (CV.model == mm)]
        else:
            r = CV[(CV.scheme == sc) & (CV["set"] == st) & (CV.model == m)]
        if r.empty:
            continue
        x = i + (j - 1.5) * 0.17
        v = r.rmse.iloc[0]
        ax.scatter([x], [v], s=70, color=MODEL_COL[m], edgecolor="white", lw=1.2, zorder=3, marker="o" if m == "MGWR" else "s")
        ax.plot([x, x], [0, v], color=MODEL_COL[m], lw=1.4, alpha=0.6)
        if m == "MGWR":
            ax.text(x, v - 0.12, f"{v:.2f}", ha="center", va="top", fontsize=8, color=INK)
ax.set_xticks(range(len(settings))); ax.set_xticklabels([s[2] for s in settings], fontsize=8.5)
ax.set_ylabel("Held-out RMSE (\u00b0C)"); ax.set_ylim(0, 5.0)
clean(ax)
h = [plt.Line2D([], [], color=MODEL_COL[m], marker="o" if m == "MGWR" else "s", lw=0, ms=7, label=MODEL_LAB[m]) for m in models]
ax.legend(handles=h, loc="upper left", fontsize=8, frameon=False, ncol=2)
tag(ax, "(a)", "Out-of-sample error")

# (b) agreement with the reported model at the planning scale (mahalle means) and across interpolators
ax = fig.add_subplot(gs[0, 1])
PS = pd.read_csv(TABLES / "mgwr_stability_planning_scale.csv")
PS = PS[PS.fit != "orig"]  # v2: the reported model is the fit of this sample
COEF_COL4 = {"ndvi": "#1E8A5A", "ndbi": "#B5432C", "alb": "#B98A1E", "C": "#2F6FB0"}
COEF_LAB4 = {"ndvi": "NDVI", "ndbi": "NDBI", "alb": "Albedo", "C": "Surface heat term"}
lab = {"orig": "Refit, same sample", "rep1": "New sample 1", "rep2": "New sample 2", "rep3": "New sample 3", "rep4": "New sample 4",
       "rep5": "New sample 5", "n2000": "n = 2,000", "aug": "+ elevation, canopy"}
rows = [(lab.get(r.fit, r.fit), {k: r[f"mah_r_{k}"] for k in COEF_COL4}) for _, r in PS.iterrows()]
ilab = {"idw_p1_k12": "IDW power 1", "idw_p2_k6": "IDW 6 neighbours", "idw_p2_k24": "IDW 24 neighbours", "nearest": "Nearest neighbour",
        "linear": "Delaunay linear", "lcz_strat_idw": "IDW within built/natural", "kernel_direct": "Kernel-consistent"}
irows = [(ilab.get(r.method, r.method), {"ndvi": r.r_ndvi, "ndbi": r.r_ndbi, "alb": r.r_alb}) for _, r in IS[IS.method != "idw_p2_k12"].iterrows()]
allrows = rows + irows
y = np.arange(len(allrows))[::-1]
for yi, (name, d) in zip(y, allrows):
    vals = list(d.values())
    ax.plot([min(vals), max(vals)], [yi, yi], color="#C3C8CF", lw=1, zorder=1)
    for k, v in d.items():
        ax.scatter([v], [yi], color=COEF_COL4[k], s=46 if k == "C" else 36, edgecolor="white", lw=0.8, zorder=3,
                   marker="D" if k == "C" else "o")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in allrows], fontsize=8)
ax.axhline(y[len(rows) - 1] - 0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
ax.text(-0.18, y[0] + 0.45, "other fits: mahalle means", fontsize=7.5, color=MUTED)
ax.text(0.25, y[len(rows)] + 0.40, "atlas interpolators: 500 m grid", fontsize=7.5, color=MUTED, va="top")
ax.set_xlim(-0.2, 1.03)
ax.set_xlabel("Pearson r with the reported model")
clean(ax)
h = [plt.Line2D([], [], color=COEF_COL4[k], marker="D" if k == "C" else "o", lw=0, ms=7, label=COEF_LAB4[k]) for k in COEF_COL4]
ax.legend(handles=h, loc="lower left", fontsize=8, frameon=False, ncol=2)
tag(ax, "(b)", "Agreement with the reported model")

# (c) LCZ-class medians across fits
ax = fig.add_subplot(gs[1, 0])
fits = ["r0", "rep1", "rep2", "rep3", "rep4", "rep5", "n2000"]
M2 = MED[MED.fit.isin(fits)]
order = [3, 6, 8, 9, 11, 12, 13, 14]
for i, c in enumerate(order):
    for j, k in enumerate(["ndvi", "ndbi", "albedo"]):
        v = M2[M2.lcz == c][k]
        x = i + (j - 1) * 0.22
        ax.plot([x, x], [v.min(), v.max()], color=COEF_COL[k], lw=3, alpha=0.45, solid_capstyle="round")
        ax.scatter([x], [M2[(M2.lcz == c) & (M2.fit == "r0")][k].iloc[0]], color=COEF_COL[k], s=30, edgecolor="white", lw=0.8, zorder=3)
ax.axhline(0, color=MUTED, lw=0.8)
ax.set_xticks(range(len(order))); ax.set_xticklabels([f"LCZ {LCZ_LABEL[c]}" for c in order], fontsize=8.5)
ax.set_ylabel("Median local coefficient (standardised)"); ax.set_ylim(-0.1, 0.6)
clean(ax)
ax.text(0.99, 0.97, "dot: reported model; bar: range over\nfive new samples and n = 2,000",
        transform=ax.transAxes, ha="right", va="top", fontsize=8, color=MUTED)
h3 = [plt.Line2D([], [], color=COEF_COL[k], marker="o", lw=0, ms=7, label=COEF_LAB[k]) for k in COEF_COL]
ax.legend(handles=h3, loc="upper left", fontsize=8, frameon=False, ncol=3)
tag(ax, "(c)", "LCZ-class coefficients across samples")

# (d) full-grid hold-out error by distance to the nearest calibration cell
ax = fig.add_subplot(gs[1, 1])
D = CV[CV["set"].astype(str).str.startswith("dist_")].copy()
labs = {"dist_0_500": "< 0.5", "dist_500_1000": "0.5\u20131", "dist_1000_2000": "1\u20132", "dist_2000_99999": "> 2"}
D["lab"] = D["set"].map(labs)
ax.bar(range(len(D)), D.rmse, color="#B5432C", alpha=0.85, width=0.6)
for i, (_, r) in enumerate(D.iterrows()):
    ax.text(i, r.rmse + 0.03, f"R\u00b2 {r.r2:.2f}\nn = {int(r.n):,}", ha="center", va="bottom", fontsize=8, color=INK)
ref = CV[(CV["set"] == "grid_holdout") & (CV.model == "OLSfull")].rmse
if len(ref):
    ax.axhline(ref.iloc[0], color="#B98A1E", lw=1.4, ls=(0, (4, 3)))
    ax.text(-0.45, ref.iloc[0] + 0.05, "OLS, full specification", color="#B98A1E", fontsize=8, ha="left", va="bottom")
ax.set_xticks(range(len(D))); ax.set_xticklabels(D["lab"])
ax.set_xlabel("Distance to nearest calibration cell (km)"); ax.set_ylabel("MGWR hold-out RMSE (\u00b0C)")
ax.set_ylim(0, max(D.rmse.max(), ref.iloc[0] if len(ref) else 0) * 1.35)
clean(ax)
tag(ax, "(d)", "Full-grid prediction by distance")

out = Path(r"REPO_ROOT\outputs\figures\paper1\final\fig09_mgwr_validation_stability.png")
fig.savefig(out, dpi=300)
fig.savefig(FIGS / out.name, dpi=150)
print(out)
