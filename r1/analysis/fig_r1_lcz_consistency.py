"""New Figure: temporal and morphological consistency of the LCZ-LST evidence."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

INK, MUTED, GRID = "#1F2933", "#667085", "#E6E8EB"
C1, C2, C3, C4 = "#B5432C", "#2F6FB0", "#B98A1E", "#1E8A5A"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#B9BEC5", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK})

M = pd.read_csv(TABLES / "lcz_mean_lst_by_year.csv", index_col=0)
M.columns = M.columns.astype(int)
E = pd.read_csv(TABLES / "lcz_eta2_by_year.csv")
T1 = pd.read_csv(TABLES / "lcz_inventory_morphology.csv")
T3 = pd.read_csv(TABLES / "inventory_class_lst.csv")
U = pd.read_csv(TABLES / "lcz_lst_uncertainty.csv").set_index("lcz")

fig = plt.figure(figsize=(13, 14.8))
gs = fig.add_gridspec(3, 2, hspace=0.36, wspace=0.22, left=0.06, right=0.97, top=0.965, bottom=0.085, height_ratios=[1, 1, 1.05])


def tag(ax, s, title):
    ax.text(-0.02, 1.06, f"{s} {title}", transform=ax.transAxes, fontsize=13, fontweight="bold", color=INK, ha="left", va="bottom")


def clean(ax):
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# (a) class trajectories
ax = fig.add_subplot(gs[0, 0])
years = list(M.columns)
hi = {3: ("LCZ 3 compact low-rise", C1), 8: ("LCZ 8 large low-rise", C3), 14: ("LCZ D low plants", C2), 11: ("LCZ A dense trees", C4)}
for k in [6, 9, 12, 13, 15, 16, 17]:
    ax.plot(years, M.loc[k], color="#C3C8CF", lw=1.2, zorder=1)
    if k in (12, 15, 17):
        ax.text(years[-1] + 0.25, M.loc[k, years[-1]], LCZ_LABEL[k], color=MUTED, fontsize=8, va="center")
for k, (lab, col) in hi.items():
    ax.plot(years, M.loc[k], color=col, lw=2.2, marker="o", ms=4, zorder=3)
    ax.text(years[-1] + 0.25, M.loc[k, years[-1]], LCZ_LABEL[k], color=INK, fontsize=9, fontweight="bold", va="center")
ax.axvline(2018, color=MUTED, lw=1, ls=(0, (3, 3)))
ax.text(2018.1, ax.get_ylim()[1] if False else M.max().max() + 0.4, "LCZ map epoch", color=MUTED, fontsize=8, va="bottom")
ax.set_xticks(years); ax.set_xticklabels([str(y)[2:] for y in years])
ax.set_xlabel("Summer (JJA) composite, 20xx"); ax.set_ylabel("Class-mean LST (\u00b0C)")
ax.set_xlim(2013.6, 2026.9)
clean(ax)
h = [plt.Line2D([], [], color=c, lw=2.2, marker="o", ms=4, label=l) for l, c in hi.values()]
h.append(plt.Line2D([], [], color="#C3C8CF", lw=1.2, label="other LCZ types"))
ax.legend(handles=h, loc="lower left", fontsize=8, frameon=False, ncol=2)
tag(ax, "(a)", "LCZ thermal ordering, 2014\u20132026")

# (b) eta2 by year
ax = fig.add_subplot(gs[0, 1])
cols = ["#9AA5B1" if y != 2018 else C1 for y in E.year]
cols[list(E.year).index(2025)] = INK  # analysed summer
ax.bar(E.year, E.eta2, color=cols, width=0.72)
for y, e, r in zip(E.year, E.eta2, E.rho_2025_types):
    ax.text(y, e + 0.006, f"{e:.2f}", ha="center", fontsize=8, color=INK)
    ax.text(y, 0.012, f"{r:.2f}", ha="center", fontsize=7.5, color="white", rotation=90, va="bottom")
ax.set_ylim(0, 0.5); ax.set_ylabel("Kruskal\u2013Wallis $\\eta^2_H$ (LCZ)")
ax.set_xticks(E.year); ax.set_xticklabels([str(y)[2:] for y in E.year]); ax.set_xlabel("Summer, 20xx")
clean(ax)
ax.text(0.99, 0.97, "bar label: $\\eta^2_H$   inside bar: Spearman $\\rho$ of\nthe 14 class means with 2025", transform=ax.transAxes,
        ha="right", va="top", fontsize=8, color=MUTED)
tag(ax, "(b)", "LCZ differentiation by year")

# (c) inventory morphology vs Stewart-Oke ranges
ax = fig.add_subplot(gs[1, 0])
SO = {2: (40, 70, 10, 25), 3: (40, 70, 3, 10), 6: (20, 40, 3, 10), 8: (30, 50, 3, 10), 9: (10, 20, 3, 10)}
colmap = {2: C1, 3: "#D97A5E", 6: C2, 8: C3, 9: C4}
for k, (b0, b1, h0, h1) in SO.items():
    ax.add_patch(Rectangle((b0, h0), b1 - b0, h1 - h0, fc=colmap[k], alpha=0.10, ec=colmap[k], lw=0.8, ls="--"))
for _, r in T1.iterrows():
    k = int(r.lcz)
    if k not in colmap:
        continue
    x, y = 100 * r.bsf3_med, r.h3_med
    ax.plot([100 * r.bsf3_q25, 100 * r.bsf3_q75], [y, y], color=colmap[k], lw=1.6)
    ax.plot([x, x], [r.h3_q25, r.h3_q75], color=colmap[k], lw=1.6)
    ax.scatter([x], [y], s=60, color=colmap[k], edgecolor="white", lw=1.5, zorder=3)
    dx, dy, ha = {6: (0.3, 1.7, "left"), 8: (1.8, -1.6, "left"), 9: (1.2, -1.4, "left")}.get(k, (1.2, 0.6, "left"))
    ax.text(x + dx, y + dy, f"LCZ {k} (n={int(r.n):,})", fontsize=8.5, color=INK, ha=ha)
ax.set_xlim(0, 75); ax.set_ylim(0, 27)
ax.set_xlabel("Building surface fraction, 300 m window (%)"); ax.set_ylabel("Footprint-weighted height (m)")
clean(ax)
ax.text(0.99, 0.03, "dot: median; bars: interquartile range\ndashed boxes: Stewart & Oke (2012) ranges",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED)
tag(ax, "(c)", "Municipal building inventory by WUDAPT class")

# (d) LST by inventory class vs WUDAPT class
ax = fig.add_subplot(gs[1, 1])
order = ["2", "3", "5", "6", "8", "9"]
names = {"2": "Compact\nmid-rise", "3": "Compact\nlow-rise", "5": "Open\nmid-rise", "6": "Open\nlow-rise", "8": "Large\nlow-rise", "9": "Sparsely\nbuilt"}
T3["inv"] = T3["inv"].astype(str)
T3 = T3.set_index("inv")
xs = np.arange(len(order))
for i, k in enumerate(order):
    r = T3.loc[k]
    ax.plot([i - 0.12, i - 0.12], [r.ci_lo, r.ci_hi], color=C1, lw=2)
    ax.scatter([i - 0.12], [r["mean"]], color=C1, s=55, zorder=3, edgecolor="white", lw=1.2)
    ax.text(i - 0.12, r.ci_lo - 0.35, f"{int(r.n):,}", ha="center", va="top", fontsize=7.5, color=C1)
    u = U.loc[int(k)]
    ax.plot([i + 0.12, i + 0.12], [u.ci_lo, u.ci_hi], color=C2, lw=2)
    ax.scatter([i + 0.12], [u["mean"]], color=C2, s=55, zorder=3, edgecolor="white", lw=1.2, marker="s")
    ax.text(i + 0.12, u.ci_lo - 0.35, f"{int(u.n):,}", ha="center", va="top", fontsize=7.5, color=C2)
ax.axhline(U.loc[11, "mean"], color=C4, lw=1.2, ls=(0, (4, 3)))
ax.text(len(order) - 0.5, U.loc[11, "mean"] + 0.2, "LCZ A dense trees", color=C4, fontsize=8, ha="right")
ax.set_xticks(xs); ax.set_xticklabels([names[k] for k in order], fontsize=8.5)
ax.set_ylabel("Mean summer 2025 LST (\u00b0C), 95% block-bootstrap CI")
ax.set_ylim(36.5, 51.5)
clean(ax)
h = [plt.Line2D([], [], color=C1, marker="o", lw=2, label="inventory-derived class (2025\u201326 buildings)"),
     plt.Line2D([], [], color=C2, marker="s", lw=2, label="WUDAPT global LCZ (2018)")]
ax.legend(handles=h, loc="upper left", fontsize=8, frameon=False)
tag(ax, "(d)", "Thermal ordering under a local typology")

# (e)-(f) emerging hot- and cold-spot patterns of the 2014-2026 cube (computed in fig_r2_datacube.py)
from matplotlib.colors import ListedColormap, BoundaryNorm
E = np.load(CACHE / "ehsa_1km.npz", allow_pickle=True)
ORD, COLS, PRES = list(E["order"]), list(E["col"]), E["present"]
ax = fig.add_subplot(gs[2, 0])
ax.imshow(E["code"], extent=E["ext"], cmap=ListedColormap(COLS), norm=BoundaryNorm(np.arange(-0.5, len(ORD)), len(ORD)), interpolation="nearest")
ax.set_axis_off()
h = [plt.Rectangle((0, 0), 1, 1, color=COLS[i]) for i in range(len(ORD)) if PRES[i]]
tag(ax, "(e)", "Emerging hot- and cold-spot patterns, 2014\u20132026")
ax = fig.add_subplot(gs[2, 1])
tab = pd.read_csv(TABLES / "ehsa_by_lcz_group.csv", index_col=0)
left = np.zeros(len(tab))
for i, o in enumerate(ORD):
    vv = tab[o].to_numpy() if o in tab else np.zeros(len(tab))
    if vv.sum() == 0:
        continue
    ax.barh(range(len(tab)), vv, left=left, color=COLS[i], height=0.6, edgecolor="white", linewidth=0.6)
    left += vv
ax.set_yticks(range(len(tab))); ax.set_yticklabels([t.replace(" LCZ", "\nLCZ").replace("vegetation and", "vegetation\nand") for t in tab.index], fontsize=9); ax.set_xlim(0, 1)
ax.legend(h, [o for i, o in enumerate(ORD) if PRES[i]], loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=3, fontsize=7.8, frameon=False)
ax.set_xlabel("share of 1 km cells"); ax.grid(True, axis="x", color="#E6E8EB")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
tag(ax, "(f)", "Pattern shares by dominant LCZ group")

out = Path(r"REPO_ROOT\outputs\figures\paper1\final\fig05_lcz_evidence_consistency.png")
fig.savefig(out, dpi=300)
fig.savefig(FIGS / out.name, dpi=150)
print(out)
