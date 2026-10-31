"""Figure 5: temporal, morphological and space-time consistency of the LCZ-LST evidence.

(a) class-mean LST of every summer 2014-2026 with the yearly LCZ differentiation (eta^2_H) and the rank agreement
    with 2025 (Spearman rho) in a strip above; (b) mean 2025 LST of the built types under the municipal-inventory
    typology and under the WUDAPT map, with the inventory morphology of each WUDAPT class; (c) emerging hot- and
    cold-spot patterns of the 2014-2026 cube on the 1 km grid, with the pattern shares by dominant LCZ group inset.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

INK, MUTED, GRID = "#1F2933", "#7A8590", "#E7EBEA"
# study palette in (a) and (b); (c) keeps its own pattern colours
C1, C2, C3, C4 = "#A65F58", "#667F95", "#CF9A83", "#5F9C8E"
STEEL, LINE, HOLD = "#4F6780", "#CBD3D2", "#344150"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#B9BEC5", "axes.labelcolor": INK,
                     "xtick.color": "#667085", "ytick.color": "#667085", "axes.titlecolor": INK})

M = pd.read_csv(TABLES / "lcz_mean_lst_by_year.csv", index_col=0)
M.columns = M.columns.astype(int)
E = pd.read_csv(TABLES / "lcz_eta2_by_year.csv")
T1 = pd.read_csv(TABLES / "lcz_inventory_morphology.csv").set_index("lcz")
T3 = pd.read_csv(TABLES / "inventory_class_lst.csv")
U = pd.read_csv(TABLES / "lcz_lst_uncertainty.csv").set_index("lcz")

fig = plt.figure(figsize=(13, 16.2))
top = fig.add_gridspec(1, 2, left=0.075, right=0.845, top=0.955, bottom=0.645, wspace=0.34, width_ratios=[1.25, 1])
ax_map_box = [0.01, 0.005, 0.98, 0.555]


def tag(ax, s, title, x=-0.02, y=1.03):
    ax.text(x, y, f"{s} {title}", transform=ax.transAxes, fontsize=13.5, fontweight="bold", color=INK, ha="left", va="bottom")


def clean(ax):
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------- (a) class trajectories + yearly differentiation strip
sub = top[0, 0].subgridspec(2, 1, height_ratios=[1.15, 3.4], hspace=0.05)
axs, ax = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
years = list(M.columns)
for x0 in (2025,):
    for a in (axs, ax):
        a.axvspan(x0 - 0.45, x0 + 0.45, color="#F1F3F1", zorder=0)
hi = {3: ("LCZ 3 compact low-rise", C1), 8: ("LCZ 8 large low-rise", C3), 14: ("LCZ D low plants", C2), 11: ("LCZ A dense trees", C4)}
for k in [6, 9, 12, 13, 15, 16, 17]:
    ax.plot(years, M.loc[k], color=LINE, lw=1.1, zorder=1)
    if k in (12, 15, 17):
        ax.text(years[-1] + 0.3, M.loc[k, years[-1]], LCZ_LABEL[k], color=MUTED, fontsize=8, va="center")
for k, (lab, col) in hi.items():
    ax.plot(years, M.loc[k], color=col, lw=2.2, marker="o", ms=3.8, zorder=3)
    ax.text(years[-1] + 0.3, M.loc[k, years[-1]], LCZ_LABEL[k], color=col, fontsize=9.5, fontweight="bold", va="center")
ax.axvline(2018, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
ax.set_xticks(years); ax.set_xticklabels([str(y)[2:] for y in years], fontsize=9)
ax.set_xlabel("Summer (JJA), 20xx"); ax.set_ylabel("Class-mean LST (\u00b0C)")
ax.set_xlim(2013.5, 2027.0)
clean(ax)
h = [plt.Line2D([], [], color=c, lw=2.2, marker="o", ms=4, label=l) for l, c in hi.values()]
h.append(plt.Line2D([], [], color=LINE, lw=1.2, label="other LCZ types"))
ax.legend(handles=h, loc="lower left", fontsize=8.2, frameon=False, ncol=2, borderaxespad=0.2)
yl = ax.get_ylim(); ax.set_ylim(yl[0], yl[1] + 0.8)
ax.text(2018.15, yl[1] + 0.7, "LCZ map epoch", color=MUTED, fontsize=8, va="top")
ax.text(2025, yl[1] + 0.7, "analysed", color=MUTED, fontsize=8, va="top", ha="center")
# strip: eta^2 bars, rho as text
cols = [C1 if y == 2018 else (HOLD if y == 2025 else "#B9C7C6") for y in E.year]
axs.bar(E.year, E.eta2, color=cols, width=0.62, zorder=2)
for y, e, r in zip(E.year, E.eta2, E.rho_2025_types):
    axs.text(y, e + 0.012, f"{e:.2f}", ha="center", va="bottom", fontsize=7.6, color=INK)
    axs.text(y, -0.075, f"{r:.2f}", ha="center", va="center", fontsize=7.3, color=MUTED)
axs.set_ylim(-0.13, 0.56); axs.set_xlim(ax.get_xlim())
axs.set_xticks([]); axs.set_yticks([])
for s in ("top", "right", "left", "bottom"):
    axs.spines[s].set_visible(False)
axs.text(2013.35, 0.20, "LCZ $\\eta^2_H$", fontsize=8.6, color=INK, va="center", ha="right")
axs.text(2013.35, -0.075, "$\\rho$ with 2025", fontsize=7.8, color=MUTED, va="center", ha="right")
ax_a_main, ax_a_strip = ax, axs  # titles are placed at figure level at the end

# ---------------- (b) built types under the inventory typology vs WUDAPT, with WUDAPT-class morphology
ax = fig.add_subplot(top[0, 1])
order = [8, 3, 6, 2, 5, 9]  # rows top to bottom
names = {2: "Compact mid-rise", 3: "Compact low-rise", 5: "Open mid-rise", 6: "Open low-rise", 8: "Large low-rise", 9: "Sparsely built"}
T3["inv"] = T3["inv"].astype(int)
T3 = T3.set_index("inv")
ref = U.loc[11, "mean"]
ax.axvline(ref, color=C4, lw=1.3, ls=(0, (4, 3)), zorder=1)
ax.text(ref + 0.12, len(order) - 0.45, "LCZ A dense trees", color=C4, fontsize=8.2, ha="left", va="center")


def msize(n):
    return 18 + 22 * np.log10(max(n, 10))


for i, k in enumerate(order):
    y = len(order) - 1 - i
    r, u = T3.loc[k], U.loc[k]
    ax.plot([r["mean"], u["mean"]], [y + 0.13, y - 0.13], color=LINE, lw=1.2, zorder=1)
    ax.plot([r.ci_lo, r.ci_hi], [y + 0.13] * 2, color=C1, lw=2.2, solid_capstyle="butt", zorder=2)
    ax.scatter([r["mean"]], [y + 0.13], s=msize(r.n), color=C1, edgecolor="white", lw=1.0, zorder=3)
    ax.plot([u.ci_lo, u.ci_hi], [y - 0.13] * 2, color=C2, lw=2.2, solid_capstyle="butt", zorder=2)
    ax.scatter([u["mean"]], [y - 0.13], s=msize(u.n), color=C2, marker="s", edgecolor="white", lw=1.0, zorder=3)
    # right-hand columns: cell counts and the inventory morphology of the WUDAPT class
    tr = ax.get_yaxis_transform()
    ax.text(1.03, y, f"{int(r.n):,}\n{int(u.n):,}", transform=tr, ha="left", va="center", fontsize=10, color=INK, linespacing=1.25)
    mo = T1.loc[k]
    ax.text(1.27, y - 0.13, f"{100 * mo.bsf3_med:.0f}%, {mo.h3_med:.0f} m", transform=tr, ha="left", va="center", fontsize=10, color=STEEL)
ax.text(1.03, 1.005, "cells\ninv. / LCZ", transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=MUTED, linespacing=1.05)
ax.text(1.27, 1.005, "LCZ cells:\nBSF, height", transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=STEEL, linespacing=1.05)
ax.set_yticks(range(len(order))); ax.set_yticklabels([names[k] for k in order[::-1]], fontsize=9.5, color=INK)
ax.set_ylim(-0.6, len(order) - 0.2)
ax.set_xlim(36.4, 49.4)
ax.set_xlabel("Mean 2025 LST (\u00b0C), 95% block-bootstrap interval")
clean(ax); ax.grid(False, axis="y")
h = [plt.Line2D([], [], color=C1, marker="o", lw=2, label="inventory typology (2025\u201326 buildings)"),
     plt.Line2D([], [], color=C2, marker="s", lw=2, label="WUDAPT global LCZ (2018)")]
ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.55, -0.11), ncol=2, fontsize=8.2, frameon=False)
ax_b = ax

# ---------------- (c) emerging hot- and cold-spot patterns, full width, shares inset
Z = np.load(CACHE / "ehsa_1km.npz", allow_pickle=True)
ORD, COLS, PRES = list(Z["order"]), list(Z["col"]), Z["present"]
ext = Z["ext"]
ax = fig.add_axes(ax_map_box)
ax.imshow(Z["code"], extent=ext, cmap=ListedColormap(COLS), norm=BoundaryNorm(np.arange(-0.5, len(ORD)), len(ORD)), interpolation="nearest")
ax.set_axis_off()
TITLE_C = "(c) Emerging hot- and cold-spot patterns of the summer LST cube, 2014\u20132026 (1 km cells)"
h = [plt.Rectangle((0, 0), 1, 1, color=COLS[i]) for i in range(len(ORD)) if PRES[i]]
ax.legend(h, [o for i, o in enumerate(ORD) if PRES[i]], loc="upper left", bbox_to_anchor=(0.045, 0.90), fontsize=10.6, frameon=False,
          handlelength=1.7, handleheight=1.3, labelspacing=0.5)
# north arrow and segmented scale bar in the style of the study-area maps (Figure 2); true scale, extent in metres
SOFT, SEG = "#91A4AD", ("#D5E0E3", "#A9BCC3")
x0, y0, seg, hb = ext[0] + 0.72 * (ext[1] - ext[0]), ext[2] + 0.055 * (ext[3] - ext[2]), 10000.0, 1500.0
for i in range(4):
    ax.add_patch(plt.Rectangle((x0 + i * seg, y0), seg, hb, facecolor=SEG[i % 2], edgecolor="white", linewidth=0.8, zorder=22))
for i in range(5):
    ax.plot([x0 + i * seg] * 2, [y0, y0 - 1300], color=SOFT, lw=0.8, zorder=23)
    if i % 2 == 0:
        ax.text(x0 + i * seg, y0 - 2100, f"{10 * i}" + (" km" if i == 4 else ""), ha="center", va="top", fontsize=9.5, color=SOFT, zorder=23)
ax.annotate("N", xy=(x0 + 4 * seg + 9500, y0 + 12500), xytext=(x0 + 4 * seg + 9500, y0 + 1800), ha="center", va="center", fontsize=12, fontweight="bold",
            color=SOFT, arrowprops=dict(arrowstyle="-|>", color=SOFT, lw=1.2, shrinkA=0, shrinkB=0), zorder=23)
# inset: pattern shares by dominant LCZ group (horizontal stacked bars) in the empty north-east of the map
ins = ax.inset_axes([0.635, 0.45, 0.345, 0.25])
tab = pd.read_csv(TABLES / "ehsa_by_lcz_group.csv", index_col=0)
tab = tab.loc[[g for g in ["Built LCZ 2-10", "Low vegetation and bare LCZ C-F", "Trees LCZ A-B", "Water LCZ G"] if g in tab.index]]
left = np.zeros(len(tab))
for i, o in enumerate(ORD):
    vv = tab[o].to_numpy() if o in tab else np.zeros(len(tab))
    if vv.sum() == 0:
        continue
    ins.barh(range(len(tab)), vv, left=left, color=COLS[i], height=0.66, edgecolor="white", linewidth=0.6)
    left += vv
lab = {"Built LCZ 2-10": "Built LCZ 2\u201310", "Low vegetation and bare LCZ C-F": "Low veg./bare C\u2013F",
       "Trees LCZ A-B": "Trees LCZ A\u2013B", "Water LCZ G": "Water LCZ G"}
ins.set_yticks(range(len(tab))); ins.set_yticklabels([lab[t] for t in tab.index], fontsize=8.6, color=INK)
ins.invert_yaxis()
ins.set_xlim(0, 1); ins.set_xticks([0, 0.25, 0.5, 0.75, 1]); ins.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=8)
ins.tick_params(length=0)
for s in ("top", "right", "left"):
    ins.spines[s].set_visible(False)
ins.set_title("Pattern shares by dominant LCZ group", fontsize=9.6, color=INK, loc="left", fontweight="bold")

# ---------------- panel titles aligned at figure level: (a) and (b) on one baseline, (a) and (c) on one left edge,
# each title starting at the left edge of its panel's labels
fig.canvas.draw()
r = fig.canvas.get_renderer()
inv = fig.transFigure.inverted()


def fbox(*axes):
    bb = [inv.transform_bbox(a.get_tightbbox(r)) for a in axes]
    return min(b.x0 for b in bb), max(b.y1 for b in bb)


xa, ya = fbox(ax_a_main, ax_a_strip)
xb, yb = fbox(ax_b)
ytop = max(ya, yb) + 0.006
TS = dict(fontsize=13.5, fontweight="bold", color=INK, ha="left", va="bottom")
fig.text(xa, ytop, "(a) LCZ thermal ordering, 2014\u20132026", **TS)
fig.text(xb, ytop, "(b) Built types under a local typology", **TS)
fig.text(xa, 0.575, TITLE_C, **TS)

out = Path(r"REPO_ROOT\outputs\figures\paper1\final\fig05_lcz_evidence_consistency.png")
fig.savefig(out, dpi=300)
fig.savefig(FIGS / out.name, dpi=150)
print(out)
