"""Generate core Study 1 figures for BBTMK and manuscript drafting."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _config import FIGURES_DIR, DATA_PROCESSED, TABLES_DIR, ensure_dirs  # noqa: E402
from _config import get_logger  # noqa: E402


MODEL_GRID = DATA_PROCESSED / "paper1_model_grid_100m_current.csv"
LCZ_SUMMARY = TABLES_DIR / "paper1_lcz_lst_summary.csv"
CORRELATIONS = TABLES_DIR / "paper1_predictor_correlations.csv"
OLS_COEFS = TABLES_DIR / "paper1_ols_coefficients.csv"

FIG_LCZ_LST = FIGURES_DIR / "paper1_lcz_lst_mean_bar.png"
FIG_CORRELATION = FIGURES_DIR / "paper1_spearman_correlation_heatmap.png"
FIG_OLS_COEFS = FIGURES_DIR / "paper1_ols_key_coefficients.png"
FIG_SCATTER = FIGURES_DIR / "paper1_lst_ndvi_ndbi_scatter.png"


def save_lcz_bar() -> None:
    """LCZ mean LST bar chart."""
    df = pd.read_csv(LCZ_SUMMARY).sort_values("mean", ascending=False)
    plt.figure(figsize=(9, 5.2))
    ax = sns.barplot(df, x="lcz", y="mean", hue="lcz", palette="mako", legend=False)
    ax.errorbar(
        x=range(len(df)),
        y=df["mean"],
        yerr=df["std"],
        fmt="none",
        ecolor="#333333",
        elinewidth=0.8,
        capsize=2,
    )
    ax.set_xlabel("LCZ class")
    ax.set_ylabel("Mean LST 2025 (deg C)")
    ax.set_title("Mean 2025 Summer LST by LCZ Class")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_LCZ_LST, dpi=300)
    plt.close()


def save_correlation_heatmap() -> None:
    """Spearman correlation heatmap."""
    corr = pd.read_csv(CORRELATIONS, index_col=0)
    keep = [
        "lst_2025",
        "ndvi_2025",
        "ndbi_2025",
        "albedo_2025",
        "tree_canopy_height_2025",
        "dsm_sym_mean_100m",
        "lcz_probability",
        "pop_total",
        "child_0_14_share",
        "elderly_65_plus_share",
    ]
    corr = corr.loc[keep, keep]
    plt.figure(figsize=(9.5, 7))
    ax = sns.heatmap(
        corr,
        vmin=-1,
        vmax=1,
        cmap="vlag",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "Spearman rho"},
    )
    ax.set_title("Spearman Correlations Among Key Variables")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG_CORRELATION, dpi=300)
    plt.close()


def save_ols_coefficients() -> None:
    """Key standardized OLS coefficients with robust confidence intervals."""
    coefs = pd.read_csv(OLS_COEFS)
    key_terms = [
        "ndvi_2025",
        "ndbi_2025",
        "albedo_2025",
        "tree_canopy_height_2025",
        "dsm_sym_mean_100m",
        "lcz_probability",
        "pop_total",
        "female_share",
        "child_0_14_share",
        "elderly_65_plus_share",
    ]
    plot_df = coefs[coefs["term"].isin(key_terms)].copy()
    order = (
        plot_df.groupby("term")["coef"]
        .apply(lambda values: values.abs().max())
        .sort_values()
        .index
        .tolist()
    )
    plt.figure(figsize=(9, 6))
    ax = sns.pointplot(
        plot_df,
        x="coef",
        y="term",
        hue="model",
        order=order,
        dodge=0.45,
        linestyle="none",
        palette=["#2563EB", "#C2410C"],
    )
    for _, row in plot_df.iterrows():
        y = order.index(row["term"])
        offset = -0.12 if row["model"] == "physical_environmental" else 0.12
        ax.plot([row["ci_low"], row["ci_high"]], [y + offset, y + offset], color="#333333", linewidth=0.8)
    ax.axvline(0, color="#111111", linewidth=0.9)
    ax.set_xlabel("Standardized coefficient (HC3 95% CI)")
    ax.set_ylabel("")
    ax.set_title("Baseline OLS Effects on 2025 LST")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_OLS_COEFS, dpi=300)
    plt.close()


def save_scatter_sample() -> None:
    """Sampled scatterplots for LST against NDVI and NDBI."""
    usecols = ["lst_2025", "ndvi_2025", "ndbi_2025"]
    df = pd.read_csv(MODEL_GRID, usecols=usecols)
    sample = df.sample(min(30_000, len(df)), random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7), sharey=True)
    sns.regplot(sample, x="ndvi_2025", y="lst_2025", scatter_kws={"s": 4, "alpha": 0.08}, line_kws={"color": "#111111"}, ax=axes[0])
    axes[0].set_title("LST vs NDVI")
    axes[0].set_xlabel("NDVI 2025")
    axes[0].set_ylabel("LST 2025 (deg C)")
    sns.regplot(sample, x="ndbi_2025", y="lst_2025", scatter_kws={"s": 4, "alpha": 0.08}, line_kws={"color": "#111111"}, ax=axes[1])
    axes[1].set_title("LST vs NDBI")
    axes[1].set_xlabel("NDBI 2025")
    axes[1].set_ylabel("")
    for ax in axes:
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIG_SCATTER, dpi=300)
    plt.close()


def main() -> None:
    ensure_dirs()
    logger = get_logger("p1_08_figures")
    logger.info("Generating Study 1 core figures.")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.9)
    save_lcz_bar()
    save_correlation_heatmap()
    save_ols_coefficients()
    save_scatter_sample()
    logger.info("Wrote figure: %s", FIG_LCZ_LST)
    logger.info("Wrote figure: %s", FIG_CORRELATION)
    logger.info("Wrote figure: %s", FIG_OLS_COEFS)
    logger.info("Wrote figure: %s", FIG_SCATTER)


if __name__ == "__main__":
    main()
