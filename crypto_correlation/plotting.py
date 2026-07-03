"""Visualization: annotated heatmap with significance markers and rolling plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_heatmap(
    corr: pd.DataFrame,
    pvals: pd.DataFrame | None = None,
    method: str = "pearson",
    n_obs: int | None = None,
    output: str | Path = "correlation_heatmap.png",
    alpha: float = 0.05,
) -> Path:
    """Save an annotated correlation heatmap.

    Coefficients whose p-value is NOT below `alpha` are marked with '(ns)'
    (not significant) so weak, noisy relationships are visually flagged.
    """
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(8, n * 1.1), max(6.5, n * 0.95)))

    annot = corr.round(2).astype(str)
    if pvals is not None:
        mask_ns = (pvals >= alpha) & ~np.eye(n, dtype=bool)
        annot = annot.where(~mask_ns, annot + "\n(ns)")

    sns.heatmap(
        corr,
        annot=annot,
        fmt="",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": f"{method.capitalize()} correlation"},
        annot_kws={"fontsize": 9},
        ax=ax,
    )
    title = f"Cryptocurrency Correlation Matrix ({method.capitalize()}, daily log returns)"
    if n_obs:
        title += f"\n{n_obs} overlapping daily observations · '(ns)' = not significant at α={alpha}"
    ax.set_title(title, fontsize=12, pad=14)
    plt.tight_layout()
    output = Path(output)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def plot_rolling(
    rolling: pd.DataFrame,
    benchmark: str,
    window: int,
    output: str | Path = "rolling_correlation.png",
) -> Path:
    """Save a line chart of rolling correlations vs. the benchmark asset."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in rolling.columns:
        ax.plot(rolling.index, rolling[col], label=col, linewidth=1.4)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel(f"{window}-day rolling correlation vs {benchmark}")
    ax.set_title(f"Rolling Correlation vs {benchmark} ({window}-day window, daily log returns)")
    ax.legend(loc="lower left", ncols=min(4, len(rolling.columns)), fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    output = Path(output)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
