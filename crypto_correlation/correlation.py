"""Correlation analysis with proper statistical testing.

Key design decisions (why this tool is "correct", not just a pretty heatmap):

1. Correlations are computed on LOG RETURNS, never on raw prices. Prices are
   non-stationary; correlating them produces spurious, inflated correlations.
   Log returns are (approximately) stationary and additive over time.

2. Every pairwise coefficient comes with a p-value (test of H0: rho = 0) and a
   95% confidence interval via the Fisher z-transformation, so you can tell a
   real relationship from noise.

3. Pearson, Spearman, and Kendall are all supported. Crypto returns are heavy-
   tailed, so rank-based measures (Spearman/Kendall) are often more robust than
   Pearson, which is sensitive to outliers.

4. Rolling correlations show how co-movement changes over time — a single
   full-sample number hides regime shifts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

VALID_METHODS = ("pearson", "spearman", "kendall")


def compute_returns(prices: pd.DataFrame, kind: str = "log") -> pd.DataFrame:
    """Convert a price DataFrame into daily returns.

    Args:
        prices: DataFrame of daily close prices (one column per asset).
        kind: "log" (recommended) or "simple".
    """
    if kind == "log":
        returns = np.log(prices / prices.shift(1))
    elif kind == "simple":
        returns = prices.pct_change()
    else:
        raise ValueError(f"kind must be 'log' or 'simple', got {kind!r}")
    return returns.dropna(how="any")


def correlation_matrix(returns: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Full correlation matrix of returns using the chosen method."""
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}, got {method!r}")
    return returns.corr(method=method)


def _pairwise_test(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float, float]:
    """Return (coefficient, p-value) for one asset pair."""
    if method == "pearson":
        r, p = stats.pearsonr(x, y)
    elif method == "spearman":
        r, p = stats.spearmanr(x, y)
    else:
        r, p = stats.kendalltau(x, y)
    return float(r), float(p)


def fisher_confidence_interval(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95% CI for a correlation coefficient via the Fisher z-transformation."""
    r = float(np.clip(r, -0.999999, 0.999999))
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - z_crit * se)), float(np.tanh(z + z_crit * se))


def correlation_with_pvalues(
    returns: pd.DataFrame, method: str = "pearson", alpha: float = 0.05
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Correlation matrix plus p-values and a tidy pairwise summary table.

    Returns:
        corr: correlation matrix.
        pvals: matrix of p-values for H0: rho = 0.
        summary: long-format DataFrame with one row per unique pair, including
                 the coefficient, p-value, 95% CI, sample size, and a
                 significance flag (Bonferroni-adjusted for multiple testing).
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}, got {method!r}")

    cols = list(returns.columns)
    n_assets = len(cols)
    n_obs = len(returns)
    n_pairs = n_assets * (n_assets - 1) // 2
    bonferroni_alpha = alpha / max(n_pairs, 1)

    corr = pd.DataFrame(np.eye(n_assets), index=cols, columns=cols)
    pvals = pd.DataFrame(np.zeros((n_assets, n_assets)), index=cols, columns=cols)
    rows = []

    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            x = returns[cols[i]].to_numpy()
            y = returns[cols[j]].to_numpy()
            r, p = _pairwise_test(x, y, method)
            ci_low, ci_high = fisher_confidence_interval(r, n_obs, alpha)
            corr.iat[i, j] = corr.iat[j, i] = r
            pvals.iat[i, j] = pvals.iat[j, i] = p
            rows.append(
                {
                    "asset_a": cols[i],
                    "asset_b": cols[j],
                    "method": method,
                    "correlation": round(r, 4),
                    "p_value": p,
                    "ci_95_low": round(ci_low, 4),
                    "ci_95_high": round(ci_high, 4),
                    "n_observations": n_obs,
                    "significant_bonferroni": p < bonferroni_alpha,
                }
            )

    summary = pd.DataFrame(rows).sort_values("correlation", ascending=False).reset_index(drop=True)
    return corr, pvals, summary


def rolling_correlation(
    returns: pd.DataFrame, benchmark: str, window: int = 30
) -> pd.DataFrame:
    """Rolling Pearson correlation of every asset against a benchmark (e.g. BTC)."""
    if benchmark not in returns.columns:
        raise ValueError(f"benchmark {benchmark!r} not in returns columns")
    others = [c for c in returns.columns if c != benchmark]
    out = {}
    for col in others:
        out[col] = returns[col].rolling(window).corr(returns[benchmark])
    return pd.DataFrame(out).dropna(how="all")


def descriptive_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualized return/volatility, skewness, kurtosis, and normality test per asset.

    The Jarque-Bera p-value tells you whether returns look Gaussian (they
    almost never do in crypto) — a reason to also check Spearman/Kendall.
    """
    rows = []
    for col in returns.columns:
        r = returns[col].dropna()
        jb_stat, jb_p = stats.jarque_bera(r)
        rows.append(
            {
                "asset": col,
                "ann_return_pct": round(float(r.mean()) * 365 * 100, 2),
                "ann_volatility_pct": round(float(r.std()) * np.sqrt(365) * 100, 2),
                "skewness": round(float(stats.skew(r)), 3),
                "excess_kurtosis": round(float(stats.kurtosis(r)), 3),
                "jarque_bera_p": float(jb_p),
                "normal_returns": jb_p > 0.05,
            }
        )
    return pd.DataFrame(rows).set_index("asset")
