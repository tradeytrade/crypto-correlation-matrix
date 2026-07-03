"""Unit tests for the correlation engine (no network required)."""

import numpy as np
import pandas as pd
import pytest

from crypto_correlation.correlation import (
    compute_returns,
    correlation_matrix,
    correlation_with_pvalues,
    descriptive_stats,
    fisher_confidence_interval,
    rolling_correlation,
)

rng = np.random.default_rng(42)


def make_prices(n=300):
    """Synthetic prices where A and B are strongly correlated and C is independent."""
    base = rng.normal(0, 0.02, n)
    ret_a = base + rng.normal(0, 0.005, n)
    ret_b = base + rng.normal(0, 0.005, n)          # highly correlated with A
    ret_c = rng.normal(0, 0.02, n)                   # independent
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    prices = pd.DataFrame(
        {
            "A": 100 * np.exp(np.cumsum(ret_a)),
            "B": 200 * np.exp(np.cumsum(ret_b)),
            "C": 50 * np.exp(np.cumsum(ret_c)),
        },
        index=idx,
    )
    return prices


def test_log_returns_shape_and_values():
    prices = make_prices()
    returns = compute_returns(prices, kind="log")
    assert len(returns) == len(prices) - 1
    # log return definition check on first row
    expected = np.log(prices.iloc[1] / prices.iloc[0])
    assert np.allclose(returns.iloc[0], expected)


def test_invalid_return_kind_raises():
    prices = make_prices(50)
    with pytest.raises(ValueError):
        compute_returns(prices, kind="magic")


def test_correlated_pair_detected():
    returns = compute_returns(make_prices())
    corr = correlation_matrix(returns, "pearson")
    assert corr.loc["A", "B"] > 0.9
    assert abs(corr.loc["A", "C"]) < 0.25
    # matrix properties
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr.values, corr.values.T)


def test_pvalues_and_significance():
    returns = compute_returns(make_prices())
    corr, pvals, summary = correlation_with_pvalues(returns, "pearson")
    # strong pair must be significant, independent pair should have larger p
    ab = summary[(summary.asset_a == "A") & (summary.asset_b == "B")].iloc[0]
    assert ab.p_value < 1e-10
    assert bool(ab.significant_bonferroni)
    assert ab.ci_95_low < ab.correlation < ab.ci_95_high
    # p-value matrix symmetric with zero diagonal
    assert np.allclose(pvals.values, pvals.values.T)


def test_matches_scipy_reference():
    from scipy import stats

    returns = compute_returns(make_prices())
    corr, pvals, _ = correlation_with_pvalues(returns, "spearman")
    ref_r, ref_p = stats.spearmanr(returns["A"], returns["C"])
    assert corr.loc["A", "C"] == pytest.approx(ref_r)
    assert pvals.loc["A", "C"] == pytest.approx(ref_p)


def test_fisher_ci_contains_r():
    low, high = fisher_confidence_interval(0.8, 100)
    assert low < 0.8 < high
    assert -1 <= low and high <= 1
    # larger samples give tighter intervals
    low2, high2 = fisher_confidence_interval(0.8, 1000)
    assert (high2 - low2) < (high - low)


def test_rolling_correlation():
    returns = compute_returns(make_prices())
    rolling = rolling_correlation(returns, benchmark="A", window=30)
    assert set(rolling.columns) == {"B", "C"}
    assert rolling["B"].mean() > 0.85
    assert rolling.abs().max().max() <= 1.0 + 1e-9


def test_rolling_bad_benchmark_raises():
    returns = compute_returns(make_prices(60))
    with pytest.raises(ValueError):
        rolling_correlation(returns, benchmark="ZZZ")


def test_descriptive_stats_columns():
    returns = compute_returns(make_prices())
    table = descriptive_stats(returns)
    assert {"ann_return_pct", "ann_volatility_pct", "jarque_bera_p"} <= set(table.columns)
    assert list(table.index) == ["A", "B", "C"]
