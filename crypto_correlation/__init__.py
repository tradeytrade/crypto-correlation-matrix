"""Crypto Correlation Matrix — statistically rigorous correlation analysis for cryptocurrencies."""

__version__ = "1.0.0"

from .fetchers import fetch_prices
from .correlation import (
    compute_returns,
    correlation_matrix,
    correlation_with_pvalues,
    rolling_correlation,
)

__all__ = [
    "fetch_prices",
    "compute_returns",
    "correlation_matrix",
    "correlation_with_pvalues",
    "rolling_correlation",
]
