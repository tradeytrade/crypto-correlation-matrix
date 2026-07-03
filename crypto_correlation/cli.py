"""Command-line interface.

Example:
    python -m crypto_correlation --symbols BTC ETH SOL XRP DOGE BNB HYPE --days 365
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .correlation import (
    compute_returns,
    correlation_with_pvalues,
    descriptive_stats,
    rolling_correlation,
)
from .fetchers import fetch_prices
from .plotting import plot_heatmap, plot_rolling

DEFAULT_SYMBOLS = ["BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "HYPE"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="crypto-correlation",
        description="Statistically rigorous cryptocurrency correlation matrix from free public APIs.",
    )
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, metavar="SYM",
                   help=f"Ticker symbols to analyze (default: {' '.join(DEFAULT_SYMBOLS)})")
    p.add_argument("--days", type=int, default=365,
                   help="Number of daily observations to fetch (default: 365)")
    p.add_argument("--method", choices=["pearson", "spearman", "kendall"], default="pearson",
                   help="Correlation method (default: pearson)")
    p.add_argument("--rolling-window", type=int, default=30,
                   help="Window size in days for rolling correlation (default: 30)")
    p.add_argument("--benchmark", default="BTC",
                   help="Benchmark asset for rolling correlation (default: BTC)")
    p.add_argument("--output-dir", default="output",
                   help="Directory for CSVs and charts (default: ./output)")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="Significance level for hypothesis tests (default: 0.05)")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = [s.upper() for s in args.symbols]
    print(f"Fetching {args.days} days of daily prices for: {', '.join(symbols)} ...")
    try:
        prices, sources = fetch_prices(symbols, args.days)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Data sources used:")
    for sym, src in sources.items():
        print(f"  {sym:<6} -> {src}")
    print(f"Overlapping observations: {len(prices)} days "
          f"({prices.index.min().date()} to {prices.index.max().date()})\n")

    returns = compute_returns(prices, kind="log")
    corr, pvals, summary = correlation_with_pvalues(returns, args.method, args.alpha)
    stats_table = descriptive_stats(returns)

    print(f"=== {args.method.capitalize()} correlation matrix (daily log returns) ===")
    print(corr.round(3).to_string(), "\n")

    print("=== Pairwise results with significance tests ===")
    display = summary.copy()
    display["p_value"] = display["p_value"].map(lambda v: f"{v:.2e}")
    print(display.to_string(index=False), "\n")

    print("=== Descriptive statistics (annualized) ===")
    print(stats_table.to_string(), "\n")

    # Save artifacts
    prices.to_csv(out_dir / "prices.csv")
    returns.to_csv(out_dir / "log_returns.csv")
    corr.to_csv(out_dir / f"correlation_{args.method}.csv")
    pvals.to_csv(out_dir / f"pvalues_{args.method}.csv")
    summary.to_csv(out_dir / "pairwise_summary.csv", index=False)
    stats_table.to_csv(out_dir / "descriptive_stats.csv")

    heatmap = plot_heatmap(
        corr, pvals, args.method, n_obs=len(returns),
        output=out_dir / "correlation_heatmap.png", alpha=args.alpha,
    )

    benchmark = args.benchmark.upper()
    if benchmark in returns.columns and len(returns.columns) > 1:
        rolling = rolling_correlation(returns, benchmark, args.rolling_window)
        rolling.to_csv(out_dir / f"rolling_correlation_vs_{benchmark}.csv")
        plot_rolling(rolling, benchmark, args.rolling_window,
                     output=out_dir / "rolling_correlation.png")
        print(f"Saved rolling correlation chart -> {out_dir / 'rolling_correlation.png'}")

    print(f"Saved heatmap -> {heatmap}")
    print(f"All CSVs and charts written to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
