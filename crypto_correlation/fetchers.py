"""Price data fetchers using free, no-key-required public APIs.

Sources (in priority order):
    1. Binance public REST API   — https://api.binance.com (daily klines, no key)
    2. Coinbase Exchange API     — https://api.exchange.coinbase.com (daily candles, no key)
    3. Kraken public API         — https://api.kraken.com (daily OHLC, no key)
    4. CoinGecko free API        — https://api.coingecko.com (market chart, no key, rate-limited)
    5. CryptoCompare free API    — https://min-api.cryptocompare.com (histoday, key optional)

Exchange APIs are tried first because they have generous rate limits; CoinGecko
covers long-tail assets not listed on the big exchanges (e.g. new tokens).

Each fetcher returns a pandas Series of daily closing prices (UTC) indexed by date.
`fetch_prices` tries each source in order and merges all assets into one DataFrame.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import pandas as pd
import requests

logger = logging.getLogger(__name__)

USER_AGENT = "crypto-correlation-matrix/1.0 (https://github.com)"
REQUEST_TIMEOUT = 15

# Symbol -> CoinGecko coin id mapping for common assets.
COINGECKO_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "XRP": "ripple",
    "DOGE": "dogecoin",
    "BNB": "binancecoin",
    "HYPE": "hyperliquid",
    "ADA": "cardano",
    "AVAX": "avalanche-2",
    "DOT": "polkadot",
    "LINK": "chainlink",
    "TRX": "tron",
    "LTC": "litecoin",
    "MATIC": "matic-network",
    "SHIB": "shiba-inu",
    "UNI": "uniswap",
    "ATOM": "cosmos",
    "XLM": "stellar",
    "NEAR": "near",
    "APT": "aptos",
    "ARB": "arbitrum",
    "OP": "optimism",
    "SUI": "sui",
    "TON": "the-open-network",
    "PEPE": "pepe",
}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return s


# ---------------------------------------------------------------------------
# Binance
# ---------------------------------------------------------------------------

def fetch_binance(symbol: str, days: int, session: requests.Session | None = None) -> pd.Series:
    """Fetch daily close prices for SYMBOL/USDT from Binance public klines."""
    session = session or _session()
    pair = f"{symbol.upper()}USDT"
    url = "https://api.binance.com/api/v3/klines"

    closes: list[tuple[datetime, float]] = []
    end_time = None
    remaining = days + 1  # +1 so we have enough closes for `days` returns

    while remaining > 0:
        params = {"symbol": pair, "interval": "1d", "limit": min(1000, remaining)}
        if end_time is not None:
            params["endTime"] = end_time
        resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        klines = resp.json()
        if not klines:
            break
        for k in klines:
            open_ms, close_price = int(k[0]), float(k[4])
            dt = datetime.fromtimestamp(open_ms / 1000, tz=timezone.utc)
            closes.append((dt, close_price))
        remaining -= len(klines)
        end_time = int(klines[0][0]) - 1
        if len(klines) < params["limit"]:
            break

    if not closes:
        raise ValueError(f"Binance returned no data for {pair}")

    series = pd.Series(
        {dt.date(): price for dt, price in closes}, name=symbol.upper()
    ).sort_index()
    series.index = pd.to_datetime(series.index)
    return series.tail(days + 1)


# ---------------------------------------------------------------------------
# Coinbase Exchange
# ---------------------------------------------------------------------------

def fetch_coinbase(symbol: str, days: int, session: requests.Session | None = None) -> pd.Series:
    """Fetch daily close prices for SYMBOL-USD from Coinbase Exchange candles."""
    session = session or _session()
    product = f"{symbol.upper()}-USD"
    url = f"https://api.exchange.coinbase.com/products/{product}/candles"

    records: dict = {}
    now = datetime.now(timezone.utc)
    remaining = days + 1
    end = now
    while remaining > 0:
        chunk = min(300, remaining)  # Coinbase caps 300 candles per request
        start = end - pd.Timedelta(days=chunk)
        params = {
            "granularity": 86400,
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
        resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        candles = resp.json()  # [time, low, high, open, close, volume]
        if not candles:
            break
        for c in candles:
            dt = datetime.fromtimestamp(int(c[0]), tz=timezone.utc).date()
            records[dt] = float(c[4])
        remaining -= chunk
        end = start
        time.sleep(0.15)

    if not records:
        raise ValueError(f"Coinbase returned no data for {product}")
    series = pd.Series(records, name=symbol.upper()).sort_index()
    series.index = pd.to_datetime(series.index)
    return series.tail(days + 1)


# ---------------------------------------------------------------------------
# Kraken
# ---------------------------------------------------------------------------

def fetch_kraken(symbol: str, days: int, session: requests.Session | None = None) -> pd.Series:
    """Fetch daily close prices for SYMBOL/USD from Kraken public OHLC."""
    session = session or _session()
    if days + 1 > 720:
        raise ValueError("Kraken OHLC returns at most 720 daily candles")
    pair = f"{symbol.upper()}USD"
    resp = session.get(
        "https://api.kraken.com/0/public/OHLC",
        params={"pair": pair, "interval": 1440},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("error"):
        raise ValueError(f"Kraken error for {pair}: {payload['error']}")
    result = payload.get("result", {})
    data_key = next((k for k in result if k != "last"), None)
    if data_key is None:
        raise ValueError(f"Kraken returned no data for {pair}")

    records = {}
    for row in result[data_key]:  # [time, open, high, low, close, ...]
        dt = datetime.fromtimestamp(int(row[0]), tz=timezone.utc).date()
        records[dt] = float(row[4])
    series = pd.Series(records, name=symbol.upper()).sort_index()
    series.index = pd.to_datetime(series.index)
    return series.tail(days + 1)


# ---------------------------------------------------------------------------
# CoinGecko
# ---------------------------------------------------------------------------

def _get_with_backoff(
    session: requests.Session, url: str, params: dict, retries: int = 4, base_wait: float = 15.0
) -> requests.Response:
    """GET with exponential backoff on HTTP 429 (CoinGecko free-tier rate limit)."""
    for attempt in range(retries):
        resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 429:
            return resp
        wait = base_wait * (attempt + 1)
        logger.warning("Rate limited by %s — waiting %.0fs (attempt %d/%d)",
                       url.split("/")[2], wait, attempt + 1, retries)
        time.sleep(wait)
    return resp


def fetch_coingecko(symbol: str, days: int, session: requests.Session | None = None) -> pd.Series:
    """Fetch daily close prices in USD from the CoinGecko free API."""
    session = session or _session()
    coin_id = COINGECKO_IDS.get(symbol.upper())
    if coin_id is None:
        coin_id = _coingecko_search(symbol, session)

    # Free tier caps historical range at 365 days.
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": min(days + 1, 365), "interval": "daily"}
    resp = _get_with_backoff(session, url, params)
    resp.raise_for_status()
    prices = resp.json().get("prices", [])
    if not prices:
        raise ValueError(f"CoinGecko returned no data for {coin_id}")

    records = {}
    for ms, price in prices:
        dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date()
        records[dt] = float(price)  # keep last observation per day
    series = pd.Series(records, name=symbol.upper()).sort_index()
    series.index = pd.to_datetime(series.index)
    return series.tail(days + 1)


def _coingecko_search(symbol: str, session: requests.Session) -> str:
    """Resolve an unknown ticker to a CoinGecko coin id via the search endpoint."""
    resp = session.get(
        "https://api.coingecko.com/api/v3/search",
        params={"query": symbol},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    coins = resp.json().get("coins", [])
    for coin in coins:
        if coin.get("symbol", "").upper() == symbol.upper():
            return coin["id"]
    if coins:
        return coins[0]["id"]
    raise ValueError(f"Could not resolve symbol '{symbol}' on CoinGecko")


# ---------------------------------------------------------------------------
# CryptoCompare
# ---------------------------------------------------------------------------

def fetch_cryptocompare(symbol: str, days: int, session: requests.Session | None = None) -> pd.Series:
    """Fetch daily close prices in USD from the CryptoCompare free API."""
    session = session or _session()
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = {"fsym": symbol.upper(), "tsym": "USD", "limit": min(days + 1, 2000)}
    resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("Response") != "Success":
        raise ValueError(f"CryptoCompare error for {symbol}: {payload.get('Message')}")

    records = {}
    for row in payload["Data"]["Data"]:
        if row["close"] > 0:
            dt = datetime.fromtimestamp(row["time"], tz=timezone.utc).date()
            records[dt] = float(row["close"])
    if not records:
        raise ValueError(f"CryptoCompare returned no usable data for {symbol}")
    series = pd.Series(records, name=symbol.upper()).sort_index()
    series.index = pd.to_datetime(series.index)
    return series.tail(days + 1)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

FETCHERS = [
    ("Binance", fetch_binance),
    ("Coinbase", fetch_coinbase),
    ("Kraken", fetch_kraken),
    ("CoinGecko", fetch_coingecko),
    ("CryptoCompare", fetch_cryptocompare),
]


def fetch_asset(symbol: str, days: int, session: requests.Session | None = None) -> tuple[pd.Series, str]:
    """Fetch one asset, trying each source in order. Returns (series, source_name)."""
    session = session or _session()
    errors = []
    for name, fetcher in FETCHERS:
        try:
            series = fetcher(symbol, days, session)
            if len(series) >= 30:  # need a minimum sample for meaningful stats
                logger.info("Fetched %s from %s (%d observations)", symbol, name, len(series))
                return series, name
            errors.append(f"{name}: only {len(series)} observations")
        except Exception as exc:  # noqa: BLE001 — fall through to the next source
            errors.append(f"{name}: {exc}")
    raise RuntimeError(f"All sources failed for {symbol}: " + " | ".join(errors))


def fetch_prices(symbols: list[str], days: int = 365) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch daily USD close prices for all symbols and align them on shared dates.

    Returns:
        prices: DataFrame indexed by date, one column per symbol.
        sources: mapping symbol -> API used.
    """
    session = _session()
    series_list, sources = [], {}
    for symbol in symbols:
        symbol = symbol.upper().strip()
        series, source = fetch_asset(symbol, days, session)
        series_list.append(series)
        sources[symbol] = source
        time.sleep(0.35)  # be polite to free APIs

    prices = pd.concat(series_list, axis=1)
    # Keep only dates where every asset has a price (fair pairwise comparison).
    prices = prices.dropna(how="any").sort_index()
    if len(prices) < 30:
        raise RuntimeError(
            f"Only {len(prices)} overlapping daily observations across all assets; "
            "reduce the asset list or check that all symbols are valid."
        )
    return prices, sources
