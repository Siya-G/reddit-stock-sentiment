"""Engineer per-(ticker, date) features from scored posts and yfinance prices."""
from __future__ import annotations

import logging
import os
import sqlite3

import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "raw", "reddit_posts.db")
FEATURES_PATH = os.path.join(ROOT, "data", "processed", "features.parquet")
LOG_PATH = os.path.join(ROOT, "logs", "feature_engineering.log")

logger = logging.getLogger("feature_engineering")


def configure_logging() -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def load_posts(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query(
        """
        SELECT ticker, created_utc, sentiment_positive, sentiment_negative,
               sentiment_neutral, num_comments, score
        FROM posts
        WHERE ticker IS NOT NULL
          AND sentiment_positive IS NOT NULL
        """,
        conn,
    )
    df["date"] = pd.to_datetime(df["created_utc"], unit="s", utc=True).dt.tz_convert(None)
    df["date"] = df["date"].dt.normalize()
    return df


def aggregate_daily(posts: pd.DataFrame) -> pd.DataFrame:
    daily = (
        posts.groupby(["ticker", "date"], as_index=False)
        .agg(
            avg_positive=("sentiment_positive", "mean"),
            avg_negative=("sentiment_negative", "mean"),
            avg_neutral=("sentiment_neutral", "mean"),
            post_volume=("sentiment_positive", "count"),
            total_comments=("num_comments", "sum"),
            avg_score=("score", "mean"),
        )
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    logger.info("Aggregated to %d (ticker, date) rows", len(daily))
    return daily


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()
    grouped = df.groupby("ticker", group_keys=False)

    df["sentiment_positive_3d"] = grouped["avg_positive"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    df["sentiment_positive_7d"] = grouped["avg_positive"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    df["sentiment_negative_3d"] = grouped["avg_negative"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    df["sentiment_negative_7d"] = grouped["avg_negative"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    df["sentiment_momentum"] = (
        df["sentiment_positive_3d"] - df["sentiment_positive_7d"]
    )
    df["post_volume_3d"] = grouped["post_volume"].transform(
        lambda s: s.rolling(3, min_periods=1).sum()
    )
    return df


def _extract_close_series(raw: pd.DataFrame) -> pd.Series:
    if raw.empty:
        return pd.Series(dtype=float)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        close = raw["Close"]

    close.index = pd.to_datetime(close.index)
    if close.index.tz is not None:
        close.index = close.index.tz_convert(None)
    close.index = close.index.normalize()
    return close


def fetch_ticker_prices(ticker: str) -> pd.DataFrame:
    try:
        raw = yf.download(ticker, period="max", progress=False, auto_adjust=True)
        close = _extract_close_series(raw)
        if close.empty:
            logger.warning("No price data for %s", ticker)
            return pd.DataFrame()

        prices = close.reset_index()
        prices.columns = ["date", "close"]
        prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
        prices["ticker"] = ticker

        prices = prices.sort_values("date")
        prices["price_direction"] = (
            prices["close"].shift(-1) > prices["close"]
        ).astype("Int64")
        prices["price_momentum"] = (
            prices["close"] - prices["close"].shift(3)
        ) / prices["close"].shift(3)

        return prices[["ticker", "date", "close", "price_direction", "price_momentum"]]
    except Exception as exc:
        logger.warning("Skipping ticker %s: %s", ticker, exc)
        return pd.DataFrame()


def get_valid_tickers() -> set[str]:
    """Download official NASDAQ and NYSE ticker lists and return valid symbols as a set."""
    import urllib.request
    import io

    valid = set()
    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                content = response.read().decode("utf-8")
            df = pd.read_csv(io.StringIO(content), sep="|")
            if "Symbol" in df.columns:
                symbols = df["Symbol"].dropna().astype(str).str.strip()
                symbols = symbols[symbols.str.match(r'^[A-Z]{1,5}$')]
                valid.update(symbols.tolist())
        except Exception as exc:
            logger.warning("Could not fetch ticker list from %s: %s", url, exc)

    logger.info("Loaded %d valid tickers from NASDAQ/NYSE listings", len(valid))
    return valid


def fetch_all_prices(daily: pd.DataFrame, valid_tickers: set[str], min_posts: int = 50) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter tickers by validity and post volume, then fetch prices."""
    # Count total posts per ticker across all dates
    ticker_counts = daily.groupby("ticker")["post_volume"].sum()

    # Filter 1: must be a real ticker
    real = ticker_counts[ticker_counts.index.isin(valid_tickers)]
    logger.info("After NASDAQ/NYSE validation: %d tickers (from %d total)", len(real), len(ticker_counts))

    # Filter 2: must have at least min_posts total posts
    qualified = real[real >= min_posts].index.tolist()
    logger.info("After min %d posts filter: %d tickers", min_posts, len(qualified))

    # Filter daily dataframe to only qualified tickers
    daily_filtered = daily[daily["ticker"].isin(qualified)].copy()

    # Fetch prices
    frames = []
    for ticker in sorted(qualified):
        prices = fetch_ticker_prices(ticker)
        if not prices.empty:
            frames.append(prices)

    if not frames:
        return pd.DataFrame(columns=["ticker", "date", "close", "price_direction", "price_momentum"]), daily_filtered

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Fetched price data for %d / %d qualified tickers", combined["ticker"].nunique(), len(qualified))
    return combined, daily_filtered


def join_and_clean(features: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    merged = features.merge(prices, on=["ticker", "date"], how="left")
    before = len(merged)
    merged = merged.dropna(subset=["price_direction"]).copy()
    merged["price_direction"] = merged["price_direction"].astype(int)
    logger.info(
        "Rows after price join: %d (dropped %d without price data)",
        len(merged),
        before - len(merged),
    )
    return merged


def print_summary(df: pd.DataFrame) -> None:
    up = int((df["price_direction"] == 1).sum())
    down = int((df["price_direction"] == 0).sum())

    print("\n=== Feature engineering summary ===")
    print(f"  Rows:            {len(df):,}")
    print(f"  Unique tickers:  {df['ticker'].nunique():,}")
    print(f"  Date range:      {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Class balance:   UP={up:,}  DOWN={down:,}")


def main() -> None:
    configure_logging()
    os.makedirs(os.path.dirname(FEATURES_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    posts = load_posts(conn)
    conn.close()
    logger.info("Loaded %d scored posts", len(posts))

    daily = aggregate_daily(posts)
    daily = add_rolling_features(daily)

    valid_tickers = get_valid_tickers()
    prices, daily_filtered = fetch_all_prices(daily, valid_tickers, min_posts=50)

    features = join_and_clean(daily_filtered, prices)
    features.to_parquet(FEATURES_PATH, index=False)
    logger.info("Saved features to %s", FEATURES_PATH)

    print_summary(features)


if __name__ == "__main__":
    main()
