"""Build the modeling dataset and write it to data/processed/features.parquet.

Combines scored Reddit sentiment with yfinance price data into a per-ticker,
per-day feature table suitable for training.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PRICES_DIR = ROOT / "data" / "prices"
PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"


def build_features(sentiment: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Join daily sentiment aggregates with forward price returns.

    Expects `sentiment` with columns [date, ticker, sentiment, score, num_comments]
    and `prices` with columns [date, ticker, close].
    """
    daily = (
        sentiment.groupby(["date", "ticker"])
        .agg(
            mean_sentiment=("sentiment", "mean"),
            post_count=("sentiment", "size"),
            total_score=("score", "sum"),
            total_comments=("num_comments", "sum"),
        )
        .reset_index()
    )

    prices = prices.sort_values(["ticker", "date"]).copy()
    prices["fwd_return"] = prices.groupby("ticker")["close"].pct_change().shift(-1)
    prices["target"] = (prices["fwd_return"] > 0).astype(int)

    return daily.merge(prices, on=["date", "ticker"], how="inner")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    # TODO: load scored sentiment from data/raw and prices from data/prices,
    #       then call build_features(...) and write the result.
    print(f"Write features to {FEATURES_PATH}")


if __name__ == "__main__":
    main()
