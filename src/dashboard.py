"""Streamlit dashboard for exploring sentiment, predictions, and backtest results.

Run with: streamlit run src/dashboard.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from backtest import backtest, summarize

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"
MODEL_PATH = ROOT / "models" / "lgbm_model.joblib"


@st.cache_data
def load_features() -> pd.DataFrame:
    return pd.read_parquet(FEATURES_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def main() -> None:
    st.set_page_config(page_title="Reddit Sentiment Trading", layout="wide")
    st.title("Reddit Sentiment Trading")

    if not FEATURES_PATH.exists() or not MODEL_PATH.exists():
        st.warning("Run the pipeline first to generate features and a trained model.")
        return

    df = load_features()
    model = load_model()
    result = backtest(df, model)

    stats = summarize(result)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total return", f"{stats['total_return']:.1%}")
    col2.metric("Sharpe", f"{stats['sharpe']:.2f}")
    col3.metric("Trades", stats["trades"])

    st.subheader("Equity curve")
    st.line_chart(result.set_index("date")["equity_curve"])

    st.subheader("Data")
    st.dataframe(result)


if __name__ == "__main__":
    main()
