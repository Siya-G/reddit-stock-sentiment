"""Backtest a simple long/flat strategy driven by model predictions."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"
MODEL_PATH = ROOT / "models" / "lgbm_model.joblib"

FEATURE_COLS = ["mean_sentiment", "post_count", "total_score", "total_comments"]
THRESHOLD = 0.55


def backtest(df: pd.DataFrame, model) -> pd.DataFrame:
    """Go long when predicted up-probability exceeds THRESHOLD, else stay flat."""
    df = df.sort_values("date").copy()
    df["pred_proba"] = model.predict_proba(df[FEATURE_COLS])[:, 1]
    df["position"] = (df["pred_proba"] > THRESHOLD).astype(int)
    df["strategy_return"] = df["position"] * df["fwd_return"]
    df["equity_curve"] = (1 + df["strategy_return"].fillna(0)).cumprod()
    return df


def summarize(df: pd.DataFrame) -> dict:
    returns = df["strategy_return"].dropna()
    sharpe = (
        np.sqrt(252) * returns.mean() / returns.std()
        if returns.std() > 0
        else float("nan")
    )
    return {
        "total_return": float(df["equity_curve"].iloc[-1] - 1) if len(df) else 0.0,
        "sharpe": float(sharpe),
        "trades": int(df["position"].sum()),
    }


def main() -> None:
    df = pd.read_parquet(FEATURES_PATH)
    model = joblib.load(MODEL_PATH)
    result = backtest(df, model)
    print(summarize(result))


if __name__ == "__main__":
    main()
