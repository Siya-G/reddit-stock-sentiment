"""Train a LightGBM classifier on the engineered features and save the model."""
from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "lgbm_model.joblib"

FEATURE_COLS = ["mean_sentiment", "post_count", "total_score", "total_comments"]
TARGET_COL = "target"


def train(df: pd.DataFrame) -> lgb.LGBMClassifier:
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    print(f"Validation AUC: {auc:.4f}")
    return model


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(FEATURES_PATH)
    model = train(df)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
