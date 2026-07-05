"""Score the sentiment of scraped Reddit text using FinBERT.

Loads the ProsusAI/finbert model and returns probabilities for
positive / negative / neutral, plus a signed sentiment score.
"""
from __future__ import annotations

from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "ProsusAI/finbert"
LABELS = ["positive", "negative", "neutral"]


@lru_cache(maxsize=1)
def _load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def score_texts(texts: list[str], batch_size: int = 16) -> list[dict]:
    """Return a list of {positive, negative, neutral, sentiment} dicts."""
    tokenizer, model = _load_model()
    results: list[dict] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

        for row in probs:
            scores = {label: float(row[i]) for i, label in enumerate(LABELS)}
            scores["sentiment"] = scores["positive"] - scores["negative"]
            results.append(scores)

    return results


if __name__ == "__main__":
    sample = ["TSLA is going to the moon!", "I think this stock will crash hard."]
    for text, score in zip(sample, score_texts(sample)):
        print(f"{text!r} -> {score}")
