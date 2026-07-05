"""Score Reddit posts in the SQLite database using ProsusAI/finbert."""
from __future__ import annotations

import json
import logging
import os
import sqlite3

import torch
from tqdm import tqdm
from transformers import pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "raw", "reddit_posts.db")
LOG_PATH = os.path.join(ROOT, "logs", "finbert.log")
CHECKPOINT_PATH = os.path.join(ROOT, "checkpoints", "finbert_checkpoint.json")

MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 32
COMMIT_INTERVAL = 1000
CHECKPOINT_INTERVAL = 1000
PROGRESS_LOG_INTERVAL = 10_000
MAX_TEXT_LEN = 512

logger = logging.getLogger("finbert")


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


def ensure_sentiment_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute("PRAGMA table_info(posts)")}
    for column in ("sentiment_positive", "sentiment_negative", "sentiment_neutral"):
        if column not in existing:
            conn.execute(f"ALTER TABLE posts ADD COLUMN {column} REAL")
    conn.commit()


def build_text(title: str | None, body: str | None) -> str:
    text = f"{title or ''} {body or ''}".strip()
    return text[:MAX_TEXT_LEN] if text else ""


def parse_scores(output: list) -> list[tuple[float, float, float]]:
    """Parse FinBERT batch output into (positive, negative, neutral) per post."""
    results: list[tuple[float, float, float]] = []
    for item in output:
        scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        label_scores = item if isinstance(item, list) else [item]
        for label_score in label_scores:
            label = label_score["label"].lower()
            if label in scores:
                scores[label] = float(label_score["score"])
        results.append((scores["positive"], scores["negative"], scores["neutral"]))
    return results


def write_checkpoint(last_id: str, scored: int) -> None:
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump({"last_id": last_id, "scored": scored}, f)


def load_classifier(device: str):
    pipe_device = 0 if device == "cuda" else -1
    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        return_all_scores=True,
        top_k=3,
        device=pipe_device,
    )


def fetch_posts_to_score(conn: sqlite3.Connection) -> list[tuple[str, str, str]]:
    cursor = conn.execute(
        """
        SELECT id, title, body
        FROM posts
        WHERE ticker IS NOT NULL
          AND sentiment_positive IS NULL
        ORDER BY id
        """
    )
    return cursor.fetchall()


def count_skipped(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM posts WHERE ticker IS NULL"
    ).fetchone()
    return int(row[0])


def update_post_scores(
    conn: sqlite3.Connection,
    post_id: str,
    positive: float,
    negative: float,
    neutral: float,
) -> None:
    conn.execute(
        """
        UPDATE posts
        SET sentiment_positive = ?,
            sentiment_negative = ?,
            sentiment_neutral = ?
        WHERE id = ?
        """,
        (positive, negative, neutral, post_id),
    )


def score_posts(conn: sqlite3.Connection, classifier) -> tuple[int, dict[str, float]]:
    posts = fetch_posts_to_score(conn)
    total_to_score = len(posts)

    if total_to_score == 0:
        logger.info("No posts to score.")
        return 0, {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    logger.info("Posts to score: %d", total_to_score)

    scored = 0
    since_commit = 0
    last_id = ""
    sum_positive = 0.0
    sum_negative = 0.0
    sum_neutral = 0.0

    batch_ids: list[str] = []
    batch_texts: list[str] = []

    def flush_batch() -> None:
        nonlocal scored, since_commit, sum_positive, sum_negative, sum_neutral

        if not batch_ids:
            return

        outputs = classifier(batch_texts)
        score_tuples = parse_scores(outputs)

        for post_id, (positive, negative, neutral) in zip(batch_ids, score_tuples):
            update_post_scores(conn, post_id, positive, negative, neutral)

            scored += 1
            since_commit += 1
            sum_positive += positive
            sum_negative += negative
            sum_neutral += neutral

            if scored % PROGRESS_LOG_INTERVAL == 0:
                logger.info("Progress: %d/%d posts scored", scored, total_to_score)

            if since_commit >= COMMIT_INTERVAL:
                conn.commit()
                write_checkpoint(post_id, scored)
                since_commit = 0

        batch_ids.clear()
        batch_texts.clear()

    for post_id, title, body in tqdm(posts, desc="Scoring posts", unit="post"):
        last_id = post_id
        batch_ids.append(post_id)
        batch_texts.append(build_text(title, body))

        if len(batch_ids) >= BATCH_SIZE:
            flush_batch()

    flush_batch()
    conn.commit()

    if scored > 0:
        write_checkpoint(last_id, scored)

    averages = {
        "positive": sum_positive / scored,
        "negative": sum_negative / scored,
        "neutral": sum_neutral / scored,
    }
    return scored, averages


def print_summary(
    scored: int,
    skipped: int,
    averages: dict[str, float],
) -> None:
    print("\n=== FinBERT scoring summary ===")
    print(f"  Posts scored: {scored:,}")
    print(f"  Posts skipped (no ticker): {skipped:,}")
    if scored > 0:
        print(f"  Avg positive:  {averages['positive']:.4f}")
        print(f"  Avg negative:  {averages['negative']:.4f}")
        print(f"  Avg neutral:   {averages['neutral']:.4f}")


def main() -> None:
    configure_logging()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    ensure_sentiment_columns(conn)

    skipped = count_skipped(conn)
    logger.info("Posts skipped (no ticker): %d", skipped)

    classifier = load_classifier(device)
    scored, averages = score_posts(conn, classifier)

    logger.info("Finished scoring %d posts", scored)
    print_summary(scored, skipped, averages)

    conn.close()
    logger.info("Updated database at %s", DB_PATH)


if __name__ == "__main__":
    main()
