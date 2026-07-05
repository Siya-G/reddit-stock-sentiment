"""Load Reddit post data from Kaggle datasets into a SQLite database."""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, "data", "raw", "reddit_posts.db")
LOG_PATH = os.path.join(ROOT, "logs", "ingestion.log")
CHECKPOINT_PATH = os.path.join(ROOT, "checkpoints", "ingestion_checkpoint.json")

SOURCE1_CSV = (
    "/Users/taru/.cache/kagglehub/datasets/unanimad/reddit-rwallstreetbets/"
    "versions/2/r_wallstreetbets_posts.csv"
)
SOURCE2_DIR = (
    "/Users/taru/.cache/kagglehub/datasets/shergreen/"
    "wallstreetbets-subreddit-submissions/versions/1"
)
SOURCE2_FILES = [
    "wallstreetbets_submission.json",
    "investing_submission.json",
    "options_submission.json",
    "SecurityAnalysis_submission.json",
]

TICKER_RE = re.compile(r"\$([A-Z]{1,5})\b")
CHECKPOINT_INTERVAL = 1000

logger = logging.getLogger("ingestion")


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


def create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS posts (
            id TEXT PRIMARY KEY,
            title TEXT,
            body TEXT,
            score INTEGER,
            num_comments INTEGER,
            created_utc INTEGER,
            subreddit TEXT,
            ticker TEXT
        )
        """
    )
    conn.commit()


def extract_ticker(title: str, body: str) -> str | None:
    match = TICKER_RE.search(f"{title} {body}")
    return match.group(1) if match else None


def write_checkpoint(filename: str, rows_inserted: int) -> None:
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump({"last_file": filename, "rows_inserted": rows_inserted}, f)


def insert_post(conn: sqlite3.Connection, post: dict) -> bool:
    ticker = extract_ticker(post["title"], post["body"])
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR IGNORE INTO posts
            (id, title, body, score, num_comments, created_utc, subreddit, ticker)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            post["id"],
            post["title"],
            post["body"],
            post["score"],
            post["num_comments"],
            post["created_utc"],
            post["subreddit"],
            ticker,
        ),
    )
    return cursor.rowcount > 0


def process_row(
    conn: sqlite3.Connection,
    post: dict,
    filename: str,
    total_inserted: int,
) -> tuple[int, int]:
    """Insert one row. Returns (inserted_delta, skipped_delta)."""
    try:
        if insert_post(conn, post):
            total_inserted += 1
            if total_inserted % CHECKPOINT_INTERVAL == 0:
                conn.commit()
                write_checkpoint(filename, total_inserted)
            return 1, 0
        return 0, 1
    except Exception as exc:
        logger.warning("Skipping bad row in %s: %s", filename, exc)
        return 0, 1


def ingest_csv(conn: sqlite3.Connection, filepath: str, total_inserted: int) -> int:
    filename = os.path.basename(filepath)
    inserted = 0
    skipped = 0

    try:
        df = pd.read_csv(filepath)
        for _, row in df.iterrows():
            try:
                post = {
                    "id": str(row["id"]),
                    "title": str(row["title"]) if pd.notna(row["title"]) else "",
                    "body": "",
                    "score": int(row["score"]) if pd.notna(row["score"]) else 0,
                    "num_comments": int(row["num_comments"])
                    if pd.notna(row["num_comments"])
                    else 0,
                    "created_utc": int(row["created_utc"])
                    if pd.notna(row["created_utc"])
                    else 0,
                    "subreddit": "wallstreetbets",
                }
            except Exception as exc:
                logger.warning("Skipping bad row in %s: %s", filename, exc)
                skipped += 1
                continue

            delta_ins, delta_skip = process_row(conn, post, filename, total_inserted)
            inserted += delta_ins
            skipped += delta_skip
            total_inserted += delta_ins

        conn.commit()
        write_checkpoint(filename, total_inserted)
        logger.info(
            "%s: inserted=%d skipped=%d", filename, inserted, skipped
        )
    except Exception as exc:
        logger.error("Failed to process %s: %s", filename, exc)

    return total_inserted


def ingest_jsonl(conn: sqlite3.Connection, filepath: str, total_inserted: int) -> int:
    filename = os.path.basename(filepath)
    inserted = 0
    skipped = 0

    try:
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    post = {
                        "id": str(record["id"]),
                        "title": str(record.get("title") or ""),
                        "body": str(record.get("selftext") or ""),
                        "score": int(record.get("score") or 0),
                        "num_comments": int(record.get("num_comments") or 0),
                        "created_utc": int(record.get("created_utc") or 0),
                        "subreddit": str(record.get("subreddit") or ""),
                    }
                except Exception as exc:
                    logger.warning("Skipping bad row in %s: %s", filename, exc)
                    skipped += 1
                    continue

                delta_ins, delta_skip = process_row(
                    conn, post, filename, total_inserted
                )
                inserted += delta_ins
                skipped += delta_skip
                total_inserted += delta_ins

        conn.commit()
        write_checkpoint(filename, total_inserted)
        logger.info(
            "%s: inserted=%d skipped=%d", filename, inserted, skipped
        )
    except Exception as exc:
        logger.error("Failed to process %s: %s", filename, exc)

    return total_inserted


def print_summary(conn: sqlite3.Connection) -> None:
    cursor = conn.execute(
        """
        SELECT subreddit, COUNT(*) AS count
        FROM posts
        GROUP BY subreddit
        ORDER BY subreddit
        """
    )
    rows = cursor.fetchall()

    print("\n=== Ingestion summary (rows per subreddit) ===")
    total = 0
    for subreddit, count in rows:
        print(f"  {subreddit}: {count:,}")
        total += count
    print(f"  TOTAL: {total:,}")


def main() -> None:
    configure_logging()
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    create_schema(conn)

    total_inserted = 0
    total_inserted = ingest_csv(conn, SOURCE1_CSV, total_inserted)

    for json_file in SOURCE2_FILES:
        filepath = os.path.join(SOURCE2_DIR, json_file)
        total_inserted = ingest_jsonl(conn, filepath, total_inserted)

    print_summary(conn)
    conn.close()
    logger.info("Done. Database saved to %s", DB_PATH)


if __name__ == "__main__":
    main()
