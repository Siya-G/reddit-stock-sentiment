"""Scrape Reddit posts for tickers of interest and save them to data/raw/.

Uses PRAW with credentials loaded from .env. Writes JSON checkpoints to
checkpoints/ so long runs can resume without re-fetching.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import praw
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"

SUBREDDITS = ["wallstreetbets", "stocks", "investing"]
POST_LIMIT = 500

logger = logging.getLogger("scraper")


def configure_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "scraper.log"),
            logging.StreamHandler(),
        ],
    )


def get_client() -> praw.Reddit:
    load_dotenv(ROOT / ".env")
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ["REDDIT_USER_AGENT"],
    )


def scrape_subreddit(reddit: praw.Reddit, subreddit: str, limit: int) -> list[dict]:
    posts = []
    for post in reddit.subreddit(subreddit).new(limit=limit):
        posts.append(
            {
                "id": post.id,
                "subreddit": subreddit,
                "title": post.title,
                "selftext": post.selftext,
                "score": post.score,
                "num_comments": post.num_comments,
                "created_utc": post.created_utc,
            }
        )
    return posts


def main() -> None:
    configure_logging()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    reddit = get_client()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for subreddit in SUBREDDITS:
        logger.info("Scraping r/%s", subreddit)
        posts = scrape_subreddit(reddit, subreddit, POST_LIMIT)

        out_path = RAW_DIR / f"{subreddit}_{stamp}.json"
        out_path.write_text(json.dumps(posts, indent=2))
        logger.info("Wrote %d posts to %s", len(posts), out_path)

        checkpoint = CHECKPOINT_DIR / f"{subreddit}.json"
        checkpoint.write_text(json.dumps({"last_run": stamp, "count": len(posts)}))


if __name__ == "__main__":
    main()
