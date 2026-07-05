# reddit-stock-sentiment

Predict short-term stock moves from Reddit sentiment. The pipeline scrapes posts,
scores them with FinBERT, builds features, trains a LightGBM model, and backtests
a trading strategy. A Streamlit dashboard visualizes the results.

## Project structure

```
.
├── data/
│   ├── raw/          # scraped posts go here
│   ├── processed/    # features.parquet goes here
│   └── prices/       # yfinance data goes here
├── models/           # saved LightGBM model
├── logs/             # scraper.log goes here
├── notebooks/        # EDA notebook
├── src/
│   ├── scraper.py            # pull posts from Reddit (PRAW)
│   ├── finbert_scorer.py     # sentiment scoring with FinBERT
│   ├── feature_engineering.py# build model features
│   ├── train.py              # train LightGBM
│   ├── backtest.py           # backtest the strategy
│   └── dashboard.py          # Streamlit dashboard
├── checkpoints/      # JSON checkpoints from scraper
├── requirements.txt
├── .env              # Reddit API credentials (never committed)
├── .gitignore
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your Reddit API credentials
```

## Pipeline

```bash
python src/scraper.py              # 1. scrape posts -> data/raw/
python src/finbert_scorer.py       # 2. score sentiment
python src/feature_engineering.py  # 3. build data/processed/features.parquet
python src/train.py                # 4. train model -> models/
python src/backtest.py             # 5. backtest the strategy
streamlit run src/dashboard.py     # 6. explore results
```
