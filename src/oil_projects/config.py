import os

from pathlib import Path
import pandas as pd


# Resolve paths from this study folder so the project can be moved or uploaded
# without depending on a local machine-specific directory.
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

START_DATE = "2018-01-01"
NEWS_START_DATE = "2023-01-01"
WEB_NEWS_START_DATE = "2025-01-01"
TRAIN_END = "2023-12-31"
VALID_END = "2024-12-31"
TEST_START = "2025-01-01"
WAR_SPLIT_DATE = "2025-06-13"
IRAN_WAR_WINDOW_DAYS = 60

RANDOM_SEED = 42
MARKET_CLOSE_HOUR = 16

OIL_TICKERS = {
    "XOM": "Exxon Mobil",
    "CVX": "Chevron",
    "COP": "ConocoPhillips",
    "EOG": "EOG Resources",
    "OXY": "Occidental Petroleum",
    "SHEL": "Shell",
    "BP": "BP",
    "TTE": "TotalEnergies",
}

FACTOR_TICKERS = {
    "CL=F": "WTI",
    "BZ=F": "Brent",
    "XLE": "Energy ETF",
    "SPY": "S&P 500 ETF",
    "^VIX": "VIX",
    "UUP": "US Dollar ETF",
}

BENCHMARK_TICKER = "XLE"
EXCESS_RETURN_THRESHOLD = 0.0025
USE_WEB_NEWS_TOOLS = os.environ.get("OIL_PROJECTS_USE_WEB_NEWS", "0") == "1"

IRAN_WAR_END_DATE = (pd.Timestamp(WAR_SPLIT_DATE) + pd.Timedelta(days=IRAN_WAR_WINDOW_DAYS)).strftime("%Y-%m-%d")
