from __future__ import annotations

import time

import pandas as pd
import requests

from oil_projects.config import PROCESSED_DIR, WEB_NEWS_START_DATE


GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_RATE_LIMIT_SLEEP = 3.0
GDELT_RETRY_BASE_SLEEP = 5.0

COMPANY_TERMS = {
    "XOM": ["exxon", "exxon mobil"],
    "CVX": ["chevron"],
    "COP": ["conocophillips", "conoco phillips"],
    "EOG": ["eog resources"],
    "OXY": ["occidental", "occidental petroleum"],
}

TOPIC_TERMS = {
    "war": ["iran", "israel", "attack", "strike", "missile", "drone", "war", "ceasefire"],
    "shipping": ["strait of hormuz", "hormuz", "shipping", "tanker", "vessel", "port"],
    "supply": ["oil", "crude", "supply", "refinery", "outage", "disruption", "exports"],
}

def _term(term: str) -> str:
    return f'"{term}"' if " " in term else term


def _query_string(ticker: str, topic: str) -> str:
    company_clause = " OR ".join(_term(term) for term in COMPANY_TERMS[ticker])
    topic_clause = " OR ".join(_term(term) for term in TOPIC_TERMS[topic])
    return f"({company_clause}) AND ({topic_clause}) AND sourceLang:english"


def _fetch_timeline(query: str, start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "query": query,
        "mode": "TimelineVol",
        "format": "json",
        "startdatetime": pd.Timestamp(start_date).strftime("%Y%m%d000000"),
        "enddatetime": pd.Timestamp(end_date).strftime("%Y%m%d235959"),
    }
    for attempt in range(2):
        response = requests.get(GDELT_API_URL, params=params, timeout=30)
        if response.status_code == 429:
            print(f"GDELT rate limited, retry {attempt + 1}: {query[:80]}")
            time.sleep(GDELT_RETRY_BASE_SLEEP + attempt * 4)
            continue
        response.raise_for_status()
        try:
            payload = response.json()
        except Exception:
            print(f"GDELT non-JSON response, retry {attempt + 1}: {query[:80]}")
            time.sleep(GDELT_RETRY_BASE_SLEEP + attempt * 4)
            continue
        timeline = payload.get("timeline", [])
        if not timeline:
            return pd.DataFrame(columns=["date", "value"])
        data = timeline[0].get("data", [])
        if not data:
            return pd.DataFrame(columns=["date", "value"])
        frame = pd.DataFrame(data)
        frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
        return frame[["date", "value"]]
    return pd.DataFrame(columns=["date", "value"])


def fetch_gdelt_topic_features(start_date: str = WEB_NEWS_START_DATE, end_date: str | None = None) -> pd.DataFrame:
    end = pd.Timestamp.today().normalize() if end_date is None else pd.Timestamp(end_date).normalize()
    cache_path = PROCESSED_DIR / "gdelt_topic_features.parquet"
    if cache_path.exists():
        frame = pd.read_parquet(cache_path)
        frame["date"] = pd.to_datetime(frame["date"])
        return frame

    frames: list[pd.DataFrame] = []
    for ticker in COMPANY_TERMS:
        ticker_frame = pd.DataFrame({"date": pd.date_range(start=start_date, end=end, freq="D")})
        ticker_frame["ticker"] = ticker
        for topic in TOPIC_TERMS:
            base_query = _query_string(ticker, topic)
            print(f"GDELT timeline {ticker} {topic}")
            base_series = _fetch_timeline(base_query, start_date=start_date, end_date=str(end.date()))
            ticker_frame = ticker_frame.merge(
                base_series.rename(columns={"value": f"gdelt_{topic}_volume"}),
                on="date",
                how="left",
            )
            time.sleep(GDELT_RATE_LIMIT_SLEEP)

        frames.append(ticker_frame)

    panel = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    volume_columns = [column for column in panel.columns if column.startswith("gdelt_")]
    panel[volume_columns] = panel[volume_columns].fillna(0.0)
    panel["gdelt_total_volume"] = panel[[f"gdelt_{topic}_volume" for topic in TOPIC_TERMS]].sum(axis=1)
    for column in [name for name in panel.columns if name.startswith("gdelt_") and name.endswith("_volume")] + ["gdelt_total_volume"]:
        panel[f"{column}_5d"] = panel.groupby("ticker")[column].transform(lambda s: s.rolling(5, min_periods=1).mean())

    panel.to_parquet(cache_path, index=False)
    return panel
