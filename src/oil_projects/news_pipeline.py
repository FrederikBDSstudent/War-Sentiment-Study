from __future__ import annotations

import json
import time

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
try:
    from transformers import pipeline as hf_pipeline
except Exception:  # pragma: no cover - fallback when transformers is unavailable
    hf_pipeline = None
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf

from oil_projects.common import build_excess_return_target
from oil_projects.config import (
    BENCHMARK_TICKER,
    MARKET_CLOSE_HOUR,
    NEWS_START_DATE,
    PROCESSED_DIR,
    START_DATE,
    USE_WEB_NEWS_TOOLS,
)
from oil_projects.theory_agents import add_theory_agent_features
from oil_projects.official_sources import build_official_event_features
from oil_projects.web_news_tools import fetch_gdelt_topic_features


SEC_HEADERS = {"User-Agent": "Codex oil-project research frederik@example.com"}
SEC_TICKERS = {
    "XOM": {"company_name": "Exxon Mobil", "cik": "0000034088"},
    "CVX": {"company_name": "Chevron", "cik": "0000093410"},
    "COP": {"company_name": "ConocoPhillips", "cik": "0001163165"},
    "EOG": {"company_name": "EOG Resources", "cik": "0000821189"},
    "OXY": {"company_name": "Occidental Petroleum", "cik": "0000797468"},
}
TARGET_FORMS = {"8-K", "10-Q", "10-K"}
GPR_DAILY_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
EVENT_PATTERNS = {
    "guidance_flag": r"guidance|outlook|forecast|expects|expected",
    "earnings_flag": r"earnings|results|income|revenue|profit|loss",
    "acquisition_flag": r"acquisition|acquire|merger|purchase|divest|asset sale",
    "production_flag": r"production|output|barrels|volumes|reserve|reserves",
    "legal_flag": r"litigation|lawsuit|settlement|investigation|penalty|fine",
    "geopolitics_flag": r"iran|israel|sanction|opec|middle east|conflict|war",
    "supply_disruption_flag": r"outage|shutdown|disruption|interrupt|halted|offline|force majeure",
    "shipping_flag": r"shipping|tanker|strait of hormuz|vessel|freight|port",
    "refining_flag": r"refinery|refining|throughput|crack spread|downstream",
    "upstream_flag": r"drilling|rig|well|upstream|exploration|production sharing",
    "sanctions_flag": r"sanction|embargo|restriction|export ban|price cap",
}


def _load_macro_political_features() -> pd.DataFrame:
    cache_path = PROCESSED_DIR / "macro_political_features.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    gpr = pd.read_excel(GPR_DAILY_URL)
    gpr["date"] = pd.to_datetime(gpr["date"]).dt.tz_localize(None)
    gpr = gpr.rename(
        columns={
            "GPRD": "gpr_level",
            "GPRD_ACT": "gpr_act",
            "GPRD_THREAT": "gpr_threat",
            "GPRD_MA7": "gpr_ma7",
            "GPRD_MA30": "gpr_ma30",
        }
    )[["date", "gpr_level", "gpr_act", "gpr_threat", "gpr_ma7", "gpr_ma30"]]
    gpr["gpr_surprise"] = gpr["gpr_level"] - gpr["gpr_ma30"]
    gpr["gpr_threat_share"] = gpr["gpr_threat"] / gpr["gpr_level"].replace(0, np.nan)
    gpr["gpr_act_share"] = gpr["gpr_act"] / gpr["gpr_level"].replace(0, np.nan)

    market_frames = []
    for ticker, prefix in [("^OVX", "ovx"), ("USO", "uso"), ("XOP", "xop")]:
        frame = yf.download(ticker, start=START_DATE, progress=False, auto_adjust=True)
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame.columns = [str(column).lower() for column in frame.columns]
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        out = pd.DataFrame({"date": frame.index})
        out[f"{prefix}_close"] = frame["close"].values
        out[f"{prefix}_ret_1"] = frame["close"].pct_change(fill_method=None).values
        out[f"{prefix}_ret_5"] = frame["close"].pct_change(5, fill_method=None).values
        market_frames.append(out.set_index("date"))

    macro = gpr.set_index("date")
    for frame in market_frames:
        macro = macro.join(frame, how="outer")
    macro = macro.sort_index().ffill().reset_index()
    macro.to_parquet(cache_path, index=False)
    return macro


def _download_target_prices() -> dict[str, pd.DataFrame]:
    data = {}
    benchmark_frame = yf.download(BENCHMARK_TICKER, start=START_DATE, progress=False, auto_adjust=True)
    if isinstance(benchmark_frame.columns, pd.MultiIndex):
        benchmark_frame.columns = benchmark_frame.columns.get_level_values(0)
    benchmark_frame.columns = [str(column).lower() for column in benchmark_frame.columns]
    benchmark_frame.index = pd.to_datetime(benchmark_frame.index).tz_localize(None)
    benchmark_close = benchmark_frame["close"]

    for ticker in SEC_TICKERS:
        frame = yf.download(ticker, start=START_DATE, progress=False, auto_adjust=True)
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame.columns = [str(column).lower() for column in frame.columns]
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        target_frame = build_excess_return_target(frame["close"], benchmark_close)
        frame["target"] = target_frame["target"]
        frame["excess_ret_fwd"] = target_frame["excess_ret_fwd"]
        frame["prev_ret_1"] = frame["close"].pct_change(fill_method=None)
        frame["prev_ret_5"] = frame["close"].pct_change(5, fill_method=None)
        data[ticker] = frame.sort_index()
    return data


def _fetch_submissions(cik: str) -> pd.DataFrame:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    payload = requests.get(url, headers=SEC_HEADERS, timeout=30).json()
    recent = pd.DataFrame(payload["filings"]["recent"])
    return recent


def _fetch_filing_text(cik: str, accession_number: str, primary_document: str) -> str:
    accession_compact = accession_number.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_compact}/{primary_document}"
    response = requests.get(url, headers=SEC_HEADERS, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    return " ".join(soup.stripped_strings)


def _fetch_filings() -> pd.DataFrame:
    cache_path = PROCESSED_DIR / "sec_filings.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    rows = []
    for ticker, meta in SEC_TICKERS.items():
        print(f"Fetching SEC filings for {ticker}")
        submissions = _fetch_submissions(meta["cik"])
        submissions = submissions[submissions["form"].isin(TARGET_FORMS)].copy()
        submissions["filingDate"] = pd.to_datetime(submissions["filingDate"])
        submissions = submissions[submissions["filingDate"] >= pd.Timestamp(NEWS_START_DATE)]
        submissions = submissions.sort_values("acceptanceDateTime")
        print(f"  {len(submissions)} filings to scrape")

        for filing in submissions.to_dict(orient="records"):
            try:
                text = _fetch_filing_text(meta["cik"], filing["accessionNumber"], filing["primaryDocument"])
            except Exception:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "company_name": meta["company_name"],
                    "form": filing["form"],
                    "filing_date": filing["filingDate"],
                    "accepted_at": filing["acceptanceDateTime"],
                    "accession_number": filing["accessionNumber"],
                    "primary_document": filing["primaryDocument"],
                    "text": text[:12000],
                }
            )
            time.sleep(0.2)
        if rows:
            pd.DataFrame(rows).drop_duplicates(subset=["ticker", "accession_number"]).to_parquet(cache_path, index=False)

    filings = pd.DataFrame(rows).drop_duplicates(subset=["ticker", "accession_number"])
    if filings.empty:
        raise RuntimeError("No SEC filing text could be scraped.")
    filings.to_parquet(cache_path, index=False)
    return filings


def _assign_trading_day(accepted_at: pd.Timestamp, trading_days: list[pd.Timestamp]) -> pd.Timestamp:
    eastern_time = accepted_at.tz_convert("America/New_York")
    market_day = pd.Timestamp(eastern_time.date())
    if eastern_time.hour >= MARKET_CLOSE_HOUR:
        market_day = market_day + pd.Timedelta(days=1)
    while market_day not in trading_days:
        market_day = market_day + pd.Timedelta(days=1)
    return market_day


def _score_filings(filings: pd.DataFrame) -> pd.DataFrame:
    scored = filings.copy()
    scored["accepted_at"] = pd.to_datetime(scored["accepted_at"], utc=True)
    text_series = scored["text"].fillna("").astype(str)
    scored["compound"] = _finbert_scores(text_series)
    scored["positive_flag"] = (scored["compound"] > 0.15).astype(int)
    scored["negative_flag"] = (scored["compound"] < -0.15).astype(int)
    scored["text_length"] = text_series.str.len()
    scored["novelty_score"] = _novelty_scores(scored)
    scored["is_8k"] = (scored["form"] == "8-K").astype(int)
    scored["is_periodic"] = scored["form"].isin({"10-Q", "10-K"}).astype(int)
    for feature_name, pattern in EVENT_PATTERNS.items():
        scored[feature_name] = text_series.str.contains(pattern, case=False, regex=True).astype(int)
    return scored


def _finbert_scores(text_series: pd.Series) -> pd.Series:
    cache_path = PROCESSED_DIR / "finbert_scores.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    truncated = text_series.str.slice(0, 1500)
    keys = [str(abs(hash(value))) for value in truncated.tolist()]
    missing_positions = [idx for idx, key in enumerate(keys) if key not in cache]
    if hf_pipeline is not None:
        try:
            if missing_positions:
                classifier = hf_pipeline("sentiment-analysis", model="ProsusAI/finbert", truncation=True)
                missing_text = [truncated.iloc[idx] for idx in missing_positions]
                outputs = classifier(missing_text, batch_size=8)
                label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
                for pos, output in zip(missing_positions, outputs, strict=False):
                    cache[keys[pos]] = label_map[output["label"].lower()] * float(output["score"])
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
            return pd.Series([float(cache[key]) for key in keys], index=text_series.index)
        except Exception:
            pass
    analyzer = SentimentIntensityAnalyzer()
    for pos in missing_positions:
        cache[keys[pos]] = analyzer.polarity_scores(truncated.iloc[pos])["compound"]
    if missing_positions:
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    return pd.Series([float(cache[key]) for key in keys], index=text_series.index)


def _novelty_scores(filings: pd.DataFrame) -> pd.Series:
    scores = pd.Series(1.0, index=filings.index, dtype=float)
    for ticker, group in filings.groupby("ticker"):
        ordered = group.sort_values("accepted_at")
        if len(ordered) <= 1:
            continue
        vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
        matrix = vectorizer.fit_transform(ordered["text"].fillna("").str.slice(0, 4000))
        sims = (matrix * matrix.T).toarray()
        for pos, idx in enumerate(ordered.index):
            if pos == 0:
                scores.loc[idx] = 1.0
                continue
            prior_max = float(np.max(sims[pos, :pos]))
            scores.loc[idx] = max(0.0, 1.0 - prior_max)
    return scores


def _build_daily_features(filings: pd.DataFrame, target_prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    scored = _score_filings(filings)
    trading_days = sorted(set(target_prices["XOM"].index))
    scored["trade_date"] = scored["accepted_at"].apply(lambda dt: _assign_trading_day(dt, trading_days))
    macro = _load_macro_political_features()
    official_events = build_official_event_features()
    gdelt_columns = [
        "gdelt_war_volume",
        "gdelt_shipping_volume",
        "gdelt_supply_volume",
        "gdelt_total_volume",
        "gdelt_war_volume_5d",
        "gdelt_shipping_volume_5d",
        "gdelt_supply_volume_5d",
        "gdelt_total_volume_5d",
    ]
    gdelt_grouped = pd.DataFrame(columns=["ticker", "trade_date"] + gdelt_columns)
    if USE_WEB_NEWS_TOOLS:
        gdelt_topics = fetch_gdelt_topic_features()
        gdelt_topics["trade_date"] = pd.to_datetime(gdelt_topics["date"])
        gdelt_grouped = gdelt_topics.rename(columns={"date": "raw_date"})

    grouped = (
        scored.groupby(["ticker", "trade_date"])
        .agg(
            filing_count=("accession_number", "count"),
            avg_sentiment=("compound", "mean"),
            sentiment_sum=("compound", "sum"),
            positive_share=("positive_flag", "mean"),
            negative_share=("negative_flag", "mean"),
            avg_text_length=("text_length", "mean"),
            novelty_mean=("novelty_score", "mean"),
            filing_8k_share=("is_8k", "mean"),
            filing_periodic_share=("is_periodic", "mean"),
            guidance_share=("guidance_flag", "mean"),
            earnings_share=("earnings_flag", "mean"),
            acquisition_share=("acquisition_flag", "mean"),
            production_share=("production_flag", "mean"),
            legal_share=("legal_flag", "mean"),
            geopolitics_share=("geopolitics_flag", "mean"),
            supply_disruption_share=("supply_disruption_flag", "mean"),
            shipping_share=("shipping_flag", "mean"),
            refining_share=("refining_flag", "mean"),
            upstream_share=("upstream_flag", "mean"),
            sanctions_share=("sanctions_flag", "mean"),
        )
        .reset_index()
    )
    all_rows = []
    for ticker, price_df in target_prices.items():
        merged = pd.DataFrame({"date": price_df.index})
        merged["ticker"] = ticker
        merged = merged.merge(grouped[grouped["ticker"] == ticker], left_on=["ticker", "date"], right_on=["ticker", "trade_date"], how="left")
        merged = merged.merge(official_events, on="date", how="left")
        if USE_WEB_NEWS_TOOLS:
            merged = merged.merge(
                gdelt_grouped[gdelt_grouped["ticker"] == ticker],
                left_on=["ticker", "date"],
                right_on=["ticker", "trade_date"],
                how="left",
                suffixes=("", "_web"),
            )
        for column in gdelt_columns:
            if column not in merged.columns:
                merged[column] = 0.0
        for column in [
            "filing_count",
            "avg_sentiment",
            "sentiment_sum",
            "positive_share",
            "negative_share",
            "avg_text_length",
            "novelty_mean",
            "filing_8k_share",
            "filing_periodic_share",
            "guidance_share",
            "earnings_share",
            "acquisition_share",
            "production_share",
            "legal_share",
            "geopolitics_share",
            "supply_disruption_share",
            "shipping_share",
            "refining_share",
            "upstream_share",
            "sanctions_share",
            "gdelt_war_volume",
            "gdelt_shipping_volume",
            "gdelt_supply_volume",
            "gdelt_total_volume",
            "gdelt_war_volume_5d",
            "gdelt_shipping_volume_5d",
            "gdelt_supply_volume_5d",
            "gdelt_total_volume_5d",
            "official_event_weight",
            "official_conflict_escalation_weight",
            "official_opec_supply_policy_weight",
            "official_sanctions_oil_weight",
            "official_sanctions_policy_weight",
            "official_shipping_enforcement_weight",
            "official_event_count",
            "official_event_weight_5d",
            "official_conflict_escalation_weight_5d",
            "official_opec_supply_policy_weight_5d",
            "official_sanctions_oil_weight_5d",
            "official_sanctions_policy_weight_5d",
            "official_shipping_enforcement_weight_5d",
        ]:
            merged[column] = merged[column].fillna(0)
        merged["filing_count_5d"] = merged["filing_count"].rolling(5).sum()
        merged["avg_sentiment_5d"] = merged["avg_sentiment"].rolling(5).mean()
        merged["geopolitics_sentiment"] = merged["avg_sentiment"] * merged["geopolitics_share"]
        merged["supply_risk_intensity"] = merged["novelty_mean"] * merged["geopolitics_share"] * (1 + merged["filing_count"])
        merged["supply_chain_intensity"] = merged["novelty_mean"] * (
            merged["shipping_share"] + merged["supply_disruption_share"] + merged["sanctions_share"]
        )
        merged["gdelt_supply_threat_intensity"] = (
            merged["gdelt_war_volume"] + merged["gdelt_shipping_volume"]
        ) * (1 + merged["gdelt_total_volume_5d"])
        merged["gdelt_supply_confirmation"] = merged["gdelt_supply_volume"] * (1 + merged["gdelt_war_volume"])
        merged["official_policy_confirmation"] = merged["official_event_weight"] * (1 + merged["geopolitics_share"])
        merged["official_supply_confirmation"] = (
            merged["official_sanctions_oil_weight"]
            + merged["official_shipping_enforcement_weight"]
            + merged["official_opec_supply_policy_weight"]
        ) * (1 + merged["supply_chain_intensity"])
        merged = merged.merge(macro, on="date", how="left")
        for column in [
            "gpr_level",
            "gpr_act",
            "gpr_threat",
            "gpr_ma7",
            "gpr_ma30",
            "gpr_surprise",
            "gpr_threat_share",
            "gpr_act_share",
            "ovx_close",
            "ovx_ret_1",
            "ovx_ret_5",
            "uso_close",
            "uso_ret_1",
            "uso_ret_5",
            "xop_close",
            "xop_ret_1",
            "xop_ret_5",
        ]:
            merged[column] = merged[column].ffill().fillna(0.0)
        merged["gpr_geopolitics_interaction"] = merged["gpr_level"] * merged["geopolitics_share"]
        merged["gpr_threat_sentiment"] = merged["gpr_threat"] * merged["avg_sentiment"]
        merged["ovx_supply_stress"] = merged["ovx_ret_1"] * merged["supply_risk_intensity"]
        merged["target"] = price_df["target"].values
        merged["excess_ret_fwd"] = price_df["excess_ret_fwd"].values
        merged["prev_ret_1"] = price_df["prev_ret_1"].values
        merged["prev_ret_5"] = price_df["prev_ret_5"].values
        all_rows.append(merged)

    panel = pd.concat(all_rows, ignore_index=True)
    drop_columns = [column for column in ["trade_date", "trade_date_web", "raw_date"] if column in panel.columns]
    if drop_columns:
        panel = panel.drop(columns=drop_columns)
    panel = panel.dropna(subset=["target", "prev_ret_1", "prev_ret_5"]).reset_index(drop=True)
    panel = add_theory_agent_features(panel)
    return panel
