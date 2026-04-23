from __future__ import annotations

from datetime import datetime
import json
import os
import time
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
try:
    from transformers import pipeline as hf_pipeline
except Exception:  # pragma: no cover - fallback when transformers is unavailable
    hf_pipeline = None
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf

from oil_projects.common import (
    bootstrap_accuracy_lift_by_regime,
    build_excess_return_target,
    build_supply_uncertainty_regimes,
    choose_classification_threshold,
    ensure_directories,
    evaluate_by_period,
    evaluate_regression_by_period,
    evaluate_regression_supply_uncertainty,
    evaluate_supply_uncertainty,
    evaluate_by_ticker,
    monthly_test_windows,
    save_json,
    set_seed,
    shift_probabilities_for_threshold,
)
from oil_projects.config import (
    ARTIFACT_DIR,
    BENCHMARK_TICKER,
    MARKET_CLOSE_HOUR,
    NEWS_START_DATE,
    PROCESSED_DIR,
    RANDOM_SEED,
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


def _run_matched_walkforward_model(
    model_frame: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    sample_weight_column: str | None = None,
    model_kind: str = "xgb",
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    predictions_list = []
    for ticker, ticker_frame in model_frame.groupby("ticker"):
        ticker_frame = ticker_frame.sort_values("date").reset_index(drop=True)
        ticker_test_dates = ticker_frame.loc[pd.to_datetime(ticker_frame["date"]) >= pd.Timestamp("2025-01-01"), "date"]
        for window_start, window_end in monthly_test_windows(ticker_test_dates):
            train_mask = pd.to_datetime(ticker_frame["date"]) < window_start
            test_mask = (pd.to_datetime(ticker_frame["date"]) >= window_start) & (pd.to_datetime(ticker_frame["date"]) <= window_end)
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
            x_train = ticker_frame.loc[train_mask, feature_columns]
            y_train = ticker_frame.loc[train_mask, "target"]
            x_test = ticker_frame.loc[test_mask, feature_columns]
            y_test = ticker_frame.loc[test_mask, "target"]
            if y_train.nunique() < 2:
                continue

            pipeline = _build_classifier_pipeline(model_kind)
            fit_kwargs = {}
            if sample_weight_column is not None:
                fit_kwargs["model__sample_weight"] = ticker_frame.loc[train_mask, sample_weight_column].to_numpy(dtype=float)
            pipeline.fit(x_train, y_train, **fit_kwargs)
            probabilities = pipeline.predict_proba(x_test)[:, 1]
            predictions_list.append(
                pd.DataFrame(
                    {
                        "date": ticker_frame.loc[test_mask, "date"].values,
                        "ticker": ticker,
                        "target": y_test.values,
                        "probability": probabilities,
                        "model_name": model_name,
                    }
                )
            )

    predictions = pd.concat(predictions_list, ignore_index=True).sort_values(["date", "ticker"])
    metrics = evaluate_by_period(predictions)
    ticker_metrics = evaluate_by_ticker(predictions)
    return predictions, metrics, ticker_metrics


def _run_regression_walkforward_model(
    model_frame: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    sample_weight_column: str | None = None,
    model_kind: str = "xgb",
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    predictions_list = []
    for ticker, ticker_frame in model_frame.groupby("ticker"):
        ticker_frame = ticker_frame.sort_values("date").reset_index(drop=True)
        ticker_test_dates = ticker_frame.loc[pd.to_datetime(ticker_frame["date"]) >= pd.Timestamp("2025-01-01"), "date"]
        for window_start, window_end in monthly_test_windows(ticker_test_dates):
            train_mask = pd.to_datetime(ticker_frame["date"]) < window_start
            test_mask = (pd.to_datetime(ticker_frame["date"]) >= window_start) & (pd.to_datetime(ticker_frame["date"]) <= window_end)
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
            x_train = ticker_frame.loc[train_mask, feature_columns]
            y_train = ticker_frame.loc[train_mask, "excess_ret_fwd"]
            x_test = ticker_frame.loc[test_mask, feature_columns]
            y_test = ticker_frame.loc[test_mask, "excess_ret_fwd"]

            if model_kind == "rf":
                pipeline = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                        (
                            "model",
                            RandomForestRegressor(
                                n_estimators=500,
                                max_depth=6,
                                min_samples_leaf=8,
                                random_state=RANDOM_SEED,
                            ),
                        ),
                    ]
                )
            else:
                pipeline = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            XGBRegressor(
                                n_estimators=250,
                                max_depth=3,
                                learning_rate=0.04,
                                subsample=0.9,
                                colsample_bytree=0.8,
                                random_state=RANDOM_SEED,
                            ),
                        ),
                    ]
                )
            fit_kwargs = {}
            if sample_weight_column is not None:
                fit_kwargs["model__sample_weight"] = ticker_frame.loc[train_mask, sample_weight_column].to_numpy(dtype=float)
            pipeline.fit(x_train, y_train, **fit_kwargs)
            predictions_list.append(
                pd.DataFrame(
                    {
                        "date": ticker_frame.loc[test_mask, "date"].values,
                        "ticker": ticker,
                        "actual": y_test.values,
                        "prediction": pipeline.predict(x_test),
                        "model_name": model_name,
                    }
                )
            )

    predictions = pd.concat(predictions_list, ignore_index=True).sort_values(["date", "ticker"])
    metrics = evaluate_regression_by_period(predictions)
    ticker_metrics = {
        str(ticker): evaluate_regression_by_period(group.assign(date=pd.to_datetime(group["date"])))["overall"]
        for ticker, group in predictions.groupby("ticker")
    }
    return predictions, metrics, ticker_metrics


def _build_classifier_pipeline(model_kind: str) -> Pipeline:
    if model_kind == "logit":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, C=0.7)),
            ]
        )
    if model_kind == "rf":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=6,
                        min_samples_leaf=8,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
    if model_kind == "et":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=8,
                        min_samples_leaf=6,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", StandardScaler()),
            (
                "model",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.04,
                    subsample=0.9,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def _split_train_validation_frame(train_frame: pd.DataFrame, validation_rows: int = 80) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = train_frame.sort_values("date").reset_index(drop=True)
    validation_rows = min(validation_rows, max(20, len(ordered) // 4))
    if validation_rows >= len(ordered):
        validation_rows = max(10, len(ordered) // 3)
    valid_frame = ordered.tail(validation_rows).copy()
    fit_frame = ordered.iloc[:-validation_rows].copy()
    if fit_frame.empty or fit_frame["target"].nunique() < 2 or valid_frame["target"].nunique() < 2:
        fallback_rows = min(40, max(10, len(ordered) // 5))
        valid_frame = ordered.tail(fallback_rows).copy()
        fit_frame = ordered.iloc[:-fallback_rows].copy()
    return fit_frame, valid_frame


def _run_auto_walkforward_model(
    model_frame: pd.DataFrame,
    candidate_specs: list[dict[str, object]],
    model_name: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], dict[str, dict[str, float]], pd.DataFrame]:
    predictions_list = []
    choice_rows = []
    for ticker, ticker_frame in model_frame.groupby("ticker"):
        ticker_frame = ticker_frame.sort_values("date").reset_index(drop=True)
        ticker_test_dates = ticker_frame.loc[pd.to_datetime(ticker_frame["date"]) >= pd.Timestamp("2025-01-01"), "date"]
        for window_start, window_end in monthly_test_windows(ticker_test_dates):
            train_mask = pd.to_datetime(ticker_frame["date"]) < window_start
            test_mask = (pd.to_datetime(ticker_frame["date"]) >= window_start) & (pd.to_datetime(ticker_frame["date"]) <= window_end)
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue
            train_frame = ticker_frame.loc[train_mask].copy()
            fit_frame, valid_frame = _split_train_validation_frame(train_frame)
            if fit_frame.empty or valid_frame.empty:
                continue

            best_choice = None
            for candidate in candidate_specs:
                feature_columns = list(candidate["feature_columns"])
                model_kind = str(candidate["model_kind"])
                candidate_name = str(candidate["candidate_name"])
                sample_weight_column = candidate.get("sample_weight_column")
                if fit_frame["target"].nunique() < 2 or valid_frame["target"].nunique() < 2:
                    continue
                pipeline = _build_classifier_pipeline(model_kind)
                fit_kwargs = {}
                if sample_weight_column is not None:
                    fit_kwargs["model__sample_weight"] = fit_frame[str(sample_weight_column)].to_numpy(dtype=float)
                pipeline.fit(fit_frame[feature_columns], fit_frame["target"], **fit_kwargs)
                valid_probabilities = pipeline.predict_proba(valid_frame[feature_columns])[:, 1]
                threshold = choose_classification_threshold(valid_frame["target"], valid_probabilities)
                validation_accuracy = float(accuracy_score(valid_frame["target"], (valid_probabilities >= threshold).astype(int)))
                if best_choice is None or validation_accuracy > best_choice["validation_accuracy"]:
                    best_choice = {
                        "candidate_name": candidate_name,
                        "feature_columns": feature_columns,
                        "model_kind": model_kind,
                        "sample_weight_column": sample_weight_column,
                        "threshold": float(threshold),
                        "validation_accuracy": validation_accuracy,
                    }

            if best_choice is None:
                continue

            final_pipeline = _build_classifier_pipeline(best_choice["model_kind"])
            final_fit_kwargs = {}
            if best_choice["sample_weight_column"] is not None:
                final_fit_kwargs["model__sample_weight"] = train_frame[str(best_choice["sample_weight_column"])].to_numpy(dtype=float)
            final_pipeline.fit(train_frame[best_choice["feature_columns"]], train_frame["target"], **final_fit_kwargs)
            test_probabilities = final_pipeline.predict_proba(ticker_frame.loc[test_mask, best_choice["feature_columns"]])[:, 1]
            calibrated_probabilities = shift_probabilities_for_threshold(test_probabilities, best_choice["threshold"])
            predictions_list.append(
                pd.DataFrame(
                    {
                        "date": ticker_frame.loc[test_mask, "date"].values,
                        "ticker": ticker,
                        "target": ticker_frame.loc[test_mask, "target"].values,
                        "probability": calibrated_probabilities,
                        "model_name": model_name,
                    }
                )
            )
            choice_rows.append(
                {
                    "ticker": ticker,
                    "window_start": window_start,
                    "window_end": window_end,
                    "candidate_name": best_choice["candidate_name"],
                    "model_kind": best_choice["model_kind"],
                    "threshold": best_choice["threshold"],
                    "validation_accuracy": best_choice["validation_accuracy"],
                    "train_rows": int(len(train_frame)),
                    "validation_rows": int(len(valid_frame)),
                    "test_rows": int(test_mask.sum()),
                    "train_max_date": pd.to_datetime(train_frame["date"]).max(),
                    "validation_start_date": pd.to_datetime(valid_frame["date"]).min(),
                    "validation_end_date": pd.to_datetime(valid_frame["date"]).max(),
                }
            )

    predictions = pd.concat(predictions_list, ignore_index=True).sort_values(["date", "ticker"])
    metrics = evaluate_by_period(predictions)
    ticker_metrics = evaluate_by_ticker(predictions)
    choices = pd.DataFrame(choice_rows).sort_values(["ticker", "window_start"])
    return predictions, metrics, ticker_metrics, choices


def run_news_project() -> dict:
    ensure_directories()
    set_seed(RANDOM_SEED)

    target_prices = _download_target_prices()
    filings = _fetch_filings()
    filings.to_parquet(PROCESSED_DIR / "raw_news.parquet", index=False)

    panel = _build_daily_features(filings, target_prices)
    panel.to_parquet(PROCESSED_DIR / "news_panel.parquet", index=False)

    baseline_feature_columns = [
        "filing_count",
        "avg_text_length",
        "novelty_mean",
        "filing_8k_share",
        "filing_periodic_share",
        "filing_count_5d",
        "prev_ret_1",
        "prev_ret_5",
        "ovx_close",
        "ovx_ret_1",
        "ovx_ret_5",
        "uso_close",
        "uso_ret_1",
        "uso_ret_5",
        "xop_close",
        "xop_ret_1",
        "xop_ret_5",
    ]
    sentiment_feature_columns = baseline_feature_columns + [
        "avg_sentiment",
        "sentiment_sum",
        "positive_share",
        "negative_share",
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
        "avg_sentiment_5d",
        "gdelt_war_volume",
        "gdelt_shipping_volume",
        "gdelt_supply_volume",
        "gdelt_total_volume",
        "gdelt_war_volume_5d",
        "gdelt_shipping_volume_5d",
        "gdelt_supply_volume_5d",
        "gdelt_total_volume_5d",
        "gdelt_supply_threat_intensity",
        "gdelt_supply_confirmation",
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
        "official_policy_confirmation",
        "official_supply_confirmation",
        "geopolitics_sentiment",
        "supply_risk_intensity",
        "supply_chain_intensity",
        "gpr_level",
        "gpr_act",
        "gpr_threat",
        "gpr_ma7",
        "gpr_ma30",
        "gpr_surprise",
        "gpr_threat_share",
        "gpr_act_share",
        "gpr_geopolitics_interaction",
        "gpr_threat_sentiment",
        "ovx_close",
        "ovx_ret_1",
        "ovx_ret_5",
        "uso_close",
        "uso_ret_1",
        "uso_ret_5",
        "xop_close",
        "xop_ret_1",
        "xop_ret_5",
        "ovx_supply_stress",
        "agent_political_risk",
        "agent_company_exposure",
        "agent_market_stress",
        "agent_disclosure_signal",
        "agent_war_alignment",
        "agent_supply_alignment",
        "agent_text_market_confirmation",
        "agent_uncertainty_confirmation",
    ]
    war_feature_columns = [
        "avg_sentiment",
        "sentiment_sum",
        "positive_share",
        "negative_share",
        "gdelt_war_volume",
        "gdelt_shipping_volume",
        "gdelt_supply_volume",
        "gdelt_total_volume",
        "gdelt_war_volume_5d",
        "gdelt_shipping_volume_5d",
        "gdelt_supply_volume_5d",
        "gdelt_total_volume_5d",
        "gdelt_supply_threat_intensity",
        "gdelt_supply_confirmation",
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
        "official_policy_confirmation",
        "official_supply_confirmation",
        "geopolitics_share",
        "supply_disruption_share",
        "shipping_share",
        "sanctions_share",
        "geopolitics_sentiment",
        "supply_risk_intensity",
        "supply_chain_intensity",
        "gpr_level",
        "gpr_act",
        "gpr_threat",
        "gpr_surprise",
        "gpr_threat_share",
        "gpr_act_share",
        "gpr_geopolitics_interaction",
        "gpr_threat_sentiment",
        "ovx_close",
        "ovx_ret_1",
        "ovx_ret_5",
        "ovx_supply_stress",
        "uso_ret_1",
        "xop_ret_1",
        "prev_ret_1",
        "prev_ret_5",
        "agent_political_risk",
        "agent_company_exposure",
        "agent_market_stress",
        "agent_disclosure_signal",
        "agent_war_alignment",
        "agent_supply_alignment",
        "agent_text_market_confirmation",
        "agent_uncertainty_confirmation",
        "official_event_weight",
        "official_conflict_escalation_weight",
        "official_sanctions_oil_weight",
        "official_shipping_enforcement_weight",
        "official_event_weight_5d",
        "official_policy_confirmation",
        "official_supply_confirmation",
    ]
    agent_feature_columns = [
        "avg_sentiment",
        "sentiment_sum",
        "positive_share",
        "negative_share",
        "geopolitics_share",
        "supply_disruption_share",
        "shipping_share",
        "sanctions_share",
        "gpr_level",
        "gpr_threat",
        "gpr_surprise",
        "gpr_threat_share",
        "ovx_ret_1",
        "ovx_ret_5",
        "uso_ret_1",
        "xop_ret_1",
        "prev_ret_1",
        "prev_ret_5",
        "agent_political_risk",
        "agent_company_exposure",
        "agent_market_stress",
        "agent_disclosure_signal",
        "agent_war_alignment",
        "agent_supply_alignment",
        "agent_text_market_confirmation",
        "agent_uncertainty_confirmation",
    ]
    jackpot_feature_columns = [
        "avg_sentiment",
        "sentiment_sum",
        "geopolitics_share",
        "shipping_share",
        "sanctions_share",
        "supply_disruption_share",
        "supply_chain_intensity",
        "geopolitics_sentiment",
        "gpr_level",
        "gpr_threat",
        "gpr_surprise",
        "gpr_threat_share",
        "ovx_ret_1",
        "ovx_ret_5",
        "official_event_weight",
        "official_conflict_escalation_weight",
        "official_sanctions_oil_weight",
        "official_shipping_enforcement_weight",
        "official_policy_confirmation",
        "official_supply_confirmation",
        "agent_political_risk",
        "agent_company_exposure",
        "agent_market_stress",
        "agent_disclosure_signal",
        "agent_war_alignment",
        "agent_supply_alignment",
        "agent_text_market_confirmation",
        "agent_uncertainty_confirmation",
        "prev_ret_1",
        "prev_ret_5",
    ]
    jackpot_plus_feature_columns = jackpot_feature_columns + [
        "official_x_war_alignment",
        "threat_x_shipping",
        "threat_x_sanctions",
        "ovx_x_war_alignment",
        "prev_x_official",
    ]

    regime_frame = build_supply_uncertainty_regimes()
    model_frame = panel[["date", "ticker", "target", "excess_ret_fwd"] + sorted(set(baseline_feature_columns + sentiment_feature_columns + war_feature_columns + agent_feature_columns))].copy()
    model_frame = model_frame.merge(
        regime_frame[["date", "iran_war_window", "supply_uncertainty", "normal_regime"]],
        on="date",
        how="left",
    )
    model_frame["supply_uncertainty"] = model_frame["supply_uncertainty"].fillna(0).astype(int)
    model_frame["regime_weight"] = np.where(model_frame["supply_uncertainty"] == 1, 3.0, 1.0)
    model_frame["war_weight"] = np.where(
        model_frame["iran_war_window"].fillna(0).astype(int) == 1,
        16.0,
        np.where(model_frame["supply_uncertainty"] == 1, 8.0, 1.0),
    )
    model_frame["official_x_war_alignment"] = model_frame["official_event_weight"] * model_frame["agent_war_alignment"]
    model_frame["threat_x_shipping"] = model_frame["gpr_threat"] * model_frame["shipping_share"]
    model_frame["threat_x_sanctions"] = model_frame["gpr_threat"] * model_frame["sanctions_share"]
    model_frame["ovx_x_war_alignment"] = model_frame["ovx_ret_1"] * model_frame["agent_war_alignment"]
    model_frame["prev_x_official"] = model_frame["prev_ret_1"] * model_frame["official_event_weight"]

    baseline_predictions, baseline_metrics, baseline_ticker_metrics = _run_matched_walkforward_model(
        model_frame,
        baseline_feature_columns,
        "baseline_without_sentiment",
    )
    predictions, metrics, ticker_metrics = _run_matched_walkforward_model(
        model_frame,
        sentiment_feature_columns,
        "sentiment_with_text",
    )
    specialist_predictions, specialist_metrics, specialist_ticker_metrics = _run_matched_walkforward_model(
        model_frame,
        sentiment_feature_columns,
        "sentiment_regime_specialist",
        sample_weight_column="regime_weight",
    )
    war_specialist_predictions, war_specialist_metrics, war_specialist_ticker_metrics = _run_matched_walkforward_model(
        model_frame,
        war_feature_columns,
        "sentiment_war_specialist",
        sample_weight_column="war_weight",
        model_kind="logit",
    )
    agent_predictions, agent_metrics, agent_ticker_metrics = _run_matched_walkforward_model(
        model_frame,
        agent_feature_columns,
        "sentiment_agent_specialist",
        sample_weight_column="war_weight",
        model_kind="xgb",
    )
    jackpot_predictions, jackpot_metrics, jackpot_ticker_metrics, jackpot_window_choices = _run_auto_walkforward_model(
        model_frame,
        [
            {
                "candidate_name": "jackpot_logit",
                "feature_columns": jackpot_feature_columns,
                "model_kind": "logit",
            },
            {
                "candidate_name": "jackpot_plus_logit",
                "feature_columns": jackpot_plus_feature_columns,
                "model_kind": "logit",
            },
            {
                "candidate_name": "jackpot_xgb",
                "feature_columns": jackpot_feature_columns,
                "model_kind": "xgb",
            },
            {
                "candidate_name": "jackpot_plus_xgb",
                "feature_columns": jackpot_plus_feature_columns,
                "model_kind": "xgb",
            },
            {
                "candidate_name": "war_xgb_weighted",
                "feature_columns": war_feature_columns,
                "model_kind": "xgb",
                "sample_weight_column": "war_weight",
            },
            {
                "candidate_name": "agent_logit",
                "feature_columns": agent_feature_columns,
                "model_kind": "logit",
            },
            {
                "candidate_name": "agent_xgb_weighted",
                "feature_columns": agent_feature_columns,
                "model_kind": "xgb",
                "sample_weight_column": "war_weight",
            },
        ],
        "sentiment_jackpot_auto",
    )
    regression_predictions, regression_metrics, regression_ticker_metrics = _run_regression_walkforward_model(
        model_frame,
        agent_feature_columns,
        "sentiment_agent_regression",
        sample_weight_column="war_weight",
    )
    jackpot_regression_predictions, jackpot_regression_metrics, jackpot_regression_ticker_metrics = _run_regression_walkforward_model(
        model_frame,
        jackpot_feature_columns,
        "sentiment_jackpot_regression_rf",
        sample_weight_column="war_weight",
        model_kind="rf",
    )
    baseline_regime_metrics = evaluate_supply_uncertainty(baseline_predictions, regime_frame)
    regime_metrics = evaluate_supply_uncertainty(predictions, regime_frame)
    specialist_regime_metrics = evaluate_supply_uncertainty(specialist_predictions, regime_frame)
    war_specialist_regime_metrics = evaluate_supply_uncertainty(war_specialist_predictions, regime_frame)
    agent_regime_metrics = evaluate_supply_uncertainty(agent_predictions, regime_frame)
    jackpot_regime_metrics = evaluate_supply_uncertainty(jackpot_predictions, regime_frame)
    regression_regime_metrics = evaluate_regression_supply_uncertainty(regression_predictions, regime_frame)
    jackpot_regression_regime_metrics = evaluate_regression_supply_uncertainty(jackpot_regression_predictions, regime_frame)
    comparison = {
        "overall_accuracy_lift": metrics["overall"]["accuracy"] - baseline_metrics["overall"]["accuracy"],
        "iran_war_window_accuracy_lift": regime_metrics["iran_war_window"]["accuracy"] - baseline_regime_metrics["iran_war_window"]["accuracy"],
        "supply_uncertainty_accuracy_lift": regime_metrics["supply_uncertainty"]["accuracy"] - baseline_regime_metrics["supply_uncertainty"]["accuracy"],
        "normal_regime_accuracy_lift": regime_metrics["normal_regime"]["accuracy"] - baseline_regime_metrics["normal_regime"]["accuracy"],
    }
    lift_confidence = bootstrap_accuracy_lift_by_regime(predictions, baseline_predictions, regime_frame, n_bootstrap=400)
    specialist_comparison = {
        "overall_accuracy_lift": specialist_metrics["overall"]["accuracy"] - baseline_metrics["overall"]["accuracy"],
        "iran_war_window_accuracy_lift": specialist_regime_metrics["iran_war_window"]["accuracy"] - baseline_regime_metrics["iran_war_window"]["accuracy"],
        "supply_uncertainty_accuracy_lift": specialist_regime_metrics["supply_uncertainty"]["accuracy"] - baseline_regime_metrics["supply_uncertainty"]["accuracy"],
        "normal_regime_accuracy_lift": specialist_regime_metrics["normal_regime"]["accuracy"] - baseline_regime_metrics["normal_regime"]["accuracy"],
    }
    war_specialist_comparison = {
        "overall_accuracy_lift": war_specialist_metrics["overall"]["accuracy"] - baseline_metrics["overall"]["accuracy"],
        "iran_war_window_accuracy_lift": war_specialist_regime_metrics["iran_war_window"]["accuracy"] - baseline_regime_metrics["iran_war_window"]["accuracy"],
        "supply_uncertainty_accuracy_lift": war_specialist_regime_metrics["supply_uncertainty"]["accuracy"] - baseline_regime_metrics["supply_uncertainty"]["accuracy"],
        "normal_regime_accuracy_lift": war_specialist_regime_metrics["normal_regime"]["accuracy"] - baseline_regime_metrics["normal_regime"]["accuracy"],
    }
    agent_comparison = {
        "overall_accuracy_lift": agent_metrics["overall"]["accuracy"] - baseline_metrics["overall"]["accuracy"],
        "iran_war_window_accuracy_lift": agent_regime_metrics["iran_war_window"]["accuracy"] - baseline_regime_metrics["iran_war_window"]["accuracy"],
        "supply_uncertainty_accuracy_lift": agent_regime_metrics["supply_uncertainty"]["accuracy"] - baseline_regime_metrics["supply_uncertainty"]["accuracy"],
        "normal_regime_accuracy_lift": agent_regime_metrics["normal_regime"]["accuracy"] - baseline_regime_metrics["normal_regime"]["accuracy"],
    }
    jackpot_comparison = {
        "overall_accuracy_lift": jackpot_metrics["overall"]["accuracy"] - baseline_metrics["overall"]["accuracy"],
        "iran_war_window_accuracy_lift": jackpot_regime_metrics["iran_war_window"]["accuracy"] - baseline_regime_metrics["iran_war_window"]["accuracy"],
        "supply_uncertainty_accuracy_lift": jackpot_regime_metrics["supply_uncertainty"]["accuracy"] - baseline_regime_metrics["supply_uncertainty"]["accuracy"],
        "normal_regime_accuracy_lift": jackpot_regime_metrics["normal_regime"]["accuracy"] - baseline_regime_metrics["normal_regime"]["accuracy"],
    }
    predictions.to_csv(ARTIFACT_DIR / "news_predictions.csv", index=False)
    baseline_predictions.to_csv(ARTIFACT_DIR / "news_baseline_predictions.csv", index=False)
    specialist_predictions.to_csv(ARTIFACT_DIR / "news_specialist_predictions.csv", index=False)
    war_specialist_predictions.to_csv(ARTIFACT_DIR / "news_war_specialist_predictions.csv", index=False)
    agent_predictions.to_csv(ARTIFACT_DIR / "news_agent_predictions.csv", index=False)
    regression_predictions.to_csv(ARTIFACT_DIR / "news_regression_predictions.csv", index=False)
    jackpot_predictions.to_csv(ARTIFACT_DIR / "news_jackpot_predictions.csv", index=False)
    jackpot_window_choices.to_csv(ARTIFACT_DIR / "news_jackpot_window_choices.csv", index=False)
    jackpot_regression_predictions.to_csv(ARTIFACT_DIR / "news_jackpot_regression_predictions.csv", index=False)
    jackpot_window_choice_records = jackpot_window_choices.assign(
        window_start=jackpot_window_choices["window_start"].astype(str),
        window_end=jackpot_window_choices["window_end"].astype(str),
        train_max_date=jackpot_window_choices["train_max_date"].astype(str),
        validation_start_date=jackpot_window_choices["validation_start_date"].astype(str),
        validation_end_date=jackpot_window_choices["validation_end_date"].astype(str),
    ).to_dict(orient="records")
    save_json(
        ARTIFACT_DIR / "news_metrics.json",
        {
            "project_name": "sec_filing_sentiment",
            "baseline_features": baseline_feature_columns,
            "features": sentiment_feature_columns,
            "source": "SEC EDGAR primary filings",
            "forms": sorted(TARGET_FORMS),
            "target": "next-day excess return vs XLE with 25bp threshold",
            "walk_forward": "monthly expanding-window retraining",
            "sentiment_model": "ProsusAI/finbert with VADER fallback",
            "web_news_tools": USE_WEB_NEWS_TOOLS,
            "baseline_metrics": baseline_metrics,
            "baseline_regime_metrics": baseline_regime_metrics,
            "baseline_ticker_metrics": baseline_ticker_metrics,
            "metrics": metrics,
            "regime_metrics": regime_metrics,
            "ticker_metrics": ticker_metrics,
            "comparison": comparison,
            "lift_confidence": lift_confidence,
            "specialist_metrics": specialist_metrics,
            "specialist_regime_metrics": specialist_regime_metrics,
            "specialist_ticker_metrics": specialist_ticker_metrics,
            "specialist_comparison": specialist_comparison,
            "war_specialist_features": war_feature_columns,
            "war_specialist_metrics": war_specialist_metrics,
            "war_specialist_regime_metrics": war_specialist_regime_metrics,
            "war_specialist_ticker_metrics": war_specialist_ticker_metrics,
            "war_specialist_comparison": war_specialist_comparison,
            "agent_features": agent_feature_columns,
            "agent_metrics": agent_metrics,
            "agent_regime_metrics": agent_regime_metrics,
            "agent_ticker_metrics": agent_ticker_metrics,
            "agent_comparison": agent_comparison,
            "jackpot_features": jackpot_feature_columns,
            "jackpot_plus_features": jackpot_plus_feature_columns,
            "jackpot_threshold": "validation_selected_by_window",
            "jackpot_metrics": jackpot_metrics,
            "jackpot_regime_metrics": jackpot_regime_metrics,
            "jackpot_ticker_metrics": jackpot_ticker_metrics,
            "jackpot_comparison": jackpot_comparison,
            "jackpot_window_choices": jackpot_window_choice_records,
            "regression_target": "next-day excess return vs XLE",
            "regression_metrics": regression_metrics,
            "regression_regime_metrics": regression_regime_metrics,
            "regression_ticker_metrics": regression_ticker_metrics,
            "jackpot_regression_metrics": jackpot_regression_metrics,
            "jackpot_regression_regime_metrics": jackpot_regression_regime_metrics,
            "jackpot_regression_ticker_metrics": jackpot_regression_ticker_metrics,
        },
    )

    print("\nNews project")
    print("=" * 80)
    print("Matched baseline without sentiment")
    for period_name, period_metrics in baseline_metrics.items():
        print(
            f"{period_name:<10} accuracy={period_metrics['accuracy']:.4f} "
            f"balanced_accuracy={period_metrics['balanced_accuracy']:.4f} "
            f"auc={period_metrics['auc']:.4f} n={period_metrics['n']}"
        )
    print("Sentiment model")
    for period_name, period_metrics in metrics.items():
        print(
            f"{period_name:<10} accuracy={period_metrics['accuracy']:.4f} "
            f"balanced_accuracy={period_metrics['balanced_accuracy']:.4f} "
            f"auc={period_metrics['auc']:.4f} n={period_metrics['n']}"
        )
    print("Supply-uncertainty focus baseline")
    for period_name, period_metrics in baseline_regime_metrics.items():
        print(
            f"{period_name:<18} accuracy={period_metrics['accuracy']:.4f} "
            f"balanced_accuracy={period_metrics['balanced_accuracy']:.4f} "
            f"auc={period_metrics['auc']:.4f} n={period_metrics['n']}"
        )
    print("Supply-uncertainty focus sentiment")
    for period_name, period_metrics in regime_metrics.items():
        print(
            f"{period_name:<18} accuracy={period_metrics['accuracy']:.4f} "
            f"balanced_accuracy={period_metrics['balanced_accuracy']:.4f} "
            f"auc={period_metrics['auc']:.4f} n={period_metrics['n']}"
        )
    print("Sentiment lift over matched baseline")
    for name, value in comparison.items():
        print(f"{name:<32} {value:+.4f}")
    print("Bootstrap confidence for sentiment lift")
    for name, stats in lift_confidence.items():
        print(
            f"{name:<18} lift={stats['lift']:+.4f} "
            f"ci=[{stats['ci_low']:+.4f}, {stats['ci_high']:+.4f}] n={stats['n']}"
        )
    print("Regime-specialist sentiment model")
    for period_name, period_metrics in specialist_regime_metrics.items():
        print(
            f"{period_name:<18} accuracy={period_metrics['accuracy']:.4f} "
            f"balanced_accuracy={period_metrics['balanced_accuracy']:.4f} "
            f"auc={period_metrics['auc']:.4f} n={period_metrics['n']}"
        )
    print("Regime-specialist lift over matched baseline")
    for name, value in specialist_comparison.items():
        print(f"{name:<32} {value:+.4f}")
    print("War-specialist sentiment model")
    for period_name, period_metrics in war_specialist_regime_metrics.items():
        print(
            f"{period_name:<18} accuracy={period_metrics['accuracy']:.4f} "
            f"balanced_accuracy={period_metrics['balanced_accuracy']:.4f} "
            f"auc={period_metrics['auc']:.4f} n={period_metrics['n']}"
        )
    print("War-specialist lift over matched baseline")
    for name, value in war_specialist_comparison.items():
        print(f"{name:<32} {value:+.4f}")
    print("Agent-specialist sentiment model")
    for period_name, period_metrics in agent_regime_metrics.items():
        print(
            f"{period_name:<18} accuracy={period_metrics['accuracy']:.4f} "
            f"balanced_accuracy={period_metrics['balanced_accuracy']:.4f} "
            f"auc={period_metrics['auc']:.4f} n={period_metrics['n']}"
        )
    print("Agent-specialist lift over matched baseline")
    for name, value in agent_comparison.items():
        print(f"{name:<32} {value:+.4f}")
    print("Jackpot sentiment model")
    for period_name, period_metrics in jackpot_regime_metrics.items():
        print(
            f"{period_name:<18} accuracy={period_metrics['accuracy']:.4f} "
            f"balanced_accuracy={period_metrics['balanced_accuracy']:.4f} "
            f"auc={period_metrics['auc']:.4f} n={period_metrics['n']}"
        )
    print("Jackpot lift over matched baseline")
    for name, value in jackpot_comparison.items():
        print(f"{name:<32} {value:+.4f}")
    print("Agent-specialist regression model")
    for period_name, period_metrics in regression_regime_metrics.items():
        print(
            f"{period_name:<18} rmse={period_metrics['rmse']:.4f} "
            f"mae={period_metrics['mae']:.4f} "
            f"sign_accuracy={period_metrics['sign_accuracy']:.4f} "
            f"within_50bp={period_metrics['within_tolerance_accuracy']:.4f} n={period_metrics['n']}"
        )
    print("Jackpot regression model")
    for period_name, period_metrics in jackpot_regression_regime_metrics.items():
        print(
            f"{period_name:<18} rmse={period_metrics['rmse']:.4f} "
            f"mae={period_metrics['mae']:.4f} "
            f"sign_accuracy={period_metrics['sign_accuracy']:.4f} "
            f"within_50bp={period_metrics['within_tolerance_accuracy']:.4f} n={period_metrics['n']}"
        )

    return {
        "project_name": "sec_filing_sentiment",
        "metrics": metrics,
        "regime_metrics": regime_metrics,
        "baseline_metrics": baseline_metrics,
        "baseline_regime_metrics": baseline_regime_metrics,
        "comparison": comparison,
        "specialist_metrics": specialist_metrics,
        "specialist_regime_metrics": specialist_regime_metrics,
        "specialist_comparison": specialist_comparison,
        "war_specialist_metrics": war_specialist_metrics,
        "war_specialist_regime_metrics": war_specialist_regime_metrics,
        "war_specialist_comparison": war_specialist_comparison,
        "agent_metrics": agent_metrics,
        "agent_regime_metrics": agent_regime_metrics,
        "agent_comparison": agent_comparison,
        "jackpot_metrics": jackpot_metrics,
        "jackpot_regime_metrics": jackpot_regime_metrics,
        "jackpot_comparison": jackpot_comparison,
        "regression_metrics": regression_metrics,
        "regression_regime_metrics": regression_regime_metrics,
        "jackpot_regression_metrics": jackpot_regression_metrics,
        "jackpot_regression_regime_metrics": jackpot_regression_regime_metrics,
    }
