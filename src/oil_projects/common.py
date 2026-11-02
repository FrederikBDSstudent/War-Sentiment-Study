import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
import yfinance as yf

from oil_projects.config import (
    ARTIFACT_DIR,
    BENCHMARK_TICKER,
    DATA_DIR,
    EXCESS_RETURN_THRESHOLD,
    IRAN_WAR_END_DATE,
    PROCESSED_DIR,
    RAW_DIR,
    RANDOM_SEED,
    START_DATE,
    TEST_START,
    TRAIN_END,
    VALID_END,
    WAR_SPLIT_DATE,
)


def ensure_directories() -> None:
    for path in (DATA_DIR, RAW_DIR, PROCESSED_DIR, ARTIFACT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0).rolling(window).mean()
    losses = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gains / losses.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series) -> pd.Series:
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def build_excess_return_target(
    asset_close: pd.Series,
    benchmark_close: pd.Series,
    threshold: float = EXCESS_RETURN_THRESHOLD,
) -> pd.DataFrame:
    asset_ret_fwd = asset_close.shift(-1) / asset_close - 1
    benchmark_aligned = benchmark_close.reindex(asset_close.index).ffill()
    benchmark_ret_fwd = benchmark_aligned.shift(-1) / benchmark_aligned - 1
    excess_ret_fwd = asset_ret_fwd - benchmark_ret_fwd
    target = pd.Series(np.where(excess_ret_fwd > threshold, 1, np.where(excess_ret_fwd < -threshold, 0, np.nan)), index=asset_close.index)
    return pd.DataFrame(
        {
            "asset_ret_fwd": asset_ret_fwd,
            "benchmark_ret_fwd": benchmark_ret_fwd,
            "excess_ret_fwd": excess_ret_fwd,
            "target": target,
        }
    )


def choose_classification_threshold(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    thresholds: Iterable[float] | None = None,
) -> float:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_prob_arr = np.asarray(list(y_prob), dtype=float)
    if len(y_true_arr) == 0:
        return 0.5
    best_threshold = 0.5
    best_score = -1.0
    threshold_grid = thresholds if thresholds is not None else np.linspace(0.3, 0.7, 41)
    for threshold in threshold_grid:
        score = float(accuracy_score(y_true_arr, (y_prob_arr >= threshold).astype(int)))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def shift_probabilities_for_threshold(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    shifted = np.asarray(probabilities, dtype=float) - float(threshold) + 0.5
    return np.clip(shifted, 0.0, 1.0)


def build_supply_uncertainty_regimes() -> pd.DataFrame:
    tickers = ["CL=F", "BZ=F", "^VIX", BENCHMARK_TICKER]
    frames = {}
    for ticker in tickers:
        frame = yf.download(ticker, start=START_DATE, progress=False, auto_adjust=True)
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame.columns = [str(column).lower() for column in frame.columns]
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        frames[ticker] = frame

    regime = pd.DataFrame(index=frames["CL=F"].index.union(frames["BZ=F"].index).union(frames["^VIX"].index).union(frames[BENCHMARK_TICKER].index))
    regime["cl_ret"] = frames["CL=F"]["close"].reindex(regime.index).pct_change(fill_method=None)
    regime["bz_ret"] = frames["BZ=F"]["close"].reindex(regime.index).pct_change(fill_method=None)
    regime["vix_ret"] = frames["^VIX"]["close"].reindex(regime.index).pct_change(fill_method=None)
    regime["xle_ret"] = frames[BENCHMARK_TICKER]["close"].reindex(regime.index).pct_change(fill_method=None)
    regime["iran_war_window"] = (
        (regime.index >= pd.Timestamp(WAR_SPLIT_DATE)) & (regime.index <= pd.Timestamp(IRAN_WAR_END_DATE))
    ).astype(int)
    regime["supply_uncertainty"] = (
        regime["iran_war_window"].astype(bool)
        | (regime["cl_ret"].abs() >= 0.025)
        | (regime["bz_ret"].abs() >= 0.025)
        | (regime["vix_ret"] >= 0.10)
    ).astype(int)
    regime["oil_shock_day"] = (
        (regime["cl_ret"].abs() >= 0.03)
        | (regime["bz_ret"].abs() >= 0.03)
    ).astype(int)
    regime["market_stress_day"] = (regime["vix_ret"] >= 0.12).astype(int)
    regime["normal_regime"] = (1 - regime["supply_uncertainty"]).astype(int)
    regime = regime.reset_index()
    regime = regime.rename(columns={regime.columns[0]: "date"})
    return regime


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_valid_test_label(date_value: pd.Timestamp) -> str:
    if date_value <= pd.Timestamp(TRAIN_END):
        return "train"
    if date_value <= pd.Timestamp(VALID_END):
        return "valid"
    if date_value >= pd.Timestamp(TEST_START):
        return "test"
    return "drop"


def monthly_test_windows(dates: pd.Series | pd.Index) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    monthly = pd.Series(pd.to_datetime(dates)).sort_values().dt.to_period("M").drop_duplicates().sort_values()
    windows = []
    for period in monthly:
        start = period.to_timestamp()
        end = period.to_timestamp(how="end").normalize()
        windows.append((start, end))
    return windows
