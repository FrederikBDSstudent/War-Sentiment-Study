import json
import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
import yfinance as yf

from oil_projects.config import (
    ARTIFACT_DIR,
    BENCHMARK_TICKER,
    DATA_DIR,
    EXCESS_RETURN_THRESHOLD,
    FACTOR_TICKERS,
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


def date_split_mask(index: pd.Index, start: str | None = None, end: str | None = None) -> np.ndarray:
    dates = pd.to_datetime(index)
    mask = np.ones(len(dates), dtype=bool)
    if start is not None:
        mask &= dates >= pd.Timestamp(start)
    if end is not None:
        mask &= dates <= pd.Timestamp(end)
    return mask


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


def directional_metrics(y_true: Iterable[int], y_prob: Iterable[float]) -> dict[str, float]:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_prob_arr = np.asarray(list(y_prob), dtype=float)
    y_pred_arr = (y_prob_arr >= 0.5).astype(int)
    auc = float("nan")
    if len(np.unique(y_true_arr)) > 1:
        auc = float(roc_auc_score(y_true_arr, y_prob_arr))
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "auc": auc,
        "n": int(len(y_true_arr)),
    }


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float], tolerance: float = 0.005) -> dict[str, float]:
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    if len(y_true_arr) == 0:
        return {
            "mae": math.nan,
            "rmse": math.nan,
            "r2": math.nan,
            "sign_accuracy": math.nan,
            "within_tolerance_accuracy": math.nan,
            "n": 0,
        }
    return {
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
        "sign_accuracy": float(np.mean(np.sign(y_true_arr) == np.sign(y_pred_arr))),
        "within_tolerance_accuracy": float(np.mean(np.abs(y_true_arr - y_pred_arr) <= tolerance)),
        "n": int(len(y_true_arr)),
    }


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


def evaluate_by_period(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    periods = {
        "overall": frame,
        "pre_war": frame[frame["date"] < pd.Timestamp(WAR_SPLIT_DATE)],
        "post_war": frame[frame["date"] >= pd.Timestamp(WAR_SPLIT_DATE)],
    }
    metrics = {}
    for name, subset in periods.items():
        if subset.empty:
            metrics[name] = {"accuracy": math.nan, "balanced_accuracy": math.nan, "auc": math.nan, "n": 0}
        else:
            metrics[name] = directional_metrics(subset["target"], subset["probability"])
    return metrics


def evaluate_by_ticker(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    metrics = {}
    for ticker, subset in predictions.groupby("ticker"):
        metrics[str(ticker)] = directional_metrics(subset["target"], subset["probability"])
    return metrics


def evaluate_regression_by_period(predictions: pd.DataFrame, target_col: str = "actual", pred_col: str = "prediction") -> dict[str, dict[str, float]]:
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    periods = {
        "overall": frame,
        "pre_war": frame[frame["date"] < pd.Timestamp(WAR_SPLIT_DATE)],
        "post_war": frame[frame["date"] >= pd.Timestamp(WAR_SPLIT_DATE)],
    }
    metrics = {}
    for name, subset in periods.items():
        metrics[name] = regression_metrics(subset[target_col], subset[pred_col])
    return metrics


def evaluate_regression_supply_uncertainty(
    predictions: pd.DataFrame,
    regimes: pd.DataFrame,
    target_col: str = "actual",
    pred_col: str = "prediction",
) -> dict[str, dict[str, float]]:
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    merged = frame.merge(regimes, on="date", how="left")
    outputs = {}
    for name, mask in {
        "iran_war_window": merged["iran_war_window"] == 1,
        "supply_uncertainty": merged["supply_uncertainty"] == 1,
        "normal_regime": merged["normal_regime"] == 1,
    }.items():
        subset = merged.loc[mask]
        outputs[name] = regression_metrics(subset[target_col], subset[pred_col])
    return outputs


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


def evaluate_supply_uncertainty(predictions: pd.DataFrame, regimes: pd.DataFrame) -> dict[str, dict[str, float]]:
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    merged = frame.merge(regimes, on="date", how="left")
    outputs = {}
    for name, mask in {
        "iran_war_window": merged["iran_war_window"] == 1,
        "supply_uncertainty": merged["supply_uncertainty"] == 1,
        "normal_regime": merged["normal_regime"] == 1,
    }.items():
        subset = merged.loc[mask]
        outputs[name] = directional_metrics(subset["target"], subset["probability"]) if not subset.empty else {
            "accuracy": math.nan,
            "balanced_accuracy": math.nan,
            "auc": math.nan,
            "n": 0,
        }
    return outputs


def bootstrap_accuracy_lift(
    left: pd.DataFrame,
    right: pd.DataFrame,
    n_bootstrap: int = 1000,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    merged = left.merge(
        right,
        on=["date", "ticker", "target"],
        suffixes=("_left", "_right"),
        how="inner",
    )
    if merged.empty:
        return {"lift": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": 0}

    left_correct = (merged["probability_left"] >= 0.5).astype(int) == merged["target"].astype(int)
    right_correct = (merged["probability_right"] >= 0.5).astype(int) == merged["target"].astype(int)
    point_lift = float(left_correct.mean() - right_correct.mean())

    rng = np.random.default_rng(seed)
    lifts = []
    idx = np.arange(len(merged))
    for _ in range(n_bootstrap):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        lifts.append(float(left_correct.iloc[sample_idx].mean() - right_correct.iloc[sample_idx].mean()))
    return {
        "lift": point_lift,
        "ci_low": float(np.quantile(lifts, 0.025)),
        "ci_high": float(np.quantile(lifts, 0.975)),
        "n": int(len(merged)),
    }


def bootstrap_accuracy_lift_by_regime(
    left: pd.DataFrame,
    right: pd.DataFrame,
    regimes: pd.DataFrame,
    n_bootstrap: int = 1000,
) -> dict[str, dict[str, float]]:
    outputs = {}
    for name, column in {
        "overall": None,
        "iran_war_window": "iran_war_window",
        "supply_uncertainty": "supply_uncertainty",
        "normal_regime": "normal_regime",
    }.items():
        if column is None:
            left_subset = left
            right_subset = right
        else:
            regime_dates = set(regimes.loc[regimes[column] == 1, "date"])
            left_subset = left[left["date"].isin(regime_dates)]
            right_subset = right[right["date"].isin(regime_dates)]
        outputs[name] = bootstrap_accuracy_lift(left_subset, right_subset, n_bootstrap=n_bootstrap)
    return outputs


def bootstrap_difference_in_lifts(
    left: pd.DataFrame,
    right: pd.DataFrame,
    regimes: pd.DataFrame,
    regime_a: str,
    regime_b: str,
    n_bootstrap: int = 1000,
    seed: int = RANDOM_SEED,
) -> dict[str, float]:
    merged = left.merge(
        right,
        on=["date", "ticker", "target"],
        suffixes=("_left", "_right"),
        how="inner",
    ).merge(regimes[["date", regime_a, regime_b]].drop_duplicates(), on="date", how="left")
    merged = merged[(merged[regime_a] == 1) | (merged[regime_b] == 1)].copy()
    if merged.empty:
        return {"difference": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": 0}

    left_correct = ((merged["probability_left"] >= 0.5).astype(int) == merged["target"].astype(int)).astype(int)
    right_correct = ((merged["probability_right"] >= 0.5).astype(int) == merged["target"].astype(int)).astype(int)
    merged["lift"] = left_correct - right_correct

    group_a = merged[merged[regime_a] == 1]
    group_b = merged[merged[regime_b] == 1]
    if group_a.empty or group_b.empty:
        return {"difference": math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": int(len(merged))}

    point = float(group_a["lift"].mean() - group_b["lift"].mean())
    rng = np.random.default_rng(seed)
    draws = []
    idx_a = np.arange(len(group_a))
    idx_b = np.arange(len(group_b))
    for _ in range(n_bootstrap):
        sample_a = group_a.iloc[rng.choice(idx_a, size=len(group_a), replace=True)]["lift"].mean()
        sample_b = group_b.iloc[rng.choice(idx_b, size=len(group_b), replace=True)]["lift"].mean()
        draws.append(float(sample_a - sample_b))
    return {
        "difference": point,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n": int(len(merged)),
    }


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
