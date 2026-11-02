from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from oil_projects.common import (
    build_excess_return_target,
    build_supply_uncertainty_regimes,
    compute_macd,
    compute_rsi,
    train_valid_test_label,
)
from oil_projects.config import (
    BENCHMARK_TICKER,
    FACTOR_TICKERS,
    OIL_TICKERS,
    START_DATE,
)
from oil_projects.official_sources import build_official_event_features


def _download_history(tickers: list[str]) -> dict[str, pd.DataFrame]:
    data = {}
    for ticker in tickers:
        frame = yf.download(ticker, start=START_DATE, progress=False, auto_adjust=True)
        if frame.empty:
            raise RuntimeError(f"No price history returned for {ticker}")
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame.columns = [str(column).lower() for column in frame.columns]
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        data[ticker] = frame.sort_index()
    return data


def _build_feature_panel() -> pd.DataFrame:
    price_map = _download_history(list(OIL_TICKERS))
    factor_map = _download_history(list(FACTOR_TICKERS))
    regime_frame = build_supply_uncertainty_regimes().set_index("date")
    official_events = build_official_event_features().set_index("date")
    benchmark_close = factor_map[BENCHMARK_TICKER]["close"]
    factor_returns = []
    for ticker, factor_df in factor_map.items():
        subset = pd.DataFrame(index=factor_df.index)
        subset[f"{ticker}_ret_1"] = factor_df["close"].pct_change(fill_method=None)
        subset[f"{ticker}_ret_5"] = factor_df["close"].pct_change(5, fill_method=None)
        subset[f"{ticker}_vol_10"] = subset[f"{ticker}_ret_1"].rolling(10).std()
        factor_returns.append(subset)
    merged_factors = pd.concat(factor_returns, axis=1).sort_index()
    benchmark_ret_1 = merged_factors[f"{BENCHMARK_TICKER}_ret_1"]
    oil_ret_1 = merged_factors["CL=F_ret_1"]

    frames = []
    for ticker, price_df in price_map.items():
        df = pd.DataFrame(index=price_df.index)
        df["ticker"] = ticker
        df["close"] = price_df["close"]
        df["volume"] = price_df["volume"]
        df["ret_1"] = price_df["close"].pct_change(fill_method=None)
        df["ret_2"] = price_df["close"].pct_change(2, fill_method=None)
        df["ret_3"] = price_df["close"].pct_change(3, fill_method=None)
        df["ret_5"] = price_df["close"].pct_change(5, fill_method=None)
        df["ret_10"] = price_df["close"].pct_change(10, fill_method=None)
        df["ret_20"] = price_df["close"].pct_change(20, fill_method=None)
        df["vol_5"] = df["ret_1"].rolling(5).std()
        df["vol_10"] = df["ret_1"].rolling(10).std()
        df["vol_20"] = df["ret_1"].rolling(20).std()
        df["range_1"] = (price_df["high"] - price_df["low"]) / price_df["close"]
        df["volume_z_20"] = (np.log1p(price_df["volume"]) - np.log1p(price_df["volume"]).rolling(20).mean()) / (
            np.log1p(price_df["volume"]).rolling(20).std()
        )
        df["sma_5_gap"] = price_df["close"] / price_df["close"].rolling(5).mean() - 1
        df["sma_20_gap"] = price_df["close"] / price_df["close"].rolling(20).mean() - 1
        df["rsi_14"] = compute_rsi(price_df["close"], 14) / 100.0
        df["macd_diff"] = compute_macd(price_df["close"])
        df["beta_xle_20"] = df["ret_1"].rolling(20).cov(benchmark_ret_1) / benchmark_ret_1.rolling(20).var()
        df["beta_oil_20"] = df["ret_1"].rolling(20).cov(oil_ret_1) / oil_ret_1.rolling(20).var()
        df["residual_xle_1"] = df["ret_1"] - benchmark_ret_1
        df["residual_oil_1"] = df["ret_1"] - oil_ret_1
        df["corr_xle_20"] = df["ret_1"].rolling(20).corr(benchmark_ret_1)
        df["corr_oil_20"] = df["ret_1"].rolling(20).corr(oil_ret_1)
        df["shock_beta_spread"] = df["beta_oil_20"] - df["beta_xle_20"]
        df["shock_corr_spread"] = df["corr_oil_20"] - df["corr_xle_20"]
        df["momentum_reversal_5_1"] = df["ret_5"] - df["ret_1"]
        df["vol_regime_ratio"] = df["vol_5"] / df["vol_20"].replace(0, np.nan)
        df["volume_shock_interaction"] = df["volume_z_20"] * df["range_1"]
        target_frame = build_excess_return_target(price_df["close"], benchmark_close)
        df["asset_ret_fwd"] = target_frame["asset_ret_fwd"]
        df["benchmark_ret_fwd"] = target_frame["benchmark_ret_fwd"]
        df["excess_ret_fwd"] = target_frame["excess_ret_fwd"]
        df["target"] = target_frame["target"]
        df = df.join(merged_factors, how="left")
        df = df.join(regime_frame, how="left")
        df = df.join(official_events, how="left")
        frames.append(df)

    panel = pd.concat(frames).sort_index()
    fill_columns = [
        "iran_war_window",
        "supply_uncertainty",
        "oil_shock_day",
        "market_stress_day",
        "normal_regime",
        "cl_ret",
        "bz_ret",
        "vix_ret",
        "xle_ret",
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
    ]
    for column in fill_columns:
        if column in panel.columns:
            panel[column] = panel[column].fillna(0.0)
    panel["split"] = [train_valid_test_label(date) for date in panel.index]
    panel = panel[panel["split"] != "drop"]
    panel = panel.dropna().reset_index()
    first_column = panel.columns[0]
    panel = panel.rename(columns={first_column: "date"})
    return panel
