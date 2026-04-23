from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from torch.utils.data import DataLoader, Dataset
import yfinance as yf

from oil_projects.common import (
    build_excess_return_target,
    build_supply_uncertainty_regimes,
    choose_classification_threshold,
    compute_macd,
    compute_rsi,
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
    train_valid_test_label,
)
from oil_projects.config import (
    ARTIFACT_DIR,
    BENCHMARK_TICKER,
    FACTOR_TICKERS,
    OIL_TICKERS,
    PROCESSED_DIR,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
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


class SequenceDataset(Dataset):
    def __init__(self, sequences: np.ndarray, ticker_ids: np.ndarray, targets: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.ticker_ids = torch.tensor(ticker_ids, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.ticker_ids[idx], self.targets[idx]


class QuantGRU(nn.Module):
    def __init__(self, input_size: int, ticker_count: int):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.embedding = nn.Embedding(ticker_count, 8)
        self.head = nn.Sequential(
            nn.Linear(64 + 8, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, ticker_ids: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        seq_repr = hidden[-1]
        ticker_repr = self.embedding(ticker_ids)
        logits = self.head(torch.cat([seq_repr, ticker_repr], dim=1))
        return logits.squeeze(1)


@dataclass
class SequenceBundle:
    features: np.ndarray
    tabular_features: np.ndarray
    ticker_ids: np.ndarray
    targets: np.ndarray
    dates: np.ndarray
    tickers: np.ndarray


def _make_sequences(panel: pd.DataFrame) -> tuple[SequenceBundle, list[str], dict[str, int]]:
    feature_columns = [
        col
        for col in panel.columns
        if col not in {"date", "ticker", "close", "target", "split", "asset_ret_fwd", "benchmark_ret_fwd", "excess_ret_fwd"}
    ]
    ticker_to_id = {ticker: idx for idx, ticker in enumerate(sorted(panel["ticker"].unique()))}
    sequences = []
    tabular_features = []
    ticker_ids = []
    targets = []
    dates = []
    tickers = []

    for ticker, group in panel.groupby("ticker", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        values = group[feature_columns].to_numpy(dtype=float)
        for end_idx in range(SEQUENCE_LENGTH - 1, len(group)):
            window = values[end_idx - SEQUENCE_LENGTH + 1 : end_idx + 1]
            sequences.append(window)
            tabular_features.append(values[end_idx])
            ticker_ids.append(ticker_to_id[ticker])
            targets.append(int(group.loc[end_idx, "target"]))
            dates.append(pd.Timestamp(group.loc[end_idx, "date"]))
            tickers.append(ticker)

    bundle = SequenceBundle(
        features=np.asarray(sequences, dtype=np.float32),
        tabular_features=np.asarray(tabular_features, dtype=np.float32),
        ticker_ids=np.asarray(ticker_ids, dtype=np.int64),
        targets=np.asarray(targets, dtype=np.float32),
        dates=np.asarray(dates),
        tickers=np.asarray(tickers),
    )
    return bundle, feature_columns, ticker_to_id


def _fit_scaler(features: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(features.reshape(-1, features.shape[-1]))
    return scaler


def _transform_sequences(features: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    flat = scaler.transform(features.reshape(-1, features.shape[-1]))
    return flat.reshape(features.shape)


def _train_model(
    model: QuantGRU,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    epochs: int = 30,
) -> QuantGRU:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    best_state = None
    best_valid_loss = float("inf")
    patience = 0

    model.to(device)
    for _ in range(epochs):
        model.train()
        for x_batch, ticker_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            ticker_batch = ticker_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch, ticker_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for x_batch, ticker_batch, y_batch in valid_loader:
                x_batch = x_batch.to(device)
                ticker_batch = ticker_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch, ticker_batch)
                valid_losses.append(float(criterion(logits, y_batch).item()))
        avg_valid_loss = float(np.mean(valid_losses))
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict(model: QuantGRU, dataset: SequenceDataset, device: torch.device) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    model.eval()
    probabilities = []
    with torch.no_grad():
        for x_batch, ticker_batch, _ in loader:
            logits = model(x_batch.to(device), ticker_batch.to(device))
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.asarray(probabilities, dtype=float)


def _window_masks(dates: np.ndarray, window_start: pd.Timestamp, window_end: pd.Timestamp) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts = pd.to_datetime(dates)
    train_mask = np.asarray(ts < window_start)
    valid_mask = np.asarray((ts >= (window_start - pd.DateOffset(months=6))) & (ts < window_start))
    valid_mask &= train_mask
    test_mask = np.asarray((ts >= window_start) & (ts <= window_end))
    return train_mask, valid_mask, test_mask


def run_quant_project() -> dict:
    ensure_directories()
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    panel = _build_feature_panel()
    panel.to_parquet(PROCESSED_DIR / "quant_panel.parquet", index=False)
    regime_frame = build_supply_uncertainty_regimes()

    bundle, feature_columns, ticker_to_id = _make_sequences(panel)
    split_labels = np.array([train_valid_test_label(pd.Timestamp(date)) for date in bundle.dates], dtype=object)
    test_dates = pd.to_datetime(bundle.dates[split_labels == "test"])
    windows = monthly_test_windows(test_dates)
    predictions_list = []
    window_config_rows = []

    for window_start, window_end in windows:
        print(f"Quant walk-forward window {window_start.date()} -> {window_end.date()}")
        train_mask, valid_mask, test_mask = _window_masks(bundle.dates, window_start, window_end)
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        if valid_mask.sum() == 0:
            valid_mask = train_mask.copy()
        fit_mask = train_mask & ~valid_mask
        if fit_mask.sum() == 0:
            fit_mask = train_mask.copy()

        scaler = _fit_scaler(bundle.features[fit_mask])
        scaled_features = _transform_sequences(bundle.features, scaler)
        scaled_tabular = scaler.transform(bundle.tabular_features)

        train_dataset = SequenceDataset(scaled_features[fit_mask], bundle.ticker_ids[fit_mask], bundle.targets[fit_mask])
        valid_dataset = SequenceDataset(scaled_features[valid_mask], bundle.ticker_ids[valid_mask], bundle.targets[valid_mask])
        test_dataset = SequenceDataset(scaled_features[test_mask], bundle.ticker_ids[test_mask], bundle.targets[test_mask])

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)

        model = QuantGRU(input_size=len(feature_columns), ticker_count=len(ticker_to_id))
        model = _train_model(model, train_loader, valid_loader, device, epochs=8)
        model.to(device)

        deep_valid_probabilities = _predict(model, valid_dataset, device)
        deep_test_probabilities = _predict(model, test_dataset, device)
        xgb_model = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
        )
        sample_weight = 1.0 + 2.0 * bundle.tabular_features[fit_mask][:, feature_columns.index("supply_uncertainty")]
        xgb_model.fit(
            scaled_tabular[fit_mask],
            bundle.targets[fit_mask].astype(int),
            sample_weight=sample_weight,
        )
        xgb_valid_probabilities = xgb_model.predict_proba(scaled_tabular[valid_mask])[:, 1]
        xgb_test_probabilities = xgb_model.predict_proba(scaled_tabular[test_mask])[:, 1]
        valid_targets = bundle.targets[valid_mask].astype(int)
        best_weight = 1.0
        best_threshold = 0.5
        best_validation_accuracy = float("nan")
        if valid_targets.size > 0 and np.unique(valid_targets).size > 1:
            best_validation_accuracy = -1.0
            for ensemble_weight in np.linspace(0.0, 1.0, 11):
                ensemble_valid = ensemble_weight * deep_valid_probabilities + (1.0 - ensemble_weight) * xgb_valid_probabilities
                threshold = choose_classification_threshold(valid_targets, ensemble_valid, thresholds=np.linspace(0.3, 0.7, 41))
                validation_accuracy = float(np.mean((ensemble_valid >= threshold).astype(int) == valid_targets))
                if validation_accuracy > best_validation_accuracy:
                    best_validation_accuracy = validation_accuracy
                    best_weight = float(ensemble_weight)
                    best_threshold = float(threshold)
        ensemble_test_probabilities = best_weight * deep_test_probabilities + (1.0 - best_weight) * xgb_test_probabilities
        calibrated_probabilities = shift_probabilities_for_threshold(ensemble_test_probabilities, best_threshold)
        predictions_list.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(bundle.dates[test_mask]),
                    "ticker": bundle.tickers[test_mask],
                    "target": bundle.targets[test_mask].astype(int),
                    "probability": calibrated_probabilities,
                    "probability_ensemble": ensemble_test_probabilities,
                    "probability_deep": deep_test_probabilities,
                    "probability_xgb": xgb_test_probabilities,
                    "ensemble_weight_deep": best_weight,
                    "ensemble_weight_xgb": 1.0 - best_weight,
                    "calibration_threshold": best_threshold,
                    "validation_accuracy": best_validation_accuracy,
                }
            )
        )
        window_config_rows.append(
            {
                "window_start": window_start,
                "window_end": window_end,
                "train_rows": int(train_mask.sum()),
                "fit_rows": int(fit_mask.sum()),
                "validation_rows": int(valid_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "ensemble_weight_deep": best_weight,
                "ensemble_weight_xgb": 1.0 - best_weight,
                "calibration_threshold": best_threshold,
                "validation_accuracy": best_validation_accuracy,
                "train_max_date": pd.to_datetime(bundle.dates[train_mask]).max(),
                "validation_start_date": pd.to_datetime(bundle.dates[valid_mask]).min(),
                "validation_end_date": pd.to_datetime(bundle.dates[valid_mask]).max(),
            }
        )

    predictions = pd.concat(predictions_list, ignore_index=True).sort_values(["date", "ticker"])
    predictions = predictions.merge(
        regime_frame[["date", "iran_war_window", "supply_uncertainty", "normal_regime"]],
        on="date",
        how="left",
    )
    window_configs = pd.DataFrame(window_config_rows).sort_values("window_start")

    metrics = evaluate_by_period(predictions)
    evaluation_predictions = predictions.drop(columns=["iran_war_window", "supply_uncertainty", "normal_regime"])
    regime_metrics = evaluate_supply_uncertainty(evaluation_predictions, regime_frame)
    ticker_metrics = evaluate_by_ticker(predictions)

    regression_feature_columns = [
        col
        for col in panel.columns
        if col not in {"date", "ticker", "close", "target", "split", "asset_ret_fwd", "benchmark_ret_fwd", "excess_ret_fwd"}
    ]
    regression_predictions_list = []
    for ticker, ticker_frame in panel.groupby("ticker"):
        ticker_frame = ticker_frame.sort_values("date").reset_index(drop=True)
        ticker_test_dates = ticker_frame.loc[pd.to_datetime(ticker_frame["date"]) >= pd.Timestamp("2025-01-01"), "date"]
        for window_start, window_end in monthly_test_windows(ticker_test_dates):
            train_mask = pd.to_datetime(ticker_frame["date"]) < window_start
            test_mask = (pd.to_datetime(ticker_frame["date"]) >= window_start) & (pd.to_datetime(ticker_frame["date"]) <= window_end)
            if train_mask.sum() < 80 or test_mask.sum() == 0:
                continue
            model = XGBRegressor(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.9,
                colsample_bytree=0.8,
                random_state=RANDOM_SEED,
            )
            x_train = ticker_frame.loc[train_mask, regression_feature_columns].fillna(0.0)
            y_train = ticker_frame.loc[train_mask, "excess_ret_fwd"]
            x_test = ticker_frame.loc[test_mask, regression_feature_columns].fillna(0.0)
            weights = 1.0 + 2.0 * ticker_frame.loc[train_mask, "supply_uncertainty"].fillna(0.0) + 2.0 * ticker_frame.loc[train_mask, "iran_war_window"].fillna(0.0)
            model.fit(x_train, y_train, sample_weight=weights)
            regression_predictions_list.append(
                pd.DataFrame(
                    {
                        "date": ticker_frame.loc[test_mask, "date"].values,
                        "ticker": ticker,
                        "actual": ticker_frame.loc[test_mask, "excess_ret_fwd"].values,
                        "prediction": model.predict(x_test),
                    }
                )
            )

    regression_predictions = pd.concat(regression_predictions_list, ignore_index=True).sort_values(["date", "ticker"])
    regression_metrics = evaluate_regression_by_period(regression_predictions)
    regression_regime_metrics = evaluate_regression_supply_uncertainty(regression_predictions, regime_frame)
    regression_ticker_metrics = {
        str(ticker): evaluate_regression_by_period(group.assign(date=pd.to_datetime(group["date"])))["overall"]
        for ticker, group in regression_predictions.groupby("ticker")
    }
    predictions.to_csv(ARTIFACT_DIR / "quant_predictions.csv", index=False)
    window_configs.to_csv(ARTIFACT_DIR / "quant_window_choices.csv", index=False)
    regression_predictions.to_csv(ARTIFACT_DIR / "quant_regression_predictions.csv", index=False)
    regime_frame.to_csv(ARTIFACT_DIR / "supply_uncertainty_regimes.csv", index=False)
    window_choice_records = window_configs.assign(
        window_start=window_configs["window_start"].astype(str),
        window_end=window_configs["window_end"].astype(str),
        train_max_date=window_configs["train_max_date"].astype(str),
        validation_start_date=window_configs["validation_start_date"].astype(str),
        validation_end_date=window_configs["validation_end_date"].astype(str),
    ).to_dict(orient="records")
    save_json(
        ARTIFACT_DIR / "quant_metrics.json",
        {
            "project_name": "quant_deep_ensemble",
            "features": feature_columns,
            "target": "next-day excess return vs XLE with 25bp threshold",
            "walk_forward": "monthly expanding-window retraining",
            "model_family": "GRU + XGBoost ensemble with validation-selected weights and thresholds",
            "metrics": metrics,
            "regime_metrics": regime_metrics,
            "ticker_metrics": ticker_metrics,
            "window_choices": window_choice_records,
            "regression_target": "next-day excess return vs XLE",
            "regression_metrics": regression_metrics,
            "regression_regime_metrics": regression_regime_metrics,
            "regression_ticker_metrics": regression_ticker_metrics,
        },
    )

    print("\nQuant project")
    print("=" * 80)
    for period_name, period_metrics in metrics.items():
        print(
            f"{period_name:<10} accuracy={period_metrics['accuracy']:.4f} "
            f"balanced_accuracy={period_metrics['balanced_accuracy']:.4f} "
            f"auc={period_metrics['auc']:.4f} n={period_metrics['n']}"
        )
    print("Supply-uncertainty focus")
    for period_name, period_metrics in regime_metrics.items():
        print(
            f"{period_name:<18} accuracy={period_metrics['accuracy']:.4f} "
            f"balanced_accuracy={period_metrics['balanced_accuracy']:.4f} "
            f"auc={period_metrics['auc']:.4f} n={period_metrics['n']}"
        )
    print("Regression focus")
    for period_name, period_metrics in regression_regime_metrics.items():
        print(
            f"{period_name:<18} rmse={period_metrics['rmse']:.4f} "
            f"mae={period_metrics['mae']:.4f} "
            f"sign_accuracy={period_metrics['sign_accuracy']:.4f} "
            f"within_50bp={period_metrics['within_tolerance_accuracy']:.4f} n={period_metrics['n']}"
        )

    return {
        "project_name": "quant_deep_ensemble",
        "metrics": metrics,
        "regime_metrics": regime_metrics,
        "regression_metrics": regression_metrics,
        "regression_regime_metrics": regression_regime_metrics,
    }
