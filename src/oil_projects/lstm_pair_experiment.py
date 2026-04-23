from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import yfinance as yf

from oil_projects.common import (
    choose_classification_threshold,
    ensure_directories,
    monthly_test_windows,
    save_json,
    set_seed,
    shift_probabilities_for_threshold,
)
from oil_projects.config import ARTIFACT_DIR, BASE_DIR, BENCHMARK_TICKER, RANDOM_SEED, START_DATE, WAR_SPLIT_DATE
from oil_projects.statement_sources import build_statement_event_features


TARGET_HORIZON_DAYS = 7
TARGET_THRESHOLD = 0.01
DEFAULT_SEQUENCE_LENGTH = 20
MIN_TRAIN_ROWS = 260
MIN_VALID_ROWS = 60
MIN_TEST_ROWS = 20

PRE_WAR_START = pd.Timestamp(WAR_SPLIT_DATE) - pd.Timedelta(days=60)
PRE_WAR_END = pd.Timestamp(WAR_SPLIT_DATE) - pd.Timedelta(days=1)
WAR_START = pd.Timestamp(WAR_SPLIT_DATE)
WAR_END = pd.Timestamp(WAR_SPLIT_DATE) + pd.Timedelta(days=59)


QUANT_FEATURES = [
    "ret_1",
    "ret_2",
    "ret_3",
    "ret_5",
    "ret_10",
    "vol_5",
    "vol_10",
    "range_1",
    "volume_z_20",
    "sma_5_gap",
    "sma_20_gap",
    "rsi_14",
    "macd_diff",
    "beta_xle_20",
    "beta_oil_20",
    "residual_xle_1",
    "residual_oil_1",
    "corr_xle_20",
    "corr_oil_20",
    "shock_beta_spread",
    "shock_corr_spread",
    "CL=F_ret_1",
    "CL=F_ret_5",
    "BZ=F_ret_1",
    "BZ=F_ret_5",
    "SPY_ret_1",
    "SPY_ret_5",
    "^VIX_ret_1",
    "^VIX_ret_5",
    "UUP_ret_1",
    "UUP_ret_5",
    "cl_ret",
    "bz_ret",
    "vix_ret",
    "xle_ret",
    "official_event_weight",
    "official_conflict_escalation_weight",
    "official_sanctions_oil_weight",
    "official_shipping_enforcement_weight",
]

BASELINE_NEWS_FEATURES = [
    "filing_count",
    "avg_text_length",
    "novelty_mean",
    "filing_8k_share",
    "filing_periodic_share",
    "filing_count_5d",
    "prev_ret_1",
    "prev_ret_5",
    "official_event_weight",
    "official_conflict_escalation_weight",
    "official_sanctions_oil_weight",
    "official_shipping_enforcement_weight",
    "gpr_level",
    "gpr_threat",
    "gpr_surprise",
    "gpr_threat_share",
    "statement_event_weight",
    "statement_event_weight_3d",
    "statement_shipping_weight_3d",
    "statement_hormuz_weight_3d",
    "statement_strike_weight_3d",
    "statement_sanctions_weight_3d",
    "sector_fill_used",
]

SENTIMENT_EXTRA_FEATURES = [
    "avg_sentiment",
    "sentiment_sum",
    "positive_share",
    "negative_share",
    "geopolitics_share",
    "shipping_share",
    "sanctions_share",
    "supply_disruption_share",
    "geopolitics_sentiment",
    "supply_risk_intensity",
    "supply_chain_intensity",
    "agent_political_risk",
    "agent_company_exposure",
    "agent_market_stress",
    "agent_disclosure_signal",
    "agent_war_alignment",
    "agent_uncertainty_confirmation",
    "statement_shipping_alignment",
    "statement_hormuz_alignment",
    "statement_strike_alignment",
    "statement_sanctions_alignment",
    "statement_market_confirmation",
]

ENRICHED_SENTIMENT_EXTRA_FEATURES = [
    "gdelt_war_volume",
    "gdelt_shipping_volume",
    "gdelt_opec_volume",
    "gdelt_supply_volume",
    "gdelt_total_volume",
    "gdelt_war_volume_5d",
    "gdelt_shipping_volume_5d",
    "gdelt_opec_volume_5d",
    "gdelt_supply_volume_5d",
    "gdelt_total_volume_5d",
    "gdelt_supply_threat_intensity",
    "gdelt_supply_confirmation",
    "official_event_count",
    "official_event_weight_5d",
    "official_conflict_escalation_weight_5d",
    "official_opec_supply_policy_weight",
    "official_opec_supply_policy_weight_5d",
    "official_sanctions_oil_weight_5d",
    "official_shipping_enforcement_weight_5d",
    "official_policy_confirmation",
    "official_supply_confirmation",
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
    "agent_supply_alignment",
    "agent_text_market_confirmation",
]


def _download_history(tickers: list[str]) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        frame = yf.download(ticker, start=START_DATE, progress=False, auto_adjust=True)
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame.columns = [str(column).lower() for column in frame.columns]
        frame.index = pd.to_datetime(frame.index).tz_localize(None)
        output[ticker] = frame.sort_index()
    return output


def _build_target_frame(tickers: list[str], target_horizon_days: int, target_threshold: float) -> pd.DataFrame:
    price_map = _download_history(tickers + [BENCHMARK_TICKER])
    benchmark_close = price_map[BENCHMARK_TICKER]["close"]
    rows = []
    for ticker in tickers:
        prices = price_map[ticker]["close"]
        benchmark_aligned = benchmark_close.reindex(prices.index).ffill()
        excess_return = (prices.shift(-target_horizon_days) / prices - 1.0) - (
            benchmark_aligned.shift(-target_horizon_days) / benchmark_aligned - 1.0
        )
        rows.append(
            pd.DataFrame(
                {
                    "date": prices.index,
                    "ticker": ticker,
                    "target": np.where(excess_return > target_threshold, 1, np.where(excess_return < -target_threshold, 0, np.nan)),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _build_panel(config: LSTMExperimentConfig) -> pd.DataFrame:
    news = pd.read_parquet(BASE_DIR / "data/processed/news_panel.parquet")
    quant = pd.read_parquet(BASE_DIR / "data/processed/quant_panel.parquet")
    news["date"] = pd.to_datetime(news["date"])
    quant["date"] = pd.to_datetime(quant["date"])
    news = news.drop(columns=[column for column in news.columns if column == "target"], errors="ignore")
    quant = quant.drop(columns=[column for column in quant.columns if column == "target"], errors="ignore")

    news_tickers = sorted(set(news["ticker"]))
    quant_tickers = sorted(set(quant["ticker"]))
    target_tickers = quant_tickers if config.expand_to_quant_tickers else sorted(set(news_tickers).intersection(set(quant_tickers)))
    news = news[news["ticker"].isin(set(target_tickers).intersection(set(news_tickers)))].copy()
    quant = quant[quant["ticker"].isin(target_tickers)].copy()

    statement_features = build_statement_event_features()
    statement_features["date"] = pd.to_datetime(statement_features["date"])
    news = news.merge(statement_features, on="date", how="left")
    statement_columns = [column for column in statement_features.columns if column != "date"]
    for column in statement_columns:
        news[column] = news[column].fillna(0.0)

    news["statement_shipping_alignment"] = news["statement_shipping_weight_3d"] * (1.0 + news["shipping_share"].fillna(0.0))
    news["statement_hormuz_alignment"] = news["statement_hormuz_weight_3d"] * (1.0 + news["agent_war_alignment"].fillna(0.0))
    news["statement_strike_alignment"] = news["statement_strike_weight_3d"] * (1.0 + news["geopolitics_share"].fillna(0.0))
    news["statement_sanctions_alignment"] = news["statement_sanctions_weight_3d"] * (1.0 + news["sanctions_share"].fillna(0.0))
    news["statement_market_confirmation"] = news["statement_event_weight_3d"] * (1.0 + news["ovx_ret_1"].fillna(0.0).clip(lower=0.0))

    if config.expand_to_quant_tickers:
        numeric_news_columns = [
            column
            for column in news.columns
            if column not in {"date", "ticker"} and pd.api.types.is_numeric_dtype(news[column])
        ]
        sector_by_date = news.groupby("date")[numeric_news_columns].mean().reset_index()
        model_keys = quant[["date", "ticker"]].drop_duplicates()
        news = model_keys.merge(news, on=["date", "ticker"], how="left")
        missing_native = news["filing_count"].isna() if "filing_count" in news.columns else news[numeric_news_columns[0]].isna()
        news = news.merge(
            sector_by_date.rename(columns={column: f"{column}_sector" for column in numeric_news_columns}),
            on="date",
            how="left",
        )
        for column in numeric_news_columns:
            news[column] = news[column].fillna(news[f"{column}_sector"])
        news["sector_fill_used"] = missing_native.astype(float)
        news = news.drop(columns=[f"{column}_sector" for column in numeric_news_columns])
    else:
        news["sector_fill_used"] = 0.0

    target_frame = _build_target_frame(
        target_tickers,
        target_horizon_days=config.target_horizon_days,
        target_threshold=config.target_threshold,
    )
    model_frame = news.merge(
        quant[["date", "ticker"] + QUANT_FEATURES],
        on=["date", "ticker"],
        how="left",
    ).merge(target_frame, on=["date", "ticker"], how="left")

    feature_columns = sorted(set(QUANT_FEATURES + BASELINE_NEWS_FEATURES + SENTIMENT_EXTRA_FEATURES))
    for column in feature_columns:
        if column not in model_frame.columns:
            model_frame[column] = 0.0
        model_frame[column] = model_frame[column].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    salience_columns = [
        "statement_event_weight_3d",
        "statement_shipping_weight_3d",
        "statement_hormuz_weight_3d",
        "statement_strike_weight_3d",
        "statement_sanctions_weight_3d",
        "official_event_weight",
        "official_conflict_escalation_weight",
        "official_sanctions_oil_weight",
        "official_shipping_enforcement_weight",
        "gpr_threat",
        "gpr_surprise",
        "geopolitics_share",
        "shipping_share",
        "sanctions_share",
        "supply_disruption_share",
    ]
    salience_scaled = []
    for column in salience_columns:
        if column not in model_frame.columns:
            continue
        series = model_frame[column].replace([np.inf, -np.inf], np.nan).fillna(0.0).abs()
        scale = float(series.quantile(0.95))
        scale = scale if scale > 1e-6 else 1.0
        salience_scaled.append((series / scale).clip(0.0, 1.5))
    if salience_scaled:
        salience_signal = pd.concat(salience_scaled, axis=1).max(axis=1)
        model_frame["salience_weight"] = 1.0 + config.salience_alpha * salience_signal
    else:
        model_frame["salience_weight"] = 1.0

    model_frame = model_frame.dropna(subset=["target"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    return model_frame


@dataclass
class SequenceBundle:
    sequences: np.ndarray
    static_features: np.ndarray
    targets: np.ndarray
    sample_weights: np.ndarray
    ticker_ids: np.ndarray
    dates: np.ndarray
    tickers: np.ndarray


@dataclass(frozen=True)
class LSTMExperimentConfig:
    target_horizon_days: int = TARGET_HORIZON_DAYS
    target_threshold: float = TARGET_THRESHOLD
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH
    use_enriched_sentiment_features: bool = False
    hidden_size: int = 64
    num_layers: int = 2
    lstm_dropout: float = 0.2
    static_hidden_size: int = 24
    static_dropout: float = 0.1
    head_hidden_size: int = 32
    head_dropout: float = 0.2
    learning_rate: float = 1e-3
    epochs: int = 14
    batch_size: int = 64
    bidirectional: bool = False
    use_attention_pooling: bool = False
    use_static_gate: bool = False
    expand_to_quant_tickers: bool = False
    use_salience_weights: bool = False
    salience_alpha: float = 1.0


class SequenceDataset(Dataset):
    def __init__(
        self,
        sequences: np.ndarray,
        static_features: np.ndarray,
        ticker_ids: np.ndarray,
        targets: np.ndarray,
        sample_weights: np.ndarray,
    ):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.static_features = torch.tensor(static_features, dtype=torch.float32)
        self.ticker_ids = torch.tensor(ticker_ids, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.static_features[idx], self.ticker_ids[idx], self.targets[idx], self.sample_weights[idx]


class PairLSTM(nn.Module):
    def __init__(self, input_size: int, ticker_count: int, static_input_size: int, config: LSTMExperimentConfig):
        super().__init__()
        self.bidirectional = config.bidirectional
        self.use_attention_pooling = config.use_attention_pooling
        self.use_static_gate = config.use_static_gate
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.lstm_dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
        )
        self.embedding = nn.Embedding(ticker_count, 8)
        static_layers: list[nn.Module] = []
        static_hidden_size = 0
        if static_input_size > 0:
            static_layers = [
                nn.Linear(static_input_size, config.static_hidden_size),
                nn.ReLU(),
                nn.Dropout(config.static_dropout),
            ]
            static_hidden_size = config.static_hidden_size
        self.static_head = nn.Sequential(*static_layers) if static_layers else None
        sequence_hidden_size = config.hidden_size * (2 if config.bidirectional else 1)
        attention_hidden_size = max(16, sequence_hidden_size // 2)
        self.attention = (
            nn.Sequential(
                nn.Linear(sequence_hidden_size, attention_hidden_size),
                nn.Tanh(),
                nn.Linear(attention_hidden_size, 1),
            )
            if config.use_attention_pooling
            else None
        )
        self.sequence_gate = (
            nn.Linear(static_hidden_size, sequence_hidden_size)
            if config.use_static_gate and static_hidden_size > 0
            else None
        )
        self.head = nn.Sequential(
            nn.Linear(sequence_hidden_size + 8 + static_hidden_size, config.head_hidden_size),
            nn.ReLU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(config.head_hidden_size, 1),
        )

    def forward(self, x: torch.Tensor, static_features: torch.Tensor, ticker_ids: torch.Tensor) -> torch.Tensor:
        outputs, (hidden, _) = self.lstm(x)
        static_repr = self.static_head(static_features) if self.static_head is not None else None
        if self.attention is not None:
            attention_inputs = outputs
            if static_repr is not None and self.sequence_gate is not None:
                attention_inputs = attention_inputs * (1.0 + torch.sigmoid(self.sequence_gate(static_repr)).unsqueeze(1))
            attention_scores = self.attention(attention_inputs).squeeze(-1)
            attention_weights = torch.softmax(attention_scores, dim=1)
            seq_repr = torch.sum(outputs * attention_weights.unsqueeze(-1), dim=1)
        elif self.bidirectional:
            seq_repr = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            seq_repr = hidden[-1]
        if static_repr is not None and self.sequence_gate is not None:
            seq_repr = seq_repr * torch.sigmoid(self.sequence_gate(static_repr))
        ticker_repr = self.embedding(ticker_ids)
        inputs = [seq_repr, ticker_repr]
        if static_repr is not None:
            inputs.append(static_repr)
        logits = self.head(torch.cat(inputs, dim=1))
        return logits.squeeze(1)


def _build_sequences(
    panel: pd.DataFrame,
    sequence_feature_columns: list[str],
    static_feature_columns: list[str] | None = None,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> tuple[SequenceBundle, dict[str, int]]:
    static_feature_columns = static_feature_columns or []
    ticker_to_id = {ticker: idx for idx, ticker in enumerate(sorted(panel["ticker"].unique()))}
    sequences = []
    static_features = []
    targets = []
    sample_weights = []
    ticker_ids = []
    dates = []
    tickers = []

    for ticker, group in panel.groupby("ticker", sort=True):
        group = group.sort_values("date").reset_index(drop=True)
        sequence_values = group[sequence_feature_columns].to_numpy(dtype=float)
        static_values = group[static_feature_columns].to_numpy(dtype=float) if static_feature_columns else None
        for end_idx in range(sequence_length - 1, len(group)):
            sequences.append(sequence_values[end_idx - sequence_length + 1 : end_idx + 1])
            if static_values is not None:
                static_features.append(static_values[end_idx])
            else:
                static_features.append(np.empty(0, dtype=np.float32))
            targets.append(int(group.loc[end_idx, "target"]))
            sample_weights.append(float(group.loc[end_idx, "salience_weight"]) if "salience_weight" in group.columns else 1.0)
            ticker_ids.append(ticker_to_id[ticker])
            dates.append(pd.Timestamp(group.loc[end_idx, "date"]))
            tickers.append(ticker)

    return (
        SequenceBundle(
            sequences=np.asarray(sequences, dtype=np.float32),
            static_features=np.asarray(static_features, dtype=np.float32),
            targets=np.asarray(targets, dtype=np.float32),
            sample_weights=np.asarray(sample_weights, dtype=np.float32),
            ticker_ids=np.asarray(ticker_ids, dtype=np.int64),
            dates=np.asarray(dates),
            tickers=np.asarray(tickers),
        ),
        ticker_to_id,
    )


def _fit_scaler(features: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(features.reshape(-1, features.shape[-1]))
    return scaler


def _transform_sequences(features: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    flat = scaler.transform(features.reshape(-1, features.shape[-1]))
    return flat.reshape(features.shape)


def _fit_static_scaler(features: np.ndarray) -> StandardScaler | None:
    if features.ndim != 2 or features.shape[1] == 0:
        return None
    scaler = StandardScaler()
    scaler.fit(features)
    return scaler


def _transform_static(features: np.ndarray, scaler: StandardScaler | None) -> np.ndarray:
    if scaler is None or features.ndim != 2 or features.shape[1] == 0:
        return features
    return scaler.transform(features)


def _train_model(
    model: PairLSTM,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    pos_weight: float,
    config: LSTMExperimentConfig,
) -> PairLSTM:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device), reduction="none")
    best_state = None
    best_valid_loss = float("inf")
    patience = 0

    model.to(device)
    for _ in range(config.epochs):
        model.train()
        for sequences, static_features, ticker_ids, targets, sample_weights in train_loader:
            sequences = sequences.to(device)
            static_features = static_features.to(device)
            ticker_ids = ticker_ids.to(device)
            targets = targets.to(device)
            sample_weights = sample_weights.to(device)
            optimizer.zero_grad()
            logits = model(sequences, static_features, ticker_ids)
            loss = criterion(logits, targets)
            loss = (loss * sample_weights).mean()
            loss.backward()
            optimizer.step()

        model.eval()
        losses = []
        with torch.no_grad():
            for sequences, static_features, ticker_ids, targets, sample_weights in valid_loader:
                sequences = sequences.to(device)
                static_features = static_features.to(device)
                ticker_ids = ticker_ids.to(device)
                targets = targets.to(device)
                sample_weights = sample_weights.to(device)
                logits = model(sequences, static_features, ticker_ids)
                loss = criterion(logits, targets)
                losses.append(float((loss * sample_weights).mean().item()))
        valid_loss = float(np.mean(losses)) if losses else float("inf")
        if valid_loss + 1e-5 < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 3:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict_probabilities(model: PairLSTM, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for sequences, static_features, ticker_ids, _, _ in loader:
            sequences = sequences.to(device)
            static_features = static_features.to(device)
            ticker_ids = ticker_ids.to(device)
            logits = model(sequences, static_features, ticker_ids)
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty(0)


def _window_predictions(
    bundle: SequenceBundle,
    ticker_count: int,
    label: str,
    sequence_feature_columns: list[str],
    config: LSTMExperimentConfig,
    static_feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    static_feature_columns = static_feature_columns or []
    dates = pd.to_datetime(bundle.dates)
    months = monthly_test_windows(pd.Series(dates[dates >= pd.Timestamp("2025-01-01")]))
    predictions = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for window_start, window_end in months:
        train_mask = dates < window_start
        test_mask = (dates >= window_start) & (dates <= window_end)
        if int(train_mask.sum()) < MIN_TRAIN_ROWS or int(test_mask.sum()) < MIN_TEST_ROWS:
            continue

        validation_start = window_start - pd.Timedelta(days=90)
        valid_mask = train_mask & (dates >= validation_start)
        fit_mask = train_mask & (dates < validation_start)
        if int(valid_mask.sum()) < MIN_VALID_ROWS:
            valid_indices = np.where(train_mask)[0][-MIN_VALID_ROWS:]
            valid_mask = np.zeros_like(train_mask, dtype=bool)
            valid_mask[valid_indices] = True
            fit_mask = train_mask & ~valid_mask
        if int(fit_mask.sum()) < MIN_TRAIN_ROWS // 2:
            fit_mask = train_mask

        scaler = _fit_scaler(bundle.sequences[fit_mask])
        static_scaler = _fit_static_scaler(bundle.static_features[fit_mask])
        fit_sequences = _transform_sequences(bundle.sequences[fit_mask], scaler)
        valid_sequences = _transform_sequences(bundle.sequences[valid_mask], scaler)
        test_sequences = _transform_sequences(bundle.sequences[test_mask], scaler)
        fit_static = _transform_static(bundle.static_features[fit_mask], static_scaler)
        valid_static = _transform_static(bundle.static_features[valid_mask], static_scaler)
        test_static = _transform_static(bundle.static_features[test_mask], static_scaler)

        fit_sample_weights = bundle.sample_weights[fit_mask] if config.use_salience_weights else np.ones(int(fit_mask.sum()), dtype=np.float32)
        valid_sample_weights = bundle.sample_weights[valid_mask] if config.use_salience_weights else np.ones(int(valid_mask.sum()), dtype=np.float32)
        test_sample_weights = np.ones(int(test_mask.sum()), dtype=np.float32)
        fit_dataset = SequenceDataset(
            fit_sequences,
            fit_static,
            bundle.ticker_ids[fit_mask],
            bundle.targets[fit_mask],
            fit_sample_weights,
        )
        valid_dataset = SequenceDataset(
            valid_sequences,
            valid_static,
            bundle.ticker_ids[valid_mask],
            bundle.targets[valid_mask],
            valid_sample_weights,
        )
        test_dataset = SequenceDataset(
            test_sequences,
            test_static,
            bundle.ticker_ids[test_mask],
            bundle.targets[test_mask],
            test_sample_weights,
        )

        fit_loader = DataLoader(fit_dataset, batch_size=config.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        fit_targets = bundle.targets[fit_mask]
        positives = float(fit_targets.sum())
        negatives = float(len(fit_targets) - positives)
        pos_weight = negatives / positives if positives > 0 else 1.0

        model = PairLSTM(
            input_size=len(sequence_feature_columns),
            ticker_count=ticker_count,
            static_input_size=len(static_feature_columns),
            config=config,
        )
        model = _train_model(model, fit_loader, valid_loader, device, pos_weight=pos_weight, config=config)
        validation_probabilities = _predict_probabilities(model, valid_loader, device)
        threshold = choose_classification_threshold(bundle.targets[valid_mask], validation_probabilities)

        full_train_sequences = _transform_sequences(bundle.sequences[train_mask], scaler)
        full_train_static = _transform_static(bundle.static_features[train_mask], static_scaler)
        full_train_dataset = SequenceDataset(
            full_train_sequences,
            full_train_static,
            bundle.ticker_ids[train_mask],
            bundle.targets[train_mask],
            bundle.sample_weights[train_mask] if config.use_salience_weights else np.ones(int(train_mask.sum()), dtype=np.float32),
        )
        full_train_loader = DataLoader(full_train_dataset, batch_size=config.batch_size, shuffle=True)
        model = PairLSTM(
            input_size=len(sequence_feature_columns),
            ticker_count=ticker_count,
            static_input_size=len(static_feature_columns),
            config=config,
        )
        model = _train_model(model, full_train_loader, valid_loader, device, pos_weight=pos_weight, config=config)
        raw_probabilities = _predict_probabilities(model, test_loader, device)
        probabilities = shift_probabilities_for_threshold(raw_probabilities, threshold)
        predictions.append(
            pd.DataFrame(
                {
                    "date": dates[test_mask],
                    "ticker": bundle.tickers[test_mask],
                    "target": bundle.targets[test_mask].astype(int),
                    "probability": probabilities,
                    "model": label,
                }
            )
        )

    if not predictions:
        raise RuntimeError(f"No predictions generated for {label}")
    return pd.concat(predictions, ignore_index=True).sort_values(["date", "ticker"])


def _window_metrics(predictions: pd.DataFrame) -> dict[str, object]:
    frame = predictions.copy()
    frame["predicted_class"] = (frame["probability"] >= 0.5).astype(int)
    frame["correct"] = (frame["predicted_class"] == frame["target"].astype(int)).astype(int)
    outputs = {}
    for label, mask in {
        "overall": np.ones(len(frame), dtype=bool),
        "pre_war": (frame["date"] >= PRE_WAR_START) & (frame["date"] <= PRE_WAR_END),
        "war": (frame["date"] >= WAR_START) & (frame["date"] <= WAR_END),
    }.items():
        subset = frame.loc[mask]
        outputs[label] = {
            "accuracy": float(subset["correct"].mean()) if len(subset) else float("nan"),
            "n": int(len(subset)),
        }
    return outputs


def run_lstm_pair_experiment(
    config: LSTMExperimentConfig | None = None,
    artifact_prefix: str = "lstm_pair",
) -> dict[str, object]:
    ensure_directories()
    set_seed(RANDOM_SEED)
    config = config or LSTMExperimentConfig()
    panel = _build_panel(config)

    baseline_sequence_columns = sorted(set(QUANT_FEATURES + BASELINE_NEWS_FEATURES))
    sentiment_static_columns = sorted(set(SENTIMENT_EXTRA_FEATURES))
    if config.use_enriched_sentiment_features:
        sentiment_static_columns = sorted(set(sentiment_static_columns + ENRICHED_SENTIMENT_EXTRA_FEATURES))
    sentiment_static_columns = [column for column in sentiment_static_columns if column not in baseline_sequence_columns]

    baseline_bundle, ticker_to_id = _build_sequences(
        panel[["date", "ticker", "target"] + baseline_sequence_columns].copy(),
        sequence_feature_columns=baseline_sequence_columns,
        sequence_length=config.sequence_length,
    )
    sentiment_bundle, _ = _build_sequences(
        panel[["date", "ticker", "target"] + baseline_sequence_columns + sentiment_static_columns].copy(),
        sequence_feature_columns=baseline_sequence_columns,
        static_feature_columns=sentiment_static_columns,
        sequence_length=config.sequence_length,
    )

    baseline_predictions = _window_predictions(
        baseline_bundle,
        ticker_count=len(ticker_to_id),
        label="baseline_lstm",
        sequence_feature_columns=baseline_sequence_columns,
        config=config,
    )
    sentiment_predictions = _window_predictions(
        sentiment_bundle,
        ticker_count=len(ticker_to_id),
        label="sentiment_lstm",
        sequence_feature_columns=baseline_sequence_columns,
        config=config,
        static_feature_columns=sentiment_static_columns,
    )

    baseline_predictions.to_csv(ARTIFACT_DIR / f"{artifact_prefix}_baseline_predictions.csv", index=False)
    sentiment_predictions.to_csv(ARTIFACT_DIR / f"{artifact_prefix}_sentiment_predictions.csv", index=False)

    output = {
        "project_name": artifact_prefix,
        "target": (
            f"{config.target_horizon_days}-day excess return vs {BENCHMARK_TICKER} "
            f"with {config.target_threshold:.2%} threshold"
        ),
        "sequence_length": config.sequence_length,
        "history_start": START_DATE,
        "config": asdict(config),
        "baseline_sequence_feature_count": len(baseline_sequence_columns),
        "sentiment_static_feature_count": len(sentiment_static_columns),
        "baseline_sequence_features": baseline_sequence_columns,
        "sentiment_static_features": sentiment_static_columns,
        "baseline_metrics": _window_metrics(baseline_predictions),
        "sentiment_metrics": _window_metrics(sentiment_predictions),
    }
    save_json(ARTIFACT_DIR / f"{artifact_prefix}_metrics.json", output)
    return output
