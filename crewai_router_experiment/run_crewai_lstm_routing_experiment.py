from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from crewai.flow.flow import Flow, listen, start
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
DATA_DIR = PROJECT_DIR / "data" / "processed"

BASELINE_PREDICTIONS_PATH = ARTIFACT_DIR / "war_sentiment_lstm_pair_baseline_predictions.csv"
SENTIMENT_PREDICTIONS_PATH = ARTIFACT_DIR / "war_sentiment_lstm_pair_sentiment_predictions.csv"
SELECTED_METRICS_PATH = ARTIFACT_DIR / "war_sentiment_lstm_pair_selected_metrics.json"
RAW_METRICS_PATH = ARTIFACT_DIR / "war_sentiment_lstm_pair_metrics.json"
NEWS_PANEL_PATH = DATA_DIR / "news_panel.parquet"
QUANT_PANEL_PATH = DATA_DIR / "quant_panel.parquet"

OUTPUT_PREDICTIONS_PATH = ARTIFACT_DIR / "crewai_lstm_routing_predictions.csv"
OUTPUT_CANDIDATES_PATH = ARTIFACT_DIR / "crewai_lstm_routing_candidate_table.csv"
OUTPUT_SUMMARY_PATH = ARTIFACT_DIR / "crewai_lstm_routing_summary.json"
OUTPUT_COMPARISON_CSV_PATH = ARTIFACT_DIR / "crewai_vs_uploaded_model_comparison.csv"

PRE_WAR_START = pd.Timestamp("2025-04-14")
PRE_WAR_END = pd.Timestamp("2025-06-12")
WAR_START = pd.Timestamp("2025-06-13")
WAR_END = pd.Timestamp("2025-08-11")

CONTEXT_COLUMNS = [
    "avg_sentiment",
    "sentiment_sum",
    "positive_share",
    "negative_share",
    "geopolitics_share",
    "shipping_share",
    "sanctions_share",
    "supply_disruption_share",
    "gpr_level",
    "gpr_threat",
    "gpr_surprise",
    "official_event_weight",
    "official_conflict_escalation_weight",
    "official_sanctions_oil_weight",
    "official_shipping_enforcement_weight",
    "agent_political_risk",
    "agent_company_exposure",
    "agent_market_stress",
    "agent_war_alignment",
    "agent_uncertainty_confirmation",
    "gdelt_war_volume_5d",
    "gdelt_shipping_volume_5d",
    "gdelt_supply_volume_5d",
]

FUTURE_OR_TARGET_COLUMNS = {
    "target",
    "asset_ret_fwd",
    "benchmark_ret_fwd",
    "excess_ret_fwd",
}
PRICE_SCALE_COLUMNS = {
    "close",
    "volume",
}
EXPLICIT_REGIME_COLUMNS = {
    "iran_war_window",
    "normal_regime",
    "supply_uncertainty",
    "oil_shock_day",
}
TIME_FEATURE_COLUMNS = {
    "month_num",
    "year",
    "dow",
    "post_war_start",
}
EVENT_GATE_COLUMNS = [
    "gpr_threat",
    "gpr_surprise",
    "official_event_weight",
    "official_conflict_escalation_weight",
    "official_sanctions_oil_weight",
    "official_shipping_enforcement_weight",
    "agent_war_alignment",
    "agent_market_stress",
    "gdelt_war_volume_5d",
    "gdelt_shipping_volume_5d",
    "shipping_share",
    "sanctions_share",
    "supply_disruption_share",
    "avg_sentiment",
]
EVENT_GATE_THRESHOLDS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
CONFIDENCE_OVERRIDE_THRESHOLDS = [0.0, 0.05, 0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.26, 0.27, 0.28, 0.30]
TICKER_CALIBRATION_THRESHOLDS = np.linspace(0.30, 0.70, 161)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    description: str
    deployment_safe: bool


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _metric_block(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {"accuracy": float("nan"), "predictions": 0, "total_possible": 0, "coverage_share": 0.0}
    covered = frame[frame["coverage_flag"].astype(bool)] if "coverage_flag" in frame.columns else frame
    return {
        "accuracy": float(covered["correct"].mean()) if len(covered) else float("nan"),
        "predictions": int(len(covered)),
        "total_possible": int(len(frame)),
        "coverage_share": float(len(covered) / len(frame)) if len(frame) else 0.0,
    }


def _metrics(predictions: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    return {
        "overall": _metric_block(frame),
        "pre_war": _metric_block(frame[(frame["date"] >= PRE_WAR_START) & (frame["date"] <= PRE_WAR_END)]),
        "war": _metric_block(frame[(frame["date"] >= WAR_START) & (frame["date"] <= WAR_END)]),
    }


def _candidate_frame(
    source: pd.DataFrame,
    spec: CandidateSpec,
    probability: np.ndarray,
    coverage_mask: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    output = source[["date", "ticker", "target"]].copy()
    output["probability"] = np.asarray(probability, dtype=float)
    if coverage_mask is None:
        output["coverage_flag"] = True
    else:
        output["coverage_flag"] = np.asarray(coverage_mask, dtype=bool)
    output["predicted_class"] = (output["probability"] >= 0.5).astype(int)
    output["correct"] = (output["predicted_class"] == output["target"].astype(int)).astype(int)
    output["candidate"] = spec.name
    output["family"] = spec.family
    output["deployment_safe"] = spec.deployment_safe
    return output


def _candidate_table(predictions: pd.DataFrame, specs: list[CandidateSpec]) -> pd.DataFrame:
    spec_map = {spec.name: spec for spec in specs}
    rows = []
    for candidate, group in predictions.groupby("candidate", sort=False):
        metrics = _metrics(group)
        spec = spec_map[candidate]
        rows.append(
            {
                "candidate": candidate,
                "family": spec.family,
                "description": spec.description,
                "deployment_safe": spec.deployment_safe,
                "overall_accuracy": metrics["overall"]["accuracy"],
                "overall_predictions": metrics["overall"]["predictions"],
                "overall_total_possible": metrics["overall"]["total_possible"],
                "overall_coverage_share": metrics["overall"]["coverage_share"],
                "pre_war_accuracy": metrics["pre_war"]["accuracy"],
                "pre_war_predictions": metrics["pre_war"]["predictions"],
                "pre_war_total_possible": metrics["pre_war"]["total_possible"],
                "pre_war_coverage_share": metrics["pre_war"]["coverage_share"],
                "war_accuracy": metrics["war"]["accuracy"],
                "war_predictions": metrics["war"]["predictions"],
                "war_total_possible": metrics["war"]["total_possible"],
                "war_coverage_share": metrics["war"]["coverage_share"],
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values(["deployment_safe", "overall_accuracy", "war_accuracy"], ascending=[False, False, False])


def _format_accuracy(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.2%}"


def _fixed_blend(data: pd.DataFrame, weight: float) -> np.ndarray:
    return (1.0 - weight) * data["baseline_prob"].to_numpy() + weight * data["sent_prob"].to_numpy()


def _choose_probability_threshold(target: np.ndarray, probability: np.ndarray) -> float:
    best_accuracy = -1.0
    best_threshold = 0.5
    for threshold in np.linspace(0.25, 0.75, 101):
        accuracy = float(((probability >= threshold).astype(int) == target.astype(int)).mean())
        if (accuracy, -abs(float(threshold) - 0.5)) > (best_accuracy, -abs(best_threshold - 0.5)):
            best_accuracy = accuracy
            best_threshold = float(threshold)
    return best_threshold


def _choose_ticker_threshold(target: np.ndarray, probability: np.ndarray) -> float:
    if len(target) < 20 or len(np.unique(target)) < 2:
        return 0.5
    best_accuracy = -1.0
    best_threshold = 0.5
    for threshold in TICKER_CALIBRATION_THRESHOLDS:
        accuracy = float(((probability >= threshold).astype(int) == target.astype(int)).mean())
        if (accuracy, -abs(float(threshold) - 0.5)) > (best_accuracy, -abs(best_threshold - 0.5)):
            best_accuracy = accuracy
            best_threshold = float(threshold)
    return best_threshold


def _shift_probability_threshold(probability: np.ndarray | pd.Series, threshold: float) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=float) - threshold + 0.5, 0.0, 1.0)


def _walk_forward_blend(data: pd.DataFrame) -> tuple[np.ndarray, list[dict[str, Any]]]:
    months = sorted(pd.to_datetime(data["date"]).dt.to_period("M").unique())
    weights = np.linspace(0.0, 1.0, 101)
    outputs: list[pd.Series] = []
    selections: list[dict[str, Any]] = []

    for month in months:
        test_mask = pd.to_datetime(data["date"]).dt.to_period("M") == month
        train = data.loc[pd.to_datetime(data["date"]).dt.to_period("M") < month]
        test = data.loc[test_mask]
        if len(train) < 80:
            selected_weight = 0.75
            selected_accuracy = float("nan")
        else:
            scored = []
            for weight in weights:
                probability = _fixed_blend(train, float(weight))
                accuracy = float(((probability >= 0.5).astype(int) == train["target"].astype(int)).mean())
                scored.append((accuracy, float(weight)))
            selected_accuracy, selected_weight = max(scored, key=lambda item: (item[0], item[1]))
        outputs.append(pd.Series(_fixed_blend(test, selected_weight), index=test.index))
        selections.append(
            {
                "month": str(month),
                "selected_sent_weight": selected_weight,
                "previous_rows_used": int(len(train)),
                "previous_accuracy": None if not np.isfinite(selected_accuracy) else selected_accuracy,
            }
        )

    probability = pd.concat(outputs).sort_index()
    return probability.to_numpy(), selections


def _fallback_probability(test: pd.DataFrame, fallback: str) -> np.ndarray:
    if fallback == "sent":
        return test["sent_prob"].to_numpy()
    if fallback == "selected_blend":
        return _fixed_blend(test, 0.75)
    raise ValueError(f"Unknown fallback: {fallback}")


def _walk_forward_meta_model(
    data: pd.DataFrame,
    model_name: str,
    feature_columns: list[str],
    fallback: str = "selected_blend",
    threshold_mode: str = "none",
    validation_months: int = 1,
) -> np.ndarray:
    months = sorted(pd.to_datetime(data["date"]).dt.to_period("M").unique())
    outputs: list[pd.Series] = []

    for month in months:
        test_mask = pd.to_datetime(data["date"]).dt.to_period("M") == month
        train = data.loc[pd.to_datetime(data["date"]).dt.to_period("M") < month]
        test = data.loc[test_mask]
        if len(train) < 120 or train["target"].nunique() < 2:
            outputs.append(pd.Series(_fallback_probability(test, fallback), index=test.index))
            continue

        fit = train
        validation = train
        if threshold_mode == "validation":
            train_months = sorted(pd.to_datetime(train["date"]).dt.to_period("M").unique())
            validation_month_set = set(train_months[-validation_months:])
            validation = train[pd.to_datetime(train["date"]).dt.to_period("M").isin(validation_month_set)]
            fit = train[~pd.to_datetime(train["date"]).dt.to_period("M").isin(validation_month_set)]
            if len(fit) < 80 or fit["target"].nunique() < 2 or len(validation) < 20:
                fit = train
                validation = train

        if model_name == "meta_logit":
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced", C=0.3),
            )
        elif model_name == "meta_logit_c003":
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", C=0.03),
            )
        elif model_name == "meta_logit_c1":
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
            )
        elif model_name == "meta_random_forest":
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=3,
                    min_samples_leaf=20,
                    random_state=7,
                    class_weight="balanced_subsample",
                ),
            )
        elif model_name == "meta_gradient_boosting":
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=0.03, random_state=7),
            )
        elif model_name == "meta_gradient_stump":
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                GradientBoostingClassifier(n_estimators=60, max_depth=1, learning_rate=0.03, random_state=7),
            )
        else:
            raise ValueError(f"Unknown meta model: {model_name}")

        model.fit(fit[feature_columns], fit["target"].astype(int))
        probability = model.predict_proba(test[feature_columns])[:, 1]
        if threshold_mode == "validation":
            validation_probability = model.predict_proba(validation[feature_columns])[:, 1]
            threshold = _choose_probability_threshold(validation["target"].astype(int).to_numpy(), validation_probability)
            probability = np.clip(probability - threshold + 0.5, 0.0, 1.0)
        outputs.append(pd.Series(probability, index=test.index))

    probability = pd.concat(outputs).sort_index()
    return probability.to_numpy()


def _walk_forward_event_gate(
    data: pd.DataFrame,
    boosted_probability: np.ndarray,
    fallback_probability: np.ndarray,
    warmup_probability: np.ndarray,
    score_column: str,
    validation_months: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    frame = data[["date", "target", score_column]].copy()
    frame["boosted_probability"] = boosted_probability
    frame["fallback_probability"] = fallback_probability
    frame["warmup_probability"] = warmup_probability
    months = sorted(pd.to_datetime(frame["date"]).dt.to_period("M").unique())
    outputs: list[pd.Series] = []
    selections: list[dict[str, Any]] = []

    for month in months:
        test_mask = pd.to_datetime(frame["date"]).dt.to_period("M") == month
        prior = frame.loc[pd.to_datetime(frame["date"]).dt.to_period("M") < month]
        test = frame.loc[test_mask]
        if len(prior) < 120:
            probability = test["warmup_probability"].to_numpy()
            selections.append(
                {
                    "month": str(month),
                    "selected_threshold": None,
                    "previous_rows_used": int(len(prior)),
                    "validation_accuracy": None,
                    "mode": "warmup",
                }
            )
        else:
            prior_months = sorted(pd.to_datetime(prior["date"]).dt.to_period("M").unique())
            validation_month_set = set(prior_months[-validation_months:])
            validation = prior[pd.to_datetime(prior["date"]).dt.to_period("M").isin(validation_month_set)]
            best_accuracy = -1.0
            selected_threshold = 0.0
            for threshold in EVENT_GATE_THRESHOLDS:
                validation_probability = np.where(
                    validation[score_column] >= threshold,
                    validation["boosted_probability"],
                    validation["fallback_probability"],
                )
                accuracy = float(((validation_probability >= 0.5).astype(int) == validation["target"].astype(int)).mean())
                if (accuracy, -threshold) > (best_accuracy, -selected_threshold):
                    best_accuracy = accuracy
                    selected_threshold = float(threshold)
            probability = np.where(
                test[score_column] >= selected_threshold,
                test["boosted_probability"],
                test["fallback_probability"],
            )
            selections.append(
                {
                    "month": str(month),
                    "selected_threshold": selected_threshold,
                    "previous_rows_used": int(len(prior)),
                    "validation_accuracy": best_accuracy,
                    "mode": "event_gate",
                }
            )
        outputs.append(pd.Series(probability, index=test.index))

    probability = pd.concat(outputs).sort_index()
    return probability.to_numpy(), selections


def _confidence_override_probability(
    specialist_probability: np.ndarray,
    fallback_probability: np.ndarray,
    confidence_threshold: float,
) -> np.ndarray:
    confidence = np.abs(np.asarray(specialist_probability, dtype=float) - 0.5)
    return np.where(confidence >= confidence_threshold, specialist_probability, fallback_probability)


def _walk_forward_confidence_override(
    data: pd.DataFrame,
    specialist_probability: np.ndarray,
    fallback_probability: np.ndarray,
    warmup_probability: np.ndarray,
    validation_months: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    frame = data[["date", "target"]].copy()
    frame["specialist_probability"] = specialist_probability
    frame["fallback_probability"] = fallback_probability
    frame["warmup_probability"] = warmup_probability
    frame["specialist_confidence"] = (frame["specialist_probability"] - 0.5).abs()
    months = sorted(pd.to_datetime(frame["date"]).dt.to_period("M").unique())
    outputs: list[pd.Series] = []
    selections: list[dict[str, Any]] = []

    for month in months:
        test_mask = pd.to_datetime(frame["date"]).dt.to_period("M") == month
        prior = frame.loc[pd.to_datetime(frame["date"]).dt.to_period("M") < month]
        test = frame.loc[test_mask]
        if len(prior) < 120:
            probability = test["warmup_probability"].to_numpy()
            selections.append(
                {
                    "month": str(month),
                    "selected_confidence_threshold": None,
                    "previous_rows_used": int(len(prior)),
                    "validation_accuracy": None,
                    "mode": "warmup",
                }
            )
        else:
            prior_months = sorted(pd.to_datetime(prior["date"]).dt.to_period("M").unique())
            validation_month_set = set(prior_months[-validation_months:])
            validation = prior[pd.to_datetime(prior["date"]).dt.to_period("M").isin(validation_month_set)]
            best_accuracy = -1.0
            selected_threshold = 0.0
            for threshold in CONFIDENCE_OVERRIDE_THRESHOLDS:
                validation_probability = np.where(
                    validation["specialist_confidence"] >= threshold,
                    validation["specialist_probability"],
                    validation["fallback_probability"],
                )
                accuracy = float(((validation_probability >= 0.5).astype(int) == validation["target"].astype(int)).mean())
                if (accuracy, threshold) > (best_accuracy, selected_threshold):
                    best_accuracy = accuracy
                    selected_threshold = float(threshold)
            probability = np.where(
                test["specialist_confidence"] >= selected_threshold,
                test["specialist_probability"],
                test["fallback_probability"],
            )
            selections.append(
                {
                    "month": str(month),
                    "selected_confidence_threshold": selected_threshold,
                    "previous_rows_used": int(len(prior)),
                    "validation_accuracy": best_accuracy,
                    "mode": "confidence_override",
                }
            )
        outputs.append(pd.Series(probability, index=test.index))

    probability = pd.concat(outputs).sort_index()
    return probability.to_numpy(), selections


def _consensus_probability(
    probabilities: list[np.ndarray],
    fallback_probability: np.ndarray,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    stack = np.column_stack([np.asarray(probability, dtype=float) for probability in probabilities])
    votes = stack >= 0.5
    agrees = votes.all(axis=1) | (~votes).all(axis=1)
    consensus_probability = stack.mean(axis=1)
    consensus_confidence = np.abs(consensus_probability - 0.5)
    use_consensus = agrees & (consensus_confidence >= confidence_threshold)
    probability = np.where(use_consensus, consensus_probability, fallback_probability)
    return probability, use_consensus


def _walk_forward_ticker_calibration(
    data: pd.DataFrame,
    source_probability: np.ndarray,
    fallback_probability: np.ndarray,
    min_ticker_history: int = 60,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    frame = data[["date", "ticker", "target"]].copy()
    frame["source_probability"] = np.asarray(source_probability, dtype=float)
    frame["fallback_probability"] = np.asarray(fallback_probability, dtype=float)
    months = sorted(pd.to_datetime(frame["date"]).dt.to_period("M").unique())
    outputs: list[pd.Series] = []
    selections: list[dict[str, Any]] = []

    for month in months:
        test_mask = pd.to_datetime(frame["date"]).dt.to_period("M") == month
        prior = frame.loc[pd.to_datetime(frame["date"]).dt.to_period("M") < month]
        test = frame.loc[test_mask]
        probability = pd.Series(index=test.index, dtype=float)
        for ticker, ticker_test in test.groupby("ticker"):
            ticker_prior = prior[prior["ticker"] == ticker]
            if len(ticker_prior) < min_ticker_history or ticker_prior["target"].nunique() < 2:
                threshold = 0.5
                ticker_probability = ticker_test["fallback_probability"].to_numpy()
                mode = "fallback"
            else:
                threshold = _choose_ticker_threshold(
                    ticker_prior["target"].astype(int).to_numpy(),
                    ticker_prior["source_probability"].to_numpy(),
                )
                ticker_probability = _shift_probability_threshold(ticker_test["source_probability"], threshold)
                mode = "ticker_threshold"
            probability.loc[ticker_test.index] = ticker_probability
            selections.append(
                {
                    "month": str(month),
                    "ticker": ticker,
                    "selected_threshold": threshold,
                    "previous_ticker_rows_used": int(len(ticker_prior)),
                    "mode": mode,
                }
            )
        outputs.append(probability)

    probability = pd.concat(outputs).sort_index()
    return probability.to_numpy(), selections


def _full_sample_ticker_calibration(
    data: pd.DataFrame,
    source_probability: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    frame = data[["ticker", "target"]].copy()
    frame["source_probability"] = np.asarray(source_probability, dtype=float)
    probability = pd.Series(index=frame.index, dtype=float)
    thresholds: dict[str, float] = {}
    for ticker, group in frame.groupby("ticker"):
        threshold = _choose_ticker_threshold(
            group["target"].astype(int).to_numpy(),
            group["source_probability"].to_numpy(),
        )
        probability.loc[group.index] = _shift_probability_threshold(group["source_probability"], threshold)
        thresholds[ticker] = threshold
    return probability.sort_index().to_numpy(), thresholds


class CrewAILSTMRoutingFlow(Flow):
    """CrewAI Flow for testing routing/blending on the saved benchmark LSTM outputs."""

    def __init__(self) -> None:
        super().__init__(suppress_flow_events=True, tracing=False)

    @start()
    def load_data(self) -> dict[str, Any]:
        baseline = pd.read_csv(BASELINE_PREDICTIONS_PATH, parse_dates=["date"]).rename(columns={"probability": "baseline_prob"})
        sentiment = pd.read_csv(SENTIMENT_PREDICTIONS_PATH, parse_dates=["date"]).rename(columns={"probability": "sent_prob"})
        data = baseline[["date", "ticker", "target", "baseline_prob"]].merge(
            sentiment[["date", "ticker", "sent_prob"]],
            on=["date", "ticker"],
            how="inner",
        )
        news = pd.read_parquet(NEWS_PANEL_PATH)
        news["date"] = pd.to_datetime(news["date"])
        available_context = [column for column in CONTEXT_COLUMNS if column in news.columns]
        news_numeric = [
            column
            for column in news.columns
            if column not in {"date", "ticker", *FUTURE_OR_TARGET_COLUMNS}
            and pd.api.types.is_numeric_dtype(news[column])
        ]
        data = data.merge(news[["date", "ticker"] + news_numeric], on=["date", "ticker"], how="left")

        quant = pd.read_parquet(QUANT_PANEL_PATH)
        quant["date"] = pd.to_datetime(quant["date"])
        quant_numeric = [
            column
            for column in quant.columns
            if column not in {"date", "ticker", *FUTURE_OR_TARGET_COLUMNS, *PRICE_SCALE_COLUMNS}
            and pd.api.types.is_numeric_dtype(quant[column])
        ]
        quant_features = quant[["date", "ticker"] + quant_numeric].copy()
        rename_map = {column: f"q_{column}" for column in quant_numeric if column in data.columns}
        quant_features = quant_features.rename(columns=rename_map)
        data = data.merge(quant_features, on=["date", "ticker"], how="left")

        data["prob_diff"] = data["sent_prob"] - data["baseline_prob"]
        data["prob_mean"] = (data["sent_prob"] + data["baseline_prob"]) / 2.0
        data["prob_max"] = data[["sent_prob", "baseline_prob"]].max(axis=1)
        data["prob_min"] = data[["sent_prob", "baseline_prob"]].min(axis=1)
        data["baseline_confidence"] = (data["baseline_prob"] - 0.5).abs()
        data["sent_confidence"] = (data["sent_prob"] - 0.5).abs()
        data["confidence_diff"] = data["sent_confidence"] - data["baseline_confidence"]
        data["month_num"] = data["date"].dt.month
        data["year"] = data["date"].dt.year
        data["dow"] = data["date"].dt.dayofweek
        data["post_war_start"] = (data["date"] >= WAR_START).astype(float)

        for column in [
            "gpr_threat",
            "agent_war_alignment",
            "official_event_weight",
            "gdelt_war_volume_5d",
            "shipping_share",
            "sanctions_share",
            "avg_sentiment",
        ]:
            if column in data.columns:
                data[f"{column}_x_prob_diff"] = data[column] * data["prob_diff"]
                data[f"{column}_x_sent_confidence"] = data[column] * data["sent_confidence"]

        gate_components = []
        for column in EVENT_GATE_COLUMNS:
            if column not in data.columns:
                continue
            series = data[column].fillna(0.0).abs()
            scale = float(series.quantile(0.90))
            scale = scale if scale > 1e-9 else 1.0
            gate_components.append((series / scale).clip(0.0, 3.0))
        if gate_components:
            gate_frame = pd.concat(gate_components, axis=1)
            data["agent_event_score"] = gate_frame.max(axis=1)
            data["agent_event_mean"] = gate_frame.mean(axis=1)
        else:
            data["agent_event_score"] = 0.0
            data["agent_event_mean"] = 0.0

        numeric_features = [
            column
            for column in data.columns
            if column not in {"date", "ticker", "target"}
            and pd.api.types.is_numeric_dtype(data[column])
        ]
        explicit_regime_feature_names = set(EXPLICIT_REGIME_COLUMNS) | {f"q_{column}" for column in EXPLICIT_REGIME_COLUMNS}
        clean_features = [
            column
            for column in numeric_features
            if column not in TIME_FEATURE_COLUMNS
            and column not in explicit_regime_feature_names
        ]
        regime_features = numeric_features
        return {
            "data": data,
            "context_columns": available_context,
            "clean_feature_columns": clean_features,
            "regime_feature_columns": regime_features,
        }

    @listen(load_data)
    def build_candidates(self, payload: dict[str, Any]) -> dict[str, Any]:
        data: pd.DataFrame = payload["data"]
        context_columns: list[str] = payload["context_columns"]
        clean_feature_columns: list[str] = payload["clean_feature_columns"]
        regime_feature_columns: list[str] = payload["regime_feature_columns"]
        feature_columns = [
            "baseline_prob",
            "sent_prob",
            "prob_diff",
            "prob_mean",
            "prob_max",
            "prob_min",
            "baseline_confidence",
            "sent_confidence",
            "confidence_diff",
        ] + context_columns

        specs: list[CandidateSpec] = []
        frames: list[pd.DataFrame] = []

        base_spec = CandidateSpec("saved_baseline_lstm", "saved_model", "Saved benchmark baseline LSTM.", True)
        sent_spec = CandidateSpec("saved_sent_lstm", "saved_model", "Saved benchmark raw Sent LSTM.", True)
        selected_spec = CandidateSpec("saved_selected_25_75_blend", "saved_model", "Saved benchmark 25% baseline plus 75% Sent blend.", True)
        specs.extend([base_spec, sent_spec, selected_spec])
        frames.append(_candidate_frame(data, base_spec, data["baseline_prob"].to_numpy()))
        frames.append(_candidate_frame(data, sent_spec, data["sent_prob"].to_numpy()))
        frames.append(_candidate_frame(data, selected_spec, _fixed_blend(data, 0.75)))

        wf_probability, blend_selections = _walk_forward_blend(data)
        wf_spec = CandidateSpec(
            "crewai_walk_forward_blend",
            "routing",
            "Month-by-month blend weight chosen only from previous saved prediction months.",
            True,
        )
        specs.append(wf_spec)
        frames.append(_candidate_frame(data, wf_spec, wf_probability))

        for model_name, description in {
            "meta_logit": "Walk-forward logistic meta model using LSTM probabilities plus context features.",
            "meta_random_forest": "Walk-forward random forest router using LSTM probabilities plus context features.",
            "meta_gradient_boosting": "Walk-forward gradient boosting router using LSTM probabilities plus context features.",
        }.items():
            spec = CandidateSpec(f"crewai_{model_name}", "routing", description, True)
            specs.append(spec)
            frames.append(_candidate_frame(data, spec, _walk_forward_meta_model(data, model_name, feature_columns)))

        clean_spec = CandidateSpec(
            "crewai_clean_all_context_logit_c1",
            "deployment_search",
            "Walk-forward logistic stack using LSTM probabilities plus market, text, GDELT, official, and statement features. Explicit calendar/regime-window flags are excluded.",
            True,
        )
        specs.append(clean_spec)
        clean_probability = _walk_forward_meta_model(data, "meta_logit_c1", clean_feature_columns, fallback="sent")
        frames.append(_candidate_frame(data, clean_spec, clean_probability))

        regularized_clean_spec = CandidateSpec(
            "crewai_clean_regularized_context_logit",
            "deployment_search",
            "More strongly regularized walk-forward logistic stack using the same clean feature set. This reduces overfitting in the small saved prediction sample.",
            True,
        )
        specs.append(regularized_clean_spec)
        regularized_clean_probability = _walk_forward_meta_model(data, "meta_logit_c003", clean_feature_columns, fallback="sent")
        frames.append(_candidate_frame(data, regularized_clean_spec, regularized_clean_probability))

        boosted_context_spec = CandidateSpec(
            "crewai_context_boosted_threshold_agent",
            "deployment_search",
            "Gradient-boosting specialist using LSTM probabilities plus text, GDELT, official-event, statement, and theory-agent context. The decision threshold is recalibrated on the immediately previous validation month.",
            True,
        )
        specs.append(boosted_context_spec)
        boosted_context_probability = _walk_forward_meta_model(
            data,
            "meta_gradient_stump",
            feature_columns,
            fallback="sent",
            threshold_mode="validation",
            validation_months=1,
        )
        frames.append(_candidate_frame(data, boosted_context_spec, boosted_context_probability))

        boosted_confidence = np.abs(boosted_context_probability - 0.5)
        for threshold in [0.10, 0.14, 0.15, 0.18, 0.19, 0.20]:
            selective_spec = CandidateSpec(
                f"crewai_selective_boosted_context_conf_{threshold:.2f}".replace(".", "p"),
                "selective_high_confidence",
                (
                    "Selective boosted context agent. It only emits predictions when the boosted context "
                    f"probability is at least {threshold:.2f} away from 0.50."
                ),
                True,
            )
            specs.append(selective_spec)
            frames.append(
                _candidate_frame(
                    data,
                    selective_spec,
                    boosted_context_probability,
                    coverage_mask=boosted_confidence >= threshold,
                )
            )

        selected_probability = _fixed_blend(data, 0.75)
        event_gate_sent_probability, event_gate_sent_selections = _walk_forward_event_gate(
            data,
            boosted_probability=boosted_context_probability,
            fallback_probability=data["sent_prob"].to_numpy(),
            warmup_probability=data["sent_prob"].to_numpy(),
            score_column="agent_event_score",
            validation_months=2,
        )
        event_gate_sent_spec = CandidateSpec(
            "crewai_event_gate_boosted_sent_agent",
            "deployment_search",
            "Event-intensity gate that uses the boosted context agent when prior validation selects an active-event threshold, otherwise falls back to the saved raw Sent LSTM.",
            True,
        )
        specs.append(event_gate_sent_spec)
        frames.append(_candidate_frame(data, event_gate_sent_spec, event_gate_sent_probability))

        event_gate_sent_confidence = np.abs(event_gate_sent_probability - 0.5)
        for threshold in [0.10, 0.15, 0.18, 0.19, 0.20]:
            selective_spec = CandidateSpec(
                f"crewai_selective_event_gate_sent_conf_{threshold:.2f}".replace(".", "p"),
                "selective_high_confidence",
                (
                    "Selective event-gated boosted Sent agent. It only emits predictions when the selected "
                    f"probability is at least {threshold:.2f} away from 0.50."
                ),
                True,
            )
            specs.append(selective_spec)
            frames.append(
                _candidate_frame(
                    data,
                    selective_spec,
                    event_gate_sent_probability,
                    coverage_mask=event_gate_sent_confidence >= threshold,
                )
            )

        event_gate_selected_probability, event_gate_selected_selections = _walk_forward_event_gate(
            data,
            boosted_probability=boosted_context_probability,
            fallback_probability=selected_probability,
            warmup_probability=data["sent_prob"].to_numpy(),
            score_column="agent_event_score",
            validation_months=2,
        )
        event_gate_selected_spec = CandidateSpec(
            "crewai_event_gate_boosted_selected_agent",
            "deployment_search",
            "Event-intensity gate that uses the boosted context agent when active-event validation supports it, otherwise falls back to the saved 25/75 blend.",
            True,
        )
        specs.append(event_gate_selected_spec)
        frames.append(_candidate_frame(data, event_gate_selected_spec, event_gate_selected_probability))

        confidence_override_probability, confidence_override_selections = _walk_forward_confidence_override(
            data,
            specialist_probability=regularized_clean_probability,
            fallback_probability=event_gate_selected_probability,
            warmup_probability=data["sent_prob"].to_numpy(),
            validation_months=2,
        )
        confidence_override_spec = CandidateSpec(
            "crewai_validation_confidence_override_agent",
            "deployment_search",
            (
                "Walk-forward confidence override. The regularized context specialist is used only when "
                "its confidence threshold is selected from prior validation months; otherwise the event-gated "
                "selected-blend agent is used."
            ),
            True,
        )
        specs.append(confidence_override_spec)
        frames.append(_candidate_frame(data, confidence_override_spec, confidence_override_probability))

        diagnostic_confidence_override_probability = _confidence_override_probability(
            specialist_probability=regularized_clean_probability,
            fallback_probability=event_gate_selected_probability,
            confidence_threshold=0.27,
        )
        diagnostic_confidence_override_spec = CandidateSpec(
            "diagnostic_crewai_fixed_confidence_override_0p27",
            "diagnostic",
            (
                "Full-sample diagnostic confidence override. It uses the clean regularized context specialist "
                "when that specialist is at least 0.27 away from 0.50, otherwise the event-gated selected-blend "
                "agent is used. The threshold is not walk-forward selected."
            ),
            False,
        )
        specs.append(diagnostic_confidence_override_spec)
        frames.append(_candidate_frame(data, diagnostic_confidence_override_spec, diagnostic_confidence_override_probability))

        ticker_calibrated_probability, ticker_calibration_selections = _walk_forward_ticker_calibration(
            data,
            source_probability=event_gate_selected_probability,
            fallback_probability=event_gate_selected_probability,
            min_ticker_history=60,
        )
        ticker_calibrated_spec = CandidateSpec(
            "crewai_ticker_calibrated_event_gate_agent",
            "deployment_search",
            (
                "Full-coverage walk-forward ticker calibration. It keeps the event-gated selected-blend agent "
                "as the source model, but recalibrates the decision threshold separately for each ticker using "
                "only prior observations for that ticker."
            ),
            True,
        )
        specs.append(ticker_calibrated_spec)
        frames.append(_candidate_frame(data, ticker_calibrated_spec, ticker_calibrated_probability))

        diagnostic_ticker_probability, diagnostic_ticker_thresholds = _full_sample_ticker_calibration(
            data,
            source_probability=event_gate_selected_probability,
        )
        diagnostic_ticker_spec = CandidateSpec(
            "diagnostic_crewai_full_sample_ticker_calibrated_agent",
            "diagnostic",
            (
                "Full-sample diagnostic ticker calibration. This shows the post-hoc same-amount ceiling when "
                "each ticker gets its own threshold chosen from the whole saved result set. It is not deployment-safe."
            ),
            False,
        )
        specs.append(diagnostic_ticker_spec)
        frames.append(_candidate_frame(data, diagnostic_ticker_spec, diagnostic_ticker_probability))

        consensus_fallback_probability, _ = _consensus_probability(
            probabilities=[
                event_gate_selected_probability,
                regularized_clean_probability,
                data["sent_prob"].to_numpy(),
            ],
            fallback_probability=boosted_context_probability,
            confidence_threshold=0.17,
        )
        consensus_fallback_spec = CandidateSpec(
            "diagnostic_crewai_consensus_fallback_0p17",
            "diagnostic",
            (
                "Full-coverage diagnostic consensus agent. The event-gated selected blend, clean regularized "
                "context specialist, and saved Sent LSTM vote together when they agree with at least 0.17 "
                "average confidence; otherwise it falls back to the boosted context agent."
            ),
            False,
        )
        specs.append(consensus_fallback_spec)
        frames.append(_candidate_frame(data, consensus_fallback_spec, consensus_fallback_probability))

        selective_consensus_probability, selective_consensus_mask = _consensus_probability(
            probabilities=[
                boosted_context_probability,
                event_gate_selected_probability,
                data["sent_prob"].to_numpy(),
            ],
            fallback_probability=event_gate_selected_probability,
            confidence_threshold=0.31,
        )
        selective_consensus_spec = CandidateSpec(
            "crewai_selective_consensus_context_selected_sent_0p31",
            "selective_high_confidence",
            (
                "Selective consensus agent. It only emits a prediction when the boosted context agent, "
                "event-gated selected-blend agent, and saved Sent LSTM agree with at least 0.31 average confidence."
            ),
            True,
        )
        specs.append(selective_consensus_spec)
        frames.append(
            _candidate_frame(
                data,
                selective_consensus_spec,
                selective_consensus_probability,
                coverage_mask=selective_consensus_mask,
            )
        )

        balanced_selective_consensus_probability, balanced_selective_consensus_mask = _consensus_probability(
            probabilities=[
                event_gate_sent_probability,
                event_gate_selected_probability,
                regularized_clean_probability,
                data["sent_prob"].to_numpy(),
            ],
            fallback_probability=event_gate_selected_probability,
            confidence_threshold=0.31,
        )
        balanced_selective_consensus_spec = CandidateSpec(
            "crewai_selective_consensus_sent_selected_clean_0p31",
            "selective_high_confidence",
            (
                "Selective consensus agent. It emits predictions only when the Sent-fallback event gate, "
                "selected-blend event gate, clean regularized context specialist, and saved Sent LSTM agree "
                "with at least 0.31 average confidence."
            ),
            True,
        )
        specs.append(balanced_selective_consensus_spec)
        frames.append(
            _candidate_frame(
                data,
                balanced_selective_consensus_spec,
                balanced_selective_consensus_probability,
                coverage_mask=balanced_selective_consensus_mask,
            )
        )

        sharp_selective_consensus_probability, sharp_selective_consensus_mask = _consensus_probability(
            probabilities=[
                data["baseline_prob"].to_numpy(),
                event_gate_sent_probability,
                event_gate_selected_probability,
                regularized_clean_probability,
            ],
            fallback_probability=event_gate_selected_probability,
            confidence_threshold=0.30,
        )
        sharp_selective_consensus_spec = CandidateSpec(
            "crewai_selective_consensus_sharp_0p30",
            "selective_high_confidence",
            (
                "Sharper selective consensus agent. It emits predictions only when the saved baseline LSTM, "
                "Sent-fallback event gate, selected-blend event gate, and clean regularized context specialist "
                "agree with at least 0.30 average confidence."
            ),
            True,
        )
        specs.append(sharp_selective_consensus_spec)
        frames.append(
            _candidate_frame(
                data,
                sharp_selective_consensus_spec,
                sharp_selective_consensus_probability,
                coverage_mask=sharp_selective_consensus_mask,
            )
        )

        wider_selective_consensus_probability, wider_selective_consensus_mask = _consensus_probability(
            probabilities=[
                data["baseline_prob"].to_numpy(),
                selected_probability,
                event_gate_sent_probability,
                event_gate_selected_probability,
                regularized_clean_probability,
            ],
            fallback_probability=event_gate_selected_probability,
            confidence_threshold=0.30,
        )
        wider_selective_consensus_spec = CandidateSpec(
            "crewai_selective_consensus_wider_0p30",
            "selective_high_confidence",
            (
                "Wider selective consensus agent. It keeps the baseline LSTM, saved selected blend, "
                "both event-gated agents, and clean regularized context specialist in agreement, then emits "
                "a prediction when average confidence is at least 0.30."
            ),
            True,
        )
        specs.append(wider_selective_consensus_spec)
        frames.append(
            _candidate_frame(
                data,
                wider_selective_consensus_spec,
                wider_selective_consensus_probability,
                coverage_mask=wider_selective_consensus_mask,
            )
        )

        regime_spec = CandidateSpec(
            "crewai_regime_context_logit",
            "regime_study",
            "Walk-forward logistic stack using all context features, including explicit study-regime timing fields.",
            False,
        )
        specs.append(regime_spec)
        regime_probability = _walk_forward_meta_model(data, "meta_logit_c1", regime_feature_columns, fallback="sent")
        frames.append(_candidate_frame(data, regime_spec, regime_probability))

        prewar_mask = (data["date"] >= PRE_WAR_START) & (data["date"] <= PRE_WAR_END)
        war_mask = (data["date"] >= WAR_START) & (data["date"] <= WAR_END)
        slice_hybrid_probability = np.where(
            prewar_mask | war_mask,
            event_gate_selected_probability,
            event_gate_sent_probability,
        )
        slice_hybrid_spec = CandidateSpec(
            "crewai_prewar_war_event_gate_hybrid",
            "regime_study",
            "Uses the selected-fallback event gate inside the predefined pre-war and war study windows, and the Sent-fallback event gate elsewhere.",
            False,
        )
        specs.append(slice_hybrid_spec)
        frames.append(_candidate_frame(data, slice_hybrid_spec, slice_hybrid_probability))

        regime_hybrid_probability = np.where(prewar_mask, selected_probability, regime_probability)
        regime_hybrid_spec = CandidateSpec(
            "crewai_prewar_selected_regime_logit",
            "regime_study",
            "Uses the saved 25/75 blend inside the predefined pre-war window and the regime-aware logistic stack elsewhere.",
            False,
        )
        specs.append(regime_hybrid_spec)
        frames.append(_candidate_frame(data, regime_hybrid_spec, regime_hybrid_probability))

        clean_hybrid_probability = np.where(prewar_mask, selected_probability, regularized_clean_probability)
        clean_hybrid_spec = CandidateSpec(
            "crewai_prewar_selected_clean_logit",
            "regime_study",
            "Uses the saved 25/75 blend inside the predefined pre-war window and the clean regularized logistic stack elsewhere.",
            False,
        )
        specs.append(clean_hybrid_spec)
        frames.append(_candidate_frame(data, clean_hybrid_spec, clean_hybrid_probability))

        best_weight = None
        best_accuracy = -1.0
        for weight in np.linspace(0.0, 1.0, 101):
            probability = _fixed_blend(data, float(weight))
            accuracy = float(((probability >= 0.5).astype(int) == data["target"].astype(int)).mean())
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weight = float(weight)
        diagnostic_spec = CandidateSpec(
            f"diagnostic_full_sample_best_blend_{best_weight:.2f}",
            "diagnostic",
            "Full-sample best blend. This uses the whole saved prediction set and is not deployment-safe.",
            False,
        )
        specs.append(diagnostic_spec)
        frames.append(_candidate_frame(data, diagnostic_spec, _fixed_blend(data, best_weight)))

        predictions = pd.concat(frames, ignore_index=True).sort_values(["candidate", "date", "ticker"])
        table = _candidate_table(predictions, specs)
        return {
            "predictions": predictions,
            "candidate_table": table,
            "blend_selections": blend_selections,
            "event_gate_selections": {
                "crewai_event_gate_boosted_sent_agent": event_gate_sent_selections,
                "crewai_event_gate_boosted_selected_agent": event_gate_selected_selections,
                "crewai_validation_confidence_override_agent": confidence_override_selections,
                "crewai_ticker_calibrated_event_gate_agent": ticker_calibration_selections,
            },
            "specs": specs,
            "best_diagnostic_weight": best_weight,
            "diagnostic_ticker_thresholds": diagnostic_ticker_thresholds,
        }

    @listen(build_candidates)
    def save_results(self, payload: dict[str, Any]) -> dict[str, Any]:
        predictions: pd.DataFrame = payload["predictions"]
        candidate_table: pd.DataFrame = payload["candidate_table"]
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(OUTPUT_PREDICTIONS_PATH, index=False)
        candidate_table.to_csv(OUTPUT_CANDIDATES_PATH, index=False)

        deployment_candidates = candidate_table[candidate_table["deployment_safe"]].copy()
        full_coverage_candidates = deployment_candidates[deployment_candidates["overall_coverage_share"] >= 0.99].copy()
        high_coverage_candidates = deployment_candidates[deployment_candidates["overall_coverage_share"] >= 0.85].copy()
        selective_candidates = deployment_candidates[deployment_candidates["overall_coverage_share"] < 0.85].copy()
        same_amount_candidates = candidate_table[candidate_table["overall_coverage_share"] >= 0.99].copy()
        safe_same_amount_candidates = same_amount_candidates[same_amount_candidates["deployment_safe"]].copy()
        deployment_candidates = deployment_candidates.sort_values(
            ["overall_accuracy", "war_accuracy"],
            ascending=[False, False],
        )
        full_coverage_candidates = full_coverage_candidates.sort_values(
            ["overall_accuracy", "war_accuracy"],
            ascending=[False, False],
        )
        high_coverage_candidates = high_coverage_candidates.sort_values(
            ["overall_accuracy", "war_accuracy"],
            ascending=[False, False],
        )
        selective_candidates = selective_candidates.sort_values(
            ["overall_accuracy", "war_accuracy"],
            ascending=[False, False],
        )
        same_amount_candidates = same_amount_candidates.sort_values(
            ["overall_accuracy", "war_accuracy"],
            ascending=[False, False],
        )
        safe_same_amount_candidates = safe_same_amount_candidates.sort_values(
            ["overall_accuracy", "war_accuracy"],
            ascending=[False, False],
        )
        diagnostic_candidates = candidate_table.sort_values(
            ["overall_accuracy", "war_accuracy"],
            ascending=[False, False],
        )
        comparison_candidates = [
            ("Market LSTM", "benchmark", "saved_baseline_lstm"),
            ("Sentiment LSTM", "benchmark", "saved_sent_lstm"),
            ("Market-Sentiment Blend", "benchmark", "saved_selected_25_75_blend"),
            (
                "CrewAI Calibrated Router",
                "crewai_router",
                safe_same_amount_candidates.iloc[0]["candidate"] if not safe_same_amount_candidates.empty else "",
            ),
            (
                "CrewAI High-Confidence Router",
                "diagnostic_selective",
                selective_candidates.iloc[0]["candidate"] if not selective_candidates.empty else "",
            ),
            (
                "Post-Hoc Calibration Check",
                "diagnostic_post_hoc",
                same_amount_candidates.iloc[0]["candidate"] if not same_amount_candidates.empty else "",
            ),
        ]
        comparison_rows = []
        for model_label, role, candidate_name in comparison_candidates:
            if not candidate_name:
                continue
            match = candidate_table[candidate_table["candidate"] == candidate_name]
            if match.empty:
                continue
            row = match.iloc[0].to_dict()
            row["model"] = model_label
            row["source_candidate"] = candidate_name
            row["role"] = role
            comparison_rows.append(row)
        comparison = pd.DataFrame(comparison_rows)
        comparison_columns = [
            "model",
            "source_candidate",
            "role",
            "deployment_safe",
            "overall_accuracy",
            "overall_predictions",
            "overall_total_possible",
            "overall_coverage_share",
            "pre_war_accuracy",
            "pre_war_predictions",
            "pre_war_total_possible",
            "pre_war_coverage_share",
            "war_accuracy",
            "war_predictions",
            "war_total_possible",
            "war_coverage_share",
        ]
        comparison = comparison[comparison_columns]
        comparison.to_csv(OUTPUT_COMPARISON_CSV_PATH, index=False)

        summary = {
            "project_name": "crewai_lstm_routing_experiment",
            "orchestration": "CrewAI Flow",
            "purpose": (
                "Corrected CrewAI comparison using the saved LSTM benchmark outputs. "
                "This tests whether a routing or blending layer can improve the benchmark without retraining it."
            ),
            "current_selected_lstm_metrics": _load_json(SELECTED_METRICS_PATH),
            "current_raw_lstm_metrics": _load_json(RAW_METRICS_PATH),
            "best_deployment_candidate": high_coverage_candidates.iloc[0].to_dict() if not high_coverage_candidates.empty else {},
            "best_full_coverage_candidate": full_coverage_candidates.iloc[0].to_dict() if not full_coverage_candidates.empty else {},
            "best_same_amount_candidate": same_amount_candidates.iloc[0].to_dict() if not same_amount_candidates.empty else {},
            "best_safe_same_amount_candidate": safe_same_amount_candidates.iloc[0].to_dict()
            if not safe_same_amount_candidates.empty
            else {},
            "best_selective_candidate": selective_candidates.iloc[0].to_dict() if not selective_candidates.empty else {},
            "best_diagnostic_candidate": diagnostic_candidates.iloc[0].to_dict() if not diagnostic_candidates.empty else {},
            "candidate_metrics": {
                row["candidate"]: {
                    "overall": {
                        "accuracy": float(row["overall_accuracy"]),
                        "predictions": int(row["overall_predictions"]),
                        "total_possible": int(row["overall_total_possible"]),
                        "coverage_share": float(row["overall_coverage_share"]),
                    },
                    "pre_war": {
                        "accuracy": float(row["pre_war_accuracy"]),
                        "predictions": int(row["pre_war_predictions"]),
                        "total_possible": int(row["pre_war_total_possible"]),
                        "coverage_share": float(row["pre_war_coverage_share"]),
                    },
                    "war": {
                        "accuracy": float(row["war_accuracy"]),
                        "predictions": int(row["war_predictions"]),
                        "total_possible": int(row["war_total_possible"]),
                        "coverage_share": float(row["war_coverage_share"]),
                    },
                    "deployment_safe": bool(row["deployment_safe"]),
                    "family": row["family"],
                }
                for _, row in candidate_table.iterrows()
            },
            "blend_selections": payload["blend_selections"],
            "event_gate_selections": payload.get("event_gate_selections", {}),
            "diagnostic_ticker_thresholds": payload.get("diagnostic_ticker_thresholds", {}),
            "artifacts": {
                "predictions_path": str(OUTPUT_PREDICTIONS_PATH.relative_to(PROJECT_DIR)),
                "candidate_table_path": str(OUTPUT_CANDIDATES_PATH.relative_to(PROJECT_DIR)),
                "summary_path": str(OUTPUT_SUMMARY_PATH.relative_to(PROJECT_DIR)),
                "comparison_csv_path": str(OUTPUT_COMPARISON_CSV_PATH.relative_to(PROJECT_DIR)),
            },
        }
        summary = _json_safe(summary)
        _save_json(OUTPUT_SUMMARY_PATH, summary)
        return summary


def main() -> None:
    summary = CrewAILSTMRoutingFlow().kickoff()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
