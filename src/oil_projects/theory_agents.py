from __future__ import annotations

import numpy as np
import pandas as pd


def add_theory_agent_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()

    # Political-risk agent: salience of threats to oil supply from conflict and sanctions.
    enriched["agent_political_risk"] = (
        1.8 * enriched.get("gpr_threat_share", 0.0)
        + 1.2 * enriched.get("gpr_surprise", 0.0).clip(lower=0)
        + 1.5 * enriched.get("geopolitics_share", 0.0)
        + 1.1 * enriched.get("sanctions_share", 0.0)
        + 0.9 * enriched.get("shipping_share", 0.0)
    )

    # Company-exposure agent: how directly the firm is exposed to upstream and flow disruptions.
    enriched["agent_company_exposure"] = (
        1.3 * enriched.get("production_share", 0.0)
        + 1.2 * enriched.get("upstream_share", 0.0)
        + 1.1 * enriched.get("shipping_share", 0.0)
        + 0.8 * enriched.get("refining_share", 0.0)
        + 0.7 * enriched.get("filing_8k_share", 0.0)
        + 0.5 * enriched.get("novelty_mean", 0.0)
    )

    # Market-state agent: whether the market is pricing a supply shock already.
    enriched["agent_market_stress"] = (
        enriched.get("ovx_ret_1", 0.0).clip(lower=0)
        + 0.6 * enriched.get("ovx_ret_5", 0.0).clip(lower=0)
        + 0.8 * enriched.get("uso_ret_1", 0.0).abs()
        + 0.8 * enriched.get("xop_ret_1", 0.0).abs()
    )

    # Disclosure agent: whether the filing itself has unusual event content.
    enriched["agent_disclosure_signal"] = (
        enriched.get("avg_sentiment", 0.0) * (1 + enriched.get("novelty_mean", 0.0))
        + 0.8 * enriched.get("supply_disruption_share", 0.0)
        + 0.7 * enriched.get("geopolitics_share", 0.0)
        + 0.5 * enriched.get("negative_share", 0.0)
        - 0.5 * enriched.get("positive_share", 0.0)
    )

    enriched["agent_war_alignment"] = (
        enriched["agent_political_risk"]
        * (1 + enriched["agent_company_exposure"])
        * (1 + enriched["agent_market_stress"])
    )
    enriched["agent_supply_alignment"] = (
        enriched.get("supply_chain_intensity", 0.0)
        + enriched.get("supply_risk_intensity", 0.0)
        + enriched["agent_company_exposure"] * (1 + enriched.get("shipping_share", 0.0))
    )
    enriched["agent_text_market_confirmation"] = (
        enriched["agent_disclosure_signal"] * (1 + enriched["agent_market_stress"])
    )
    enriched["agent_uncertainty_confirmation"] = (
        enriched["agent_political_risk"] * (1 + enriched.get("gpr_level", 0.0))
        + enriched["agent_market_stress"] * (1 + enriched.get("gpr_threat_share", 0.0))
    )

    return enriched.replace([np.inf, -np.inf], 0.0).fillna(0.0)

