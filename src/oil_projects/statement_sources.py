from __future__ import annotations

import pandas as pd

from oil_projects.config import ARTIFACT_DIR


STATEMENT_EVENT_ROWS = [
    {
        "date": "2025-02-04",
        "source_type": "white_house",
        "event_type": "maximum_pressure",
        "weight": 1.0,
        "strike_weight": 0.0,
        "shipping_weight": 0.0,
        "hormuz_weight": 0.0,
        "sanctions_weight": 1.0,
        "title": "White House restores maximum pressure on Iran",
        "url": "https://www.whitehouse.gov/fact-sheets/2025/02/fact-sheet-president-donald-j-trump-restores-maximum-pressure-on-iran/",
    },
    {
        "date": "2025-02-24",
        "source_type": "treasury",
        "event_type": "shadow_fleet_sanctions",
        "weight": 1.1,
        "strike_weight": 0.0,
        "shipping_weight": 0.4,
        "hormuz_weight": 0.0,
        "sanctions_weight": 1.0,
        "title": "Treasury targets Iran shadow fleet",
        "url": "https://home.treasury.gov/news/press-releases/sb0026",
    },
    {
        "date": "2025-03-13",
        "source_type": "treasury",
        "event_type": "oil_minister_sanctions",
        "weight": 1.1,
        "strike_weight": 0.0,
        "shipping_weight": 0.2,
        "hormuz_weight": 0.0,
        "sanctions_weight": 1.0,
        "title": "Treasury sanctions Iranian oil minister and fleet operators",
        "url": "https://home.treasury.gov/news/press-releases/sb0049",
    },
    {
        "date": "2025-06-13",
        "source_type": "cnbc",
        "event_type": "strike_escalation",
        "weight": 1.5,
        "strike_weight": 1.0,
        "shipping_weight": 0.2,
        "hormuz_weight": 0.3,
        "sanctions_weight": 0.0,
        "title": "Oil prices surge after Israeli strikes on Iran",
        "url": "https://www.cnbc.com/2025/06/13/asia-pacific-markets-live-us-china-trade.html",
    },
    {
        "date": "2025-06-22",
        "source_type": "cnbc",
        "event_type": "hormuz_closure_threat",
        "weight": 1.7,
        "strike_weight": 0.2,
        "shipping_weight": 0.8,
        "hormuz_weight": 1.0,
        "sanctions_weight": 0.0,
        "title": "Iran parliament backs blocking Strait of Hormuz",
        "url": "https://www.cnbc.com/2025/06/23/irans-parliament-approves-blocking-strait-of-hormuz.html",
    },
    {
        "date": "2025-06-20",
        "source_type": "treasury",
        "event_type": "shipping_enforcement",
        "weight": 1.0,
        "strike_weight": 0.0,
        "shipping_weight": 1.0,
        "hormuz_weight": 0.4,
        "sanctions_weight": 0.8,
        "title": "Treasury sanctions Houthi oil trading and shipping",
        "url": "https://home.treasury.gov/news/press-releases/sb0174",
    },
    {
        "date": "2025-07-02",
        "source_type": "reuters_reported",
        "event_type": "mine_preparation_alarm",
        "weight": 1.4,
        "strike_weight": 0.0,
        "shipping_weight": 0.8,
        "hormuz_weight": 1.0,
        "sanctions_weight": 0.0,
        "title": "Reuters-reported US alarm over Iranian mine preparations near Hormuz",
        "url": "https://www.iranintl.com/en/202507029554",
    },
    {
        "date": "2025-07-05",
        "source_type": "opec",
        "event_type": "opec_supply_policy",
        "weight": 0.8,
        "strike_weight": 0.0,
        "shipping_weight": 0.0,
        "hormuz_weight": 0.0,
        "sanctions_weight": 0.0,
        "title": "OPEC+ reaffirms market stability",
        "url": "https://www.opec.org/pr-detail/569-05-july-2025.html",
    },
]


def load_statement_event_frame() -> pd.DataFrame:
    frame = pd.DataFrame(STATEMENT_EVENT_ROWS)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date").reset_index(drop=True)


def build_statement_event_features() -> pd.DataFrame:
    events = load_statement_event_frame()
    features = pd.DataFrame({"date": pd.date_range(events["date"].min(), events["date"].max(), freq="D")})
    features = features.merge(
        events[
            [
                "date",
                "weight",
                "strike_weight",
                "shipping_weight",
                "hormuz_weight",
                "sanctions_weight",
            ]
        ].groupby("date", as_index=False).sum(),
        on="date",
        how="left",
    ).fillna(0.0)
    features = features.rename(
        columns={
            "weight": "statement_event_weight",
            "strike_weight": "statement_strike_weight",
            "shipping_weight": "statement_shipping_weight",
            "hormuz_weight": "statement_hormuz_weight",
            "sanctions_weight": "statement_sanctions_weight",
        }
    )
    features["statement_event_count"] = (features["statement_event_weight"] > 0).astype(float)
    for column in [
        "statement_event_weight",
        "statement_strike_weight",
        "statement_shipping_weight",
        "statement_hormuz_weight",
        "statement_sanctions_weight",
    ]:
        features[f"{column}_3d"] = features[column].rolling(3, min_periods=1).sum()
        features[f"{column}_5d"] = features[column].rolling(5, min_periods=1).sum()

    load_statement_event_frame().to_csv(ARTIFACT_DIR / "statement_event_sources.csv", index=False)
    return features
