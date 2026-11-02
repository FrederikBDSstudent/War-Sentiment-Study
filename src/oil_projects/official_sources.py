from __future__ import annotations

import pandas as pd

from oil_projects.config import ARTIFACT_DIR


OFFICIAL_EVENT_ROWS = [
    {
        "date": "2025-02-04",
        "source_type": "white_house",
        "event_type": "sanctions_policy",
        "weight": 1.2,
        "title": "NSPM-2 restores maximum pressure on Iran",
        "url": "https://www.whitehouse.gov/presidential-actions/2025/02/national-security-presidential-memorandum-nspm-2/",
    },
    {
        "date": "2025-02-24",
        "source_type": "treasury",
        "event_type": "sanctions_oil",
        "weight": 1.1,
        "title": "Treasury imposes additional sanctions on Iran's shadow fleet",
        "url": "https://home.treasury.gov/news/press-releases/sb0026",
    },
    {
        "date": "2025-03-13",
        "source_type": "treasury",
        "event_type": "sanctions_oil",
        "weight": 1.2,
        "title": "Treasury sanctions Iranian oil minister and shadow fleet operators",
        "url": "https://home.treasury.gov/news/press-releases/sb0049",
    },
    {
        "date": "2025-06-12",
        "source_type": "white_house",
        "event_type": "conflict_escalation",
        "weight": 1.4,
        "title": "White House statement after Israeli action against Iran",
        "url": "https://www.whitehouse.gov/briefings-statements/2025/06/statement-from-secretary-of-state-marco-rubio/",
    },
    {
        "date": "2025-06-20",
        "source_type": "treasury",
        "event_type": "shipping_enforcement",
        "weight": 1.0,
        "title": "Treasury sanctions Houthi illicit oil trading and shipping",
        "url": "https://home.treasury.gov/news/press-releases/sb0174",
    },
    {
        "date": "2025-07-05",
        "source_type": "opec",
        "event_type": "opec_supply_policy",
        "weight": 0.9,
        "title": "OPEC+ adjusts production and reaffirms market stability",
        "url": "https://www.opec.org/pr-detail/569-05-july-2025.html",
    },
]

def load_official_event_frame() -> pd.DataFrame:
    frame = pd.DataFrame(OFFICIAL_EVENT_ROWS)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date").reset_index(drop=True)


def build_official_event_features() -> pd.DataFrame:
    events = load_official_event_frame()
    features = pd.DataFrame({"date": pd.date_range(events["date"].min(), events["date"].max(), freq="D")})
    features = features.merge(events[["date", "weight"]], on="date", how="left")
    features["official_event_weight"] = features["weight"].fillna(0.0)
    features = features.drop(columns="weight")

    for event_type in sorted(events["event_type"].unique()):
        event_dates = events.loc[events["event_type"] == event_type, ["date", "weight"]].rename(columns={"weight": f"official_{event_type}_weight"})
        features = features.merge(event_dates, on="date", how="left")
        features[f"official_{event_type}_weight"] = features[f"official_{event_type}_weight"].fillna(0.0)

    features["official_event_count"] = (features["official_event_weight"] > 0).astype(float)
    features["official_event_weight_5d"] = features["official_event_weight"].rolling(5, min_periods=1).sum()
    for column in [name for name in features.columns if name.startswith("official_") and name.endswith("_weight")]:
        features[f"{column}_5d"] = features[column].rolling(5, min_periods=1).sum()

    events.to_csv(ARTIFACT_DIR / "official_event_sources.csv", index=False)
    return features
