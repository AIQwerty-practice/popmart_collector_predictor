from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "popmart_collectors.csv"
RANDOM_SEED = 42
ROW_COUNT = 2_000


def build_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    ages = rng.integers(16, 56, ROW_COUNT)
    monthly_budget = np.round(rng.gamma(shape=4.0, scale=18.0, size=ROW_COUNT) + 20, 2)
    monthly_budget = np.clip(monthly_budget, 20, 350)
    collection_size = rng.poisson(lam=18, size=ROW_COUNT) + rng.integers(0, 12, ROW_COUNT)
    collection_size = np.clip(collection_size, 0, 180)
    monthly_purchases = rng.poisson(lam=3.0, size=ROW_COUNT)
    monthly_purchases = np.clip(monthly_purchases, 0, 20)
    resale_interest = rng.integers(0, 11, ROW_COUNT)
    social_media_engagement = rng.integers(0, 11, ROW_COUNT)
    blind_box_risk_tolerance = rng.integers(0, 11, ROW_COUNT)

    favorite_series = rng.choice(
        ["Labubu", "Molly", "Skullpanda", "Dimoo", "Hirono", "Crybaby"],
        size=ROW_COUNT,
        p=[0.24, 0.18, 0.18, 0.16, 0.14, 0.10],
    )
    collector_type = rng.choice(
        ["casual", "display-focused", "completionist", "reseller", "trend-chaser"],
        size=ROW_COUNT,
        p=[0.30, 0.22, 0.20, 0.10, 0.18],
    )
    region = rng.choice(
        ["North America", "Europe", "East Asia", "Southeast Asia", "Oceania"],
        size=ROW_COUNT,
        p=[0.34, 0.18, 0.24, 0.18, 0.06],
    )

    series_boost = np.select(
        [
            favorite_series == "Labubu",
            favorite_series == "Skullpanda",
            favorite_series == "Molly",
            favorite_series == "Hirono",
        ],
        [0.55, 0.35, 0.20, 0.15],
        default=0.0,
    )
    type_boost = np.select(
        [
            collector_type == "completionist",
            collector_type == "trend-chaser",
            collector_type == "reseller",
            collector_type == "display-focused",
        ],
        [0.55, 0.45, 0.35, 0.10],
        default=-0.25,
    )
    region_boost = np.select(
        [region == "East Asia", region == "Southeast Asia", region == "North America"],
        [0.20, 0.16, 0.10],
        default=0.0,
    )

    score = (
        -3.3
        + monthly_budget / 95
        + collection_size / 55
        + monthly_purchases / 4.8
        + social_media_engagement / 5.8
        + blind_box_risk_tolerance / 6.0
        + resale_interest / 10.0
        + series_boost
        + type_boost
        + region_boost
        - np.maximum(ages - 38, 0) / 80
        + rng.normal(0, 0.55, ROW_COUNT)
    )
    probability = 1 / (1 + np.exp(-score))
    will_buy_next_release = (rng.random(ROW_COUNT) < probability).astype(int)

    return pd.DataFrame(
        {
            "age": ages,
            "monthly_budget_usd": monthly_budget,
            "collection_size": collection_size,
            "monthly_purchases": monthly_purchases,
            "resale_interest": resale_interest,
            "social_media_engagement": social_media_engagement,
            "blind_box_risk_tolerance": blind_box_risk_tolerance,
            "favorite_series": favorite_series,
            "collector_type": collector_type,
            "region": region,
            "will_buy_next_release": will_buy_next_release,
        }
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset()
    dataset.to_csv(DATA_PATH, index=False)
    print(f"Wrote {len(dataset):,} rows to {DATA_PATH}")


if __name__ == "__main__":
    main()
