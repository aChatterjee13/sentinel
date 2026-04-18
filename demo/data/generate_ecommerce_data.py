"""Generate synthetic interaction + recommendation data for Playbook D.

Produces parquet files under ./demo/data/ecommerce/:
  - baseline_interactions.parquet — 6 weeks of clicks/purchases (broad catalogue)
  - today_clean_recos.parquet     — recommendations covering the long tail
  - today_drifted_recos.parquet   — recommendations collapsed onto top-1% items
                                     (the long-tail collapse failure mode)

Usage:
    python demo/data/generate_ecommerce_data.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
OUT = Path(__file__).parent / "ecommerce"
OUT.mkdir(parents=True, exist_ok=True)

CATALOGUE_SIZE = 5_000
USER_SIZE = 2_000


def _baseline_interactions(rng: np.random.Generator) -> pd.DataFrame:
    n = 80_000
    # Zipf-distributed item popularity but the long tail still gets traffic
    item_id = rng.zipf(a=1.4, size=n)
    item_id = np.clip(item_id, 1, CATALOGUE_SIZE)
    user_id = rng.integers(low=1, high=USER_SIZE + 1, size=n)
    interaction_type = rng.choice(
        ["click", "click", "click", "add_to_cart", "purchase"], size=n
    )
    user_segment = rng.choice(["new", "returning", "loyal", "vip"], size=n,
                              p=[0.3, 0.4, 0.2, 0.1])
    return pd.DataFrame({
        "user_id": user_id,
        "product_id": item_id,
        "interaction_type": interaction_type,
        "user_segment": user_segment,
    })


def _recos(rng: np.random.Generator, *, collapse: bool) -> pd.DataFrame:
    n = 20_000
    if collapse:
        # All recommendations come from top-1% items: long-tail collapse
        top = max(50, CATALOGUE_SIZE // 100)
        item_id = rng.integers(low=1, high=top + 1, size=n)
    else:
        item_id = rng.zipf(a=1.6, size=n)
        item_id = np.clip(item_id, 1, CATALOGUE_SIZE)
    user_id = rng.integers(low=1, high=USER_SIZE + 1, size=n)
    user_segment = rng.choice(["new", "returning", "loyal", "vip"], size=n,
                              p=[0.3, 0.4, 0.2, 0.1])
    rank = rng.integers(low=1, high=11, size=n)
    return pd.DataFrame({
        "user_id": user_id,
        "product_id": item_id,
        "user_segment": user_segment,
        "rank": rank,
    })


def main() -> None:
    rng = np.random.default_rng(SEED)
    baseline = _baseline_interactions(rng)
    clean = _recos(rng, collapse=False)
    drifted = _recos(rng, collapse=True)

    baseline.to_parquet(OUT / "baseline_interactions.parquet", index=False)
    clean.to_parquet(OUT / "today_clean_recos.parquet", index=False)
    drifted.to_parquet(OUT / "today_drifted_recos.parquet", index=False)

    unique_clean = clean["product_id"].nunique()
    unique_drifted = drifted["product_id"].nunique()
    coverage_clean = unique_clean / CATALOGUE_SIZE
    coverage_drifted = unique_drifted / CATALOGUE_SIZE
    print(f"[ecommerce] baseline interactions: {len(baseline)}")
    print(f"[ecommerce] clean recos coverage: {coverage_clean:.1%}")
    print(f"[ecommerce] drifted recos coverage: {coverage_drifted:.1%}  <- demo target")


if __name__ == "__main__":
    main()
