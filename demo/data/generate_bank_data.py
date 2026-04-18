"""Generate synthetic credit decisioning datasets for Playbook A.

Produces three parquet files under ./demo/data/bank/:
  - baseline.parquet   — 12 months of "normal" origination data
  - today_clean.parquet — a current window with no drift
  - today_drifted.parquet — a current window with income + LTV drift

Usage:
    python demo/data/generate_bank_data.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
OUT = Path(__file__).parent / "bank"
OUT.mkdir(parents=True, exist_ok=True)


def _sample(n: int, *, shift: bool, rng: np.random.Generator) -> pd.DataFrame:
    # Baseline feature distributions (UK personal loans book)
    income = rng.lognormal(mean=10.6, sigma=0.55, size=n)           # ~£40k median
    age = rng.normal(loc=42, scale=12, size=n).clip(18, 80)
    ltv = rng.beta(a=2.5, b=2.0, size=n) * 100                      # 0..100
    credit_score = rng.normal(loc=680, scale=60, size=n).clip(300, 850)
    months_employed = rng.gamma(shape=3.0, scale=18.0, size=n).clip(0, 600)
    existing_debt = rng.gamma(shape=2.0, scale=4000.0, size=n)

    if shift:
        # Cost-of-living shock: lower incomes, higher LTV, higher existing debt
        income *= 0.85
        ltv = np.clip(ltv + 12, 0, 100)
        existing_debt *= 1.35
        credit_score -= 25

    default_logit = (
        -3.0
        + 0.00002 * (60000 - income)
        + 0.04 * (ltv - 60)
        + 0.00005 * existing_debt
        - 0.008 * (credit_score - 650)
    )
    default_prob = 1 / (1 + np.exp(-default_logit))
    default_flag = (rng.uniform(size=n) < default_prob).astype(int)

    return pd.DataFrame(
        {
            "application_id": [f"APP-{i:07d}" for i in range(n)],
            "income_gbp": income.round(2),
            "age": age.round(0).astype(int),
            "ltv_pct": ltv.round(2),
            "credit_score": credit_score.round(0).astype(int),
            "months_employed": months_employed.round(0).astype(int),
            "existing_debt_gbp": existing_debt.round(2),
            "default_flag": default_flag,
        }
    )


def main() -> None:
    rng = np.random.default_rng(SEED)
    baseline = _sample(12_000, shift=False, rng=rng)
    today_clean = _sample(2_000, shift=False, rng=rng)
    today_drifted = _sample(2_000, shift=True, rng=rng)

    baseline.to_parquet(OUT / "baseline.parquet", index=False)
    today_clean.to_parquet(OUT / "today_clean.parquet", index=False)
    today_drifted.to_parquet(OUT / "today_drifted.parquet", index=False)

    print(f"[bank] wrote {len(baseline)} baseline rows to {OUT / 'baseline.parquet'}")
    print(f"[bank] wrote {len(today_clean)} clean rows to {OUT / 'today_clean.parquet'}")
    print(f"[bank] wrote {len(today_drifted)} drifted rows to {OUT / 'today_drifted.parquet'}")


if __name__ == "__main__":
    main()
