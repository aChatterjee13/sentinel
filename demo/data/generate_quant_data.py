"""Generate synthetic equity volatility data for Playbook F.

Produces parquet files under ./demo/data/quant/:
  - baseline.parquet        — 3 years of normal-regime daily volatility
  - today_clean.parquet     — recent month that follows the baseline regime
  - today_regime_shift.parquet — recent month with a clear regime change
                                   (higher mean volatility + trend break)

Usage:
    python demo/data/generate_quant_data.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
OUT = Path(__file__).parent / "quant"
OUT.mkdir(parents=True, exist_ok=True)

TRADING_DAYS_YEAR = 252


def _series(n: int, *, regime_shift: bool, rng: np.random.Generator,
            start_date: str) -> pd.DataFrame:
    # Baseline vol: GARCH-like — weak autocorrelation around 0.15
    vol = np.zeros(n)
    vol[0] = 0.16
    for t in range(1, n):
        mean_reversion = 0.05 * (0.16 - vol[t - 1])
        shock = rng.normal(0, 0.012)
        vol[t] = max(0.05, vol[t - 1] + mean_reversion + shock)

    # Weekly + yearly seasonality (day-of-week and calendar effects)
    days = np.arange(n)
    weekly = 0.005 * np.sin(2 * np.pi * days / 5)
    yearly = 0.01 * np.sin(2 * np.pi * days / TRADING_DAYS_YEAR)
    vol = vol + weekly + yearly

    if regime_shift:
        # Step change: +60% mean vol, +40% variance, drift in trend
        vol = vol * 1.6 + np.linspace(0, 0.08, n) + rng.normal(0, 0.006, size=n)

    dates = pd.date_range(start=start_date, periods=n, freq="B")
    return pd.DataFrame({
        "date": dates,
        "ticker": "SPX",
        "realised_vol_1d": vol.round(5),
        "log_return": rng.normal(0, vol / np.sqrt(TRADING_DAYS_YEAR)).round(5),
    })


def main() -> None:
    rng = np.random.default_rng(SEED)
    baseline = _series(3 * TRADING_DAYS_YEAR, regime_shift=False, rng=rng,
                       start_date="2023-01-03")
    today_clean = _series(22, regime_shift=False, rng=rng, start_date="2026-03-02")
    today_shift = _series(22, regime_shift=True, rng=rng, start_date="2026-03-02")

    baseline.to_parquet(OUT / "baseline.parquet", index=False)
    today_clean.to_parquet(OUT / "today_clean.parquet", index=False)
    today_shift.to_parquet(OUT / "today_regime_shift.parquet", index=False)

    print(f"[quant] baseline: {len(baseline)} rows, "
          f"mean vol {baseline['realised_vol_1d'].mean():.4f}")
    print(f"[quant] today clean: mean vol {today_clean['realised_vol_1d'].mean():.4f}")
    print(f"[quant] today regime shift: mean vol {today_shift['realised_vol_1d'].mean():.4f}")


if __name__ == "__main__":
    main()
