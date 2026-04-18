# Sentinel for Quants and Asset Managers

**Calendar-aware drift. Per-horizon forecast quality. SR 11-7 audit package. One config.**

---

## The problem we heard from you

- Your drift tool treats every day as independent and screams at you every
  January. It has no idea about seasonality.
- You have 40 forecasting models in production and no consistent way to
  track per-horizon skill (useful at 1d, useless at 63d — and you don't
  know which is which).
- SR 11-7 requires full lineage. Today you build that lineage by hand at
  audit time. It takes 2 weeks per model.
- Regime shifts should trigger a full model review. Today they trigger a
  slack message that gets lost.
- Every model risk committee meeting spends 90% of its time reconstructing
  "what changed". The other 10% is decisions.

## What Sentinel does about it

| Capability | How Sentinel delivers |
|---|---|
| Calendar-aware drift | `drift.method: calendar_test, compare_against: same_period` |
| STL decomposition monitoring | Trend / seasonal / residual tracked independently |
| Stationarity tests | ADF on sliding windows, `adf_significance: 0.05` |
| Per-horizon quality | `horizon_tracking: [1, 5, 22, 63]` — knows where skill ends |
| MASE + Winkler + coverage | `metrics: [mase, coverage, directional_accuracy, winkler]` |
| No auto-promote | `approval.mode: human_in_loop` — model risk gate |
| 7-year audit retention | `retention_days: 2555` with SHAP explanations |
| SR 11-7 lineage pack | `compliance_frameworks: [sr_11_7, pra_ss1_23, eu_ai_act]` |
| Shadow-first deployment | `strategy: shadow, duration: 336h` — 14 days paper-trading parity |

## What you saw in the demo

1. A clean forward month passed cleanly — because the calendar test
   compared March 2026 against March 2025, not against a global average.
2. A **regime-shifted month** (vol +60%, trend break) tripped three
   independent alerts:
   - Calendar drift
   - STL trend-slope change
   - ADF stationarity loss
3. Per-horizon quality showed the model holding at 1-day and 5-day but
   degrading at 22-day and 63-day — exactly the type of surface a model
   risk committee wants to see.
4. The audit trail has the full lineage: data → training → model → baseline
   → prediction → drift → decision. Exportable as an SR 11-7 evidence pack.

## Commercial headline

- **Seasonality-aware drift** — no more false alarms in January.
- **Per-horizon quality tracking** — the first thing your model risk
  committee asks for is now on the dashboard.
- **Full SR 11-7 lineage pack** exported with one command.
- **Shadow-first deployment** with 14-day paper-trading parity required.
- **No auto-promote, ever** — human model risk approval is enforced.

## What we need from you to run a pilot

1. One forecasting model currently in production — we prefer a model that
   has had a false positive from your current drift tool in the last 6 months.
2. Read-only access to a backfill of that model's inputs and outputs.
3. A named model owner + a named model risk reviewer.
4. 3 weeks of working sessions with one quant + one model risk analyst.
5. At the end: a side-by-side of your current drift tool vs Sentinel on
   the same 12 months of history, plus one SR 11-7 evidence pack.

## The ask

A 1-hour session with your head of model risk. We walk through the
backfill comparison and you tell us which of the 40 models you'd pilot on.

---

*Contact:* your Zensar account team • *Demo repo:* `demo/scripts/quant_demo.sh`
