#!/usr/bin/env bash
# Playbook F — Quant / asset manager volatility forecasting
#
# 5-act narrative:
#   1. PSI is wrong for time series — show why
#   2. Calendar-aware drift: same period vs same period
#   3. STL decomposition monitoring catches the regime shift
#   4. Per-horizon quality (1d, 5d, 22d, 63d)
#   5. SR 11-7 model risk management compliance package
#
# Run from repo root:
#   source demo/envs/quant.env
#   bash demo/scripts/quant_demo.sh

set -euo pipefail

CONFIG=${SENTINEL_CONFIG_PATH:-./demo/configs/volatility_forecast.yaml}
DATA=./demo/data/quant

say() { printf "\n\033[1;36m▸ %s\033[0m\n" "$*"; }
pause() { printf "\033[2m  (press enter)\033[0m"; read -r _; }

# Optional: launch the Sentinel dashboard alongside the demo.
# Enable with `export SENTINEL_DEMO_DASHBOARD=1` before running.
if [ "${SENTINEL_DEMO_DASHBOARD:-0}" = "1" ]; then
  PORT=${SENTINEL_DEMO_DASHBOARD_PORT:-8000}
  say "Starting Sentinel dashboard on http://127.0.0.1:${PORT} (background)"
  sentinel dashboard --config "$CONFIG" --port "$PORT" >/tmp/sentinel-dashboard.log 2>&1 &
  DASHBOARD_PID=$!
  trap 'kill $DASHBOARD_PID 2>/dev/null || true' EXIT
  sleep 2
  printf "  → open http://127.0.0.1:%s and follow each act in the browser\n" "$PORT"
  pause
fi

say "Act 0 — generate synthetic SPX volatility series"
python demo/data/generate_quant_data.py
pause

say "Act 1 — show the timeseries domain section"
awk '/^domains:/,/^feature_health:/' "$CONFIG"
pause

say "Act 2 — validate"
sentinel validate --config "$CONFIG"
pause

say "Act 3 — drift check on a CLEAN month (same regime as baseline)"
sentinel check --config "$CONFIG" \
  --reference "$DATA/baseline.parquet" \
  --current "$DATA/today_clean.parquet"
pause

say "Act 4 — drift check on a REGIME-SHIFT month (vol +60%, trend break)"
sentinel check --config "$CONFIG" \
  --reference "$DATA/baseline.parquet" \
  --current "$DATA/today_regime_shift.parquet"
pause

say "Act 5 — status (per-horizon quality, decomposition, ADF stationarity)"
sentinel status --config "$CONFIG"
pause

say "Act 6 — registry: SR 11-7 requires full lineage on every champion change"
sentinel registry list --config "$CONFIG"
pause

say "Act 7 — audit trail (7 year retention, no auto-promote)"
sentinel audit --config "$CONFIG" --limit 10
pause

say "Demo complete. Talking points:"
cat <<'EOF'
  • Standard PSI on time series is wrong — it confuses seasonality with drift.
  • Calendar test compares Mar 2026 vs Mar 2025, not Mar 2026 vs the global baseline.
  • STL decomposition tracks trend / seasonal / residual independently.
    Trend slope change → regime alert.
  • Per-horizon quality shows where the model loses skill (e.g. fine at 1d,
    useless at 63d). Surfaces the model's true useful range.
  • Auto-promote is OFF. SR 11-7 requires human model risk approval.
  • 7-year audit retention. Full SHAP explanations on every prediction logged.
EOF
