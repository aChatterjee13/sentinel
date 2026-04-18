#!/usr/bin/env bash
# Playbook A — UK regulated bank credit decisioning demo
#
# 5-act narrative:
#   1. The pain — 5 tools, weeks of plumbing
#   2. The one config file (40 lines)
#   3. Drift fires, alert fires, audit log appears
#   4. Compliance — show the audit/registry trail
#   5. The close — 7-year retention by default
#
# Run from repo root:
#   source demo/envs/bank.env
#   bash demo/scripts/bank_demo.sh

set -euo pipefail

CONFIG=${SENTINEL_CONFIG_PATH:-./demo/configs/credit_decisioning.yaml}
DATA=./demo/data/bank

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

say "Act 0 — generate synthetic origination data"
python demo/data/generate_bank_data.py
pause

say "Act 1 — show the config (this is the whole governance model)"
sed -n '1,40p' "$CONFIG"
pause

say "Act 2 — validate the config (fails fast with clear errors)"
sentinel validate --config "$CONFIG"
pause

say "Act 3 — fit baseline + run drift check on a CLEAN window"
sentinel check --config "$CONFIG" \
  --reference "$DATA/baseline.parquet" \
  --current "$DATA/today_clean.parquet"
pause

say "Act 4 — same model, drifted window (cost-of-living shock)"
sentinel check --config "$CONFIG" \
  --reference "$DATA/baseline.parquet" \
  --current "$DATA/today_drifted.parquet"
pause

say "Act 5 — show current status (registry + drift + audit retention)"
sentinel status --config "$CONFIG"
pause

say "Act 6 — show the audit trail (every event, immutable)"
sentinel audit --config "$CONFIG" --type drift_detected --limit 5
pause

say "Act 7 — promote a new version with canary (5% → 25% → 50% → 100%)"
sentinel deploy --config "$CONFIG" --version 3.2.2 --strategy canary --traffic 5
pause

say "Demo complete. Talking points:"
cat <<'EOF'
  • One YAML file. 40 lines. FCA Consumer Duty + PRA SS1/23 ready.
  • 7-year audit retention is the DEFAULT for this config.
  • Auto-rollback wired to error_rate_increase + latency_p99_increase.
  • Cascade alerts to pricing_engine and collections_priority.
  • Every step you just saw is in the audit trail. Show it to the regulator tomorrow.
EOF
