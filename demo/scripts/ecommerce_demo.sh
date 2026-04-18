#!/usr/bin/env bash
# Playbook D — E-commerce product recommendation
#
# 5-act narrative:
#   1. NDCG isn't enough — show beyond-accuracy metrics
#   2. The long-tail collapse failure mode
#   3. Drift fires — coverage drops below threshold
#   4. Group fairness check across segments
#   5. Auto-rollback wired to revenue per session
#
# Run from repo root:
#   source demo/envs/ecommerce.env
#   bash demo/scripts/ecommerce_demo.sh

set -euo pipefail

CONFIG=${SENTINEL_CONFIG_PATH:-./demo/configs/product_reco.yaml}
DATA=./demo/data/ecommerce

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

say "Act 0 — generate synthetic interactions and recommendations"
python demo/data/generate_ecommerce_data.py
pause

say "Act 1 — show the recommendation domain section (beyond-accuracy metrics)"
awk '/^domains:/,/^cost_monitor:/' "$CONFIG"
pause

say "Act 2 — validate"
sentinel validate --config "$CONFIG"
pause

say "Act 3 — drift check on a CLEAN recommendation window"
sentinel check --config "$CONFIG" \
  --reference "$DATA/baseline_interactions.parquet" \
  --current "$DATA/today_clean_recos.parquet"
pause

say "Act 4 — drift check on a COLLAPSED window (top-1% items take everything)"
sentinel check --config "$CONFIG" \
  --reference "$DATA/baseline_interactions.parquet" \
  --current "$DATA/today_drifted_recos.parquet"
pause

say "Act 5 — status: coverage, diversity, popularity bias, fairness"
sentinel status --config "$CONFIG"
pause

say "Act 6 — promote the new model with revenue-aware canary"
sentinel deploy --config "$CONFIG" --version 5.0.4 --strategy canary --traffic 10
pause

say "Demo complete. Talking points:"
cat <<'EOF'
  • Vendors give you NDCG. Sentinel also gives you coverage, diversity,
    novelty, popularity bias and group fairness on the same dashboard.
  • Long-tail collapse is the silent killer of recommender quality.
    You just saw it caught automatically.
  • Group fairness gap > 10% across user segments → alert.
    Defensible to a regulator.
  • Auto-rollback wired to revenue per session, not just model error.
EOF
