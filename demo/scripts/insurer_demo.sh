#!/usr/bin/env bash
# Playbook B — Insurance modernizer: tabular fraud + RAG claims with full LLMOps
#
# 5-act narrative:
#   1. Same SDK, same audit trail — for tabular AND for RAG
#   2. Show the LLMOps section of the config (PII / jailbreak / topic / token)
#   3. Run a batch of clean queries — they pass
#   4. Run the attack queries — guardrails catch each one
#   5. Show the audit trail with guardrail violations recorded
#
# Run from repo root:
#   source demo/envs/insurer.env
#   bash demo/scripts/insurer_demo.sh

set -euo pipefail

CONFIG=${SENTINEL_CONFIG_PATH:-./demo/configs/claims_rag.yaml}
DATA=./demo/data/insurer

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

say "Act 0 — generate synthetic claims corpus and queries"
python demo/data/generate_insurer_data.py
pause

say "Act 1 — validate the RAG config"
sentinel validate --config "$CONFIG"
pause

say "Act 2 — show the LLMOps guardrails section"
awk '/^llmops:/,/^audit:/' "$CONFIG"
pause

say "Act 3 — run CLEAN queries through the guardrail pipeline"
python demo/agents/rag_demo.py --queries "$DATA/queries_clean.jsonl" --config "$CONFIG"
pause

say "Act 4 — run ATTACK queries (PII + jailbreak + off-topic)"
python demo/agents/rag_demo.py --queries "$DATA/queries_attacks.jsonl" --config "$CONFIG"
pause

say "Act 5 — status (drift + token economics + guardrail violation rate)"
sentinel status --config "$CONFIG"
pause

say "Act 6 — audit trail of guardrail blocks"
sentinel audit --config "$CONFIG" --type guardrail_violation --limit 10
pause

say "Demo complete. Talking points:"
cat <<'EOF'
  • Same SDK governs your tabular fraud model AND your RAG claims pipeline.
  • PII never reaches the LLM provider. Compliance breathes again.
  • Token budgets are enforced at $0.50/query, $500/day. Cost regression alerts on day 1.
  • Groundedness check on every output — hallucination defence.
  • Audit trail has the same shape as your tabular models.
    One process for two world classes of model.
EOF
