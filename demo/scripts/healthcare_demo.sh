#!/usr/bin/env bash
# Playbook C — Healthcare clinical decision support agent
#
# 5-act narrative:
#   1. The fear: agents in clinical context
#   2. The mitigations: tracing, human-in-the-loop, sandboxing
#   3. PHI redaction in flight
#   4. Confidence-based escalation in flight
#   5. The audit trail meets HIPAA + SaMD requirements
#
# Run from repo root:
#   source demo/envs/healthcare.env
#   bash demo/scripts/healthcare_demo.sh

set -euo pipefail

CONFIG=${SENTINEL_CONFIG_PATH:-./demo/configs/clinical_decision.yaml}
DATA=./demo/data/healthcare

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

say "Act 0 — generate synthetic clinical cases (no real PHI, ever)"
python demo/data/generate_healthcare_data.py
pause

say "Act 1 — show the safety section of the config"
awk '/safety:/,/agent_registry:/' "$CONFIG"
pause

say "Act 2 — validate the agent config"
sentinel validate --config "$CONFIG"
pause

say "Act 3 — process a CLEAN case (should run end-to-end)"
python demo/agents/low_confidence_agent.py --case C-3001 --config "$CONFIG" || true
pause

say "Act 4 — process a LOW-CONFIDENCE case (should escalate to clinician)"
python demo/agents/low_confidence_agent.py --case C-3003 --config "$CONFIG" || true
pause

say "Act 5 — status: traces stored, escalations counted, retention 7 years"
sentinel status --config "$CONFIG"
pause

say "Act 6 — audit trail filtered to escalations"
sentinel audit --config "$CONFIG" --type human_handoff --limit 10
pause

say "Demo complete. Talking points:"
cat <<'EOF'
  • PHI is hashed before any LLM call. The provider never sees it.
  • Loop ceiling is 30 iterations (tighter than default for clinical context).
  • Confidence below 0.70 → automatic clinician handoff. No exceptions.
  • Every reasoning step is in the audit trail. 7-year retention.
  • The agent is incapable of writing to the EHR — tool permissions enforced.
EOF
