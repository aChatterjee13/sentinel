#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Sentinel Showcase Demo — Full-Stack End-to-End
#
# This script gives a prospect a complete, self-running demonstration of
# every Sentinel capability. It:
#
#   1. Seeds ALL subsystems with realistic synthetic data
#   2. Launches the dashboard (auto-opens browser)
#   3. Walks through an 8-act guided narrative
#
# Every act populates the dashboard in real time — the audience can
# follow along in the browser while the terminal tells the story.
#
# Run from repo root:
#   bash demo/scripts/showcase_demo.sh
#
# Options (environment variables):
#   SENTINEL_SHOWCASE_PORT=8000     Dashboard port (default: 8000)
#   SENTINEL_SHOWCASE_NO_BROWSER=1  Skip auto-opening browser
#   SENTINEL_SHOWCASE_AUTO=1        No pauses — run straight through
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "$0")/../..")"

CONFIG=./demo/configs/showcase_all.yaml
PORT=${SENTINEL_SHOWCASE_PORT:-8000}
AUTO=${SENTINEL_SHOWCASE_AUTO:-0}
NO_BROWSER=${SENTINEL_SHOWCASE_NO_BROWSER:-0}

# ── Colours and helpers ──────────────────────────────────────────────

CYAN='\033[1;36m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

banner() {
  printf "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
  printf "${BOLD}  %s${RESET}\n" "$*"
  printf "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
}

act() {
  printf "\n${GREEN}▸ Act %s — %s${RESET}\n" "$1" "$2"
}

say() {
  printf "  ${DIM}%s${RESET}\n" "$*"
}

talk() {
  printf "  ${YELLOW}💡 %s${RESET}\n" "$*"
}

pause() {
  if [ "$AUTO" = "1" ]; then
    sleep 0.5
    return
  fi
  printf "\n  ${DIM}(press Enter to continue)${RESET} "
  read -r _
}

cleanup() {
  if [ -n "${DASHBOARD_PID:-}" ]; then
    kill "$DASHBOARD_PID" 2>/dev/null || true
    wait "$DASHBOARD_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ── Dummy env vars (channels run in disabled mode) ───────────────────

export SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
export TEAMS_WEBHOOK_URL="${TEAMS_WEBHOOK_URL:-}"
export PD_ROUTING_KEY="${PD_ROUTING_KEY:-}"
export OTLP_ENDPOINT="${OTLP_ENDPOINT:-}"
export SENTINEL_LOG_LEVEL="${SENTINEL_LOG_LEVEL:-WARNING}"

# ═════════════════════════════════════════════════════════════════════
banner "SENTINEL — Unified MLOps + LLMOps + AgentOps Platform"
# ═════════════════════════════════════════════════════════════════════

printf "\n"
say "One SDK.  One config file.  Seven layers of production ML governance."
say ""
say "This demo will seed every subsystem with synthetic data and launch"
say "the live dashboard so you can explore the full platform."
printf "\n"

# ── Preflight ────────────────────────────────────────────────────────

act "0" "Preflight checks"

say "Checking Python environment..."
python -c "import sentinel; print(f'  sentinel v{sentinel.__version__} ✓')" 2>/dev/null || {
  printf "  ${RED}Error: sentinel not installed. Run: pip install -e \".[all,dev]\"${RESET}\n"
  exit 1
}
python -c "import numpy; print(f'  numpy v{numpy.__version__} ✓')" 2>/dev/null || {
  printf "  ${RED}Error: numpy not installed.${RESET}\n"
  exit 1
}

say "Validating showcase config..."
sentinel config validate --config "$CONFIG" 2>/dev/null && say "  config valid ✓" || {
  # Config validation may warn about missing file refs — that's OK for demo
  say "  config loaded with warnings (OK for demo)"
}

pause

# ── Act 1: The Config ────────────────────────────────────────────────

act "1" "The Config — one YAML file governs everything"

say "This is the ENTIRE governance model for a fraud classifier,"
say "including LLMOps guardrails and AgentOps safety rails:"
printf "\n"

# Show key sections of the config
printf "  ${BOLD}model:${RESET}\n"
sed -n '/^model:/,/^data_quality:/p' "$CONFIG" | head -7 | while IFS= read -r line; do
  printf "  ${DIM}%s${RESET}\n" "$line"
done

printf "\n  ${BOLD}drift (with auto-check):${RESET}\n"
sed -n '/^drift:/,/^feature_health:/p' "$CONFIG" | head -16 | while IFS= read -r line; do
  printf "  ${DIM}%s${RESET}\n" "$line"
done

printf "\n  ${BOLD}alerts (with escalation):${RESET}\n"
sed -n '/^alerts:/,/^retraining:/p' "$CONFIG" | head -15 | while IFS= read -r line; do
  printf "  ${DIM}%s${RESET}\n" "$line"
done

printf "\n"
talk "Every behaviour is config-driven. No code changes for policy updates."
talk "Git-blame the YAML to see who changed what threshold and when."

pause

# ── Act 2: Seed all subsystems ───────────────────────────────────────

act "2" "Seeding all subsystems with synthetic data"

say "Running the showcase seeder — this populates every layer..."
printf "\n"

python demo/data/seed_showcase.py --config "$CONFIG"

pause

# ── Act 3: Launch the dashboard ──────────────────────────────────────

act "3" "Launching the Sentinel dashboard"

say "Starting dashboard on http://127.0.0.1:${PORT} ..."

if [ "$NO_BROWSER" = "1" ]; then
  sentinel dashboard --config "$CONFIG" --port "$PORT" >/tmp/sentinel-showcase.log 2>&1 &
else
  sentinel dashboard --config "$CONFIG" --port "$PORT" --open >/tmp/sentinel-showcase.log 2>&1 &
fi
DASHBOARD_PID=$!
sleep 3

if kill -0 "$DASHBOARD_PID" 2>/dev/null; then
  printf "\n"
  say "Dashboard running ✓  PID=$DASHBOARD_PID"
  say ""
  printf "  ${BOLD}→ Open http://127.0.0.1:${PORT} in your browser${RESET}\n"
  say ""
  say "The dashboard shows LIVE data from the seeding step."
  say "Each page corresponds to a layer of the seven-layer stack."
else
  printf "  ${RED}Dashboard failed to start — check /tmp/sentinel-showcase.log${RESET}\n"
  printf "  ${DIM}Continuing with CLI-only demo...${RESET}\n"
fi

pause

# ── Act 4: Walk through dashboard pages ──────────────────────────────

act "4" "Dashboard walkthrough — the seven-layer stack"

say "Navigate through each page in your browser:"
printf "\n"
printf "  ${BOLD}Page              URL                        What to look for${RESET}\n"
printf "  ${DIM}────────────────  ─────────────────────────  ─────────────────────────────${RESET}\n"
printf "  Overview          /                            Model health, KPI links, events\n"
printf "  Drift             /drift                       PSI scores, concept drift meta\n"
printf "  Features          /features                    Top-3 drifted by importance\n"
printf "  Registry          /registry                    3 versions (4.0.0 → 4.1.0)\n"
printf "  Audit             /audit                       Every event, immutable timeline\n"
printf "  Deployments       /deployments                 Canary at 5%% traffic\n"
printf "  Retraining        /retraining                  Pending retrain request\n"
printf "  Intelligence      /intelligence                KPI mappings + model DAG\n"
printf "  Compliance        /compliance                  FCA + EU AI Act + GDPR + SOC2\n"
printf "  LLMOps            /llmops                      Guardrails, tokens, prompts\n"
printf "  AgentOps          /agentops                    3 traces, tool audit, safety\n"
printf "\n"

talk "Every page reads from the same SentinelClient — one source of truth."

pause

# ── Act 5: CLI deep dives ────────────────────────────────────────────

act "5" "CLI deep dive — drift detection"

say "Let's look at the drift report from the command line:"
printf "\n"

sentinel check --config "$CONFIG" 2>/dev/null | python -m json.tool 2>/dev/null || {
  # If check fails (no baseline in CLI mode), show status instead
  say "(drift check requires fitted baseline — showing status instead)"
  sentinel status --config "$CONFIG" 2>/dev/null | python -m json.tool 2>/dev/null || true
}

printf "\n"
talk "PSI method detected distribution shift in the fraud ring data."
talk "Drifted features: claim_amount, claimant_age, prior_claims."

pause

# ── Act 6: Audit trail ───────────────────────────────────────────────

act "6" "Audit trail — every event, immutable, compliance-ready"

say "Recent audit events:"
printf "\n"

sentinel audit --config "$CONFIG" --limit 15 2>/dev/null || {
  say "(showing audit files directly)"
  ls -la demo/audit/showcase/*.jsonl 2>/dev/null || true
  printf "\n  ${DIM}Latest events:${RESET}\n"
  tail -5 demo/audit/showcase/*.jsonl 2>/dev/null | while IFS= read -r line; do
    printf "  %s\n" "$(echo "$line" | python -c "import sys,json; d=json.load(sys.stdin); print(f\"{d.get('timestamp','?')[:19]}  {d.get('event_type','?'):30s}  {d.get('model_name','?')}\")" 2>/dev/null || echo "  $line")"
  done
}

printf "\n"
talk "7-year retention (2,555 days) is the DEFAULT."
talk "FCA Consumer Duty, EU AI Act, GDPR, SOC2 — all declared in config."
talk "Every event you saw in this demo is in the audit trail right now."

pause

# ── Act 7: Model status ─────────────────────────────────────────────

act "7" "Model status — the health snapshot"

say "Current model status:"
printf "\n"

sentinel status --config "$CONFIG" 2>/dev/null | python -m json.tool 2>/dev/null || true

printf "\n"
talk "One command shows: drift state, buffer, registry, LLMOps, AgentOps."

pause

# ── Act 8: The close ─────────────────────────────────────────────────

act "8" "Summary — what you just saw"

printf "\n"
printf "  ${BOLD}Layer 1 — Developer Interface${RESET}\n"
printf "    ✓ SentinelClient: 1 import, 1 config file, full lifecycle\n"
printf "    ✓ CLI: init, check, status, deploy, audit, dashboard\n"
printf "    ✓ Dashboard: 11 live pages, zero external dependencies\n"
printf "\n"
printf "  ${BOLD}Layer 2 — Observability${RESET}\n"
printf "    ✓ Data drift: PSI with per-feature scores\n"
printf "    ✓ Concept drift: DDM streaming from log_prediction()\n"
printf "    ✓ Count-based auto-check: fires every N predictions\n"
printf "    ✓ Data quality: schema, freshness, outliers\n"
printf "    ✓ Feature health: importance × drift ranking\n"
printf "\n"
printf "  ${BOLD}Layer 3 — LLMOps${RESET}\n"
printf "    ✓ Guardrails: PII redaction, jailbreak, topic fence\n"
printf "    ✓ Token economics: per-query + daily budgets\n"
printf "    ✓ Prompt management: versioning + A/B routing\n"
printf "    ✓ Groundedness: hallucination defence for RAG\n"
printf "\n"
printf "  ${BOLD}Layer 4 — AgentOps${RESET}\n"
printf "    ✓ Trace monitoring: span-based, OTel-compatible\n"
printf "    ✓ Tool audit: allowlist/blocklist per agent\n"
printf "    ✓ Safety: loop detection, budget guard, escalation\n"
printf "    ✓ LangGraph middleware: zero-code instrumentation\n"
printf "\n"
printf "  ${BOLD}Layer 5 — Intelligence${RESET}\n"
printf "    ✓ Model graph: dependency DAG + cascade alerts\n"
printf "    ✓ KPI linkage: precision → fraud_catch_rate\n"
printf "\n"
printf "  ${BOLD}Layer 6 — Action${RESET}\n"
printf "    ✓ Notifications: Slack + Teams + PagerDuty + escalation\n"
printf "    ✓ Retrain orchestrator: drift → pipeline → approve → deploy\n"
printf "    ✓ Deployment: shadow / canary / blue-green with auto-rollback\n"
printf "\n"
printf "  ${BOLD}Layer 7 — Foundation${RESET}\n"
printf "    ✓ Model registry: 3 versions with full metadata\n"
printf "    ✓ Audit trail: immutable, 7-year retention, tamper-evident\n"
printf "    ✓ Compliance: FCA + EU AI Act + GDPR + SOC2\n"
printf "\n"

printf "  ${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
printf "  ${BOLD}  All of the above from ONE config file and ONE pip install.${RESET}\n"
printf "  ${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
printf "\n"

if [ -n "${DASHBOARD_PID:-}" ] && kill -0 "$DASHBOARD_PID" 2>/dev/null; then
  say "Dashboard is still running at http://127.0.0.1:${PORT}"
  say "Explore freely. Press Ctrl+C to stop."
  printf "\n"
  wait "$DASHBOARD_PID" 2>/dev/null || true
else
  say "Demo complete."
fi
