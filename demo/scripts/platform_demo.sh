#!/usr/bin/env bash
# Playbook E — Enterprise agent platform team
#
# 5-act narrative:
#   1. The estate problem: 12 agents, 3 frameworks, no shared governance
#   2. One Sentinel config for all of them
#   3. Run a normal task (LangGraph + Semantic Kernel)
#   4. Run adversarial tasks — each safety rail catches its target
#   5. Audit + per-team budgets + golden-suite eval
#
# Run from repo root:
#   source demo/envs/platform.env
#   bash demo/scripts/platform_demo.sh

set -euo pipefail

CONFIG=${SENTINEL_CONFIG_PATH:-./demo/configs/multi_team_agents.yaml}

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

say "Act 0 — generate synthetic agent manifest + tasks"
python demo/data/generate_platform_data.py
pause

say "Act 1 — show the auto-instrument and per-agent permissions sections"
awk '/auto_instrument:/,/replay:/' "$CONFIG"
pause

say "Act 2 — validate the multi-framework agent config"
sentinel validate --config "$CONFIG"
pause

say "Act 3 — run a normal LangGraph task (sales research)"
python demo/agents/langgraph_agent.py --task "research Acme Corp"
pause

say "Act 4 — run a normal Semantic Kernel task (followup email draft)"
python demo/agents/semantic_kernel_agent.py --task "draft followup email for Acme"
pause

say "Act 5 — adversarial: a buggy looping agent"
python demo/agents/buggy_loop_agent.py || true
pause

say "Act 6 — status (per-agent traces, costs, success rates)"
sentinel status --config "$CONFIG"
pause

say "Act 7 — audit trail of safety rail firings"
sentinel audit --config "$CONFIG" --type safety_rail_triggered --limit 10
pause

say "Demo complete. Talking points:"
cat <<'EOF'
  • One config governs LangGraph + Semantic Kernel + CrewAI + Google ADK.
  • Per-team budgets ($2,000/day total) and per-query caps ($1.00).
  • Buggy loop killed by max_repeated_tool_calls — not by ops at 3am.
  • Tool permissions = sales agent CANNOT send the email it just drafted.
  • A2A registry lets agents discover each other by capability.
  • Golden-suite eval gates version_change in CI. Regression is impossible.
EOF
