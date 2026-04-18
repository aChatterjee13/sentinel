# Sentinel for Enterprise Agent Platform Teams

**One config governs every agent in your estate — across every framework — with per-team budgets, per-agent tool permissions, and one audit trail.**

---

## The problem we heard from you

- You have 10–30 agents in production or on the way. Three frameworks.
  Four teams. Two vendors.
- Each team built its own monitoring. None of it talks to each other.
- Ops gets paged at 3am by runaway agents that nobody can kill from outside
  the process.
- Your CFO is asking why the LLM bill went up 40% this quarter. You cannot
  attribute cost to a team, let alone to an agent or a prompt version.
- The security team wants to know which tools each agent can call. The
  answer lives in a README that's already out of date.
- Your auditor wants proof that a specific agent took a specific action. You
  have logs, but not a trace.

## What Sentinel does about it

| Problem | Sentinel capability |
|---|---|
| Multi-framework sprawl | `auto_instrument: { langgraph, semantic_kernel, crewai, google_adk }` |
| Runaway agents | `loop_detection: { max_iterations, max_repeated_tool_calls, thrash_window }` |
| Budget blowout | `budget: { max_tokens_per_run, max_cost_per_run, max_time_per_run }` |
| Cost attribution | `token_economics.track_by: [model, agent_name, team, environment]` |
| Tool permission drift | `tool_audit.permissions.<agent>: { allowed: [...], blocked: [...] }` |
| Destructive ops | `sandbox.mode: approve_first` on write/delete/execute/transfer |
| A2A discovery | `agent_registry.a2a: { protocol: a2a_v1, discovery: registry }` |
| Golden-suite regression | `evaluation.golden_datasets.run_on: [version_change, daily]` |

## What you saw in the demo

1. A LangGraph agent and a Semantic Kernel agent, governed by the same config.
2. A **buggy looping agent was killed by the safety rail** — not by ops.
3. A sales agent was **blocked from sending an email it had just drafted**
   because the `email_send` tool was in its blocklist.
4. The audit trail has every reasoning step for every agent in the estate,
   searchable by agent name, team, task type.

## Commercial headline

- **One config, every framework**: LangGraph, Semantic Kernel, CrewAI, Google ADK.
- **Per-team cost attribution from day 1**: No more mystery LLM bills.
- **Hard safety rails**: Loops, budgets, tool permissions, destructive-op sandboxing.
- **Golden-suite CI gate**: No agent version ships that regresses the golden suite.
- **A2A registry**: Agents discover each other by capability, not by URL.
- **OpenTelemetry compatible**: Plug your existing observability stack in.

## What we need from you to run a pilot

1. 3–5 agents currently in production across 2 frameworks.
2. Slack + PagerDuty access for the escalation chain.
3. A named owner for each agent.
4. 3 weeks of part-time effort from your platform team lead.
5. At the end: per-team budget reports for a month, golden-suite CI gate
   wired into your deploy pipeline, and a single dashboard showing every
   agent's trace, cost, and safety rail firings.

## The ask

A technical deep-dive with your platform architect and your SRE lead. We
pick one of your current agents, instrument it live, and show you the first
5 minutes of traces in your own OTEL backend.

---

*Contact:* your Zensar account team • *Demo repo:* `demo/scripts/platform_demo.sh`
