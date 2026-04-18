# Slide 07 — AgentOps: tracing, safety rails, tool audit

## The three problems nobody has solved yet

1. **Runaway agents** — no way to kill from outside the process.
2. **Tool permission sprawl** — which agent can call which API?
3. **Reasoning trace for audit** — where does the agent's decision come from?

## The Sentinel config

```yaml
agentops:
  tracing:
    sample_rate: 1.0               # trace every run, retention 2555 days
    auto_instrument:
      langgraph: true
      semantic_kernel: true
      crewai: true
      google_adk: true
  tool_audit:
    permissions:
      claims_agent:
        allowed: [policy_lookup, kb_search]
        blocked: [payment_execute, customer_delete]
    parameter_validation: true     # schemas enforced
    rate_limits:
      default: 100/min
  safety:
    loop_detection:
      max_iterations: 50
      max_repeated_tool_calls: 5
      max_delegation_depth: 5
    budget:
      max_tokens_per_run: 50000
      max_cost_per_run: 5.00
      max_time_per_run: 300s
      on_exceeded: graceful_stop
    escalation:
      triggers:
        - condition: confidence_below
          threshold: 0.3
          action: human_handoff
    sandbox:
      destructive_ops: [write, delete, execute, transfer]
      mode: approve_first
```

## What this delivers

- **One config covers four frameworks**
- **Every tool call is audited** with parameter validation
- **Destructive ops are gated** by approve-first or sandbox-first
- **Runaway agents are killed** by the SDK, not by ops at 3am
- **Low-confidence outputs escalate** to a human automatically
- **OpenTelemetry-compatible traces** go to your existing backend

---

*Speaker note:* This slide is the single biggest differentiator. No vendor in
the market has all of this under one config. Pause on the "one config covers
four frameworks" line.
