# Slide 11 — One dashboard, every module, no SaaS

```
┌──────────────────────────────────────────────────────────────────────┐
│  ▸ Project Sentinel                              fraud_v3.2 / canary │
├──────────────────────────────────────────────────────────────────────┤
│  Overview │ Drift │ Features │ Registry │ Audit │ LLMOps │ AgentOps │
├──────────────────────────────────────────────────────────────────────┤
│  Drift timeline (PSI, 14d window)            ┃  Top features        │
│      ▁▂▂▃▃▅▅▆▇▇█▇▆▆ ←── threshold 0.20      ┃  • amount_log   .31  │
│                                              ┃  • merchant_cc  .27  │
│                                              ┃  • txn_velocity .22  │
│  Recent alerts                                                       │
│  [HIGH ] 14:02  data drift psi=0.27 > 0.20                          │
│  [INFO ] 13:58  canary 25% → 50% (auto-promoted, error +0.0%)       │
└──────────────────────────────────────────────────────────────────────┘
```

### What it actually shows

| Page | What's on it | Where the data comes from |
|---|---|---|
| **Overview** | Model name, version, drift status, recent alerts, active deployments | `client.status()` + `client.audit.latest()` |
| **Drift** | PSI/KS timeline (Plotly), feature scores, drilldown to each report | `client.audit.query(event_type="drift_checked")` |
| **Features** | Feature health table with importance + drift score + null rate | `client.get_feature_health()` |
| **Registry** | All registered models, versions, baseline metrics, version comparison | `client.registry` |
| **Audit** | Filterable immutable trail (event type, model, date range) | `client.audit.query()` |
| **LLMOps** | Prompt versions + A/B routing, guardrail violations, daily token cost | `client.llmops.*` |
| **AgentOps** | Recent traces + timeline view, tool stats, agent registry | `client.agentops.tracer / tool_audit / registry` |
| **Deployments** | Active strategy, traffic split, history feed | `client.deployment_manager.list_active()` |
| **Compliance** | Frameworks declared in config + per-framework event counts | `client.config.audit.compliance_frameworks` |

### Why it matters

- **Local-first**: `sentinel dashboard --config sentinel.yaml` and you have
  a UI on `localhost:8000`. Zero SaaS, zero data leaving the machine.
- **Same source of truth**: Every page reads from the live `SentinelClient`
  that's already powering the application — no separate database, no ETL,
  no replication lag.
- **Embeddable**: `SentinelDashboardRouter(client).attach(app, prefix="/sentinel")`
  drops the same UI inside an existing FastAPI service. Customers add
  Sentinel pages to their internal portal in 3 lines.
- **Stays out of the way**: Optional extra (`pip install sentinel-mlops[dashboard]`).
  Customers who want a CLI-only deploy never pay the FastAPI cost.

---

*Speaker note:* Open the dashboard live before saying any of this. The
left-hand drift chart updating in real time as you fire the drifted-window
demo command in the terminal **is the slide**. Don't read the table — let
them watch the alert appear on screen.

Run-along command:

```bash
export SENTINEL_DEMO_DASHBOARD=1
bash demo/scripts/bank_demo.sh
```
