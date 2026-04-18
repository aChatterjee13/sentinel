# Sentinel for UK Retail Banks

**One SDK. FCA Consumer Duty + PRA SS1/23 ready in 40 lines of YAML.**

---

## The problem we heard from you

- 5–7 separate tools stitched together per model: Evidently, MLflow, custom
  alerting scripts, home-grown deployment, a spreadsheet for model inventory.
- Every audit cycle costs weeks of evidence-gathering.
- Canary rollouts are hand-run. Rollbacks are hand-run. Blame for the last
  outage is still live.
- SS1/23 and Consumer Duty documents are re-typed for every new model version.

## What Sentinel does about it

| Capability | How Sentinel delivers |
|---|---|
| Data + concept + model drift | PSI / KS / DDM on a single config block |
| Feature health with SHAP | `feature_health.importance_method: shap` |
| FCA + PRA audit trail | `retention_days: 2555` is the default |
| Auto-rollback canary | `canary.rollback_on: { error_rate_increase, latency_p99 }` |
| Human-in-the-loop retrain | `approval.mode: human_in_loop` with named approvers |
| Cascade alerts | `model_graph.cascade_alerts: true` |
| Slack + Teams + PagerDuty | One `alerts.channels` list with escalation chain |

## The 40-line demo you just saw

```yaml
model:  { name: credit_decisioning_v3, domain: tabular, framework: xgboost }
drift:  { data: { method: psi, threshold: 0.2 }, concept: { method: ddm } }
retraining: { approval: { mode: human_in_loop,
                          approvers: [model-risk@bank.example] } }
deployment: { strategy: canary,
              canary: { rollback_on: { error_rate_increase: 0.02 } } }
audit:  { retention_days: 2555,
          compliance_frameworks: [fca_consumer_duty, pra_ss1_23] }
```

## Commercial headline

- **Integration effort**: <50 lines of Python. One YAML file.
- **Weeks-to-hours**: Model onboarding collapses from 3–4 weeks to one day.
- **Audit prep**: Regulator-ready lineage is a 1-click export.
- **Framework agnostic**: Works with your existing XGBoost / LightGBM /
  sklearn / PyTorch models — no model changes required.
- **Cloud portable**: Azure-first, same config runs on AWS / GCP / on-prem.

## What we need from you to run a pilot

1. One non-critical model (e.g. a champion challenger already in shadow).
2. Azure ML workspace access OR a local dev environment.
3. 1 week of 2-hour working sessions with one of your ML platform engineers.
4. At the end: a Slack alert fires on synthetic drift, you show your auditor
   the trail from that alert back to the baseline, and we jointly agree
   whether to expand to 5–10 models.

## The ask

A 1-hour deep-dive with your model risk team to walk through the SS1/23
evidence pack Sentinel can produce on day one. We bring the SDK, the config,
and a synthetic dataset. You bring the auditor questions.

---

*Contact:* your Zensar account team • *Demo repo:* `demo/scripts/bank_demo.sh`
