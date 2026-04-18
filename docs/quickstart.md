# Quickstart

This guide takes you from `pip install` to a fully monitored model with drift detection, alerts, model registry, and an audit trail — in about 10 minutes. Everything works locally with no cloud dependencies.

## What you'll build

A small classification model wrapped in a FastAPI service. Sentinel will:

- Validate every prediction against a schema
- Compute PSI drift on a rolling window
- Fire a Slack alert (or any channel you configure) when drift exceeds the threshold
- Track every model version in a local registry (or Azure ML, MLflow, SageMaker, Vertex AI, Databricks)
- Log every event to an immutable audit trail
- Track per-cohort performance with fairness disparity detection
- Explain predictions at row, global, and cohort levels
- Monitor LLM calls with guardrails, token economics, and semantic drift
- Trace agent reasoning steps with safety budgets and tool audit
- Auto-select domain-appropriate drift detectors for time series, NLP, recommendation, and graph models

---

## 1. Installation

```bash
pip install sentinel-mlops
```

For this quickstart you'll also want the optional drift extras (scipy + sklearn) so the full set of detectors is available:

```bash
pip install "sentinel-mlops[drift]"
```

Verify the install:

```bash
sentinel --version
```

Common extras:

| Extra | What it adds |
|-------|-------------|
| `drift` | scipy, scikit-learn — advanced drift detectors |
| `llmops` | tiktoken, openai, presidio — LLM guardrails & token tracking |
| `agentops` | opentelemetry — agent tracing & export |
| `timeseries` | statsmodels, pmdarima — seasonal drift & forecast quality |
| `graph` | networkx, torch-geometric — topology drift |
| `dashboard` | FastAPI, uvicorn, Jinja2, Plotly — local UI |
| `azure` | azure-ai-ml, azure-storage-blob — Azure backends |
| `mlflow` | mlflow — MLflow registry backend |
| `all` | everything above (lightweight; excludes torch/presidio) |

```bash
pip install "sentinel-mlops[all]"
```

---

## 2. Configuration

### Generate a starter config

```bash
sentinel init --name fraud_classifier --out sentinel.yaml
```

This drops a `sentinel.yaml` next to your code with sensible defaults. Open it and you'll see something like:

```yaml
version: "1.0"

model:
  name: fraud_classifier
  type: classification
  domain: tabular

drift:
  data:
    method: psi
    threshold: 0.2
    window: 7d
    reference: baseline

alerts:
  channels:
    - type: slack
      webhook_url: ${SLACK_WEBHOOK_URL}
      channel: "#ml-alerts"
  policies:
    cooldown: 1h
    rate_limit_per_hour: 60

audit:
  storage: local
  path: ./audit/
  retention_days: 2555
```

### Validate

```bash
sentinel validate --config sentinel.yaml
```

You should see `Config OK`. If you see a Pydantic error, the message will tell you exactly which field is wrong.

Before you ship the config to production, run the strict validator. Strict
mode also fails on unset `${VAR}` substitutions and missing file references
(baselines, schemas, holdout datasets) — exactly the things that pass plain
`validate` but blow up on first use:

```bash
sentinel config validate --config sentinel.yaml --strict
```

### Inspect the resolved config

Useful when you have a chain of `extends:` parents. Secrets are masked by default:

```bash
sentinel config show --config sentinel.yaml
# webhook_url: <REDACTED>
# routing_key: <REDACTED>

sentinel config show --config sentinel.yaml --format json
sentinel config show --config sentinel.yaml --unmask  # prints plaintext secrets
```

---

## 3. Core usage

### Wire Sentinel into your serving code

Here's a minimal FastAPI app. The whole loop — quality check, prediction, drift logging — is five lines.

```python
# app.py
from fastapi import FastAPI
from sentinel import SentinelClient
import joblib

app = FastAPI()
model = joblib.load("models/fraud.pkl")
sentinel = SentinelClient.from_config("sentinel.yaml")

@app.on_event("startup")
async def startup() -> None:
    # Idempotent — safe to call on every boot
    sentinel.register_model_if_new(version="1.0.0", framework="sklearn")

@app.post("/predict")
async def predict(features: dict) -> dict:
    quality = sentinel.check_data_quality(features)
    if quality.has_critical_issues:
        return {"error": "Input validation failed", "issues": quality.issues}

    prediction = model.predict([features])
    prediction_id = sentinel.log_prediction(features=features, prediction=prediction)

    return {"prediction": prediction.tolist(), "prediction_id": prediction_id}

@app.get("/health")
async def health() -> dict:
    return {
        "model": sentinel.model_name,
        "drift": sentinel.check_drift().summary,
        "feature_health": sentinel.get_feature_health().summary,
    }
```

That's it. You now have:

- **Schema enforcement** on every input
- **Prediction logging** with a unique `prediction_id` for later ground-truth attachment
- **A drift check endpoint** for your liveness probes
- **Model registration** on startup

### Log ground truth with `log_actual()`

Ground truth often arrives days or weeks after the prediction. Use the `prediction_id` returned by `log_prediction()` to attach actuals later:

```python
prediction_id = sentinel.log_prediction(features=X, prediction=y_pred)

# ... days later, when the ground truth arrives ...
sentinel.log_actual(prediction_id, actual=y_true)
```

This enables concept drift detection (DDM, ADWIN) which needs paired prediction/actual values.

### Fit a baseline

Drift detection needs a reference distribution. The simplest way to fit one is from your training data:

```python
import pandas as pd
from sentinel import SentinelClient

sentinel = SentinelClient.from_config("sentinel.yaml")
reference = pd.read_parquet("data/training.parquet")
sentinel.fit_baseline(reference)
```

Or, from the CLI, with a CSV/Parquet file:

```bash
sentinel check --config sentinel.yaml --reference data/training.parquet --current data/today.parquet
```

Sentinel will:

1. Fit the baseline against `--reference` if not already fitted
2. Compute drift statistics against `--current`
3. Print a `DriftReport` to stdout
4. Append a drift event to the audit trail
5. Fire any configured alerts if drift exceeds the threshold

### Re-baseline after retraining

After retraining a model on new data, reset the drift baseline so future comparisons use the new reference:

```python
sentinel.reset_drift_baseline(reference=new_training_data)
```

Calling without arguments clears the fitted state and resets all concept drift observations.

### Export and clear the prediction buffer

The SDK buffers predictions in memory for drift windows. Export and optionally clear:

```python
n = sentinel.flush_buffer("predictions.jsonl")
print(f"Exported {n} predictions")

sentinel.clear_buffer()  # optional: free memory
```

### Configure an alert channel

Set the webhook URL as an environment variable:

```bash
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T000/B000/XXX
```

The `${SLACK_WEBHOOK_URL}` token in `sentinel.yaml` is substituted at load time. The first time drift is detected, you'll see a message like:

```
[WARNING] fraud_classifier: data drift detected
Source: drift_detector
Method: psi   Score: 0.27   Threshold: 0.20
Top features: amount_log, merchant_country, txn_velocity
```

Cooldown defaults to 1 hour, so duplicate alerts won't flood the channel. Rate limiting (60 alerts/hour by default) catches anything that slips through.

---

## 4. Explainability

Sentinel wraps SHAP for per-prediction, global, and per-cohort explanations. Install the extra:

```bash
pip install "sentinel-mlops[explain]"
```

You **must** call `set_model_for_explanations()` before any explain method:

```python
sentinel.set_model_for_explanations(
    model=model,
    feature_names=["income", "age", "tenure", "balance"],
    background_data=X_train[:100],
)

# Per-row explanations
attributions = sentinel.explain(X_test[:5])
# [{"income": 0.42, "age": -0.13, ...}, ...]

# Global feature importance (mean |SHAP| across rows)
global_imp = sentinel.explain_global(X_test)
# {"income": 0.42, "age": 0.31, "tenure": 0.18, ...}

# Per-cohort comparison
cohort_imp = sentinel.explain_cohorts(X_test, cohort_labels=labels)
# {"premium": {"income": 0.5, ...}, "standard": {"income": 0.3, ...}}
```

All results are visible in the dashboard at `/explanations`.

---

## 5. Cohort analysis

If your model serves different sub-populations (e.g., customer segments, regions, product lines), enable cohort tracking in `sentinel.yaml`:

```yaml
cohort_analysis:
  enabled: true
  cohort_column: customer_segment
  disparity_threshold: 0.1
```

Then log predictions with cohort labels:

```python
sentinel.log_prediction(features=X, prediction=y_pred, actual=y_true)
# cohort auto-derived from features["customer_segment"]

# Or specify explicitly:
sentinel.log_prediction(features=X, prediction=y_pred, cohort_id="premium")
```

Compare performance across cohorts:

```python
comparison = sentinel.compare_cohorts()
for alert in comparison.disparity_alerts:
    print(f"Cohort {alert.cohort_id}: {alert.deviation_pct:+.1f}% from global")
```

Both cohort analysis and explainability are visible in the dashboard at `/cohorts`.

---

## 6. Model registry

### Register and manage versions

```python
sentinel.register_model(
    version="1.1.0",
    framework="sklearn",
    metrics={"f1": 0.91, "auc": 0.96},
)
```

```bash
sentinel registry list --config sentinel.yaml
sentinel registry show --config sentinel.yaml --version 1.0.0
```

### Multi-backend registry

The default backend is `local` (filesystem). For production, configure a cloud backend in `sentinel.yaml`:

**Azure ML:**

```yaml
registry:
  backend: azure_ml
  subscription_id: ${AZURE_SUBSCRIPTION_ID}
  resource_group: ${AZURE_RESOURCE_GROUP}
  workspace_name: ${AZURE_ML_WORKSPACE}
```

**MLflow:**

```yaml
registry:
  backend: mlflow
  tracking_uri: ${MLFLOW_TRACKING_URI}
```

**SageMaker:**

```yaml
registry:
  backend: sagemaker
  region_name: us-east-1
  role_arn: ${SAGEMAKER_ROLE_ARN}
  s3_bucket: ${SAGEMAKER_BUCKET}
```

**Vertex AI:**

```yaml
registry:
  backend: vertex_ai
  project: ${GCP_PROJECT}
  location: us-central1
  gcs_bucket: ${GCS_BUCKET}
```

**Databricks:**

```yaml
registry:
  backend: databricks
  host: ${DATABRICKS_HOST}
  token: ${DATABRICKS_TOKEN}
  catalog: ml
  schema_name: models
```

All six backends implement the same interface — switching is a one-line YAML change.

---

## 7. Deployment

### Deploy a new version

When you train a new model, register it and roll it out with a canary:

```bash
sentinel deploy --config sentinel.yaml --version 1.1.0 --strategy canary --traffic 5
```

Sentinel will start the canary at 5% traffic. Each `advance` call (or your scheduler) walks the ramp steps. If the configured `rollback_on` metrics breach their thresholds, Sentinel auto-rolls back and logs the rollback to the audit trail.

### Dry-run mode

Preview what a deployment would do without actually deploying:

```bash
sentinel deploy --config sentinel.yaml --version 1.1.0 --strategy canary --traffic 5 --dry-run
```

Returns a JSON validation result showing the plan without executing it.

### Shadow testing

For zero-traffic shadow testing (new model runs in parallel, predictions logged but not served):

```bash
sentinel deploy --config sentinel.yaml --version 1.1.0 --strategy shadow
```

Available strategies: `shadow`, `canary`, `blue_green`, `direct`.

---

## 8. LLMOps

Enable LLMOps in your config:

```yaml
llmops:
  enabled: true
  mode: rag  # rag | completion | chat | agent
```

### Guardrails

The guardrail pipeline runs input and output checks on every LLM call:

```python
from sentinel.llmops.guardrails import GuardrailPipeline

pipeline = GuardrailPipeline.from_config(sentinel.config.llmops)

# Pre-call check — PII redaction, jailbreak detection, topic fencing
input_result = pipeline.check_input(user_message, context=retrieved_chunks)
if input_result.blocked:
    return {"error": input_result.reason}

# ... LLM call happens here ...

# Post-call check — groundedness, toxicity, format compliance
output_result = pipeline.check_output(
    response=llm_response,
    context=retrieved_chunks,
    original_query=user_message,
)
if output_result.blocked:
    return {"error": "Response did not pass safety checks"}
```

### Custom guardrails DSL

Define custom rules in YAML — no Python code required:

```yaml
llmops:
  guardrails:
    input:
      - type: custom
        name: sql_injection_guard
        rules:
          - { type: regex_absent, value: "DROP TABLE|DELETE FROM|TRUNCATE" }
          - { type: min_length, value: 5 }
          - { type: not_empty }
        combine: "any"
        action: block
```

11 built-in rule types: `regex_match`, `regex_absent`, `keyword_present`, `keyword_absent`, `min_length`, `max_length`, `json_schema`, `sentiment`, `language`, `word_count`, `not_empty`.

For full custom logic, use a **plugin guardrail**:

```yaml
llmops:
  guardrails:
    output:
      - type: plugin
        module: "my_company.guardrails"
        class_name: "ComplianceGuardrail"
        config: { strict_mode: true }
        action: warn
```

The plugin class must implement `check(content, context) -> GuardrailResult`.

### Prompt management

```python
from sentinel.llmops.prompt_manager import PromptManager

pm = PromptManager(config=sentinel.config.llmops)

pm.register(
    name="claims_qa",
    version="1.0",
    system_prompt="You are an insurance claims analyst...",
    template="Summarise the following claim: {{claim_text}}",
)

# A/B traffic routing
pm.set_traffic("claims_qa", {"1.0": 90, "1.1": 10})
prompt = pm.resolve("claims_qa", context={"user_segment": "gold"})
```

### Token economics

```python
from sentinel.llmops.token_economics import TokenTracker

tracker = TokenTracker(config=sentinel.config.llmops.token_economics)
usage = tracker.record(
    model="gpt-4o",
    input_tokens=450,
    output_tokens=120,
)
print(f"Cost: ${usage.cost:.4f}")
print(f"Daily total: ${tracker.daily_total():.2f}")
```

### Semantic drift baseline

For LLM output drift detection, fit a baseline from a representative sample of responses:

```python
sentinel.fit_semantic_baseline(outputs=sample_responses)
```

This embeds the sample outputs and stores the distribution centroid. Future calls are compared against this baseline to detect semantic drift.

---

## 9. AgentOps

Enable AgentOps in your config:

```yaml
agentops:
  enabled: true
  tracing:
    backend: local  # local | otlp | arize_phoenix
    sample_rate: 1.0
```

### Trace agent reasoning

```python
from sentinel.agentops.trace.tracer import AgentTracer

tracer = AgentTracer(config=sentinel.config.agentops.tracing)

with tracer.trace(agent_name="claims_processor") as trace:
    with tracer.span("plan"):
        plan = agent.plan(claim_id)

    with tracer.span("tool_call", tool="policy_search"):
        policy = search_policy(claim_id)

    with tracer.span("synthesise"):
        result = agent.synthesise(policy)

last = tracer.get_last_trace()
print(f"Steps: {last.step_count}, Tokens: {last.total_tokens}")
```

### Tool audit

Control which tools each agent can call:

```yaml
agentops:
  tool_audit:
    permissions:
      claims_processor:
        allowed: [sharepoint_search, policy_lookup, llm_extraction]
        blocked: [payment_execute, user_delete]
    rate_limits:
      default: 100/min
      overrides:
        payment_execute: 5/min
```

### Safety budgets

```yaml
agentops:
  safety:
    loop_detection:
      max_iterations: 50
      max_repeated_tool_calls: 5
    budget:
      max_tokens_per_run: 50000
      max_cost_per_run: 5.00
      max_time_per_run: 300s
      on_exceeded: graceful_stop  # graceful_stop | escalate | hard_kill
```

### Golden datasets

Store curated input → expected output → expected trajectory test suites:

```yaml
agentops:
  evaluation:
    golden_datasets:
      path: tests/golden/
      run_on: [version_change, daily]
    task_completion:
      min_success_rate: 0.85
```

---

## 10. Domain adapters

Switch `model.domain` in your config and Sentinel automatically uses domain-appropriate drift detectors and quality metrics — no code changes needed:

```yaml
model:
  name: demand_forecast_v3
  domain: timeseries  # tabular | timeseries | nlp | recommendation | graph
  type: regression
```

### Available domains

**Tabular** (default) — PSI/KS drift, standard classification/regression metrics. This is the default when no domain is specified; existing configs work unchanged.

**Time series** — seasonality-aware drift (calendar tests, ACF monitoring, stationarity checks), forecast quality metrics (MASE, coverage, directional accuracy), STL decomposition monitoring:

```yaml
model:
  domain: timeseries
domains:
  timeseries:
    frequency: daily
    seasonality_periods: [7, 365]
    drift:
      method: calendar_test
      compare_against: same_period
    quality:
      metrics: [mase, coverage, directional_accuracy]
      horizon_tracking: [1, 7, 14, 30]
```

**NLP** — vocabulary drift (OOV rate), embedding space monitoring (MMD), label distribution shift, text statistics. For non-LLM NLP models (NER, classification, sentiment):

```yaml
model:
  domain: nlp
domains:
  nlp:
    task: ner  # ner | classification | sentiment | topic_modelling
    drift:
      vocabulary:
        oov_rate_threshold: 0.05
      embedding:
        method: mmd
```

**Recommendation** — item/user distribution drift, beyond-accuracy metrics (coverage, diversity, novelty, popularity bias, fairness):

```yaml
model:
  domain: recommendation
domains:
  recommendation:
    model_type: collaborative_filtering
    quality:
      ranking_metrics: [ndcg_at_10, map_at_10]
      beyond_accuracy:
        coverage:
          min_threshold: 0.4
        popularity_bias:
          max_threshold: 0.8
```

**Graph** — topology drift (degree distribution, density, clustering coefficient), node/edge feature drift, knowledge graph metrics (Hits@K, MRR, relation coverage):

```yaml
model:
  domain: graph
domains:
  graph:
    task: link_prediction
    graph_type: knowledge_graph
    drift:
      topology:
        degree_distribution:
          method: ks_test
```

See [`configs/examples/`](../configs/examples/) for full production-shaped configs for each domain.

---

## 11. Dashboard

The dashboard ships under the optional `[dashboard]` extra. Install it once:

```bash
pip install "sentinel-mlops[dashboard]"
```

Then point it at the same config you've been using:

```bash
sentinel dashboard --config sentinel.yaml
```

This boots a local FastAPI app on `http://127.0.0.1:8000` with one page per
SDK module — overview, drift, feature health, cohorts, explanations, registry, audit trail,
compliance, LLMOps (prompts / guardrails / token economics), AgentOps
(traces / tools / agent registry), deployments, retraining, and intelligence.
21 pages total with 12 interactive Plotly charts, dark mode, and auto-refresh.
The dashboard reads from the live `SentinelClient` — there is no extra
database to provision and nothing to keep in sync.

Useful flags:

- `--host 0.0.0.0` — bind on all interfaces (overrides `dashboard.server.host`)
- `--port 8080` — override `dashboard.server.port`
- `--reload` — uvicorn auto-reload for dev iteration
- `--open` — open the dashboard in your default browser on boot

### Embed in an existing FastAPI app

```python
from fastapi import FastAPI
from sentinel import SentinelClient, SentinelDashboardRouter

client = SentinelClient.from_config("sentinel.yaml")
app = FastAPI()

dashboard = SentinelDashboardRouter(client)
dashboard.attach(app, prefix="/sentinel")  # mounts pages + JSON API
```

Now `/sentinel/` (HTML) and `/sentinel/api/...` (JSON for HTMX + Plotly) live
inside your existing app, with the same `SentinelClient` powering both your
predictions and the dashboard.

---

## 12. Compliance and audit

### Audit trail

Query the audit trail from the CLI:

```bash
sentinel audit --config sentinel.yaml --type drift_detected --limit 20
```

For regulated deployments, opt into tamper-evident audit trails — every
event will carry an HMAC-SHA256 signature plus a `previous_hash` chain
pointer, so any insert / delete / edit is detectable on
re-verification:

```yaml
audit:
  storage: local
  path: ./audit/
  tamper_evidence: true
  signing_key_env: SENTINEL_AUDIT_KEY
```

```bash
export SENTINEL_AUDIT_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
sentinel audit verify --config sentinel.yaml
# OK — 1,402 events / 1,402 signed / chain head: 9b2f4c…
```

`sentinel audit verify` exits `0` on a clean trail and `1` on any
tampering — wire it into a nightly cron and alert on non-zero exits.
See [`docs/security.md`](security.md) for the full hash-chain
threat model and key-rotation guidance.

### Signed configs

Sign the **config file itself** so production refuses to
boot against an unreviewed YAML edit:

```bash
export SENTINEL_CONFIG_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
sentinel config sign --config sentinel.yaml
# signed sentinel.yaml → sentinel.yaml.sig
sentinel config verify-signature --config sentinel.yaml
# OK — sentinel.yaml matches signature
```

Then set `dashboard.server.require_signed_config: true` and `sentinel
dashboard` will refuse to start if the sidecar is missing or stale.

### Compliance reports

Generate regulatory compliance reports from the audit trail:

```python
from sentinel.foundation.audit.compliance import ComplianceReporter

reporter = ComplianceReporter(trail=sentinel.audit)

# FCA Consumer Duty — model fairness, bias monitoring, outcomes
fca_report = reporter.generate("fca_consumer_duty", model_name="fraud_classifier", period_days=90)

# EU AI Act — risk classification, transparency documentation
eu_report = reporter.generate("eu_ai_act", model_name="fraud_classifier", period_days=90)

# Internal audit — full lifecycle report for any model version
internal = reporter.generate("internal_audit", model_name="fraud_classifier", period_days=90)
```

### Datasets and experiments

Sentinel's **dataset registry** tracks dataset metadata without moving your data. Combined with the **experiment tracker**, you get full lineage from data → training run → model version.

```python
sentinel.dataset_registry.register(
    name="claims_training",
    version="2.1.0",
    path="s3://bucket/claims_train.parquet",
    format="parquet",
    num_rows=50_000,
    num_features=42,
    tags={"split": "train"},
)

sentinel.dataset_registry.verify("claims_training@2.1.0")  # SHA-256 hash check
```

```python
tracker = sentinel.experiment_tracker
run = tracker.create_run("fraud_experiment", params={"lr": 0.001, "epochs": 50})
for epoch in range(50):
    tracker.log_metric(run.run_id, "f1", value=0.80 + epoch * 0.002, step=epoch)
tracker.end_run(run.run_id, status="completed")
tracker.link_dataset(run.run_id, "claims_training@2.1.0")
tracker.link_model(run.run_id, model_version="2.3.1")
```

Both are visible in the dashboard at `/datasets` and `/experiments`.

---

## 13. CLI reference

### Core commands

```bash
sentinel init --name MODEL --out sentinel.yaml   # Generate starter config
sentinel validate --config sentinel.yaml          # Validate config (basic)
sentinel config validate --config sentinel.yaml --strict  # Strict: check env vars + file refs
sentinel config show --config sentinel.yaml       # Show resolved config (secrets masked)
sentinel config show --config sentinel.yaml --unmask      # Show with secrets
sentinel config sign --config sentinel.yaml               # Sign config for tamper protection
sentinel config verify-signature --config sentinel.yaml   # Verify config signature
sentinel status --config sentinel.yaml            # Show model + drift status
```

### Drift and monitoring

```bash
sentinel check --config sentinel.yaml --reference data/training.parquet --current data/today.parquet
```

### Registry

```bash
sentinel registry list --config sentinel.yaml
sentinel registry show --config sentinel.yaml --version 1.0.0
```

### Audit

```bash
sentinel audit --config sentinel.yaml --type drift_detected --limit 20
sentinel audit verify --config sentinel.yaml
sentinel audit chain-info --config sentinel.yaml
```

### Deployment

```bash
sentinel deploy --config sentinel.yaml --version 1.1.0 --strategy canary --traffic 5
sentinel deploy --config sentinel.yaml --version 1.1.0 --strategy shadow
sentinel deploy --config sentinel.yaml --version 1.1.0 --strategy canary --traffic 5 --dry-run
```

### Cloud backend testing

Probe every configured cloud backend (Key Vault, registry, audit shipper, notifications) with pass/fail and elapsed time. No records are written and no models are deployed:

```bash
sentinel cloud test --config sentinel.yaml
sentinel cloud test --config sentinel.yaml --only registry
```

### Dashboard

```bash
sentinel dashboard --config sentinel.yaml
sentinel dashboard --config sentinel.yaml --host 0.0.0.0 --port 8080 --open
```

### Shell completion

```bash
sentinel completion bash >> ~/.bashrc
sentinel completion zsh  >> ~/.zshrc
sentinel completion fish >> ~/.config/fish/completions/sentinel.fish
```

---

## 14. Where to go next

- **[Config reference](config-reference.md)** — every YAML field with type, default, and example
- **[Architecture](architecture.md)** — the seven-layer stack and how to extend it
- **[Security](security.md)** — tamper-evident audit trails, RBAC, CSRF, rate limiting, Bearer JWT, signed configs
- **[Azure integration](azure.md)** — Key Vault secrets, Azure ML registry, Blob audit shipping, deployment targets
- **[`configs/examples/`](../configs/examples/)** — production-shaped configs for each domain (tabular, time series, NLP, recommendation, graph, RAG, multi-agent)

---

## Troubleshooting

**`Config validation error` on `sentinel validate`**
The error message names the field and the violation. Common causes: typo in a `Literal` field (e.g. `method: ks_test` instead of `ks`), missing required `model.name`, or environment variable not set. When `extends:` is in play, the error line includes the originating file in `[from parent.yaml]` so you can tell which level of the chain owns the broken field.

**`references unset environment variable(s)` on `validate --strict`**
You're running with `--strict` and a `${VAR}` token has no matching env var (or the env var is set but empty — `VAR=` from a shell file is the most common cause). The error names the JSON path (e.g. `alerts.channels.0.webhook_url`) and the variable. Either export the variable or add a `${VAR:-default}` fallback in the YAML.

**Drift always reports `not fitted`**
You haven't called `fit_baseline()` yet, and `model.baseline_dataset` is not set in the config. Either fit programmatically or provide a baseline path in the config.

**Alerts never fire**
Check `sentinel audit --type alert_sent` to see if the engine is dispatching. If alerts are dispatched but never delivered, check the channel-specific result: a Slack channel with no `webhook_url` is created in disabled mode and silently logs `channel disabled`.

**`sentinel audit verify` exits 1 with `hash_mismatch` / `broken_chain`**
The trail has been edited or had events inserted/deleted. Inspect
the offending file (`./audit/<date>.jsonl`) against the chain head
reported by `sentinel audit chain-info`. If the tampering was a
legitimate operator action (e.g. retention pruning), regenerate the
chain by archiving the old trail and starting fresh — the SDK does
not silently re-key in place. See `docs/security.md` section 1.

**`ConfigSignatureError: config signature does not match`**
The resolved config differs from what was signed. Re-run
`sentinel config sign` after the YAML edit, or revert the edit if it
wasn't intentional. Note that the signature is computed against the
**resolved** config, so a change to a parent file in an `extends:`
chain — or to an `${VAR}` value — will also invalidate the signature.

**`SentinelError: set_model_for_explanations() must be called first`**
You called `explain()`, `explain_global()`, or `explain_cohorts()` before attaching a model. Call `sentinel.set_model_for_explanations(model, feature_names, background_data)` once before using any explain method.

**`ModuleNotFoundError: scipy`** (or sklearn, presidio, statsmodels, etc.)
You need an optional extra. The most common ones:

```bash
pip install "sentinel-mlops[drift]"        # scipy + sklearn for advanced detectors
pip install "sentinel-mlops[explain]"      # shap, lime
pip install "sentinel-mlops[llmops]"       # tiktoken, openai, presidio
pip install "sentinel-mlops[agentops]"     # opentelemetry
pip install "sentinel-mlops[timeseries]"   # statsmodels, pmdarima
pip install "sentinel-mlops[graph]"        # networkx, torch-geometric
pip install "sentinel-mlops[dashboard]"    # FastAPI, Plotly, Jinja2
pip install "sentinel-mlops[all]"          # everything (lightweight)
```

**`CohortAnalysisConfig: cohort_column is required when cohort_analysis is enabled`**
You set `cohort_analysis.enabled: true` but didn't specify which feature column to use for cohort segmentation. Add `cohort_column: your_column_name` to the `cohort_analysis` section.

**Dashboard shows empty cohort/explanation pages**
Cohort data requires `log_prediction()` calls with either `cohort_id` or a valid `cohort_column` in the features dict. Explainability requires `set_model_for_explanations()` followed by at least one `explain_global()` or `explain()` call to populate the page.

**`sentinel cloud test` fails for a specific backend**
Use `--only registry` (or `keyvault`, `audit`, `notifications`) to isolate the failing backend. Check that the relevant environment variables are set and that your credentials have the required RBAC permissions. See [`docs/azure.md`](azure.md) for RBAC snippets and a troubleshooting table.
