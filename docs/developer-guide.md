# Project Sentinel — Step-by-Step Developer Guide

> **Before you start — which guide do you actually need?**
>
> - **You are integrating Sentinel into your own model-serving app and want a step-by-step tutorial.** → You are in the right place. Keep reading.
> - **You are joining the Sentinel project as a contributor and need to understand the codebase itself** — what every folder does, what every YAML field does, how to add a new drift detector / channel / deployment target — → Read [`codebase-guide.md`](codebase-guide.md) first. It is a comprehensive tour of the repository, designed for new developers who have never seen the project before.
> - **You want a 5-minute conceptual overview** of what Sentinel is. → Start with [`quickstart.md`](quickstart.md).
> - **You need the full YAML field reference.** → [`config-reference.md`](config-reference.md).

Target: ship one model with full monitoring, alerting, audit trail, and dashboard in a single afternoon. Every step tells you exactly what to run, what to expect, and how to verify.

This guide is the hands-on companion to [`quickstart.md`](quickstart.md) (conceptual overview) and [`config-reference.md`](config-reference.md) (every YAML field). Follow these steps in order — each one builds on the previous.

---

## Step 0 — Prerequisites (2 min)

You need:

- Python 3.10, 3.11, 3.12, or 3.13 (`python --version`)
- A working virtualenv or conda env
- A model you can import — any sklearn / xgboost / lightgbm model works
- A reference dataset (the data your model was trained on, or a representative sample)
- A Slack incoming webhook URL (optional but recommended for step 7)

Verify:

```bash
python --version        # ≥ 3.10
which pip               # should point inside your venv
```

---

## Step 1 — Install Sentinel (1 min)

```bash
pip install "sentinel-mlops[all,dashboard]"
```

Verify:

```bash
sentinel --version      # → sentinel, version 0.1.0
python -c "from sentinel import SentinelClient; print('ok')"
```

If you get a `typing_extensions` / `pydantic_core` import error, force-reinstall typing_extensions:

```bash
pip install --force-reinstall --no-deps "typing_extensions>=4.13"
```

> **Python 3.13 note.** The `[all]` extra is intentionally light — it does not pull `torch` / `sentence-transformers` / `presidio` so it resolves on Python 3.13 (where torch has no wheels). If you explicitly need embedding-based semantic drift, jailbreak detection, topic fencing, or PII analysis, install `[ml-extras]` on top — but only on Python ≤3.12.

---

## Step 2 — Generate a starter config (1 min)

```bash
sentinel init --name claims_fraud_v2 --out sentinel.yaml
```

Verify:

```bash
sentinel validate --config sentinel.yaml
# → OK — model=claims_fraud_v2 domain=tabular
```

Open `sentinel.yaml` in your editor. You'll see `model`, `drift`, `alerts`, `audit`, `deployment` sections already populated with sensible defaults. Do **not** edit it yet — we'll make it work end-to-end first, then tune.

Commit it:

```bash
git add sentinel.yaml
git commit -m "chore: add sentinel config for claims_fraud_v2"
```

From this point on, `sentinel.yaml` is version-controlled alongside your model code.

---

## Step 3 — Wire `SentinelClient` into your serving code (5 min)

Create a minimal FastAPI app if you don't have one, or edit your existing serving module. Add exactly three things: import, client init, and a `log_prediction` call.

```python
# app.py
from fastapi import FastAPI
from sentinel import SentinelClient
import joblib

app = FastAPI()
model = joblib.load("model.pkl")

# Module-level singleton — thread-safe, safe for FastAPI workers
sentinel = SentinelClient.from_config("sentinel.yaml")

@app.post("/predict")
async def predict(features: dict):
    pred = model.predict([list(features.values())])[0]
    sentinel.log_prediction(features=features, prediction=pred)
    return {"prediction": int(pred)}

@app.get("/health")
async def health():
    return sentinel.status()
```

Verify it starts without exceptions:

```bash
uvicorn app:app --port 8001 &
curl -s http://127.0.0.1:8001/health | python -m json.tool
```

You should see a JSON blob with `model`, `version` (shows "unregistered" until step 4), `drift_status`, and `audit_event_count`.

---

## Step 4 — Register your model (2 min)

Either from code:

```python
sentinel.register_model_if_new(
    model=model,
    version="1.0.0",
    metadata={
        "framework": "sklearn",
        "trained_on": "2026-03-01",
        "metrics": {"accuracy": 0.91, "f1": 0.88, "auc": 0.94},
    },
)
```

Or from the CLI (useful for CI/CD):

```bash
sentinel registry list --config sentinel.yaml
```

Verify:

```bash
sentinel registry list
# → claims_fraud_v2: 1.0.0
sentinel registry show --version 1.0.0
```

---

## Step 5 — Fit the baseline (3 min)

Sentinel needs a reference distribution to compare incoming data against. Fit it **once** per model version, from a representative sample.

```python
import pandas as pd

reference = pd.read_parquet("data/training_sample.parquet")   # or CSV, or numpy array
sentinel.fit_baseline(reference)
```

Or from the CLI:

```bash
sentinel check \
  --config sentinel.yaml \
  --reference data/training_sample.parquet \
  --current data/training_sample.parquet
```

Verify: the JSON output should show `"is_drifted": false` and per-feature `feature_scores` near zero. This confirms PSI/KS is computing correctly on your data shape.

---

## Step 6 — Prove drift detection works (3 min)

Create a deliberately-shifted dataset and verify Sentinel catches it. This is your "smoke test" before trusting the monitoring in production.

```python
import numpy as np
import pandas as pd

reference = pd.read_parquet("data/training_sample.parquet")

# Shift the mean of two features
drifted = reference.copy()
drifted["amount_log"] += 1.5
drifted["txn_velocity"] *= 2.0

sentinel.fit_baseline(reference)
report = sentinel.check_drift(drifted)

print(f"drifted={report.is_drifted}  severity={report.severity}")
print("top drifting features:")
for feat, score in sorted(report.feature_scores.items(), key=lambda x: -x[1])[:5]:
    print(f"  {feat}: {score:.3f}")
```

Expected output:

```
drifted=True  severity=high
top drifting features:
  amount_log: 0.847
  txn_velocity: 0.612
  ...
```

If `is_drifted=False` on a deliberately drifted dataset, lower `drift.data.threshold` in `sentinel.yaml` (default is 0.2; try 0.1).

---

## Step 7 — Turn on alerts (3 min)

Edit `sentinel.yaml`:

```yaml
alerts:
  channels:
    - type: slack
      webhook_url: ${SLACK_WEBHOOK_URL}
      channel: "#ml-alerts"
  policies:
    cooldown: 1h
```

Export the webhook and re-run the drift smoke test:

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
python -c "
from sentinel import SentinelClient
import pandas as pd
c = SentinelClient.from_config('sentinel.yaml')
ref = pd.read_parquet('data/training_sample.parquet')
drifted = ref.copy(); drifted['amount_log'] += 1.5
c.fit_baseline(ref)
c.check_drift(drifted)
"
```

Verify: a Slack message shows up in `#ml-alerts` with the drift severity and top drifting features. If nothing arrives, check:

```bash
sentinel audit --type alert_sent --limit 5
```

If the audit trail shows `alert_sent` events but Slack is quiet, your webhook URL is wrong. If there are no `alert_sent` events, the drift didn't trigger — go back to step 6.

---

## Step 8 — Verify the audit trail (1 min)

Every event Sentinel fires is immutably logged. Tail it:

```bash
sentinel audit --limit 20
```

You should see a stream of events: `model_registered`, `baseline_fitted`, `drift_checked`, `alert_sent`. This is the same data that compliance auditors will read later. Nothing else to configure — it's on by default.

Filter by type:

```bash
sentinel audit --type drift_checked --limit 5
sentinel audit --type alert_sent --limit 5
```

---

## Step 9 — Launch the dashboard (1 min)

```bash
sentinel dashboard --config sentinel.yaml --port 8000 --open
```

Your browser opens to `http://127.0.0.1:8000`. Click through:

- `/` — overview with model status + recent alerts
- `/drift` — drift timeline chart
- `/features` — feature health table
- `/registry` — your registered model versions
- `/audit` — filterable event log
- `/compliance` — framework coverage
- `/cohorts` — cohort comparison with disparity detection
- `/cohorts/{id}` — single cohort deep-dive
- `/explanations` — global feature importance chart

All data is read live from the same `SentinelClient` your app is using. **There is no separate database to provision.**

Keep the dashboard running in a second terminal while you work — you'll see new events land as you hit `/predict` in your app.

---

## Step 10 — Set up feature health monitoring (2 min)

Add to `sentinel.yaml`:

```yaml
feature_health:
  importance_method: shap        # or: permutation | builtin
  alert_on_top_n_drift: 3        # alert if top-3 important features drift
  recalculate_importance: weekly
```

From code:

```python
report = sentinel.get_feature_health()
for feat in report.ranked_features[:5]:
    print(f"{feat.name}: importance={feat.importance:.3f} drift={feat.drift_score:.3f}")
```

Verify the dashboard's `/features` page now shows ranked features with drift scores.

---

## Step 11 — Add a second model version and deploy it safely (5 min)

Register a new version:

```python
sentinel.register_model_if_new(
    model=model_v2, version="1.1.0",
    metadata={"metrics": {"accuracy": 0.93, "f1": 0.90}},
)
```

Add a deployment section to `sentinel.yaml`:

```yaml
deployment:
  strategy: canary
  canary:
    initial_traffic_pct: 5
    ramp_steps: [5, 25, 50, 100]
    ramp_interval: 1h
    rollback_on:
      error_rate_increase: 0.02
      latency_p99_increase_ms: 50
```

Start the canary from the CLI:

```bash
sentinel deploy --version 1.1.0 --strategy canary --traffic 5
```

**Dry-run first** — validate the deployment config without actually starting:

```bash
sentinel deploy --version 1.1.0 --strategy canary --traffic 5 --dry-run
```

This prints a JSON report showing what *would* happen (target, strategy, traffic split, rollback rules) and exits. Use it in CI to catch deployment config errors before hitting production.

**Deployment targets** — the `deployment.target` field controls *where* the model is deployed. Supported targets:

```yaml
deployment:
  strategy: canary
  target: local                   # default — no-op target for dev/testing
  # target: azure_ml_endpoint    # Azure ML managed endpoint
  # target: azure_app_service    # Azure App Service slot swap
  # target: aks                  # Azure Kubernetes Service
  # target: sagemaker_endpoint   # AWS SageMaker
  # target: vertex_ai_endpoint   # Google Vertex AI
```

The strategy × target combination is validated at config load time. For example, `canary` + `azure_app_service` is rejected because App Service uses slot swaps (blue-green only). Use `sentinel validate` to catch mismatches early.

Verify:

```bash
sentinel audit --type deployment_started --limit 1
```

The dashboard's `/deployments` page now shows the active canary with its traffic ramp.

---

## Step 12 — Wire Sentinel into CI (3 min)

Add a pre-merge check to your CI workflow that validates the config on every PR:

```yaml
# .github/workflows/sentinel.yml
- run: pip install "sentinel-mlops[all]"
- run: sentinel validate --config sentinel.yaml
- run: sentinel check --config sentinel.yaml \
         --reference data/training_sample.parquet \
         --current data/latest_production_sample.parquet
```

Any PR that breaks the config fails the build. Any PR that would silently introduce drift fails the build.

---

## Step 13 — Ship to production (5 min)

**Three changes** for production readiness:

**13a. Move the audit trail to cloud storage:**

```yaml
# Azure Blob
audit:
  storage: azure_blob
  retention_days: 2555
  azure_blob:
    account_url: https://sentinelaudit.blob.core.windows.net
    container: sentinel-audit

# AWS S3
audit:
  storage: s3
  retention_days: 2555
  s3:
    bucket: sentinel-audit-prod
    region: eu-west-1

# Google Cloud Storage (requires pip install "sentinel-mlops[gcp]")
audit:
  storage: gcs
  retention_days: 2555
  gcs:
    bucket: sentinel-audit-prod
    project: my-gcp-project
```

**13b. Move the model registry to your cloud ML platform:**

```yaml
registry:
  backend: azure_ml            # or: mlflow | s3
  workspace: ${AZURE_ML_WORKSPACE}
```

**13c. Pin the SDK version in your `requirements.txt`:**

```
sentinel-mlops[all,dashboard]==0.1.0
```

Redeploy your app. Run the drift smoke test from step 6 once against the production config to verify cloud backends are wired up:

```bash
SENTINEL_CONFIG_PATH=/etc/sentinel/claims.yaml sentinel status
```

---

## Step 14 — Level up: pick **one** advanced capability (optional)

Once steps 1–13 are done, add one of these — each is a single-config-section change, no code:

| Capability | YAML section to add | What you get |
|---|---|---|
| Concept drift (streaming) | `drift.concept` | DDM/EDDM/ADWIN/Page-Hinkley fed per-prediction from `log_prediction(actual=...)` |
| Count-based auto-drift | `drift.auto_check` | Auto-trigger `check_drift()` every N predictions — no scheduler needed |
| Retrain + auto-deploy | `retraining` with `deploy_on_promote: true` | Drift → pipeline → validate → approve → promote → auto-deploy |
| Business KPI linking | `business_kpi.mappings` | Model metric → business outcome correlation |
| Cost monitoring | `cost_monitor` | Latency/throughput/cost-per-prediction alerts |
| Multi-model cascade | `model_graph.dependencies` | Upstream drift → downstream alerts |
| Cohort analysis | `cohort_analysis` | Per-subpopulation performance + fairness disparity detection |
| Multi-level explainability | `explainability` | Global, cohort, and row-level SHAP/permutation feature importance |
| GCS audit storage | `audit.storage: gcs` + `audit.gcs` | Ship audit logs to Google Cloud Storage |
| Dataset registry | `dataset_registry` (see below) | Metadata-only dataset tracking with SHA-256 verification |
| Experiment tracking | `experiment_tracker` (see below) | Named experiments, nested runs, metric history, search, run comparison |
| Custom guardrails DSL | `llmops.guardrails` with `type: custom` | Declarative rule-based guardrails — no code needed |
| Plugin guardrails | `llmops.guardrails` with `type: plugin` | Bring your own guardrail class, loaded dynamically |
| Compliance reporting | `audit.compliance_frameworks` (see below) | FCA Consumer Duty, EU AI Act, and internal audit reports from your audit trail |
| Domain adapters | `model.domain` (see below) | Seasonality-aware drift, NLP vocab drift, recsys beyond-accuracy metrics, graph topology drift |
| Multi-backend registry | `registry` (see below) | Azure ML, MLflow, SageMaker, Vertex AI, Databricks model registry backends |
| Deployment targets | `deployment.target` (see below) | Azure ML Endpoint, App Service, AKS, SageMaker, Vertex AI deployment targets |
| Config signing | `sentinel config sign` (see below) | HMAC-SHA256 config signatures — tamper-proof YAML for regulated environments |
| `--dry-run` deploy | `sentinel deploy --dry-run` (see below) | Validate deployment config without starting a real deploy |
| LLMOps (for RAG apps) | `llmops` (see below) | Guardrails, prompt versions, token economics |
| AgentOps (for agent apps) | `agentops` (see below) | Trace monitoring, tool audit, loop detection, budget guard |

### If you're adding LLMOps (RAG / chat app)

```yaml
llmops:
  enabled: true
  mode: rag
  guardrails:
    input:
      - { type: pii_detection, action: redact }
      - { type: jailbreak_detection, action: block }
    output:
      - { type: toxicity, threshold: 0.7, action: block }
      - { type: groundedness, min_score: 0.6, action: warn }
  token_economics:
    budgets: { daily_max_cost: 500.00, per_query_max_cost: 0.50 }
```

Then in code:

```python
llmops = sentinel.llmops
in_check = llmops.check_input(query)
if in_check.blocked:
    return {"error": in_check.reason}
# ... your LLM call ...
out_check = llmops.check_output(response, context={"chunks": chunks})
llmops.log_call(
    prompt_name="claims_qa", prompt_version="1.0",
    query=query, response=response, context_chunks=chunks,
    model="gpt-4o-mini",
    input_tokens=usage.in_, output_tokens=usage.out_,
    guardrail_results={"input": in_check, "output": out_check},
)
```

### If you're adding AgentOps (LangGraph / Semantic Kernel agent)

```yaml
agentops:
  enabled: true
  tracing: { backend: local, sample_rate: 1.0 }
  safety:
    loop_detection: { max_iterations: 50, max_repeated_tool_calls: 5 }
    budget: { max_tokens_per_run: 50000, max_cost_per_run: 5.00, max_time_per_run: 300s }
```

Then wrap your compiled graph with zero-code instrumentation:

```python
from sentinel.agentops.integrations import LangGraphMiddleware
from sentinel.agentops.trace.tracer import AgentTracer

tracer = AgentTracer()
middleware = LangGraphMiddleware(tracer)

# Wrap your compiled LangGraph graph — no code changes to your agent
monitored = middleware.wrap(compiled_graph, agent_name="claims_processor")

# invoke() and ainvoke() are both traced automatically
result = monitored.invoke({"input": "process claim 12345"})

# Every node in the graph becomes a traced span
trace = tracer.get_last_trace()
print(f"Steps: {len(trace.spans)}, Agent: {trace.agent_name}")
```

The middleware works via duck-typing — it wraps any object with `stream()` / `astream()` methods. It does not import `langgraph` at module level, so it works regardless of whether you have the `langgraph` package installed.

### If you're adding count-based drift auto-check

```yaml
drift:
  auto_check:
    enabled: true
    every_n_predictions: 500    # auto-check every 500 predictions
```

This triggers `check_drift()` automatically after every 500 predictions in a background daemon thread. No scheduler needed — the check is driven purely by prediction volume. The counter resets on both auto-check and manual `check_drift()` calls.

### If you're adding streaming concept drift

```yaml
drift:
  concept:
    method: ddm          # ddm | eddm | adwin | page_hinkley
    warning_level: 2.0
    drift_level: 3.0
    min_samples: 100
```

Then pass `actual` values in your `log_prediction()` calls:

```python
sentinel.log_prediction(features=X, prediction=y_pred, actual=y_true)
```

The concept drift detector is fed a per-prediction error signal (0.0 for correct, 1.0 for incorrect in classification; absolute error for regression). The streaming state is merged into the `DriftReport` returned by `check_drift()`.

### If you're adding cohort-based performance analysis

```yaml
cohort_analysis:
  enabled: true
  cohort_column: customer_segment    # auto-derive cohort from this feature
  min_samples_per_cohort: 30
  max_cohorts: 50
  disparity_threshold: 0.1           # flag cohorts >10% worse than global
  buffer_size: 5000
```

```python
# Log predictions with a cohort label
sentinel.log_prediction(
    features=X, prediction=y_pred, actuals=y_true,
    cohort_id="premium_customers"        # explicit cohort
)

# Or let the SDK derive the cohort from the feature column
sentinel.log_prediction(features=X, prediction=y_pred, actuals=y_true)
# (uses cohort_column from config)

# Get per-cohort performance
report = sentinel.get_cohort_report("premium_customers")
# report.accuracy, report.count, report.mean_prediction

# Cross-cohort comparison with disparity detection
comparison = sentinel.compare_cohorts()
for alert in comparison.disparity_alerts:
    print(f"Cohort {alert.cohort_id}: accuracy {alert.accuracy:.3f} "
          f"({alert.deviation_pct:+.1f}% from global)")
```

The cohort comparison page is available at `/cohorts` in the dashboard.

---

### If you're adding multi-level explainability

```python
import numpy as np

# Row-level: explain individual predictions
report = sentinel.explain(X)
# report.attributions → dict[str, float] per row

# Global: mean |SHAP| across all rows → feature ranking
global_report = sentinel.explain_global(X)
# global_report.feature_importances → {"income": 0.42, "age": 0.31, ...}

# Cohort: per-cohort feature importance comparison
cohort_labels = np.array(["young"] * 200 + ["senior"] * 300)
cohort_report = sentinel.explain_cohorts(X, cohort_labels)
# cohort_report.cohort_importances → {"young": {...}, "senior": {...}}
```

All three levels use the same SHAP backend (or permutation fallback when SHAP is unavailable). Results are logged to the audit trail. The global importance chart is visible at `/explanations` in the dashboard.

---

### If you're using the reset and maintenance APIs

Sentinel provides public methods for resetting state without restarting:

```python
# Reset drift baseline (e.g., after a retrain)
sentinel.reset_drift_baseline(reference=new_training_data)

# Clear accumulated cohort data
sentinel.clear_cohort_data()

# Access sub-components via public properties
analyzer = sentinel.cohort_analyzer      # CohortAnalyzer instance (or None)
engine = sentinel.explainability_engine  # ExplainabilityEngine instance (or None)
```

These are useful in CI/CD pipelines where you retrain and re-baseline without restarting the serving process.

---

### If you're adding retrain-to-deploy automation

```yaml
retraining:
  trigger: drift_confirmed
  pipeline: azureml://pipelines/retrain_fraud
  deploy_on_promote: true       # auto-deploy after approval
  approval:
    mode: hybrid
    auto_promote_if:
      metric: f1
      improvement_pct: 2.0

deployment:
  strategy: canary
  canary:
    ramp_steps: [5, 25, 50, 100]
    ramp_interval: 1h
```

When `deploy_on_promote: true`, the retrain orchestrator automatically calls `DeploymentManager.start()` after a model is promoted — whether auto-approved or manually approved. No manual `client.deploy()` call is needed.

---

### If you're adding dataset registry

The dataset registry tracks dataset **metadata** without moving your data. Reference datasets by `name@version` anywhere in the SDK. Access it via the `client.datasets` property.

```python
datasets = sentinel.datasets    # DatasetRegistry instance

# Register a dataset version
datasets.register(
    name="claims_training",
    version="2.1.0",
    path="s3://bucket/claims_train.parquet",
    format="parquet",
    num_rows=50_000,
    num_features=42,
    tags={"split": "train", "quarter": "Q1-2026"},
)

# Look up and search
ds = datasets.get("claims_training@2.1.0")
versions = datasets.list_versions("claims_training")
names = datasets.list_names()            # → ["claims_training", "holdout_set", ...]
results = datasets.search(tags={"split": "train"})

# Compare two versions
diff = datasets.compare("claims_training@2.0.0", "claims_training@2.1.0")

# Link to experiments and models
datasets.link_to_experiment("claims_training@2.1.0", experiment="fraud_v3")
datasets.link_to_model("claims_training@2.1.0", model_version="2.3.1")

# Verify data integrity (SHA-256 content hash)
datasets.verify("claims_training@2.1.0")
```

`DatasetVersion` stores: `name`, `version`, `path`, `format`, `split`, `num_rows`, `num_features`, `content_hash` (SHA-256), `schema`, `tags`, and `lineage`. The dashboard page at `/datasets` lists all registered versions.

---

### If you're adding enhanced experiment tracking

The experiment tracker provides named experiments, nested runs, metric time-series, search, and comparison. It is backward-compatible with the old `foundation/experiments/tracker.py` API. Access it via the `client.experiments` property.

```python
tracker = sentinel.experiments    # ExperimentTracker instance

# Create an experiment (a named group of runs)
exp = tracker.create_experiment("fraud_detection_v3")

# Create a run within the experiment
run = tracker.create_run(
    "fraud_detection_v3",
    params={"lr": 0.001, "epochs": 50, "model": "xgboost"},
)

# Log metrics with time-series history (value + step)
for epoch in range(50):
    tracker.log_metric(run.run_id, "f1", value=0.80 + epoch * 0.002, step=epoch)
    tracker.log_metric(run.run_id, "loss", value=1.0 - epoch * 0.015, step=epoch)

# End the run
tracker.end_run(run.run_id, status="completed")

# Nested runs via parent_run_id
child = tracker.create_run("fraud_detection_v3", parent_run_id=run.run_id)

# Search with AND filter syntax
results = tracker.search_runs("fraud_detection_v3",
    filter_expr="metrics.f1 > 0.85 AND params.lr < 0.01")

# OR syntax — find runs matching either condition
results = tracker.search_runs("fraud_detection_v3",
    filter_expr="metrics.f1 > 0.85 OR params.model = 'xgboost'")

# Compare multiple runs side-by-side (params diff, latest metrics, duration)
comparison = tracker.compare_runs([run_a.run_id, run_b.run_id])
# comparison → {"params_diff": {...}, "metrics_latest": {...}, "status": {...}}

# Link to datasets and models
tracker.log_dataset(run.run_id, "claims_training@2.1.0")
tracker.link_to_model(run.run_id, model_name="claims_fraud", model_version="2.3.1")
```

The dashboard at `/experiments` shows experiment lists, run tables, and metric history charts (interactive Plotly).

---

### If you're adding custom guardrails (DSL or plugin)

Custom guardrails let you define validation rules declaratively in YAML. No Python code needed for common patterns.

**DSL guardrails** (`type: custom`):

```yaml
llmops:
  guardrails:
    input:
      - type: custom
        name: input_quality_gate
        rules:
          - { type: min_length, value: 10 }
          - { type: regex_absent, value: "DROP TABLE|DELETE FROM" }
          - { type: keyword_absent, value: "ignore previous instructions" }
          - { type: not_empty }
        combine: "any"       # block on FIRST rule failure (strict)
        action: block
    output:
      - type: custom
        name: output_format_check
        rules:
          - { type: max_length, value: 5000 }
          - { type: json_schema, value: '{"type": "object", "required": ["summary"]}' }
        combine: "all"       # block only if ALL rules fail (lenient)
        action: warn
```

**Available DSL rule types:**

| Rule type | Description | `value` field |
|---|---|---|
| `regex_match` | Content must match regex | Regex pattern |
| `regex_absent` | Content must NOT match regex | Regex pattern |
| `keyword_present` | Content must contain keyword | Keyword string |
| `keyword_absent` | Content must NOT contain keyword | Keyword string |
| `min_length` | Minimum character length | Integer |
| `max_length` | Maximum character length | Integer |
| `json_schema` | Content must validate against JSON Schema | JSON Schema string |
| `sentiment` | Content sentiment check | Threshold float |
| `language` | Content language detection | Language code |
| `word_count` | Minimum/maximum word count | Integer |
| `not_empty` | Content must not be empty/whitespace | (none) |

**`combine` modes:**
- `"any"` (strict): Block on the **first** rule failure. Use for hard safety constraints.
- `"all"` (lenient): Block only if **every** rule fails. Use for soft quality hints.

**Plugin guardrails** (`type: plugin`):

For fully custom logic, implement a Python class and load it dynamically:

```yaml
llmops:
  guardrails:
    output:
      - type: plugin
        module: "my_company.guardrails"
        class_name: "ComplianceGuardrail"
        config: { strict_mode: true, max_risk_score: 0.7 }
        action: warn
```

Your class must implement `check(content, context) -> GuardrailResult`:

```python
# my_company/guardrails.py
from sentinel.llmops.guardrails.base import BaseGuardrail, GuardrailResult

class ComplianceGuardrail(BaseGuardrail):
    def __init__(self, config: dict):
        self.strict_mode = config.get("strict_mode", False)
        self.max_risk_score = config.get("max_risk_score", 0.8)

    def check(self, content: str, context: dict | None = None) -> GuardrailResult:
        risk_score = self._assess_risk(content)
        if risk_score > self.max_risk_score:
            return GuardrailResult(passed=False, reason=f"Risk score {risk_score:.2f}")
        return GuardrailResult(passed=True)
```

Both custom DSL and plugin guardrails integrate seamlessly with the existing `GuardrailPipeline` — they run in the same chain as built-in guardrails, respect `action: block|warn`, and log results to the audit trail.

---

### If you're adding compliance reporting

Sentinel's audit trail already logs every model event. `ComplianceReporter` reads those events and generates structured reports for FCA Consumer Duty, EU AI Act, and internal audit.

```python
from sentinel.foundation.audit.compliance import ComplianceReporter

reporter = ComplianceReporter(trail=sentinel.audit, risk_level="high")

# FCA Consumer Duty — fairness, bias monitoring, human oversight
fca_report = reporter.generate(
    "fca_consumer_duty",
    model_name="claims_fraud_v2",
    period_days=90,
)
print(fca_report["fairness_monitoring"])   # cohort analyses + disparity detection
print(fca_report["human_oversight"])        # approval decisions + details
print(fca_report["outcome_tracking"])
print(fca_report["model_governance"])       # versions deployed, retrain triggers

# EU AI Act — risk classification, transparency, monitoring
eu_report = reporter.generate(
    "eu_ai_act",
    model_name="claims_fraud_v2",
    period_days=90,
)
print(eu_report["risk_classification"])
print(eu_report["transparency"])            # registered versions + deployment history
print(eu_report["monitoring_summary"])      # drift detections, retrains, alerts
print(eu_report["human_oversight"])

# Internal audit — full event dump for any model over any period
internal = reporter.generate(
    "internal_audit",
    model_name="claims_fraud_v2",
    period_days=365,
)
print(f"Total events: {internal['event_counts']}")
print(f"Period: {internal['first_event']} → {internal['last_event']}")
```

The compliance report page is visible at `/compliance` in the dashboard. `generate()` returns a plain `dict` so you can serialise to JSON, upload to a GRC tool, or attach to an email.

Verify: run `reporter.generate("fca_consumer_duty", ...)` after logging some predictions with cohort labels — the `fairness_monitoring` section should show cohort analyses.

---

### If you're adding domain adapters

Sentinel auto-selects domain-appropriate drift detectors and quality metrics based on the `model.domain` field. Switch one line in YAML to change from tabular monitoring to time series, NLP, recommendation, or graph ML monitoring.

```yaml
model:
  name: demand_forecast_v3
  domain: timeseries        # tabular | timeseries | nlp | recommendation | graph
  type: regression
```

**What each domain gives you:**

| Domain | Drift detection | Quality metrics | Extra install |
|---|---|---|---|
| `tabular` (default) | PSI, KS, chi-squared | Accuracy, F1, AUC | none |
| `timeseries` | Seasonality-aware calendar tests, temporal covariate shift, ACF monitoring, stationarity (ADF/KPSS) | MASE, MAPE, coverage, interval width, directional accuracy, per-horizon tracking | `pip install "sentinel-mlops[timeseries]"` (statsmodels, pmdarima) |
| `nlp` | Vocabulary drift (OOV rate), embedding space shift (MMD), label distribution, text statistics | Token F1, span exact match, macro/micro F1, perplexity trend | `pip install "sentinel-mlops[nlp-domain]"` (sentence-transformers, spacy) |
| `recommendation` | Item/user distribution shift, cold-start ratio, long-tail collapse | NDCG@K, MAP@K, coverage, diversity, novelty, serendipity, popularity bias, group fairness | `pip install "sentinel-mlops[recommendation]"` (implicit, recbole) |
| `graph` | Topology drift (degree distribution, density, clustering coefficient, connected components), node/edge feature drift | AUC-ROC, Hits@K, MRR, modularity, embedding isotropy | `pip install "sentinel-mlops[graph]"` (networkx, torch-geometric) |

No code changes are needed — the adapter is resolved at `SentinelClient` init time. All domain-specific drift results flow through the same notification engine, audit trail, and dashboard as tabular drift. If no `domain` is specified, the default is `tabular` — existing configs are fully backward-compatible.

Verify:

```bash
sentinel validate --config sentinel.yaml
# → OK — model=demand_forecast_v3 domain=timeseries
```

---

### If you're adding multi-backend model registry

The `registry` config section controls where model artifacts and metadata are stored. Six backends are supported:

```yaml
# Local filesystem (default — for dev/testing)
registry:
  backend: local
  path: ./registry

# Azure ML model registry
registry:
  backend: azure_ml
  subscription_id: ${AZURE_SUBSCRIPTION_ID}
  resource_group: ${AZURE_RESOURCE_GROUP}
  workspace_name: ${AZURE_ML_WORKSPACE}

# MLflow-compatible (self-hosted or managed)
registry:
  backend: mlflow
  tracking_uri: http://mlflow.internal:5000

# AWS SageMaker
registry:
  backend: sagemaker
  region_name: us-east-1
  role_arn: ${SAGEMAKER_ROLE_ARN}
  s3_bucket: ${SAGEMAKER_BUCKET}

# Google Vertex AI
registry:
  backend: vertex_ai
  project: ${GCP_PROJECT}
  location: us-central1
  gcs_bucket: ${GCS_BUCKET}

# Databricks Unity Catalog
registry:
  backend: databricks
  host: ${DATABRICKS_HOST}
  token: ${DATABRICKS_TOKEN}
  catalog: ml
  schema_name: default
```

All backends implement the same `BaseRegistryBackend` interface — `register()`, `get()`, `list_versions()`, and `get_baseline()` work identically regardless of backend. Switch backends by changing the YAML; no code changes needed.

Verify:

```bash
sentinel cloud test --config sentinel.yaml --only registry
# → registry: OK  (elapsed: 340ms)
```

---

### If you're adding alert escalation chains

Escalation chains fire increasingly aggressive alerts over time if an alert hasn't been acknowledged. The `after` field is a duration string. Channels accumulate — later steps add channels on top of earlier ones.

```yaml
alerts:
  channels:
    - type: slack
      webhook_url: ${SLACK_WEBHOOK_URL}
      channel: "#ml-alerts"
    - type: pagerduty
      routing_key: ${PD_ROUTING_KEY}
  policies:
    cooldown: 1h
    escalation:
      - after: 0m                       # immediately
        channels: [slack]
        severity: [medium, high, critical]
      - after: 30m                      # 30 min with no ack
        channels: [slack, pagerduty]    # adds PagerDuty
        severity: [high, critical]
      - after: 2h                       # 2 hours with no ack
        channels: [slack, pagerduty]
        severity: [critical]            # only critical escalates this far
```

**How it works:**
1. When a drift alert fires at severity `high`, the first matching step sends to Slack immediately.
2. If the alert is not acknowledged within 30 minutes, the second step fires — now Slack *and* PagerDuty receive the alert.
3. The `cooldown` policy prevents re-alerting for the *same issue* within 1 hour, but escalation steps are independent of cooldown — they fire on schedule regardless.

The escalation timeline is visible in the audit trail (`sentinel audit --type alert_sent`) and on the dashboard's overview page.

---

### If you're adding config signing

Config signing prevents tampering in regulated environments. The signature is computed against the **resolved** config (after `extends:` chains and `${VAR}` substitution), so it's valid regardless of how the config was assembled.

```bash
# Generate a signing key (store this securely — Key Vault, secrets manager, etc.)
export SENTINEL_CONFIG_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')

# Sign the config — writes a .sig file alongside the YAML
sentinel config sign --config sentinel.yaml

# Verify the signature
sentinel config verify-signature --config sentinel.yaml
```

To enforce signature verification on dashboard startup, add to your YAML:

```yaml
dashboard:
  server:
    require_signed_config: true
```

With this flag, the dashboard refuses to boot if the config's signature is missing or invalid. Use this in production to ensure nobody can modify monitoring thresholds or alert channels without re-signing.

Verify:

```bash
# Tamper with a threshold, then verify — should fail:
sentinel config verify-signature --config sentinel.yaml
# → ERROR: signature mismatch
```

---

## Step 15 — Write one integration test (3 min)

Lock in the behaviour so a future refactor doesn't silently break monitoring:

```python
# tests/test_sentinel_integration.py
import numpy as np
import pytest
from sentinel import SentinelClient
from sentinel.config.schema import (
    SentinelConfig, ModelConfig, AuditConfig, DriftConfig, DataDriftConfig,
)

@pytest.fixture
def client(tmp_path):
    cfg = SentinelConfig(
        model=ModelConfig(name="test_model", type="classification"),
        drift=DriftConfig(data=DataDriftConfig(method="psi", threshold=0.2)),
        audit=AuditConfig(storage="local", path=str(tmp_path / "audit")),
    )
    return SentinelClient(cfg)

def test_drift_fires_on_shifted_data(client):
    rng = np.random.default_rng(42)
    ref = rng.normal(0, 1, size=(500, 3))
    client.fit_baseline(ref)

    shifted = rng.normal(2, 1, size=(500, 3))    # mean shifted by 2σ
    report = client.check_drift(shifted)

    assert report.is_drifted
    assert report.severity in ("high", "critical")
```

Run it:

```bash
pytest tests/test_sentinel_integration.py -v
```

---

## Step 16 — Plan your testing (10 min)

Steps 1–15 get one model wired up. **This step is the testing plan you run before every release** — it treats Sentinel itself as a dependency you need to verify alongside your model code. Work through the five layers top-to-bottom on every release candidate; each layer runs in seconds to minutes and fails fast.

### Layer 1 — Static checks (seconds)

Runs on every PR. No model, no data, no network.

```bash
sentinel validate --config sentinel.yaml --strict        # fails on unresolved ${VAR}
sentinel config show   --config sentinel.yaml            # prints with <REDACTED> masking
sentinel config verify-signature --config sentinel.yaml  # if you sign configs
```

What this catches: typos, missing env vars, broken `extends:` chains, unsigned or tampered configs, malformed file-reference paths.

### Layer 2 — Unit + contract tests (seconds)

Your integration test from step 15, plus any module-specific tests you add. Keep them in `tests/` and run them the same way Sentinel does:

```bash
pytest tests/ -v                                          # your app tests
pytest tests/test_sentinel_integration.py -v              # the step-15 drift contract
```

The Sentinel repo ships **1,640 upstream unit tests** (84% line coverage) that cover every pluggable module. If you fork or extend Sentinel, run them too:

```bash
pytest tests/unit/ -v                                     # full upstream suite
pytest tests/unit/ --cov=sentinel --cov-report=term-missing
```

What this catches: drift detector regressions, schema validation gaps, new modules breaking public API contracts.

### Layer 3 — Drift smoke test (1 min)

Re-run the deliberately-shifted dataset from step 6 against the live config on every release. Commit it as a repeatable script:

```bash
# scripts/drift_smoke.py
python scripts/drift_smoke.py --config sentinel.yaml \
       --reference data/training_sample.parquet
# expect: drifted=True  severity in {high, critical}
```

Wire it into CI as a gating test. If Sentinel ever stops detecting the canned drift, you catch the regression before production.

### Layer 4 — Cloud backend smoke test (2 min)

Runs on the production config against real cloud endpoints. Verifies every `${azkv:…}` reference resolves, every backend is reachable, and credentials are correct **before** deployment.

```bash
sentinel cloud test --config /etc/sentinel/claims.yaml
# expected: four OK lines for keyvault, registry, audit, deploy + exit 0

# Scope to one backend while debugging:
sentinel cloud test --config /etc/sentinel/claims.yaml --only keyvault
sentinel cloud test --config /etc/sentinel/claims.yaml --only registry
sentinel cloud test --config /etc/sentinel/claims.yaml --only audit
sentinel cloud test --config /etc/sentinel/claims.yaml --only deploy
```

What this catches: expired credentials, missing RBAC role assignments, wrong subscription ID, Key Vault secret deleted, Azure ML workspace moved.

### Layer 5 — End-to-end scenario per paradigm (5 min each, pick the ones you ship)

Run exactly one scenario per ML paradigm in your app, against a seeded environment:

| Paradigm | Smoke test |
|---|---|
| Traditional ML | Step 6 drift smoke + `sentinel audit --type alert_sent --limit 1` |
| LLMOps (RAG / chat) | Send a known-bad PII query; assert `guardrail_results.input.blocked == True`; assert a `guardrail_violation` audit event fires |
| AgentOps (LangGraph / SK) | Run a golden trajectory through the wrapped graph; assert `tracer.get_last_trace().step_count` matches the expected trajectory; assert budget guard didn't trip |
| Time series | Fit baseline on one seasonal period, check drift on the same period + 1 year; assert calendar-aware comparison returns `is_drifted=False` |
| Graph ML | Fit baseline on a reference graph, check drift on a densified variant; assert topology drift score exceeds threshold |

Every scenario asserts on the **audit trail** as the ground truth, not on stdout — the audit trail is what compliance will read.

### Audit chain integrity (weekly, if tamper-evidence is on)

If you enabled security hardening's tamper-evident audit trail (`audit.tamper_evidence: true`), run the chain verifier on a schedule:

```bash
sentinel audit verify --config sentinel.yaml           # full chain check
sentinel audit chain-info --config sentinel.yaml       # HMAC key id + chain head
```

Fail the build if verification ever returns non-zero — that means somebody rotated a key improperly or tampered with the log.

### Testing checklist

- [ ] Layer 1 (static) runs on every PR via CI
- [ ] Layer 2 (unit/contract) runs on every PR via CI
- [ ] Layer 3 (drift smoke) runs on every release candidate
- [ ] Layer 4 (`sentinel cloud test`) runs on deploy to each environment
- [ ] Layer 5 scenarios are scripted for every paradigm you ship
- [ ] Audit chain verifier runs on a weekly schedule if tamper-evidence is on

---

## Step 17 — Plan your demo (10 min)

Sentinel is most effective when the person signing the cheque sees it work end-to-end in a single session. Use this playbook to plan a 10-minute stakeholder demo that covers monitoring, alerting, audit, and — if relevant — LLMOps/AgentOps/deployment.

### Demo environment options

Pick one **before** the meeting starts. Don't try to demo against production.

| Option | When to use | Setup cost |
|---|---|---|
| `scripts/run_dashboard.py` seeded demo | Every stakeholder demo by default | 0 — `python scripts/run_dashboard.py` seeds every page |
| `demo/scripts/<vertical>_demo.sh` | Vertical-specific deep dive (bank, insurer, healthcare, ecommerce, platform, quant) | 1 min — each script seeds and opens the dashboard on a dedicated port |
| Your dev config against a test model | Engineer-facing demo where they want to see their own data | 5 min — the full step 1–13 walkthrough above |
| Staging subscription with `sentinel cloud test` | CTO / platform eng demos that need real Azure proof points | 15 min — needs cloud credentials pre-provisioned |

The seeded dashboard is the default because it shows **every** page populated with realistic data in under 10 seconds. No data wrangling, no "oops, that page is empty" moments.

### The 10-minute stakeholder script

Time-boxed so you always finish on time. Expand any section that gets questions; cut the rest.

| Minute | Scene | What to show | Key message |
|---|---|---|---|
| 0:00–0:30 | Problem framing | One slide: "3–5 tools → 1 SDK" | Sentinel replaces stitched-together monitoring, registry, alerts, deployment |
| 0:30–2:00 | **Config over code** | Open `configs/examples/insurance_fraud.yaml` side-by-side with the dashboard | Every behaviour is YAML. Non-engineers can review it. Git-tracked. |
| 2:00–3:00 | Overview page | `/` — model status, recent alerts, event counts | One place to see everything |
| 3:00–4:30 | Drift in action | `/drift` — feature scores, timeline chart | "This is what drift looks like. Top-3 important features + severity." |
| 4:30–5:30 | Audit trail | `/audit` filter by `alert_sent`, `drift_checked`, `deployment_started` | Every action is logged immutably. FCA / EU AI Act ready. |
| 5:30–6:30 | Deployment safety | `/deployments` — canary ramp with rollback rules | Shadow → canary → promote with auto-rollback, no custom scripts |
| 6:30–8:00 | **Pick one differentiator** | (see matrix below) | Your vertical-specific hook |
| 8:00–9:00 | Cloud proof | Terminal: `sentinel cloud test --config azure_full.yaml` → four OK lines | Not vapourware — real Azure integration, verified by CLI |
| 9:00–10:00 | Q&A landing slide | "Pilot in 2 weeks" slide with contact + repo link | Leave them with a concrete next action |

### The differentiator menu (pick one for minutes 6:30–8:00)

Match to the stakeholder's context. Don't try to show all of them.

| Stakeholder | Differentiator | Dashboard page | Script |
|---|---|---|---|
| Compliance / risk officer | Compliance report generator | `/compliance` | "FCA Consumer Duty and EU AI Act reports are one click — here's the raw evidence" |
| LLM / RAG team | Guardrails + token economics | `/llmops` | "PII redaction, jailbreak detection, cost per query. Here's a jailbreak attempt being blocked live." |
| Agent team | Trace viewer + budget guard | `/agentops` | "Every reasoning step is traced. Here's an agent hitting its budget and being stopped before it runs away." |
| Forecasting team | Seasonality-aware drift | `/drift` with demand_forecast config | "Standard PSI false-alerts on January vs July. This compares same-period calendar windows." |
| CTO / platform engineering | `sentinel cloud test` + RBAC dashboard | Terminal + `/settings` | "One CLI confirms Key Vault, Azure ML, audit, and deploy in 3 seconds." |
| Data science lead | Feature health + explainability | `/features` | "Ranked by SHAP importance. Alerts fire only when a **top-3** feature drifts." |

### Pre-demo checklist (run 15 min before the meeting)

- [ ] `python scripts/run_dashboard.py` opens with all pages rendering
- [ ] Refresh `/drift`, `/features`, `/audit`, `/registry`, `/deployments`, `/llmops`, `/agentops`, `/compliance` — no 500s, no empty states
- [ ] Terminal window ready with `sentinel cloud test` command pre-typed (if you're showing cloud proof)
- [ ] `configs/examples/insurance_fraud.yaml` open in a second window for the "config over code" moment
- [ ] Slack `#ml-alerts` channel visible so the live alert actually appears on screen
- [ ] Screen-share tested on the actual meeting tool (webhook URLs in the YAML are already `${VAR}` so the `config show` command is safe to project)
- [ ] Backup: `demo/scripts/<vertical>_demo.sh` on standby in case the primary demo env breaks

### Vertical playbooks

Don't reinvent the wheel — six vertical playbooks ship in [`demo/scripts/`](../demo/scripts/) with seeded data and talking points:

- `bank_demo.sh` — retail banking fraud, FCA focus
- `insurer_demo.sh` — claims fraud + underwriting agent
- `healthcare_demo.sh` — clinical NER + compliance
- `ecommerce_demo.sh` — recommendation drift + cost monitoring
- `platform_demo.sh` — CTO-level, multi-model graph + cloud integration
- `quant_demo.sh` — time series forecasting drift + concept drift

Each script: `(1) seeds a dedicated audit dir, (2) launches the dashboard on a dedicated port, (3) prints the exact talking points for that vertical`. Walk through the matching script the day before your meeting.

### Demo checklist

- [ ] One demo environment chosen and verified 15 min before the meeting
- [ ] 10-minute script rehearsed at least once end-to-end
- [ ] Vertical playbook picked (one of the six in `demo/scripts/`)
- [ ] Backup environment on standby
- [ ] Single-page leave-behind with contact + repo link + pilot-in-2-weeks ask

---

## Troubleshooting cheat sheet

| Symptom | Fix |
|---|---|
| `ImportError: cannot import name 'Sentinel' from 'typing_extensions'` | `pip install --force-reinstall --no-deps "typing_extensions>=4.13"` |
| `pip install sentinel-mlops[all]` fails on Python 3.13 | That's expected if you tried to add `[ml-extras]` — torch has no wheels. Use `[all,dashboard]` only. |
| `sentinel dashboard` → "Dashboard requires `pip install sentinel-mlops[dashboard]`" | `pip install "sentinel-mlops[dashboard]"` |
| Drift never fires even on obviously shifted data | Lower `drift.data.threshold` to `0.1`; verify with step 6 |
| Slack alerts don't arrive but `sentinel audit --type alert_sent` shows events | Wrong webhook URL — test it with `curl -X POST -d '{"text":"test"}' $SLACK_WEBHOOK_URL` |
| Script hangs on `client.llmops.log_call(...)` for 30+ seconds | You have `sentence-transformers` installed. Either disable semantic drift in YAML, or inject a fake `embed_fn` as in `scripts/run_dashboard.py:_install_fake_embedder` |
| `sentinel.yaml` has unresolved `${VAR}` references | Export the env vars in the shell that runs your app, or source an `.env` file |
| `cohort_column is required when cohort_analysis is enabled` | Add `cohort_column: your_feature_name` to the `cohort_analysis` section |
| Dashboard cohort/explanation pages are empty | Call `log_prediction()` with `cohort_id` or valid `cohort_column`; call `explain_global(X)` at least once |
| `SentinelError: unknown domain 'xyz'` | Valid domains: `tabular`, `timeseries`, `nlp`, `recommendation`, `graph` |
| Config validation: `invalid type/domain combination` | Some combos are invalid (e.g., `type: ranking` + `domain: timeseries`). Check the error message for valid pairs |
| `sentinel config verify-signature` → `signature mismatch` | Re-sign after any config change: `sentinel config sign --config sentinel.yaml`. Signature is computed on the resolved config. |
| `sentinel cloud test` → `registry: FAIL` | Check credentials: `--only registry` to isolate. For Azure ML, verify `AZURE_SUBSCRIPTION_ID`, `AZURE_RESOURCE_GROUP`, `AZURE_ML_WORKSPACE` are set and RBAC is assigned. |
| `sentinel deploy --dry-run` shows strategy × target error | Not all strategy/target combos are valid (e.g., `canary` + `azure_app_service`). Use `blue_green` for App Service slot swaps. |
| `ImportError: statsmodels` when using `domain: timeseries` | Install the domain extra: `pip install "sentinel-mlops[timeseries]"`. Domain-specific deps are lazy-loaded. |
| `ComplianceReporter.generate()` returns empty `fairness_monitoring` | Log predictions with `cohort_id` or enable `cohort_analysis` first — fairness reports need cohort data in the audit trail. |
| Dashboard refuses to start with `require_signed_config: true` | Sign the config first: `export SENTINEL_CONFIG_KEY=... && sentinel config sign --config sentinel.yaml` |

---

## The "am I done?" checklist

You're ready to call step 1–17 complete when all of these are true:

**Integration (steps 1–13)**

- [ ] `sentinel validate --config sentinel.yaml` exits 0
- [ ] `sentinel status` shows your model, version, and domain
- [ ] `sentinel registry list` shows at least one registered version
- [ ] `sentinel.yaml` is committed to git
- [ ] Secrets (webhook URLs, cloud credentials) are in env vars, **not** in the YAML
- [ ] A deliberately-drifted dataset fires a Slack alert
- [ ] `sentinel audit --limit 5` shows recent events including `drift_checked` and `alert_sent`
- [ ] The dashboard at `http://127.0.0.1:8000` renders every nav page without errors
- [ ] If using cohorts: `sentinel.compare_cohorts()` returns a valid report with no errors
- [ ] If using explainability: `sentinel.explain_global(X)` returns feature importances
- [ ] One pytest integration test covers the drift-detection behaviour
- [ ] CI runs `sentinel validate` on every PR
- [ ] The audit backend and registry backend point at cloud storage (not local) in production

**Advanced capabilities (step 14, if applicable)**

- [ ] If using compliance: `ComplianceReporter.generate("fca_consumer_duty", ...)` returns a non-empty report
- [ ] If using domain adapters: `sentinel validate` shows the correct domain (e.g., `domain=timeseries`)
- [ ] If using non-local registry: `sentinel cloud test --only registry` passes
- [ ] If using config signing: `sentinel config verify-signature` passes
- [ ] If using `--dry-run` deploy: `sentinel deploy --dry-run` returns a valid JSON report

**Testing (step 16)**

- [ ] Layer 1 static checks wired into CI (`sentinel validate --strict`)
- [ ] Layer 3 drift smoke script exists and runs on every release
- [ ] Layer 4 `sentinel cloud test` runs on every deploy to each environment
- [ ] Layer 5 scenario test exists for every paradigm you ship
- [ ] Audit chain verifier scheduled weekly if tamper-evidence is on

**Demo readiness (step 17)**

- [ ] `python scripts/run_dashboard.py` opens with all pages rendering
- [ ] 10-minute stakeholder script rehearsed end-to-end at least once
- [ ] Matching vertical playbook picked from `demo/scripts/`
- [ ] Backup demo environment verified on standby

Once every box is ticked, the integration is production-grade **and** stakeholder-demo-ready. Everything else — LLMOps, AgentOps, domain adapters, multi-model graphs, business KPI linking — is incremental YAML-only additions that don't require revisiting your app code.

---

## Where to go next

- [`quickstart.md`](quickstart.md) — the conceptual tour of the SDK
- [`architecture.md`](architecture.md) — the seven-layer stack and feedback loops
- [`config-reference.md`](config-reference.md) — every YAML field documented
- [`../configs/examples/`](../configs/examples/) — 8 ready-to-adapt configs (fraud, RAG, agents, forecasting, NER, recsys, graph)
- [`../demo/`](../demo/) — 6 customer playbooks with runnable end-to-end scripts
- [`../scripts/run_dashboard.py`](../scripts/run_dashboard.py) — fastest way to see every dashboard page with seeded data
- [`../tests/unit/`](../tests/unit/) — canonical test patterns for each module
