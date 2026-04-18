<div align="center">

# Project Sentinel

### The Unified MLOps + LLMOps + AgentOps SDK for Production AI

**One SDK. Three paradigms. Five domains. Zero glue code.**

[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)](#)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-2%2C212%20passing-brightgreen.svg)](#-testing)
[![Coverage](https://img.shields.io/badge/coverage-87%25-green.svg)](#-testing)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](#license)

[Quickstart](#-quickstart) ·
[Features](#-features) ·
[Architecture](#-architecture) ·
[Dashboard](#-self-serve-dashboard) ·
[Examples](#-end-to-end-examples) ·
[Docs](#-documentation)

</div>

---

## What is Sentinel?

Project Sentinel is a **config-driven Python SDK** that gives ML/AI teams a single `pip install` to **monitor, govern, and operate** production machine learning models, LLM applications, and autonomous agent systems.

It replaces the 3-5 separate tools that enterprise teams typically stitch together (Evidently, WhyLabs, MLflow, custom alerting, deployment scripts, prompt managers, agent tracers) with one coherent library — all controlled through a single YAML file.

```bash
pip install sentinel-mlops
```

```python
from sentinel import SentinelClient

client = SentinelClient.from_config("sentinel.yaml")
client.log_prediction(features=X, prediction=y_pred)
report = client.check_drift()
if report.is_drifted:
    client.notify(report)
```

That's the entire monitoring loop. Drift detection, alert routing, model registry, audit trail — all driven by config.

---

## Why Sentinel?

Most enterprise ML teams reinvent the same monitoring and deployment plumbing for every project. Sentinel collapses setup time from **weeks to hours**. A developer goes from `pip install` to a fully monitored, alerting, auto-deploying model in under 50 lines of code.

### Built for regulated industries

Sentinel targets ML/AI teams in **banking, insurance, and healthcare** where model governance, audit trails, and human-in-the-loop controls are non-negotiable. Every prediction is logged. Every drift event is auditable. Every deployment has an approval gate.

### Three paradigms, one interface

| Paradigm | What it covers | Example use cases |
|----------|---------------|-------------------|
| **Traditional MLOps** | Drift detection, data quality, model registry, deployment | Fraud detection, credit scoring, claims prediction |
| **LLMOps** | Prompt management, guardrails, response quality, token costs | RAG pipelines, claims summarisation, document Q&A |
| **AgentOps** | Trace monitoring, tool audit, safety enforcement, multi-agent | Underwriting agents, automated claims processing |

### How Sentinel compares

| Capability | Evidently | WhyLabs | NannyML | MLflow | Arize | LangSmith | Guardrails AI | AgentOps.ai | CrewAI | **Sentinel** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Data drift detection | ✓ | ✓ | ✓ | — | ✓ | — | — | — | — | **✓** |
| Concept drift | — | — | ✓ | — | — | — | — | — | — | **✓** |
| Cohort-based analysis | — | — | — | — | ✓ | — | — | — | — | **✓** |
| Multi-level explainability | — | — | — | — | — | — | — | — | — | **✓** |
| Model registry (6 backends) | — | — | — | ✓ | — | — | — | — | — | **✓** |
| Alert routing + escalation | — | ✓ | — | — | ✓ | — | — | — | — | **✓** |
| Deployment automation | — | — | — | — | — | — | — | — | — | **✓** |
| LLM guardrails + DSL | — | — | — | — | — | — | ✓ | — | — | **✓** |
| Prompt versioning + A/B | — | — | — | — | — | ✓ | — | — | — | **✓** |
| Agent tracing (OTel) | — | — | — | — | — | ✓ | — | ✓ | ✓ | **✓** |
| Agent safety controls | — | — | — | — | — | — | — | — | — | **✓** |
| Domain adapters (5 domains) | — | — | — | — | — | — | — | — | — | **✓** |
| Tamper-evident audit | — | — | — | — | — | — | — | — | — | **✓** |
| Multi-cloud (Azure/AWS/GCP) | — | — | — | — | — | — | — | — | — | **✓** |
| Config-as-code | — | — | — | — | — | — | — | — | — | **✓** |
| Self-hosted / on-prem | ✓ | — | ✓ | ✓ | — | — | ✓ | — | ✓ | **✓** |

> **Sentinel is the only SDK that unifies traditional MLOps + LLMOps + AgentOps under one config-driven interface with first-class regulatory compliance support.**

---

## 🚀 Quickstart

### 1. Install

```bash
pip install sentinel-mlops                  # core (numpy + pydantic only)
pip install "sentinel-mlops[drift]"         # + scipy/sklearn for drift detection
pip install "sentinel-mlops[dashboard]"     # + FastAPI/Plotly for the web UI
pip install "sentinel-mlops[all]"           # everything except heavy ML libs
```

### 2. Generate a starter config

```bash
sentinel init --name my_model --out sentinel.yaml
```

This creates a fully-commented YAML config with sensible defaults. Every SDK behaviour is driven by this file.

### 3. Wire into your serving code

```python
from fastapi import FastAPI
from sentinel import SentinelClient
import joblib

app = FastAPI()
model = joblib.load("models/fraud.pkl")
sentinel = SentinelClient.from_config("sentinel.yaml")

@app.on_event("startup")
async def startup():
    sentinel.register_model_if_new(version="1.0.0", framework="sklearn")

@app.post("/predict")
async def predict(features: dict):
    # Validate inputs against schema
    quality = sentinel.check_data_quality(features)
    if quality.has_critical_issues:
        return {"error": "Input validation failed", "issues": quality.issues}

    # Predict + log (drift computed in background on batched windows)
    prediction = model.predict([features])
    sentinel.log_prediction(features=features, prediction=prediction)
    return {"prediction": prediction.tolist()}

@app.get("/health")
async def health():
    drift = sentinel.check_drift()
    return {"model": sentinel.model_name, "drift": drift.summary}
```

### 4. Check drift from the CLI

```bash
sentinel check --config sentinel.yaml \
  --reference data/training.parquet \
  --current data/today.parquet
```

### 5. Launch the dashboard

```bash
sentinel dashboard --config sentinel.yaml --open
```

### 6. Deploy a new model version

```bash
sentinel deploy --config sentinel.yaml \
  --version 1.1.0 --strategy canary --traffic 5
```

---

## 🔍 Features

### Observability (Layer 2)

| Feature | Description |
|---------|-------------|
| **Data drift** | PSI, KS, Jensen-Shannon, Chi-squared, Wasserstein — pick via config |
| **Concept drift** | DDM, EDDM, ADWIN, Page-Hinkley — detects X→y relationship changes |
| **Model performance decay** | Track accuracy, F1, AUC against registered baselines |
| **Feature health** | Per-feature drift ranked by SHAP/permutation importance |
| **Cohort analysis** | Segment predictions by sub-population, detect performance disparity |
| **Data quality** | Schema enforcement, freshness checks, null/duplicate detection, outlier detection (IF, Z-score, IQR) |
| **Cost monitoring** | Inference latency (p50/p99), throughput, cost-per-prediction |

### LLMOps (Layer 3)

| Feature | Description |
|---------|-------------|
| **Prompt management** | Versioned bundles (system + template + few-shot), A/B traffic routing |
| **Input guardrails** | PII detection/redaction (Presidio), jailbreak detection, topic fencing, token budget |
| **Output guardrails** | Toxicity, groundedness (NLI/chunk-overlap), format compliance, regulatory language |
| **Custom guardrails DSL** | Declarative rule-based guardrails (regex, keyword, length, JSON schema, sentiment, language) — no code needed |
| **Plugin guardrails** | Bring your own guardrail class — dynamically loaded at runtime via `type: plugin` |
| **Response quality** | LLM-as-judge with configurable rubrics, heuristic fallback, reference-based |
| **Semantic drift** | Embedding-distribution shift over sliding windows |
| **RAG quality** | Retrieval relevance, chunk utilisation, faithfulness, answer coverage |
| **Token economics** | Per-query/user/model cost tracking with budgets and trend alerts |
| **Prompt drift** | Composite signal detecting prompt effectiveness degradation |

### AgentOps (Layer 4)

| Feature | Description |
|---------|-------------|
| **Span-based tracing** | OpenTelemetry-compatible; export to Jaeger, Arize Phoenix, OTLP |
| **Auto-instrumentation** | LangGraph middleware, Semantic Kernel plugin, Google ADK hooks |
| **Tool audit** | Success/failure/latency tracking, allowlist/blocklist, parameter validation |
| **Tool replay** | Record and replay tool calls for debugging with bounded cache |
| **Loop detection** | Infinite loops, circular delegation, thrashing, depth limits |
| **Budget enforcement** | Token, cost, time, and tool-call budgets per agent run |
| **Human escalation** | Confidence-based, failure-based, sensitive-data, regulatory triggers |
| **Action sandboxing** | Destructive operations require approval, dry-run, or sandbox-then-apply |
| **Agent registry** | Versioned capability manifests, A2A discovery, health checks |
| **Multi-agent monitoring** | Delegation chains, consensus tracking, bottleneck detection |
| **Trajectory evaluation** | LCS comparison against golden datasets with CI/CD integration |

### Model Lifecycle (Layers 5–7)

| Feature | Description |
|---------|-------------|
| **Model registry** | Version, baseline, metadata, lineage — 6 backends: local, Azure ML, MLflow, SageMaker, Vertex AI, Databricks |
| **Deployment automation** | Shadow, canary, blue-green, direct strategies with auto-rollback — 6 targets: local, Azure ML Endpoint, Azure App Service, AKS, SageMaker Endpoint, Vertex AI Endpoint |
| **Notifications** | 6 channels (Slack, Teams, PagerDuty, Email, Webhook, custom) with escalation chains, cooldown, digest mode, rate limiting, Jinja2 templates |
| **Retrain orchestration** | Drift / scheduled / manual triggers → pipeline → validate → human approval → promote → auto-deploy |
| **Multi-model graphs** | Dependency DAG with cascade alert propagation, eager cycle detection, thread-safe |
| **Business KPI linking** | Model metrics ↔ business outcomes mapping with auto-refresh scheduler |
| **Explainability** | SHAP/permutation at three levels: per-row, global, and per-cohort |
| **Experiment tracking** | Named experiments with nested runs, metric history, search (`metrics.f1 > 0.85 AND params.lr < 0.01`), run comparison, dataset + model linkage |
| **Dataset registry** | Metadata-only tracking with `name@version` references, SHA-256 content hash verification, lineage linking to experiments and models |

### Domain Adapters

Switch one config field and Sentinel uses domain-appropriate algorithms:

```yaml
model:
  name: demand_forecast_v3
  domain: timeseries        # tabular | timeseries | nlp | recommendation | graph
  type: regression
```

| Domain | Drift Detection | Quality Metrics |
|--------|-----------------|-----------------|
| **tabular** (default) | PSI, KS, Chi-squared | Accuracy, F1, AUC, precision, recall |
| **timeseries** | STL decomposition, calendar-aware tests, stationarity (ADF/KPSS) | MASE, MAPE, coverage, interval width, Winkler, directional accuracy |
| **nlp** | Vocabulary drift (OOV rate), embedding-space MMD, label distribution | Token-level F1, span exact match, Cohen's Kappa |
| **recommendation** | Item/user distribution shift, cold-start ratio | NDCG@K, MAP@K, coverage, diversity, novelty, Gini bias, group fairness |
| **graph** | Topology drift (degree, density, clustering, components) | AUC-ROC, Hits@K, MRR, modularity, embedding isotropy |

### 📊 Self-Serve Dashboard

A production-ready web UI ships with the SDK — no separate deployment needed.

```bash
sentinel dashboard --config sentinel.yaml --port 8080 --open
```

**Or embed in an existing FastAPI app:**

```python
from sentinel.dashboard import SentinelDashboardRouter

router = SentinelDashboardRouter(client)
router.attach(app)  # mounts at /sentinel/
```

**What's included:**

- **21 pages** — Overview, drift, features, cohorts, explanations, registry, audit, compliance, LLMOps (prompts, guardrails, tokens), AgentOps (traces, tools, agents), deployments, retraining, intelligence, datasets, experiments
- **12 interactive Plotly charts** — Drift timeline, feature importance, compliance events, token costs, cost-by-model, tool success rates, guardrail violations, agent trace waterfall, overview health, cohort comparison, global feature importance, audit event distribution
- **Dark mode**, auto-refresh (configurable interval), toast notifications
- **RBAC** with viewer / operator / admin roles and JWT Bearer auth
- **CSRF protection**, rate limiting, security headers, signed config enforcement

### 🔒 Security & Compliance

| Feature | Description |
|---------|-------------|
| **Tamper-evident audit trail** | HMAC-SHA256 hash chain; `sentinel audit verify` detects any modification |
| **Signed configs** | `sentinel config sign` ensures production boots only against reviewed YAML |
| **RBAC** | Viewer / operator / admin roles with transitive inheritance and namespaced permissions |
| **CSRF protection** | Double-submit-cookie pattern with HTMX header hooks |
| **Rate limiting** | Per-IP (unauthenticated) and per-user (authenticated) token-bucket |
| **Security headers** | HSTS, CSP, X-Frame-Options, X-Content-Type-Options |
| **Azure Key Vault** | `${azkv:vault-url/secret-name}` syntax for zero-plaintext configs |
| **AWS Secrets Manager** | `${awssm:secret-name}` or `${awssm:secret-name/key}` syntax |
| **GCP Secret Manager** | `${gcpsm:project/secret-name}` syntax |
| **FCA / EU AI Act** | Built-in compliance report generators for regulated industries |

---

## 📐 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 1 — Developer Interface                                   │
│  SDK Core API  │  Config-as-Code (YAML)  │  CLI  │  Dashboard    │
├──────────────────────────────────────────────────────────────────┤
│  Layer 2 — Observability                                         │
│  Data Quality  │  Drift Detection  │  Feature Health  │  Cost    │
├──────────────────────────────────────────────────────────────────┤
│  Layer 3 — LLMOps                                                │
│  Prompt Mgmt  │  Guardrails  │  Response Quality  │  Token Cost  │
├──────────────────────────────────────────────────────────────────┤
│  Layer 4 — AgentOps                                              │
│  Tracing  │  Tool Audit  │  Safety  │  Multi-Agent  │  Eval      │
├──────────────────────────────────────────────────────────────────┤
│  Layer 5 — Intelligence                                          │
│  Model Graph  │  Business KPI Link  │  Explainability  │ Cohorts │
├──────────────────────────────────────────────────────────────────┤
│  Layer 6 — Action                                                │
│  Notifications  │  Retrain Orchestrator  │  Deploy Automation     │
├──────────────────────────────────────────────────────────────────┤
│  Layer 7 — Foundation                                            │
│  Model Registry  │  Audit Trail  │  Experiment Tracking           │
└──────────────────────────────────────────────────────────────────┘
```

**Data flow:** Prediction → Data Quality → Drift Detection → Feature Health → Intelligence (correlate signals) → Action (alert / retrain / deploy) → Audit Trail (log everything)

**Feedback loops:** Retrain results reset observability baselines. Agent trace failures tighten guardrail policies. Prompt drift signals trigger the retrain orchestrator.

Read the [architecture deep-dive](docs/architecture.md).

---

## 💻 End-to-End Examples

### Traditional ML — Fraud classifier with monitoring

```python
from sentinel import SentinelClient

sentinel = SentinelClient.from_config("sentinel.yaml")
sentinel.register_model_if_new(version="2.3.1", framework="xgboost")

# In your prediction loop:
sentinel.log_prediction(features=X, prediction=y_pred, actuals=y_true)

# Periodic checks:
drift = sentinel.check_drift()             # DriftReport
quality = sentinel.check_data_quality(X)   # QualityReport
health = sentinel.get_feature_health()     # FeatureHealthReport

# Cohort analysis:
sentinel.log_prediction(features=X, prediction=y_pred, cohort_id="segment_a")
disparity = sentinel.compare_cohorts()     # CohortComparisonReport

# Explainability:
global_imp = sentinel.explain_global(X)    # GlobalExplanation
cohort_imp = sentinel.explain_cohorts(X, cohort_labels=labels)
```

### LLMOps — RAG pipeline with guardrails

```python
from sentinel import SentinelClient
from sentinel.llmops import PromptManager, GuardrailPipeline, TokenTracker

sentinel = SentinelClient.from_config("sentinel.yaml")
prompts = PromptManager.from_config("sentinel.yaml")
guardrails = GuardrailPipeline.from_config("sentinel.yaml")

# 1. Input guardrails — PII redaction, jailbreak check, topic fence
input_check = guardrails.check_input(user_query)
if input_check.blocked:
    return {"error": input_check.reason}

# 2. Resolve prompt version (handles A/B routing)
prompt = prompts.resolve("claims_qa", context={"user_segment": user_id})

# 3. Call your LLM...
response = await call_llm(prompt.render(query=input_check.sanitised_input))

# 4. Output guardrails — groundedness, toxicity, format
output_check = guardrails.check_output(response=response, context=chunks)
if output_check.blocked:
    return {"error": "Response failed safety checks"}

# 5. Log everything
sentinel.log_llm_call(
    prompt_name="claims_qa", prompt_version=prompt.version,
    input_tokens=450, output_tokens=120, response=response,
)
```

### Dataset Management — Track and version datasets

```python
from sentinel import SentinelClient

sentinel = SentinelClient.from_config("sentinel.yaml")

# Register a dataset version (metadata-only — data stays in place)
sentinel.dataset_registry.register(
    name="claims_training",
    version="2.1.0",
    path="s3://bucket/claims_train.parquet",
    format="parquet",
    num_rows=50_000,
    num_features=42,
    tags={"split": "train", "quarter": "Q1-2026"},
)

# Reference datasets by name@version
sentinel.dataset_registry.link_to_experiment("claims_training@2.1.0", experiment="fraud_v3")

# Verify data integrity
sentinel.dataset_registry.verify("claims_training@2.1.0")  # SHA-256 content hash check
```

### Enhanced Experiment Tracking — Runs, metrics, and comparison

```python
from sentinel import SentinelClient

sentinel = SentinelClient.from_config("sentinel.yaml")
tracker = sentinel.experiment_tracker

# Create an experiment and log runs with metric history
run = tracker.create_run("fraud_experiment", params={"lr": 0.001, "epochs": 50})
for epoch in range(50):
    tracker.log_metric(run.run_id, "f1", value=0.80 + epoch * 0.002, step=epoch)
tracker.end_run(run.run_id, status="completed")

# Search runs with filter syntax
results = tracker.search_runs("metrics.f1 > 0.85 AND params.lr < 0.01")

# Compare two runs side-by-side
diff = tracker.compare_runs(run_id_a, run_id_b)

# Link to dataset and model versions
tracker.link_dataset(run.run_id, "claims_training@2.1.0")
tracker.link_model(run.run_id, model_version="2.3.1")
```

### Custom Guardrails DSL — Declarative rules, no code

```yaml
# sentinel.yaml — guardrails section
llmops:
  guardrails:
    input:
      - type: custom
        name: input_quality_gate
        rules:
          - { type: min_length, value: 10 }
          - { type: regex_absent, value: "DROP TABLE|DELETE FROM" }
          - { type: not_empty }
        combine: "any"         # block on first rule failure
        action: block
    output:
      - type: plugin
        module: "my_company.guardrails"
        class_name: "ComplianceGuardrail"
        config: { strict_mode: true }
        action: warn
```

### AgentOps — LangGraph agent with tracing

```python
from langgraph.graph import StateGraph
from sentinel.agentops import AgentTracer
from sentinel.agentops.integrations import LangGraphMiddleware

tracer = AgentTracer.from_config("sentinel.yaml")

# Your LangGraph agent
graph = StateGraph(AgentState)
graph.add_node("plan", plan_node)
graph.add_node("search", search_node)
graph.add_node("synthesise", synthesise_node)
compiled = graph.compile()

# Wrap with Sentinel — zero code changes to your agent
monitored = LangGraphMiddleware(tracer).wrap(compiled)

# Every step is traced, every tool call audited, budgets enforced
result = await monitored.ainvoke(
    {"claim_id": "12345"},
    config={"sentinel": {"agent_name": "claims_processor",
                         "budget": {"max_tokens": 30000}}}
)

trace = tracer.get_last_trace()
print(f"Steps: {trace.step_count}, Cost: ${trace.total_cost:.4f}")
```

---

## ⚙️ Configuration

Everything is driven by a single YAML file. Here's a representative example:

```yaml
version: "1.0"

model:
  name: claims_fraud_v2
  type: classification
  domain: tabular               # tabular | timeseries | nlp | recommendation | graph

drift:
  data:
    method: psi                 # psi | ks | js_divergence | chi_squared | wasserstein
    threshold: 0.2
    window: 7d
  concept:
    method: ddm
    min_samples: 100

alerts:
  channels:
    - type: slack
      webhook_url: ${SLACK_WEBHOOK_URL}    # env var substitution
      channel: "#ml-alerts"
  policies:
    cooldown: 1h
    escalation:
      - after: 0m
        channels: [slack]
        severity: [high, critical]

deployment:
  strategy: canary
  canary:
    ramp_steps: [5, 25, 50, 100]
    rollback_on:
      error_rate_increase: 0.02

audit:
  storage: local                # local | azure_blob | s3 | gcs
  tamper_evidence: true         # HMAC-SHA256 hash chain
  compliance_frameworks: [fca_consumer_duty, eu_ai_act]
```

### Example configs

| Config | Use case |
|--------|----------|
| [`minimal.yaml`](configs/examples/minimal.yaml) | Bare minimum — model name + drift |
| [`insurance_fraud.yaml`](configs/examples/insurance_fraud.yaml) | Full BFSI with FCA compliance |
| [`demand_forecast.yaml`](configs/examples/demand_forecast.yaml) | Time series with STL decomposition |
| [`ner_entity_extraction.yaml`](configs/examples/ner_entity_extraction.yaml) | NLP NER with vocabulary drift |
| [`product_reco.yaml`](configs/examples/product_reco.yaml) | RecSys with fairness constraints |
| [`fraud_graph.yaml`](configs/examples/fraud_graph.yaml) | Graph ML with topology drift |
| [`rag_claims_agent.yaml`](configs/examples/rag_claims_agent.yaml) | RAG pipeline with guardrails |
| [`multi_agent_underwriting.yaml`](configs/examples/multi_agent_underwriting.yaml) | Multi-agent with full tracing |

---

## 🖥️ CLI Reference

```
sentinel init       Generate a starter sentinel.yaml config
sentinel validate   Validate config (--strict checks env vars + file refs)
sentinel check      Run drift detection against reference/current datasets
sentinel status     Show current model status and health
sentinel deploy     Deploy a model version (--dry-run to validate without deploying)

sentinel registry list   List registered model versions
sentinel registry show   Show details for a specific version

sentinel audit query         Query the audit trail by event type
sentinel audit verify        Verify audit trail hash chain integrity
sentinel audit chain-info    Show hash chain metadata

sentinel config validate          Validate config (--strict mode)
sentinel config show              Display resolved config (secrets redacted, --unmask to reveal)
sentinel config sign              Sign a config for production enforcement
sentinel config verify-signature  Verify a config signature

sentinel cloud test    Probe cloud backends (Key Vault, registry, shipper, target)
sentinel dashboard     Launch the self-serve web dashboard
sentinel completion    Generate shell completion script (bash/zsh/fish)
```

---

## 📦 Installation Extras

Sentinel has zero heavy dependencies in its core — just `numpy` and `pydantic`. Install only what you need:

| Extra | What it adds | Install |
|-------|-------------|---------|
| `drift` | scipy, scikit-learn | `pip install "sentinel-mlops[drift]"` |
| `explain` | SHAP, LIME | `pip install "sentinel-mlops[explain]"` |
| `dashboard` | FastAPI, Plotly, Jinja2, HTMX | `pip install "sentinel-mlops[dashboard]"` |
| `azure` | Azure ML, Blob Storage, Key Vault | `pip install "sentinel-mlops[azure]"` |
| `aws` | Boto3, SageMaker | `pip install "sentinel-mlops[aws]"` |
| `notify-slack` | Slack SDK | `pip install "sentinel-mlops[notify-slack]"` |
| `notify-teams` | pymsteams | `pip install "sentinel-mlops[notify-teams]"` |
| `notify-pagerduty` | pdpyras | `pip install "sentinel-mlops[notify-pagerduty]"` |
| `llmops` | tiktoken, OpenAI, Presidio, sentence-transformers | `pip install "sentinel-mlops[llmops]"` |
| `agentops` | OpenTelemetry API + SDK + OTLP exporter | `pip install "sentinel-mlops[agentops]"` |
| `timeseries` | statsmodels, pmdarima | `pip install "sentinel-mlops[timeseries]"` |
| `nlp-domain` | sentence-transformers, spaCy | `pip install "sentinel-mlops[nlp-domain]"` |
| `recommendation` | implicit, RecBole | `pip install "sentinel-mlops[recommendation]"` |
| `graph` | NetworkX, PyTorch Geometric | `pip install "sentinel-mlops[graph]"` |
| `mlflow` | MLflow SDK | `pip install "sentinel-mlops[mlflow]"` |
| `all` | Everything above (excluding heavy torch/spacy) | `pip install "sentinel-mlops[all]"` |
| `ml-extras` | sentence-transformers, Presidio, spaCy | `pip install "sentinel-mlops[ml-extras]"` |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/unit/ -v

# Run with coverage
pytest tests/unit/ --cov=sentinel --cov-report=term-missing

# Run integration tests (requires fixtures)
pytest tests/integration/ -v -m integration
```

**Current status:** 2,212 tests passing · 87% line coverage · 211 source files · 129 test files

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[Quickstart](docs/quickstart.md)** | Get to your first alert in 10 minutes |
| **[Config Reference](docs/config-reference.md)** | Every YAML field with type, default, and example |
| **[Architecture](docs/architecture.md)** | Seven-layer stack, extensibility points, data flow |
| **[Security](docs/security.md)** | Audit trails, RBAC, CSRF, JWT, signed configs, threat model |
| **[Azure Integration](docs/azure.md)** | Key Vault, ML registry, Blob audit, RBAC snippets |
| **[Cloud Integration](docs/cloud-integration-guide.md)** | Multi-cloud: Azure, AWS, GCP, Databricks backends |
| **[Developer Guide](docs/developer-guide.md)** | Step-by-step integration tutorial (17 steps) |
| **[Codebase Guide](docs/codebase-guide.md)** | Module-by-module contributor walkthrough |
| **[Demo Guide](docs/demo-guide.md)** | 10-part runnable demo script for stakeholder presentations |
| **[Offering Guide](docs/offering-guide.md)** | Packaging Sentinel as a commercial product |

---

## 🗺️ Project Structure

```
sentinel/
├── sentinel/
│   ├── __init__.py              # Public API surface
│   ├── config/                  # YAML loader, Pydantic schema, defaults, Key Vault, signing
│   ├── core/                    # SentinelClient, hooks, types, exceptions
│   ├── observability/           # Data quality, drift (data/concept/model), feature health, cost, cohorts
│   ├── llmops/                  # Prompt manager, guardrails, quality, token economics, prompt drift
│   ├── agentops/                # Tracing, tool audit, safety, agent registry, multi-agent, eval
│   ├── intelligence/            # Model graph, KPI linker, explainability
│   ├── action/                  # Notifications, retrain orchestrator, deployment strategies
│   ├── foundation/              # Model registry, audit trail, experiments, dataset registry
│   ├── domains/                 # Adapters: tabular, timeseries, nlp, recommendation, graph
│   ├── integrations/            # Azure, AWS, GCP cloud backends
│   ├── dashboard/               # FastAPI + Jinja2 + HTMX + Plotly web UI
│   └── cli/                     # Click-based CLI
├── tests/
│   ├── unit/                    # 2,212 tests — fast, no I/O
│   └── integration/             # Requires fixtures
├── configs/examples/            # 8 reference YAML configs
└── docs/                        # Quickstart, architecture, security, cloud, guides
```

---

## 🤝 Contributing

We welcome contributions! Please read the [Developer Guide](docs/developer-guide.md) before submitting PRs.

**Key conventions:**
- Python 3.10+ with full type hints (`mypy --strict`)
- Google-style docstrings on all public APIs
- `structlog` for all logging (never `print()` or stdlib `logging`)
- Every new module needs a corresponding test file in `tests/unit/`
- ABC + registry pattern for all pluggable components
- Lazy imports for heavy optional dependencies

```bash
# Development setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
pre-commit install

# Lint, format, type-check
ruff check sentinel/ && ruff format sentinel/
mypy sentinel/ --strict

# Test
pytest tests/unit/ -v --cov=sentinel
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
>>>>>>> 302e202 (Merged sentinel-ops into main repo)
