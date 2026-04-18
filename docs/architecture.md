# Architecture

Sentinel is a Python SDK that gives ML teams a single `pip install` to monitor, govern, and operate production machine learning models, LLM applications, and autonomous agent systems. It covers three operational paradigms under a unified, config-driven interface:

- **Traditional MLOps** — drift detection, data quality, automated retraining and deployment for sklearn, XGBoost, PyTorch, TensorFlow, LightGBM, and ONNX models.
- **LLMOps** — prompt management, guardrails, response quality, token economics, and semantic drift for OpenAI, Azure OpenAI, Anthropic, and open-source LLMs.
- **AgentOps** — trace monitoring, tool audit, safety enforcement, and multi-agent orchestration for LangGraph, Semantic Kernel, CrewAI, AutoGen, and Google ADK.

The SDK is organised as a seven-layer stack with explicit feedback loops. The layering is enforced by the package structure — modules in lower layers do not import from higher layers, and cloud-specific code is quarantined in `sentinel/integrations/`.

---

## The seven-layer stack

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 1 — Developer Interface                               │
│  SentinelClient  │  Config as Code  │  CLI  │  Dashboard     │
├──────────────────────────────────────────────────────────────┤
│  Layer 2 — Observability                                     │
│  Data Quality │ Drift (data/concept/model) │ Feature Health  │
│  Cost Monitor │ Cohort Analysis                              │
├──────────────────────────────────────────────────────────────┤
│  Layer 3 — LLMOps                                            │
│  Prompt Mgmt │ Guardrails │ Response Quality │ Token Econ    │
│  Semantic Drift │ Prompt Drift                               │
├──────────────────────────────────────────────────────────────┤
│  Layer 4 — AgentOps                                          │
│  Trace System │ Tool Audit │ Safety │ Agent Registry         │
│  Multi-Agent │ Evaluation                                    │
├──────────────────────────────────────────────────────────────┤
│  Layer 5 — Intelligence                                      │
│  Model Graph (DAG) │ KPI Linker │ Explainability             │
├──────────────────────────────────────────────────────────────┤
│  Layer 6 — Action                                            │
│  Notifications │ Retrain Orchestrator │ Deployment Automation │
├──────────────────────────────────────────────────────────────┤
│  Layer 7 — Foundation                                        │
│  Model Registry │ Audit Trail │ Lineage │ Experiments        │
│  Dataset Registry                                            │
└──────────────────────────────────────────────────────────────┘

Feedback loops:
  → Retrain results feed back into Observability (reset baselines)
  → Foundation baselines inform Observability (define "normal")
  → Agent trace failures feed back into Guardrails (tighten policies)
  → Prompt drift signals feed into Retrain Orchestrator (trigger prompt tuning)
  → Audit trail feeds into Compliance reports (FCA / EU AI Act)
```

---

## Layer 1 — Developer Interface

**Package:** `sentinel.core`, `sentinel.config`, `sentinel.cli`, `sentinel.dashboard`

The single entry point. Everything a user does passes through this layer.

### SentinelClient (`sentinel/core/client.py`)

The public Python API. Thread-safe. All heavy work is delegated to lower layers; the client itself is a thin orchestrator.

```python
from sentinel import SentinelClient

client = SentinelClient.from_config("sentinel.yaml")
client.log_prediction(features=X, prediction=y_pred)
client.check_drift()          # → DriftReport
client.check_data_quality(X)  # → QualityReport
client.get_feature_health()   # → FeatureHealthReport
client.deploy(version="2.3.1", strategy="canary", traffic_pct=5)
```

The constructor wires together every layer based on the config:

```
AuditTrail → ModelRegistry → DomainAdapter → DataQualityChecker
           → DriftDetector → FeatureHealthMonitor → CostMonitor
           → CohortAnalyzer → NotificationEngine → DeploymentManager
           → RetrainOrchestrator → ModelGraph → KPILinker → HookManager
```

LLMOps and AgentOps clients are built **lazily** — they only initialise when accessed via `client.llmops` or `client.agentops`, so tabular-only deployments never import tiktoken or OpenTelemetry.

### Config as Code (`sentinel/config/`)

Every SDK behaviour is driven by YAML config files. The config subsystem provides:

- **`loader.py`** — YAML parsing, `${VAR}` environment variable expansion, `extends:` inheritance chains (with circular-reference detection), and secret resolution (`${azkv:…}`, `${awssm:…}`, `${gcpsm:…}` tokens).
- **`schema.py`** — Pydantic v2 models for every config section. Validates on load with clear error messages. `SecretStr` fields for webhook URLs, routing keys, and tokens.
- **`defaults.py`** — Sensible defaults for all modules.
- **`references.py`** — File-reference validation for baseline, schema, holdout, and audit paths.
- **`signing.py`** — Config signing (`sentinel config sign`) and verification against resolved content.
- **`source.py`** — Per-field source tracing in validation errors.

### CLI (`sentinel/cli/main.py`)

Click-based commands:

| Command | Purpose |
|---------|---------|
| `sentinel init` | Generate a config template |
| `sentinel config validate` | Validate config (optional `--strict` for env vars) |
| `sentinel config show` | Print resolved config with `<REDACTED>` secrets |
| `sentinel config sign` / `verify-signature` | Sign and verify config integrity |
| `sentinel check` | Run drift detection |
| `sentinel status` | Show model and deployment status |
| `sentinel deploy` | Deploy a model version with a chosen strategy |
| `sentinel registry list` / `show` | Inspect the model registry |
| `sentinel audit verify` / `chain-info` | Verify audit trail integrity |
| `sentinel cloud test` | Probe all configured cloud backends |
| `sentinel dashboard` | Launch the local dashboard server |

### Dashboard (`sentinel/dashboard/`)

Optional FastAPI + Jinja2 + HTMX + Plotly UI under the `[dashboard]` extra. Available via:

- **CLI**: `sentinel dashboard --config sentinel.yaml`
- **Embeddable**: `SentinelDashboardRouter(client).attach(app)` in an existing FastAPI app

Every page is a thin pure-function view over the live `SentinelClient` — no separate data store. Pages cover: overview, drift, features, registry, audit, LLMOps, AgentOps, deployments, and compliance. Security features include RBAC (viewer/operator/admin roles), Bearer JWT auth, double-submit-cookie CSRF, and rate limiting.

---

## Layer 2 — Observability

**Package:** `sentinel.observability`

Detects "something is wrong" — without yet deciding what to do about it.

### Data Quality (`data_quality.py`)

- Schema enforcement against JSON Schema definitions
- Freshness checks (alert if no new data in configurable window)
- Outlier detection: isolation forest, z-score, IQR
- Null ratio, duplicate ratio, and type-mismatch tracking

### Drift Detection (`drift/`)

Three independent drift modules, all extending `BaseDriftDetector` with `fit()` / `detect()` / `reset()`:

| Module | Methods | Mode |
|--------|---------|------|
| **Data drift** (`data_drift.py`) | PSI, KS, Jensen-Shannon, Chi-squared, Wasserstein | Batch — computed on configurable windows (time or count) |
| **Concept drift** (`concept_drift.py`) | DDM, EDDM, ADWIN, Page-Hinkley | Streaming — fed per-prediction error signals when actuals are provided |
| **Model drift** (`model_drift.py`) | Performance metric tracking (accuracy, F1, AUC, etc.) | Batch — compared against registered baselines |

**Count-based auto-check:** when enabled via `drift.auto_check`, the client automatically triggers `check_drift()` in a background thread after every N predictions.

### Feature Health (`feature_health.py`)

Per-feature drift scores weighted by SHAP importance (or permutation, or built-in feature importance). Alerts when the top-N most important features drift simultaneously.

### Cost Monitor (`cost_monitor.py`)

Tracks inference latency (p50/p95/p99), throughput (requests per second), cost-per-prediction, and compute utilisation with configurable alert thresholds.

### Cohort Analysis (`cohort_analyzer.py`)

Slices predictions by configurable dimensions (user segment, region, product type) and computes per-cohort performance metrics. Surfaces cohorts where the model underperforms relative to the population — critical for fairness monitoring in regulated industries.

Every detector produces a typed report (`DriftReport`, `QualityReport`, `FeatureHealthReport`, `CohortPerformanceReport`) defined in `sentinel/core/types.py`. The shape is identical regardless of which detector was used — downstream layers don't care.

---

## Layer 3 — LLMOps

**Package:** `sentinel.llmops`

The LLM-specific monitoring layer. Optional — only loaded when `llmops.enabled: true`.

### Prompt Manager (`prompt_manager.py`)

- Versioned prompt bundles (system prompt + template + few-shot examples)
- A/B routing with configurable traffic splits between prompt versions
- Performance metadata per version (quality score, token usage, latency, guardrail violation rate)
- Pluggable backend: local filesystem, Azure Blob, S3

### Guardrails Pipeline (`guardrails/`)

Pre- and post-LLM safety checks. The pipeline runs each rule and short-circuits on the first `block`. Every guardrail extends `BaseGuardrail` with `check(content, context) → GuardrailResult`.

**Input guardrails** (before LLM call):

| Guardrail | What it does |
|-----------|-------------|
| PII detection | Regex + Presidio-based detection; redact, block, or warn. Supports person, SSN, email, phone, account numbers |
| Jailbreak detection | Heuristic patterns + embedding similarity against known attack prompts |
| Topic fence | Reject or flag queries outside the agent's intended scope |
| Token budget | Block inputs that would exceed cost thresholds |

**Output guardrails** (after LLM call):

| Guardrail | What it does |
|-----------|-------------|
| Toxicity | Block responses containing toxic or harmful content |
| Groundedness | NLI / chunk-overlap / LLM-judge hallucination detection for RAG |
| Format compliance | Verify output matches expected JSON schema or structure |
| Regulatory language | Block prohibited phrases per compliance rules |

### Response Quality (`quality/`)

- **LLM-as-judge evaluator** — configurable rubrics (relevance, completeness, clarity, safety) scored by a cheaper judge model
- **Heuristic scoring** — rule-based checks: length, format compliance, keyword presence, readability
- **Reference-based** — BLEU, ROUGE, BERTScore against golden answers
- **Semantic drift detection** — embed LLM outputs, track centroid shift using cosine distance over configurable windows
- **RAG retrieval quality** — relevance, chunk utilisation, faithfulness, answer coverage

### Token Economics (`token_economics.py`)

- Token usage and cost tracking per query, model, prompt version, and user segment
- Configurable pricing tables (OpenAI, Azure OpenAI, open-source models via custom entries)
- Budget alerts: daily cost, per-query max tokens, cost-per-query trend increase
- Model routing decision logging

### Prompt Drift (`prompt_drift.py`)

Composite signal from:
- Quality scores trending downward
- Guardrail violation rate increasing
- Token usage per query increasing
- Semantic drift in outputs (embedding distribution shift)

---

## Layer 4 — AgentOps

**Package:** `sentinel.agentops`

The agent-specific monitoring and safety layer. Optional — only loaded when `agentops.enabled: true`.

### Trace System (`trace/`)

Span-based tracing compatible with OpenTelemetry. Each agent run produces a tree of spans capturing reasoning steps, tool calls, and outputs.

- **Decorator pattern**: `@tracer.trace_agent("claims_processor")`
- **Middleware pattern**: `LangGraphMiddleware(tracer).wrap(compiled_graph)` — duck-typed, no framework import at module level
- **Exporters**: OTLP (Jaeger/Zipkin), Arize Phoenix, local JSON
- **Auto-instrumentation**: LangGraph, Semantic Kernel, Google ADK via config toggle

### Tool Audit (`tool_audit/`)

- **`monitor.py`** — tool call success/failure/latency tracking (p50, p95, p99)
- **`permissions.py`** — per-agent tool allowlist/blocklist enforcement
- **`replay.py`** — record and replay tool calls for debugging; mock tool responses for testing
- Parameter validation against schemas before execution
- Per-tool, per-agent rate limits

### Safety (`safety/`)

| Module | What it enforces |
|--------|-----------------|
| **Loop detector** (`loop_detector.py`) | Max iterations, max repeated tool calls, max delegation depth, thrashing detection within a sliding window |
| **Budget guard** (`budget_guard.py`) | Token budget, cost budget, time budget, action budget per agent run. On exceeded: graceful stop, escalate, or hard kill |
| **Escalation** (`escalation.py`) | Confidence below threshold → human handoff. Consecutive tool failures → human handoff. Sensitive data or regulatory context detected → human approval |
| **Sandbox** (`sandbox.py`) | Destructive operations (write, delete, execute, transfer) intercepted with approve-first, dry-run, or sandbox-then-apply modes |

Safety modules **never silently swallow errors** — every detection and action is logged to the audit trail before enforcement.

### Agent Registry (`agent_registry.py`)

- Agent versioning with capability manifests
- Tool permission sets per agent
- Performance baselines (task completion rate, latency, cost)
- A2A (agent-to-agent) discovery and health checks

### Multi-Agent Monitoring (`multi_agent/`)

- **`delegation.py`** — delegation chain tracking across agents (who delegated to whom, with what input, what was returned)
- **`consensus.py`** — agreement/disagreement tracking for multi-agent decisions; configurable min agreement threshold and conflict action (escalate, majority vote, weighted vote)
- **`orchestration.py`** — bottleneck detection, fan-out/fan-in pattern tracking, straggler alerts

### Evaluation (`eval/`)

- **`task_completion.py`** — binary and graded task success tracking, per task type
- **`trajectory.py`** — compare actual step sequences against optimal trajectories; penalise unnecessary steps
- **`golden_datasets.py`** — curated input → expected output → expected trajectory test suites; auto-run on version changes or daily schedule

---

## Layer 5 — Intelligence

**Package:** `sentinel.intelligence`

Correlates signals from Layers 2–4 with business context. Answers the question "does this drift actually matter?"

### Model Graph (`model_graph.py`)

Multi-model dependency DAG. When an upstream model drifts, cascade alerts propagate to all downstream consumers. Configured via `model_graph.dependencies` in YAML.

A 0.3 PSI score is meaningless on its own; "PSI 0.3 on `merchant_country`, top-3 important feature, downstream auto-adjudication model affected, projected fraud_catch_rate impact -2.4%" is actionable.

### KPI Linker (`kpi_linker.py`)

Maps model metrics to business KPIs (e.g. `precision → fraud_catch_rate`) with optional warehouse data sources. Supports auto-refresh from configured data sources so that business impact assessment uses current values.

### Explainability (`explainability.py`)

- SHAP integration for per-prediction explanations
- Permutation importance as a lighter alternative
- Compliance report generators that attach explanation summaries to audit events

---

## Layer 6 — Action

**Package:** `sentinel.action`

Does something about the signals from Layers 2–5. This is where Sentinel's "config over code" promise pays off.

### Notifications (`notifications/`)

**Architecture**: `engine.py` → alert router → `channels/` → delivery

| Component | Details |
|-----------|---------|
| **Channels** | 6 built-in: Slack, Teams, PagerDuty, email, webhook (generic JSON POST), SNS (pending). All extend `BaseChannel` |
| **Severity levels** | `info`, `warning`, `high`, `critical` |
| **Cooldown** | Suppress repeated alerts for the same issue within a configurable window |
| **Rate limiting** | Max alerts per hour per channel to prevent alert fatigue |
| **Escalation** | Background `EscalationTimer` (heap-based priority queue, daemon thread) fires time-delayed callbacks. Escalation chains defined in YAML |
| **Digest mode** | Batch alerts into periodic summaries instead of firing individually |
| **Templates** | Jinja2-based customisable message templates per channel |
| **Fingerprinting** | Policy-based alert deduplication via `policies.py` |

### Retrain Orchestrator (`retrain/`)

Full pipeline: drift detection → trigger evaluation → pipeline execution → holdout validation → human approval → model promotion → optional auto-deployment.

| Component | Details |
|-----------|---------|
| **Triggers** (`triggers.py`) | `drift_confirmed`, `scheduled`, `manual` |
| **Approval** (`approval.py`) | Three modes: `auto` (promote if validation passes), `human_in_loop` (require explicit approval with timeout), `hybrid` (auto-promote if improvement exceeds threshold, otherwise require human) |
| **Validation** | Evaluate retrained model against holdout dataset with configurable minimum performance thresholds |
| **Auto-deploy** | When `deploy_on_promote: true`, the orchestrator calls `DeploymentManager.start()` after promotion, closing the retrain-to-deploy loop |

### Deployment Automation (`deployment/`)

**4 strategies** (all extending `BaseDeploymentStrategy`):

| Strategy | Behaviour |
|----------|-----------|
| **Shadow** | New model runs in parallel; predictions logged but not served |
| **Canary** | Route a small % of traffic to new model; ramp up or rollback automatically |
| **Blue-green** | Two identical environments; atomic traffic switch |
| **Direct** | Replace in-place (non-critical models only) |

**6 deployment targets** (all extending `BaseDeploymentTarget`):

| Target | Cloud | Method |
|--------|-------|--------|
| `local` | — | Local process management |
| `azure_ml_endpoint` | Azure | Azure ML managed endpoint traffic split |
| `azure_app_service` | Azure | App Service slot swaps |
| `aks` | Azure | Kubernetes replica scaling |
| `sagemaker_endpoint` | AWS | SageMaker production variant weights |
| `vertex_ai_endpoint` | GCP | Vertex AI endpoint traffic split |

**Champion-challenger comparison** (`promotion.py`): evaluates candidate vs incumbent on holdout or live traffic. Auto-promotion rules and configurable rollback triggers (error rate spike, latency degradation).

Strategy × target compatibility is validated at config load time (e.g. `canary` + `azure_app_service` slot-swap is rejected).

---

## Layer 7 — Foundation

**Package:** `sentinel.foundation`

The persistence and lineage backbone. Every other layer depends on this one — and only on this one.

### Model Registry (`registry/`)

Stores model artifacts, metadata, performance baselines, feature schemas, and training data references.

**6 backends** (all extending `BaseRegistryBackend`):

| Backend | Config value | Cloud |
|---------|-------------|-------|
| Local filesystem | `local` | — |
| Azure ML | `azure_ml` | Azure |
| MLflow | `mlflow` | Any |
| SageMaker | `sagemaker` | AWS |
| Vertex AI | `vertex_ai` | GCP |
| Databricks Unity Catalog | `databricks` | Databricks |

All backends support: `save`, `load`, `list_versions`, `list_models`, `delete`, `exists`, plus optional artifact storage (`save_artifact`, `load_artifact`, `has_artifact`). Semantic versioning via `versioning.py`.

### Audit Trail (`audit/`)

The **immutable spine** of the system. Every model action is logged:

- Model registration, promotion, deprecation, rollback
- Drift detection (with full statistical details)
- Alert dispatch (channel, severity, payload)
- Retrain triggers (reason, pipeline ID)
- Deployment changes (strategy, traffic split, approval details)
- Predictions and explanations (optional, configurable)

| Component | Details |
|-----------|---------|
| **`trail.py`** | Append-only JSONL log with daily file rotation and configurable retention |
| **`integrity.py`** | HMAC-SHA256 hash chain for tamper evidence. Each event carries `previous_hash` and `event_hmac`. Verified via `sentinel audit verify` |
| **`keystore.py`** | Pluggable key storage (env-based or file-based) for the HMAC signing key |
| **`lineage.py`** | Data → training → model → prediction lineage graph with save/load persistence |
| **`compliance.py`** | FCA Consumer Duty, EU AI Act, and internal audit report generators |
| **`shipper.py`** | `BaseAuditShipper` with `ThreadedShipper` base. Cloud shippers: `AzureBlobShipper`, `S3Shipper`, `GcsShipper` — background upload with retry, never blocking the hot write path |

### Experiment Tracker (`experiments/tracker.py`)

Links training experiments to production deployments:
- Create and manage experiment runs with metric history
- Nested run support (parent/child experiments)
- Filter experiments by status, tags, and metrics (with OR support)
- Compare runs across metrics

### Dataset Registry (`datasets/`)

- **`registry.py`** — register, query, compare, and link dataset versions with content hashing
- **`lineage.py`** — dataset version lineage graphs
- **`hashing.py`** — deterministic content hashing for reproducibility

---

## Domain Adapters

**Package:** `sentinel.domains`

Most ML monitoring tools assume tabular data. Sentinel uses a **domain adapter pattern** so the same SDK covers five ML paradigms without polluting the core.

```python
# sentinel/domains/base.py
class BaseDomainAdapter(ABC):
    @abstractmethod
    def get_drift_detectors(self) -> list[BaseDriftDetector]: ...

    @abstractmethod
    def get_quality_metrics(self) -> list[BaseMetric]: ...

    @abstractmethod
    def get_schema_validator(self) -> BaseSchemaValidator: ...
```

The adapter is resolved at `SentinelClient` init time based on `model.domain` in the config:

### Tabular (default)

**Adapter:** `TabularAdapter` — the default when no domain is specified. Backward compatible with all pre-domain-adapter configs.

- **Drift:** PSI, KS, Jensen-Shannon, Chi-squared, Wasserstein
- **Quality:** accuracy, F1, AUC, precision, recall

### Time Series

**Adapter:** `TimeSeriesAdapter` — for demand forecasting, pricing models, claims volume prediction. Extra: `sentinel[timeseries]` (statsmodels, pmdarima).

What makes time series special: observations are temporally dependent. A "drift" in January vs July might just be seasonality, not a problem. The adapter handles this with:

- **Seasonality-aware baselines** — STL decomposition into trend, seasonal, residual. Compare components separately.
- **Calendar-aware drift tests** — compare current window against same calendar period from reference (Jan 2026 vs Jan 2025).
- **Stationarity monitoring** — running ADF and KPSS tests on sliding windows.
- **Forecast quality** — MASE, MAPE, coverage, interval width, Winkler score, directional accuracy.
- **Per-horizon tracking** — quality metrics at each forecast horizon (1-step, 7-step, 30-step).

### NLP (non-LLM)

**Adapter:** `NLPAdapter` — for NER, sentiment analysis, text classification. Extra: `sentinel[nlp-domain]` (sentence-transformers, spacy).

- **Vocabulary drift** — OOV rate tracking, new token surfacing, vocabulary entropy monitoring.
- **Embedding space drift** — MMD or Mahalanobis distance on sentence embeddings.
- **Label distribution** — Chi-squared test on predicted label frequencies.
- **Text statistics** — document length, language distribution, special character patterns.
- **Quality** — token-level F1, span exact match (NER), macro/micro F1 (classification).

### Recommendation

**Adapter:** `RecommendationAdapter` — for collaborative filtering, content-based, and neural recommenders. Extra: `sentinel[recommendation]` (implicit, recbole).

- **Item distribution drift** — catalogue cold-start ratio, long-tail collapse detection.
- **User distribution drift** — segment composition changes, interaction pattern shifts.
- **Beyond-accuracy metrics** — NDCG@K, MAP@K, coverage, diversity (ILS), novelty, serendipity, popularity bias (Gini coefficient), group fairness.

### Graph ML

**Adapter:** `GraphAdapter` — for GNNs, knowledge graph completion, transaction networks. Extra: `sentinel[graph]` (networkx, torch-geometric).

- **Topology drift** — degree distribution shift (KS test), density monitoring, clustering coefficient, connected components, diameter tracking.
- **Node/edge feature drift** — standard PSI/KS applied to graph features.
- **KG-specific** — relation coverage, entity OOV rate, plausibility score trends.
- **Quality** — AUC-ROC (link prediction), Hits@K, MRR (KG completion), node classification F1, embedding isotropy.

### Design rules for all adapters

1. **Stateless** — all configuration comes from the YAML config under `domains.<name>`.
2. **Lazy imports** — heavy optional deps imported inside methods only. `import sentinel` never requires statsmodels, networkx, or torch-geometric.
3. **Same ABC** — domain drift detectors extend `BaseDriftDetector` and produce the same `DriftReport` schema. They flow through the same notification, audit, and dashboard pipelines.
4. **No core schema expansion** — each adapter parses its own subtree under `domains.<name>`.

---

## Data Flow

### Prediction-time (synchronous)

```
1. Request hits your serving framework (FastAPI, Flask, Triton, ...)
        │
        ▼
2. SentinelClient.check_data_quality(features)        ← Layer 2
        │  validates schema, freshness, outliers
        ▼
3. (LLMOps) GuardrailPipeline.check_input(text)       ← Layer 3
        │  PII redact, jailbreak, topic fence, token budget
        ▼
4. (AgentOps) AgentTracer.start_span("plan")           ← Layer 4
        │  span begins, budget guard armed
        ▼
5. Your model.predict(features)
        │
        ▼
6. (LLMOps) GuardrailPipeline.check_output(response)   ← Layer 3
        │  groundedness, toxicity, format compliance
        ▼
7. SentinelClient.log_prediction(features, prediction)
        │  buffers for windowed drift computation      ← Layer 2
        │  appends to audit trail (optional)           ← Layer 7
        ▼
8. Return to caller
```

When `actual` values are provided, streaming concept drift detectors (DDM, EDDM, ADWIN, Page-Hinkley) are fed per-prediction error signals in real-time.

### Background processing (asynchronous)

On a configurable cadence (time-based, count-based, or via `drift.auto_check`):

```
A. DriftDetector.detect(buffered_window)              ← Layer 2
        │  produces DriftReport
        ▼
B. KPILinker.assess_business_impact(report)           ← Layer 5
        │  is this drift meaningful to the business?
        ▼
C. ModelGraph.cascade(report)                         ← Layer 5
        │  which downstream models are affected?
        ▼
D. NotificationEngine.dispatch(alert)                 ← Layer 6
        │  cooldown, rate limit, escalation, digest
        │  delivers to Slack/Teams/PagerDuty/email/webhook
        │  schedules future escalation steps via EscalationTimer
        ▼
E. RetrainOrchestrator.maybe_trigger(report)          ← Layer 6
        │  trigger=drift_confirmed: kick off pipeline
        │  validate → human approval → promote
        │  deploy_on_promote=true: auto-deploy via DeploymentManager
        ▼
F. AuditTrail.log_event(...)                          ← Layer 7
        │  tamper-evident HMAC hash chain
        │  cloud shipper uploads day-rotated files
```

Every step in this flow is configurable, swappable, or skippable from YAML.

### Feedback loops

1. **Retrain → Observability.** A successful retrain promotes a new baseline; the drift detector resets its reference.
2. **Foundation → Observability.** New model registration flows baseline metrics and feature schema into data quality and feature health monitors.
3. **Agent traces → Guardrails.** Tool call failures and loop detections inform guardrail policy tuning.
4. **Prompt drift → Retrain orchestrator.** Quality decline + semantic shift can trigger prompt A/B tests the same way model drift triggers retraining.
5. **Audit trail → Compliance.** The compliance module reads the audit trail to generate regulatory reports — no separate logging needed.

---

## Cross-Cutting Concerns

### Thread safety

- `SentinelClient` is thread-safe. Internal state protected by `RLock`s; prediction buffer is a bounded thread-safe queue.
- Heavy computations (drift, SHAP, semantic drift embeddings) run in background threads. They never block prediction serving.
- The audit trail is append-only with a process-level write lock.
- Notification delivery is async — a slow webhook never stalls the calling thread.
- Deployment state transitions are serialised per `deployment_id` to prevent races.
- All daemon threads (`DriftScheduler`, `EscalationTimer`, `ThreadedShipper`) follow the same pattern: `threading.Event` for stop signalling, `daemon=True`, and `close()` for clean shutdown.

### Config validation

- Pydantic v2 schema validates all config on load with clear error messages and fix suggestions.
- `SecretStr` fields for all sensitive values; `sentinel config show` masks them as `<REDACTED>`.
- Strategy × target compatibility matrix validated at load time.
- `model_validator` checks ensure required sub-configs are present for each backend choice.
- Environment variable substitution, cloud secret resolution, and `extends:` inheritance all resolve before validation.

### Error hierarchy

All custom exceptions extend `SentinelError`. Module-specific exceptions (`DriftDetectionError`, `DeploymentError`, `ConfigError`, `RegistryError`, `AuditError`, `ModelNotFoundError`) extend module-specific bases for targeted handling.

### Structured logging

All modules use `structlog` for structured, JSON-compatible logging. Never `print()` or stdlib `logging` directly.

### Dependency rules

1. **Lower layers never import higher layers.** Foundation has no knowledge of Action or LLMOps.
2. **Cloud SDKs live in `sentinel/integrations/`.** Core modules never import `azure-ai-ml`, `boto3`, or `google-cloud-*` directly.
3. **Optional dependencies are lazy.** `import sentinel` requires only: numpy, pydantic, pyyaml, click, structlog, croniter.
4. **LLMOps and AgentOps are opt-in.** Config flags default to `enabled: false`.
5. **Reports have a stable shape.** Every detector produces the same `DriftReport` — downstream layers never branch on the source.
6. **Audit trail is the source of truth.** Any regulatory-significant action must be persisted before it is considered complete.

---

## Extension Points

Every pluggable component has an ABC. To add a new implementation, subclass the ABC and register it.

| Extension point | ABC | Registry | Used for |
|---|---|---|---|
| Drift detector | `BaseDriftDetector` | `DRIFT_REGISTRY` | Custom statistical tests, domain-specific drift |
| Notification channel | `BaseChannel` | `CHANNEL_REGISTRY` | New alerting destinations (OpsGenie, ServiceNow, Datadog) |
| Deployment strategy | `BaseDeploymentStrategy` | `STRATEGY_REGISTRY` | Custom rollout patterns |
| Deployment target | `BaseDeploymentTarget` | resolved by config | New cloud endpoints |
| Registry backend | `BaseRegistryBackend` | resolved by config | Alternative model stores |
| Audit shipper | `BaseAuditShipper` / `ThreadedShipper` | resolved by config | New cloud storage backends |
| Guardrail | `BaseGuardrail` | `GUARDRAIL_REGISTRY` | New input/output safety checks |
| Quality evaluator | `BaseEvaluator` | `EVALUATOR_REGISTRY` | Custom LLM scoring methods |
| Trace exporter | `BaseExporter` | `EXPORTER_REGISTRY` | New observability backends (Datadog APM, Honeycomb) |
| Domain adapter | `BaseDomainAdapter` | `DOMAIN_REGISTRY` | New ML paradigms |

### Example: adding a custom notification channel

```python
from sentinel.action.notifications.channels.base import BaseChannel
from sentinel.action.notifications import register_channel
from sentinel.core.types import Alert, DeliveryResult

class OpsGenieChannel(BaseChannel):
    name = "opsgenie"

    def __init__(self, api_key: str, team: str, **config):
        super().__init__(**config)
        self.api_key = api_key
        self.team = team

    def send(self, alert: Alert) -> DeliveryResult:
        # ... HTTP POST to OpsGenie API ...
        return DeliveryResult(channel=self.name, delivered=True)

register_channel("opsgenie", OpsGenieChannel)
```

### Example: adding a new domain adapter

1. Create `sentinel/domains/my_domain/adapter.py` implementing `BaseDomainAdapter`
2. Provide `get_drift_detectors()`, `get_quality_metrics()`, `get_schema_validator()`
3. Register in `sentinel/domains/__init__.py` under `DOMAIN_REGISTRY`
4. Add optional dependencies to `pyproject.toml` as a new extra
5. Use lazy imports so `import sentinel` stays lightweight

### Example: adding a new deployment target

1. Implement `BaseDeploymentTarget` with `set_traffic_split()`, `health_check()`, `rollback_to()`
2. Place under `sentinel/integrations/{provider}/`
3. Add config sub-model to `sentinel/config/schema.py`
4. Add factory branch to `DeploymentManager._build_target()`
5. Update the strategy × target compatibility matrix

---

## Configuration Philosophy

Sentinel is **config-first by design**, not as an aesthetic preference:

1. **Auditability.** A YAML config has `git blame`. Compliance officers can review threshold changes without reading code.
2. **Non-engineer review.** Risk officers and compliance teams can read and approve YAML.
3. **GitOps deployment.** Merge a config change → CI/CD applies it. No code changes for a threshold tweak, a new alert channel, or a rollout strategy change.

Code is reserved for genuinely novel logic — a new drift detector, a new guardrail, a new domain adapter. Operating the system never requires writing code.

---

## Why These Layers?

- **Observability must be independent of action.** Detection produces reports; action consumes them. Mixing them creates systems that can't be tested or tuned in isolation.
- **LLMOps and AgentOps need separate layers** because their failure modes are fundamentally different. Token budgets, jailbreaks, and tool loops have no analogue in traditional ML.
- **Intelligence sits between observation and action** because raw signals lie. A 0.3 PSI on a low-importance feature is noise. A 0.05 PSI on the top SHAP feature is an emergency.
- **Foundation is at the bottom** because every layer above needs immutable state. Versioning, auditing, and lineage aren't features — they're the substrate that makes the rest defensible to a regulator.

What's novel isn't any individual layer — it's having all seven in one SDK, accessible through one config file, governed by one audit trail.

---

## Further Reading

- **[Quickstart](quickstart.md)** — from `pip install` to a first alert in 10 minutes
- **[Config reference](config-reference.md)** — every YAML field, type, and default
- **[Cloud integration guide](cloud-integration-guide.md)** — Azure, AWS, GCP backends and `sentinel cloud test`
- **[Security](security.md)** — RBAC, CSRF, audit trail integrity, config signing
- **[Azure guide](azure.md)** — Azure-specific RBAC, Key Vault, and deployment details
- **[`configs/examples/`](../configs/examples/)** — realistic configs spanning every supported scenario
- **[`CLAUDE.md`](../CLAUDE.md)** — canonical project specification
