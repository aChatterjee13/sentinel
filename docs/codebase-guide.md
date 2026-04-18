# Codebase guide — Project Sentinel

> **Audience:** developers joining the Sentinel project for the first time.
> You want to understand *the repository itself* — what every folder does,
> what every YAML section means, and how the pieces fit together — before
> you start contributing.
>
> **If instead you are a user who wants to integrate Sentinel into your
> own model-serving application**, read [`developer-guide.md`](developer-guide.md).
> That guide is a step-by-step integration tutorial; this guide is a tour
> of the codebase.

---

## Table of contents

1. [Who this guide is for](#1-who-this-guide-is-for)
2. [30-second orientation](#2-30-second-orientation)
3. [The three mental models you need](#3-the-three-mental-models-you-need)
4. [Repository layout — top level](#4-repository-layout--top-level)
5. [The `sentinel/` package — layer-by-layer tour](#5-the-sentinel-package--layer-by-layer-tour)
6. [YAML config — philosophy and loading pipeline](#6-yaml-config--philosophy-and-loading-pipeline)
7. [Config-to-code mapping](#7-config-to-code-mapping)
8. [Example configs — what each one demonstrates](#8-example-configs--what-each-one-demonstrates)
9. [Data flow — a prediction's journey through Sentinel](#9-data-flow--a-predictions-journey-through-sentinel)
10. [How to extend Sentinel](#10-how-to-extend-sentinel)
11. [Test layout](#11-test-layout)
12. [Dashboard internals](#12-dashboard-internals)
13. [CLI internals](#13-cli-internals)
14. [Conventions and standards](#14-conventions-and-standards)
15. [Thread safety patterns](#15-thread-safety-patterns)
16. [Your first 30 minutes](#16-your-first-30-minutes)
17. [Your first hour](#17-your-first-hour)
18. [Glossary](#18-glossary)
19. [Where to go next](#19-where-to-go-next)

---

## 1. Who this guide is for

You are one of the following:

- **A new developer contributing to Sentinel itself.** You need to understand
  what every folder is for, what every file does, and how to add a new
  drift detector / guardrail / deployment strategy without breaking the
  existing contracts.
- **A reviewer picking up a Sentinel pull request.** You want a mental
  map of where things live so you know which files to scrutinise.
- **A technical lead evaluating Sentinel** before approving it for your
  team's stack. You want to understand the architecture at the code
  level, not just the marketing level.

This guide assumes you are comfortable with:

- Python 3.10+ (type hints, dataclasses, `match` statements, `Protocol`s)
- YAML syntax
- The basics of machine learning operations (what "drift" means, what a
  model registry is, what canary deployment is)
- A little bit of Pydantic v2 (we use it heavily for config validation)

You do **not** need prior exposure to LLMs, agent frameworks, Azure, or
any specific ML library. Sentinel treats all of those as optional
extras behind lazy imports.

---

## 2. 30-second orientation

**What Sentinel is:** one `pip install` that gives an ML team a
config-driven SDK covering drift detection, data quality, model
registry, audit trail, alerting, deployment automation, LLMOps
(prompts, guardrails, token economics), AgentOps (tracing, tool
audit, budget guards), and a local-first dashboard.

**What makes it unusual:**

1. **Every behaviour is in YAML.** A single `sentinel.yaml` drives
   the whole SDK. Python code is only needed at the integration
   seam (`client.log_prediction(...)` in your serving endpoint).
2. **One SDK, three operational paradigms.** Traditional MLOps,
   LLMOps, and AgentOps all share the same client, the same audit
   trail, the same notification engine, and the same config file.
3. **Five domain adapters.** Tabular, time series, NLP,
   recommendation, and graph ML — swap `domain: timeseries` and
   the SDK uses seasonality-aware drift detection automatically.
4. **Compliance first.** Every model event is logged with an
   HMAC-SHA256 hash chain so audit logs are tamper-evident.
5. **Cloud portable.** Azure-first (Key Vault, Azure ML, Blob, App
   Service, AKS), with shipped backends for AWS (Secrets Manager, S3,
   SageMaker), GCP (Secret Manager, GCS, Vertex AI), and Databricks
   (Unity Catalog). Nothing is cloud-locked at the core.

**What Sentinel is not:**

- Not a model training framework. Bring your own sklearn / PyTorch /
  XGBoost / LangGraph / CrewAI / whatever.
- Not a SaaS product. It is a Python library with a local FastAPI
  dashboard. You run it inside your own infrastructure.
- Not a replacement for your serving framework. You still run
  FastAPI / Flask / Flyte / Airflow / Ray Serve. Sentinel is called
  from inside your handlers.

---

## 3. The three mental models you need

Before diving into folders, internalise these three patterns. **Every
subsystem in the codebase uses them.** Once you see them, the rest of
the code is predictable.

### 3.1 `SentinelClient` is the only front door

```
your application code
        │
        ▼
   SentinelClient   ◄── built from sentinel.yaml via from_config()
        │
        ├─► ModelRegistry        (backend: local | azure_ml | mlflow | sagemaker | vertex_ai | databricks)
        ├─► DriftEngine          (detectors chosen by YAML)
        ├─► DataQuality          (schema + freshness + outliers)
        ├─► FeatureHealth
        ├─► NotificationEngine   (channels chosen by YAML)
        ├─► AuditTrail           (backend: local | azure_blob | s3 | gcs)
        ├─► DeploymentManager    (strategy + target chosen by YAML)
        ├─► LLMOpsClient         (only if llmops.enabled)
        ├─► AgentOpsClient       (only if agentops.enabled)
        ├─► CohortAnalyzer       (only if cohort_analysis.enabled)
        ├─► ExplainabilityEngine  (only if explain extras installed)
        ├─► DatasetRegistry       (always available)
        ├─► ExperimentTracker     (always available)
        ├─► ModelGraph            (only if model_graph.dependencies set)
        ├─► KPILinker             (only if business_kpi.mappings set)
        └─► DashboardRouter      (only if dashboard extras installed)
```

Users never construct any of the sub-components directly. They call
`SentinelClient.from_config("sentinel.yaml")` and get one object that
wires everything. That wiring lives in
`sentinel/core/client.py:SentinelClient.__init__` (~1,300 lines) —
it's the "factory" that turns a validated config into live
subsystems.

**Implication for contributors:** almost every new feature ends with
a new field in `SentinelConfig`, a new attribute on `SentinelClient`,
and one line in `SentinelClient.__init__` that constructs it. If
your PR doesn't touch `client.py` or `schema.py`, you are probably
only touching a leaf module, which is usually a good sign.

### 3.2 ABC + registry + config `Literal` = every plugin

Every pluggable component in Sentinel — drift detectors, notification
channels, deployment strategies, deployment targets, registry
backends, audit shippers, guardrails, quality evaluators, trace
exporters, domain adapters — follows the same three-piece pattern:

1. **An Abstract Base Class** (`BaseDriftDetector`, `BaseChannel`,
   `BaseDeploymentStrategy`, `BaseDeploymentTarget`,
   `BaseRegistryBackend`, `BaseAuditShipper`, `BaseGuardrail`,
   `BaseEvaluator`, `BaseExporter`, `BaseDomainAdapter`). The ABC
   defines the contract (what methods a subclass must implement).
2. **A module-level registry dict**, e.g.
   `DRIFT_DETECTOR_REGISTRY = {"psi": PSIDriftDetector, "ks": KSDriftDetector, ...}`.
   The registry maps string keys (used in YAML) to concrete classes.
3. **A `Literal` field in the Pydantic schema**, e.g.
   `method: Literal["psi", "ks", "js", "chi_squared", "wasserstein"]`.
   The `Literal` enforces that only registered names are valid at
   config-load time, before any runtime dispatch.

Adding a new drift detector is always the same four steps:

1. Implement `MyDetector(BaseDriftDetector)` in a new file.
2. Add `"my_method": MyDetector` to the registry dict.
3. Add `"my_method"` to the `Literal[...]` in `schema.py`.
4. Write tests and update `docs/config-reference.md`.

You never touch `SentinelClient` unless you are adding a whole new
subsystem. See section 10 for worked examples.

### 3.3 The audit trail is the source of truth

Every meaningful event in Sentinel is written to the audit trail:

- a prediction is logged
- drift is detected
- an alert fires
- a model is registered / promoted / rolled back
- a retrain is triggered
- a deployment advances to the next canary step
- a guardrail blocks a request
- an agent exceeds its budget
- a config is signed or loaded

The audit trail lives in `sentinel/foundation/audit/trail.py`. As
of security hardening it is **tamper-evident**: every event is hashed together
with the previous event's hash using HMAC-SHA256, producing a
chain that cannot be edited without invalidating all subsequent
entries (`sentinel audit verify` checks the chain).

**Implication for contributors:** when you add a new feature that
does something meaningful, you almost certainly need to call
`audit_trail.record(event)` at the right moment. Ask "can a
regulator ask 'when did this happen and why?' about this event?"
If yes, it belongs in the audit trail.

---

## 4. Repository layout — top level

```
project-sentinel/
├── CLAUDE.md              # Project constitution — read this before ANY PR
├── CHANGELOG.md           # Release notes, grouped by workstream
├── README.md              # Public-facing quickstart
├── pyproject.toml         # Package config, dependencies, extras
│
├── sentinel/              # The actual SDK — 211 Python files
├── tests/                 # unit + integration + fixtures (2,212 tests, 87% line coverage)
├── docs/                  # User-facing documentation (this file lives here)
├── configs/               # Example YAML configs (see section 8)
├── demo/                  # End-to-end demo scripts + sample data
├── scripts/               # One-off helper scripts (codegen, audits)
├── agents/                # Sample agent definitions (for AgentOps demos)
├── audit/                 # Default local audit log directory (created at runtime)
├── prompts/               # Default local prompt registry (created at runtime)
├── registry/              # Default local model registry (created at runtime)
│
├── .coverage              # pytest-cov output (gitignored)
├── .mypy_cache/           # mypy type-check cache (gitignored)
├── .pytest_cache/         # pytest cache (gitignored)
└── .ruff_cache/           # ruff cache (gitignored)
```

### Files you must read before your first PR

| File | Why |
|---|---|
| `CLAUDE.md` | The project constitution. Architecture diagram, module specs, coding standards, roadmap. If your PR contradicts `CLAUDE.md`, your PR is wrong. |
| `pyproject.toml` | Source of truth for dependencies, Python version, optional extras, ruff + mypy config. |
| `docs/developer-guide.md` | Step-by-step integration tutorial (for SDK *users*, but still worth skimming so you know what we promise). |
| `docs/config-reference.md` | Every YAML field documented with accepted values. This is the contract we expose to users. |
| `docs/architecture.md` | High-level layer diagram and cross-layer data flows. |
| `docs/security.md` | Audit hash chain, RBAC, CSRF, signed configs. |
| `docs/azure.md` | Key Vault, Azure ML registry, Azure deploy targets. |

### Directories you can ignore for now

- `audit/`, `prompts/`, `registry/` — these are **runtime outputs**.
  Sentinel creates them when it runs locally. They are in
  `.gitignore`. Don't commit anything here.
- `.coverage`, `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/` —
  tool caches. Ignore.
- `sentinel_mlops.egg-info/` — pip install metadata. Ignore.

### Directories worth a quick skim

- `demo/` — the end-to-end "salesy" demo. Useful if you want to
  see Sentinel working without wiring anything yourself.
- `scripts/` — operational scripts (regenerating schema samples,
  running audits). Nothing here is imported by the SDK at runtime.
- `agents/` — sample agent definitions (LangGraph + Semantic
  Kernel skeletons) used by the AgentOps demo configs in
  `configs/examples/multi_agent_underwriting.yaml`.

---

## 5. The `sentinel/` package — layer-by-layer tour

This is the heart of the guide. Every subpackage is explained in
four parts:

1. **What it does** — one sentence.
2. **Key files** — what each file contains.
3. **Public API** — what external code is expected to touch.
4. **Extension points** — what ABCs + registries live here.

The order below follows the 7-layer architecture diagram in
`CLAUDE.md`, starting from the bottom (foundation) and going up
(developer interface).

### 5.1 `sentinel/__init__.py` — public API surface

**What it does.** Re-exports everything users are meant to import
with `from sentinel import ...`. It is intentionally small (~94 lines).
If a symbol is not re-exported here, it is considered internal and
can break between releases.

**What it exports:**

- `SentinelClient` — the main entry point (see 5.2).
- Exception classes: `SentinelError` (base), `ConfigError`,
  `DriftDetectionError`, `DeploymentError`, `RegistryError`,
  `AuditError`, `GuardrailError`, `AgentError`,
  `BudgetExceededError`, `LoopDetectedError`, `DashboardError`,
  `DashboardNotInstalledError`.
- Data-transfer types: `DriftReport`, `QualityReport`,
  `FeatureHealthReport`, `Alert`, `AlertSeverity`, `PredictionRecord`.
- Dashboard symbols `create_dashboard_app` and
  `SentinelDashboardRouter` — these are **lazy-loaded** via
  `__getattr__` so that `import sentinel` doesn't fail when the
  `[dashboard]` extra isn't installed.
- `__version__`.

**Rule of thumb.** If a user has to do
`from sentinel.foundation.audit.trail import AuditTrail`, we have
a documentation / API-design problem. Anything a user needs should
be reachable from `sentinel.*` directly.

### 5.2 `sentinel/core/` — the client and shared types

```
sentinel/core/
├── __init__.py
├── client.py         # SentinelClient — ~1,300 lines, the factory + facade
├── exceptions.py     # SentinelError hierarchy
├── hooks.py          # Plugin/hook system for third-party extensions
├── scheduler.py      # Background drift scheduler (daemon thread)
└── types.py          # Shared dataclasses: DriftReport, Alert, etc.
```

**`client.py` — the single most important file.** Everything else
in the SDK converges here. Read it cover-to-cover on day 2 of
onboarding; you will understand 70% of the architecture. It is now
~1,300 lines with 55+ methods.

Key methods:

- `SentinelClient.from_config(path)` — load YAML, validate,
  instantiate every subsystem.
- `SentinelClient.__init__` — the factory that wires registry,
  drift engine, audit trail, notification engine, deployment
  manager, LLMOps client, AgentOps client.
- `log_prediction(features, prediction, actuals=None)` — the
  most-called method in production code.
- `check_drift()`, `check_data_quality(X)`, `get_feature_health()`.
- `register_model(model, metadata)`, `deploy(version, strategy,
  traffic_pct)`.
- Private `_build_*` helpers: `_build_audit_trail`,
  `_build_audit_shipper`, `_build_registry_backend`,
  `_build_notification_engine`, `_build_deployment_manager`.
  These are the config-to-object bridges — each one takes a
  validated config block and returns a constructed component.

**`exceptions.py`.** Every exception in Sentinel inherits from
`SentinelError`. Module-specific exceptions inherit from
module-specific bases (`DriftDetectionError`, `DeploymentError`,
etc.). Never raise `ValueError` or `RuntimeError` from SDK code —
always raise a `SentinelError` subclass so users can catch our
errors specifically.

**`types.py`.** Frozen dataclasses / Pydantic models for all
data-transfer objects. `DriftReport`, `QualityReport`, `Alert`,
`PredictionRecord`, `FeatureHealthReport`. These are the shapes
that flow between subsystems. They are **immutable by convention**;
if you find yourself mutating a `DriftReport` field, you are
probably doing something wrong.

**`hooks.py`.** A small pub-sub hook system for third-party
extensions. Not heavily used yet — most extension happens via
ABCs + registries, not hooks. Skim it once and move on.

### 5.3 `sentinel/config/` — load, validate, trace secrets

```
sentinel/config/
├── __init__.py
├── aws_secrets.py    # ${awssm:secret-name} resolver for AWS Secrets Manager
├── defaults.py       # Fallback defaults for all modules (DEFAULT_PRICING, etc.)
├── gcp_secrets.py    # ${gcpsm:project/secret} resolver for GCP Secret Manager
├── keyvault.py       # ${azkv:vault/secret} resolver for Azure Key Vault
├── loader.py         # 375 lines — YAML → dict → validate → SentinelConfig
├── references.py     # File-reference validation (baseline paths, schemas)
├── schema.py         # ~1,073 lines — EVERY Pydantic model for the YAML schema
├── secrets.py        # SecretStr wrappers + unwrap() helper
├── signing.py        # Config signing for tamper-evident configs
└── source.py         # Per-field source tracing for validation errors
```

**This is the most important non-`core` package.** Every new
feature that changes config behaviour lives here.

**`schema.py` (~1,073 lines).** The Pydantic v2 schema for
`sentinel.yaml`. Read it top-to-bottom at least once. The model
hierarchy is:

```
SentinelConfig  ◄── the root model, one per YAML file
├── version: str
├── extends: str | None           # for config inheritance
├── model: ModelConfig
├── data_quality: DataQualityConfig
├── drift: DriftConfig
│   ├── data: DataDriftConfig
│   ├── concept: ConceptDriftConfig
│   └── model: ModelDriftConfig
├── feature_health: FeatureHealthConfig
├── alerts: AlertConfig
│   ├── channels: list[ChannelConfig]   # discriminated union
│   └── policies: PolicyConfig
├── retraining: RetrainConfig
├── deployment: DeploymentConfig
│   ├── strategy: Literal[...]
│   ├── target: Literal[...]            # cloud integration: typed deploy targets
│   └── target sub-configs (azure_ml_endpoint, azure_app_service, aks)
├── cost_monitor: CostMonitorConfig
├── business_kpi: BusinessKPIConfig
├── audit: AuditConfig                  # cloud integration: storage sub-configs
├── registry: RegistryConfig            # cloud integration: backend dispatch
├── model_graph: ModelGraphConfig
├── llmops: LLMOpsConfig                # entire LLMOps layer toggle
├── agentops: AgentOpsConfig            # entire AgentOps layer toggle
├── domains: DomainsConfig              # tabular | timeseries | nlp | reco | graph
└── dashboard: DashboardConfig          # FastAPI UI + RBAC + auth
```

Two rules for editing `schema.py`:

1. Every new field must have a sensible default so old configs
   still load. Breaking old configs is a version-bump-worthy
   change.
2. Every `Literal[...]` that enumerates valid backend names must
   stay in sync with the corresponding registry dict. If you add
   `"my_method"` to the registry, add it to the `Literal` too.

**`loader.py` (~375 lines).** The loader is a pipeline:

```
YAML file
   │
   ▼
  _load_yaml          read file, parse with ruamel.yaml (line numbers preserved)
   │
   ▼
  _resolve_extends    recursively merge parent configs (with cycle detection)
   │
   ▼
  _substitute_env     replace ${VAR_NAME} and ${azkv:vault/secret}
   │                  (lazy Azure SDK import here)
   │
   ▼
  _validate_references  check baseline paths, schema files, holdout datasets
   │                    actually exist on disk
   │
   ▼
  SentinelConfig.model_validate(data)   Pydantic validation with source tracing
   │
   ▼
  return SentinelConfig
```

Each stage is a private function in `loader.py`. If you ever
hit a config bug, set a breakpoint at the end of each stage and
watch how the dict evolves. That's always how I debug loader
issues.

**`secrets.py`.** Wraps sensitive strings (webhook URLs, API keys,
basic-auth passwords) in Pydantic `SecretStr` so they never leak
into logs, error messages, or `sentinel config show` output. Use
`unwrap(secret)` when you need the plaintext inside the SDK.

**`signing.py`.** `sentinel config sign` computes a hash
over the fully-resolved config (after extends + env-var
substitution) and writes a `.sig` file. The dashboard's
`require_signed_config` enforcement verifies this on every load.
Survives `extends:` chains because it signs the resolved form.

**`keyvault.py`.** Lazy Azure SDK import. Caches one
`SecretClient` per vault URL. The regex `_AZKV_PATTERN` matches
`${azkv:vault-name/secret-name}`. Uses `DefaultAzureCredential`
for authentication.

**`aws_secrets.py`.** AWS Secrets Manager resolver. Tokens of the
form `${awssm:secret-name}` or `${awssm:secret-name/key}` are
resolved at the same config-load stage as env-var substitution.
Uses `boto3`'s default credential chain (env vars, `~/.aws/credentials`,
IAM role). Caches one Secrets Manager client per region. JSON-valued
secrets support a `/key` suffix to extract a single field.

**`gcp_secrets.py`.** GCP Secret Manager resolver. Tokens of the
form `${gcpsm:project/secret-name}` or
`${gcpsm:project/secret-name/version}` are resolved at config-load
time. Uses Google Application Default Credentials. A singleton
`SecretManagerServiceClient` is cached at module level with a
per-secret-version result cache. Defaults to `latest` version.

**`source.py`.** When a Pydantic validation error fires,
we annotate it with the YAML file + line number where the offending
field came from. This is the magic that lets
`sentinel config validate` show errors like:

```
alerts.channels[0].webhook_url: missing environment variable
  at sentinel.yaml line 47
  (inherited from base.yaml line 22)
```

**`defaults.py`.** Fallback values for modules that don't get
config overrides. `DEFAULT_PRICING` (per-token costs for every
LLM provider we know about) lives here. If a customer sets
`llmops.token_economics.pricing: {...}` in their config, the
override wins; otherwise defaults apply.

**`references.py`.** Verifies that every file path in the
config (baseline dataset, schema JSON, holdout dataset, audit
directory) actually resolves on disk before the SDK tries to
use it. Fails fast at config-load time rather than at first
prediction.

### 5.4 `sentinel/foundation/` — registry, audit, experiments, datasets

```
sentinel/foundation/
├── __init__.py
├── registry/
│   ├── __init__.py
│   ├── model_registry.py     # ModelRegistry facade
│   ├── versioning.py         # Semantic version helpers
│   └── backends/
│       ├── __init__.py       # resolve_backend() dispatch
│       ├── base.py           # BaseRegistryBackend ABC
│       ├── local.py          # filesystem backend (dev default)
│       ├── azure_ml.py       # Azure ML workspace backend
│       ├── mlflow.py         # MLflow-compatible backend
│       ├── sagemaker.py      # AWS SageMaker Model Registry
│       ├── vertex_ai.py      # GCP Vertex AI Model Registry
│       └── databricks.py     # Databricks Unity Catalog
├── audit/
│   ├── __init__.py
│   ├── trail.py              # ~528 lines — tamper-evident audit log with HMAC chain and file-level indexes
│   ├── lineage.py            # Data → training → model → prediction lineage (thread-safe, JSON persistence)
│   ├── compliance.py         # FCA / EU AI Act report generators with configurable risk classification
│   ├── shipper.py            # BaseAuditShipper + NullShipper + ThreadedShipper + AzureBlobShipper + S3Shipper
│   ├── integrity.py          # Hash chain verification utilities
│   └── keystore.py           # Pluggable HMAC key storage (env or file)
├── datasets/
│   ├── __init__.py
│   ├── hashing.py            # SHA-256 content hash computation
│   ├── lineage.py            # Dataset lineage tracking
│   └── registry.py           # DatasetRegistry + DatasetVersion — metadata-only tracking
└── experiments/
    ├── __init__.py
    └── tracker.py            # ExperimentTracker — named experiments, nested runs, metric history, OR filters
```

**`foundation/audit/trail.py` (~528 lines).** The immutable audit
log. Every event is a JSON line with: timestamp, event type,
payload, previous hash, current HMAC. Read `_build_event`,
`_compute_hmac`, `_last_hash`, and `verify_integrity` — those
four functions are the entire hash chain. The trail also maintains
file-level indexes for efficient querying by event type and
timestamp range.

The trail also has an optional `shipper` that uploads
completed day-files to Azure Blob or S3 on day rotation. The
local hash chain is untouched; shipping is a downstream
downstream-only concern.

**`foundation/registry/`.** Model versions, metadata, baselines,
feature schemas, training-data references, lineage pointers.
The `ModelRegistry` class is the facade; the real work happens
in a `BaseRegistryBackend` implementation chosen by YAML.

Backends (6 total):

- `local.py` — filesystem. One directory per model, one
  subdirectory per version, `meta.json` + `baseline.parquet` +
  `model.pkl`. Great for dev.
- `azure_ml.py` — wraps `azure.ai.ml.MLClient`. Lazy import so
  you can run without the `[azure]` extra. Full parity with
  `local`.
- `mlflow.py` — wraps `mlflow.tracking.MlflowClient`. Models are
  registered in MLflow's own model registry.
- `sagemaker.py` — wraps `boto3` SageMaker Model Registry APIs.
  Requires `[aws]` extra.
- `vertex_ai.py` — wraps `google.cloud.aiplatform`. Requires
  `[gcp]` extra (future).
- `databricks.py` — wraps Databricks Unity Catalog via REST API.
  Requires `databricks-sdk` (future).

**`foundation/audit/keystore.py`.** Pluggable HMAC key
storage. The `env` keystore reads the key from
`SENTINEL_AUDIT_HMAC_KEY`. The `file` keystore reads it from a
file with strict permissions.

**`foundation/audit/integrity.py`.** Hash chain verification
utilities extracted from `trail.py`. Provides standalone
verification logic that can be used by the `sentinel audit verify`
CLI command independently of the running trail instance.

**`foundation/audit/lineage.py`.** Data → training → model →
prediction lineage tracking. Thread-safe with `threading.Lock`.
Now supports **JSON persistence** via `save(path)` / `load(path)`
methods, so lineage graphs survive process restarts. Used by the
compliance report generators to produce full model lifecycle
documentation.

**`foundation/experiments/tracker.py`.** The `ExperimentTracker`
provides named experiments (groups of related runs), nested runs
via `parent_run_id`, metric time-series logging (`value` + `step`),
and search with filter syntax (`"metrics.f1 > 0.85 AND params.lr < 0.01"`).
Runs can be compared side-by-side and linked to dataset versions
(`name@version` refs) and model versions. The dashboard pages at
`/experiments` render metric history charts. The API is backward-compatible
with the old thin `tracker.py` that only linked MLflow experiment
IDs to production model versions.

**`foundation/datasets/registry.py`.** The `DatasetRegistry`
tracks dataset **metadata** without moving data. Each
`DatasetVersion` stores: `name`, `version`, `path`, `format`,
`split`, `num_rows`, `num_features`, `content_hash` (SHA-256),
`schema`, `tags`, and `lineage`. Datasets are referenced via
`name@version` syntax throughout the SDK. Key methods: `register`,
`get`, `list_versions`, `search`, `compare`, `link_to_experiment`,
`link_to_model`, and `verify` (re-hashes data on disk and compares
against the stored `content_hash`). The dashboard page at
`/datasets` lists all registered versions.

### 5.5 `sentinel/observability/` — drift, data quality, feature health, cost

```
sentinel/observability/
├── __init__.py
├── data_quality.py       # Schema validation + freshness + outlier detection
├── drift/
│   ├── __init__.py       # DRIFT_DETECTOR_REGISTRY
│   ├── base.py           # BaseDriftDetector ABC + DriftReport
│   ├── data_drift.py     # 367 lines — PSI, KS, JS, chi-squared, Wasserstein
│   ├── concept_drift.py  # DDM, EDDM, ADWIN, Page-Hinkley
│   └── model_drift.py    # Performance decay vs baseline
├── feature_health.py     # Per-feature drift + importance ranking
└── cost_monitor.py       # Latency + throughput + cost-per-prediction
```

**`drift/base.py`.** The `BaseDriftDetector` ABC has three methods:

- `fit(reference_data)` — compute statistics from the baseline.
- `detect(current_data) -> DriftReport` — compare current vs baseline.
- `reset()` — clear internal state (used by streaming detectors).

Every detector must return a `DriftReport` with the same fields:
`is_drifted`, `severity`, `test_statistic`, `p_value`,
`feature_scores`, `timestamp`. Uniformity here is what lets the
notification engine treat PSI, KS, DDM, and ADWIN results
interchangeably.

**`drift/data_drift.py`.** PSI, KS, JS divergence, chi-squared,
Wasserstein. These are the most battle-tested detectors and the
ones users most frequently configure. Unit tests verify
statistical properties with synthetic data (use `hypothesis`).

**`drift/concept_drift.py`.** Concept drift requires ground-truth
labels, which often arrive with lag. DDM and EDDM track error
rates with running variance; ADWIN maintains an adaptive sliding
window; Page-Hinkley is a CUSUM variant. Each is useful in a
different scenario — see the docstrings for guidance.

These detectors are now **streaming-enabled**: when `drift.concept`
is configured, `SentinelClient` builds the appropriate detector
at init time and feeds it per-prediction error signals inside
`log_prediction()` whenever `actual` values are provided. The
concept drift state is merged into the `DriftReport` returned by
`check_drift()` — if concept drift is detected it can promote
the overall report to `is_drifted=True` even when data drift
alone is below threshold.

**`drift/model_drift.py`.** Tracks production metrics (accuracy,
F1, AUC, latency) against the baseline stored in the model
registry. Fires alerts when the delta exceeds config thresholds.

**`data_quality.py`.** Three sub-checks: JSON Schema validation,
freshness (how stale is the newest sample), and outlier detection
(isolation forest, z-score, IQR). Returns a `QualityReport` with
issue severities.

**`feature_health.py`.** For each feature: per-feature drift score
+ feature importance rank (from SHAP / permutation / built-in).
Surfaces the most-important drifting features so users don't
drown in per-feature alerts.

**`cost_monitor.py`.** Tracks latency, throughput, cost per
prediction. Alerts when latency p99 exceeds thresholds or
cost-per-1k-predictions climbs.

### 5.6 `sentinel/intelligence/` — cross-signal layer

```
sentinel/intelligence/
├── __init__.py
├── explainability.py   # SHAP/LIME wrapper + per-prediction explanations
├── kpi_linker.py       # Model metric → business KPI mapping
└── model_graph.py      # Multi-model dependency DAG + cascade alerts
```

**`model_graph.py`.** A directed graph of model dependencies.
When the upstream `feature_engineering_pipeline` drifts, cascade
alerts propagate to every downstream model that depends on it.
Now thread-safe (uses `threading.Lock`) and performs **eager cycle
detection** on graph construction — cycles in the dependency
declaration are rejected at config-load time with a clear error.

**`kpi_linker.py`.** Connects model metrics (precision, recall)
to business KPIs (fraud catch rate, false positive rate). Reads
business KPI values from a configured warehouse query and
computes the correlation with model-reported metrics. This is
what lets compliance officers say "this drift alert translates
to N false positives per week, which costs £X". Now includes an
**auto-refresh scheduler** that periodically re-fetches business
KPI values on a configurable interval.

**`explainability.py`.** Wrapper around SHAP and permutation
importance. Produces explanations at three levels: per-prediction,
global, and per-cohort. Attaches to audit log entries. Used by
the compliance report generator for FCA Consumer Duty per-decision
rationale. Also exposed as `ExplainabilityEngine` on
`SentinelClient` when the `[explain]` extra is installed.

### `sentinel/observability/cohort_analyzer.py`

Per-subpopulation performance tracking with fairness disparity detection.
Thread-safe (uses `threading.Lock`), config-driven via `CohortAnalysisConfig`.

- `log_prediction(features, prediction, actual, cohort_id)` — buffer predictions per cohort
- `get_report(cohort_id)` → `CohortReport` with accuracy, count, mean prediction
- `compare_cohorts()` → `CohortComparisonReport` with disparity alerts
- `clear()` — reset all accumulated cohort data

Auto-derives `cohort_id` from `features[cohort_column]` if no explicit ID is passed.
Uses bounded `deque(maxlen=buffer_size)` per cohort for memory safety.

### 5.7 `sentinel/action/` — notifications, retrain, deployment

```
sentinel/action/
├── __init__.py
├── notifications/
│   ├── __init__.py
│   ├── engine.py            # NotificationEngine — router with cooldown + escalation
│   ├── escalation.py        # EscalationTimer — background daemon for time-delayed escalations
│   ├── policies.py          # Severity rules, digest vs realtime, remaining escalation steps
│   └── channels/
│       ├── __init__.py      # CHANNEL_REGISTRY
│       ├── base.py          # BaseChannel ABC
│       ├── slack.py
│       ├── teams.py
│       ├── pagerduty.py
│       ├── email.py
│       └── webhook.py       # Generic webhook channel (custom endpoints)
├── retrain/
│   ├── __init__.py
│   ├── orchestrator.py      # Drift → retrain → validate → promote pipeline
│   ├── triggers.py          # Configurable trigger conditions
│   └── approval.py          # Human-in-the-loop approval gates
└── deployment/
    ├── __init__.py          # STRATEGY_REGISTRY + TARGET_REGISTRY
    ├── manager.py           # DeploymentManager — lifecycle coordinator
    ├── promotion.py         # Champion-challenger comparison
    ├── strategies/
    │   ├── __init__.py
    │   ├── base.py          # BaseDeploymentStrategy ABC
    │   ├── shadow.py        # Log, don't serve
    │   ├── canary.py        # % traffic ramp with auto-rollback
    │   ├── blue_green.py    # Atomic swap
    │   └── direct.py        # Replace in-place (non-critical models)
    └── targets/
        ├── __init__.py      # resolve_target()
        ├── base.py          # BaseDeploymentTarget ABC
        ├── local.py         # no-op, logs only
        ├── azure_ml_endpoint.py
        ├── azure_app_service.py
        ├── aks.py
        ├── sagemaker.py     # AWS SageMaker endpoint target
        └── vertex_ai.py     # GCP Vertex AI endpoint target
```

**`notifications/engine.py`.** The `NotificationEngine` is a
router. It receives an `Alert`, consults `policies.py` to decide
severity / cooldown / digest, and fans the alert out to the
appropriate channels. Cooldown is enforced per `(model, check,
severity)` tuple to prevent alert storms. After dispatching an
alert, the engine schedules any remaining escalation steps with
the `EscalationTimer` and exposes a `close()` method for clean
shutdown (called by `SentinelClient.close()`).

**`notifications/escalation.py`.** The `EscalationTimer` is a
background daemon thread that fires time-delayed escalation
callbacks. It uses a heap-based priority queue of
`(fire_at, fingerprint, alert, step)` tuples. A single worker
thread sleeps until the next fire time, pops the entry, and calls
the engine's `_on_escalation` callback. Supports `cancel(fp)` to
remove pending escalations for acknowledged alerts. Follows the
same daemon-thread + Event pattern as `DriftScheduler`.

**`notifications/policies.py`.** Severity rules, cooldown,
fingerprinting, digest queue, and the `remaining_escalation_steps()`
method that returns escalation steps whose `after` duration has
not yet elapsed relative to the alert's first-seen time.

**`notifications/channels/`.** One file per channel. Each
implements `BaseChannel.send(alert) -> DeliveryResult`. All
channels use `httpx` (already a core dep) for HTTP; no per-channel
SDKs at the core level.

**`retrain/orchestrator.py`.** The state machine: drift detected
→ trigger pipeline → wait for completion → validate on holdout →
compare against champion → request approval → promote → optionally
deploy. Pipelines are external (Azure ML pipeline, custom script)
— Sentinel just coordinates. The orchestrator now accepts an
optional `deployment_manager` and, when `deploy_on_promote: true`
is set in config, automatically calls
`DeploymentManager.start()` after a model is promoted (both in
`run()` for auto-approved models and in `approve()` for
manually-approved ones).

**`retrain/approval.py`.** The human-in-the-loop gate. When a
model needs approval, an alert fires through the notification
engine. The approver acts via the dashboard or a CLI command.
Approvals are persisted in the audit trail.

**`deployment/manager.py`.** Orchestrates a deploy:
instantiates the configured strategy (`canary`, `blue_green`,
`shadow`, `direct`) and target (`local`, `azure_ml_endpoint`,
`azure_app_service`, `aks`, `sagemaker`, `vertex_ai`) and drives
them through the lifecycle.

**`deployment/strategies/`.** The *how* of deployment (what
fraction of traffic goes where, when to roll back). All strategies
accept a `BaseDeploymentTarget` and delegate the *what* to it.

**`deployment/targets/`.** The *what* of deployment (how
to actually change traffic weights on Azure ML Online Endpoints,
how to swap Azure App Service slots, how to scale AKS replicas,
how to update SageMaker endpoint variants, how to update Vertex AI
traffic splits).
This separation means `CanaryStrategy` + `AzureMLEndpointTarget`
is a canary rollout on Azure ML; `CanaryStrategy` + `AKSDeploymentTarget`
is a canary rollout on AKS; no code change, just config.

Strategy/target compatibility is enforced at config-validation
time. `canary` + `azure_app_service` raises a clear error because
App Service slot traffic routing is brittle.

### 5.8 `sentinel/llmops/` — prompts, guardrails, quality, tokens

```
sentinel/llmops/
├── __init__.py
├── client.py             # LLMOpsClient — sub-facade for LLM ops
├── prompt_manager.py     # Prompt versioning + A/B routing
├── prompt_drift.py       # Composite prompt drift detector
├── token_economics.py    # Per-query cost tracking + budget enforcement
├── guardrails/
│   ├── __init__.py       # GUARDRAIL_REGISTRY
│   ├── base.py           # BaseGuardrail ABC + GuardrailResult
│   ├── engine.py         # GuardrailPipeline — short-circuit runner
│   ├── pii.py            # PII detection/redaction (Presidio)
│   ├── jailbreak.py      # Jailbreak/prompt-injection detection
│   ├── toxicity.py       # Toxic/harmful content detection
│   ├── groundedness.py   # NLI / chunk-overlap / LLM-judge
│   ├── topic_fence.py    # Off-topic / out-of-scope detection
│   ├── format_compliance.py  # Output schema validation
│   ├── regulatory.py     # Prohibited-phrases / regulatory language scan
│   ├── token_budget.py   # Input token budget enforcement
│   ├── custom.py         # Custom DSL guardrails — declarative rule engine
│   └── plugin.py         # Plugin guardrails — dynamic class loading
└── quality/
    ├── __init__.py
    ├── base.py           # BaseEvaluator ABC
    ├── evaluator.py      # LLM-as-judge + heuristic scoring
    ├── judge_factory.py  # Judge model configuration and instantiation
    ├── semantic_drift.py # Embedding-distribution drift
    └── retrieval_quality.py  # RAG metrics — relevance, utilisation, etc.
```

**`llmops/client.py`.** `SentinelClient.llmops` returns an
`LLMOpsClient`. It wraps the prompt manager, guardrail pipeline,
evaluator, semantic drift detector, and token tracker. Users
call `client.llmops.log_call(...)` or `client.llmops.guardrails.check_input(...)`.

**`guardrails/engine.py`.** The `GuardrailPipeline` runs a chain
of `BaseGuardrail`s against an input or output. Each guardrail
returns `pass`, `warn`, or `block`. The pipeline short-circuits
on the first `block`. Latency per guardrail is tracked and
attached to the result so slow guardrails surface in the cost
monitor.

**`guardrails/`.** PII detection (Presidio under an optional
extra), jailbreak detection (embedding similarity against known
attack prompts), topic fencing, token budget checks, toxicity,
groundedness (NLI or chunk-overlap for RAG), format compliance,
regulatory language scans.

**`guardrails/custom.py`.** The custom DSL guardrail engine.
Interprets `type: custom` config entries with declarative rules —
11 rule types (`regex_match`, `regex_absent`, `keyword_present`,
`keyword_absent`, `min_length`, `max_length`, `json_schema`,
`sentiment`, `language`, `word_count`, `not_empty`). `combine: "any"`
means strict (block on first failure); `combine: "all"` means
lenient (block only if all rules fail). No Python code needed.

**`guardrails/plugin.py`.** The plugin guardrail loader. Interprets
`type: plugin` config entries by dynamically importing `module`
and instantiating `class_name` with `config`. The loaded class
must implement `check(content, context) -> GuardrailResult`.
Enables users to bring their own guardrail logic without forking
the SDK.

**`quality/evaluator.py`.** LLM-as-judge scoring. A cheaper
judge model (default `gpt-4o-mini`) scores responses on
configurable rubrics. Heuristic fallback for high-volume, low-cost
cases.

**`quality/semantic_drift.py`.** Embeds LLM outputs and tracks
the centroid of the embedding distribution over time. Cosine
distance between this week's centroid and last week's is the
drift signal. Catches silent model updates from providers.

**`quality/retrieval_quality.py`.** RAG-specific: retrieval
relevance, chunk utilisation, answer coverage, faithfulness. All
computed from the `context_chunks` passed to `log_llm_call`.

**`token_economics.py`.** Tracks token usage + cost per query,
per model, per prompt version, per user. Enforces daily/per-query
budgets. `estimate_cost(model, input_tokens, output_tokens)` is
a pure lookup against `DEFAULT_PRICING` (see `config/defaults.py`).
Supports custom pricing for Azure OpenAI (`azure/gpt-4o`), vLLM
self-hosted, etc.

**`prompt_manager.py`.** The prompt registry. Stores versioned
prompt bundles (system prompt + template + few-shot examples).
Resolves which version to serve given A/B test config. Logs
performance metadata per version.

**`prompt_drift.py`.** Composite detector: quality score decline
+ guardrail violation rate increase + token-usage increase +
semantic drift. When all four fire, the prompt is probably
regressing.

### 5.9 `sentinel/agentops/` — trace, tool audit, safety, evaluation

```
sentinel/agentops/
├── __init__.py
├── client.py              # AgentOpsClient — sub-facade
├── agent_registry.py      # Agent versioning + capability manifests
├── integrations/
│   ├── __init__.py        # Re-exports LangGraphMiddleware, MonitoredGraph
│   └── langgraph.py       # LangGraph auto-instrumentation middleware
├── trace/
│   ├── __init__.py
│   ├── tracer.py          # Span-based tracer (AgentTracer)
│   ├── visualiser.py      # Timeline + decision tree rendering
│   └── exporters/
│       ├── __init__.py    # EXPORTER_REGISTRY
│       ├── base.py        # BaseExporter ABC
│       ├── jsonl.py       # Local JSON-lines export
│       └── otlp.py        # OpenTelemetry export
├── tool_audit/
│   ├── __init__.py
│   ├── monitor.py         # Tool call tracking
│   ├── permissions.py     # Allowlist/blocklist enforcement
│   └── replay.py          # Record + replay for debugging
├── safety/
│   ├── __init__.py
│   ├── loop_detector.py   # Infinite loop detection
│   ├── budget_guard.py    # Token/cost/time budget enforcement
│   ├── escalation.py      # Agent → human handoff triggers
│   └── sandbox.py         # Destructive-op sandboxing
├── multi_agent/
│   ├── __init__.py
│   ├── orchestration.py   # Multi-agent workflow monitoring
│   ├── delegation.py      # Delegation chain tracking
│   └── consensus.py       # Agreement/conflict detection
└── eval/
    ├── __init__.py
    ├── task_completion.py  # End-to-end task success rate
    ├── trajectory.py       # Optimal vs actual step comparison
    └── golden_datasets.py  # Golden test suite management
```

**`integrations/langgraph.py`.** The `LangGraphMiddleware`
provides zero-code instrumentation for LangGraph compiled graphs.
`middleware.wrap(graph)` returns a `MonitoredGraph` proxy that
intercepts `invoke()` and `ainvoke()` calls. The proxy uses the
graph's `stream()` / `astream()` methods to intercept per-node
events, recording each as a traced span with output key
attributes. The middleware does **not** import `langgraph` at
module level — it works via duck-typing, wrapping any object with
compatible `stream()` / `astream()` methods.

Usage: `from sentinel.agentops.integrations import LangGraphMiddleware`

**`trace/tracer.py`.** The `AgentTracer` is the entry point. It
creates a `Span` per reasoning step and nests spans to build a
tree. Spans have OTel-compatible attributes (name, start time,
duration, status, parent span id). Export via `OTLPExporter` or
`ArizePhoenixExporter`.

**`tool_audit/monitor.py`.** Every tool call passes through
here: success/failure, latency, input schema validation, rate
limits. Results feed into the audit trail.

**`tool_audit/permissions.py`.** Allowlist + blocklist per agent.
A claims agent should not have access to the payment execution
tool. Enforced before the tool call happens — blocked calls
raise `GuardrailError`.

**`safety/loop_detector.py`.** Three detection strategies: max
iterations, max repeated tool calls with the same input, and
"thrash" detection (alternating between two states). Fires
`LoopDetectedError` when triggered.

**`safety/budget_guard.py`.** Per-run budget: max tokens, max
cost, max wall-clock time, max tool calls. On exceeded: graceful
stop, escalate, or hard kill (configurable).

**`safety/escalation.py`.** Confidence-based, failure-count-based,
sensitive-data-based, and regulatory-context-based triggers. All
escalations go through the notification engine to the configured
approver channel.

**`agent_registry.py`.** Agent versioning + capability manifests.
Orchestrator agents use capability queries to discover specialist
agents (A2A discovery).

**`multi_agent/`.** Tracks delegation chains across agents.
Consensus detection for "three underwriting agents disagree on
this claim". Bottleneck identification for slow multi-agent
pipelines.

**`eval/`.** Task completion rate tracking, trajectory comparison
against golden datasets, golden suite CI/CD runner. The trajectory
evaluator penalises unnecessary steps and rewards efficient paths.

### 5.10 `sentinel/domains/` — tabular, time series, NLP, reco, graph

```
sentinel/domains/
├── __init__.py
├── base.py             # BaseDomainAdapter ABC
├── tabular/adapter.py  # Default (PSI, KS, standard metrics)
├── timeseries/
│   ├── adapter.py
│   ├── drift.py        # Calendar tests, ACF monitoring
│   ├── quality.py      # MASE, coverage, directional accuracy
│   └── decomposition.py  # STL decomposition monitoring
├── nlp/
│   ├── adapter.py
│   ├── drift.py        # Vocabulary drift, embedding space shift
│   ├── quality.py      # NER F1, classification metrics
│   └── text_stats.py
├── recommendation/
│   ├── adapter.py
│   ├── drift.py        # Item/user distribution shift
│   ├── quality.py      # NDCG, MAP, coverage, diversity, novelty
│   └── bias.py         # Popularity bias, group fairness
└── graph/
    ├── adapter.py
    ├── drift.py        # Topology drift — degree, density, clustering
    ├── quality.py      # Link prediction AUC, Hits@K, MRR
    └── structure.py    # Graph density, connected components
```

**`domains/base.py`.** `BaseDomainAdapter` exposes three methods:
`get_drift_detectors()`, `get_quality_metrics()`,
`get_schema_validator()`. An adapter is a **factory** for the
rest of the SDK — when `SentinelClient` builds its drift engine
and quality checker, it asks the adapter for the domain-appropriate
implementations.

**`tabular/adapter.py`.** The default. Returns the existing
`PSIDriftDetector`, `KSDriftDetector`, standard classification
metrics, JSON-Schema validator. Used when `domain` is omitted —
fully backward compatible.

**`timeseries/adapter.py`.** Returns `CalendarAwareDriftDetector`,
`ACFMonitor`, `STLDecompositionMonitor`, and forecast metrics
(MASE, coverage, directional accuracy). Lazy imports `statsmodels`
under `sentinel[timeseries]`.

**`nlp/adapter.py`.** Returns vocabulary drift detectors,
embedding space monitors, NER/classification F1 metrics, text
stats monitors.

**`recommendation/adapter.py`.** Returns item-distribution drift
detectors, beyond-accuracy metrics (NDCG, coverage, diversity,
novelty), popularity-bias detectors, group fairness checks.

**`graph/adapter.py`.** Returns topology drift detectors (degree
distribution, density, clustering coefficient, connected
components), node/edge feature drift, graph-quality metrics.
Lazy imports `networkx` under `sentinel[graph]`.

**Extension rule.** New domain adapters must not import their
heavy deps at module load — use function-local imports. This
keeps `import sentinel` fast and keeps the `[tabular]` path
dependency-free.

### 5.11 `sentinel/integrations/` — cloud-specific code lives here

```
sentinel/integrations/
├── __init__.py
├── aws/
│   ├── __init__.py
│   └── s3_audit.py       # S3AuditStorage + S3Shipper
├── azure/
│   ├── __init__.py
│   ├── blob_audit.py     # AzureBlobAuditStorage + AzureBlobShipper
│   └── pipeline_runner.py  # Azure ML pipeline trigger for retrain orchestrator
└── gcp/
    ├── __init__.py        # Re-exports GcsAuditStorage, GcsShipper
    └── gcs_audit.py       # GcsAuditStorage + GcsShipper
```

**This is the one hard rule.** Core modules must **never** import
`azure.*`, `boto3`, or `google.*` directly. Anything cloud-specific
lives under `sentinel/integrations/{azure,aws,gcp}/` and is
**lazy-imported** at the point of use.

The cloud deploy targets (`sentinel/action/deployment/targets/azure_*.py`,
`sagemaker.py`, `vertex_ai.py`) break the surface of this rule
slightly — they are under `action/`, not `integrations/` — because
they implement a cross-cutting ABC (`BaseDeploymentTarget`) that
`DeploymentManager` needs to dispatch against. They still follow
the lazy-import rule for the underlying SDK.

**`azure/blob_audit.py`.** `AzureBlobAuditStorage` is the manual
helper (explicit upload). `AzureBlobShipper` is the automatic
adapter that the audit trail calls on day rotation.

**`aws/s3_audit.py`.** Same pattern — `S3AuditStorage` (manual) +
`S3Shipper` (automatic).

**`gcp/gcs_audit.py`.** Same pattern — `GcsAuditStorage` (manual
upload to GCS buckets) + `GcsShipper` (automatic, extends
`ThreadedShipper`). Lazy imports `google.cloud.storage`. Requires
the `[gcp]` extra (`pip install "sentinel-mlops[gcp]"`). Uses
Application Default Credentials via `google.cloud.storage.Client()`.
Follows the same immutable-bucket contract as the Azure Blob and
S3 shippers — `enforce_retention()` returns 0 (never deletes
from remote).

### 5.12 `sentinel/dashboard/` — local FastAPI UI

```
sentinel/dashboard/
├── __init__.py
├── server.py             # create_dashboard_app() + SentinelDashboardRouter
├── state.py              # In-memory state cache for the dashboard
├── deps.py               # FastAPI Depends() helpers
├── security/
│   ├── __init__.py
│   ├── auth.py           # Basic + Bearer JWT auth orchestration
│   ├── bearer.py         # Bearer JWT validation with JWKS-URL + kid cache
│   ├── csrf.py           # Double-submit-cookie CSRF
│   ├── headers.py        # HSTS + CSP + frame guards
│   ├── principal.py      # Authenticated user principal model
│   ├── rate_limit.py     # Token-bucket rate limiter (per-IP + per-user)
│   ├── rbac.py           # Role-based access control engine
│   └── route_perms.py    # Per-route permission definitions
├── routes/
│   ├── __init__.py
│   ├── pages.py          # Page route registration
│   ├── api.py            # JSON + CSV API endpoints (export, rollback, metrics)
│   ├── overview.py       # / — system status
│   ├── drift.py
│   ├── features.py
│   ├── registry.py
│   ├── audit.py
│   ├── llmops.py
│   ├── agentops.py
│   ├── deployments.py
│   ├── compliance.py
│   ├── cohorts.py        # /cohorts — cohort comparison + /cohorts/{id} detail
│   ├── explanations.py   # /explanations — global feature importance
│   ├── datasets.py       # /datasets — dataset registry browser
│   └── experiments.py    # /experiments — experiment runs + metric history charts
├── views/
│   └── (Jinja2 template helpers)
├── templates/            # Jinja2 HTML templates (HTMX + Plotly)
│   ├── base.html
│   ├── overview.html
│   └── ... (one per route)
└── static/               # CSS, JS, images
```

**`server.py`.** Two entry points:

1. `create_dashboard_app(client)` — returns a standalone FastAPI
   app. Used by `sentinel dashboard --config sentinel.yaml`.
2. `SentinelDashboardRouter(client).attach(app)` — mounts the
   dashboard routes onto an existing FastAPI app. Used when a
   customer wants to embed the dashboard in their own service.

Both are lazy-loaded via `sentinel/__init__.py:__getattr__` so
that `import sentinel` works without the `[dashboard]` extra.

**`security/auth.py`.** Three layers stacked:

1. **Basic auth** — simple username/password for dev.
2. **Bearer JWT** — validated against a JWKS URL with a
   `kid`-keyed cache. No full OIDC client on purpose — bearer-only
   is sufficient for the dashboard's scope.
3. **RBAC** — viewer / operator / admin roles with transitive
   inheritance and namespaced permissions
   (`drift:read`, `deploy:promote`, etc.). Every route has a
   `Depends(require_permission(...))` guard.

See my memory note: **never use `request: Request` in FastAPI
dependency functions when the module has
`from __future__ import annotations`** — FastAPI's forward-ref
resolver cannot handle it and silently treats the parameter as a
query string, producing a 422. Use `Header(default=None)` instead.

**`security/csrf.py`.** Double-submit-cookie pattern. The
base template injects a CSRF token into every form and HTMX
request header. Guards all state-mutating routes.

**`security/rate_limit.py`.** In-memory token-bucket rate
limiter (per-IP for unauthenticated routes, per-username for
authenticated ones). Single-worker by design — scaling to
multi-worker is a operational observability concern.

**`security/headers.py`.** HSTS (on HTTPS), CSP, frame
guards, X-Content-Type-Options. Applied via a FastAPI middleware.

**`routes/`.** One file per top-level dashboard page. Each route
reads from `SentinelClient` via `deps.py` helpers and renders a
Jinja2 template with HTMX + Plotly widgets. Nothing fancy — the
goal is zero-build-step, zero-npm, local-first.

**`state.py`.** A small in-memory cache to avoid recomputing
drift / feature health on every page load. Invalidated on
`log_prediction` boundaries.

### 5.13 `sentinel/cli/` — Click-based CLI

```
sentinel/cli/
├── __init__.py
└── main.py            # All subcommands
```

**`cli/main.py`.** A Click `Group` of subcommands:

- `sentinel init` — generate a `sentinel.yaml` template.
- `sentinel check` — run drift detection on a dataset.
- `sentinel status` — show current model status.
- `sentinel deploy --version X --strategy canary --traffic 5 [--dry-run]`.
- `sentinel config validate [--strict]`.
- `sentinel config show [--unmask]`.
- `sentinel config sign` / `verify-signature`.
- `sentinel registry list` / `sentinel registry show`.
- `sentinel audit query --event-type X --model-name Y`.
- `sentinel audit verify` — verify the HMAC hash chain.
- `sentinel audit chain-info` — show chain head + length.
- `sentinel cloud test [--only audit|registry|deploy|keyvault]`
  — smoke-test cloud backends without a live run.
- `sentinel dashboard --config sentinel.yaml` — start the local UI.
- `sentinel completion [bash|zsh|fish]` — generate shell completion script.

Each subcommand is a thin wrapper: parse args, construct
`SentinelClient`, delegate to an SDK method, print a table.
Business logic lives in the SDK modules, not the CLI.

---

## 6. YAML config — philosophy and loading pipeline

This section exists because we've heard the feedback: new developers
look at a `sentinel.yaml` and don't know what they're seeing. Here's
what every part of a config file is, why it's there, and how it gets
turned into live objects.

### 6.1 Why YAML and not Python?

**It's a deliberate architectural choice, not a convenience.** Every
Sentinel behaviour is defined in YAML because:

- **Auditability.** `git blame sentinel.yaml` tells you who changed
  the fraud model's drift threshold and when. Try doing that with
  a `DriftConfig(method="psi", threshold=0.2)` buried in Python.
- **Non-developer review.** Risk officers and compliance teams need
  to approve alert thresholds, retrain policies, and deployment
  strategies. They can read YAML. They can't read Python.
- **Environment separation.** The same model ships three YAMLs —
  `sentinel-dev.yaml`, `sentinel-staging.yaml`, `sentinel-prod.yaml` —
  that differ only in secret references and thresholds. With Python,
  you'd be maintaining three scripts.
- **GitOps deployment.** Config changes flow through the same
  pull-request + review + CI pipeline as code. No separate
  "operational console" to manage.
- **Signing and tamper evidence.** A YAML file has a stable
  hash. `sentinel config sign` produces a signature over the
  fully-resolved config. Python objects are harder to sign.

The cost is that schema evolution is a bit noisier — every new
feature needs a schema field, a validator, docs, and a migration
story. Worth it.

### 6.2 Top-level structure of a `sentinel.yaml`

```yaml
version: "1.0"                  # schema version — required, Pydantic validates
extends: base.yaml              # optional — inherit from a parent config
model: {...}                    # which model is being monitored (required)

# --- Observability (MLOps core) ---
data_quality: {...}             # schema validation, freshness, outliers
drift: {...}                    # data drift, concept drift, model drift
feature_health: {...}           # per-feature monitoring + importance
cost_monitor: {...}             # latency, throughput, cost-per-prediction

# --- Action layer ---
alerts: {...}                   # channels + cooldown + escalation policies
retraining: {...}               # trigger + pipeline + approval + validation
deployment: {...}               # strategy + target

# --- Foundation ---
audit: {...}                    # storage backend + retention + compliance
registry: {...}                 # backend: local | azure_ml | mlflow | sagemaker | vertex_ai | databricks

# --- Intelligence ---
model_graph: {...}              # upstream/downstream model dependencies
business_kpi: {...}             # model metric ↔ business KPI mappings

# --- Domain adapter ---
domains: {...}                  # tabular | timeseries | nlp | reco | graph

# --- Optional subsystems ---
llmops: {...}                   # only if llmops.enabled: true
agentops: {...}                 # only if agentops.enabled: true
dashboard: {...}                # only if you want the FastAPI UI
```

**Every top-level key maps to a submodule under `sentinel/`:**

| YAML key | Pydantic model | Runtime owner |
|---|---|---|
| `model` | `ModelConfig` | `SentinelClient` |
| `data_quality` | `DataQualityConfig` | `observability/data_quality.py` |
| `drift` | `DriftConfig` | `observability/drift/` |
| `feature_health` | `FeatureHealthConfig` | `observability/feature_health.py` |
| `cost_monitor` | `CostMonitorConfig` | `observability/cost_monitor.py` |
| `alerts` | `AlertConfig` | `action/notifications/` |
| `retraining` | `RetrainConfig` | `action/retrain/` |
| `deployment` | `DeploymentConfig` | `action/deployment/` |
| `audit` | `AuditConfig` | `foundation/audit/` |
| `registry` | `RegistryConfig` | `foundation/registry/` |
| `model_graph` | `ModelGraphConfig` | `intelligence/model_graph.py` |
| `business_kpi` | `BusinessKPIConfig` | `intelligence/kpi_linker.py` |
| `domains` | `DomainsConfig` | `domains/<name>/adapter.py` |
| `llmops` | `LLMOpsConfig` | `llmops/` |
| `agentops` | `AgentOpsConfig` | `agentops/` |
| `dashboard` | `DashboardConfig` | `dashboard/` |

### 6.3 The loading pipeline — what happens when you call `from_config`

When you call `SentinelClient.from_config("sentinel.yaml")`, this
sequence runs (all stages live in `sentinel/config/loader.py`):

```
┌──────────────────────────────────────────────────────────────────┐
│  Stage 1 — Read file                                             │
│  Open YAML with ruamel.yaml. Line numbers are preserved for      │
│  error reporting. Invalid YAML → ConfigError with the exact      │
│  line + column.                                                  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 2 — Resolve `extends:` chain                              │
│  If the config has `extends: base.yaml`, recursively load the    │
│  parent and deep-merge (child keys override parent). config hardening added  │
│  cycle detection so `a.yaml → b.yaml → a.yaml` fails fast with   │
│  a clear error.                                                  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 3 — Substitute secrets and environment variables          │
│  Walk every string value in the dict. Replace `${VAR_NAME}` with │
│  os.environ['VAR_NAME']. Replace `${azkv:vault/secret}` with     │
│  Azure Key Vault, `${awssm:secret-name}` with AWS Secrets        │
│  Manager, `${gcpsm:project/secret}` with GCP Secret Manager      │
│  (lazy SDK imports + cached clients). Strict mode raises on      │
│  missing vars; lenient mode preserves the literal.               │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 4 — Validate file references                       │
│  Check that baseline dataset paths, schema JSON files, holdout   │
│  dataset paths, and audit directories actually resolve on disk.  │
│  Fail here rather than at first prediction.                      │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 5 — Pydantic validation                                   │
│  SentinelConfig.model_validate(data). Every field is type-       │
│  checked, every Literal enforced, every cross-field validator    │
│  fires. Sensitive fields (webhook_url, routing_key, basic-auth   │
│  password) are wrapped in SecretStr. Errors are annotated with   │
│  source file + line via sentinel/config/source.py.               │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 6 — Signature verification (security hardening, opt-in)                 │
│  If `dashboard.server.require_signed_config: true`, verify the   │
│  config signature against the .sig file. Signature is computed   │
│  over the resolved dict, so it survives extends + env-var        │
│  substitution.                                                   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  Stage 7 — SentinelClient.__init__                               │
│  The factory: for each enabled subsystem, call the corresponding │
│  _build_* helper to construct a live object. Wire them up. Audit │
│  trail is constructed first (every other subsystem needs it).    │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                       live SentinelClient
```

**Debugging tip.** If config loading fails, run
`sentinel config validate --strict --verbose` — it runs all seven
stages and prints exactly which one failed with full context.

### 6.4 Secrets, `SecretStr`, and `<REDACTED>`

Every sensitive field in the schema is declared as
`pydantic.SecretStr`. This means:

- The plaintext is never serialised into logs, error messages, or
  dashboard output. `repr(config)` shows `SecretStr('**********')`.
- `sentinel config show` replaces every secret with `<REDACTED>`
  by default. Use `--unmask` to see plaintext (with a warning).
- To use a secret inside the SDK, call
  `unwrap(config.some.secret_field)` from
  `sentinel/config/secrets.py`. This is the only sanctioned way.
- Secrets can come from five sources: literal YAML string, env var
  substitution (`${VAR}`), Azure Key Vault (`${azkv:vault/secret}`),
  AWS Secrets Manager (`${awssm:secret-name}`), or GCP Secret
  Manager (`${gcpsm:project/secret}`). All five flow through the
  same substitution stage before validation, so Pydantic's
  `SecretStr` wrapping always happens at the end regardless of
  source.

Which fields are `SecretStr`?

- `alerts.channels[*].webhook_url` (Slack + Teams)
- `alerts.channels[*].routing_key` (PagerDuty)
- `alerts.channels[*].password` (email SMTP)
- `dashboard.server.basic_auth_password`
- `dashboard.server.jwt_secret` (for signed session cookies)
- `foundation.audit.keystore.key` (HMAC key for the hash chain)

Don't add any new field that holds a credential without wrapping
it in `SecretStr`. Code review will catch this, but it's easier
to get right the first time.

---

## 7. Config-to-code mapping

This is a lookup table you will refer to constantly in your first
few weeks. Given a YAML field, which file owns it? Given a file,
which config section drives it?

### 7.1 Dissecting `configs/examples/minimal.yaml`

The smallest possible config. Every line is annotated with which
file reads it and what it does.

```yaml
version: "1.0"                    # [1] SentinelConfig.version — loader.py rejects other versions
                                  #
model:                            # [2] ModelConfig — schema.py
  name: hello_world_model         # [3] Used as the audit trail key + registry key + dashboard title
  type: classification            # [4] Literal[classification|regression|ranking] — picks default metric set
  domain: tabular                 # [5] Picks the domain adapter — domains/tabular/adapter.py
                                  #
drift:                            # [6] DriftConfig — schema.py; wired in client.py _build_drift_engine
  data:                           # [7] DataDriftConfig
    method: psi                   # [8] Literal — looked up in DRIFT_DETECTOR_REGISTRY → PSIDriftDetector
    threshold: 0.2                # [9] PSI > 0.2 triggers a drift alert
    window: 7d                    # [10] Sliding window used to compute the current distribution
                                  #
alerts:                           # [11] AlertConfig — action/notifications/engine.py
  channels:                       # [12] list[ChannelConfig] — discriminated union on "type"
    - type: slack                 # [13] type: slack → SlackChannel from channels/slack.py
      webhook_url: ${SLACK_WEBHOOK_URL}  # [14] env substitution at loader stage 3 → SecretStr
      channel: "#ml-alerts"       # [15] Slack channel name — plain string, not secret
  policies:                       # [16] PolicyConfig — notifications/policies.py
    cooldown: 1h                  # [17] Deduplicate identical alerts within this window
                                  #
audit:                            # [18] AuditConfig — foundation/audit/trail.py
  storage: local                  # [19] Literal[local|azure_blob|s3|gcs] — picks the shipper
  path: ./audit/                  # [20] Local filesystem path — validated by config/references.py
```

Lines [6]–[10]: a single PSI drift detector on the default
tabular feature set. No per-feature overrides, no concept drift,
no model drift. The minimum viable MLOps loop.

Lines [11]–[17]: one Slack channel, no escalation, no digest,
simple 1-hour cooldown. Enough to stop alert storms in dev.

Lines [18]–[20]: local-disk audit log in `./audit/`, default
hash-chain enabled. Every event is tamper-evident out of
the box.

### 7.2 Lookup table — field → file

| YAML path | Read by |
|---|---|
| `version` | `config/loader.py:_check_schema_version` |
| `extends` | `config/loader.py:_resolve_extends` |
| `model.name` | `core/client.py:SentinelClient.__init__` |
| `model.type` | `core/client.py`, `domains/*/adapter.py` |
| `model.domain` | `core/client.py:_build_domain_adapter` |
| `model.framework` | logged to audit trail; used by `foundation/registry/` |
| `data_quality.schema` | `observability/data_quality.py:SchemaValidator` |
| `data_quality.freshness` | `observability/data_quality.py:FreshnessCheck` |
| `data_quality.outlier_detection` | `observability/data_quality.py:OutlierDetector` |
| `drift.data.method` | `observability/drift/__init__.py:DRIFT_DETECTOR_REGISTRY` |
| `drift.data.threshold` | the selected `BaseDriftDetector` subclass |
| `drift.data.window` | `observability/drift/base.py:WindowedDetector` |
| `drift.concept.method` | `observability/drift/concept_drift.py` |
| `drift.auto_check.enabled` | `core/client.py:log_prediction` — count-based auto-check |
| `drift.auto_check.every_n_predictions` | `core/client.py:log_prediction` |
| `drift.model.metrics` | `observability/drift/model_drift.py` |
| `feature_health.importance_method` | `observability/feature_health.py` |
| `feature_health.alert_on_top_n_drift` | same file |
| `alerts.channels[*].type` | `action/notifications/channels/__init__.py:CHANNEL_REGISTRY` |
| `alerts.channels[*].webhook_url` | the selected `BaseChannel` subclass — SecretStr |
| `alerts.policies.cooldown` | `action/notifications/policies.py:CooldownPolicy` |
| `alerts.policies.escalation` | `action/notifications/policies.py:EscalationPolicy` |
| `retraining.trigger` | `action/retrain/triggers.py` |
| `retraining.pipeline` | `action/retrain/orchestrator.py` |
| `retraining.deploy_on_promote` | `action/retrain/orchestrator.py:_maybe_deploy` |
| `retraining.approval` | `action/retrain/approval.py:ApprovalGate` |
| `deployment.strategy` | `action/deployment/__init__.py:STRATEGY_REGISTRY` |
| `deployment.target` | `action/deployment/targets/__init__.py:TARGET_REGISTRY` |
| `deployment.canary` | `action/deployment/strategies/canary.py` |
| `deployment.blue_green` | `action/deployment/strategies/blue_green.py` |
| `audit.storage` | `core/client.py:_build_audit_shipper` — supports `local`, `azure_blob`, `s3`, `gcs` |
| `audit.gcs.*` | `integrations/gcp/gcs_audit.py:GcsShipper` |
| `audit.tamper_evidence` | `foundation/audit/trail.py` |
| `audit.keystore` | `foundation/audit/keystore.py` |
| `audit.compliance_frameworks` | `foundation/audit/compliance.py` |
| `registry.backend` | `foundation/registry/backends/__init__.py:resolve_backend` |
| `business_kpi.mappings` | `intelligence/kpi_linker.py` |
| `model_graph.dependencies` | `intelligence/model_graph.py` |
| `domains.timeseries.*` | `domains/timeseries/adapter.py` |
| `domains.nlp.*` | `domains/nlp/adapter.py` |
| `domains.recommendation.*` | `domains/recommendation/adapter.py` |
| `domains.graph.*` | `domains/graph/adapter.py` |
| `llmops.enabled` | `core/client.py:__init__` — toggles entire layer |
| `llmops.prompts` | `llmops/prompt_manager.py` |
| `llmops.guardrails.input` | `llmops/guardrails/engine.py:GuardrailPipeline` |
| `llmops.guardrails.output` | same |
| `llmops.guardrails.*.type=custom` | `llmops/guardrails/custom.py:CustomGuardrail` |
| `llmops.guardrails.*.type=plugin` | `llmops/guardrails/plugin.py:PluginGuardrail` |
| `llmops.quality.evaluator` | `llmops/quality/evaluator.py` |
| `llmops.quality.semantic_drift` | `llmops/quality/semantic_drift.py` |
| `llmops.quality.retrieval_quality` | `llmops/quality/retrieval_quality.py` |
| `llmops.token_economics` | `llmops/token_economics.py` |
| `llmops.prompt_drift` | `llmops/prompt_drift.py` |
| `agentops.enabled` | `core/client.py:__init__` — toggles entire layer |
| `agentops.tracing` | `agentops/trace/tracer.py` |
| `agentops.tool_audit` | `agentops/tool_audit/monitor.py` |
| `agentops.safety.loop_detection` | `agentops/safety/loop_detector.py` |
| `agentops.safety.budget` | `agentops/safety/budget_guard.py` |
| `agentops.safety.escalation` | `agentops/safety/escalation.py` |
| `agentops.agent_registry` | `agentops/agent_registry.py` |
| `agentops.multi_agent` | `agentops/multi_agent/*` |
| `agentops.evaluation` | `agentops/eval/*` |
| `dashboard.server.host` | `dashboard/server.py:create_dashboard_app` |
| `dashboard.server.rbac` | `dashboard/security/auth.py` |
| `dashboard.server.csrf` | `dashboard/security/csrf.py` |
| `dashboard.server.rate_limit` | `dashboard/security/rate_limit.py` |

### 7.3 Lookup table — file → config section

If you're editing a file and want to know which YAML section
users configure it through:

| File | YAML section |
|---|---|
| `foundation/datasets/registry.py` | accessed via `SentinelClient.dataset_registry` |
| `foundation/experiments/tracker.py` | accessed via `SentinelClient.experiment_tracker` |
| `core/client.py` | the whole config (it's the factory) |
| `config/schema.py` | defines all YAML sections |
| `observability/data_quality.py` | `data_quality` |
| `observability/drift/data_drift.py` | `drift.data` |
| `observability/drift/concept_drift.py` | `drift.concept` |
| `observability/drift/model_drift.py` | `drift.model` |
| `observability/feature_health.py` | `feature_health` |
| `observability/cost_monitor.py` | `cost_monitor` |
| `action/notifications/engine.py` | `alerts.policies` |
| `action/notifications/channels/slack.py` | `alerts.channels[type=slack]` |
| `action/notifications/channels/teams.py` | `alerts.channels[type=teams]` |
| `action/notifications/channels/pagerduty.py` | `alerts.channels[type=pagerduty]` |
| `action/notifications/channels/email.py` | `alerts.channels[type=email]` |
| `action/notifications/channels/webhook.py` | `alerts.channels[type=webhook]` |
| `action/retrain/orchestrator.py` | `retraining` (incl. `deploy_on_promote`) |
| `action/notifications/escalation.py` | `alerts.policies.escalation` |
| `action/deployment/manager.py` | `deployment` |
| `action/deployment/strategies/canary.py` | `deployment.canary` |
| `action/deployment/strategies/blue_green.py` | `deployment.blue_green` |
| `action/deployment/strategies/direct.py` | `deployment` with `strategy: direct` |
| `action/deployment/targets/azure_ml_endpoint.py` | `deployment.azure_ml_endpoint` |
| `action/deployment/targets/sagemaker.py` | `deployment` with `target: sagemaker_endpoint` |
| `action/deployment/targets/vertex_ai.py` | `deployment` with `target: vertex_ai_endpoint` |
| `foundation/audit/trail.py` | `audit` |
| `foundation/audit/integrity.py` | `audit.tamper_evidence` |
| `foundation/audit/shipper.py` | `audit.storage` (non-local backends) |
| `foundation/registry/backends/local.py` | `registry` with `backend: local` |
| `foundation/registry/backends/azure_ml.py` | `registry` with `backend: azure_ml` |
| `foundation/registry/backends/mlflow.py` | `registry` with `backend: mlflow` |
| `foundation/registry/backends/sagemaker.py` | `registry` with `backend: sagemaker` |
| `foundation/registry/backends/vertex_ai.py` | `registry` with `backend: vertex_ai` |
| `foundation/registry/backends/databricks.py` | `registry` with `backend: databricks` |
| `config/aws_secrets.py` | `${awssm:...}` tokens in any config value |
| `config/gcp_secrets.py` | `${gcpsm:...}` tokens in any config value |
| `config/signing.py` | `sentinel config sign` / `verify-signature` |
| `llmops/prompt_manager.py` | `llmops.prompts` |
| `llmops/guardrails/engine.py` | `llmops.guardrails` |
| `llmops/guardrails/custom.py` | `llmops.guardrails` entries with `type: custom` |
| `llmops/guardrails/plugin.py` | `llmops.guardrails` entries with `type: plugin` |
| `llmops/token_economics.py` | `llmops.token_economics` |
| `agentops/integrations/langgraph.py` | `agentops.tracing.auto_instrument.langgraph` |
| `agentops/trace/tracer.py` | `agentops.tracing` |
| `agentops/safety/budget_guard.py` | `agentops.safety.budget` |
| `agentops/tool_audit/monitor.py` | `agentops.tool_audit` |
| `domains/timeseries/adapter.py` | `domains.timeseries` |
| `dashboard/server.py` | `dashboard.server` |
| `dashboard/security/auth.py` | `dashboard.server.rbac` + `dashboard.server.auth` |

---

## 8. Example configs — what each one demonstrates

The `configs/examples/` directory is your **best learning
resource** after this guide and `CLAUDE.md`. Every config is a
worked example of a real production scenario. Don't copy-paste
them blindly — open each one and trace how its YAML maps to live
code using the table in section 7.

### 8.1 `minimal.yaml` — 28 lines, the "hello world"

**Scenario.** You have a sklearn classifier and want Slack
alerts when PSI exceeds 0.2.

**What's in it.** Model metadata, one PSI detector, one Slack
channel with a 1h cooldown, local audit storage.

**What's NOT in it.** No data quality checks, no concept/model
drift, no feature health, no retraining, no deployment automation,
no registry (uses local default), no LLMOps, no AgentOps, no
dashboard. **This is intentional.** It's the *smallest* config
that produces value.

**Use it to learn.** Load it into `SentinelClient`, call
`log_prediction` a few times with synthetic data, and watch the
audit log fill up.

### 8.2 `insurance_fraud.yaml` — the full BFSI monty

**Scenario.** A claims fraud classifier in an FCA-regulated
insurer. Needs drift detection, escalation through Slack →
Teams → PagerDuty, human-approval retraining, canary deployment
on Azure, full audit trail with FCA Consumer Duty compliance.

**What's in it.**

- `data_quality` with JSON Schema validation and freshness checks
- `drift.data` (PSI), `drift.concept` (DDM), `drift.model` (F1 decay)
- `feature_health` with SHAP-based importance
- Three notification channels with escalation chains
- `retraining` with human-in-loop approval via email
- `deployment` canary on Azure ML Online Endpoints
- `audit` with tamper evidence + FCA Consumer Duty framework
- `business_kpi` mapping precision → fraud catch rate
- `model_graph` showing upstream feature-engineering dependency

**Use it to learn.** This is the config you'd show to a customer
to prove Sentinel "does the real thing". Read it top to bottom,
then open each referenced file in `sentinel/` to see how the
YAML maps to code.

### 8.3 `ecommerce_reco.yaml` — recommendation model

**Scenario.** A product recommendation model. Needs beyond-accuracy
metrics (NDCG, coverage, diversity, novelty) and A/B test
integration.

**What's different.** `domains.recommendation` is enabled with
the full suite of ranking metrics. Drift is configured per the
recommendation adapter's detectors (item/user distribution drift,
cold-start ratio, long-tail monitoring).

*Note: this example config may be missing from the repo right
now — it's the only one mentioned in `CLAUDE.md` that hadn't been
committed by the cloud integration audit. If you need it, file a task to add it.*

### 8.4 `rag_claims_agent.yaml` — LLMOps + RAG

**Scenario.** A RAG-based claims QA assistant. Retrieves policy
documents, answers customer questions, must never give financial
advice.

**What's in it.**

- `llmops.enabled: true`
- `llmops.prompts` with versioning and A/B routing
- `llmops.guardrails.input` — PII detection, jailbreak detection,
  topic fencing to insurance topics only, token budget
- `llmops.guardrails.output` — toxicity, groundedness (NLI
  method), format compliance, regulatory language scan against
  a prohibited-phrases file
- `llmops.quality.evaluator` — LLM-as-judge with a 4-dimension
  rubric (relevance, completeness, clarity, safety)
- `llmops.quality.semantic_drift` — weekly embedding drift check
- `llmops.quality.retrieval_quality` — faithfulness + coverage
- `llmops.token_economics` with daily $500 budget and per-query
  $0.50 cap
- `llmops.prompt_drift` — composite regression detector

**Use it to learn.** This is the most complete LLMOps config in
the repo. If you're contributing to `sentinel/llmops/`, read this
config before you touch any code.

### 8.5 `multi_agent_underwriting.yaml` — full AgentOps

**Scenario.** Three agents collaborating on an underwriting
decision: a data collector, a risk assessor, and a decision
agent. Needs tracing, tool audit, budget guards, consensus
monitoring.

**What's in it.**

- `agentops.enabled: true`
- `agentops.tracing` with OTLP export to a local collector
- `agentops.tool_audit.permissions` — per-agent tool
  allowlist/blocklist (decision agent can't touch payment APIs)
- `agentops.safety.loop_detection` — max 50 iterations, no more
  than 5 repeated tool calls with the same input, max delegation
  depth 5
- `agentops.safety.budget` — 50k tokens, $5, 5min wall time per run
- `agentops.safety.escalation` — confidence < 0.3 → human
- `agentops.safety.sandbox` — destructive ops need approval
- `agentops.multi_agent.consensus` — ⅔ majority rule
- `agentops.evaluation.golden_datasets` — tests run daily on
  a golden suite under `tests/golden/`

**Use it to learn.** This is the most complete AgentOps config.
Again, the entry cost is high — you need to understand tracing
and span hierarchy before it clicks.

### 8.6 `demand_forecast.yaml` — time series

**Scenario.** A daily demand forecast with weekly + yearly
seasonality. Needs seasonality-aware drift detection and
calibration-aware forecast quality metrics.

**What's in it.**

- `model.domain: timeseries`
- `domains.timeseries.frequency: daily`
- `domains.timeseries.seasonality_periods: [7, 365]`
- `domains.timeseries.decomposition: stl`
- `domains.timeseries.drift` — calendar test (compare
  January 2026 to January 2025, not against the global baseline)
- `domains.timeseries.quality` — MASE, coverage, directional
  accuracy, per-horizon tracking

**Use it to learn.** The time series adapter is the best
argument for domain adapters existing at all. Open this config
side-by-side with `sentinel/domains/timeseries/drift.py` and
see how the calendar-test strategy turns into real code.

### 8.7 `ner_entity_extraction.yaml` — NLP non-LLM

**Scenario.** A spaCy-based NER model for extracting party
names from legal documents. Needs vocabulary drift + embedding
space monitoring.

**What's in it.**

- `model.domain: nlp`
- `domains.nlp.task: ner`
- `domains.nlp.drift.vocabulary` — OOV rate alert
- `domains.nlp.drift.embedding` — MMD over a sliding window
- `domains.nlp.quality` — token-level + span-level F1 per entity
  type

**Use it to learn.** NLP drift is underrepresented in the open
source world. Read this config, then open
`sentinel/domains/nlp/drift.py` to see how vocabulary + embedding
drift compose.

### 8.8 `product_reco.yaml` — recommendation (alternate)

**Scenario.** Alternative recommendation example with a different
focus — popularity bias + fairness across user segments.

**What's in it.**

- `model.domain: recommendation`
- Popularity bias tracking with Gini coefficient threshold
- Group fairness: max 10% NDCG disparity across user segments

**Use it to learn.** Good illustration of beyond-accuracy metrics
being first-class citizens, not afterthoughts.

### 8.9 `fraud_graph.yaml` — graph ML

**Scenario.** A GNN for transaction graph fraud detection. Needs
topology drift (degree distribution, density, clustering
coefficient) and link prediction quality (AUC, Hits@K, MRR).

**What's in it.**

- `model.domain: graph`
- `domains.graph.task: link_prediction`
- `domains.graph.graph_type: transaction_graph`
- `domains.graph.drift.topology` — degree distribution KS test,
  density monitoring, clustering coefficient tracking
- `domains.graph.quality` — AUC, Hits@10, MRR, embedding
  isotropy

**Use it to learn.** The graph adapter is the most specialised
of the five. If you've never monitored a GNN before, this is
the "why domain adapters matter" example.

### 8.10 How to use these configs in your own dev work

1. Pick the closest example to your scenario.
2. Copy it to your working directory as `sentinel.yaml`.
3. `sentinel config validate --strict sentinel.yaml` — fix any
   missing env vars or file paths.
4. Edit the model name and drift thresholds.
5. Run `SentinelClient.from_config("sentinel.yaml")` in a REPL
   and explore the returned client object.
6. When everything works, commit your config alongside your
   model serving code.

---

## 9. Data flow — a prediction's journey through Sentinel

The fastest way to understand the whole SDK is to trace a single
prediction from the moment it enters the SDK to the moment an
alert gets fired. This section does exactly that.

### 9.1 The happy path — no drift, no alerts

```
1.  user code calls client.log_prediction(features=X, prediction=y, actuals=None)
         │
         ▼
2.  SentinelClient.log_prediction  (core/client.py)
    - thread-safe append to in-memory prediction buffer
    - if actual provided + concept drift configured:
      feed error signal to streaming concept detector (DDM/EDDM/ADWIN/PH)
    - if count-based auto-check enabled:
      increment counter → if threshold reached, spawn daemon thread
      to run check_drift() in the background
    - returns immediately (non-blocking)
         │
         ▼
3.  A background thread (started in SentinelClient.__init__) wakes up
    on its window boundary (every 1000 preds or every 10 min)
         │
         ├──►  observability/data_quality.py
         │     schema validation + freshness + outlier check
         │     → QualityReport
         │
         ├──►  observability/drift/data_drift.py
         │     PSI / KS / JS on the buffered window vs baseline
         │     → DriftReport
         │
         ├──►  observability/feature_health.py
         │     per-feature drift × importance rank
         │     → FeatureHealthReport
         │
         └──►  observability/cost_monitor.py
               latency p99, throughput, cost per prediction
               → CostReport
         │
         ▼
4.  foundation/audit/trail.py
    - every report becomes an audit event
    - HMAC-SHA256 chain adds a new link
    - JSON line appended to the configured backend
    - if storage=azure_blob, the day-rotation shipper ships
      yesterday's file to Azure Blob in a background thread
         │
         ▼
5.  intelligence/model_graph.py
    - if upstream model drifted, cascade alert
      downstream dependents
         │
         ▼
6.  if DriftReport.is_drifted == False:
         └─► stop here. No alert. No action.
```

In the happy path, steps 1–5 take single-digit milliseconds. The
user's serving endpoint is unaffected — everything heavy runs in
the background thread.

### 9.2 The drift-detected path

```
Continuing from step 6 above when DriftReport.is_drifted == True:
         │
         ▼
7.  action/notifications/engine.py
    NotificationEngine.fire(alert)
    - consult policies.py: is this alert in cooldown?
      - yes → suppress, log to audit, stop
      - no → continue
    - pick severity from drift.severity field
    - fan out to configured channels
         │
         ├──►  channels/slack.py     → POST webhook
         ├──►  channels/teams.py     → POST webhook
         └──►  channels/pagerduty.py → create incident
         │
         ▼
8.  if retraining.trigger == "drift_confirmed":
         │
         ▼
    action/retrain/orchestrator.py
    - start the configured retrain pipeline (Azure ML pipeline ID)
    - wait for completion
    - validate candidate on holdout dataset
    - compare against champion
    - if improvement > auto_promote_threshold:
      - auto-promote (if approval mode allows)
    - else:
      - action/retrain/approval.py
      - send approval request via notification engine
      - wait for human approval (with timeout)
         │
         ▼
9.  if approved and deployment.strategy set:
         │
         ▼
    action/deployment/manager.py
    - resolve strategy (canary/blue_green/shadow/direct)
    - resolve target (local/azure_ml_endpoint/azure_app_service/aks) [cloud integration]
    - start the deploy
    - on each ramp step: health check → traffic split → sleep
    - on failure: target.rollback_to(previous_version)
         │
         ▼
10. foundation/audit/trail.py
    - every step above is audited
    - hash chain preserves ordering
```

The whole chain runs **without any direct Python API call from
the user**. The only thing the user's code did was
`log_prediction(...)`. Everything else is driven by the YAML.

### 9.3 LLMOps path — a RAG query

```
1.  user calls client.llmops.log_call(
        prompt_name="claims_qa",
        query=user_msg,
        context_chunks=retrieved,
        response=llm_output,
        usage=...
    )
         │
         ├──► llmops/guardrails/engine.py (if called separately)
         │    input pipeline: PII → jailbreak → topic → budget
         │    output pipeline: toxicity → groundedness → format → regulatory
         │
         ├──► llmops/quality/evaluator.py
         │    sample_rate=0.1 → LLM-as-judge scores this call
         │
         ├──► llmops/quality/semantic_drift.py
         │    embed response, update sliding-window centroid
         │
         ├──► llmops/quality/retrieval_quality.py
         │    relevance + utilisation + faithfulness from chunks
         │
         ├──► llmops/token_economics.py
         │    input_tokens × input_price + output_tokens × output_price
         │    check against daily budget
         │
         └──► llmops/prompt_drift.py
              composite signal: quality ↓ + violations ↑ + tokens ↑
         │
         ▼
2.  foundation/audit/trail.py — log the full LLM call
3.  if any guardrail blocked → notification
4.  if prompt drift detected → optional A/B test trigger
```

### 9.4 AgentOps path — an agent run

```
1.  user invokes an agent wrapped with LangGraphMiddleware(tracer).wrap(graph)
         │
         ▼
2.  agentops/trace/tracer.py
    - creates the root span "agent_run"
    - starts budget_guard.start(run_id)
         │
         ▼
3.  for each agent step:
         │
         ├─► tracer.span("plan") → LLM call → span ends with duration
         │   tool_audit/monitor.py records the call
         │   tool_audit/permissions.py checks allowlist
         │   safety/budget_guard.py decrements remaining budget
         │   safety/loop_detector.py checks for repetition
         │
         ├─► tracer.span("tool_call: search_policy")
         │   same checks
         │
         ├─► tracer.span("synthesise")
         │
         └─► tracer.span("respond")
         │
         ▼
4.  trace export via agentops/trace/exporters/otlp.py
         │
         ▼
5.  agentops/eval/task_completion.py
    - task marked completed or failed
    - agentops/eval/trajectory.py compares against golden trajectory
         │
         ▼
6.  foundation/audit/trail.py — full trace persisted
7.  if budget exceeded / loop detected → safety/escalation.py → notification
```

This is the mental model to keep. In every path, the same three
things happen: **compute something → write to audit trail → maybe
fire an alert**.

---

## 10. How to extend Sentinel

Five worked examples. These are the most common contributions and
they cover 90% of real PRs. Three more extension patterns follow
(sections 10.6–10.8) covering registry backends, guardrails, and
trace exporters.

### 10.1 Add a new drift detector (Wasserstein-style example)

**Scenario.** You want to add a new drift method called
`earth_movers_2d` for 2-D feature spaces.

**Steps.**

1. Create `sentinel/observability/drift/earth_movers_2d.py`:

   ```python
   from sentinel.observability.drift.base import BaseDriftDetector
   from sentinel.core.types import DriftReport

   class EarthMovers2DDriftDetector(BaseDriftDetector):
       def __init__(self, threshold: float = 0.1):
           self.threshold = threshold
           self._ref = None

       def fit(self, reference_data):
           self._ref = self._preprocess(reference_data)

       def detect(self, current_data) -> DriftReport:
           cur = self._preprocess(current_data)
           distance = self._compute_em_distance(self._ref, cur)
           return DriftReport(
               is_drifted=distance > self.threshold,
               severity="high" if distance > self.threshold * 2 else "medium",
               test_statistic=distance,
               p_value=None,
               feature_scores={},
               timestamp=datetime.now(UTC),
           )

       def reset(self):
           self._ref = None
   ```

2. Register in `sentinel/observability/drift/__init__.py`:

   ```python
   from sentinel.observability.drift.earth_movers_2d import EarthMovers2DDriftDetector

   DRIFT_DETECTOR_REGISTRY = {
       "psi": PSIDriftDetector,
       "ks": KSDriftDetector,
       # ...existing entries...
       "earth_movers_2d": EarthMovers2DDriftDetector,
   }
   ```

3. Add to the `Literal` in `sentinel/config/schema.py`:

   ```python
   class DataDriftConfig(_Base):
       method: Literal["psi", "ks", "js", "chi_squared", "wasserstein", "earth_movers_2d"] = "psi"
   ```

4. Write tests in
   `tests/unit/observability/drift/test_earth_movers_2d.py`. Use
   `hypothesis` to verify statistical properties on synthetic data.

5. Update `docs/config-reference.md` drift section with the new
   method name and when to use it.

**You did not touch** `core/client.py`, `action/notifications/*`,
or any subsystem above `observability/drift/`. That's the whole
point of the ABC + registry pattern.

### 10.2 Add a new notification channel (Discord example)

**Scenario.** You want Discord alerts.

**Steps.**

1. Create `sentinel/action/notifications/channels/discord.py`:

   ```python
   import httpx
   from sentinel.action.notifications.channels.base import BaseChannel
   from sentinel.core.types import Alert, DeliveryResult

   class DiscordChannel(BaseChannel):
       def __init__(self, webhook_url, username="Sentinel"):
           self.webhook_url = webhook_url
           self.username = username

       async def send(self, alert: Alert) -> DeliveryResult:
           payload = {
               "username": self.username,
               "embeds": [{
                   "title": alert.title,
                   "description": alert.message,
                   "color": self._severity_color(alert.severity),
               }],
           }
           async with httpx.AsyncClient() as client:
               response = await client.post(
                   self.webhook_url.get_secret_value(),  # SecretStr!
                   json=payload,
               )
           return DeliveryResult(
               success=response.status_code == 204,
               channel="discord",
               timestamp=datetime.now(UTC),
           )
   ```

2. Register in `sentinel/action/notifications/channels/__init__.py`:

   ```python
   CHANNEL_REGISTRY = {
       "slack": SlackChannel,
       "teams": TeamsChannel,
       "discord": DiscordChannel,  # new
       ...
   }
   ```

3. Add a schema variant in `sentinel/config/schema.py`:

   ```python
   class DiscordChannelConfig(_Base):
       type: Literal["discord"]
       webhook_url: SecretStr
       username: str = "Sentinel"

   # Add to the discriminated union:
   ChannelConfig = Annotated[
       SlackChannelConfig | TeamsChannelConfig | DiscordChannelConfig | ...,
       Field(discriminator="type"),
   ]
   ```

4. Test with a mocked `httpx.AsyncClient`:
   `tests/unit/action/notifications/channels/test_discord.py`.

5. Document in `docs/config-reference.md` alerts section.

### 10.3 Add a new deployment target (Cloud Run example)

**Scenario.** You want GCP Cloud Run as a deployment target.

**Steps.**

1. Create `sentinel/action/deployment/targets/cloud_run.py`.
2. Implement `BaseDeploymentTarget`:
   - `set_traffic_split(model_name, weights)` — use
     `google.cloud.run_v2` client to update revision traffic.
   - `health_check(model_name, version)` — hit `/healthz`.
   - `rollback_to(model_name, version)` — set traffic back.
3. **Lazy import** `google.cloud.run_v2` inside `__init__`, raise
   a clear `DeploymentError` if the extra isn't installed.
4. Register in `sentinel/action/deployment/targets/__init__.py`:

   ```python
   TARGET_REGISTRY = {
       "local": LocalDeploymentTarget,
       "azure_ml_endpoint": AzureMLEndpointTarget,
       "azure_app_service": AzureAppServiceTarget,
       "aks": AKSDeploymentTarget,
       "cloud_run": CloudRunTarget,  # new
   }
   ```

5. Add a `CloudRunTargetConfig` sub-model and the `"cloud_run"`
   variant to `DeploymentConfig.target` Literal in `schema.py`.
6. Add a strategy/target compat matrix entry — which strategies
   work with Cloud Run? (canary works; blue_green works; shadow
   works; direct works.)
7. Add a `[gcp]` extra in `pyproject.toml` with the GCP SDK.
8. Tests mock `google.cloud.run_v2.ServicesClient`.
9. Add a new `docs/gcp.md` section (or append to a future
   `docs/deployment-targets.md`).

### 10.4 Add a new domain adapter (survival analysis example)

**Scenario.** You want to add a survival analysis domain with
time-to-event-specific drift and metrics (concordance index,
Brier score).

**Steps.**

1. Create `sentinel/domains/survival/adapter.py`:

   ```python
   from sentinel.domains.base import BaseDomainAdapter

   class SurvivalAdapter(BaseDomainAdapter):
       def get_drift_detectors(self):
           return [CensoringRateDriftDetector(), HazardRatioDriftDetector()]

       def get_quality_metrics(self):
           return [ConcordanceIndex(), BrierScore(), IntegratedBrierScore()]

       def get_schema_validator(self):
           return SurvivalSchemaValidator()
   ```

2. Create `sentinel/domains/survival/drift.py` and
   `sentinel/domains/survival/quality.py` with the detector +
   metric implementations. **Lazy import `lifelines`** only
   inside functions that need it.

3. Register in `sentinel/domains/__init__.py`:

   ```python
   DOMAIN_REGISTRY = {
       "tabular": TabularAdapter,
       "timeseries": TimeSeriesAdapter,
       "nlp": NLPAdapter,
       "recommendation": RecommendationAdapter,
       "graph": GraphAdapter,
       "survival": SurvivalAdapter,  # new
   }
   ```

4. Add `"survival"` to `ModelConfig.domain` Literal in
   `schema.py`. Add a `DomainsConfig.survival` sub-config.

5. Add a `[survival]` optional extra with `lifelines>=0.27`.

6. Create a worked example:
   `configs/examples/survival_churn.yaml`.

7. Write tests in `tests/unit/domains/survival/`.

8. Update `docs/domains.md` (create if missing) with the new
   adapter.

### 10.5 Add a new top-level YAML field (e.g. `feature_store`)

**Scenario.** You want to declare a feature store endpoint in
the config so Sentinel can join features at inference time.

**Steps.**

1. Add the Pydantic model in `sentinel/config/schema.py`:

   ```python
   class FeatureStoreConfig(_Base):
       backend: Literal["feast", "tecton", "custom"] = "feast"
       endpoint: str
       api_key: SecretStr | None = None
       default_feature_view: str | None = None
   ```

2. Add the field to `SentinelConfig`:

   ```python
   class SentinelConfig(_Base):
       # ...existing fields...
       feature_store: FeatureStoreConfig | None = None
   ```

   Optional default = `None` so old configs still load.

3. Create a new module `sentinel/integrations/feature_store/`
   with `BaseFeatureStore`, `FeastFeatureStore`,
   `TectonFeatureStore`. Lazy imports.

4. Wire into `SentinelClient.__init__`:

   ```python
   self.feature_store = self._build_feature_store(config)
   ```

   where `_build_feature_store` checks `config.feature_store is
   not None` and instantiates the right backend.

5. Expose `client.feature_store.get_features(entity_ids)` as the
   public API.

6. Tests + docs + a CHANGELOG entry.

**Checklist** (pin this somewhere) for adding any new feature:

- [ ] Schema: `sentinel/config/schema.py` field added
- [ ] Defaults: sensible default so old configs load
- [ ] Implementation: new module under the right subsystem
- [ ] Registry: `LITERAL_FIELD` ↔ registry dict in sync
- [ ] Factory: `SentinelClient._build_*` helper added
- [ ] Audit: events logged to the audit trail
- [ ] Dashboard: new view if it's user-facing
- [ ] CLI: new subcommand if it's operator-facing
- [ ] Tests: unit + integration, ≥ 90% coverage
- [ ] Docs: `docs/config-reference.md` entry
- [ ] Docs: worked example in `configs/examples/`
- [ ] Changelog: entry in `CHANGELOG.md`
- [ ] Extras: `pyproject.toml` optional extra if heavy deps

### 10.6 Add a new registry backend (DynamoDB example)

**Scenario.** You want to store model versions in AWS DynamoDB.

**Steps.**

1. Create `sentinel/foundation/registry/backends/dynamodb.py`:

   ```python
   from sentinel.foundation.registry.backends.base import BaseRegistryBackend

   class DynamoDBRegistryBackend(BaseRegistryBackend):
       def __init__(self, table_name: str, region: str = "us-east-1"):
           # Lazy import
           import boto3
           self._table = boto3.resource("dynamodb", region_name=region).Table(table_name)

       def register(self, model_name, version, metadata):
           ...

       def get(self, model_name, version):
           ...

       def list_versions(self, model_name):
           ...
   ```

2. Register in `sentinel/foundation/registry/backends/__init__.py`:

   ```python
   BACKEND_REGISTRY = {
       "local": LocalRegistryBackend,
       # ...existing...
       "dynamodb": _lazy_dynamodb,
   }
   ```

3. Add `"dynamodb"` to `RegistryConfig.backend` Literal in `schema.py`.
4. Add a `DynamoDBRegistryConfig` sub-model if needed.
5. Tests mock `boto3.resource`.

### 10.7 Add a new guardrail (compliance check example)

**Scenario.** You want to add a built-in guardrail that checks for
financial advice language in LLM responses.

**Steps.**

1. Create `sentinel/llmops/guardrails/financial_advice.py`:

   ```python
   from sentinel.llmops.guardrails.base import BaseGuardrail, GuardrailResult

   class FinancialAdviceGuardrail(BaseGuardrail):
       def __init__(self, patterns: list[str] | None = None):
           self.patterns = patterns or ["you should invest", "I recommend buying"]

       def check(self, content: str, context: dict | None = None) -> GuardrailResult:
           for pattern in self.patterns:
               if pattern.lower() in content.lower():
                   return GuardrailResult(action="block", reason=f"Financial advice detected: {pattern}")
           return GuardrailResult(action="pass")
   ```

2. Register in `sentinel/llmops/guardrails/__init__.py`:

   ```python
   GUARDRAIL_REGISTRY = {
       # ...existing...
       "financial_advice": FinancialAdviceGuardrail,
   }
   ```

3. Add `"financial_advice"` to the guardrail type Literal in `schema.py`.
4. Tests in `tests/unit/llmops/guardrails/test_financial_advice.py`.
5. Document in `docs/config-reference.md` guardrails section.

Note: for one-off or organisation-specific guardrails, consider using
`type: custom` (DSL rules, no code) or `type: plugin` (dynamic class
loading) instead of adding to the SDK core.

### 10.8 Add a new trace exporter (Datadog example)

**Scenario.** You want to export agent traces to Datadog APM.

**Steps.**

1. Create `sentinel/agentops/trace/exporters/datadog.py`:

   ```python
   from sentinel.agentops.trace.exporters.base import BaseExporter

   class DatadogExporter(BaseExporter):
       def __init__(self, api_key: str, site: str = "datadoghq.com"):
           # Lazy import
           from ddtrace import tracer
           self._tracer = tracer
           self._api_key = api_key

       def export(self, trace):
           # Convert Sentinel spans to Datadog spans
           ...

       def flush(self):
           ...
   ```

2. Register in `sentinel/agentops/trace/exporters/__init__.py`:

   ```python
   EXPORTER_REGISTRY = {
       "jsonl": JsonlExporter,
       "otlp": OTLPExporter,
       "datadog": DatadogExporter,
   }
   ```

3. Add `"datadog"` to the exporter backend Literal in `schema.py`.
4. Add a `[datadog]` optional extra in `pyproject.toml`.
5. Tests mock the `ddtrace` library.

---

## 11. Test layout

**Current status:** 2,212 tests across 129 test files, 87% line
coverage, 211 source files.

```
tests/
├── conftest.py                   # Shared fixtures, synthetic datasets
├── unit/                         # Fast, no I/O, fully mocked — 129 test files
│   ├── config/                   # Config loader, env vars, signing, secrets, key vault, AWS/GCP
│   ├── core/                     # Prediction buffer, client lifecycle
│   ├── observability/
│   │   ├── drift/
│   │   │   └── test_model_drift.py
│   │   ├── test_data_quality_full.py
│   │   ├── test_cost_monitor_enhanced.py
│   │   ├── test_feature_importance.py
│   │   └── test_schema_inference.py
│   ├── action/
│   │   ├── notifications/        # Digest, escalation, dispatch, rate limit, templates
│   │   ├── retrain/              # Orchestrator, approval, deploy-on-promote
│   │   └── deployment/
│   │       └── targets/          # AKS, App Service, Azure ML, SageMaker, Vertex AI
│   ├── foundation/
│   │   ├── audit/                # Hash chain, keystore, shipper, trail integration
│   │   ├── registry/
│   │   │   └── backends/         # Azure ML, MLflow, SageMaker, Vertex AI, Databricks
│   │   ├── test_compliance_full.py
│   │   ├── test_lineage_full.py
│   │   ├── test_experiment_tracker_full.py
│   │   └── test_dataset_registry.py
│   ├── llmops/
│   │   ├── guardrails/           # Format, regulatory, token budget, custom DSL, plugin
│   │   └── quality/              # Evaluator, semantic drift, retrieval quality, prompt manager
│   ├── agentops/                 # Tracer, tool monitor, tool replay, registry, safety, eval
│   │   ├── trace/
│   │   ├── safety/
│   │   └── eval/
│   ├── domains/                  # Five adapters, each with drift + quality tests
│   ├── intelligence/
│   │   ├── test_explainability.py
│   │   ├── test_cohort_analyzer.py
│   │   ├── test_feature_health.py
│   │   └── test_kpi_linker.py
│   ├── integrations/
│   │   ├── aws/                  # S3 audit shipper
│   │   ├── azure/                # Blob audit shipper, pipeline runner
│   │   └── gcp/                  # GCS audit shipper
│   ├── dashboard/
│   │   ├── routes/               # Overview, drift, registry, audit, agentops, etc.
│   │   ├── security/             # Auth, bearer, RBAC, CSRF, rate limit, headers
│   │   └── test_dashboard_export.py
│   └── cli/                      # Audit verify, config sign, cloud test, init CI
├── integration/                  # Requires external fixtures; opt-in via -m integration
│   └── azure/                    # Tape-replay (vcrpy) tests — no live Azure needed
└── fixtures/
    ├── configs/                  # Valid and invalid sample configs
    ├── data/                     # Synthetic drift scenarios
    └── vcr/azure/                # Recorded HTTP cassettes for integration tests
```

**Rules for tests.**

1. **Unit tests** must never hit the network or the filesystem
   (except via `tmp_path`). Mock everything.
2. **Integration tests** are marked `@pytest.mark.integration`
   and skipped by default. Run with `pytest -m integration`.
3. **Tape-replay** tests (Azure) use vcrpy cassettes in
   `tests/fixtures/vcr/azure/`. Record once against a real
   subscription, commit the cassette, replay forever in CI.
4. **Property-based** tests use `hypothesis` for drift detectors.
   Verify statistical properties (monotonicity, symmetry) hold
   across random distributions.
5. **Test data** is always synthetic. Never commit real customer
   data. `conftest.py` has a suite of fixtures for common
   distributions (normal, drifted, categorical, temporal).
6. **Coverage target.** Overall coverage is 87% line. Core modules
   ≥ 90%. Hardening workstreams (config hardening, cloud integration, security hardening — all complete)
   are above 95% on their surface.

**Running tests:**

```bash
pytest tests/unit/ -v                              # fast unit suite (~30s)
pytest tests/unit/ --cov=sentinel --cov-report=term-missing  # coverage
pytest tests/integration/ -v -m integration       # integration suite
pytest tests/unit/config/ -v                      # just the config tests
pytest -k "test_psi"                              # just PSI tests
```

---

## 12. Dashboard internals

```
sentinel/dashboard/
├── server.py            # create_dashboard_app() + SentinelDashboardRouter
├── state.py             # in-memory cache
├── deps.py              # FastAPI Depends() helpers
├── security/
│   ├── auth.py          # basic + bearer JWT + RBAC
│   ├── csrf.py          # double-submit-cookie
│   ├── rate_limit.py    # token-bucket
│   └── headers.py       # HSTS + CSP + frame guards
├── routes/              # one module per top-level page (21 pages)
├── views/               # Jinja2 template helpers
├── templates/           # HTML templates (HTMX + Plotly, 12 charts)
└── static/              # CSS, JS, images
```

**Stack.** FastAPI + Jinja2 + HTMX + Plotly. **Zero build step.**
No npm, no webpack, no React. Customers can run the dashboard on
a raw Ubuntu VM with just `pip install sentinel-mlops[dashboard]`.

**Pages (21+).** Overview, drift, features, registry, audit,
llmops, agentops, deployments, compliance, cohorts,
cohort detail (`/cohorts/{id}`), explanations (global feature
importance), datasets (`/datasets`), experiments (`/experiments`
with metric history charts), plus API endpoints including
`/api/audit/chart`, `/api/export/audit.csv`, `/api/export/drift.csv`,
`/api/export/metrics.csv`, and `/api/deployments/{id}/rollback`.

**UX features.** Dark mode toggle, auto-refresh with
configurable interval, and toast notifications for real-time
alerts.

**Two deployment modes:**

1. **Standalone** — `sentinel dashboard --config sentinel.yaml`
   runs a fresh FastAPI app on port 8080. Good for local dev.
2. **Embedded** —
   `SentinelDashboardRouter(client).attach(app)` mounts the
   routes onto an existing customer FastAPI app. Good for
   integrating into an existing service (share auth, share
   middleware).

**Route pattern.** Each route file in `routes/` follows this
template:

```python
from fastapi import APIRouter, Depends
from sentinel.dashboard.deps import get_client, require_permission

router = APIRouter(prefix="/drift", tags=["drift"])

@router.get("/", response_class=HTMLResponse)
async def drift_overview(
    client = Depends(get_client),
    _ = Depends(require_permission("drift:read")),
):
    reports = client.get_recent_drift_reports(limit=10)
    return templates.TemplateResponse("drift.html", {
        "request": request,
        "reports": reports,
    })
```

**`Depends(require_permission(...))`** is the security hardening RBAC guard.
Every non-trivial route has one. The permission names are
namespaced: `drift:read`, `deploy:promote`, `registry:write`, etc.

**HTMX pattern.** Pages use HTMX for progressive enhancement.
Tables update via `hx-get`, forms submit via `hx-post`. The CSRF
middleware injects a header into every HTMX request so POST routes
are protected without code changes.

**Plotly pattern.** Charts (12 in total) are rendered server-side as JSON and
handed to Plotly.js on the client. No server-side image generation.

**Gotcha (from memory).** When writing a FastAPI dependency
function, **don't** take a `request: Request` parameter if the
module has `from __future__ import annotations` — FastAPI will
treat it as a query parameter and return 422. Use
`Header(default=None)` to read the specific header you need.
This affected `dashboard/server.py:build_basic_auth_dependency`
once; don't repeat it.

---

## 13. CLI internals

`sentinel/cli/main.py` is a Click `Group` with these subcommands:

```
sentinel
├── init                           # generate sentinel.yaml template
├── check --config sentinel.yaml   # run drift on a dataset
├── status                         # show current model status
├── deploy --version X --strategy canary --traffic 5 [--dry-run]
├── config
│   ├── validate [--strict]
│   ├── show [--unmask]
│   ├── sign
│   └── verify-signature
├── registry
│   ├── list                       # list registered model versions
│   └── show                       # show details for a version
├── audit
│   ├── query --event-type X       # query by event type / model name
│   ├── verify                     # verify HMAC chain
│   └── chain-info                 # chain head + length
├── cloud
│   └── test [--only audit|registry|deploy|keyvault]
├── dashboard --config sentinel.yaml  # start local UI
└── completion [bash|zsh|fish]     # generate shell completions
```

**Pattern.** Every subcommand is a thin adapter:

```python
@cli.command()
@click.option("--config", default="sentinel.yaml")
def check(config):
    client = SentinelClient.from_config(config)
    report = client.check_drift()
    click.echo(format_drift_report(report))
    sys.exit(0 if not report.is_drifted else 1)
```

Business logic lives in the SDK modules. The CLI **never**
contains statistical or operational code. This keeps the CLI
trivially testable and makes sure every CLI capability is also
available via the Python API.

**Adding a new subcommand** is:

1. Add a `@cli.command()` function in `main.py`.
2. Parse arguments with Click decorators.
3. Construct `SentinelClient` (or the relevant subcomponent).
4. Call an existing SDK method.
5. Format the result with `click.echo` or a rich `Table`.
6. Exit with a meaningful code (0 success, 1 failure, 2 usage).

---

## 14. Conventions and standards

### 14.1 Python style

- **Version:** Python 3.10+.
- **Formatter + linter:** `ruff` only. No black, no isort, no
  pylint. Run `ruff format sentinel/` and `ruff check sentinel/`
  before every commit.
- **Type hints:** full type hints on every public API.
  `mypy --strict` must pass on `sentinel/config/`,
  `sentinel/core/client.py`, and all cloud integration/security hardening surface areas.
- **Docstrings:** Google style. Required on all public classes
  and functions. Include `Args`, `Returns`, `Raises`, `Example`
  sections when relevant.
- **Naming:** `snake_case` for functions/variables, `PascalCase`
  for classes, `UPPER_SNAKE_CASE` for constants.
- **Imports:** absolute only. Group: stdlib → third-party → local.
  No star imports.
- **Logging:** always `structlog`, never `print()` or stdlib
  `logging` directly. Use structured key-value pairs:
  `log.info("drift.detected", model=name, severity=sev)`.

### 14.2 Architecture rules (non-negotiable)

1. **Core never imports cloud SDKs directly.** Anything
   `azure.*`, `boto3`, `google.*` lives under
   `sentinel/integrations/<cloud>/` with lazy imports. The only
   exceptions are `sentinel/action/deployment/targets/azure_*.py`
   which still follow the lazy-import rule.
2. **Optional extras stay optional.** `import sentinel` must
   work with only the core deps installed. Never put a heavy
   import at module level.
3. **Every pluggable component has an ABC.** Never hardcode a
   specific implementation in a dispatcher.
4. **Every Literal field matches a registry.** If you add to one,
   add to the other.
5. **Every state-mutating operation is audited.** If a regulator
   could ask "when did this happen?", log it to the audit trail.
6. **Secrets stay as `SecretStr`.** Unwrap only at the point of
   network use.
7. **Async for I/O.** All network calls are async. Sync wrappers
   provided for convenience when needed.
8. **Custom exceptions.** Always raise a `SentinelError`
   subclass, never `ValueError` or `RuntimeError`.
9. **Test first for drift algorithms.** Verify statistical
   properties with synthetic data before integration.
10. **Domain adapters never import heavy deps at module level.**
    Use function-local imports for `statsmodels`, `networkx`, etc.

### 14.3 Dependency management

`pyproject.toml` is the single source of truth. Rules:

- Core deps are minimal: `numpy`, `pydantic`, `pyyaml`, `click`,
  `structlog`, `croniter`, `httpx`.
- Every heavy dependency goes in an optional extra:
  `drift`, `explain`, `viz`, `azure`, `aws`, `gcp`,
  `notify-slack`, `notify-teams`, `notify-pagerduty`,
  `mlflow`, `llmops`, `agentops`, `timeseries`, `nlp-domain`,
  `recommendation`, `graph`, `dashboard`, `k8s`.
- The `all` extra intentionally excludes torch / sentence-transformers /
  presidio / spacy so it resolves on Python 3.13. Use `ml-extras`
  explicitly for those.
- Dev deps: `pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`,
  `mypy`, `pre-commit`, `hypothesis`, `vcrpy`, `pytest-recording`.

---

## 15. Thread safety patterns

Sentinel is designed to be used in multi-threaded serving frameworks
(FastAPI, Flask with threads, Gunicorn). The following locking patterns
are used throughout the codebase. Understanding them will save you
from subtle bugs.

### 15.1 The `threading.Lock` pattern

Most stateful modules use a simple `threading.Lock` to protect shared
mutable state. The lock is held for the shortest possible duration —
typically just long enough to append to a list or update a dict.

**Modules using this pattern:**

| Module | What the lock protects |
|--------|----------------------|
| `core/client.py` | Prediction buffer, auto-check counter |
| `observability/cohort_analyzer.py` | Per-cohort `deque` buffers |
| `observability/drift/concept_drift.py` | Streaming detector state (DDM/EDDM/ADWIN) |
| `intelligence/model_graph.py` | Dependency graph adjacency list |
| `intelligence/kpi_linker.py` | Auto-refresh scheduler state |
| `foundation/audit/trail.py` | Hash chain state (previous hash, event counter) |
| `foundation/audit/lineage.py` | Lineage graph persistence |
| `llmops/prompt_manager.py` | Prompt registry + A/B assignments |
| `llmops/token_economics.py` | Token usage accumulators |
| `agentops/trace/tracer.py` | Active span stack, trace list |
| `agentops/tool_audit/monitor.py` | Tool call counters |
| `agentops/tool_audit/replay.py` | Replay cache (bounded LRU) |
| `agentops/safety/budget_guard.py` | Per-run budget counters |
| `agentops/safety/loop_detector.py` | Step history ring buffer |
| `agentops/agent_registry.py` | Agent manifest store |
| `agentops/multi_agent/delegation.py` | Delegation chain tracking |
| `agentops/multi_agent/consensus.py` | Vote accumulator |
| `agentops/eval/task_completion.py` | Completion rate counters |

### 15.2 The daemon-thread + Event pattern

Long-running background tasks (escalation timers, drift schedulers,
auto-refresh) use a daemon thread with a `threading.Event` for
clean shutdown. The pattern is:

```python
class MyDaemon:
    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self.interval)
            if not self._stop_event.is_set():
                self._do_work()

    def close(self):
        self._stop_event.set()
        self._thread.join(timeout=5)
```

**Users of this pattern:** `EscalationTimer`, `DriftScheduler`,
KPI linker auto-refresh.

### 15.3 Rules for contributors

1. **Never hold a lock across I/O.** Compute under the lock, release,
   then do the network call.
2. **Never hold two locks simultaneously** unless the ordering is
   documented. Lock ordering bugs are the #1 source of deadlocks.
3. **Use `daemon=True`** for all background threads so they don't
   prevent process shutdown.
4. **Call `close()`** — `SentinelClient.close()` propagates to all
   subsystem `close()` methods. If you add a daemon thread, wire its
   shutdown into `close()`.
5. **Bounded buffers** — use `deque(maxlen=N)` or equivalent for any
   buffer that accumulates per-prediction data. Unbounded buffers are
   memory leaks in long-running services.

---

## 16. Your first 30 minutes

A recipe for setting up a working dev environment and confirming
everything runs. Do this before you touch any code.

```bash
# 1. Clone
git clone <repo-url> project-sentinel
cd project-sentinel

# 2. Create an isolated environment
python -m venv .venv
source .venv/bin/activate
python -V    # must be 3.10+

# 3. Install with the dev + all extras
pip install -e ".[all,dev]"

# 4. Install pre-commit hooks
pre-commit install

# 5. Run the unit test suite — should be green out of the box
pytest tests/unit/ -v

# 6. Run the lint + format check
ruff check sentinel/
ruff format --check sentinel/

# 7. Run mypy on the strict surface
mypy sentinel/config/ sentinel/core/client.py --strict

# 8. Generate and inspect a default config
sentinel init > /tmp/my-sentinel.yaml
sentinel config validate --strict --config /tmp/my-sentinel.yaml

# 9. Start the dashboard against the minimal example
sentinel dashboard --config configs/examples/minimal.yaml
# Open http://localhost:8080 in a browser.
```

If any of those nine steps fail, **stop** and ask in the team
chat. Don't paper over a broken setup — every downstream problem
you hit will waste more time than fixing it now.

---

## 17. Your first hour — a guided tour

Once your environment is green, open these files in order in
your editor. Read each one for 5–10 minutes. You don't need to
understand everything — the goal is to build familiarity with
the shapes.

1. **`CLAUDE.md`** — the project constitution. Skim the
   architecture diagram, module specs, and "Notes for Claude Code"
   section at the bottom.
2. **`sentinel/__init__.py`** — the public API surface. Confirm
   you understand what a user can and cannot import.
3. **`sentinel/core/client.py`** — the factory. Read
   `SentinelClient.__init__` all the way through. You won't
   understand every line, but you'll see how config becomes live
   objects.
4. **`sentinel/core/types.py`** — the shapes that flow between
   subsystems. `DriftReport`, `Alert`, `QualityReport`.
5. **`sentinel/core/exceptions.py`** — the error hierarchy.
6. **`sentinel/config/schema.py`** — read the top-level
   `SentinelConfig`, then dive into whichever sub-config interests
   you most (I recommend `DriftConfig` first, it's the simplest).
7. **`sentinel/config/loader.py`** — trace the seven-stage
   pipeline from section 6.3 of this guide against the real code.
8. **`sentinel/observability/drift/base.py`** — the
   `BaseDriftDetector` ABC. Internalise the three-method contract.
9. **`sentinel/observability/drift/data_drift.py`** — pick the
   PSI implementation and read it end-to-end. This is the
   canonical "new detector" pattern.
10. **`sentinel/foundation/audit/trail.py`** — the hash chain.
    Read `_build_event`, `_compute_hmac`, `_last_hash`, and
    `verify_integrity`. Understand why editing old entries breaks
    the chain.
11. **`sentinel/action/notifications/engine.py`** — see how an
    alert gets routed to channels based on the cooldown and
    escalation policies.
12. **`configs/examples/minimal.yaml`** — read the file side by
    side with section 7.1 of this guide.
13. **`configs/examples/insurance_fraud.yaml`** — the full BFSI
    example. Pick three sections you don't recognise and look up
    the owner files using section 7.2 / 7.3.
14. **`tests/unit/observability/drift/test_data_drift.py`** — see
    how tests mock synthetic data and assert statistical properties.
15. **`docs/config-reference.md`** — the user-facing contract for
    every YAML field. This is what your PRs must not break.

After this, you'll know enough to confidently:

- Find the file that owns any YAML field.
- Add a new drift detector, channel, or deployment strategy.
- Read a PR and know which files to scrutinise.
- Debug a config validation error without getting lost.

---

## 18. Glossary

Terms you'll see all over the codebase. Memorise these — they
are not interchangeable.

**ABC** — Abstract Base Class. Every pluggable component has one.
Defines the contract; concrete subclasses implement it.

**Adapter** — a domain-specific factory that returns the right
drift detectors, quality metrics, and schema validators for a
given ML paradigm. See `sentinel/domains/base.py`.

**AgentOps** — the operational discipline for autonomous agent
systems: tracing, tool audit, loop detection, budget guards,
human escalation. Sentinel layer 4.

**Audit trail** — the tamper-evident log of every meaningful
event. Lives in `sentinel/foundation/audit/trail.py`. security hardening added
the HMAC hash chain.

**Backend** — a concrete implementation behind an ABC. E.g.
`LocalRegistryBackend`, `AzureMLRegistryBackend`,
`MLflowRegistryBackend` are all backends for `ModelRegistry`.

**Baseline** — the reference dataset against which drift is
computed. Stored in the model registry at registration time.

**Canary** — a deployment strategy that routes a small % of
traffic to a new model version and ramps up if health checks
pass.

**Channel** — a notification delivery target (Slack, Teams,
PagerDuty, email). Each implements `BaseChannel`.

**Concept drift** — a change in the relationship between X and
y, detected via error-rate tracking. Requires ground-truth
labels, which may arrive with lag. See DDM, EDDM, ADWIN,
Page-Hinkley.

**Config as code** — the principle that every behaviour lives in
a version-controlled YAML file, not in Python code.

**Data drift** — a change in the input feature distribution.
Measured without needing labels. See PSI, KS, JS, chi-squared,
Wasserstein.

**Domain adapter** — see Adapter.

**Drift report** — a `DriftReport` dataclass with
`is_drifted`, `severity`, `test_statistic`, `p_value`,
`feature_scores`, `timestamp`. The uniform shape every detector
returns.

**Extends** — a YAML config can inherit from a parent via
`extends: base.yaml`. Resolved in `loader.py` stage 2 with
cycle detection.

**Guardrail** — a pre/post-processing check on LLM input or
output. PII detection, jailbreak detection, toxicity,
groundedness, format compliance. Each implements `BaseGuardrail`.

**Hash chain** — the HMAC-SHA256 chain in the audit trail that
makes old entries tamper-evident. See `trail.py:_compute_hmac`.

**HMAC** — Hash-based Message Authentication Code. We use
HMAC-SHA256 with a keystore-provided secret for the audit chain.

**Keystore** — pluggable storage for the HMAC key. Current
backends: `env` (environment variable), `file` (strict-perms
file). cloud integration adds an Azure Key Vault keystore.

**LLMOps** — the operational discipline for LLM applications:
prompt management, guardrails, response quality, token economics,
prompt drift detection. Sentinel layer 3.

**MLOps** — the operational discipline for traditional ML
models: drift detection, data quality, model registry, deployment
automation. Sentinel layer 2.

**Model graph** — a directed graph of model dependencies used
for cascade alerting. Upstream drift propagates to downstream
dependents.

**Policy** — a rule that governs how alerts are delivered.
Cooldown, digest mode, escalation chains. See
`action/notifications/policies.py`.

**Prompt drift** — a composite signal that a prompt's
effectiveness is degrading (quality decline + guardrail
violations + token usage increase + semantic drift in outputs).

**Registry** — two meanings. (1) Model registry: storage for
model artifacts, metadata, baselines. (2) Registry dict:
module-level `dict[str, Type]` that maps YAML backend names to
concrete classes.

**Semantic drift** — drift in the *meaning* of LLM outputs,
measured by embedding distribution centroid shift.

**Shipper** — a background worker that uploads completed audit
log files to cloud storage (`AzureBlobShipper`, `S3Shipper`) on
day rotation. See `foundation/audit/shipper.py`.

**Span** — a single timed step in an agent trace. Spans nest
to form a tree. OTel-compatible.

**Strategy** — a deployment strategy (canary, blue/green,
shadow, direct). Decides *how* traffic is split over time.

**Target** — a deployment target (local, Azure ML Online
Endpoint, Azure App Service, AKS). Decides *where* the new
version actually runs. cloud integration.

**Trace** — a tree of spans representing a full agent run.
Stored in the audit trail.

**Trajectory** — the sequence of steps an agent took to
complete a task. Compared against optimal trajectories from
golden datasets for evaluation.

**Production hardening workstreams** — post-0.1.0 hardening phases:
WS1 (config-as-code hardening — **complete**), WS2 (Azure/cloud
integrations — **complete**), WS3 (security hardening — **complete**),
WS4 (operational observability), WS5 (LLMOps/AgentOps robustness),
WS6 (dashboard write actions).

---

## 19. Where to go next

You've finished the codebase guide. The next best uses of your
time, in order:

1. **Do section 16** (first 30 minutes) if you haven't already.
2. **Do section 17** (first hour guided tour).
3. **Read `CLAUDE.md`** cover-to-cover. It's the constitution.
4. **Read `docs/config-reference.md`** to understand the
   user-facing contract.
5. **Read the workstream docs** in order of relevance to your
   first PR:
   - `docs/security.md` — audit chain, RBAC, signed
     configs.
   - `docs/azure.md` — Key Vault, registry backends,
     deploy targets.
   - `docs/cloud-integration-guide.md` — multi-cloud: Azure,
     AWS, GCP, Databricks backends.
   - (pending) operational observability `docs/observability.md` — Prometheus / OTel
     self-instrumentation (when available).
6. **Read `docs/developer-guide.md`** — the step-by-step
   integration tutorial. Gives you the user's perspective so you
   understand what we've promised them.
7. **Pick a "good first issue"** from the backlog. Typical
   first PRs: adding a new drift detector, adding a new
   notification channel, improving error messages in a
   subsystem, adding a missing example config.
8. **Write your first test.** Find a module with < 90% coverage
   (run `pytest --cov` to see) and push it above 90%. This is
   the fastest way to learn a subsystem.

Good luck. When in doubt, grep the codebase for patterns — you
will almost always find a prior example of what you're trying
to do. And when something feels more complicated than it should
be, it probably is. Ask.

---

*Maintained by: the Sentinel team. Last meaningful update:
Full codebase guide refresh — 2,212 tests (87% line coverage),
211 source files, 129 test files. All three production hardening
workstreams complete (config hardening, Azure integration, security
hardening). Covers: 6 model registry backends (local, Azure ML,
MLflow, SageMaker, Vertex AI, Databricks), 6 deployment targets,
6 notification channels, 5 domain adapters, 3 cloud secret
resolvers (Azure KV, AWS SM, GCP SM), multi-cloud audit shippers,
full dashboard security (RBAC, JWT, CSRF, rate limiting, headers,
signed configs), thread safety patterns throughout. If you find a
stale reference, a missing file path, or a dead link, fix it in
the same PR that touched the thing you noticed.*
