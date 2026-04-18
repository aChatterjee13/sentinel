# How the Config System Works

> **Audience:** Developers integrating Sentinel into their projects and
> administrators managing Sentinel configurations across teams.
>
> **See also:** [`config-reference.md`](config-reference.md) for the
> field-by-field reference of every YAML key.

---

## Table of contents

1. [Overview — why config?](#1-overview--why-config)
2. [The loading pipeline](#2-the-loading-pipeline)
3. [Config file anatomy](#3-config-file-anatomy)
4. [Schema validation](#4-schema-validation)
5. [Environment variable substitution](#5-environment-variable-substitution)
6. [Cloud secret manager integration](#6-cloud-secret-manager-integration)
7. [Config inheritance (`extends:`)](#7-config-inheritance-extends)
8. [Secret handling and `<REDACTED>` masking](#8-secret-handling-and-redacted-masking)
9. [File reference validation](#9-file-reference-validation)
10. [Config signing and verification](#10-config-signing-and-verification)
11. [Source tracing for error messages](#11-source-tracing-for-error-messages)
12. [CLI commands for config management](#12-cli-commands-for-config-management)
13. [The example configs](#13-the-example-configs)
14. [How config maps to runtime behaviour](#14-how-config-maps-to-runtime-behaviour)
15. [Best practices](#15-best-practices)

---

## 1. Overview — why config?

Every Sentinel behaviour — drift thresholds, alert channels, deployment
strategies, guardrail rules, agent safety budgets — is defined in a
single YAML file. This is a deliberate architectural choice:

- **Auditable.** `git blame` shows who changed what threshold and when.
- **Reviewable.** Risk officers and compliance teams can review a YAML
  file without reading Python.
- **Inheritable.** A base config defines org-wide policies; model-specific
  configs override only what they need.
- **GitOps-friendly.** Merge a config change, CI/CD applies it
  automatically.
- **Signable.** Production configs can be cryptographically signed to
  prevent tampering.

The SDK only needs Python code at the integration seam — one call to
`SentinelClient.from_config("sentinel.yaml")` and a few lines in your
serving endpoint.

---

## 2. The loading pipeline

When you call `SentinelClient.from_config("sentinel.yaml")`, this is
what happens under the hood:

```
sentinel.yaml
     │
     ▼
┌────────────────────┐
│  1. _read_raw()    │  Parse YAML/JSON into a raw Python dict
└────────────────────┘
     │
     ▼
┌──────────────────────────┐
│  2. _resolve_inheritance │  Walk the extends: chain, merge parent
│     ()                   │  → child with per-field source tracking
└──────────────────────────┘
     │
     ▼
┌──────────────────────────┐
│  3. harvest()            │  Strip source annotations into a
│                          │  SourceMap (kept for error reporting)
└──────────────────────────┘
     │
     ▼
┌──────────────────────────┐
│  4. _substitute_env()    │  Replace ${VAR} and ${VAR:-default}
│                          │  with os.environ values. Also resolves
│                          │  ${azkv:...}, ${awssm:...}, ${gcpsm:...}
└──────────────────────────┘
     │
     ▼
┌──────────────────────────┐
│  5. verify_signature()   │  (Optional) Check sidecar .sig file
│                          │  against an HMAC key
└──────────────────────────┘
     │
     ▼
┌──────────────────────────┐
│  6. SentinelConfig       │  Pydantic validates every field against
│     .model_validate()    │  76 typed schema classes. Fails fast
│                          │  with clear, source-traced errors.
└──────────────────────────┘
     │
     ▼
  SentinelConfig object → SentinelClient uses it to wire up all modules
```

**Key properties:**

- **Fail-fast.** If anything is wrong (missing field, wrong type, invalid
  combination), the loader raises a `ConfigValidationError` with the
  exact field path, the error message, and (if using `extends:`) which
  file the field came from.
- **Cached.** The `ConfigLoader` caches the validated config. Subsequent
  calls return the cached object unless `force=True`.
- **Immutable output.** The `SentinelConfig` object is a frozen Pydantic
  model. No one can mutate it after loading.

### Python API

```python
from sentinel.config.loader import ConfigLoader, load_config

# Simple — one-liner
config = load_config("sentinel.yaml")

# With options
loader = ConfigLoader(
    "sentinel.yaml",
    strict_env=True,             # fail if any ${VAR} is unset
    verify_signature=True,       # check .sig sidecar
    signature_keystore=keystore, # HMAC key source
)
config = loader.load()

# Access the source map (for debugging)
source_map = loader.source_map
```

### From SentinelClient

```python
from sentinel import SentinelClient

# This calls ConfigLoader internally
client = SentinelClient.from_config("sentinel.yaml")

# The config is available as:
client.config  # → SentinelConfig
```

---

## 3. Config file anatomy

A `sentinel.yaml` has up to 20 top-level sections. Only `model.name`
is required — everything else has sensible defaults.

```yaml
version: "1.0"                    # Config schema version

# ── Required ────────────────────────────────────
model:
  name: claims_fraud_v2           # The only required field
  type: classification            # classification | regression | ranking | forecasting | generation
  domain: tabular                 # tabular | timeseries | nlp | recommendation | graph
  framework: xgboost              # Auto-detected if omitted

# ── Observability ───────────────────────────────
data_quality:                     # Schema validation, freshness, outliers
drift:                            # Data drift, concept drift, model drift
feature_health:                   # Per-feature monitoring + importance ranking
cost_monitor:                     # Latency, throughput, cost tracking
cohort_analysis:                  # Subgroup performance analysis

# ── Action ──────────────────────────────────────
alerts:                           # Channels (Slack/Teams/PagerDuty/email), escalation
retraining:                       # Drift → retrain → validate → promote pipeline
deployment:                       # Shadow, canary, blue-green strategies + cloud targets

# ── Foundation ──────────────────────────────────
audit:                            # Storage, retention, compliance, tamper evidence
registry:                         # Model registry backend (local/Azure ML/MLflow/SageMaker/...)
model_graph:                      # Multi-model dependency DAG
business_kpi:                     # Model metric → business outcome mapping

# ── LLMOps ──────────────────────────────────────
llmops:                           # Prompts, guardrails, quality, token economics

# ── AgentOps ────────────────────────────────────
agentops:                         # Tracing, tool audit, safety, agent registry

# ── Domain-specific ─────────────────────────────
domains:                          # Time series, NLP, recommendation, graph adapter configs

# ── Dashboard ───────────────────────────────────
dashboard:                        # Server settings, auth, RBAC, UI preferences

# ── Data management ─────────────────────────────
datasets:                         # Dataset registry config
experiments:                      # Experiment tracking config

# ── Inheritance ─────────────────────────────────
extends: base.yaml                # (Optional) Inherit from a parent config
```

### Minimal config (10 lines)

```yaml
version: "1.0"
model:
  name: my_model
  type: classification
drift:
  data:
    method: psi
    threshold: 0.2
    window: 7d
```

This is enough to start. The SDK will:
- Use PSI for data drift detection with threshold 0.2
- Default to local filesystem for the model registry
- Default to local JSON-lines for the audit trail
- Skip alerts (no channels configured)
- Skip deployment automation (no strategy configured)

---

## 4. Schema validation

The config schema is defined as **76 Pydantic model classes** in
`sentinel/config/schema.py`. Every YAML field maps to a typed Python
field with validation rules.

### What the schema enforces

| Validation type | Example |
|----------------|---------|
| **Type checking** | `drift.data.threshold` must be a `float`, not a string |
| **Allowed values** | `model.type` must be one of: `classification`, `regression`, `ranking`, `forecasting`, `generation` |
| **Range validation** | `drift.data.threshold` must be ≥ 0 |
| **Duration format** | `alerts.policies.cooldown` must match `\d+(ms\|s\|m\|h\|d\|w)` — e.g. `1h`, `30m`, `7d` |
| **Cross-field validation** | `model.type=forecasting` + `model.domain=graph` is rejected as invalid |
| **Strategy × target compatibility** | `canary` strategy + `azure_app_service` target is rejected (App Service slots don't support gradual traffic ramp) |
| **Extra fields rejected** | A typo like `threhold: 0.2` (instead of `threshold`) causes a validation error, not silent ignoring |
| **Secret masking** | `webhook_url` fields use `SecretStr` — never logged in plaintext |

### The schema class hierarchy

```
SentinelConfig (root)
├── ModelConfig
├── DataQualityConfig
│   ├── SchemaConfig
│   ├── FreshnessConfig
│   └── OutlierConfig
├── DriftConfig
│   ├── DataDriftConfig
│   ├── ConceptDriftConfig
│   ├── ModelDriftConfig
│   ├── DriftScheduleConfig
│   └── DriftAutoCheckConfig
├── FeatureHealthConfig
├── CohortAnalysisConfig
├── AlertsConfig
│   ├── ChannelConfig (list)
│   ├── AlertPolicies
│   └── EscalationStep (list)
├── RetrainingConfig
│   ├── ApprovalConfig
│   └── ValidationConfig
├── DeploymentConfig
│   ├── CanaryConfig
│   ├── ShadowConfig
│   ├── BlueGreenConfig
│   └── Target sub-configs (Azure ML, App Service, AKS, SageMaker, Vertex AI)
├── CostMonitorConfig
├── BusinessKPIConfig → KPIMapping (list)
├── AuditConfig
│   ├── AzureBlobAuditConfig
│   ├── S3AuditConfig
│   └── GcsAuditConfig
├── RegistryConfig
├── ModelGraphConfig → ModelGraphEdge (list)
├── LLMOpsConfig
│   ├── PromptRegistryConfig
│   ├── GuardrailsConfig → GuardrailRuleConfig (list)
│   ├── LLMQualityConfig
│   │   ├── QualityEvaluatorConfig
│   │   ├── SemanticDriftConfig
│   │   └── RetrievalQualityConfig
│   ├── TokenEconomicsConfig
│   └── PromptDriftConfig
├── AgentOpsConfig
│   ├── TracingConfig
│   ├── ToolAuditConfig
│   ├── SafetyConfig
│   │   ├── LoopDetectionConfig
│   │   ├── BudgetConfig
│   │   ├── EscalationConfig → EscalationTrigger (list)
│   │   └── SandboxConfig
│   ├── AgentRegistryConfig
│   ├── MultiAgentConfig
│   └── AgentEvaluationConfig
├── DashboardConfig
│   ├── DashboardServerConfig
│   │   ├── BearerAuthConfig
│   │   ├── RBACConfig → RBACUserBinding (list)
│   │   ├── CSRFConfig
│   │   ├── RateLimitConfig
│   │   └── CSPConfig
│   └── DashboardUIConfig
├── DomainConfig
├── DatasetConfig
└── ExperimentConfig
```

### Error messages

Validation errors include the full field path and, when using
`extends:`, the originating file:

```
ConfigValidationError: config validation failed:
  - drift.data.threshold: Input should be greater than or equal to 0
    [from production.yaml]
  - alerts.channels.0.type: Input should be 'slack', 'teams',
    'pagerduty', 'email' or 'webhook' [from base.yaml]
  - deployment.strategy: value is not a valid enumeration member
```

---

## 5. Environment variable substitution

Secrets and environment-specific values should never be hardcoded in
YAML. Use `${VAR_NAME}` syntax:

```yaml
alerts:
  channels:
    - type: slack
      webhook_url: ${SLACK_WEBHOOK_URL}
      channel: "#ml-alerts"
    - type: pagerduty
      routing_key: ${PD_ROUTING_KEY}
```

### Syntax

| Pattern | Behaviour |
|---------|-----------|
| `${VAR}` | Replace with `os.environ["VAR"]`. In lenient mode (default), unset vars are preserved as literal `${VAR}`. In strict mode, raises `ConfigMissingEnvVarError`. |
| `${VAR:-default}` | Replace with `os.environ["VAR"]` if set, otherwise use `default`. |

### Strict mode

By default, unresolved `${VAR}` tokens are silently preserved (for
backward compatibility). Use **strict mode** to catch missing variables
at config load time:

```bash
# CLI
sentinel config validate --strict

# Python
config = load_config("sentinel.yaml", strict_env=True)
```

Strict mode error:

```
ConfigMissingEnvVarError: alerts.channels.0.webhook_url references
unset environment variable(s): ${SLACK_WEBHOOK_URL}
```

### Where substitution applies

Substitution runs **after** `extends:` inheritance is resolved but
**before** Pydantic validation. This means:

1. A parent config can define `${VAR}` tokens that the child doesn't
   override — they resolve from the child's runtime environment.
2. Default values in `${VAR:-default}` syntax work across inheritance.
3. The signature verifier runs on the **post-substitution** payload, so
   signed configs work regardless of which machine resolves the env vars
   (as long as the final values match).

---

## 6. Cloud secret manager integration

For production deployments, secrets should come from a managed secret
store rather than environment variables. Sentinel supports three cloud
providers natively:

### Azure Key Vault

```yaml
alerts:
  channels:
    - type: slack
      webhook_url: ${azkv:https://my-vault.vault.azure.net/slack-webhook}
```

**Syntax:** `${azkv:<vault-url>/<secret-name>}`

Resolves via `azure.identity.DefaultAzureCredential` +
`azure.keyvault.secrets.SecretClient`. Requires the `[azure]` extra.
Multiple vaults are supported — the client is cached per vault URL.

### AWS Secrets Manager

```yaml
alerts:
  channels:
    - type: slack
      webhook_url: ${awssm:my-slack-webhook}
```

**Syntax:** `${awssm:<secret-name>}` or `${awssm:<secret-name>/<json-key>}`

The `/json-key` variant extracts a specific key from a JSON secret
value. Requires the `[aws]` extra.

### GCP Secret Manager

```yaml
alerts:
  channels:
    - type: slack
      webhook_url: ${gcpsm:my-project/slack-webhook}
```

**Syntax:** `${gcpsm:<project-id>/<secret-name>}`

Resolves via Application Default Credentials. Requires a GCP extra.

### Resolution order

All three providers are resolved at the same pipeline stage as `${VAR}`
env-var substitution — they are lazy-imported only when the specific
token prefix (`${azkv:`, `${awssm:`, `${gcpsm:`) is detected in a
string value.

---

## 7. Config inheritance (`extends:`)

Large organisations don't want every model team writing a 200-line YAML
from scratch. Config inheritance lets you define a base config with
org-wide defaults and override only what's model-specific.

### How it works

```yaml
# base.yaml — org-wide defaults
version: "1.0"
alerts:
  channels:
    - type: slack
      webhook_url: ${SLACK_WEBHOOK_URL}
      channel: "#ml-alerts"
  policies:
    cooldown: 1h
audit:
  storage: azure_blob
  retention_days: 2555
  compliance_frameworks: [fca_consumer_duty]
```

```yaml
# fraud_model.yaml — model-specific overrides
extends: base.yaml

model:
  name: claims_fraud_v2
  type: classification
drift:
  data:
    method: psi
    threshold: 0.15     # Stricter than org default (0.2)
```

**Merge rules:**
- Dicts are merged recursively — child keys override parent keys.
- Lists are replaced wholesale — the child's `channels:` list replaces
  the parent's, not appends to it.
- The `extends:` key itself is stripped before validation.

### Multi-level inheritance

Chains of any depth are supported:

```
org-base.yaml → team-base.yaml → model.yaml
```

### Cycle detection

Circular inheritance (`a.yaml → b.yaml → a.yaml`) is detected at load
time and raises `ConfigCircularInheritanceError` with the full chain:

```
ConfigCircularInheritanceError: circular config inheritance detected:
a.yaml → b.yaml → a.yaml
```

### Source tracing

When a validation error occurs in a config with `extends:`, the error
message includes which file the problematic field came from:

```
  - alerts.policies.cooldown: invalid interval format '1 hour'
    [from base.yaml]
```

---

## 8. Secret handling and `<REDACTED>` masking

Sensitive fields (webhook URLs, routing keys, passwords) use Pydantic's
`SecretStr` type. This ensures they are never accidentally logged or
serialised.

### Which fields are secrets

- `alerts.channels[].webhook_url`
- `alerts.channels[].routing_key`
- `dashboard.server.basic_auth_password`
- Any field typed as `SecretStr` in the schema

### Viewing config with masking

```bash
# Shows all secrets as <REDACTED>
sentinel config show

# Reveals actual values (use with caution)
sentinel config show --unmask
```

### In Python

```python
channel_config.webhook_url                    # → SecretStr('**********')
channel_config.webhook_url.get_secret_value() # → 'https://hooks.slack.com/...'
```

---

## 9. File reference validation

The config can point at several on-disk paths:

- `model.baseline_dataset` — baseline data for drift comparison
- `data_quality.schema.path` — JSON Schema for input validation
- `retraining.validation.holdout_dataset` — holdout data
- `audit.path` — audit trail directory

In strict mode (`sentinel config validate --strict`), these paths are
validated at load time:

```bash
sentinel config validate --strict
```

```
FileReferenceError: model.baseline_dataset: path
'data/baseline.parquet' does not exist
```

**Remote paths are skipped.** URIs with `s3://`, `azure://`, `gs://`,
`http://`, `https://`, `azureml://` schemes are not checked locally —
they require cloud credentials that are validated separately via
`sentinel cloud test`.

---

## 10. Config signing and verification

For regulated environments, configs can be cryptographically signed
so that any modification (accidental or malicious) is detected.

### Signing a config

```bash
# Sign using an HMAC key from an environment variable
export SENTINEL_SIGNING_KEY="your-256-bit-key"
sentinel config sign --config sentinel.yaml
# Creates sentinel.yaml.sig
```

### Verifying a signature

```bash
sentinel config verify-signature --config sentinel.yaml
# ✓ Signature valid (key fingerprint: abc123...)
```

### How it works

1. The loader resolves `extends:` chains and `${VAR}` substitutions.
2. The **resolved** config dict is canonicalised to JSON bytes.
3. An HMAC-SHA256 is computed over those bytes.
4. The signature (HMAC + key fingerprint + timestamp) is stored in a
   sidecar `.sig` file.

Because the signature is computed on the **resolved** config, it
survives `extends:` chains and environment variable substitution.
The same `.sig` file works on any machine as long as the final
resolved config produces the same bytes.

### Enforcing signed configs

In the dashboard config:

```yaml
dashboard:
  server:
    require_signed_config: true
```

The dashboard will refuse to start if the config's signature is
missing or invalid.

---

## 11. Source tracing for error messages

When using `extends:` chains, a validation error like
`drift.data.threshold: value too small` is frustrating — which file
defined that field?

Sentinel's **source tracing** system tags every leaf value with the
file it originated from during the `extends:` merge. When Pydantic
validation fails, the error message includes the source file:

```
config validation failed:
  - drift.data.threshold: Input should be >= 0 [from team-base.yaml]
  - model.name: Field required [from fraud_model.yaml]
```

This works for chains of any depth. The source map is built during the
inheritance resolution step and kept on the `ConfigLoader` instance
for programmatic access:

```python
loader = ConfigLoader("sentinel.yaml")
config = loader.load()
source_map = loader.source_map  # SourceMap with per-field origins
```

---

## 12. CLI commands for config management

| Command | What it does |
|---------|-------------|
| `sentinel init` | Generate a starter `sentinel.yaml` with comments |
| `sentinel config validate` | Validate config against the Pydantic schema |
| `sentinel config validate --strict` | Also check env vars are set + file paths exist |
| `sentinel config show` | Display the fully resolved config (secrets `<REDACTED>`) |
| `sentinel config show --unmask` | Display with secret values revealed |
| `sentinel config sign` | Sign the resolved config, creating a `.sig` sidecar |
| `sentinel config verify-signature` | Verify a config's `.sig` file |

### Example workflow

```bash
# 1. Generate a starter config
sentinel init
# Creates sentinel.yaml with commented template

# 2. Edit the config for your model
vim sentinel.yaml

# 3. Validate (catches typos, type errors, invalid combos)
sentinel config validate --strict

# 4. View the resolved config (check inheritance + env vars)
sentinel config show

# 5. Sign for production
sentinel config sign

# 6. Deploy — the dashboard enforces the signature
sentinel dashboard --config sentinel.yaml
```

---

## 13. The example configs

Sentinel ships 8 example configs in `configs/examples/` covering
different use cases:

| Config file | Lines | Demonstrates |
|------------|-------|-------------|
| `minimal.yaml` | 27 | Bare minimum — model name + PSI drift + Slack alerts |
| `insurance_fraud.yaml` | 142 | Full BFSI: drift + alerts + deployment + compliance + audit |
| `rag_claims_agent.yaml` | 113 | LLMOps: guardrails + prompt management + RAG quality + token economics |
| `multi_agent_underwriting.yaml` | 146 | AgentOps: tracing + tool audit + safety + multi-agent consensus |
| `demand_forecast.yaml` | 69 | Time series domain: seasonality-aware drift + forecast quality |
| `ner_entity_extraction.yaml` | 72 | NLP domain: vocabulary drift + embedding monitoring + NER metrics |
| `product_reco.yaml` | 86 | Recommendation domain: coverage + diversity + fairness |
| `fraud_graph.yaml` | 84 | Graph ML domain: topology drift + knowledge graph metrics |

### Using an example as a starting point

```bash
cp configs/examples/insurance_fraud.yaml sentinel.yaml
# Edit model name, alert channels, etc.
sentinel config validate --strict
```

---

## 14. How config maps to runtime behaviour

Every top-level config section maps to one or more SDK modules. The
`SentinelClient` reads the validated config and wires up the
appropriate components:

| Config section | SDK module | What it controls |
|---------------|-----------|-----------------|
| `model` | `SentinelClient` | Model identity, domain adapter selection |
| `data_quality` | `sentinel.observability.data_quality` | Schema validation, freshness checks, outlier detection |
| `drift.data` | `sentinel.observability.drift.data_drift` | Which statistical test (PSI/KS/JS/chi²/Wasserstein), threshold, window |
| `drift.concept` | `sentinel.observability.drift.concept_drift` | DDM/EDDM/ADWIN/Page-Hinkley, warning + drift levels |
| `drift.model` | `sentinel.observability.drift.model_drift` | Performance metrics, decay thresholds |
| `feature_health` | `sentinel.observability.feature_health` | Importance method (SHAP/permutation), top-N drift alerting |
| `alerts` | `sentinel.action.notifications` | Channel routing, cooldown, escalation chains, digest mode |
| `deployment` | `sentinel.action.deployment` | Strategy (canary/shadow/blue-green) + target (Azure/AWS/GCP) |
| `retraining` | `sentinel.action.retrain` | Trigger conditions, approval gates, validation rules |
| `audit` | `sentinel.foundation.audit` | Storage backend, retention, tamper evidence, compliance reports |
| `registry` | `sentinel.foundation.registry` | Backend (local/Azure ML/MLflow/SageMaker/Vertex AI/Databricks) |
| `llmops` | `sentinel.llmops` | Prompt versioning, guardrail pipeline, quality scoring, cost tracking |
| `agentops` | `sentinel.agentops` | Tracing backend, tool permissions, safety budgets, escalation triggers |
| `domains` | `sentinel.domains` | Domain-specific drift detectors, quality metrics, schema validators |
| `dashboard` | `sentinel.dashboard` | Server config, auth, RBAC, UI preferences |

### Example: how `drift.data` becomes a detector

```yaml
drift:
  data:
    method: psi
    threshold: 0.2
    window: 7d
```

At client init time, `SentinelClient` reads `config.drift.data.method`,
looks up `"psi"` in the drift detector registry, and instantiates a
`PSIDriftDetector(threshold=0.2)`. When you call `client.check_drift()`,
the detector compares the last 7 days of logged predictions against the
baseline stored in the model registry.

### Example: how `alerts.channels` becomes notification dispatch

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
      - after: 0m
        channels: [slack]
        severity: [high, critical]
      - after: 30m
        channels: [slack, pagerduty]
        severity: [critical]
```

The `NotificationEngine` reads this config and:
1. Instantiates a `SlackChannel` and a `PagerDutyChannel`.
2. When a drift alert fires, routes it to Slack immediately.
3. Suppresses duplicate alerts for the same issue within 1 hour (cooldown).
4. If the alert is critical and not acknowledged within 30 minutes,
   escalates to PagerDuty.

---

## 15. Best practices

### For individual developers

1. **Start from an example.** Copy the closest example config and
   modify it. Don't write from scratch.
2. **Use strict validation.** Always run
   `sentinel config validate --strict` before deploying.
3. **Never hardcode secrets.** Use `${VAR}` or `${azkv:...}` for
   all webhook URLs, API keys, and routing keys.
4. **Set `domain:` explicitly.** Even though `tabular` is the default,
   being explicit makes the config self-documenting.

### For teams and organisations

1. **Use config inheritance.** Define a `base.yaml` with org-wide
   defaults (alert channels, compliance frameworks, audit retention).
   Model configs inherit via `extends: base.yaml`.
2. **Sign production configs.** Use `sentinel config sign` and enforce
   `require_signed_config: true` in the dashboard.
3. **Version-control configs.** Configs should live next to the model
   code. `git blame` is your audit trail for config changes.
4. **Use cloud secret managers in production.** `${azkv:...}` for
   Azure, `${awssm:...}` for AWS, `${gcpsm:...}` for GCP.

### Config inheritance pattern for multi-model orgs

```
configs/
├── base.yaml                    # Org-wide: alerts, audit, compliance
├── team-ml/
│   ├── team-base.yaml           # extends: ../base.yaml — team defaults
│   ├── fraud_model.yaml         # extends: team-base.yaml
│   └── churn_model.yaml         # extends: team-base.yaml
└── team-llm/
    ├── team-base.yaml           # extends: ../base.yaml — LLMOps defaults
    ├── claims_rag.yaml          # extends: team-base.yaml
    └── underwriting_agent.yaml  # extends: team-base.yaml
```

Each model config is 20–40 lines. All the boilerplate (alert channels,
audit storage, compliance frameworks) lives in the base configs and is
shared across all models.

---

## Source files

| File | Purpose |
|------|---------|
| `sentinel/config/schema.py` | 76 Pydantic model classes — the config schema |
| `sentinel/config/loader.py` | YAML/JSON parser, env-var substitution, inheritance, validation |
| `sentinel/config/defaults.py` | `sentinel init` template, default pricing tables |
| `sentinel/config/source.py` | Per-field source tracing for error messages |
| `sentinel/config/references.py` | File-path validation (`--strict` mode) |
| `sentinel/config/secrets.py` | `SecretStr` wrappers and `unwrap()` helper |
| `sentinel/config/signing.py` | Config signing and verification |
| `sentinel/config/keyvault.py` | Azure Key Vault `${azkv:...}` resolver |
| `sentinel/config/aws_secrets.py` | AWS Secrets Manager `${awssm:...}` resolver |
| `sentinel/config/gcp_secrets.py` | GCP Secret Manager `${gcpsm:...}` resolver |
