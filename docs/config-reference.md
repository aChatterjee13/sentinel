# Configuration Reference

> Every Sentinel behaviour is defined in YAML. This document is the
> authoritative field-by-field reference for all configuration sections.
>
> The schema is enforced by `sentinel.config.schema.SentinelConfig`
> (Pydantic v2, `extra="forbid"`). Unknown fields cause a validation
> error, so typos are caught at load time. Run
> `sentinel config validate --config sentinel.yaml` to check a config
> without starting any monitoring. Add `--strict` to also fail on
> unset env vars and missing referenced files.

---

## Quick Reference

A minimal working config — only `model.name` is required:

```yaml
version: "1.0"
model:
  name: my_model
```

A realistic minimal config with drift detection and Slack alerting:

```yaml
version: "1.0"
model:
  name: claims_fraud_v2
  type: classification
  framework: xgboost
  baseline_dataset: data/baseline.parquet

drift:
  data:
    method: psi
    threshold: 0.2
    window: 7d

alerts:
  channels:
    - type: slack
      webhook_url: ${SLACK_WEBHOOK_URL}
      channel: "#ml-alerts"
  policies:
    cooldown: 1h
```

---

## Top-Level Structure

```yaml
version: "1.0"               # required, currently always "1.0"
extends: base.yaml           # optional, inherit from another config

model: { ... }               # required — identifies the model
data_quality: { ... }        # schema enforcement, freshness, outliers
drift: { ... }               # data, concept, model drift detection
feature_health: { ... }      # per-feature drift weighted by importance
cohort_analysis: { ... }     # sub-population monitoring
alerts: { ... }              # notification channels and routing policies
retraining: { ... }          # retrain trigger, approval, validation
deployment: { ... }          # deployment strategy, target, rollback
cost_monitor: { ... }        # latency, throughput, cost tracking
business_kpi: { ... }        # model metric → business KPI mapping
audit: { ... }               # immutable audit trail configuration
registry: { ... }            # model registry backend
model_graph: { ... }         # multi-model dependency DAG
llmops: { ... }              # LLM monitoring and governance
agentops: { ... }            # agent tracing, safety, tool audit
dashboard: { ... }           # optional UI configuration
domains: { ... }             # domain-specific adapter overrides
datasets: { ... }            # dataset metadata registry
experiments: { ... }         # experiment tracker settings
```

Only `model` is required. Every other section has sensible defaults —
you only specify what you want to override.

---

## Environment Variable Substitution

`${VAR_NAME}` is replaced at load time with the value of the environment variable. Used everywhere — webhook URLs, cloud credentials, file paths. Default values are supported with `${VAR_NAME:-default}` syntax.

```yaml
alerts:
  channels:
    - type: slack
      webhook_url: ${SLACK_WEBHOOK_URL}
audit:
  retention_days: ${AUDIT_RETENTION:-2555}    # default 7 years
```

In **lenient mode** (default for `sentinel validate`), an unresolved `${VAR}` is left as a literal string, which usually fails later when something tries to use it. In **strict mode** (`sentinel config validate --strict`), the loader raises `ConfigMissingEnvVarError` with the JSON path of the offending field. Empty values (`VAR=`) are also treated as unset in strict mode — an empty webhook URL is just as broken as a missing one.

### Azure Key Vault References

A second substitution pass runs alongside env-var substitution: `${azkv:vault-name/secret-name}` is resolved against Azure Key Vault at config-load time.

```yaml
alerts:
  channels:
    - type: slack
      webhook_url: ${azkv:sentinel-prod/slack-webhook}
dashboard:
  server:
    basic_auth_password: ${azkv:sentinel-prod/dashboard-password}
```

- **Vault name:** 3–24 chars, alphanumeric + hyphens, must start and end with an alphanumeric character (Azure Key Vault naming rules).
- **Secret name:** 1+ chars, alphanumeric + hyphens.
- **Authentication:** always `azure.identity.DefaultAzureCredential` — `az login` locally, Managed Identity / Workload Identity in production.
- **Required RBAC:** the runtime identity needs the `Key Vault Secrets User` role on the vault.
- **Versioning:** only the latest secret version is fetched. `${azkv:vault/secret/version}` is not supported.
- **Caching:** the `SecretClient` is cached per vault URL, and each resolved secret is cached per `(vault, secret)` key for the process lifetime.
- **Strict vs lenient:** strict raises `ConfigKeyVaultError`; lenient preserves the literal token (matches env-var behaviour).

Resolution happens **before** Pydantic validation, so `SecretStr` wrapping, `<REDACTED>` masking in `sentinel config show`, and strict-mode file reference checks all work transparently. Install the `[azure]` extra — `pip install "sentinel-mlops[azure]"` — before using `${azkv:…}`. See [`docs/azure.md`](azure.md#1-azure-key-vault-secret-resolution) for the full flow and troubleshooting.

---

## Config Inheritance (`extends:`)

```yaml
extends: configs/base.yaml
model:
  name: claims_fraud_v2     # overrides base
```

Fields in the child config override the parent. Lists are replaced, not merged. The loader detects circular inheritance (`a → b → a`) and raises `ConfigCircularInheritanceError` with the full chain in the error message. Multi-level chains (`grandparent → parent → child`) are supported and validation errors include the originating file via the `[from <file> (via chain)]` suffix, making it easy to track down which file in the chain owns a misconfigured field.

---

## Secrets and `<REDACTED>` Masking

The fields below are stored internally as `pydantic.SecretStr` so they never leak into logs, error messages, or `model_dump()` output:

| Field | Where |
|---|---|
| `alerts.channels[].webhook_url` | Slack, Teams, generic webhook channels |
| `alerts.channels[].routing_key` | PagerDuty channel |
| `dashboard.server.basic_auth_password` | Dashboard HTTP Basic auth |

Use `sentinel config show` to inspect a fully-resolved config with secrets rendered as `<REDACTED>`. Pass `--unmask` only when you know what you are doing — it prints the plaintext to stdout and emits a stderr warning. Use `--format json` for machine-readable output.

```bash
# Inspect resolved config (secrets masked)
sentinel config show --config sentinel.yaml

# Reveal secrets (audit-logged via stderr warning)
sentinel config show --config sentinel.yaml --unmask --format json
```

---

## File Reference Validation

`sentinel config validate --strict` also checks that path-like config fields point to files that actually exist on disk:

| Field | Severity if missing |
|---|---|
| `model.baseline_dataset` | error |
| `data_quality.schema.path` | error |
| `retraining.validation.holdout_dataset` | error |
| `retraining.pipeline` | warning |
| `audit.path` (parent dir must be writable) | error |

Remote URIs (`s3://`, `azure://`, `azureml://`, `gs://`, `https://`, `abfss://`, `wasbs://`, `warehouse://`) are skipped — only local paths are checked. In **lenient mode**, missing files are emitted as warnings to stderr but the command still exits 0 so the dev loop stays fast.

---

## Duration / Interval Strings

Many fields accept a duration string. The format is a number followed by a unit suffix:

| Suffix | Unit | Example |
|--------|------|---------|
| `ms` | milliseconds | `500ms` |
| `s` | seconds | `300s` |
| `m` | minutes | `30m` |
| `h` | hours | `1h` |
| `d` | days | `7d` |
| `w` | weeks | `1w` |

Fractional values are supported (e.g. `1.5h`). Invalid formats cause a validation error at load time with the message: *"Invalid interval format 'X'. Expected a number followed by a unit: ms, s, m, h, d, or w"*.

---

## Section Reference

### `version`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `version` | str | `"1.0"` | Config schema version. Must be a non-empty string. Currently always `"1.0"`. |

### `extends`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `extends` | str \| null | `null` | Path to a parent config to inherit from. See [Config Inheritance](#config-inheritance-extends). |

---

### `model`

Identifies and describes the model being monitored. **This is the only required section.**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | *(required)* | Stable model identifier. Used in audit logs, alerts, and registry. |
| `type` | `"classification"` \| `"regression"` \| `"ranking"` \| `"forecasting"` \| `"generation"` | `"classification"` | Model task type. Affects which drift/quality metrics are applicable. |
| `framework` | str \| null | `null` | Model framework (e.g. `xgboost`, `pytorch`, `sklearn`). Auto-detected if not set. |
| `version` | str \| null | `null` | Current model version (semver recommended). |
| `domain` | `"tabular"` \| `"timeseries"` \| `"nlp"` \| `"recommendation"` \| `"graph"` | `"tabular"` | Selects the domain adapter for drift detection and quality metrics. |
| `baseline_dataset` | str \| null | `null` | Path or URI to a reference dataset used to fit drift baselines. Validated in strict mode. |
| `description` | str \| null | `null` | Free-text human-readable description. |

**Validation rules:**
- `name` is required; omitting it causes a validation error.
- Certain `type` + `domain` combinations are rejected:
  - `ranking` + `timeseries` ❌
  - `forecasting` + `graph` ❌
  - `generation` + `timeseries` ❌
  - `generation` + `graph` ❌

```yaml
model:
  name: claims_fraud_v2
  type: classification
  framework: xgboost
  version: "2.3.1"
  domain: tabular
  baseline_dataset: s3://bucket/baselines/fraud_v2.parquet
  description: "BFSI fraud classifier — UK retail banking"
```

---

### `data_quality`

Schema enforcement, freshness monitoring, and outlier detection on incoming features.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `schema.enforce` | bool | `true` | Whether to enforce schema validation on incoming data. |
| `schema.path` | str \| null | `null` | Path to a JSON Schema file for input validation. Validated in strict mode. |
| `freshness.max_age_hours` | int | `24` | Alert if no new data arrives within this many hours. |
| `outlier_detection.method` | `"isolation_forest"` \| `"zscore"` \| `"iqr"` | `"iqr"` | Outlier detection algorithm. |
| `outlier_detection.contamination` | float | `0.05` | Expected proportion of outliers (for isolation_forest). |
| `null_threshold` | float | `0.1` | Alert if any column exceeds this null fraction. |
| `duplicate_threshold` | float | `0.05` | Alert if the duplicate row fraction exceeds this value. |

> **Note:** The YAML key is `schema` (no underscore). Internally it maps to the `schema_` Python attribute via a Pydantic alias.

```yaml
data_quality:
  schema:
    enforce: true
    path: schemas/claims_v2.json
  freshness:
    max_age_hours: 24
  outlier_detection:
    method: isolation_forest
    contamination: 0.05
  null_threshold: 0.10
  duplicate_threshold: 0.05
```

---

### `drift`

Three independent drift subsystems plus scheduling. Data drift is always available; concept drift requires ground-truth labels; model drift requires metric streams.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `data` | `DataDriftConfig` | *(see below)* | Statistical tests on input feature distributions. |
| `concept` | `ConceptDriftConfig` \| null | `null` | Concept drift detection (requires actuals). Disabled if omitted. |
| `model` | `ModelDriftConfig` \| null | `null` | Performance decay tracking. Disabled if omitted. |
| `schedule` | `DriftScheduleConfig` | *(see below)* | Background periodic drift checking. |
| `auto_check` | `DriftAutoCheckConfig` | *(see below)* | Count-based automatic drift checking. |

#### `drift.data`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `method` | `"psi"` \| `"ks"` \| `"js_divergence"` \| `"chi_squared"` \| `"wasserstein"` | `"psi"` | Statistical test for data drift. |
| `threshold` | float | `0.2` | Drift significance threshold (interpretation depends on method). |
| `window` | str | `"7d"` | Sliding window for the current distribution. Duration string. |
| `reference` | `"baseline"` \| `"previous_window"` \| `"custom"` | `"baseline"` | What to compare the current window against. |
| `features.include` | list[str] \| `"all"` | `"all"` | Features to include in drift checks. |
| `features.exclude` | list[str] | `[]` | Features to exclude from drift checks. |

**Method guidance:**

| Method | Best for | Threshold guidance |
|--------|----------|-------------------|
| `psi` | Binned continuous + categorical | < 0.1 stable, 0.1–0.2 moderate, > 0.2 significant |
| `ks` | Continuous features | p-value < 0.05 |
| `js_divergence` | Probability distributions | > 0.1 investigate |
| `chi_squared` | Categorical features | p-value < 0.05 |
| `wasserstein` | Continuous, shape-sensitive | Domain-specific |

#### `drift.concept`

Optional — set to `null` (or omit entirely) to disable concept drift detection.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `method` | `"ddm"` \| `"eddm"` \| `"adwin"` \| `"page_hinkley"` | `"ddm"` | Concept drift detection algorithm. |
| `warning_level` | float | `2.0` | Standard deviations for warning threshold. |
| `drift_level` | float | `3.0` | Standard deviations for drift threshold. |
| `min_samples` | int | `100` | Minimum samples before detection activates. |
| `requires_actuals` | bool | `true` | Flags that ground-truth labels are needed. |

#### `drift.model`

Optional — set to `null` (or omit entirely) to disable model drift detection.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `metrics` | list[str] | `["accuracy", "f1"]` | Performance metrics to track. |
| `threshold` | dict[str, float] | `{"accuracy": 0.05}` | Per-metric maximum acceptable degradation from baseline. |
| `evaluation_window` | int | `1000` | Number of predictions in the evaluation window. |

#### `drift.schedule`

Background scheduler for periodic drift detection.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Toggle the background scheduler. |
| `interval` | str | `"7d"` | How often to run drift checks. Duration string. |
| `run_on_start` | bool | `false` | Run a drift check immediately on scheduler start. |

#### `drift.auto_check`

Count-based automatic drift checking triggered by prediction volume.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Toggle count-based auto-checking. |
| `every_n_predictions` | int | `1000` | Run `check_drift()` after this many predictions. |

When `enabled: true`, the counter increments on every `log_prediction()` call. Once both the counter and the prediction buffer reach `every_n_predictions`, a daemon thread runs `check_drift()` and the counter resets. Manual calls to `check_drift()` also reset the counter.

```yaml
drift:
  data:
    method: psi
    threshold: 0.2
    window: 7d
    reference: baseline
    features:
      include: all
      exclude: [timestamp, id]
  concept:
    method: ddm
    warning_level: 2.0
    drift_level: 3.0
    min_samples: 100
  model:
    metrics: [accuracy, f1, auc]
    threshold:
      accuracy: 0.05
      f1: 0.08
    evaluation_window: 1000
  schedule:
    enabled: true
    interval: 1d
    run_on_start: true
  auto_check:
    enabled: true
    every_n_predictions: 500
```

---

### `feature_health`

Per-feature drift monitoring weighted by feature importance.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `importance_method` | `"shap"` \| `"permutation"` \| `"builtin"` | `"builtin"` | Method for calculating feature importance. `builtin` uses the model's native importance (e.g. XGBoost gain). |
| `alert_on_top_n_drift` | int | `3` | Alert if any of the top-N most important features drift. |
| `recalculate_importance` | `"never"` \| `"daily"` \| `"weekly"` \| `"monthly"` | `"weekly"` | How often to recalculate feature importance scores. |

```yaml
feature_health:
  importance_method: shap
  alert_on_top_n_drift: 5
  recalculate_importance: weekly
```

---

### `cohort_analysis`

Sub-population monitoring. When enabled, predictions logged via `SentinelClient.log_prediction()` can carry a `cohort_id` to segment monitoring by sub-population.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Toggle cohort analysis on/off. |
| `cohort_column` | str \| null | `null` | Default feature name to derive cohort IDs from. **Required when `enabled: true`.** |
| `max_cohorts` | int | `50` | Upper limit on tracked cohorts to bound memory. |
| `min_samples_per_cohort` | int | `30` | Minimum predictions per cohort before drift/disparity checks activate. |
| `disparity_threshold` | float | `0.10` | Maximum acceptable relative performance gap between a cohort and the global mean (0.10 = 10%). |
| `buffer_size` | int | `1000` | Rolling prediction buffer per cohort. |

**Validation rules:**
- When `enabled: true`, `cohort_column` must be set (non-null, non-empty). Omitting it raises a validation error: *"cohort_column is required when cohort_analysis is enabled"*.

```yaml
cohort_analysis:
  enabled: true
  cohort_column: customer_segment
  max_cohorts: 20
  min_samples_per_cohort: 50
  disparity_threshold: 0.15
  buffer_size: 2000
```

---

### `alerts`

Notification channels and routing policies.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `channels` | list[ChannelConfig] | `[]` | List of notification channel configurations. |
| `policies` | `AlertPolicies` | *(see below)* | Alert routing, cooldown, and escalation policies. |

#### `alerts.channels[]`

Each channel is a dict with `type` and channel-specific fields. The `ChannelConfig` model uses `extra="allow"` so channels can carry custom fields for pluggable implementations.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | `"slack"` \| `"teams"` \| `"pagerduty"` \| `"email"` \| `"webhook"` | *(required)* | Channel type. |
| `webhook_url` | SecretStr \| null | `null` | Webhook URL. **Required for `slack`, `teams`, `webhook`.** Redacted in `config show`. |
| `routing_key` | SecretStr \| null | `null` | PagerDuty routing key. **Required for `pagerduty`.** Redacted in `config show`. |
| `channel` | str \| null | `null` | Slack channel name (e.g. `"#ml-alerts"`). |
| `recipients` | list[str] | `[]` | Email recipients. **Required for `email`** (at least one). |
| `severity_mapping` | dict[str, str] | `{}` | Map Sentinel severity levels to channel-specific levels. |
| `enabled` | bool | `true` | Toggle this channel on/off without removing its config. |
| `template` | str \| null | `null` | Optional Jinja2 template for message formatting (per-channel override). |

**Validation rules:**
- `slack`, `teams`, `webhook` require a non-empty `webhook_url`.
- `pagerduty` requires `routing_key`.
- `email` requires at least one entry in `recipients`.

#### `alerts.policies`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cooldown` | str | `"1h"` | Suppress repeated alerts for the same issue within this window. Duration string. |
| `digest_mode` | bool | `false` | Batch alerts into periodic digests instead of firing each individually. |
| `digest_interval` | str | `"6h"` | How often to send digest summaries (when `digest_mode: true`). Duration string. |
| `rate_limit_per_hour` | int | `60` | Maximum alerts per hour per channel. |
| `rate_limit_window` | str | `"1h"` | Window for rate limiting. Duration string. |
| `default_template` | str \| null | `null` | Global Jinja2 template applied to all channels (overridden by per-channel `template`). |
| `escalation` | list[EscalationStep] | `[]` | Ordered escalation chain. |

#### `alerts.policies.escalation[]` (EscalationStep)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `after` | str | `"0m"` | Time after initial alert before this step activates. Duration string. |
| `channels` | list[str] | `[]` | Channel types to notify at this step. |
| `severity` | list[str] | `[]` | Only escalate alerts of these severity levels. |

```yaml
alerts:
  channels:
    - type: slack
      webhook_url: ${SLACK_WEBHOOK_URL}
      channel: "#ml-alerts"
      template: "templates/slack_alert.j2"
    - type: teams
      webhook_url: ${TEAMS_WEBHOOK_URL}
    - type: pagerduty
      routing_key: ${PD_ROUTING_KEY}
      severity_mapping:
        critical: critical
        high: error
        medium: warning
    - type: email
      recipients: [ml-team@company.com, risk@company.com]
  policies:
    cooldown: 1h
    rate_limit_per_hour: 60
    rate_limit_window: 1h
    digest_mode: false
    digest_interval: 6h
    default_template: "templates/default_alert.j2"
    escalation:
      - after: 0m
        channels: [slack]
        severity: [warning, high, critical]
      - after: 30m
        channels: [slack, teams]
        severity: [high, critical]
      - after: 2h
        channels: [slack, teams, pagerduty]
        severity: [critical]
```

---

### `retraining`

Drift → trigger → validate → human approval → promote → (optionally) deploy pipeline.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `trigger` | `"drift_confirmed"` \| `"scheduled"` \| `"manual"` | `"drift_confirmed"` | What triggers a retraining run. |
| `pipeline` | str \| null | `null` | Pipeline URI or local script path (e.g. `azureml://pipelines/retrain_fraud`). |
| `schedule` | str \| null | `null` | Cron expression for scheduled retrains (when `trigger: scheduled`). |
| `deploy_on_promote` | bool | `false` | Auto-deploy after model promotion. Requires a `deployment` section. |
| `approval` | `ApprovalConfig` | *(see below)* | Human-in-the-loop approval settings. |
| `validation` | `ValidationConfig` | *(see below)* | Retrained model validation settings. |

When `deploy_on_promote: true`, the `RetrainOrchestrator` automatically calls `DeploymentManager.start()` after a model is promoted (whether auto-approved or manually approved). The deployment result is included in the orchestrator's return value and logged to the audit trail.

#### `retraining.approval`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `"auto"` \| `"human_in_loop"` \| `"hybrid"` | `"human_in_loop"` | Approval mode. `hybrid` auto-promotes if thresholds are met, otherwise requires human approval. |
| `approvers` | list[str] | `[]` | Email addresses of approvers (for `human_in_loop` and `hybrid`). |
| `auto_promote_if` | dict[str, Any] | `{}` | Conditions for auto-promotion in `hybrid` mode (e.g. `{metric: f1, improvement_pct: 2.0}`). |
| `timeout` | str | `"48h"` | Auto-reject if no approval within this window. Duration string. |

#### `retraining.validation`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `holdout_dataset` | str \| null | `null` | Path or URI to a holdout dataset for validation. Validated in strict mode. |
| `min_performance` | dict[str, float] | `{}` | Minimum acceptable performance thresholds (e.g. `{accuracy: 0.85, f1: 0.80}`). |

```yaml
retraining:
  trigger: drift_confirmed
  pipeline: azureml://pipelines/retrain_fraud
  deploy_on_promote: true
  approval:
    mode: hybrid
    approvers: [ml-team@company.com]
    auto_promote_if:
      metric: f1
      improvement_pct: 2.0
    timeout: 48h
  validation:
    holdout_dataset: s3://bucket/holdout.parquet
    min_performance:
      accuracy: 0.85
      f1: 0.80
```

---

### `deployment`

Deployment strategy, target environment, and rollback rules.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `strategy` | `"shadow"` \| `"canary"` \| `"blue_green"` \| `"direct"` | `"canary"` | Deployment strategy. |
| `target` | `"local"` \| `"azure_ml_endpoint"` \| `"azure_app_service"` \| `"aks"` \| `"sagemaker_endpoint"` \| `"vertex_ai_endpoint"` | `"local"` | Deployment target environment. |
| `canary` | `CanaryConfig` | *(see below)* | Canary-specific settings. |
| `shadow` | `ShadowConfig` | *(see below)* | Shadow-specific settings. |
| `blue_green` | `BlueGreenConfig` | *(see below)* | Blue/green-specific settings. |
| `azure_ml_endpoint` | `AzureMLEndpointTargetConfig` \| null | `null` | Required when `target: azure_ml_endpoint`. |
| `azure_app_service` | `AzureAppServiceTargetConfig` \| null | `null` | Required when `target: azure_app_service`. |
| `aks` | `AKSDeploymentTargetConfig` \| null | `null` | Required when `target: aks`. |
| `sagemaker_endpoint` | `SageMakerEndpointTargetConfig` \| null | `null` | Required when `target: sagemaker_endpoint`. |
| `vertex_ai_endpoint` | `VertexAIEndpointTargetConfig` \| null | `null` | Required when `target: vertex_ai_endpoint`. |

**Validation rules:**
- When `target` is not `local`, the corresponding sub-config must be provided.
- The strategy × target combination must be compatible (see matrix below).

#### Strategy × Target Compatibility Matrix

| Strategy | `local` | `azure_ml_endpoint` | `azure_app_service` | `aks` | `sagemaker_endpoint` | `vertex_ai_endpoint` |
|----------|---------|---------------------|---------------------|-------|----------------------|----------------------|
| `shadow` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `canary` | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| `blue_green` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `direct` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

`canary` + `azure_app_service` is rejected at validation time because App Service slot-traffic percentage routing is brittle — use `blue_green` (slot swap) instead. The error message includes both the strategy and target values and lists compatible targets.

#### `deployment.canary`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `initial_traffic_pct` | int | `5` | Initial percentage of traffic routed to the new model. |
| `ramp_steps` | list[int] | `[5, 25, 50, 100]` | Traffic percentage at each ramp step. |
| `ramp_interval` | str | `"1h"` | Time between ramp steps. Duration string. |
| `rollback_on` | dict[str, float] | `{}` | Auto-rollback conditions (e.g. `{error_rate_increase: 0.02, latency_p99_increase_ms: 50}`). |

#### `deployment.shadow`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `duration` | str | `"24h"` | How long the shadow deployment runs. Duration string. |
| `log_predictions` | bool | `true` | Whether to log shadow model predictions for comparison. |

#### `deployment.blue_green`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `health_check_url` | str \| null | `null` | Health check endpoint to probe before switching traffic. |
| `warmup_seconds` | int | `30` | Seconds to wait after deploying before switching traffic. |

#### `deployment.azure_ml_endpoint`

Required when `target: azure_ml_endpoint`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `endpoint_name` | str | *(required, min 1 char)* | Azure ML managed online endpoint name. |
| `subscription_id` | str | *(required, min 1 char)* | Azure subscription ID. |
| `resource_group` | str | *(required, min 1 char)* | Azure resource group name. |
| `workspace_name` | str | *(required, min 1 char)* | Azure ML workspace name. |
| `deployment_name_pattern` | str | `"{model_name}-{version}"` | Pattern for deployment names. Supports `{model_name}` and `{version}` placeholders. |

#### `deployment.azure_app_service`

Required when `target: azure_app_service`. Compatible with `shadow`, `blue_green`, and `direct` strategies only.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `subscription_id` | str | *(required)* | Azure subscription ID. |
| `resource_group` | str | *(required)* | Azure resource group name. |
| `site_name` | str | *(required)* | Azure App Service site name. |
| `production_slot` | str | `"production"` | Name of the production slot. |
| `staging_slot` | str | `"staging"` | Name of the staging slot. |
| `health_check_path` | str | `"/healthz"` | Health check endpoint path on the app service. |

#### `deployment.aks`

Required when `target: aks`. Canary traffic granularity is bounded by `replicas_total` — with the default `10`, the smallest non-zero canary step is 10%.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `namespace` | str | *(required)* | Kubernetes namespace. |
| `service_name` | str | *(required)* | Kubernetes service name. |
| `deployment_name_pattern` | str | `"{model_name}-{version}"` | Pattern for Kubernetes deployment names. |
| `replicas_total` | int | `10` | Total replica count across champion + challenger. |
| `kubeconfig_path` | str \| null | `null` | Path to kubeconfig file. Uses default context if not set. |

#### `deployment.sagemaker_endpoint`

Required when `target: sagemaker_endpoint`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `endpoint_name` | str | *(required)* | SageMaker endpoint name. |
| `region_name` | str \| null | `null` | AWS region. Uses default region if not set. |
| `variant_name_pattern` | str | `"{model_name}-{version}"` | Pattern for production variant names. |

#### `deployment.vertex_ai_endpoint`

Required when `target: vertex_ai_endpoint`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `endpoint_name` | str | *(required)* | Vertex AI endpoint name. |
| `project` | str | *(required)* | GCP project ID. |
| `location` | str | `"us-central1"` | GCP region. |

```yaml
# Canary on Azure ML Online Endpoints
deployment:
  strategy: canary
  target: azure_ml_endpoint
  canary:
    initial_traffic_pct: 5
    ramp_steps: [5, 25, 50, 100]
    ramp_interval: 30m
    rollback_on:
      error_rate_increase: 0.02
      latency_p99_increase_ms: 50
  azure_ml_endpoint:
    endpoint_name: claims-fraud-endpoint
    subscription_id: ${AZURE_SUBSCRIPTION_ID}
    resource_group: sentinel-prod-rg
    workspace_name: sentinel-prod-ws

# Blue/green on Azure App Service (slot swap)
deployment:
  strategy: blue_green
  target: azure_app_service
  blue_green:
    warmup_seconds: 60
  azure_app_service:
    subscription_id: ${AZURE_SUBSCRIPTION_ID}
    resource_group: sentinel-prod-rg
    site_name: claims-fraud-api
    production_slot: production
    staging_slot: staging
    health_check_path: /healthz

# Canary on AKS (replica scaling)
deployment:
  strategy: canary
  target: aks
  canary:
    ramp_steps: [10, 30, 60, 100]
    ramp_interval: 15m
  aks:
    namespace: ml-prod
    service_name: claims-fraud
    replicas_total: 10

# Direct deploy to SageMaker
deployment:
  strategy: direct
  target: sagemaker_endpoint
  sagemaker_endpoint:
    endpoint_name: claims-fraud-prod
    region_name: us-east-1

# Shadow on Vertex AI
deployment:
  strategy: shadow
  target: vertex_ai_endpoint
  shadow:
    duration: 48h
    log_predictions: true
  vertex_ai_endpoint:
    endpoint_name: claims-fraud-shadow
    project: my-gcp-project
    location: us-central1
```

See [`docs/azure.md`](azure.md#4-azure-ml-online-endpoints-deployment-target) for RBAC setup and troubleshooting per target.

---

### `cost_monitor`

Latency, throughput, and cost-per-prediction tracking with configurable alert thresholds.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `track` | list[str] | `["inference_latency_ms", "throughput_rps", "cost_per_prediction", "compute_utilisation_pct"]` | Metrics to track. See known metrics below. |
| `alert_thresholds` | dict[str, float] | `{}` | Per-metric alert thresholds. Keys must be known metrics. |

**Known metrics** (the only keys allowed in `alert_thresholds`):

| Metric | Description |
|--------|-------------|
| `inference_latency_ms` | End-to-end inference latency in milliseconds |
| `latency_p99_ms` | 99th percentile latency in milliseconds |
| `throughput_rps` | Requests per second |
| `cost_per_prediction` | Cost per individual prediction |
| `cost_per_1k_predictions` | Cost per 1000 predictions |
| `compute_utilisation_pct` | Compute utilisation percentage |
| `error_rate` | Prediction error rate |

**Validation rules:**
- Keys in `alert_thresholds` are validated against the known metrics set. Unknown metric names cause a validation error listing the unknown keys and the full set of known metrics.

```yaml
cost_monitor:
  track:
    - inference_latency_ms
    - throughput_rps
    - cost_per_prediction
    - error_rate
  alert_thresholds:
    latency_p99_ms: 200
    cost_per_1k_predictions: 5.00
    error_rate: 0.05
```

---

### `business_kpi`

Maps model metrics to business outcomes for impact reporting.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mappings` | list[KPIMapping] | `[]` | List of model metric → business KPI mappings. |

#### `business_kpi.mappings[]` (KPIMapping)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_metric` | str | *(required)* | Model metric name (e.g. `precision`, `recall`). |
| `business_kpi` | str | *(required)* | Business KPI name (e.g. `fraud_catch_rate`). |
| `data_source` | str \| null | `null` | URI for the business metric data source. |

```yaml
business_kpi:
  mappings:
    - model_metric: precision
      business_kpi: fraud_catch_rate
      data_source: warehouse://analytics.fraud_metrics
    - model_metric: recall
      business_kpi: false_positive_rate
```

---

### `audit`

Immutable audit trail. Required for regulated deployments. The trail is **always local-first** — every event is written to `audit.path` with its HMAC-SHA256 signature before any remote shipping happens. A shipper rotates completed daily files to cloud storage on day rotation without touching the hash chain.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `storage` | `"local"` \| `"azure_blob"` \| `"s3"` \| `"gcs"` | `"local"` | Storage backend. Non-local backends also require the matching sub-config. |
| `path` | str | `"./audit/"` | Local directory for audit log files. |
| `retention_days` | int | `2555` | Days to retain audit records (2555 ≈ 7 years for FCA compliance). |
| `log_predictions` | bool | `false` | Log every prediction with features. |
| `log_explanations` | bool | `false` | Attach SHAP/LIME values to each logged prediction. |
| `compliance_frameworks` | list[str] | `[]` | Compliance frameworks to generate reports for (e.g. `["fca_consumer_duty", "eu_ai_act"]`). |
| `compliance_risk_level` | str | `"high"` | EU AI Act risk classification (`high`, `limited`, `minimal`, `unacceptable`). |
| `tamper_evidence` | bool | `false` | Enable HMAC-SHA256 hash chain for tamper detection. |
| `signing_key_env` | str | `"SENTINEL_AUDIT_KEY"` | Env var name containing the HMAC signing key. |
| `signing_key_path` | str \| null | `null` | File path containing the HMAC signing key (alternative to env var). |
| `azure_blob` | `AzureBlobAuditConfig` \| null | `null` | Required when `storage: azure_blob`. |
| `s3` | `S3AuditConfig` \| null | `null` | Required when `storage: s3`. |
| `gcs` | `GcsAuditConfig` \| null | `null` | Required when `storage: gcs`. |

**Validation rules:**
- When `storage` is not `local`, the matching sub-config must be present (e.g. `storage: s3` requires `s3: {...}`).

#### `audit.azure_blob`

Required when `storage: azure_blob`. Uses `DefaultAzureCredential`. Needs `Storage Blob Data Contributor` on the container.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `account_url` | str | *(required)* | Azure Storage account URL (e.g. `https://myaccount.blob.core.windows.net`). |
| `container_name` | str | *(required)* | Blob container name. |
| `prefix` | str | `"sentinel-audit"` | Blob name prefix for uploaded files. |
| `delete_local_after_ship` | bool | `false` | Delete local audit files after successful upload. |

#### `audit.s3`

Required when `storage: s3`. Uses the default boto3 credential chain.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `bucket` | str | *(required)* | S3 bucket name. |
| `prefix` | str | `"sentinel-audit"` | S3 key prefix for uploaded files. |
| `region` | str \| null | `null` | AWS region. Uses default if not set. |
| `delete_local_after_ship` | bool | `false` | Delete local audit files after successful upload. |

#### `audit.gcs`

Required when `storage: gcs`. Requires the `[gcp]` extra. Uses Application Default Credentials (ADC).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `bucket` | str | *(required)* | GCS bucket name. |
| `prefix` | str | `"sentinel-audit"` | Object name prefix for uploaded files. |
| `project` | str \| null | `null` | GCP project ID. Uses default from ADC if not set. |
| `delete_local_after_ship` | bool | `false` | Delete local audit files after successful upload. |

**Shipper invariant:** Shippers upload the **exact bytes** that were HMAC'd on disk. You can verify the chain against the local copy, the remote copy, or any re-download.

```yaml
audit:
  storage: azure_blob
  path: ./audit/
  retention_days: 2555
  log_predictions: true
  log_explanations: true
  compliance_frameworks: [fca_consumer_duty, eu_ai_act]
  compliance_risk_level: high
  tamper_evidence: true
  signing_key_env: SENTINEL_AUDIT_KEY
  azure_blob:
    account_url: https://sentinelauditprod.blob.core.windows.net
    container_name: audit
    prefix: claims_fraud_v2/
    delete_local_after_ship: false
```

```yaml
# S3 storage
audit:
  storage: s3
  path: ./audit/
  s3:
    bucket: sentinel-audit-prod
    prefix: claims_fraud_v2/
    region: us-east-1

# GCS storage (requires [gcp] extra)
audit:
  storage: gcs
  path: ./audit/
  gcs:
    bucket: sentinel-audit-prod
    prefix: claims_fraud_v2/
    project: my-gcp-project
```

When `tamper_evidence: true`, every event carries an HMAC-SHA256 signature plus a `previous_hash` chain pointer. `sentinel audit verify` walks the trail and detects insertions, deletions, or edits. See [`docs/security.md`](security.md) for the full threat model and key-rotation guidance.

---

### `registry`

Model registry backend selection and configuration.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | `"local"` \| `"azure_ml"` \| `"mlflow"` \| `"sagemaker"` \| `"vertex_ai"` \| `"databricks"` | `"local"` | Backend selector. |
| `path` | str | `"./registry"` | Local filesystem path (used by `local` backend). |
| `subscription_id` | str \| null | `null` | Azure subscription ID. **Required for `azure_ml`.** |
| `resource_group` | str \| null | `null` | Azure resource group. **Required for `azure_ml`.** |
| `workspace_name` | str \| null | `null` | Azure ML workspace name. **Required for `azure_ml`.** |
| `tracking_uri` | str \| null | `null` | MLflow tracking server URI. Falls back to `MLFLOW_TRACKING_URI` env var for `mlflow` backend. |
| `region_name` | str \| null | `null` | AWS region (for `sagemaker` backend). |
| `role_arn` | str \| null | `null` | AWS IAM role ARN (for `sagemaker` backend). |
| `s3_bucket` | str \| null | `null` | S3 bucket for model artifacts (for `sagemaker` backend). |
| `project` | str \| null | `null` | GCP project ID (for `vertex_ai` backend). |
| `location` | str \| null | `null` | GCP region (for `vertex_ai` backend). |
| `gcs_bucket` | str \| null | `null` | GCS bucket for model artifacts (for `vertex_ai` backend). |
| `host` | str \| null | `null` | Databricks workspace URL (for `databricks` backend). |
| `token` | str \| null | `null` | Databricks personal access token (for `databricks` backend). |
| `catalog` | str \| null | `null` | Unity Catalog name (for `databricks` backend). |
| `schema_name` | str \| null | `null` | Unity Catalog schema (for `databricks` backend). |
| `serialize_artifacts` | bool | `false` | Whether to serialize and store model artifacts in the registry. |
| `serializer` | `"joblib"` \| `"pickle"` \| `"onnx"` \| `"auto"` | `"auto"` | Serialization format when `serialize_artifacts: true`. `auto` selects based on framework. |

**Validation rules:**
- `backend: azure_ml` requires `subscription_id`, `resource_group`, and `workspace_name`. Missing fields produce a clear error listing which fields are absent.
- `backend: mlflow` does not strictly require `tracking_uri` at validation time — the backend constructor falls back to the `MLFLOW_TRACKING_URI` env var.

```yaml
# Local filesystem (default)
registry:
  backend: local
  path: ./registry/
  serialize_artifacts: true
  serializer: joblib

# Azure ML workspace
registry:
  backend: azure_ml
  subscription_id: ${AZURE_SUBSCRIPTION_ID}
  resource_group: sentinel-prod-rg
  workspace_name: sentinel-prod-ws

# MLflow-compatible tracking server
registry:
  backend: mlflow
  tracking_uri: https://mlflow.internal.example.com

# SageMaker
registry:
  backend: sagemaker
  region_name: us-east-1
  s3_bucket: my-model-artifacts

# Databricks Unity Catalog
registry:
  backend: databricks
  host: https://myworkspace.databricks.com
  token: ${DATABRICKS_TOKEN}
  catalog: ml_catalog
  schema_name: production
```

---

### `model_graph`

Multi-model dependency DAG. When `cascade_alerts` is enabled, drift detected in an upstream model triggers alerts for all downstream models.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dependencies` | list[ModelGraphEdge] | `[]` | List of upstream → downstream model edges. |
| `cascade_alerts` | bool | `true` | Propagate alerts to downstream models when upstream drifts. |

#### `model_graph.dependencies[]` (ModelGraphEdge)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `upstream` | str | *(required)* | Upstream model or pipeline name. |
| `downstream` | str | *(required)* | Downstream model name. |

```yaml
model_graph:
  dependencies:
    - upstream: feature_engineering_pipeline
      downstream: claims_fraud_v2
    - upstream: claims_fraud_v2
      downstream: auto_adjudication_model
  cascade_alerts: true
```

---

### `llmops`

LLM monitoring and governance layer. Disabled by default.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Toggle the entire LLMOps layer. |
| `mode` | `"rag"` \| `"completion"` \| `"chat"` \| `"agent"` | `"completion"` | Application mode. Affects which checks are applicable. |
| `prompts` | `PromptRegistryConfig` | *(see below)* | Prompt versioning and A/B testing. |
| `guardrails` | `GuardrailsConfig` | *(see below)* | Input/output guardrail pipeline. |
| `quality` | `LLMQualityConfig` | *(see below)* | Response quality evaluation. |
| `token_economics` | `TokenEconomicsConfig` | *(see below)* | Token usage and cost tracking. |
| `prompt_drift` | `PromptDriftConfig` | *(see below)* | Prompt effectiveness degradation detection. |

#### `llmops.prompts`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `registry_backend` | `"local"` \| `"azure_blob"` \| `"s3"` | `"local"` | Where to store prompt versions. |
| `versioning` | `"semantic"` \| `"timestamp"` \| `"git_hash"` | `"semantic"` | Versioning scheme for prompts. |
| `ab_testing` | dict[str, Any] | `{}` | A/B test configuration (e.g. `{enabled: true, default_split: [90, 10]}`). |

#### `llmops.guardrails`

Input and output guardrail rule lists. Each rule has a `type` and `action` at minimum. Extra fields are passed through (`extra="allow"`).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input` | list[GuardrailRuleConfig] | `[]` | Input guardrails (run before the LLM call). |
| `output` | list[GuardrailRuleConfig] | `[]` | Output guardrails (run after the LLM call). |

#### `llmops.guardrails.input[]` / `output[]` (GuardrailRuleConfig)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | str | *(required)* | Guardrail type (`pii_detection`, `jailbreak_detection`, `toxicity`, `groundedness`, `topic_fence`, `token_budget`, `format_compliance`, `regulatory_language`, `custom`, `plugin`). |
| `action` | `"block"` \| `"warn"` \| `"redact"` | `"warn"` | Action when the guardrail triggers. |
| `threshold` | float \| null | `null` | Detection threshold (interpretation depends on type). |
| `method` | str \| null | `null` | Detection method (e.g. `embedding_similarity`, `nli`, `chunk_overlap`). |
| `critical` | bool | `false` | If `true`, pipeline init fails when this guardrail cannot load. |
| `name` | str \| null | `null` | **Required for `type: custom`.** Human-readable guardrail name. |
| `rules` | list[dict] | `[]` | **Required for `type: custom`** (at least one rule). |
| `combine` | `"all"` \| `"any"` | `"all"` | For custom guardrails: whether all or any rules must match. |
| `module` | str \| null | `null` | **Required for `type: plugin`.** Python module path. |
| `class_name` | str \| null | `null` | **Required for `type: plugin`.** Class name. |
| `config` | dict[str, Any] | `{}` | Extra configuration for plugin guardrails. |

**Validation rules:**
- `type: custom` requires both `name` and a non-empty `rules` list.
- `type: plugin` requires both `module` and `class_name`.

```yaml
llmops:
  enabled: true
  mode: rag
  guardrails:
    input:
      - type: pii_detection
        action: redact
        critical: true
      - type: jailbreak_detection
        method: embedding_similarity
        threshold: 0.85
        action: block
      - type: custom
        name: profanity_check
        action: block
        rules:
          - pattern: "badword"
            match: regex
        combine: any
      - type: plugin
        module: mycompany.guardrails
        class_name: ComplianceGuardrail
        action: block
    output:
      - type: toxicity
        threshold: 0.7
        action: block
      - type: groundedness
        method: nli
        min_score: 0.6
        action: warn
```

#### `llmops.quality`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `evaluator` | `QualityEvaluatorConfig` | *(see below)* | Response quality evaluation method. |
| `semantic_drift` | `SemanticDriftConfig` | *(see below)* | Embedding-based output distribution drift. |
| `retrieval_quality` | `RetrievalQualityConfig` | *(see below)* | RAG-specific quality metrics. |

##### `llmops.quality.evaluator`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `method` | `"llm_judge"` \| `"heuristic"` \| `"reference_based"` \| `"hybrid"` | `"heuristic"` | Evaluation method. |
| `judge_model` | str \| null | `null` | Model for LLM-as-judge (e.g. `gpt-4o-mini`). |
| `rubrics` | dict[str, dict[str, Any]] | `{}` | Scoring rubrics (e.g. `{relevance: {weight: 0.3, scale: 5}}`). |
| `sample_rate` | float | `0.1` | Fraction of responses to evaluate (0.1 = 10%). |

##### `llmops.quality.semantic_drift`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `embedding_model` | str | `"text-embedding-3-small"` | Model used to embed outputs for drift detection. |
| `window` | str | `"7d"` | Time window for drift comparison. |
| `threshold` | float | `0.15` | Cosine distance threshold for detecting drift. |
| `window_size` | int | `500` | Rolling window size for embedding observations. |

##### `llmops.quality.retrieval_quality`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `track` | list[str] | `[]` | RAG metrics to track (`relevance`, `chunk_utilisation`, `faithfulness`, `answer_coverage`). |
| `min_relevance` | float | `0.5` | Minimum acceptable retrieval relevance score. |
| `min_faithfulness` | float | `0.7` | Minimum acceptable faithfulness score. |

```yaml
llmops:
  quality:
    evaluator:
      method: llm_judge
      judge_model: gpt-4o-mini
      rubrics:
        relevance: {weight: 0.3, scale: 5}
        completeness: {weight: 0.3, scale: 5}
      sample_rate: 0.1
    semantic_drift:
      embedding_model: text-embedding-3-small
      window: 7d
      threshold: 0.15
      window_size: 500
    retrieval_quality:
      track: [relevance, chunk_utilisation, faithfulness]
      min_relevance: 0.5
      min_faithfulness: 0.7
```

#### `llmops.token_economics`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `track_by` | list[str] | `["model"]` | Dimensions for cost aggregation (e.g. `[model, prompt_version, user_segment]`). |
| `budgets` | dict[str, float] | `{}` | Budget thresholds (e.g. `{daily_max_cost: 500.00}`). |
| `alerts` | dict[str, float] | `{}` | Alert thresholds (e.g. `{daily_cost_threshold: 400.00}`). |
| `model_routing` | dict[str, Any] | `{}` | Model routing config (e.g. `{log_routing_decisions: true}`). |
| `pricing` | dict[str, dict[str, float]] | `{}` | Custom per-model pricing. Keys are model names, values have `input` and `output` costs per 1K tokens. |

**Azure OpenAI:** prefix deployment names with `azure/` for automatic pricing lookup and provider tagging. Override defaults in `pricing:` for enterprise rates. See the CLAUDE.md for the full list of default `azure/` models.

```yaml
llmops:
  token_economics:
    track_by: [model, prompt_version]
    budgets:
      daily_max_cost: 500.00
      per_query_max_tokens: 8000
    alerts:
      daily_cost_threshold: 400.00
    pricing:
      azure/gpt-4o:
        input: 0.0025
        output: 0.0075
```

#### `llmops.prompt_drift`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `detection_window` | str | `"7d"` | Time window for trend analysis. |
| `signals` | dict[str, float] | `{}` | Detection signals and thresholds. |
| `min_samples` | int | `20` | Minimum observations before drift detection activates. |

```yaml
llmops:
  prompt_drift:
    detection_window: 7d
    min_samples: 50
    signals:
      quality_score_decline: 0.1
      guardrail_violation_rate_increase: 0.05
      token_usage_increase_pct: 25
```

---

### `agentops`

Agent monitoring, tracing, safety, and governance. Disabled by default.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Toggle the entire AgentOps layer. |
| `tracing` | `TracingConfig` | *(see below)* | Span-based agent trace monitoring. |
| `tool_audit` | `ToolAuditConfig` | *(see below)* | Tool call monitoring and security. |
| `safety` | `SafetyConfig` | *(see below)* | Loop detection, budgets, escalation, sandboxing. |
| `agent_registry` | `AgentRegistryConfig` | *(see below)* | Agent versioning and capability manifests. |
| `multi_agent` | `MultiAgentConfig` | *(see below)* | Multi-agent orchestration monitoring. |
| `evaluation` | `AgentEvaluationConfig` | *(see below)* | Agent evaluation framework. |

#### `agentops.tracing`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | `"local"` \| `"otlp"` \| `"arize_phoenix"` | `"local"` | Trace storage backend. |
| `otlp_endpoint` | str \| null | `null` | OpenTelemetry collector endpoint. |
| `sample_rate` | float | `1.0` | Fraction of runs to trace (1.0 = all). |
| `export_format` | `"json"` \| `"protobuf"` | `"json"` | Wire format for trace export. |
| `retention_days` | int | `90` | Days to retain local trace data. |
| `auto_instrument` | dict[str, bool] | `{}` | Framework instrumentation toggles (e.g. `{langgraph: true}`). |

#### `agentops.tool_audit`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `permissions` | dict[str, dict[str, list[str]]] | `{}` | Per-agent tool permissions with `allowed`/`blocked` lists. |
| `parameter_validation` | bool | `true` | Validate tool call parameters against schemas. |
| `rate_limits` | dict[str, str] | `{}` | Per-tool rate limits (e.g. `{default: "100/min"}`). |
| `replay` | dict[str, Any] | `{}` | Tool call replay settings. |

#### `agentops.safety`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `loop_detection` | `LoopDetectionConfig` | *(see below)* | Stuck agent detection. |
| `budget` | `BudgetConfig` | *(see below)* | Resource budget enforcement. |
| `escalation` | `EscalationConfig` | *(see below)* | Human escalation triggers. |
| `sandbox` | `SandboxConfig` | *(see below)* | Destructive operation sandboxing. |

##### `agentops.safety.loop_detection`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_iterations` | int | `50` | Hard kill after N reasoning steps. |
| `max_repeated_tool_calls` | int | `5` | Same tool + same input N times = stuck. |
| `max_delegation_depth` | int | `5` | Max agent-to-agent delegation depth. |
| `thrash_window` | int | `10` | Detect alternating states in last N steps. |

##### `agentops.safety.budget`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_tokens_per_run` | int | `50000` | Max tokens per agent run. |
| `max_cost_per_run` | float | `5.0` | Max cost (USD) per run. |
| `max_time_per_run` | str | `"300s"` | Max wall-clock time. Duration string. |
| `max_tool_calls_per_run` | int | `30` | Max tool calls per run. |
| `on_exceeded` | `"graceful_stop"` \| `"escalate"` \| `"hard_kill"` | `"graceful_stop"` | Action on budget exceeded. |

##### `agentops.safety.escalation`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `triggers` | list[EscalationTrigger] | `[]` | Human escalation conditions. |

Each trigger has: `condition` (str, required), `threshold` (float\|null), `patterns` (list[str]), `action` (`"human_handoff"` \| `"human_approval"` \| `"block"`, default `"human_handoff"`).

##### `agentops.safety.sandbox`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `destructive_ops` | list[str] | `[]` | Operations considered destructive. |
| `mode` | `"approve_first"` \| `"dry_run"` \| `"sandbox_then_apply"` | `"approve_first"` | How to handle destructive ops. |

```yaml
agentops:
  enabled: true
  tracing:
    backend: otlp
    otlp_endpoint: ${OTLP_ENDPOINT}
    auto_instrument:
      langgraph: true
  tool_audit:
    permissions:
      claims_processor:
        allowed: [sharepoint_search, policy_lookup]
        blocked: [payment_execute]
    rate_limits:
      default: "100/min"
      payment_execute: "5/min"
  safety:
    loop_detection:
      max_iterations: 50
      max_repeated_tool_calls: 5
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
        - condition: regulatory_context
          patterns: [financial_advice, medical_diagnosis]
          action: human_approval
    sandbox:
      destructive_ops: [write, delete, execute]
      mode: approve_first
  agent_registry:
    backend: local
    capability_manifest: true
    health_check_interval: 60s
    a2a:
      protocol: a2a_v1
      discovery: registry
  multi_agent:
    delegation_tracking: true
    consensus:
      enabled: true
      min_agreement: 0.67
      conflict_action: escalate
    bottleneck_detection:
      latency_percentile: p95
      threshold_ms: 5000
  evaluation:
    golden_datasets:
      path: tests/golden/
      run_on: [version_change, daily]
    task_completion:
      track_by: [agent, task_type]
      min_success_rate: 0.85
    trajectory:
      compare_against: optimal
      penalty_per_extra_step: 0.05
```

---

### `dashboard`

Optional dashboard UI. Requires `[dashboard]` extra. Disabled by default.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Toggle the dashboard. |
| `server` | `DashboardServerConfig` | *(see below)* | HTTP server and security. |
| `ui` | `DashboardUIConfig` | *(see below)* | Presentation settings. |

#### `dashboard.server`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `host` | str | `"127.0.0.1"` | Bind address. |
| `port` | int | `8000` | Listen port. |
| `root_path` | str | `""` | ASGI root path for reverse-proxy mounts. |
| `auth` | `"none"` \| `"basic"` \| `"bearer"` | `"none"` | Authentication mode. |
| `basic_auth_username` | str \| null | `null` | Basic auth username. **Required when `auth: basic`.** |
| `basic_auth_password` | SecretStr \| null | `null` | Basic auth password. **Required when `auth: basic`.** Redacted in `config show`. |
| `bearer` | `BearerAuthConfig` | *(see below)* | JWT Bearer validation. |
| `rbac` | `RBACConfig` | *(see below)* | Role-based access control. |
| `csrf` | `CSRFConfig` | *(see below)* | CSRF protection. |
| `rate_limit` | `RateLimitConfig` | *(see below)* | Rate limiting. |
| `csp` | `CSPConfig` | *(see below)* | Content Security Policy. |
| `require_signed_config` | bool | `false` | Require HMAC config signature. |

**Validation:** `auth: bearer` requires `bearer.jwks_url`; `auth: basic` requires username + password.

##### `dashboard.server.bearer`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `jwks_url` | str \| null | `null` | JWKS endpoint URL. **Required for bearer auth.** |
| `issuer` | str \| null | `null` | Expected `iss` claim. |
| `audience` | str \| null | `null` | Expected `aud` claim. |
| `username_claim` | str | `"sub"` | JWT claim for username. |
| `roles_claim` | str | `"roles"` | JWT claim for roles. |
| `algorithms` | list[str] | `["RS256"]` | Allowed signing algorithms. |
| `cache_ttl_seconds` | int | `3600` | JWKS cache TTL. |
| `leeway_seconds` | int | `30` | Clock skew tolerance. |

##### `dashboard.server.rbac`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Toggle RBAC. |
| `default_role` | str | `"viewer"` | Default role for unbound users. |
| `users` | list[RBACUserBinding] | `[]` | User → role mappings (`username` + `roles`). |
| `role_permissions` | dict[str, list[str]] | viewer/operator/admin defaults | Role → permission map. |
| `role_hierarchy` | list[str] | `["viewer", "operator", "admin"]` | Least → most privileged. Higher inherits lower. |

**Default permissions:** `viewer` gets `*.read`; `operator` adds `audit.verify`, `deployments.promote`, `retrain.trigger`, `golden.run`; `admin` gets `*`.

##### `dashboard.server.csrf`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Toggle CSRF. |
| `cookie_name` | str | `"sentinel_csrf"` | Cookie name. |
| `header_name` | str | `"X-CSRF-Token"` | Header name. |
| `cookie_secure` | bool \| null | `null` | Secure flag (auto-detect if null). |
| `cookie_samesite` | `"lax"` \| `"strict"` \| `"none"` | `"lax"` | SameSite attribute. |

##### `dashboard.server.rate_limit`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Toggle rate limiting. |
| `default_per_minute` | int | `100` | Page request limit. |
| `api_per_minute` | int | `300` | API endpoint limit. |
| `auth_per_minute` | int | `10` | Auth endpoint limit. |
| `burst_multiplier` | float | `2.0` | Bucket capacity multiplier. |

##### `dashboard.server.csp`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `true` | Toggle CSP header. |
| `policy` | str \| null | `null` | Custom CSP string (null = built-in default). |

#### `dashboard.ui`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `title` | str | `"Project Sentinel"` | Dashboard title. |
| `theme` | `"light"` \| `"dark"` \| `"auto"` | `"auto"` | UI theme. |
| `show_modules` | list[str] | `["overview", "drift", "features", "registry", "experiments", "audit", "llmops", "agentops", "deployments", "retraining", "intelligence", "compliance"]` | Sidebar modules. |
| `refresh_interval_seconds` | int | `30` | Auto-refresh interval. |

```yaml
dashboard:
  enabled: true
  server:
    host: 0.0.0.0
    port: 8080
    auth: bearer
    bearer:
      jwks_url: https://login.example.com/.well-known/jwks.json
    rbac:
      enabled: true
      users:
        - username: alice
          roles: [admin]
  ui:
    title: "Risk MLOps"
    theme: dark
    show_modules: [overview, drift, audit, compliance]
    refresh_interval_seconds: 60
```

---

### `domains`

Domain-specific overrides. Only the section matching `model.domain` is consulted at runtime. Each sub-key is freeform `dict[str, Any]` — the domain adapter validates its own fields.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tabular` | dict[str, Any] | `{}` | Tabular domain overrides (default adapter — usually no config needed). |
| `timeseries` | dict[str, Any] | `{}` | Time series: seasonality, calendar drift, forecast quality. |
| `nlp` | dict[str, Any] | `{}` | NLP: vocabulary drift, embedding space monitoring. |
| `recommendation` | dict[str, Any] | `{}` | RecSys: item/user drift, beyond-accuracy metrics, fairness. |
| `graph` | dict[str, Any] | `{}` | Graph ML: topology drift, node/edge features, KG metrics. |

#### `domains.tabular`

The tabular adapter is the default when `model.domain` is omitted or set to `"tabular"`. It uses the core drift detectors (PSI, KS, etc.) and standard classification/regression metrics from the top-level `drift`, `feature_health`, and `data_quality` sections. No additional domain-specific config is typically needed.

```yaml
model:
  name: claims_fraud_v2
  domain: tabular    # or simply omit — tabular is the default
# No domains.tabular section required
```

#### `domains.timeseries`

Time series models need seasonality-aware drift detection and forecast-specific quality metrics. Requires the `[timeseries]` extra (`pip install "sentinel-mlops[timeseries]"`) for STL decomposition.

**Common fields:**

| Field | Type | Description |
|-------|------|-------------|
| `frequency` | str | Data frequency: `hourly`, `daily`, `weekly`, `monthly`. |
| `seasonality_periods` | list[int] | Seasonality periods in units of `frequency` (e.g. `[7, 365]` = weekly + yearly with daily data). |
| `decomposition` | str | Decomposition method: `stl`, `x13`, `custom`. |
| `drift.method` | str | Drift method: `calendar_test`, `temporal_covariate`, `acf_shift`. |
| `drift.compare_against` | str | Baseline: `same_period`, `previous_window`, `global`. |
| `drift.stationarity_check` | bool | Enable ADF/KPSS stationarity monitoring. |
| `drift.adf_significance` | float | ADF test significance level (e.g. `0.05`). |
| `quality.metrics` | list[str] | Forecast metrics: `mase`, `mape`, `smape`, `coverage`, `directional_accuracy`, `winkler`. |
| `quality.prediction_interval` | float | Nominal interval (e.g. `0.95` for 95%). |
| `quality.coverage_alert_threshold` | float | Alert if actual coverage falls below this. |
| `quality.horizon_tracking` | list[int] | Forecast horizons to track separately. |
| `decomposition_monitoring.trend_slope_change_threshold` | float | Alert on trend slope change. |
| `decomposition_monitoring.seasonal_amplitude_change_pct` | float | Alert on seasonal amplitude change (%). |
| `decomposition_monitoring.residual_variance_increase_pct` | float | Alert on residual variance increase (%). |

```yaml
model:
  name: demand_forecast_v3
  domain: timeseries
  type: forecasting

domains:
  timeseries:
    frequency: daily
    seasonality_periods: [7, 365]
    decomposition: stl
    drift:
      method: calendar_test
      compare_against: same_period
      stationarity_check: true
      adf_significance: 0.05
    quality:
      metrics: [mase, coverage, directional_accuracy]
      prediction_interval: 0.95
      coverage_alert_threshold: 0.85
      horizon_tracking: [1, 7, 14, 30]
    decomposition_monitoring:
      trend_slope_change_threshold: 0.1
      seasonal_amplitude_change_pct: 20
      residual_variance_increase_pct: 30
```

**Why calendar-aware drift?** Standard PSI/KS treats every observation as independent. For time series, a "drift" in January vs July might just be seasonality. `calendar_test` compares against the same calendar period from the reference (Jan 2026 vs Jan 2025), eliminating false positives from seasonal effects.

#### `domains.nlp`

Traditional NLP models (NER, classification, sentiment) with vocabulary and embedding drift detection. Requires `[nlp-domain]` extra for sentence-transformers.

**Common fields:**

| Field | Type | Description |
|-------|------|-------------|
| `task` | str | NLP task: `ner`, `classification`, `sentiment`, `topic_modelling`. |
| `embedding_model` | str | Sentence embedding model for semantic drift (e.g. `sentence-transformers/all-MiniLM-L6-v2`). |
| `drift.vocabulary.oov_rate_threshold` | float | Alert if out-of-vocabulary token rate exceeds this. |
| `drift.vocabulary.track_new_tokens` | bool | Track new tokens appearing in production. |
| `drift.vocabulary.top_new_tokens_k` | int | Surface top-K new tokens for review. |
| `drift.embedding.method` | str | Embedding drift method: `mmd`, `mahalanobis`, `cosine_centroid`. |
| `drift.embedding.threshold` | float | Embedding drift significance threshold. |
| `drift.embedding.window` | str | Comparison window (duration string). |
| `drift.label_distribution.method` | str | Label drift test: `chi_squared`. |
| `drift.label_distribution.threshold` | float | Significance threshold (e.g. `0.05`). |
| `drift.text_stats.track` | list[str] | Text statistics: `avg_length`, `language_dist`, `oov_rate`, `special_char_rate`. |
| `quality.metrics` | list[str] | Quality metrics: `token_f1`, `span_exact_match`, `macro_f1`, `micro_f1`, `cohens_kappa`. |
| `quality.per_entity_tracking` | bool | Track F1 per entity type (NER models). |
| `quality.min_samples_per_class` | int | Minimum samples per class before metrics activate. |

```yaml
model:
  name: claims_ner_v1
  domain: nlp
  type: classification

domains:
  nlp:
    task: ner
    embedding_model: sentence-transformers/all-MiniLM-L6-v2
    drift:
      vocabulary:
        oov_rate_threshold: 0.05
        track_new_tokens: true
        top_new_tokens_k: 50
      embedding:
        method: mmd
        threshold: 0.1
        window: 7d
      label_distribution:
        method: chi_squared
        threshold: 0.05
      text_stats:
        track: [avg_length, language_dist, oov_rate, special_char_rate]
    quality:
      metrics: [token_f1, span_exact_match]
      per_entity_tracking: true
      min_samples_per_class: 50
```

**Why vocabulary drift matters:** A rising OOV rate means the model is encountering words it has never seen — common after product launches, regulatory changes, or market shifts. The `embedding` drift detector catches semantic shift even when the vocabulary stays stable (same words used in different contexts).

#### `domains.recommendation`

Recommendation systems with item/user distribution drift, beyond-accuracy metrics, and fairness monitoring. Requires `[recommendation]` extra for RecBole/Implicit.

**Common fields:**

| Field | Type | Description |
|-------|------|-------------|
| `model_type` | str | RecSys type: `collaborative_filtering`, `content_based`, `hybrid`, `neural`. |
| `item_id_field` | str | Column name for item IDs. |
| `user_id_field` | str | Column name for user IDs. |
| `interaction_types` | list[str] | Interaction types: `click`, `purchase`, `add_to_cart`, `view`, `rating`. |
| `drift.item_distribution.track_cold_start_ratio` | bool | Track % of recommended items with few interactions. |
| `drift.item_distribution.long_tail_ratio_threshold` | float | Alert if top-1% items get more than this share. |
| `drift.item_distribution.catalogue_change_window` | str | Window for catalogue change detection. |
| `drift.user_distribution.segment_field` | str | Feature field for user segmentation. |
| `drift.user_distribution.interaction_pattern_window` | str | Window for interaction pattern analysis. |
| `quality.ranking_metrics` | list[str] | Ranking metrics: `ndcg_at_10`, `map_at_10`, `hit_rate_at_10`, `mrr`. |
| `quality.beyond_accuracy.coverage.min_threshold` | float | Minimum catalogue coverage (0.4 = 40% of items recommended). |
| `quality.beyond_accuracy.diversity.method` | str | Diversity method: `intra_list_similarity`. |
| `quality.beyond_accuracy.diversity.min_threshold` | float | Minimum diversity score. |
| `quality.beyond_accuracy.novelty.method` | str | Novelty method: `inverse_popularity`. |
| `quality.beyond_accuracy.novelty.track_trend` | bool | Track novelty trends over time. |
| `quality.beyond_accuracy.popularity_bias.method` | str | Bias method: `gini_coefficient`. |
| `quality.beyond_accuracy.popularity_bias.max_threshold` | float | Max acceptable bias (0.8 = extreme concentration). |
| `quality.fairness.protected_attribute` | str | Attribute for fairness evaluation. |
| `quality.fairness.metric` | str | Metric to check fairness on (e.g. `ndcg_at_10`). |
| `quality.fairness.max_disparity` | float | Max metric gap between groups (0.1 = 10%). |
| `ab_testing.integration` | str | A/B platform: `none`, `optimizely`, `launchdarkly`, `custom`. |
| `ab_testing.guardrail_metrics` | list[str] | Business guardrail metrics (e.g. `revenue_per_session`). |

```yaml
model:
  name: product_reco_v2
  domain: recommendation
  type: ranking

domains:
  recommendation:
    model_type: hybrid
    item_id_field: product_id
    user_id_field: user_id
    interaction_types: [click, purchase, add_to_cart]
    drift:
      item_distribution:
        track_cold_start_ratio: true
        long_tail_ratio_threshold: 0.3
        catalogue_change_window: 7d
      user_distribution:
        segment_field: user_segment
        interaction_pattern_window: 14d
    quality:
      ranking_metrics: [ndcg_at_10, map_at_10]
      beyond_accuracy:
        coverage:
          min_threshold: 0.4
        diversity:
          method: intra_list_similarity
          min_threshold: 0.3
        novelty:
          method: inverse_popularity
          track_trend: true
        popularity_bias:
          method: gini_coefficient
          max_threshold: 0.8
      fairness:
        protected_attribute: user_segment
        metric: ndcg_at_10
        max_disparity: 0.1
    ab_testing:
      integration: none
      guardrail_metrics: [revenue_per_session, engagement_rate]
```

**Beyond-accuracy metrics** are the key differentiator for recommendation monitoring. A model with perfect NDCG may still fail if it only surfaces popular items (low novelty, high popularity bias) or shows the same recommendations to every user (low diversity). These metrics surface degradation invisible to accuracy-only tracking.

#### `domains.graph`

Graph ML / knowledge graph models with topology drift, node/edge feature drift, and KG-specific metrics. Requires `[graph]` extra for NetworkX/PyG.

**Common fields:**

| Field | Type | Description |
|-------|------|-------------|
| `task` | str | Graph task: `link_prediction`, `node_classification`, `graph_classification`, `kg_completion`. |
| `graph_type` | str | Graph type: `knowledge_graph`, `social_network`, `transaction_graph`, `molecular`. |
| `framework` | str | Graph ML framework: `pyg` (PyTorch Geometric), `dgl`, `networkx`. |
| `drift.topology.degree_distribution.method` | str | Test method: `ks_test`. |
| `drift.topology.degree_distribution.window` | str | Comparison window. |
| `drift.topology.degree_distribution.threshold` | float | Drift threshold. |
| `drift.topology.density_monitoring` | bool | Track graph density (edges / possible edges). |
| `drift.topology.clustering_coefficient` | bool | Track average clustering coefficient. |
| `drift.topology.connected_components` | bool | Track connected component count and size distribution. |
| `drift.topology.diameter_tracking` | bool | Track graph diameter (expensive — sample for large graphs). |
| `drift.node_features.method` | str | Feature drift method: `psi`. |
| `drift.node_features.threshold` | float | Feature drift threshold. |
| `drift.edge_features.method` | str | Feature drift method: `psi`. |
| `drift.edge_features.threshold` | float | Feature drift threshold. |
| `drift.feature_topology_correlation.track` | bool | Monitor feature-topology correlation. |
| `drift.feature_topology_correlation.method` | str | Method: `mutual_information`. |
| `quality.metrics` | list[str] | Quality metrics: `auc_roc`, `hits_at_10`, `mrr`, `node_f1`, `modularity`. |
| `quality.embedding_isotropy.track` | bool | Monitor embedding isotropy (anisotropic = degraded quality). |
| `quality.embedding_isotropy.method` | str | Method: `partition_score`. |
| `quality.embedding_isotropy.threshold` | float | Isotropy threshold. |
| `knowledge_graph.relation_coverage.track_new_relations` | bool | Track new relation types the model has not seen. |
| `knowledge_graph.relation_coverage.min_training_triples_per_relation` | int | Min training triples per relation. |
| `knowledge_graph.entity_vocabulary.oov_entity_threshold` | float | Alert if unseen entity fraction exceeds this. |
| `knowledge_graph.plausibility_trend.window` | str | Plausibility tracking window. |
| `knowledge_graph.plausibility_trend.decline_threshold` | float | Alert on plausibility decline. |

```yaml
model:
  name: fraud_graph_v1
  domain: graph
  type: classification

domains:
  graph:
    task: link_prediction
    graph_type: transaction_graph
    framework: pyg
    drift:
      topology:
        degree_distribution:
          method: ks_test
          window: 7d
          threshold: 0.05
        density_monitoring: true
        clustering_coefficient: true
        connected_components: true
        diameter_tracking: false       # expensive for large graphs
      node_features:
        method: psi
        threshold: 0.2
      edge_features:
        method: psi
        threshold: 0.2
      feature_topology_correlation:
        track: true
        method: mutual_information
    quality:
      metrics: [auc_roc, hits_at_10, mrr]
      embedding_isotropy:
        track: true
        method: partition_score
        threshold: 0.3
    knowledge_graph:
      relation_coverage:
        track_new_relations: true
        min_training_triples_per_relation: 100
      entity_vocabulary:
        oov_entity_threshold: 0.1
      plausibility_trend:
        window: 7d
        decline_threshold: 0.05
```

**Why topology drift matters:** In graph ML, both node/edge features AND graph structure are inputs. A transaction graph where average degree jumps from 50 to 500 has fundamentally changed even if individual node features are stable. Standard drift detection sees features but is blind to structure — topology monitoring fills this gap.

#### Domain example files

See `configs/examples/` for complete worked configs per domain:

| Domain | Example file | Model type |
|--------|-------------|------------|
| `tabular` | `insurance_fraud.yaml` | Classification |
| `timeseries` | `demand_forecast.yaml` | Forecasting |
| `nlp` | `ner_entity_extraction.yaml` | NER / Classification |
| `recommendation` | `product_reco.yaml` | Ranking |
| `graph` | `fraud_graph.yaml` | Link prediction |

---

### `datasets`

Dataset metadata registry.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `registry_path` | str | `"./datasets"` | Local directory for dataset metadata. |
| `auto_hash` | bool | `true` | Auto-compute content hash on registration. |
| `require_schema` | bool | `false` | Require JSON Schema when registering datasets. |

---

### `experiments`

Experiment tracker for linking training runs to production models.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `storage_path` | str | `"./experiments"` | Local directory for experiment data. |
| `auto_log` | bool | `true` | Auto-log hyperparameters and metrics. |
| `nested_runs` | bool | `true` | Allow nested experiment runs. |
| `max_metric_history` | int | `10000` | Max metric data points per run. |

---

## Validation Rules Summary

### Global rules

- All sub-models use `extra="forbid"` **except** `ChannelConfig` and `GuardrailRuleConfig` (`extra="allow"`).
- `version` must be a non-empty string.
- `model.name` is required.
- `model.type` and `model.domain` are strict `Literal` enums.
- Duration strings match `^\d+(?:\.\d+)?(?:ms|s|m|h|d|w)$`.
- `${VAR}` env vars are substituted before validation.

### Cross-field validators

| Validator | Condition | Error |
|-----------|-----------|-------|
| `ModelConfig` | Invalid type + domain combo | *"model.type=X is not compatible with model.domain=Y"* |
| `CohortAnalysisConfig` | `enabled` without `cohort_column` | *"cohort_column is required when cohort_analysis is enabled"* |
| `ChannelConfig` | Missing required field for channel type | *"channel type 'X' requires Y"* |
| `DeploymentConfig` | Missing sub-config for target | *"deployment.target=X requires deployment.X to be set"* |
| `DeploymentConfig` | Incompatible strategy + target | *"deployment.strategy=X is not compatible with deployment.target=Y"* |
| `CostMonitorConfig` | Unknown metric in thresholds | *"unknown cost metric(s) in alert_thresholds"* |
| `AuditConfig` | Non-local storage without sub-config | *"audit.storage=X requires audit.X to be set"* |
| `RegistryConfig` | `azure_ml` without required fields | *"registry.backend=azure_ml requires registry.X to be set"* |
| `DashboardServerConfig` | Auth mode without required config | *"dashboard.server.auth=X requires Y"* |
| `GuardrailRuleConfig` | Custom/plugin without required fields | *"Custom guardrail requires..."* / *"Plugin guardrail requires..."* |

### CLI validation commands

```bash
# Lenient validation (warnings only for env vars / files)
sentinel config validate --config sentinel.yaml

# Strict validation (errors for missing env vars and files)
sentinel config validate --config sentinel.yaml --strict

# Show resolved config (secrets masked)
sentinel config show --config sentinel.yaml

# Show resolved config (secrets unmasked)
sentinel config show --config sentinel.yaml --unmask --format json
```

---

## Full Example — Traditional ML

A complete config for a BFSI fraud classifier with full monitoring, alerting, retraining, and deployment:

```yaml
version: "1.0"

model:
  name: claims_fraud_v2
  type: classification
  framework: xgboost
  version: "2.3.1"
  domain: tabular
  baseline_dataset: s3://bucket/baselines/fraud_v2.parquet
  description: "BFSI fraud classifier — UK retail banking"

data_quality:
  schema:
    enforce: true
    path: schemas/claims_v2.json
  freshness:
    max_age_hours: 24
  outlier_detection:
    method: isolation_forest
    contamination: 0.05
  null_threshold: 0.10
  duplicate_threshold: 0.05

drift:
  data:
    method: psi
    threshold: 0.2
    window: 7d
    reference: baseline
    features:
      include: all
      exclude: [timestamp, id]
  concept:
    method: ddm
    warning_level: 2.0
    drift_level: 3.0
    min_samples: 100
  model:
    metrics: [accuracy, f1, auc]
    threshold:
      accuracy: 0.05
      f1: 0.08
    evaluation_window: 1000
  schedule:
    enabled: true
    interval: 1d
    run_on_start: true
  auto_check:
    enabled: true
    every_n_predictions: 500

feature_health:
  importance_method: shap
  alert_on_top_n_drift: 3
  recalculate_importance: weekly

cohort_analysis:
  enabled: true
  cohort_column: customer_segment
  max_cohorts: 20
  min_samples_per_cohort: 50
  disparity_threshold: 0.10

alerts:
  channels:
    - type: slack
      webhook_url: ${SLACK_WEBHOOK_URL}
      channel: "#ml-alerts"
    - type: pagerduty
      routing_key: ${PD_ROUTING_KEY}
      severity_mapping:
        critical: critical
        high: error
  policies:
    cooldown: 1h
    digest_mode: false
    rate_limit_per_hour: 60
    escalation:
      - after: 0m
        channels: [slack]
        severity: [warning, high, critical]
      - after: 2h
        channels: [slack, pagerduty]
        severity: [critical]

retraining:
  trigger: drift_confirmed
  pipeline: azureml://pipelines/retrain_fraud
  deploy_on_promote: true
  approval:
    mode: hybrid
    approvers: [ml-team@company.com]
    auto_promote_if:
      metric: f1
      improvement_pct: 2.0
    timeout: 48h
  validation:
    holdout_dataset: s3://bucket/holdout.parquet
    min_performance:
      accuracy: 0.85
      f1: 0.80

deployment:
  strategy: canary
  target: azure_ml_endpoint
  canary:
    initial_traffic_pct: 5
    ramp_steps: [5, 25, 50, 100]
    ramp_interval: 1h
    rollback_on:
      error_rate_increase: 0.02
      latency_p99_increase_ms: 50
  azure_ml_endpoint:
    endpoint_name: claims-fraud-endpoint
    subscription_id: ${AZURE_SUBSCRIPTION_ID}
    resource_group: sentinel-prod-rg
    workspace_name: sentinel-prod-ws

cost_monitor:
  track:
    - inference_latency_ms
    - throughput_rps
    - cost_per_prediction
  alert_thresholds:
    latency_p99_ms: 200
    cost_per_1k_predictions: 5.00

business_kpi:
  mappings:
    - model_metric: precision
      business_kpi: fraud_catch_rate
      data_source: warehouse://analytics.fraud_metrics
    - model_metric: recall
      business_kpi: false_positive_rate

audit:
  storage: azure_blob
  path: ./audit/
  retention_days: 2555
  log_predictions: true
  log_explanations: true
  compliance_frameworks: [fca_consumer_duty, eu_ai_act]
  compliance_risk_level: high
  tamper_evidence: true
  signing_key_env: SENTINEL_AUDIT_KEY
  azure_blob:
    account_url: https://sentinelauditprod.blob.core.windows.net
    container_name: audit
    prefix: claims_fraud_v2/

registry:
  backend: azure_ml
  subscription_id: ${AZURE_SUBSCRIPTION_ID}
  resource_group: sentinel-prod-rg
  workspace_name: sentinel-prod-ws

model_graph:
  dependencies:
    - upstream: feature_engineering_pipeline
      downstream: claims_fraud_v2
    - upstream: claims_fraud_v2
      downstream: auto_adjudication_model
  cascade_alerts: true

datasets:
  registry_path: ./datasets
  auto_hash: true

experiments:
  storage_path: ./experiments
  auto_log: true
```

---

## Full Example — LLMOps RAG Pipeline

A config for a RAG-based insurance claims Q&A system with guardrails, quality scoring, and cost tracking:

```yaml
version: "1.0"

model:
  name: claims_qa_rag
  type: generation
  domain: tabular
  description: "RAG pipeline for insurance claims Q&A"

llmops:
  enabled: true
  mode: rag
  prompts:
    registry_backend: local
    versioning: semantic
    ab_testing:
      enabled: true
      default_split: [90, 10]
  guardrails:
    input:
      - type: pii_detection
        action: redact
        entities: [person, ssn, account_number, email, phone]
        redaction_strategy: mask
        critical: true
      - type: jailbreak_detection
        method: embedding_similarity
        threshold: 0.85
        action: block
      - type: topic_fence
        allowed_topics: [insurance_claims, policy_coverage, underwriting]
        action: warn
      - type: token_budget
        max_input_tokens: 4000
        action: block
    output:
      - type: toxicity
        threshold: 0.7
        action: block
      - type: groundedness
        method: nli
        min_score: 0.6
        action: warn
      - type: format_compliance
        expected_schema: schemas/claims_summary_output.json
        action: warn
      - type: regulatory_language
        prohibited_phrases_file: compliance/prohibited_terms.yaml
        action: block
  quality:
    evaluator:
      method: llm_judge
      judge_model: gpt-4o-mini
      rubrics:
        relevance: {weight: 0.3, scale: 5}
        completeness: {weight: 0.3, scale: 5}
        clarity: {weight: 0.2, scale: 5}
        safety: {weight: 0.2, scale: 5}
      sample_rate: 0.1
    semantic_drift:
      embedding_model: text-embedding-3-small
      window: 7d
      threshold: 0.15
      window_size: 500
    retrieval_quality:
      track: [relevance, chunk_utilisation, faithfulness, answer_coverage]
      min_relevance: 0.5
      min_faithfulness: 0.7
  token_economics:
    track_by: [model, prompt_version, user_segment]
    budgets:
      daily_max_cost: 500.00
      per_query_max_tokens: 8000
      per_query_max_cost: 0.50
    alerts:
      daily_cost_threshold: 400.00
      cost_per_query_trend_increase_pct: 20
    pricing:
      azure/gpt-4o:
        input: 0.0025
        output: 0.0075
  prompt_drift:
    detection_window: 7d
    min_samples: 50
    signals:
      quality_score_decline: 0.1
      guardrail_violation_rate_increase: 0.05
      token_usage_increase_pct: 25

alerts:
  channels:
    - type: slack
      webhook_url: ${SLACK_WEBHOOK_URL}
      channel: "#llm-alerts"
  policies:
    cooldown: 1h

audit:
  storage: local
  path: ./audit/
  retention_days: 2555
  log_predictions: true
  compliance_frameworks: [eu_ai_act]
```

---

## Full Example — AgentOps Multi-Agent System

A config for a multi-agent underwriting system with safety controls, tool audit, and evaluation:

```yaml
version: "1.0"

model:
  name: underwriting_agent_system
  type: generation
  domain: tabular
  description: "Multi-agent underwriting assessment system"

agentops:
  enabled: true
  tracing:
    backend: otlp
    otlp_endpoint: ${OTLP_ENDPOINT}
    sample_rate: 1.0
    export_format: json
    retention_days: 90
    auto_instrument:
      langgraph: true
      semantic_kernel: true
  tool_audit:
    permissions:
      risk_assessor:
        allowed: [risk_database, actuarial_tables, credit_check]
        blocked: [payment_execute, user_delete]
      document_processor:
        allowed: [document_ocr, sharepoint_search, llm_extraction]
        blocked: [payment_execute, database_write]
      underwriter:
        allowed: [risk_database, policy_lookup, pricing_engine]
        blocked: [payment_execute]
    parameter_validation: true
    rate_limits:
      default: "100/min"
      credit_check: "10/min"
      pricing_engine: "20/min"
    replay:
      enabled: true
      storage: azure_blob
  safety:
    loop_detection:
      max_iterations: 50
      max_repeated_tool_calls: 5
      max_delegation_depth: 5
      thrash_window: 10
    budget:
      max_tokens_per_run: 50000
      max_cost_per_run: 5.00
      max_time_per_run: 300s
      max_tool_calls_per_run: 30
      on_exceeded: graceful_stop
    escalation:
      triggers:
        - condition: confidence_below
          threshold: 0.3
          action: human_handoff
        - condition: consecutive_tool_failures
          threshold: 3
          action: human_handoff
        - condition: sensitive_data_detected
          action: human_approval
        - condition: regulatory_context
          patterns: [financial_advice, medical_diagnosis, legal_opinion]
          action: human_approval
    sandbox:
      destructive_ops: [write, delete, execute, transfer]
      mode: approve_first
  agent_registry:
    backend: local
    capability_manifest: true
    health_check_interval: 60s
    a2a:
      protocol: a2a_v1
      discovery: registry
  multi_agent:
    delegation_tracking: true
    consensus:
      enabled: true
      min_agreement: 0.67
      conflict_action: escalate
    bottleneck_detection:
      latency_percentile: p95
      threshold_ms: 5000
  evaluation:
    golden_datasets:
      path: tests/golden/
      run_on: [version_change, daily]
    task_completion:
      track_by: [agent, task_type]
      min_success_rate: 0.85
    trajectory:
      compare_against: optimal
      penalty_per_extra_step: 0.05

llmops:
  enabled: true
  mode: agent
  guardrails:
    input:
      - type: pii_detection
        action: redact
        critical: true
    output:
      - type: groundedness
        method: nli
        min_score: 0.6
        action: warn
  token_economics:
    track_by: [model, agent]
    budgets:
      daily_max_cost: 1000.00

alerts:
  channels:
    - type: slack
      webhook_url: ${SLACK_WEBHOOK_URL}
      channel: "#agent-alerts"
    - type: pagerduty
      routing_key: ${PD_ROUTING_KEY}
  policies:
    cooldown: 30m
    escalation:
      - after: 0m
        channels: [slack]
        severity: [high, critical]
      - after: 15m
        channels: [slack, pagerduty]
        severity: [critical]

audit:
  storage: azure_blob
  path: ./audit/
  retention_days: 2555
  log_predictions: true
  tamper_evidence: true
  signing_key_env: SENTINEL_AUDIT_KEY
  compliance_frameworks: [fca_consumer_duty, eu_ai_act]
  azure_blob:
    account_url: https://sentinelauditprod.blob.core.windows.net
    container_name: audit
    prefix: underwriting_agents/

dashboard:
  enabled: true
  server:
    host: 0.0.0.0
    port: 8080
    auth: bearer
    bearer:
      jwks_url: https://login.example.com/.well-known/jwks.json
      issuer: https://login.example.com/
      audience: sentinel-dashboard
      username_claim: sub
      roles_claim: roles
    rbac:
      enabled: true
      default_role: viewer
      role_hierarchy: [viewer, operator, admin]
      users:
        - username: alice
          roles: [admin]
        - username: bob
          roles: [operator]
    csrf:
      enabled: true
      cookie_samesite: lax
    rate_limit:
      enabled: true
      default_per_minute: 100
      api_per_minute: 300
      auth_per_minute: 10
    csp:
      enabled: true
    require_signed_config: true
  ui:
    title: "Underwriting Agent Ops"
    theme: auto
    show_modules: [overview, agentops, llmops, audit, compliance]
    refresh_interval_seconds: 15
```
