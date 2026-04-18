"""Pydantic schemas for Sentinel YAML configuration files.

The full config tree is rooted at `SentinelConfig`. Sub-models map 1:1 to
top-level YAML sections so the schema and the docs stay aligned.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator

_INTERVAL_RE = re.compile(r"^\d+(?:\.\d+)?(?:ms|s|m|h|d|w)$")


def _validate_interval(v: str) -> str:
    """Validate duration/interval strings like '7d', '1h', '300s'."""
    if not _INTERVAL_RE.match(v.strip()):
        raise ValueError(
            f"Invalid interval format '{v}'. "
            "Expected a number followed by a unit: ms, s, m, h, d, or w "
            "(e.g. '7d', '1h', '300s', '500ms')."
        )
    return v.strip()


class _Base(BaseModel):
    """Base config model with strict validation."""

    model_config = ConfigDict(extra="forbid", validate_default=True)


# ── Model ──────────────────────────────────────────────────────────


class ModelConfig(_Base):
    """Identifies and describes the model being monitored."""

    name: str
    type: Literal["classification", "regression", "ranking", "forecasting", "generation"] = (
        "classification"
    )
    framework: str | None = None  # auto-detected if None
    version: str | None = None
    domain: Literal["tabular", "timeseries", "nlp", "recommendation", "graph"] = "tabular"
    baseline_dataset: str | None = None
    description: str | None = None

    @model_validator(mode="after")
    def _validate_type_domain_combination(self) -> ModelConfig:
        """Reject obviously invalid model type + domain combinations."""
        _invalid_combos: dict[str, set[str]] = {
            "ranking": {"timeseries"},
            "forecasting": {"graph"},
            "generation": {"timeseries", "graph"},
        }
        blocked_domains = _invalid_combos.get(self.type, set())
        if self.domain in blocked_domains:
            raise ValueError(
                f"model.type={self.type!r} is not compatible with model.domain={self.domain!r}"
            )
        return self


# ── Data quality ───────────────────────────────────────────────────


class SchemaConfig(_Base):
    enforce: bool = True
    path: str | None = None


class FreshnessConfig(_Base):
    max_age_hours: int = 24


class OutlierConfig(_Base):
    method: Literal["isolation_forest", "zscore", "iqr"] = "iqr"
    contamination: float = 0.05


class DataQualityConfig(_Base):
    schema_: SchemaConfig = Field(default_factory=SchemaConfig, alias="schema")
    freshness: FreshnessConfig = Field(default_factory=FreshnessConfig)
    outlier_detection: OutlierConfig = Field(default_factory=OutlierConfig)
    null_threshold: float = 0.1
    duplicate_threshold: float = 0.05

    model_config = ConfigDict(populate_by_name=True, extra="forbid")


# ── Drift ──────────────────────────────────────────────────────────


class FeatureSelector(_Base):
    include: list[str] | Literal["all"] = "all"
    exclude: list[str] = Field(default_factory=list)


class DataDriftConfig(_Base):
    method: Literal["psi", "ks", "js_divergence", "chi_squared", "wasserstein"] = "psi"
    threshold: float = 0.2
    window: str = "7d"
    reference: Literal["baseline", "previous_window", "custom"] = "baseline"
    features: FeatureSelector = Field(default_factory=FeatureSelector)


class ConceptDriftConfig(_Base):
    method: Literal["ddm", "eddm", "adwin", "page_hinkley"] = "ddm"
    warning_level: float = 2.0
    drift_level: float = 3.0
    min_samples: int = 100
    requires_actuals: bool = True


class ModelDriftConfig(_Base):
    metrics: list[str] = Field(default_factory=lambda: ["accuracy", "f1"])
    threshold: dict[str, float] = Field(default_factory=lambda: {"accuracy": 0.05})
    evaluation_window: int = 1000


class DriftScheduleConfig(_Base):
    """Background scheduler for periodic drift detection (WS-C)."""

    enabled: bool = False
    interval: str = "7d"
    run_on_start: bool = False

    @field_validator("interval")
    @classmethod
    def _check_interval(cls, v: str) -> str:
        return _validate_interval(v)


class DriftAutoCheckConfig(_Base):
    """Count-based automatic drift checking."""

    enabled: bool = False
    every_n_predictions: int = 1000


class DriftConfig(_Base):
    data: DataDriftConfig = Field(default_factory=DataDriftConfig)
    concept: ConceptDriftConfig | None = None
    model: ModelDriftConfig | None = None
    schedule: DriftScheduleConfig = Field(default_factory=DriftScheduleConfig)
    auto_check: DriftAutoCheckConfig = Field(default_factory=DriftAutoCheckConfig)


# ── Feature health ─────────────────────────────────────────────────


class FeatureHealthConfig(_Base):
    importance_method: Literal["shap", "permutation", "builtin"] = "builtin"
    alert_on_top_n_drift: int = 3
    recalculate_importance: Literal["never", "daily", "weekly", "monthly"] = "weekly"


# ── Cohort analysis ────────────────────────────────────────────────


class CohortAnalysisConfig(_Base):
    """Configuration for cohort-based performance analysis.

    When enabled, predictions logged via ``SentinelClient.log_prediction``
    can carry a ``cohort_id`` to segment monitoring by sub-population.

    Attributes:
        enabled: Toggle cohort analysis on/off.
        cohort_column: Default feature name to derive cohort IDs from when
            an explicit ``cohort_id`` is not provided.  ``None`` means the
            caller must always pass ``cohort_id`` explicitly.
        max_cohorts: Upper limit on tracked cohorts to bound memory.
        min_samples_per_cohort: Minimum predictions per cohort before
            drift / disparity checks become active.
        disparity_threshold: Maximum acceptable relative performance gap
            between a cohort and the global mean (e.g. 0.10 = 10%).
        buffer_size: Rolling prediction buffer per cohort.
    """

    enabled: bool = False
    cohort_column: str | None = None
    max_cohorts: int = 50
    min_samples_per_cohort: int = 30
    disparity_threshold: float = 0.10
    buffer_size: int = 1000

    @model_validator(mode="after")
    def _require_column_when_enabled(self) -> CohortAnalysisConfig:
        if self.enabled and not self.cohort_column:
            raise ValueError("cohort_column is required when cohort_analysis is enabled")
        return self


# ── Alerts ─────────────────────────────────────────────────────────


class ChannelConfig(_Base):
    type: Literal["slack", "teams", "pagerduty", "email", "webhook"]
    # ``webhook_url`` and ``routing_key`` carry credentials. They are stored
    # as :class:`pydantic.SecretStr` so that ``repr``, ``model_dump_json``,
    # and ``masked_dump`` never leak them. Use
    # :func:`sentinel.config.secrets.unwrap` at the channel boundary to
    # retrieve the plaintext.
    webhook_url: SecretStr | None = None
    routing_key: SecretStr | None = None
    channel: str | None = None  # slack channel name
    recipients: list[str] = Field(default_factory=list)
    severity_mapping: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    template: str | None = None  # optional Jinja2 template for message formatting

    model_config = ConfigDict(extra="allow")  # let channels carry custom fields

    @model_validator(mode="after")
    def _validate_channel_fields(self) -> ChannelConfig:
        if self.type in ("slack", "teams", "webhook") and self.webhook_url is None:
            raise ValueError(f"channel type '{self.type}' requires webhook_url")
        if self.type in ("slack", "teams", "webhook") and self.webhook_url is not None:
            raw = self.webhook_url.get_secret_value().strip()
            if not raw:
                raise ValueError(f"channel type '{self.type}' requires a non-empty webhook_url")
        if self.type == "pagerduty" and self.routing_key is None:
            raise ValueError("channel type 'pagerduty' requires routing_key")
        if self.type == "email" and not self.recipients:
            raise ValueError("channel type 'email' requires at least one recipient")
        return self


class EscalationStep(_Base):
    after: str = "0m"
    channels: list[str] = Field(default_factory=list)
    severity: list[str] = Field(default_factory=list)

    @field_validator("after")
    @classmethod
    def _check_after(cls, v: str) -> str:
        return _validate_interval(v)


class AlertPolicies(_Base):
    cooldown: str = "1h"
    digest_mode: bool = False
    digest_interval: str = "6h"
    rate_limit_per_hour: int = 60
    rate_limit_window: str = "1h"
    default_template: str | None = None  # global Jinja2 template applied to all channels
    escalation: list[EscalationStep] = Field(default_factory=list)

    @field_validator("cooldown", "digest_interval", "rate_limit_window")
    @classmethod
    def _check_intervals(cls, v: str) -> str:
        return _validate_interval(v)


class AlertsConfig(_Base):
    channels: list[ChannelConfig] = Field(default_factory=list)
    policies: AlertPolicies = Field(default_factory=AlertPolicies)


# ── Retraining ─────────────────────────────────────────────────────


class ApprovalConfig(_Base):
    mode: Literal["auto", "human_in_loop", "hybrid"] = "human_in_loop"
    approvers: list[str] = Field(default_factory=list)
    auto_promote_if: dict[str, Any] = Field(default_factory=dict)
    timeout: str = "48h"

    @field_validator("timeout")
    @classmethod
    def _check_timeout(cls, v: str) -> str:
        return _validate_interval(v)


class ValidationConfig(_Base):
    holdout_dataset: str | None = None
    min_performance: dict[str, float] = Field(default_factory=dict)


class RetrainingConfig(_Base):
    trigger: Literal["drift_confirmed", "scheduled", "manual"] = "drift_confirmed"
    pipeline: str | None = None
    schedule: str | None = None  # cron expression for scheduled retrains
    approval: ApprovalConfig = Field(default_factory=ApprovalConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    deploy_on_promote: bool = False


# ── Deployment ─────────────────────────────────────────────────────


class CanaryConfig(_Base):
    initial_traffic_pct: int = 5
    ramp_steps: list[int] = Field(default_factory=lambda: [5, 25, 50, 100])
    ramp_interval: str = "1h"
    rollback_on: dict[str, float] = Field(default_factory=dict)

    @field_validator("ramp_interval")
    @classmethod
    def _check_ramp_interval(cls, v: str) -> str:
        return _validate_interval(v)


class ShadowConfig(_Base):
    duration: str = "24h"
    log_predictions: bool = True

    @field_validator("duration")
    @classmethod
    def _check_duration(cls, v: str) -> str:
        return _validate_interval(v)


class BlueGreenConfig(_Base):
    health_check_url: str | None = None
    warmup_seconds: int = 30


class AzureMLEndpointTargetConfig(_Base):
    """Config for the Azure ML Managed Online Endpoint deployment target."""

    endpoint_name: str = Field(min_length=1)
    subscription_id: str = Field(min_length=1)
    resource_group: str = Field(min_length=1)
    workspace_name: str = Field(min_length=1)
    deployment_name_pattern: str = "{model_name}-{version}"


class AzureAppServiceTargetConfig(_Base):
    """Config for the Azure App Service slot-swap deployment target."""

    subscription_id: str
    resource_group: str
    site_name: str
    production_slot: str = "production"
    staging_slot: str = "staging"
    health_check_path: str = "/healthz"


class AKSDeploymentTargetConfig(_Base):
    """Config for the Azure Kubernetes Service deployment target."""

    namespace: str
    service_name: str
    deployment_name_pattern: str = "{model_name}-{version}"
    replicas_total: int = 10
    kubeconfig_path: str | None = None


class SageMakerEndpointTargetConfig(_Base):
    """Config for the AWS SageMaker endpoint deployment target."""

    endpoint_name: str
    region_name: str | None = None
    variant_name_pattern: str = "{model_name}-{version}"


class VertexAIEndpointTargetConfig(_Base):
    """Config for the Google Vertex AI endpoint deployment target."""

    endpoint_name: str
    project: str
    location: str = "us-central1"


# Strategy + target compatibility matrix. ``canary`` against
# ``azure_app_service`` is rejected because slot-traffic % routing is
# brittle and out of scope for WS#2 — use ``blue_green`` instead.
_STRATEGY_TARGET_COMPAT: dict[str, set[str]] = {
    "shadow": {
        "local", "azure_ml_endpoint", "azure_app_service", "aks",
        "sagemaker_endpoint", "vertex_ai_endpoint",
    },
    "canary": {
        "local", "azure_ml_endpoint", "aks",
        "sagemaker_endpoint", "vertex_ai_endpoint",
    },
    "blue_green": {
        "local", "azure_ml_endpoint", "azure_app_service", "aks",
        "sagemaker_endpoint", "vertex_ai_endpoint",
    },
    "direct": {
        "local", "azure_ml_endpoint", "azure_app_service", "aks",
        "sagemaker_endpoint", "vertex_ai_endpoint",
    },
}


class DeploymentConfig(_Base):
    strategy: Literal["shadow", "canary", "blue_green", "direct"] = "canary"
    canary: CanaryConfig = Field(default_factory=CanaryConfig)
    shadow: ShadowConfig = Field(default_factory=ShadowConfig)
    blue_green: BlueGreenConfig = Field(default_factory=BlueGreenConfig)
    target: Literal[
        "local", "azure_ml_endpoint", "azure_app_service", "aks",
        "sagemaker_endpoint", "vertex_ai_endpoint",
    ] = "local"
    azure_ml_endpoint: AzureMLEndpointTargetConfig | None = None
    azure_app_service: AzureAppServiceTargetConfig | None = None
    aks: AKSDeploymentTargetConfig | None = None
    sagemaker_endpoint: SageMakerEndpointTargetConfig | None = None
    vertex_ai_endpoint: VertexAIEndpointTargetConfig | None = None

    @model_validator(mode="after")
    def _validate_target_requirements(self) -> DeploymentConfig:
        if self.target == "azure_ml_endpoint" and self.azure_ml_endpoint is None:
            raise ValueError(
                "deployment.target=azure_ml_endpoint requires "
                "deployment.azure_ml_endpoint to be set"
            )
        if self.target == "azure_app_service" and self.azure_app_service is None:
            raise ValueError(
                "deployment.target=azure_app_service requires "
                "deployment.azure_app_service to be set"
            )
        if self.target == "aks" and self.aks is None:
            raise ValueError("deployment.target=aks requires deployment.aks to be set")
        if self.target == "sagemaker_endpoint" and self.sagemaker_endpoint is None:
            raise ValueError(
                "deployment.target=sagemaker_endpoint requires "
                "deployment.sagemaker_endpoint to be set"
            )
        if self.target == "vertex_ai_endpoint" and self.vertex_ai_endpoint is None:
            raise ValueError(
                "deployment.target=vertex_ai_endpoint requires "
                "deployment.vertex_ai_endpoint to be set"
            )
        compatible = _STRATEGY_TARGET_COMPAT.get(self.strategy, set())
        if self.target not in compatible:
            raise ValueError(
                f"deployment.strategy={self.strategy!r} is not compatible "
                f"with deployment.target={self.target!r}. "
                f"Compatible targets for {self.strategy!r}: {sorted(compatible)}"
            )
        return self


# ── Cost monitor ───────────────────────────────────────────────────


class CostMonitorConfig(_Base):
    _KNOWN_METRICS: frozenset[str] = frozenset(
        {
            "inference_latency_ms",
            "latency_p99_ms",
            "throughput_rps",
            "cost_per_prediction",
            "cost_per_1k_predictions",
            "compute_utilisation_pct",
            "error_rate",
        }
    )

    track: list[str] = Field(
        default_factory=lambda: [
            "inference_latency_ms",
            "throughput_rps",
            "cost_per_prediction",
            "compute_utilisation_pct",
        ]
    )
    alert_thresholds: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_threshold_keys(self) -> CostMonitorConfig:
        unknown = set(self.alert_thresholds) - self._KNOWN_METRICS
        if unknown:
            raise ValueError(
                f"unknown cost metric(s) in alert_thresholds: {sorted(unknown)}. "
                f"Known metrics: {sorted(self._KNOWN_METRICS)}"
            )
        return self


# ── KPI / Audit ────────────────────────────────────────────────────


class KPIMapping(_Base):
    model_metric: str
    business_kpi: str
    data_source: str | None = None


class BusinessKPIConfig(_Base):
    mappings: list[KPIMapping] = Field(default_factory=list)


class AzureBlobAuditConfig(_Base):
    """Destination for the Azure Blob audit shipper.

    Required when ``AuditConfig.storage = "azure_blob"``. The shipper
    uses :class:`azure.identity.DefaultAzureCredential` at runtime —
    there are no credentials in the schema.
    """

    account_url: str
    container_name: str
    prefix: str = "sentinel-audit"
    delete_local_after_ship: bool = False


class S3AuditConfig(_Base):
    """Destination for the S3 audit shipper.

    Required when ``AuditConfig.storage = "s3"``.
    """

    bucket: str
    prefix: str = "sentinel-audit"
    region: str | None = None
    delete_local_after_ship: bool = False


class GcsAuditConfig(_Base):
    """Required when ``audit.storage = "gcs"``."""

    bucket: str
    prefix: str = "sentinel-audit"
    project: str | None = None
    delete_local_after_ship: bool = False


class AuditConfig(_Base):
    storage: Literal["local", "azure_blob", "s3", "gcs"] = "local"
    path: str = "./audit/"
    retention_days: int = 2555
    log_predictions: bool = False
    log_explanations: bool = False
    compliance_frameworks: list[str] = Field(default_factory=list)
    compliance_risk_level: str = "high"  # high | limited | minimal | unacceptable
    # Workstream #3 — audit hash-chain tamper evidence.
    #
    # When ``tamper_evidence: true`` the :class:`AuditTrail` attaches
    # an HMAC-SHA256 + ``previous_hash`` to every event it writes, and
    # ``sentinel audit verify`` can detect insertions, deletions, or
    # edits. Exactly one of ``signing_key_env`` or ``signing_key_path``
    # must resolve to bytes at client-construction time.
    tamper_evidence: bool = False
    signing_key_env: str = "SENTINEL_AUDIT_KEY"
    signing_key_path: str | None = None
    # Workstream #2 — optional shipper destinations. Exactly one
    # matching sub-config must be present when ``storage`` is not
    # ``local``.
    azure_blob: AzureBlobAuditConfig | None = None
    s3: S3AuditConfig | None = None
    gcs: GcsAuditConfig | None = None

    @model_validator(mode="after")
    def _validate_storage_requirements(self) -> AuditConfig:
        if self.storage == "azure_blob" and self.azure_blob is None:
            raise ValueError("audit.storage=azure_blob requires audit.azure_blob to be set")
        if self.storage == "s3" and self.s3 is None:
            raise ValueError("audit.storage=s3 requires audit.s3 to be set")
        if self.storage == "gcs" and self.gcs is None:
            raise ValueError("audit.storage=gcs requires audit.gcs to be set")
        return self


class ModelGraphEdge(_Base):
    upstream: str
    downstream: str


class ModelGraphConfig(_Base):
    dependencies: list[ModelGraphEdge] = Field(default_factory=list)
    cascade_alerts: bool = True


# ── Registry ───────────────────────────────────────────────────────


class RegistryConfig(_Base):
    """Selects the backend that stores model versions.

    The default ``backend="local"`` is backward compatible with every
    existing config — it puts model metadata on the local filesystem
    under ``path``. Operators who want durable storage point the
    backend at Azure ML or MLflow, which only need the relevant
    credentials (resolved from env vars or Key Vault).
    """

    backend: Literal[
        "local", "azure_ml", "mlflow", "sagemaker", "vertex_ai", "databricks"
    ] = "local"
    # Local backend
    path: str = "./registry"
    # Azure ML backend
    subscription_id: str | None = None
    resource_group: str | None = None
    workspace_name: str | None = None
    # MLflow backend
    tracking_uri: str | None = None
    # SageMaker backend
    region_name: str | None = None
    role_arn: str | None = None
    s3_bucket: str | None = None
    # Vertex AI backend
    project: str | None = None
    location: str | None = None
    gcs_bucket: str | None = None
    # Databricks backend
    host: str | None = None
    token: str | None = None
    catalog: str | None = None
    schema_name: str | None = None
    # Artifact storage (WS-A)
    serialize_artifacts: bool = False
    serializer: Literal["joblib", "pickle", "onnx", "auto"] = "auto"

    @model_validator(mode="after")
    def _validate_backend_requirements(self) -> RegistryConfig:
        if self.backend == "azure_ml":
            missing = [
                name
                for name, value in (
                    ("subscription_id", self.subscription_id),
                    ("resource_group", self.resource_group),
                    ("workspace_name", self.workspace_name),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    "registry.backend=azure_ml requires "
                    f"registry.{', registry.'.join(missing)} to be set"
                )
        # MLflow can fall back to MLFLOW_TRACKING_URI env var, so we do
        # not require ``tracking_uri`` at validation time — the backend
        # constructor handles it.
        return self


# ── LLMOps ─────────────────────────────────────────────────────────


class PromptRegistryConfig(_Base):
    registry_backend: Literal["local", "azure_blob", "s3"] = "local"
    versioning: Literal["semantic", "timestamp", "git_hash"] = "semantic"
    ab_testing: dict[str, Any] = Field(default_factory=dict)


class GuardrailRuleConfig(_Base):
    type: str
    action: Literal["block", "warn", "redact"] = "warn"
    threshold: float | None = None
    method: str | None = None
    critical: bool = False  # If True, pipeline init fails if this guardrail can't load

    # Custom guardrail DSL fields (used when type="custom")
    name: str | None = None
    rules: list[dict[str, Any]] = Field(default_factory=list)
    combine: Literal["all", "any"] = "all"

    # Plugin guardrail fields (used when type="plugin")
    module: str | None = None
    class_name: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _validate_custom_or_plugin(self) -> GuardrailRuleConfig:
        """Validate that custom/plugin guardrails have required fields."""
        if self.type == "custom":
            if not self.rules:
                raise ValueError(
                    "Custom guardrail requires at least one rule in 'rules' list."
                )
            if not self.name:
                raise ValueError("Custom guardrail requires a 'name' field.")
        if self.type == "plugin" and (not self.module or not self.class_name):
            raise ValueError(
                "Plugin guardrail requires both 'module' and 'class_name' fields."
            )
        return self


class GuardrailsConfig(_Base):
    input: list[GuardrailRuleConfig] = Field(default_factory=list)
    output: list[GuardrailRuleConfig] = Field(default_factory=list)


class QualityEvaluatorConfig(_Base):
    method: Literal["llm_judge", "heuristic", "reference_based", "hybrid"] = "heuristic"
    judge_model: str | None = None
    rubrics: dict[str, dict[str, Any]] = Field(default_factory=dict)
    sample_rate: float = 0.1


class SemanticDriftConfig(_Base):
    embedding_model: str = "text-embedding-3-small"
    window: str = "7d"
    threshold: float = 0.15
    window_size: int = 500  # Rolling window size for embedding observations


class RetrievalQualityConfig(_Base):
    track: list[str] = Field(default_factory=list)
    min_relevance: float = 0.5
    min_faithfulness: float = 0.7


class LLMQualityConfig(_Base):
    evaluator: QualityEvaluatorConfig = Field(default_factory=QualityEvaluatorConfig)
    semantic_drift: SemanticDriftConfig = Field(default_factory=SemanticDriftConfig)
    retrieval_quality: RetrievalQualityConfig = Field(default_factory=RetrievalQualityConfig)


class TokenEconomicsConfig(_Base):
    track_by: list[str] = Field(default_factory=lambda: ["model"])
    budgets: dict[str, float] = Field(default_factory=dict)
    alerts: dict[str, float] = Field(default_factory=dict)
    model_routing: dict[str, Any] = Field(default_factory=dict)
    pricing: dict[str, dict[str, float]] = Field(default_factory=dict)


class PromptDriftConfig(_Base):
    detection_window: str = "7d"
    signals: dict[str, float] = Field(default_factory=dict)
    min_samples: int = 20  # Minimum observations before drift detection activates


class LLMOpsConfig(_Base):
    enabled: bool = False
    mode: Literal["rag", "completion", "chat", "agent"] = "completion"
    prompts: PromptRegistryConfig = Field(default_factory=PromptRegistryConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    quality: LLMQualityConfig = Field(default_factory=LLMQualityConfig)
    token_economics: TokenEconomicsConfig = Field(default_factory=TokenEconomicsConfig)
    prompt_drift: PromptDriftConfig = Field(default_factory=PromptDriftConfig)


# ── AgentOps ───────────────────────────────────────────────────────


class TracingConfig(_Base):
    backend: Literal["local", "otlp", "arize_phoenix"] = "local"
    otlp_endpoint: str | None = None
    sample_rate: float = 1.0
    export_format: Literal["json", "protobuf"] = "json"
    retention_days: int = 90
    auto_instrument: dict[str, bool] = Field(default_factory=dict)


class ToolAuditConfig(_Base):
    permissions: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    parameter_validation: bool = True
    rate_limits: dict[str, str] = Field(default_factory=dict)
    replay: dict[str, Any] = Field(default_factory=dict)


class LoopDetectionConfig(_Base):
    max_iterations: int = 50
    max_repeated_tool_calls: int = 5
    max_delegation_depth: int = 5
    thrash_window: int = 10


class BudgetConfig(_Base):
    max_tokens_per_run: int = 50000
    max_cost_per_run: float = 5.0
    max_time_per_run: str = "300s"
    max_tool_calls_per_run: int = 30
    on_exceeded: Literal["graceful_stop", "escalate", "hard_kill"] = "graceful_stop"

    @field_validator("max_time_per_run")
    @classmethod
    def _check_time(cls, v: str) -> str:
        return _validate_interval(v)


class EscalationTrigger(_Base):
    condition: str
    threshold: float | None = None
    patterns: list[str] = Field(default_factory=list)
    action: Literal["human_handoff", "human_approval", "block"] = "human_handoff"


class EscalationConfig(_Base):
    triggers: list[EscalationTrigger] = Field(default_factory=list)


class SandboxConfig(_Base):
    destructive_ops: list[str] = Field(default_factory=list)
    mode: Literal["approve_first", "dry_run", "sandbox_then_apply"] = "approve_first"


class SafetyConfig(_Base):
    loop_detection: LoopDetectionConfig = Field(default_factory=LoopDetectionConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)


class AgentRegistryConfig(_Base):
    backend: Literal["local", "azure_blob", "database"] = "local"
    capability_manifest: bool = True
    health_check_interval: str = "60s"
    a2a: dict[str, Any] = Field(default_factory=dict)

    @field_validator("health_check_interval")
    @classmethod
    def _check_interval(cls, v: str) -> str:
        return _validate_interval(v)


class MultiAgentConfig(_Base):
    delegation_tracking: bool = True
    consensus: dict[str, Any] = Field(default_factory=dict)
    bottleneck_detection: dict[str, Any] = Field(default_factory=dict)


class AgentEvaluationConfig(_Base):
    golden_datasets: dict[str, Any] = Field(default_factory=dict)
    task_completion: dict[str, Any] = Field(default_factory=dict)
    trajectory: dict[str, Any] = Field(default_factory=dict)


class AgentOpsConfig(_Base):
    enabled: bool = False
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    tool_audit: ToolAuditConfig = Field(default_factory=ToolAuditConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    agent_registry: AgentRegistryConfig = Field(default_factory=AgentRegistryConfig)
    multi_agent: MultiAgentConfig = Field(default_factory=MultiAgentConfig)
    evaluation: AgentEvaluationConfig = Field(default_factory=AgentEvaluationConfig)


# ── Dashboard ──────────────────────────────────────────────────────


class RBACUserBinding(_Base):
    """Maps a username (basic auth or JWT claim) to a list of roles."""

    username: str
    roles: list[str] = Field(default_factory=list)


def _default_role_permissions() -> dict[str, list[str]]:
    """Default role → permission matrix used when none is configured."""
    return {
        "viewer": [
            "drift.read",
            "features.read",
            "registry.read",
            "audit.read",
            "llmops.read",
            "agentops.read",
            "deployments.read",
            "compliance.read",
        ],
        "operator": [
            # Inherits viewer permissions via transitive closure in
            # :class:`sentinel.dashboard.security.rbac.RBACPolicy`.
            "audit.verify",
            "deployments.promote",
            "deployments.write",
            "retrain.trigger",
            "golden.run",
        ],
        "admin": [
            # Admin inherits operator (+ viewer). Admin-only permissions
            # are wildcarded via the ``*`` token.
            "*",
        ],
    }


class RBACConfig(_Base):
    """Role-based access control for the dashboard.

    When ``enabled`` is False (the default), every authenticated user
    gets the ``default_role`` implicitly and per-route permission
    checks behave as no-ops — the local-first dev experience is
    unchanged. Flip ``enabled: true`` to wire the full principal →
    permission → route dependency chain.
    """

    enabled: bool = False
    default_role: str = "viewer"
    users: list[RBACUserBinding] = Field(default_factory=list)
    role_permissions: dict[str, list[str]] = Field(default_factory=_default_role_permissions)
    # Ordered list of roles from least to most privileged. Higher roles
    # automatically gain the permissions of every lower role.
    role_hierarchy: list[str] = Field(
        default_factory=lambda: ["viewer", "operator", "admin"],
    )


class BearerAuthConfig(_Base):
    """Config for validating ``Authorization: Bearer ...`` JWTs.

    Only a JWKS-URL flow is supported. No Authorization Code + PKCE,
    no session cookies, no reverse-proxy header trust — those are
    explicitly out of scope for workstream #3.
    """

    jwks_url: str | None = None
    issuer: str | None = None
    audience: str | None = None
    username_claim: str = "sub"
    roles_claim: str = "roles"
    algorithms: list[str] = Field(default_factory=lambda: ["RS256"])
    cache_ttl_seconds: int = 3600
    leeway_seconds: int = 30


class CSRFConfig(_Base):
    """Double-submit cookie CSRF protection settings."""

    enabled: bool = True
    cookie_name: str = "sentinel_csrf"
    header_name: str = "X-CSRF-Token"
    cookie_secure: bool | None = None  # None = auto-detect from scheme
    cookie_samesite: Literal["lax", "strict", "none"] = "lax"


class RateLimitConfig(_Base):
    """In-memory token-bucket rate limit for the dashboard."""

    enabled: bool = True
    default_per_minute: int = 100
    api_per_minute: int = 300
    auth_per_minute: int = 10
    burst_multiplier: float = 2.0


class CSPConfig(_Base):
    """Content Security Policy tuning."""

    enabled: bool = True
    policy: str | None = None  # None = use built-in dashboard-friendly default


class DashboardServerConfig(_Base):
    """HTTP server settings for the optional dashboard UI."""

    host: str = "127.0.0.1"
    port: int = 8000
    root_path: str = ""  # for reverse-proxy mounts
    auth: Literal["none", "basic", "bearer"] = "none"
    basic_auth_username: str | None = None
    # Stored as SecretStr — never appears in repr or masked dumps. The
    # dashboard's ``basic_auth_guard`` dependency unwraps it via
    # :func:`sentinel.config.secrets.unwrap` at request time.
    basic_auth_password: SecretStr | None = None

    # Workstream #3 — security hardening.
    bearer: BearerAuthConfig = Field(default_factory=BearerAuthConfig)
    rbac: RBACConfig = Field(default_factory=RBACConfig)
    csrf: CSRFConfig = Field(default_factory=CSRFConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    csp: CSPConfig = Field(default_factory=CSPConfig)
    require_signed_config: bool = False

    @model_validator(mode="after")
    def _validate_auth_mode(self) -> DashboardServerConfig:
        if self.auth == "bearer" and not self.bearer.jwks_url:
            raise ValueError(
                "dashboard.server.auth=bearer requires dashboard.server.bearer.jwks_url"
            )
        if self.auth == "basic" and (not self.basic_auth_username or not self.basic_auth_password):
            raise ValueError(
                "dashboard.server.auth=basic requires both "
                "basic_auth_username and basic_auth_password"
            )
        return self


class DashboardUIConfig(_Base):
    """Presentational settings for the dashboard UI."""

    title: str = "Project Sentinel"
    theme: Literal["light", "dark", "auto"] = "auto"
    show_modules: list[str] = Field(
        default_factory=lambda: [
            "overview",
            "drift",
            "features",
            "registry",
            "experiments",
            "audit",
            "llmops",
            "agentops",
            "deployments",
            "retraining",
            "intelligence",
            "compliance",
        ]
    )
    refresh_interval_seconds: int = 30


class DashboardConfig(_Base):
    """Top-level dashboard configuration block."""

    enabled: bool = False
    server: DashboardServerConfig = Field(default_factory=DashboardServerConfig)
    ui: DashboardUIConfig = Field(default_factory=DashboardUIConfig)


# ── Domains ────────────────────────────────────────────────────────


class DomainConfig(_Base):
    """Domain-specific overrides keyed by domain name."""

    timeseries: dict[str, Any] = Field(default_factory=dict)
    nlp: dict[str, Any] = Field(default_factory=dict)
    recommendation: dict[str, Any] = Field(default_factory=dict)
    graph: dict[str, Any] = Field(default_factory=dict)
    tabular: dict[str, Any] = Field(default_factory=dict)


# ── Datasets ───────────────────────────────────────────────────────


class DatasetConfig(_Base):
    """Configuration for the dataset metadata registry."""

    registry_path: str = "./datasets"
    auto_hash: bool = True
    require_schema: bool = False


# ── Experiments ────────────────────────────────────────────────────


class ExperimentConfig(_Base):
    """Configuration for the experiment tracker."""

    storage_path: str = "./experiments"
    auto_log: bool = True
    nested_runs: bool = True
    max_metric_history: int = 10000


# ── Root ───────────────────────────────────────────────────────────


class SentinelConfig(_Base):
    """Root configuration for a Sentinel-managed model."""

    version: str = "1.0"
    extends: str | None = None  # path to base config to inherit from
    model: ModelConfig
    data_quality: DataQualityConfig = Field(default_factory=DataQualityConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    feature_health: FeatureHealthConfig = Field(default_factory=FeatureHealthConfig)
    cohort_analysis: CohortAnalysisConfig = Field(default_factory=CohortAnalysisConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    retraining: RetrainingConfig = Field(default_factory=RetrainingConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    cost_monitor: CostMonitorConfig = Field(default_factory=CostMonitorConfig)
    business_kpi: BusinessKPIConfig = Field(default_factory=BusinessKPIConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    registry: RegistryConfig = Field(default_factory=RegistryConfig)
    model_graph: ModelGraphConfig = Field(default_factory=ModelGraphConfig)
    llmops: LLMOpsConfig = Field(default_factory=LLMOpsConfig)
    agentops: AgentOpsConfig = Field(default_factory=AgentOpsConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    domains: DomainConfig = Field(default_factory=DomainConfig)
    datasets: DatasetConfig = Field(default_factory=DatasetConfig)
    experiments: ExperimentConfig = Field(default_factory=ExperimentConfig)

    @field_validator("version")
    @classmethod
    def _check_version(cls, v: str) -> str:
        if not v:
            raise ValueError("config.version is required")
        return v
