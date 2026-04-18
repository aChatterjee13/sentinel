"""Shared type definitions and data transfer objects.

All DTOs are immutable Pydantic models. They are produced by Sentinel modules
and consumed by user code, the audit trail, and the notification engine.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def _now() -> datetime:
    """Return a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _uid() -> str:
    """Return a short unique identifier."""
    return uuid4().hex[:16]


class _Frozen(BaseModel):
    """Base for immutable DTOs."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# ── Severity ───────────────────────────────────────────────────────


class AlertSeverity(str, Enum):
    """Severity levels for alerts and reports."""

    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_score(cls, score: float, thresholds: dict[str, float] | None = None) -> AlertSeverity:
        """Map a numeric score to a severity using configurable thresholds."""
        thresholds = thresholds or {"warning": 0.1, "high": 0.2, "critical": 0.3}
        if score >= thresholds.get("critical", 0.3):
            return cls.CRITICAL
        if score >= thresholds.get("high", 0.2):
            return cls.HIGH
        if score >= thresholds.get("warning", 0.1):
            return cls.WARNING
        return cls.INFO


# ── Drift ──────────────────────────────────────────────────────────


class DriftReport(_Frozen):
    """Result of a drift detection run.

    Attributes:
        is_drifted: True when at least one feature exceeds its drift threshold.
        severity: Categorical severity derived from the worst score.
        method: Name of the statistical method used (psi, ks, ddm, ...).
        test_statistic: Aggregate test statistic across features.
        p_value: Optional p-value when the method is a hypothesis test.
        feature_scores: Per-feature drift scores keyed by column name.
        drifted_features: Names of features that crossed their thresholds.
        timestamp: When the report was produced (UTC).
        window: Description of the comparison window (e.g. ``"7d"``).
        metadata: Free-form context for downstream consumers.
    """

    report_id: str = Field(default_factory=_uid)
    model_name: str
    method: str
    is_drifted: bool
    severity: AlertSeverity
    test_statistic: float
    p_value: float | None = None
    feature_scores: dict[str, float] = Field(default_factory=dict)
    drifted_features: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_now)
    window: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def summary(self) -> str:
        """Short human-readable summary."""
        if not self.is_drifted:
            return f"stable ({self.method})"
        return f"drifted: {len(self.drifted_features)} features ({self.severity.value})"


# ── Data quality ───────────────────────────────────────────────────


class QualityIssue(_Frozen):
    """A single data quality finding."""

    feature: str | None
    rule: str
    severity: AlertSeverity
    message: str
    count: int = 0


class QualityReport(_Frozen):
    """Result of a data quality check.

    Attributes:
        profile: Per-feature statistics computed from the checked data.
            Each key is a feature name mapping to a dict with ``type``,
            ``null_rate``, ``mean``, ``std``, ``min``, ``max``, and
            ``unique_count``.
    """

    report_id: str = Field(default_factory=_uid)
    model_name: str
    is_valid: bool
    issues: list[QualityIssue] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_now)
    rows_checked: int = 0
    profile: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @property
    def has_critical_issues(self) -> bool:
        """True when any issue has critical severity."""
        return any(i.severity == AlertSeverity.CRITICAL for i in self.issues)

    @property
    def summary(self) -> str:
        """Short human-readable summary."""
        if self.is_valid:
            return f"ok ({self.rows_checked} rows)"
        return f"{len(self.issues)} issues across {self.rows_checked} rows"


# ── Feature health ─────────────────────────────────────────────────


class FeatureHealth(_Frozen):
    """Per-feature health snapshot."""

    name: str
    importance: float
    drift_score: float
    null_rate: float
    is_drifted: bool
    severity: AlertSeverity


class FeatureHealthReport(_Frozen):
    """Aggregated feature health across all features."""

    report_id: str = Field(default_factory=_uid)
    model_name: str
    features: list[FeatureHealth] = Field(default_factory=list)
    top_n_drifted: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=_now)

    @property
    def summary(self) -> str:
        """Short human-readable summary."""
        n_drifted = sum(1 for f in self.features if f.is_drifted)
        return f"{n_drifted}/{len(self.features)} features drifted"


# ── Predictions ────────────────────────────────────────────────────


class PredictionRecord(_Frozen):
    """A single logged prediction with optional ground truth."""

    prediction_id: str = Field(default_factory=_uid)
    record_id: str = Field(default_factory=_uid)
    model_name: str
    model_version: str | None = None
    features: dict[str, Any] = Field(default_factory=dict)
    prediction: Any = None
    actual: Any = None
    confidence: float | None = None
    explanation: dict[str, float] | None = None
    timestamp: datetime = Field(default_factory=_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Alerts ─────────────────────────────────────────────────────────


class Alert(_Frozen):
    """A notification dispatched by Sentinel."""

    alert_id: str = Field(default_factory=_uid)
    model_name: str
    title: str
    body: str
    severity: AlertSeverity
    source: str
    timestamp: datetime = Field(default_factory=_now)
    payload: dict[str, Any] = Field(default_factory=dict)
    fingerprint: str | None = None  # used for cooldown deduplication


class DeliveryResult(_Frozen):
    """Outcome of a single notification channel delivery attempt."""

    channel: str
    delivered: bool
    timestamp: datetime = Field(default_factory=_now)
    error: str | None = None
    response: dict[str, Any] | None = None


# ── Cost / performance ─────────────────────────────────────────────


class CostMetrics(_Frozen):
    """Latency, throughput, and cost telemetry."""

    model_name: str
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    throughput_rps: float
    cost_per_prediction: float
    compute_utilisation_pct: float
    sample_count: int = 0
    error_rate: float = 0.0
    error_count: int = 0
    success_count: int = 0
    timestamp: datetime = Field(default_factory=_now)


# ── LLM types ──────────────────────────────────────────────────────


class GuardrailResult(_Frozen):
    """Outcome of a single guardrail check."""

    name: str
    passed: bool
    blocked: bool
    severity: AlertSeverity
    score: float | None = None
    reason: str | None = None
    sanitised_content: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineResult(_Frozen):
    """Aggregated result of a guardrail pipeline run."""

    blocked: bool
    results: list[GuardrailResult] = Field(default_factory=list)
    sanitised_input: str | None = None
    reason: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when nothing in the pipeline blocked."""
        return not self.blocked


class LLMUsage(_Frozen):
    """Per-call LLM usage metrics."""

    input_tokens: int
    output_tokens: int
    model: str
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    cached: bool = False


class QualityScore(_Frozen):
    """Result of a quality evaluator."""

    overall: float
    rubric_scores: dict[str, float] = Field(default_factory=dict)
    method: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Agent types ────────────────────────────────────────────────────


class SpanStatus(str, Enum):
    """OpenTelemetry-aligned span status codes."""

    OK = "ok"
    ERROR = "error"
    CANCELLED = "cancelled"


class Span(_Frozen):
    """A single span in an agent trace."""

    span_id: str = Field(default_factory=_uid)
    parent_id: str | None = None
    name: str
    kind: str  # plan, tool_call, llm_call, delegation, ...
    start_time: datetime
    end_time: datetime | None = None
    status: SpanStatus = SpanStatus.OK
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        """Span duration in milliseconds, if ended."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds() * 1000.0


class AgentTrace(_Frozen):
    """Complete trace of a single agent run."""

    trace_id: str = Field(default_factory=_uid)
    agent_name: str
    started_at: datetime = Field(default_factory=_now)
    ended_at: datetime | None = None
    spans: list[Span] = Field(default_factory=list)
    total_tokens: int = 0
    total_cost: float = 0.0
    tool_call_count: int = 0
    status: SpanStatus = SpanStatus.OK
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def step_count(self) -> int:
        """Number of spans in the trace."""
        return len(self.spans)

    @property
    def duration_ms(self) -> float | None:
        """Total wall-clock duration of the trace."""
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds() * 1000.0


# ── Explainability ─────────────────────────────────────────────────


class ExplanationReport(_Frozen):
    """Row-level feature attributions for a batch of predictions.

    Attributes:
        model_name: Name of the model being explained.
        attributions: Per-row feature→attribution mapping.
        method: Explanation method used (``"shap"`` or ``"permutation"``).
    """

    report_id: str = Field(default_factory=_uid)
    model_name: str
    attributions: list[dict[str, float]] = Field(default_factory=list)
    method: str = "shap"
    timestamp: datetime = Field(default_factory=_now)

    @property
    def summary(self) -> str:
        """Short human-readable summary."""
        return f"{len(self.attributions)} rows explained ({self.method})"


class GlobalExplanationReport(_Frozen):
    """Aggregated feature importance across all predictions.

    Attributes:
        feature_importance: Mean absolute attribution per feature, sorted descending.
        sample_count: Number of predictions the aggregation is based on.
    """

    report_id: str = Field(default_factory=_uid)
    model_name: str
    feature_importance: dict[str, float] = Field(default_factory=dict)
    sample_count: int = 0
    method: str = "shap"
    timestamp: datetime = Field(default_factory=_now)

    @property
    def ranked_features(self) -> list[tuple[str, float]]:
        """Features ranked by importance (descending)."""
        return sorted(self.feature_importance.items(), key=lambda kv: kv[1], reverse=True)

    @property
    def summary(self) -> str:
        """Short human-readable summary."""
        top = self.ranked_features[:3]
        names = ", ".join(n for n, _ in top)
        return f"top features: {names} (n={self.sample_count})"


class CohortExplanationReport(_Frozen):
    """Per-cohort aggregated feature importance with cross-cohort comparison.

    Attributes:
        cohort_importances: Mapping of cohort_id → {feature → mean |attribution|}.
        cohort_counts: Number of samples per cohort.
        cross_cohort_std: Per-feature standard deviation across cohort means,
            highlighting features whose importance varies most between cohorts.
    """

    report_id: str = Field(default_factory=_uid)
    model_name: str
    cohort_importances: dict[str, dict[str, float]] = Field(default_factory=dict)
    cohort_counts: dict[str, int] = Field(default_factory=dict)
    cross_cohort_std: dict[str, float] = Field(default_factory=dict)
    method: str = "shap"
    timestamp: datetime = Field(default_factory=_now)

    @property
    def most_variable_features(self) -> list[tuple[str, float]]:
        """Features whose importance varies most across cohorts (descending)."""
        return sorted(self.cross_cohort_std.items(), key=lambda kv: kv[1], reverse=True)

    @property
    def summary(self) -> str:
        """Short human-readable summary."""
        n = len(self.cohort_importances)
        return f"{n} cohorts compared"


# ── Cohort analysis ────────────────────────────────────────────────


class CohortMetrics(_Frozen):
    """Performance and health metrics for a single prediction cohort.

    Attributes:
        cohort_id: Identifier for this cohort (e.g. ``"age_30_40"``).
        count: Number of predictions in this cohort.
        mean_prediction: Average predicted value.
        mean_actual: Average ground truth (None if actuals not available).
        drift_score: Aggregate drift score for this cohort's feature distribution.
        accuracy: Classification accuracy (None if not classification or no actuals).
        feature_health: Per-feature health within this cohort.
    """

    cohort_id: str
    count: int
    mean_prediction: float | None = None
    mean_actual: float | None = None
    drift_score: float = 0.0
    accuracy: float | None = None
    feature_health: list[FeatureHealth] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CohortPerformanceReport(_Frozen):
    """Performance report for a single cohort.

    Attributes:
        cohort_id: Cohort this report covers.
        metrics: Aggregated metrics for the cohort.
        drift_report: Optional drift report scoped to this cohort.
    """

    report_id: str = Field(default_factory=_uid)
    model_name: str
    cohort_id: str
    metrics: CohortMetrics
    timestamp: datetime = Field(default_factory=_now)


class CohortComparativeReport(_Frozen):
    """Cross-cohort comparison highlighting performance disparity.

    Attributes:
        cohorts: Per-cohort metrics.
        disparity_flags: Cohort IDs where performance deviates significantly
            from the global mean (e.g. accuracy >10% below average).
        global_metrics: Baseline metrics across all cohorts combined.
    """

    report_id: str = Field(default_factory=_uid)
    model_name: str
    cohorts: list[CohortMetrics] = Field(default_factory=list)
    disparity_flags: list[str] = Field(default_factory=list)
    global_mean_prediction: float | None = None
    global_accuracy: float | None = None
    timestamp: datetime = Field(default_factory=_now)

    @property
    def summary(self) -> str:
        """Short human-readable summary."""
        n = len(self.cohorts)
        d = len(self.disparity_flags)
        if d:
            return f"{d}/{n} cohorts flagged for disparity"
        return f"{n} cohorts, no disparity detected"


# ── Audit ──────────────────────────────────────────────────────────


class AuditEvent(_Frozen):
    """An immutable audit log entry.

    When the audit trail is configured with ``tamper_evidence: true``,
    every event carries two extra fields that together form a
    cryptographic hash chain:

    - ``previous_hash``: the ``event_hmac`` of the event immediately
      preceding this one (None for the first event in a fresh chain).
    - ``event_hmac``: HMAC-SHA256 over the canonical JSON of every
      other field, computed at write time with the audit signing key.

    Both default to ``None`` so JSON-lines files produced by earlier
    versions of Sentinel load unchanged.
    """

    event_id: str = Field(default_factory=_uid)
    event_type: str
    model_name: str | None = None
    model_version: str | None = None
    actor: str | None = None
    timestamp: datetime = Field(default_factory=_now)
    payload: dict[str, Any] = Field(default_factory=dict)
    previous_hash: str | None = None
    event_hmac: str | None = None
