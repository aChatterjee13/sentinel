"""Pydantic response models for the dashboard JSON API.

These models drive the OpenAPI/Swagger documentation and provide
runtime response validation for key endpoints. Complex dict-shaped
endpoints that vary by configuration still return ``dict[str, Any]``
but carry rich ``description`` strings in their route decorators.

NOTE: This module must NOT use ``from __future__ import annotations``
because FastAPI needs concrete type resolution for ``response_model``
serialisation (same constraint as ``api.py``).
"""

from pydantic import BaseModel, Field

# ── Overview / Health ────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Service liveness check."""

    status: str = Field(description="Always 'ok' when the service is running")
    model: str = Field(description="Name of the monitored model")
    version: str | None = Field(description="Current model version (null if none registered)")
    started_at: str = Field(description="ISO-8601 timestamp when the dashboard started")


# ── Drift ────────────────────────────────────────────────────────────


class DriftSummary(BaseModel):
    """High-level drift status returned by /api/drift."""

    total_reports: int = Field(description="Total number of drift reports on record")
    drifted_count: int = Field(description="Number of reports that detected drift")
    latest_check: str | None = Field(
        default=None, description="ISO-8601 timestamp of the most recent drift check"
    )


# ── Experiments ──────────────────────────────────────────────────────


class ExperimentSummary(BaseModel):
    """Summary of a single experiment in the list view."""

    name: str = Field(description="Experiment name")
    run_count: int = Field(description="Number of runs in this experiment")
    latest_run: str | None = Field(
        default=None, description="ISO-8601 timestamp of the most recent run"
    )
    tags: list[str] = Field(default_factory=list, description="User-defined tags")


class MetricPoint(BaseModel):
    """A single metric observation within a run."""

    value: float = Field(description="Metric value")
    step: int | None = Field(default=None, description="Training step (if applicable)")
    timestamp: str = Field(description="ISO-8601 timestamp when the metric was logged")


class RunComparison(BaseModel):
    """Side-by-side comparison of two experiment runs."""

    params_diff: dict = Field(description="Parameters that differ between the two runs")
    metrics_latest: dict = Field(description="Latest metric values per run")
    status: dict = Field(description="Run status for each run")


# ── Datasets ─────────────────────────────────────────────────────────


class DatasetVersionSummary(BaseModel):
    """Summary of a single dataset version."""

    name: str = Field(description="Dataset name")
    version: str = Field(description="Semantic version string")
    format: str = Field(description="Storage format (parquet, csv, delta, etc.)")
    num_rows: int | None = Field(default=None, description="Row count (if known)")
    num_features: int | None = Field(default=None, description="Feature count (if known)")


# ── Audit ────────────────────────────────────────────────────────────


class AuditChartData(BaseModel):
    """Aggregated audit event counts suitable for bar/pie charts."""

    labels: list[str] = Field(description="Event type labels")
    values: list[int] = Field(description="Event counts per type")
