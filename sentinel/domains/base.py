"""BaseDomainAdapter — interface that all domain adapters must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from sentinel.config.schema import SentinelConfig
from sentinel.observability.drift.base import BaseDriftDetector

DomainName = Literal["tabular", "timeseries", "nlp", "recommendation", "graph"]


class BaseDomainAdapter(ABC):
    """Interface for a domain-specific monitoring adapter.

    Each adapter binds a domain name (``tabular``, ``timeseries``, ``nlp``,
    ``recommendation``, ``graph``) to drift detectors, quality metrics, and
    schema validators tailored for that paradigm. The core SDK delegates
    domain-specific behaviour through this interface so that switching
    domains is a YAML config change rather than a code change.
    """

    domain: DomainName = "tabular"

    def __init__(self, config: SentinelConfig):
        self.config = config
        self.model_name = config.model.name
        self.options: dict[str, Any] = self._domain_options()

    def _domain_options(self) -> dict[str, Any]:
        """Pull the per-domain options block from the root config."""
        return getattr(self.config.domains, self.domain, {}) or {}

    # ── Required hooks ───────────────────────────────────────────

    @abstractmethod
    def get_drift_detectors(self) -> list[BaseDriftDetector]:
        """Return the drift detectors appropriate for this domain."""

    @abstractmethod
    def get_quality_metrics(self) -> list[Any]:
        """Return the quality metric calculators appropriate for this domain."""

    @abstractmethod
    def get_schema_validator(self) -> Any:
        """Return a schema validator appropriate for this domain's inputs."""

    # ── Optional hooks ───────────────────────────────────────────

    def describe(self) -> dict[str, Any]:
        """Short description of the adapter — used by status endpoints."""
        return {
            "domain": self.domain,
            "model": self.model_name,
            "options": self.options,
            "drift_detectors": [d.method_name for d in self.get_drift_detectors()],
            "quality_metrics": [
                getattr(m, "name", type(m).__name__) for m in self.get_quality_metrics()
            ],
        }
