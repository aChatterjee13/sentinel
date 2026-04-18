"""Detect when prompt effectiveness degrades over time."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from sentinel.config.schema import LLMOpsConfig, PromptDriftConfig
from sentinel.core.types import AlertSeverity, DriftReport

log = structlog.get_logger(__name__)


@dataclass
class PromptStats:
    """Rolling stats for a single prompt version."""

    quality_scores: deque[float] = field(default_factory=lambda: deque(maxlen=200))
    guardrail_violations: deque[int] = field(default_factory=lambda: deque(maxlen=200))
    token_usage: deque[int] = field(default_factory=lambda: deque(maxlen=200))
    semantic_distances: deque[float] = field(default_factory=lambda: deque(maxlen=200))


class PromptDriftDetector:
    """Composite prompt drift detector.

    Combines four signals:

    - Quality score decline (rolling mean delta)
    - Guardrail violation rate increase
    - Token usage trend (verbosity creep)
    - Semantic drift in outputs (delegated to :class:`SemanticDriftMonitor`)

    All public methods are thread-safe. A :class:`threading.Lock` serialises
    access to the mutable ``_stats`` dictionary.
    """

    def __init__(self, config: PromptDriftConfig | None = None):
        self.config = config or PromptDriftConfig()
        self._stats: dict[str, PromptStats] = {}
        self._lock = threading.Lock()
        self.signals = self.config.signals or {
            "quality_score_decline": 0.1,
            "guardrail_violation_rate_increase": 0.05,
            "token_usage_increase_pct": 25.0,
        }

    @classmethod
    def from_config(cls, config: LLMOpsConfig | str | Any) -> PromptDriftDetector:
        if isinstance(config, str):
            from sentinel.config.loader import load_config

            cfg = load_config(config)
            return cls(cfg.llmops.prompt_drift)
        if isinstance(config, LLMOpsConfig):
            return cls(config.prompt_drift)
        return cls(config.llmops.prompt_drift)  # type: ignore[union-attr]

    def observe(
        self,
        prompt_name: str,
        prompt_version: str,
        quality_score: float | None = None,
        guardrail_violations: int = 0,
        total_tokens: int = 0,
        semantic_distance: float | None = None,
    ) -> None:
        """Record one call's metrics for the given prompt version."""
        key = f"{prompt_name}@{prompt_version}"
        with self._lock:
            stats = self._stats.setdefault(key, PromptStats())
            if quality_score is not None:
                stats.quality_scores.append(quality_score)
            stats.guardrail_violations.append(guardrail_violations)
            stats.token_usage.append(total_tokens)
            if semantic_distance is not None:
                stats.semantic_distances.append(semantic_distance)

    @staticmethod
    def _ewma(values: list[float], alpha: float = 0.3) -> float:
        """Compute exponentially weighted moving average, emphasizing recent values.

        Args:
            values: Ordered observations (oldest first).
            alpha: Smoothing factor in ``(0, 1]``.  Higher values give more
                weight to recent observations.

        Returns:
            EWMA of *values*, or ``0.0`` if *values* is empty.
        """
        if not values:
            return 0.0
        ewma = values[0]
        for v in values[1:]:
            ewma = alpha * v + (1 - alpha) * ewma
        return ewma

    def detect(self, prompt_name: str, prompt_version: str) -> DriftReport:
        """Detect prompt drift for a specific prompt version.

        Args:
            prompt_name: Registered prompt name.
            prompt_version: The version string being monitored.

        Returns:
            A :class:`DriftReport` describing detected drift signals.
        """
        key = f"{prompt_name}@{prompt_version}"
        min_samples = self.config.min_samples
        with self._lock:
            stats = self._stats.get(key)
            if stats is None or len(stats.quality_scores) < min_samples:
                log.info(
                    "prompt_drift.insufficient_data",
                    prompt=prompt_name,
                    version=prompt_version,
                    samples=len(stats.quality_scores) if stats else 0,
                    required=min_samples,
                )
                return self._stable_report(prompt_name, prompt_version, "insufficient_data")

            # Snapshot mutable deques while holding the lock
            quality_scores = list(stats.quality_scores)
            guardrail_violations = list(stats.guardrail_violations)
            token_usage = list(stats.token_usage)
            semantic_distances = list(stats.semantic_distances)

        signals: dict[str, float] = {}
        drifted = []

        # Quality decline — compare early quarter mean vs recent quarter EWMA
        if quality_scores:
            quarter = max(1, len(quality_scores) // 4)
            early_mean = sum(quality_scores[:quarter]) / quarter
            recent_ewma = self._ewma(quality_scores[-quarter:])
            decline = early_mean - recent_ewma
            signals["quality_decline"] = decline
            if decline > self.signals.get("quality_score_decline", 0.1):
                drifted.append("quality_decline")

        # Guardrail violation rate — early quarter mean vs recent quarter EWMA
        if guardrail_violations:
            quarter = max(1, len(guardrail_violations) // 4)
            early_mean = sum(guardrail_violations[:quarter]) / quarter
            recent_ewma = self._ewma([float(v) for v in guardrail_violations[-quarter:]])
            rate_increase = recent_ewma - early_mean
            signals["guardrail_increase"] = rate_increase
            if rate_increase > self.signals.get("guardrail_violation_rate_increase", 0.05):
                drifted.append("guardrail_violations")

        # Token usage — early quarter mean vs recent quarter EWMA
        if token_usage:
            quarter = max(1, len(token_usage) // 4)
            early_mean = sum(token_usage[:quarter]) / quarter
            recent_ewma = self._ewma([float(v) for v in token_usage[-quarter:]])
            pct_increase = (recent_ewma - early_mean) / max(1, early_mean) * 100
            signals["token_usage_pct"] = pct_increase
            if pct_increase > self.signals.get("token_usage_increase_pct", 25.0):
                drifted.append("token_usage")

        # Semantic
        if semantic_distances:
            avg_dist = sum(semantic_distances) / len(semantic_distances)
            signals["semantic_distance"] = avg_dist
            if avg_dist > 0.2:
                drifted.append("semantic_drift")

        is_drifted = bool(drifted)
        severity = (
            AlertSeverity.HIGH
            if len(drifted) >= 2
            else (AlertSeverity.WARNING if drifted else AlertSeverity.INFO)
        )
        return DriftReport(
            model_name=prompt_name,
            method="prompt_drift",
            is_drifted=is_drifted,
            severity=severity,
            test_statistic=float(len(drifted)),
            feature_scores=signals,
            drifted_features=drifted,
            timestamp=datetime.now(timezone.utc),
            window=self.config.detection_window,
            metadata={"prompt_version": prompt_version},
        )

    def _stable_report(self, name: str, version: str, reason: str) -> DriftReport:
        return DriftReport(
            model_name=name,
            method="prompt_drift",
            is_drifted=False,
            severity=AlertSeverity.INFO,
            test_statistic=0.0,
            window=self.config.detection_window,
            metadata={"prompt_version": version, "reason": reason},
        )
