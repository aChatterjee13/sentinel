"""Latency, throughput, and cost-per-prediction tracking."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from sentinel.config.schema import CostMonitorConfig
from sentinel.core.types import CostMetrics

log = structlog.get_logger(__name__)


class CostMonitor:
    """Rolling-window tracker for inference latency and cost.

    Example:
        >>> monitor = CostMonitor(config, model_name="claims_fraud")
        >>> with monitor.time_request():
        ...     model.predict(X)
        >>> metrics = monitor.snapshot()
        >>> print(metrics.latency_ms_p95)
    """

    def __init__(
        self,
        config: CostMonitorConfig,
        model_name: str,
        window_size: int = 1000,
        cost_per_prediction: float = 0.0,
    ) -> None:
        self.config = config
        self.model_name = model_name
        self.cost_per_prediction = cost_per_prediction
        self._latencies_ms: deque[float] = deque(maxlen=window_size)
        self._timestamps: deque[float] = deque(maxlen=window_size)
        self._error_count: int = 0
        self._success_count: int = 0
        self._lock = threading.Lock()

    def record(self, latency_ms: float) -> None:
        """Record a single successful prediction latency."""
        with self._lock:
            self._latencies_ms.append(float(latency_ms))
            self._timestamps.append(time.monotonic())
            self._success_count += 1

    def record_error(self, latency_ms: float | None = None) -> None:
        """Record a failed prediction (error/exception during inference).

        Args:
            latency_ms: Optional latency of the failed request.  When
                provided the latency is included in the rolling window so
                that percentile calculations reflect both successes and
                failures.
        """
        with self._lock:
            self._error_count += 1
            if latency_ms is not None:
                self._latencies_ms.append(float(latency_ms))
                self._timestamps.append(time.monotonic())

    def time_request(self) -> _RequestTimer:
        """Context manager that records elapsed time on exit."""
        return _RequestTimer(self)

    def snapshot(self) -> CostMetrics:
        """Compute current latency / throughput / cost metrics."""
        with self._lock:
            count = len(self._latencies_ms)
            error_count = self._error_count
            success_count = self._success_count
            if count == 0:
                return CostMetrics(
                    model_name=self.model_name,
                    latency_ms_p50=float("nan"),
                    latency_ms_p95=float("nan"),
                    latency_ms_p99=float("nan"),
                    throughput_rps=0.0,
                    cost_per_prediction=self.cost_per_prediction,
                    compute_utilisation_pct=0.0,
                    sample_count=0,
                    error_rate=0.0,
                    error_count=error_count,
                    success_count=success_count,
                )
            latencies = np.array(self._latencies_ms)
            p50 = float(np.percentile(latencies, 50))
            p95 = float(np.percentile(latencies, 95))
            p99 = float(np.percentile(latencies, 99))
            throughput = self._compute_throughput_locked()

        total = error_count + success_count
        error_rate = error_count / total if total > 0 else 0.0

        return CostMetrics(
            model_name=self.model_name,
            latency_ms_p50=p50,
            latency_ms_p95=p95,
            latency_ms_p99=p99,
            throughput_rps=throughput,
            cost_per_prediction=self.cost_per_prediction,
            compute_utilisation_pct=0.0,  # populated by integrations
            sample_count=count,
            error_rate=error_rate,
            error_count=error_count,
            success_count=success_count,
        )

    def check_thresholds(self) -> list[str]:
        """Return human-readable threshold breach descriptions."""
        snap = self.snapshot()
        breaches: list[str] = []
        thresholds = self.config.alert_thresholds
        if "latency_p99_ms" in thresholds and snap.latency_ms_p99 > thresholds["latency_p99_ms"]:
            breaches.append(
                f"p99 latency {snap.latency_ms_p99:.1f}ms > {thresholds['latency_p99_ms']}ms"
            )
        if (
            "cost_per_1k_predictions" in thresholds
            and snap.cost_per_prediction * 1000 > thresholds["cost_per_1k_predictions"]
        ):
            breaches.append(
                f"cost/1k {snap.cost_per_prediction * 1000:.2f} > "
                f"{thresholds['cost_per_1k_predictions']}"
            )
        if "throughput_rps" in thresholds and snap.throughput_rps < thresholds["throughput_rps"]:
            breaches.append(
                f"throughput {snap.throughput_rps:.1f} rps < {thresholds['throughput_rps']} rps"
            )
        if "error_rate" in thresholds and snap.error_rate > thresholds["error_rate"]:
            breaches.append(
                f"error rate {snap.error_rate:.2%} > {thresholds['error_rate']:.2%}"
            )
        return breaches

    # ── Persistence ────────────────────────────────────────────────

    def flush_metrics(self, path: str | Path) -> int:
        """Append the current snapshot to a JSONL file.

        Args:
            path: Destination file.  Created if it does not exist.

        Returns:
            Total number of records in the file after the append.
        """
        snap = self.snapshot()
        record = {
            "timestamp": snap.timestamp.isoformat(),
            "latency_ms_p50": snap.latency_ms_p50,
            "latency_ms_p95": snap.latency_ms_p95,
            "latency_ms_p99": snap.latency_ms_p99,
            "throughput_rps": snap.throughput_rps,
            "cost_per_prediction": snap.cost_per_prediction,
            "error_rate": snap.error_rate,
            "sample_count": snap.sample_count,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        log.debug("cost_metrics_flushed", path=str(p))
        # Count total lines
        count = sum(1 for _ in p.open())
        return count

    @staticmethod
    def load_metrics(path: str | Path, n: int = 100) -> list[dict[str, Any]]:
        """Read the last *n* metric snapshots from a JSONL file.

        Args:
            path: Source JSONL file.
            n: Maximum number of records to return (most recent last).

        Returns:
            A list of dicts, one per snapshot line.
        """
        p = Path(path)
        if not p.exists():
            return []
        lines = p.read_text().splitlines()
        tail = lines[-n:] if len(lines) > n else lines
        results: list[dict[str, Any]] = []
        for line in tail:
            line = line.strip()
            if line:
                results.append(json.loads(line))
        return results

    # ── Internal ───────────────────────────────────────────────────

    def _compute_throughput_locked(self) -> float:
        """Compute throughput.  Caller must hold ``self._lock``."""
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        if span <= 0:
            return 0.0
        return float(len(self._timestamps) / span)

    def _compute_throughput(self) -> float:
        with self._lock:
            return self._compute_throughput_locked()


class _RequestTimer:
    """Context manager helper that calls `monitor.record()` on exit."""

    def __init__(self, monitor: CostMonitor):
        self._monitor = monitor
        self._start = 0.0

    def __enter__(self) -> _RequestTimer:
        self._start = time.monotonic()
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed = (time.monotonic() - self._start) * 1000
        self._monitor.record(elapsed)
