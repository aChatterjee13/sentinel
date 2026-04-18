"""Maps technical model metrics to business KPIs."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from typing import Any

import structlog

from sentinel.config.schema import BusinessKPIConfig

log = structlog.get_logger(__name__)

KPIFetcher = Callable[[str], float | None]
"""Signature: ``(data_source) -> kpi_value | None``."""


class KPILinker:
    """Links model metrics to business KPIs and computes impact reports.

    Example:
        >>> linker = KPILinker(config.business_kpi)
        >>> linker.set_fetcher(my_warehouse_fetcher)
        >>> impact = linker.report({"precision": 0.92, "recall": 0.81})
    """

    def __init__(self, config: BusinessKPIConfig):
        self.config = config
        self._fetcher: KPIFetcher | None = None
        self._cached_kpis: dict[str, float] = {}
        self._lock = threading.Lock()
        self._refresh_timer: threading.Timer | None = None
        self._refresh_interval: float | None = None

    def set_fetcher(self, fetcher: KPIFetcher) -> None:
        """Inject a function that fetches KPI values from a data source."""
        with self._lock:
            self._fetcher = fetcher

    def refresh(self) -> dict[str, float]:
        """Pull fresh values for every configured KPI."""
        with self._lock:
            if self._fetcher is None:
                return self._cached_kpis
            for mapping in self.config.mappings:
                if mapping.data_source:
                    try:
                        value = self._fetcher(mapping.data_source)
                    except Exception:
                        log.warning(
                            "kpi_linker.fetch_failed",
                            kpi=mapping.business_kpi,
                            data_source=mapping.data_source,
                        )
                        continue
                    if value is not None and isinstance(value, (int, float)) and math.isfinite(value):
                        self._cached_kpis[mapping.business_kpi] = float(value)
                    elif value is not None:
                        log.warning(
                            "kpi_linker.invalid_value",
                            kpi=mapping.business_kpi,
                            value=value,
                        )
            return self._cached_kpis

    def start_auto_refresh(self, interval_seconds: float = 300.0) -> None:
        """Start periodic background KPI refresh.

        Args:
            interval_seconds: Interval between refreshes in seconds.
        """
        self._refresh_interval = interval_seconds
        self._schedule_next()
        log.info(
            "kpi_linker.auto_refresh_started",
            interval_seconds=interval_seconds,
        )

    def stop_auto_refresh(self) -> None:
        """Stop the background refresh."""
        if self._refresh_timer is not None:
            self._refresh_timer.cancel()
            self._refresh_timer = None
        self._refresh_interval = None
        log.info("kpi_linker.auto_refresh_stopped")

    def _schedule_next(self) -> None:
        """Schedule the next background refresh cycle."""
        if self._refresh_interval is None:
            return
        self._refresh_timer = threading.Timer(
            self._refresh_interval, self._auto_refresh
        )
        self._refresh_timer.daemon = True
        self._refresh_timer.start()

    def _auto_refresh(self) -> None:
        """Execute a single background refresh cycle and reschedule."""
        try:
            self.refresh()
        except Exception:
            log.warning("kpi_linker.auto_refresh_failed")
        finally:
            self._schedule_next()

    def report(self, model_metrics: dict[str, float]) -> dict[str, Any]:
        """Build a report linking model metrics to KPIs."""
        with self._lock:
            impacts: list[dict[str, Any]] = []
            for mapping in self.config.mappings:
                metric_value = model_metrics.get(mapping.model_metric)
                kpi_value = self._cached_kpis.get(mapping.business_kpi)
                impacts.append(
                    {
                        "model_metric": mapping.model_metric,
                        "business_kpi": mapping.business_kpi,
                        "metric_value": metric_value,
                        "kpi_value": kpi_value,
                        "data_source": mapping.data_source,
                    }
                )
            return {"linked_kpis": impacts, "n_links": len(impacts)}
