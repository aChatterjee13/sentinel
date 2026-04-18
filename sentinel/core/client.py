"""SentinelClient — the single entry point for the SDK."""

from __future__ import annotations

import json
import re
import threading
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import structlog

from sentinel.action.deployment.manager import DeploymentManager
from sentinel.action.notifications.engine import NotificationEngine
from sentinel.action.retrain.orchestrator import RetrainOrchestrator
from sentinel.config.loader import ConfigLoader
from sentinel.config.schema import SentinelConfig
from sentinel.core.exceptions import DriftDetectionError, SentinelError
from sentinel.core.hooks import HookManager, HookType
from sentinel.core.types import (
    Alert,
    AlertSeverity,
    DriftReport,
    FeatureHealthReport,
    PredictionRecord,
    QualityReport,
)
from sentinel.domains import resolve_adapter
from sentinel.domains.base import BaseDomainAdapter
from sentinel.foundation.audit.keystore import BaseKeystore, EnvKeystore, FileKeystore
from sentinel.foundation.audit.shipper import BaseAuditShipper, NullShipper
from sentinel.foundation.audit.trail import AuditTrail
from sentinel.foundation.datasets.registry import DatasetRegistry
from sentinel.foundation.registry.backends import resolve_backend
from sentinel.foundation.registry.backends.base import BaseRegistryBackend
from sentinel.foundation.registry.backends.local import LocalRegistryBackend
from sentinel.foundation.registry.model_registry import ModelRegistry
from sentinel.intelligence.kpi_linker import KPILinker
from sentinel.intelligence.model_graph import ModelGraph
from sentinel.observability.cost_monitor import CostMonitor
from sentinel.observability.data_quality import DataQualityChecker
from sentinel.observability.drift import BaseDriftDetector, create_drift_detector
from sentinel.observability.feature_health import FeatureHealthMonitor

if TYPE_CHECKING:
    from sentinel.action.deployment.strategies.base import DeploymentState
    from sentinel.foundation.registry.model_registry import ModelVersion

log = structlog.get_logger(__name__)


class SentinelClient:
    """The unified Sentinel client.

    Provides a single entry point to drift detection, data quality, alerts,
    deployment, retraining, registry, audit, and (when enabled) LLMOps and
    AgentOps subsystems.

    Example:
        >>> client = SentinelClient.from_config("sentinel.yaml")
        >>> client.log_prediction(features={"age": 35}, prediction=0)
        >>> report = client.check_drift()
        >>> if report.is_drifted:
        ...     print(report.summary)
    """

    # ── Construction ──────────────────────────────────────────────

    def __init__(
        self,
        config: SentinelConfig,
        registry: ModelRegistry | None = None,
        audit: AuditTrail | None = None,
    ):
        self.config = config
        self.model_name = config.model.name
        self.model_version: str | None = config.model.version

        # Foundation
        self.audit = audit or AuditTrail(
            config.audit,
            keystore=self._build_audit_keystore(config),
            shipper=self._build_audit_shipper(config),
        )
        self.registry = registry or ModelRegistry(backend=self._build_registry_backend(config))

        # Domain adapter — resolves drift detectors, quality, schema validator
        self.domain_adapter: BaseDomainAdapter = self._build_domain_adapter()

        # Observability
        self._drift_detector = self._build_drift_detector()
        self.data_quality = DataQualityChecker(config.data_quality, model_name=self.model_name)
        self.feature_health = FeatureHealthMonitor(
            config.feature_health, model_name=self.model_name
        )
        self.cost_monitor = CostMonitor(config.cost_monitor, model_name=self.model_name)

        # Action
        self.notifications = NotificationEngine(config.alerts, audit_trail=self.audit)
        self.deployment_manager = DeploymentManager(
            config.deployment, registry=self.registry, audit=self.audit
        )
        self.retrain = RetrainOrchestrator(
            config.retraining,
            registry=self.registry,
            audit=self.audit,
            deployment_manager=self.deployment_manager,
        )

        # Intelligence
        self.model_graph = ModelGraph(config.model_graph)
        self.kpi_linker = KPILinker(config.business_kpi)

        # Hooks + buffers
        self.hooks = HookManager()
        self._lock = threading.RLock()
        self._prediction_buffer: deque[PredictionRecord] = deque(maxlen=10_000)
        self._actuals: dict[str, Any] = {}
        self._previous_window_data: np.ndarray | None = None

        # Model reference for importance computation & explanations
        self._model: Any = None
        self._reference_data: Any = None
        self._reference_y: Any = None
        self._last_importance_calc: datetime | None = None

        # Count-based auto-check (Gap-B)
        self._auto_check = config.drift.auto_check
        self._predictions_since_check: int = 0

        # Streaming concept drift (Gap-E)
        self._concept_drift_detector: BaseDriftDetector | None = None
        self._concept_observations: int = 0
        if config.drift.concept is not None:
            self._concept_drift_detector = self._build_concept_drift_detector(config)

        # Optional layers — lazy
        self._llmops: Any = None
        self._agentops: Any = None
        self._llmops_init_failed: bool = False
        self._agentops_init_failed: bool = False
        self._explainability: Any = None
        self._dataset_registry: DatasetRegistry | None = None
        self._experiment_tracker: Any = None

        # Cohort analysis (competitive-gap)
        self._cohort_analyzer: Any = None
        if config.cohort_analysis.enabled:
            from sentinel.observability.cohort_analyzer import CohortAnalyzer

            self._cohort_analyzer = CohortAnalyzer(config.cohort_analysis, self.model_name)

        # Background drift scheduler (WS-C)
        self._scheduler: Any = None
        if config.drift.schedule.enabled:
            from sentinel.core.scheduler import DriftScheduler

            self._scheduler = DriftScheduler(
                client=self,
                interval=config.drift.schedule.interval,
                run_on_start=config.drift.schedule.run_on_start,
            )
            self._scheduler.start()

        # Auto-wire Azure ML pipeline runner (WS-B)
        self._auto_wire_pipeline_runner()

        log.info(
            "sentinel.client_initialised",
            model=self.model_name,
            domain=config.model.domain,
            llmops=config.llmops.enabled,
            agentops=config.agentops.enabled,
        )

    @classmethod
    def from_config(cls, path: str | Path) -> SentinelClient:
        """Build a client from a YAML/JSON config file."""
        config = ConfigLoader(path).load()
        return cls(config)

    @staticmethod
    def _build_audit_keystore(config: SentinelConfig) -> BaseKeystore | None:
        """Build the signing keystore when tamper evidence is enabled.

        Returns ``None`` when ``audit.tamper_evidence`` is off. When
        on, exactly one of ``signing_key_path`` or ``signing_key_env``
        must resolve to a usable key; the keystore validates that
        lazily on first use and will raise
        :class:`AuditKeystoreError` at write time if the key is
        missing or too short.
        """
        if not config.audit.tamper_evidence:
            return None
        if config.audit.signing_key_path:
            return FileKeystore(config.audit.signing_key_path)
        return EnvKeystore(config.audit.signing_key_env)

    @staticmethod
    def _build_audit_shipper(config: SentinelConfig) -> BaseAuditShipper:
        """Build the audit shipper matching ``audit.storage``.

        The default ``storage="local"`` returns a :class:`NullShipper`
        — the trail keeps writing locally and nothing is shipped
        anywhere, preserving every existing config. ``azure_blob``
        and ``s3`` return their respective threaded shippers, which
        lazy-import their cloud SDKs so ``import sentinel`` stays
        dependency-free. ``gcs`` is not yet implemented.
        """
        audit = config.audit
        if audit.storage == "local":
            return NullShipper()
        if audit.storage == "azure_blob":
            if audit.azure_blob is None:
                raise SentinelError("azure_blob config required for azure_blob audit storage")
            from sentinel.integrations.azure.blob_audit import AzureBlobShipper

            return AzureBlobShipper(
                account_url=audit.azure_blob.account_url,
                container_name=audit.azure_blob.container_name,
                prefix=audit.azure_blob.prefix,
                delete_local_after_ship=audit.azure_blob.delete_local_after_ship,
            )
        if audit.storage == "s3":
            if audit.s3 is None:
                raise SentinelError("s3 config required for s3 audit storage")
            from sentinel.integrations.aws.s3_audit import S3Shipper

            return S3Shipper(
                bucket=audit.s3.bucket,
                prefix=audit.s3.prefix,
                region=audit.s3.region,
                delete_local_after_ship=audit.s3.delete_local_after_ship,
            )
        if audit.storage == "gcs":
            if audit.gcs is None:
                raise SentinelError("gcs config required for gcs audit storage")
            from sentinel.integrations.gcp.gcs_audit import GcsShipper

            return GcsShipper(
                bucket=audit.gcs.bucket,
                prefix=audit.gcs.prefix,
                project=audit.gcs.project,
                delete_local_after_ship=audit.gcs.delete_local_after_ship,
            )
        # Defensive — Pydantic's Literal already rejects unknown values.
        raise SentinelError(f"unknown audit storage backend: {audit.storage}")

    @staticmethod
    def _build_registry_backend(config: SentinelConfig) -> BaseRegistryBackend:
        """Instantiate the configured model-registry backend.

        The default ``backend="local"`` stays wire-compatible with
        every existing config — it returns a :class:`LocalRegistryBackend`
        pointing at ``registry.path``. ``azure_ml`` and ``mlflow``
        resolve through the lazy registry in
        :mod:`sentinel.foundation.registry.backends` so heavy cloud
        SDKs are only imported when the operator asked for them.
        """
        reg = config.registry
        backend_cls = resolve_backend(reg.backend)
        if reg.backend == "local":
            return LocalRegistryBackend(root=reg.path)
        if reg.backend == "azure_ml":
            # Validator guarantees these three fields are set.
            if reg.subscription_id is None:
                raise SentinelError("subscription_id required for azure_ml registry backend")
            if reg.resource_group is None:
                raise SentinelError("resource_group required for azure_ml registry backend")
            if reg.workspace_name is None:
                raise SentinelError("workspace_name required for azure_ml registry backend")
            return backend_cls(  # type: ignore[call-arg]
                subscription_id=reg.subscription_id,
                resource_group=reg.resource_group,
                workspace_name=reg.workspace_name,
            )
        if reg.backend == "mlflow":
            return backend_cls(tracking_uri=reg.tracking_uri)  # type: ignore[call-arg]
        if reg.backend == "sagemaker":
            return backend_cls(  # type: ignore[call-arg]
                region_name=reg.region_name or "us-east-1",
                role_arn=reg.role_arn,
                s3_bucket=reg.s3_bucket,
            )
        if reg.backend == "vertex_ai":
            if reg.project is None:
                raise SentinelError("project required for vertex_ai registry backend")
            return backend_cls(  # type: ignore[call-arg]
                project=reg.project,
                location=reg.location or "us-central1",
                gcs_bucket=reg.gcs_bucket,
            )
        if reg.backend == "databricks":
            return backend_cls(  # type: ignore[call-arg]
                host=reg.host,
                token=reg.token,
                catalog=reg.catalog or "ml",
                schema_name=reg.schema_name or "default",
            )
        # Defensive — Pydantic's Literal already rejects unknown values.
        raise SentinelError(f"unknown registry backend: {reg.backend}")

    def _auto_wire_pipeline_runner(self) -> None:
        """Auto-wire an Azure ML pipeline runner when the config points at one.

        When ``retraining.pipeline`` starts with ``azureml://`` AND the
        registry backend is ``azure_ml`` (meaning credentials are already
        available), construct an :class:`AzureMLPipelineRunner` from the
        same subscription/resource-group/workspace and inject it into the
        retrain orchestrator. Otherwise skip silently — the user can
        always call ``client.retrain.set_pipeline_runner()`` manually.
        """
        pipeline_uri = self.config.retraining.pipeline or ""
        if not pipeline_uri.startswith("azureml://"):
            return
        if self.config.registry.backend != "azure_ml":
            return
        reg = self.config.registry
        if not (reg.subscription_id and reg.resource_group and reg.workspace_name):
            return
        try:
            from sentinel.integrations.azure.pipeline_runner import AzureMLPipelineRunner

            runner = AzureMLPipelineRunner(
                subscription_id=reg.subscription_id,
                resource_group=reg.resource_group,
                workspace_name=reg.workspace_name,
            )
            self.retrain.set_pipeline_runner(runner)
            log.info("client.auto_wired_pipeline_runner", pipeline=pipeline_uri)
        except Exception:
            log.debug("client.auto_wire_pipeline_runner_skipped", pipeline=pipeline_uri)

    # ── Drift detection ───────────────────────────────────────────

    def _build_domain_adapter(self) -> BaseDomainAdapter:
        adapter_cls = resolve_adapter(self.config.model.domain)
        try:
            return adapter_cls(self.config)
        except Exception as e:
            raise SentinelError(
                f"failed to build domain adapter for {self.config.model.domain!r}: {e}"
            ) from e

    @staticmethod
    def _build_concept_drift_detector(config: SentinelConfig) -> BaseDriftDetector:
        """Instantiate the configured streaming concept drift detector."""
        from sentinel.observability.drift.concept_drift import (
            ADWINConceptDriftDetector,
            DDMConceptDriftDetector,
            EDDMConceptDriftDetector,
            PageHinkleyDriftDetector,
        )

        concept_detectors: dict[str, type[BaseDriftDetector]] = {
            "ddm": DDMConceptDriftDetector,
            "eddm": EDDMConceptDriftDetector,
            "adwin": ADWINConceptDriftDetector,
            "page_hinkley": PageHinkleyDriftDetector,
        }
        if config.drift.concept is None:
            raise SentinelError("concept drift config required but not provided")
        method = config.drift.concept.method
        cls = concept_detectors[method]
        return cls(
            model_name=config.model.name,
            warning_level=config.drift.concept.warning_level,
            drift_level=config.drift.concept.drift_level,
            min_samples=config.drift.concept.min_samples,
        )

    def _build_drift_detector(self) -> BaseDriftDetector:
        # Domain adapter owns the drift detector — fall back to the core
        # tabular factory for backwards compatibility.
        detectors = self.domain_adapter.get_drift_detectors()
        if detectors:
            return detectors[0]
        cfg = self.config.drift.data
        return create_drift_detector(
            method=cfg.method,
            model_name=self.model_name,
            threshold=cfg.threshold,
        )

    def fit_baseline(
        self,
        reference: Any,
        *,
        model: Any = None,
        y: Any | None = None,
    ) -> None:
        """Fit the drift detector and data quality checker on a reference dataset.

        Args:
            reference: The reference (baseline) feature matrix.
            model: Optional fitted model.  When provided, feature
                importances are computed automatically using the
                configured ``importance_method``.
            y: Optional target vector.  Required when
                ``importance_method`` is ``"permutation"``.
        """
        self._drift_detector.fit(reference)

        # data_quality.fit needs feature_names for numpy arrays
        import numpy as _np

        feature_names: list[str] | None = None
        if isinstance(reference, _np.ndarray):
            n_cols = reference.shape[1] if reference.ndim > 1 else 1
            feature_names = [f"f{i}" for i in range(n_cols)]
        self.data_quality.fit(reference, feature_names=feature_names)

        self._reference_data = reference
        self._reference_y = y
        self.audit.log(
            event_type="baseline_fitted",
            model_name=self.model_name,
            method=self._drift_detector.method_name,
        )

        effective_model = model or self._model
        if effective_model is not None:
            arr = _np.asarray(reference)
            names = feature_names or [f"f{i}" for i in range(arr.shape[1] if arr.ndim > 1 else 1)]
            self.feature_health.compute_importance(
                effective_model, arr, y=y, feature_names=names,
            )
            self._last_importance_calc = datetime.now(timezone.utc)
            if model is not None:
                self._model = model

    def check_drift(self, current: Any | None = None) -> DriftReport:
        """Compute a drift report against the configured reference.

        If *current* is ``None``, uses the in-memory prediction buffer
        filtered by the configured ``drift.data.window``.  When the
        reference mode is ``previous_window``, the current window's data
        becomes the reference for the next check.
        """
        self.hooks.dispatch(HookType.BEFORE_DRIFT_CHECK)

        window_str = self.config.drift.data.window
        ref_mode = self.config.drift.data.reference

        with self._lock:
            if current is not None:
                data = current
                buf_snapshot: list[PredictionRecord] = []
            else:
                buf_snapshot = self._extract_window()
                data = None

        if data is None and buf_snapshot:
            keys = sorted({k for r in buf_snapshot for k in r.features})
            if keys:
                rows = [
                    [float(r.features.get(k, 0.0) or 0.0) for k in keys]
                    for r in buf_snapshot
                ]
                data = np.array(rows)

        if data is None or (hasattr(data, "__len__") and len(data) == 0):
            raise SentinelError("no data available for drift check")

        # previous_window reference mode: use last window as baseline
        if ref_mode == "previous_window" and self._previous_window_data is not None:
            self._drift_detector.fit(self._previous_window_data)
        elif not self._drift_detector.is_fitted():
            log.warning("drift.no_baseline_fitting_from_buffer")
            self._drift_detector.fit(data)

        # Save current data for next previous_window comparison
        current_data = np.array(data) if not isinstance(data, np.ndarray) else data.copy()
        with self._lock:
            self._previous_window_data = current_data

        # Reset count-based counter on manual/auto check
        with self._lock:
            self._predictions_since_check = 0

        report = self._drift_detector.detect(data)

        # Populate the window field on the report
        report = report.model_copy(update={"window": window_str})

        # Merge concept drift results (Gap-E)
        with self._lock:
            _has_concept_obs = (
                self._concept_drift_detector is not None and self._concept_observations > 0
            )
        if _has_concept_obs:
            concept_report = self._concept_drift_detector.detect(np.array([]))
            report = report.model_copy(
                update={
                    "metadata": {
                        **report.metadata,
                        "concept_drift": concept_report.model_dump(mode="json"),
                    },
                }
            )
            if concept_report.is_drifted and not report.is_drifted:
                report = report.model_copy(
                    update={
                        "is_drifted": True,
                        "severity": max(
                            report.severity,
                            concept_report.severity,
                            key=lambda s: list(AlertSeverity).index(s),
                        ),
                    }
                )

        self.hooks.dispatch(HookType.AFTER_DRIFT_CHECK, report)
        self.audit.log(
            event_type="drift_checked",
            model_name=self.model_name,
            method=report.method,
            is_drifted=report.is_drifted,
            severity=report.severity.value,
            n_drifted=len(report.drifted_features),
        )

        # Persist drift report to history
        self._persist_drift_report(report)

        if report.is_drifted:
            self._fire_drift_alert(report)
            self._maybe_trigger_retrain(report)

        return report

    # ── Window helpers ─────────────────────────────────────────────

    @staticmethod
    def _parse_window(window: str) -> timedelta | int:
        """Parse a window string into a timedelta or a count.

        Supported formats:
            ``"7d"`` → 7 days, ``"24h"`` → 24 hours, ``"30m"`` → 30 minutes,
            ``"1000"`` → last 1000 records (integer count).

        Args:
            window: The window specification string.

        Returns:
            A ``timedelta`` for time-based windows or an ``int`` for count-based.

        Raises:
            ValueError: If the window string cannot be parsed.
        """
        stripped = window.strip()
        if stripped.isdigit():
            return int(stripped)
        m = re.fullmatch(r"(\d+)\s*([dhm])", stripped, re.IGNORECASE)
        if not m:
            raise ValueError(f"cannot parse window string: {window!r}")
        value = int(m.group(1))
        unit = m.group(2).lower()
        if unit == "d":
            return timedelta(days=value)
        if unit == "h":
            return timedelta(hours=value)
        return timedelta(minutes=value)

    def _extract_window(self) -> list[PredictionRecord]:
        """Extract predictions within the configured drift window.

        Must be called while holding ``self._lock``.

        Returns:
            Filtered list of :class:`PredictionRecord` objects.
        """
        window_str = self.config.drift.data.window
        if not window_str:
            return list(self._prediction_buffer)

        try:
            parsed = self._parse_window(window_str)
        except ValueError:
            log.warning("drift.unparseable_window", window=window_str)
            return list(self._prediction_buffer)

        if isinstance(parsed, int):
            buf = list(self._prediction_buffer)
            return buf[-parsed:] if len(buf) > parsed else buf

        # Time-based window
        cutoff = datetime.now(timezone.utc) - parsed
        return [r for r in self._prediction_buffer if r.timestamp >= cutoff]

    # ── Drift history persistence ──────────────────────────────────

    def _drift_history_path(self) -> Path:
        """Return the JSONL path for this model's drift history."""
        base = Path(self.config.audit.path) / "drift_history"
        base.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^\w\-.]", "_", self.model_name)
        return base / f"{safe_name}.jsonl"

    def _persist_drift_report(self, report: DriftReport) -> None:
        """Append a drift report to the model's history file."""
        try:
            path = self._drift_history_path()
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(report.model_dump(mode="json"), default=str) + "\n")
        except Exception:
            log.warning("drift.persist_report_failed", model=self.model_name)

    def get_drift_history(self, n: int = 10) -> list[DriftReport]:
        """Read the last *n* drift reports from the history file.

        Args:
            n: Maximum number of reports to return (most recent first).

        Returns:
            A list of :class:`DriftReport` objects, newest first.
        """
        path = self._drift_history_path()
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        reports: list[DriftReport] = []
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                reports.append(DriftReport.model_validate_json(line))
            except Exception:
                log.debug("drift.history_parse_error", line=line[:80])
            if len(reports) >= n:
                break
        return reports

    def _buffered_features(self) -> Any:
        if not self._prediction_buffer:
            return None
        keys = sorted({k for r in self._prediction_buffer for k in r.features})
        if not keys:
            return None
        import numpy as np

        rows = [
            [float(r.features.get(k, 0.0) or 0.0) for k in keys] for r in self._prediction_buffer
        ]
        return np.array(rows)

    def _fire_drift_alert(self, report: DriftReport) -> None:
        alert = Alert(
            model_name=self.model_name,
            title=f"Drift detected: {report.method}",
            body=f"{len(report.drifted_features)} features drifted (severity {report.severity.value})",
            severity=report.severity,
            source="drift_detector",
            payload={
                "method": report.method,
                "test_statistic": report.test_statistic,
                "drifted_features": report.drifted_features[:10],
            },
            fingerprint=f"drift:{self.model_name}:{report.method}",
        )
        self.notifications.dispatch(alert)

        # Cascade alerts (WS-E) — notify downstream models
        impact = self.model_graph.cascade_impact(self.model_name)
        downstream = impact.get("downstream_affected", [])
        if downstream:
            cascade_alert = Alert(
                model_name=self.model_name,
                title=f"Cascade: drift in {self.model_name} affects {len(downstream)} downstream model(s)",
                body=f"Downstream affected: {', '.join(downstream[:5])}",
                severity=AlertSeverity.HIGH,
                source="model_graph",
                payload=impact,
                fingerprint=f"cascade:{self.model_name}:{report.method}",
            )
            self.notifications.dispatch(cascade_alert)
            self.audit.log(
                event_type="cascade_alert_fired",
                model_name=self.model_name,
                downstream_affected=downstream,
            )

    def _maybe_trigger_retrain(self, report: DriftReport) -> None:
        trigger = self.retrain.on_drift(report)
        if trigger is None:
            return
        # Caller can hook in their own pipeline; we just record intent here.
        self.audit.log(
            event_type="retrain_intent",
            model_name=self.model_name,
            trigger=trigger.__dict__,
        )

    # ── Data quality ──────────────────────────────────────────────

    def check_data_quality(self, data: dict[str, Any] | list[dict[str, Any]]) -> QualityReport:
        report = self.data_quality.check(data)
        if report.has_critical_issues:
            self._fire_quality_alert(report)
        return report

    def _fire_quality_alert(self, report: QualityReport) -> None:
        alert = Alert(
            model_name=self.model_name,
            title="Data quality issues detected",
            body=report.summary,
            severity=AlertSeverity.HIGH,
            source="data_quality",
            payload={"issues": [i.model_dump(mode="json") for i in report.issues[:5]]},
            fingerprint=f"quality:{self.model_name}",
        )
        self.notifications.dispatch(alert)

    # ── Predictions ───────────────────────────────────────────────

    def log_prediction(
        self,
        features: dict[str, Any],
        prediction: Any,
        actual: Any = None,
        confidence: float | None = None,
        explanation: dict[str, float] | None = None,
        cohort_id: str | None = None,
        latency_ms: float | None = None,
        **metadata: Any,
    ) -> str:
        """Record a prediction. Drift checks fire on the configured cadence.

        Args:
            features: Input feature dictionary.
            prediction: Model output.
            actual: Optional ground-truth label.
            confidence: Optional prediction confidence score.
            explanation: Optional per-feature attribution dict.
            cohort_id: Optional cohort identifier for cohort analysis.
            latency_ms: Optional inference latency in milliseconds.  When
                provided, automatically forwarded to the cost monitor.
            **metadata: Arbitrary extra metadata stored with the record.

        Returns:
            The ``prediction_id`` assigned to this prediction. Use it to
            submit ground-truth actuals later via :meth:`log_actual`.
        """
        record = PredictionRecord(
            model_name=self.model_name,
            model_version=self.model_version,
            features=features,
            prediction=prediction,
            actual=actual,
            confidence=confidence,
            explanation=explanation,
            metadata=metadata,
        )
        self.hooks.dispatch(HookType.BEFORE_PREDICTION, record)
        with self._lock:
            self._prediction_buffer.append(record)
        if self.config.audit.log_predictions:
            self.audit.log(
                event_type="prediction_logged",
                model_name=self.model_name,
                model_version=self.model_version,
                payload=record.model_dump(mode="json"),
            )

        # Auto-record latency in cost monitor
        if latency_ms is not None:
            self.cost_monitor.record(latency_ms)

        # Cohort analysis
        if self._cohort_analyzer is not None:
            try:
                feat_floats = {
                    k: float(v) for k, v in features.items() if isinstance(v, (int, float))
                }
                self._cohort_analyzer.add_prediction(
                    features=feat_floats,
                    prediction=float(prediction) if isinstance(prediction, (int, float)) else 0.0,
                    actual=float(actual) if isinstance(actual, (int, float)) else None,
                    cohort_id=cohort_id,
                )
            except Exception:
                log.debug("cohort_analyzer.add_prediction_failed")

        # Streaming concept drift (Gap-E)
        if actual is not None and self._concept_drift_detector is not None:
            error_signal = self._compute_error_signal(prediction, actual)
            self._concept_drift_detector._update(error_signal)
            with self._lock:
                self._concept_observations += 1

        # Count-based auto-check (Gap-B)
        if self._auto_check.enabled:
            should_check = False
            with self._lock:
                self._predictions_since_check += 1
                if (
                    self._predictions_since_check >= self._auto_check.every_n_predictions
                    and len(self._prediction_buffer) >= self._auto_check.every_n_predictions
                ):
                    self._predictions_since_check = 0
                    should_check = True
            if should_check:
                threading.Thread(
                    target=self._safe_check_drift,
                    name="sentinel-auto-drift",
                    daemon=True,
                ).start()

        self.hooks.dispatch(HookType.AFTER_PREDICTION, record)
        return record.prediction_id

    def _safe_check_drift(self) -> None:
        """Auto-check drift in a background thread, swallowing errors."""
        try:
            self.check_drift()
        except (DriftDetectionError, ValueError, TypeError) as e:
            log.warning("auto_check_drift_failed", error=str(e))
        except Exception:
            log.exception("auto_check_drift_unexpected_error")

    @staticmethod
    def _compute_error_signal(prediction: Any, actual: Any) -> float:
        """Convert prediction + actual into a binary error signal."""
        try:
            if isinstance(prediction, (int, float)) and isinstance(actual, (int, float)):
                # If values look binary (0/1), treat as classification
                if prediction in (0, 1) and actual in (0, 1):
                    return 0.0 if prediction == actual else 1.0
                # Regression — normalized absolute error
                return abs(float(prediction) - float(actual))
            # Categorical / string comparison
            return 0.0 if prediction == actual else 1.0
        except Exception:
            return 1.0  # treat unparseable as error

    def buffer_size(self) -> int:
        return len(self._prediction_buffer)

    def log_actual(self, prediction_id: str, actual: Any) -> None:
        """Attach a ground-truth actual to a previously logged prediction.

        In many BFSI workflows the true label arrives days or weeks after
        the prediction was made.  Call this method with the
        ``prediction_id`` returned by :meth:`log_prediction` to record the
        actual value.

        Args:
            prediction_id: The identifier returned by ``log_prediction()``.
            actual: The ground-truth value to associate with the prediction.

        Raises:
            Nothing — if the *prediction_id* is no longer in the buffer
            (evicted or already cleared), a warning is logged instead.
        """
        with self._lock:
            found = any(r.prediction_id == prediction_id for r in self._prediction_buffer)
            if found:
                self._actuals[prediction_id] = actual
            else:
                log.warning(
                    "log_actual.prediction_not_found",
                    prediction_id=prediction_id,
                )

    def flush_buffer(self, path: str | Path, format: str = "jsonl") -> int:
        """Export the current prediction buffer to a JSONL file.

        Each line is a JSON object with keys: ``prediction_id``,
        ``features``, ``prediction``, ``actual``, ``timestamp``, and
        ``confidence``.

        The buffer is **not** cleared after writing — call
        :meth:`clear_buffer` explicitly if needed.

        Args:
            path: Destination file path (created or overwritten).
            format: Output format.  Only ``"jsonl"`` is supported today.

        Returns:
            The number of records written.

        Raises:
            ValueError: If *format* is not ``"jsonl"``.
        """
        if format != "jsonl":
            raise ValueError(f"unsupported format: {format!r} (only 'jsonl' is supported)")

        import json

        out = Path(path)
        with self._lock:
            snapshot = list(self._prediction_buffer)
            actuals_snapshot = dict(self._actuals)

        with out.open("w", encoding="utf-8") as fh:
            for rec in snapshot:
                actual = actuals_snapshot.get(rec.prediction_id, rec.actual)
                ts = rec.timestamp.isoformat() if rec.timestamp else None
                line = {
                    "prediction_id": rec.prediction_id,
                    "features": rec.features,
                    "prediction": rec.prediction,
                    "actual": actual,
                    "timestamp": ts,
                    "confidence": rec.confidence,
                }
                fh.write(json.dumps(line, default=str) + "\n")

        return len(snapshot)

    def clear_buffer(self) -> None:
        with self._lock:
            self._prediction_buffer.clear()
            self._actuals.clear()

    # ── Feature health ────────────────────────────────────────────

    _RECALC_INTERVAL_HOURS: ClassVar[dict[str, float]] = {
        "daily": 24.0,
        "weekly": 168.0,
        "monthly": 720.0,
    }

    def set_model(self, model: Any) -> None:
        """Store a model reference for importance computation and explanations.

        Args:
            model: A fitted estimator (any object with ``predict`` or
                ``predict_proba``).
        """
        self._model = model

    def get_feature_health(self) -> FeatureHealthReport:
        """Combine drift scores with importance into a feature health report.

        When ``recalculate_importance`` is set to ``"daily"``,
        ``"weekly"``, or ``"monthly"`` and a model + reference data are
        available, importances are automatically recomputed if the
        configured interval has elapsed since the last calculation.
        """
        self._maybe_recompute_importance()
        with self._lock:
            data = self._buffered_features()
        if data is None:
            raise SentinelError("no buffered predictions for feature health")
        if not self._drift_detector.is_fitted():
            self._drift_detector.fit(data)
        return self.feature_health.evaluate(self._drift_detector, data)

    def _maybe_recompute_importance(self) -> None:
        """Recompute feature importances if the recalculation interval elapsed."""
        import numpy as _np

        schedule = self.config.feature_health.recalculate_importance
        interval_hours = self._RECALC_INTERVAL_HOURS.get(schedule)
        if interval_hours is None:
            return  # "never" or unknown

        if self._model is None or self._reference_data is None:
            return

        now = datetime.now(timezone.utc)
        if self._last_importance_calc is not None:
            elapsed = (now - self._last_importance_calc).total_seconds() / 3600.0
            if elapsed < interval_hours:
                return

        arr = _np.asarray(self._reference_data)
        names = [f"f{i}" for i in range(arr.shape[1] if arr.ndim > 1 else 1)]
        self.feature_health.compute_importance(
            self._model, arr, y=self._reference_y, feature_names=names,
        )
        self._last_importance_calc = now

    # ── Registry ──────────────────────────────────────────────────

    def register_model(
        self,
        version: str,
        model: Any = None,
        **metadata: Any,
    ) -> ModelVersion:
        """Register a new version of this model.

        Args:
            version: Semantic version string.
            model: Optional trained model object. When provided AND
                ``config.registry.serialize_artifacts`` is True, the model
                is serialized and stored alongside its metadata.
            **metadata: Forwarded to the registry.

        Returns:
            The registered :class:`ModelVersion`.
        """
        if model is not None and self.config.registry.serialize_artifacts:
            mv = self.registry.register_with_artifact(
                self.model_name,
                version,
                model=model,
                serializer_name=self.config.registry.serializer,
                **metadata,
            )
        else:
            mv = self.registry.register(self.model_name, version, **metadata)
        self.model_version = version
        self.audit.log(
            event_type="model_registered",
            model_name=self.model_name,
            model_version=version,
            metadata=metadata,
        )
        return mv

    def register_model_if_new(self, version: str, **metadata: Any) -> ModelVersion:
        return self.registry.register_if_new(self.model_name, version, **metadata)

    @property
    def current_version(self) -> str | None:
        return self.model_version

    # ── Deployment ────────────────────────────────────────────────

    def deploy(
        self,
        version: str,
        strategy: str | None = None,
        traffic_pct: int | None = None,
    ) -> DeploymentState:
        """Begin a deployment for the given version."""
        if traffic_pct is not None and not (0 <= traffic_pct <= 100):
            raise ValueError(f"traffic_pct must be between 0 and 100, got {traffic_pct}")
        self.hooks.dispatch(HookType.BEFORE_DEPLOYMENT, version, strategy)
        state = self.deployment_manager.start(
            model_name=self.model_name,
            to_version=version,
            strategy_override=strategy,
        )
        if traffic_pct is not None:
            state = state.model_copy(update={"traffic_pct": traffic_pct})
        self.hooks.dispatch(HookType.AFTER_DEPLOYMENT, state)
        return state

    # ── LLMOps / AgentOps lazy accessors ──────────────────────────

    @property
    def llmops(self) -> Any:
        """Returns the LLMOps subsystem if enabled."""
        if not self.config.llmops.enabled:
            raise SentinelError("LLMOps is not enabled in config")
        if self._llmops_init_failed:
            raise SentinelError("LLMOps initialization previously failed")
        if self._llmops is None:
            try:
                from sentinel.llmops import LLMOpsClient

                self._llmops = LLMOpsClient(self.config.llmops, audit=self.audit)
            except Exception as e:
                self._llmops_init_failed = True
                raise SentinelError(f"LLMOps initialization failed: {e}") from e
        return self._llmops

    @property
    def agentops(self) -> Any:
        """Returns the AgentOps subsystem if enabled."""
        if not self.config.agentops.enabled:
            raise SentinelError("AgentOps is not enabled in config")
        if self._agentops_init_failed:
            raise SentinelError("AgentOps initialization previously failed")
        if self._agentops is None:
            try:
                from sentinel.agentops import AgentOpsClient

                self._agentops = AgentOpsClient(self.config.agentops, audit=self.audit)
            except Exception as e:
                self._agentops_init_failed = True
                raise SentinelError(f"AgentOps initialization failed: {e}") from e
        return self._agentops

    def log_llm_call(self, **kwargs: Any) -> Any:
        """Convenience proxy to llmops.log_call()."""
        return self.llmops.log_call(**kwargs)

    def fit_semantic_baseline(self, outputs: list[str]) -> None:
        """Initialize LLMOps semantic drift baseline.

        See :meth:`~sentinel.llmops.client.LLMOpsClient.fit_semantic_baseline`.
        """
        self.llmops.fit_semantic_baseline(outputs)

    # ── Dataset registry ──────────────────────────────────────────

    @property
    def datasets(self) -> DatasetRegistry:
        """Returns the dataset metadata registry (lazy init)."""
        if self._dataset_registry is None:
            ds_cfg = self.config.datasets
            self._dataset_registry = DatasetRegistry(
                storage_path=Path(ds_cfg.registry_path),
                auto_hash=ds_cfg.auto_hash,
                require_schema=ds_cfg.require_schema,
            )
        return self._dataset_registry

    # ── Status / health ───────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Quick health snapshot — used by /health endpoints."""
        return {
            "model": self.model_name,
            "version": self.model_version or "unregistered",
            "domain": self.config.model.domain,
            "buffer": len(self._prediction_buffer),
            "drift_fitted": self._drift_detector.is_fitted(),
            "registered_versions": self.registry.list_versions(self.model_name),
            "llmops_enabled": self.config.llmops.enabled,
            "agentops_enabled": self.config.agentops.enabled,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Lifecycle ──────────────────────────────────────────────────

    def reset_drift_baseline(self, reference: Any | None = None) -> None:
        """Reset drift detectors, optionally with new reference data."""
        with self._lock:
            if reference is not None and self._drift_detector:
                self._drift_detector.fit(reference)
            self._prediction_buffer.clear()
            self._actuals.clear()
            self._concept_observations = 0

    def clear_cohort_data(self) -> None:
        """Clear accumulated cohort data."""
        if self._cohort_analyzer:
            self._cohort_analyzer.clear()

    @property
    def cohort_analyzer(self) -> Any:
        """Public accessor for the cohort analyser (may be None)."""
        return self._cohort_analyzer

    @property
    def explainability_engine(self) -> Any:
        """Public accessor for the explainability engine (may be None)."""
        return self._explainability

    @property
    def experiments(self) -> Any:
        """Lazy accessor for the :class:`ExperimentTracker`.

        Returns:
            The :class:`~sentinel.foundation.experiments.ExperimentTracker`
            instance, initialised on first access from the config.
        """
        if self._experiment_tracker is None:
            from sentinel.foundation.experiments.tracker import ExperimentTracker

            self._experiment_tracker = ExperimentTracker(
                storage_path=self.config.experiments.storage_path,
                max_metric_history=self.config.experiments.max_metric_history,
            )
        return self._experiment_tracker

    def close(self) -> None:
        """Shut down background threads and release resources."""
        if self._scheduler is not None:
            self._scheduler.stop()
        self.notifications.close()
        self.audit.close()

    def __enter__(self) -> SentinelClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ── Explainability (WS-E) ─────────────────────────────────────

    def set_model_for_explanations(
        self,
        model: Any,
        feature_names: list[str],
        background_data: Any = None,
    ) -> None:
        """Attach a model for SHAP/LIME explanations (lazy import)."""
        from sentinel.intelligence.explainability import ExplainabilityEngine

        self._explainability = ExplainabilityEngine(
            model, feature_names, background_data=background_data
        )

    def explain(self, X: Any) -> list[dict[str, float]]:
        """Compute per-row feature attributions.

        Requires a prior call to :meth:`set_model_for_explanations`.

        Args:
            X: Feature matrix (n_samples x n_features).

        Returns:
            List of dicts mapping feature name → attribution value.

        Raises:
            SentinelError: If no model has been set for explanations.
        """
        if self._explainability is None:
            raise SentinelError("call set_model_for_explanations() first")
        return self._explainability.explain(X)

    def explain_global(self, X: Any) -> dict[str, float]:
        """Compute global feature importance (mean |attribution|).

        Args:
            X: Feature matrix (n_samples x n_features).

        Returns:
            Dict mapping feature name → mean absolute attribution,
            sorted descending by importance.

        Raises:
            SentinelError: If no model has been set for explanations.
        """
        if self._explainability is None:
            raise SentinelError("call set_model_for_explanations() first")
        result = self._explainability.explain_global(X)
        self.audit.log(
            event_type="global_explanation",
            model_name=self.model_name,
            n_features=len(result),
        )
        return result

    def explain_cohorts(self, X: Any, cohort_labels: list[str]) -> dict[str, dict[str, float]]:
        """Compute per-cohort feature importance for cross-cohort comparison.

        Args:
            X: Feature matrix (n_samples x n_features).
            cohort_labels: Length-n list assigning each row to a cohort.

        Returns:
            Dict mapping cohort_id → {feature → mean |attribution|}.

        Raises:
            SentinelError: If no model has been set for explanations.
        """
        if self._explainability is None:
            raise SentinelError("call set_model_for_explanations() first")
        result = self._explainability.explain_cohorts(X, cohort_labels)
        self.audit.log(
            event_type="cohort_explanation",
            model_name=self.model_name,
            n_cohorts=len(result),
        )
        return result

    # ── Cohort analysis ───────────────────────────────────────────

    def get_cohort_report(self, cohort_id: str) -> Any:
        """Get a performance report for a single cohort.

        Args:
            cohort_id: The cohort to report on.

        Returns:
            :class:`CohortPerformanceReport` or ``None`` if cohort
            analysis is disabled or the cohort is unknown.
        """
        if self._cohort_analyzer is None:
            return None
        return self._cohort_analyzer.get_cohort_report(cohort_id)

    def compare_cohorts(self) -> Any:
        """Compare all tracked cohorts and flag performance disparities.

        Returns:
            :class:`CohortComparativeReport` or ``None`` if cohort
            analysis is disabled.
        """
        if self._cohort_analyzer is None:
            return None
        report = self._cohort_analyzer.compare_cohorts()
        if report.disparity_flags:
            self.audit.log(
                event_type="cohort_disparity_detected",
                model_name=self.model_name,
                flagged_cohorts=report.disparity_flags,
            )
            alert = Alert(
                model_name=self.model_name,
                title=f"Cohort disparity: {len(report.disparity_flags)} cohort(s) flagged",
                body=report.summary,
                severity=AlertSeverity.HIGH,
                source="cohort_analyzer",
                payload={"flagged": report.disparity_flags},
                fingerprint=f"cohort_disparity:{self.model_name}",
            )
            self.notifications.dispatch(alert)
        return report
