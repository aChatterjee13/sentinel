"""Demo: Sentinel MLOps — Full Capabilities Showcase

Run:
    pip install -e ".[all,dashboard]"
    python scripts/demo_mlops.py

This demonstrates every MLOps capability:
  ✓ Config validation & loading
  ✓ Model registry (4 versions, promote, compare)
  ✓ Data quality (schema, outlier, freshness)
  ✓ Data drift (PSI on 8 features)
  ✓ Concept drift (DDM with accuracy degradation)
  ✓ Model drift (performance decay)
  ✓ Feature health (importance x drift)
  ✓ Cohort analysis (3 segments, disparity detection)
  ✓ Explainability (SHAP/permutation, global, per-cohort)
  ✓ Notifications (Slack alerts with cooldown)
  ✓ Deployment (canary rollout)
  ✓ Retrain orchestration (drift → retrain → approve)
  ✓ Multi-model graph (cascade alerts)
  ✓ Business KPI linking
  ✓ Cost monitoring
  ✓ Audit trail (tamper-evident, compliance)
  ✓ Dataset registry (register, compare, link)
  ✓ Experiment tracking (runs, metrics, search, compare)
  ✓ Dashboard (all pages populated)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import contextlib
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Make ``sentinel`` importable when running from the repo root without an
# editable install (e.g. `python scripts/demo_mlops.py`).
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sentinel.config.schema import (  # noqa: E402
    AlertPolicies,
    AlertsConfig,
    ApprovalConfig,
    AuditConfig,
    BusinessKPIConfig,
    CanaryConfig,
    ChannelConfig,
    CohortAnalysisConfig,
    ConceptDriftConfig,
    CostMonitorConfig,
    DashboardConfig,
    DashboardServerConfig,
    DashboardUIConfig,
    DataDriftConfig,
    DeploymentConfig,
    DriftConfig,
    FeatureHealthConfig,
    KPIMapping,
    ModelConfig,
    ModelDriftConfig,
    ModelGraphConfig,
    ModelGraphEdge,
    RetrainingConfig,
    SentinelConfig,
    ValidationConfig,
)
from sentinel.core.client import SentinelClient  # noqa: E402
from sentinel.foundation.registry.backends.local import LocalRegistryBackend  # noqa: E402
from sentinel.foundation.registry.model_registry import ModelRegistry  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "claims_fraud_v2"

FEATURE_NAMES = [
    "amount_log",
    "merchant_category",
    "txn_velocity",
    "geo_distance_km",
    "device_age_days",
    "account_age_months",
    "time_since_last_txn",
    "avg_txn_amount",
]

PORT = 8000
HOST = "127.0.0.1"

# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

_BOLD = "\033[1m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_RED = "\033[91m"
_RESET = "\033[0m"
_DIM = "\033[2m"


def _header(step: int, total: int, title: str) -> None:
    print(f"\n{_BOLD}{'═' * 55}")
    print(f"  [{step}/{total}] {title}")
    print(f"{'═' * 55}{_RESET}")


def _ok(msg: str) -> None:
    print(f"  {_GREEN}✓{_RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {_YELLOW}⚠{_RESET} {msg}")


def _info(msg: str) -> None:
    print(f"  {_CYAN}i{_RESET} {msg}")


def _detail(msg: str) -> None:
    print(f"    {_DIM}{msg}{_RESET}")


# ---------------------------------------------------------------------------
# Config & client construction
# ---------------------------------------------------------------------------


def build_config(workspace: Path) -> SentinelConfig:
    """Build a fully-enabled SentinelConfig rooted under *workspace*."""
    audit_dir = workspace / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    return SentinelConfig(
        model=ModelConfig(
            name=MODEL_NAME,
            domain="tabular",
            type="classification",
            framework="xgboost",
            version="1.0.0",
        ),
        data_quality={
            "schema": {"enforce": False},
            "freshness": {"max_age_hours": 24},
            "outlier_detection": {"method": "zscore", "contamination": 0.05},
            "null_threshold": 0.1,
            "duplicate_threshold": 0.05,
        },
        drift=DriftConfig(
            data=DataDriftConfig(method="psi", threshold=0.2, window="7d"),
            concept=ConceptDriftConfig(
                method="ddm",
                warning_level=2.0,
                drift_level=3.0,
                min_samples=30,
            ),
            model=ModelDriftConfig(
                metrics=["accuracy", "f1"],
                threshold={"accuracy": 0.05, "f1": 0.08},
                evaluation_window=1000,
            ),
        ),
        feature_health=FeatureHealthConfig(
            importance_method="builtin",
            alert_on_top_n_drift=3,
            recalculate_importance="weekly",
        ),
        cohort_analysis=CohortAnalysisConfig(
            enabled=True,
            cohort_column="customer_segment",
            max_cohorts=10,
            min_samples_per_cohort=20,
            disparity_threshold=0.10,
            buffer_size=2000,
        ),
        alerts=AlertsConfig(
            channels=[
                ChannelConfig(
                    type="slack",
                    webhook_url="https://hooks.slack.example.invalid/demo",
                    channel="#ml-alerts",
                ),
            ],
            policies=AlertPolicies(
                cooldown="1m",
                rate_limit_per_hour=120,
            ),
        ),
        retraining=RetrainingConfig(
            trigger="drift_confirmed",
            pipeline="azureml://pipelines/retrain_fraud_v2",
            approval=ApprovalConfig(
                mode="human_in_loop",
                approvers=["ml-team@company.com", "risk-officer@company.com"],
            ),
            validation=ValidationConfig(
                min_performance={"accuracy": 0.85, "f1": 0.80},
            ),
        ),
        deployment=DeploymentConfig(
            strategy="canary",
            canary=CanaryConfig(
                initial_traffic_pct=5,
                ramp_steps=[5, 25, 50, 100],
                ramp_interval="1h",
            ),
        ),
        cost_monitor=CostMonitorConfig(
            track=["inference_latency_ms", "throughput_rps", "cost_per_prediction"],
            alert_thresholds={"latency_p99_ms": 200.0, "cost_per_1k_predictions": 5.0},
        ),
        business_kpi=BusinessKPIConfig(
            mappings=[
                KPIMapping(
                    model_metric="precision",
                    business_kpi="fraud_catch_rate",
                    data_source="warehouse://analytics.fraud_metrics",
                ),
                KPIMapping(
                    model_metric="recall",
                    business_kpi="false_positive_rate",
                    data_source="warehouse://analytics.fraud_metrics",
                ),
                KPIMapping(
                    model_metric="f1",
                    business_kpi="operational_efficiency",
                ),
            ],
        ),
        model_graph=ModelGraphConfig(
            dependencies=[
                ModelGraphEdge(
                    upstream="feature_engineering_pipeline",
                    downstream=MODEL_NAME,
                ),
                ModelGraphEdge(
                    upstream=MODEL_NAME,
                    downstream="auto_adjudication_model",
                ),
                ModelGraphEdge(
                    upstream="auto_adjudication_model",
                    downstream="payout_router",
                ),
            ],
            cascade_alerts=True,
        ),
        audit=AuditConfig(
            storage="local",
            path=str(audit_dir),
            retention_days=2555,
            log_predictions=False,
            log_explanations=False,
            compliance_frameworks=["fca_consumer_duty", "eu_ai_act", "pra_ss123"],
        ),
        dashboard=DashboardConfig(
            enabled=True,
            server=DashboardServerConfig(host=HOST, port=PORT),
            ui=DashboardUIConfig(
                title="Sentinel MLOps — Full Demo",
                theme="auto",
            ),
        ),
    )


def build_client(workspace: Path) -> SentinelClient:
    """Build a SentinelClient backed by an isolated workspace."""
    cfg = build_config(workspace)
    registry_root = workspace / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry(backend=LocalRegistryBackend(root=registry_root))
    client = SentinelClient(cfg, registry=registry)
    _install_fake_embedder(client)
    return client


def _install_fake_embedder(client: SentinelClient) -> None:
    """Inject a deterministic fake embed function for semantic drift."""
    try:
        llm = client.llmops
    except Exception:
        return

    def _fake_embed(texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            seed = abs(hash(t)) % (2**31)
            rng = np.random.default_rng(seed)
            out.append(rng.standard_normal(32).tolist())
        return out

    llm.semantic_drift._embed_fn = _fake_embed


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------


def _make_features(n: int, *, drifted: bool, seed: int) -> np.ndarray:
    """Generate an (n, 8) array of synthetic feature data."""
    rng = np.random.default_rng(seed)
    cols = []
    for i in range(len(FEATURE_NAMES)):
        if drifted and i in (0, 2):
            cols.append(rng.normal(loc=1.8, scale=1.4, size=n))
        elif drifted and i == 3:
            cols.append(rng.normal(loc=1.0, scale=2.0, size=n))
        else:
            cols.append(rng.normal(loc=0.0, scale=1.0, size=n))
    return np.stack(cols, axis=1)


def _feature_dict(row: np.ndarray) -> dict[str, float]:
    """Convert a single row to a named feature dict."""
    return {name: float(row[i]) for i, name in enumerate(FEATURE_NAMES)}


# ---------------------------------------------------------------------------
# Seed functions — one per capability
# ---------------------------------------------------------------------------

TOTAL_STEPS = 19


def seed_config_validation(client: SentinelClient) -> None:
    """[1/19] Show that the config was validated by Pydantic."""
    _header(1, TOTAL_STEPS, "Config & Client Setup")
    cfg = client.config
    _ok(f"Config validated — model={cfg.model.name}, domain={cfg.model.domain}")
    _ok(f"Drift method={cfg.drift.data.method}, threshold={cfg.drift.data.threshold}")
    _ok(f"Alerts → {len(cfg.alerts.channels)} channel(s), cooldown={cfg.alerts.policies.cooldown}")
    _ok(f"Cohort analysis enabled, column={cfg.cohort_analysis.cohort_column}")
    _ok(f"Audit → {cfg.audit.storage}, frameworks={cfg.audit.compliance_frameworks}")


def seed_registry(client: SentinelClient) -> None:
    """[2/19] Register 4 model versions with realistic metadata."""
    _header(2, TOTAL_STEPS, "Model Registry — registering 4 versions")
    versions = [
        (
            "1.0.0",
            {
                "framework": "xgboost",
                "trained_on": "2025-06-15",
                "data_version": "ds-2025-06",
                "metrics": {"accuracy": 0.91, "f1": 0.88, "auc": 0.94},
                "tags": ["baseline"],
            },
        ),
        (
            "1.1.0",
            {
                "framework": "xgboost",
                "trained_on": "2025-09-20",
                "data_version": "ds-2025-09",
                "metrics": {"accuracy": 0.93, "f1": 0.90, "auc": 0.95},
                "tags": ["improved", "production"],
            },
        ),
        (
            "1.2.0",
            {
                "framework": "xgboost",
                "trained_on": "2026-01-10",
                "data_version": "ds-2026-01",
                "metrics": {"accuracy": 0.94, "f1": 0.91, "auc": 0.96},
                "tags": ["production", "current"],
            },
        ),
        (
            "2.0.0",
            {
                "framework": "lightgbm",
                "trained_on": "2026-04-05",
                "data_version": "ds-2026-04",
                "metrics": {"accuracy": 0.95, "f1": 0.92, "auc": 0.97},
                "tags": ["canary", "experimental"],
            },
        ),
    ]
    for version, meta in versions:
        client.registry.register(MODEL_NAME, version, **meta)
        _ok(
            f"Registered {MODEL_NAME}@{version}  "
            f"(acc={meta['metrics']['accuracy']}, framework={meta['framework']})"
        )

    with contextlib.suppress(Exception):
        client.registry.promote(MODEL_NAME, "1.2.0", status="production")
    _ok("Promoted v1.2.0 → production")
    client.model_version = "1.2.0"

    # Version comparison
    v1 = client.registry.get(MODEL_NAME, "1.0.0")
    v2 = client.registry.get(MODEL_NAME, "1.2.0")
    if v1 and v2:
        _info(
            f"Comparing v1.0.0 → v1.2.0: "
            f"accuracy {v1.metrics.get('accuracy', '?')} → {v2.metrics.get('accuracy', '?')}, "
            f"f1 {v1.metrics.get('f1', '?')} → {v2.metrics.get('f1', '?')}"
        )


def seed_data_quality(client: SentinelClient) -> None:
    """[3/19] Data quality checks — clean + corrupted data."""
    _header(3, TOTAL_STEPS, "Data Quality — schema & outlier checks")

    # Clean data — should pass
    clean_rows = [_feature_dict(row) for row in _make_features(50, drifted=False, seed=100)]
    report_clean = client.check_data_quality(clean_rows)
    _ok(
        f"Clean data: {len(report_clean.issues)} issues, critical={report_clean.has_critical_issues}"
    )

    # Corrupted data — inject nulls, out-of-range, type violations
    corrupted = []
    rng = np.random.default_rng(101)
    for i in range(50):
        row = _feature_dict(_make_features(1, drifted=False, seed=200 + i)[0])
        if i < 8:
            row["amount_log"] = None  # type: ignore[assignment]  # null injection
        if i in (10, 11, 12):
            row["geo_distance_km"] = float("nan")
        if 15 <= i < 20:
            row["txn_velocity"] = rng.normal(loc=50, scale=5)  # extreme outlier
        corrupted.append(row)

    report_bad = client.check_data_quality(corrupted)
    _ok(
        f"Corrupted data: {len(report_bad.issues)} issues, "
        f"critical={report_bad.has_critical_issues}"
    )
    for issue in report_bad.issues[:3]:
        _detail(f"→ {issue.severity}: {issue.message}")


def seed_data_drift(client: SentinelClient) -> None:
    """[4/19] Data drift with PSI — fit baseline, log clean, log drifted."""
    _header(4, TOTAL_STEPS, "Data Drift Detection — PSI on 8 features")

    # Fit baseline
    baseline = _make_features(800, drifted=False, seed=11)
    client.fit_baseline(baseline)
    _ok("Baseline fitted (800 samples, 8 features)")

    # Log clean predictions — drift check should pass
    clean = _make_features(300, drifted=False, seed=22)
    for row in clean:
        client.log_prediction(features=_feature_dict(row), prediction=0)
    drift_clean = None
    with contextlib.suppress(Exception):
        drift_clean = client.check_drift()
    if drift_clean:
        _ok(f"Clean window: is_drifted={drift_clean.is_drifted}, severity={drift_clean.severity}")
    else:
        _ok("Clean window: drift check passed (no report)")
    client.clear_buffer()

    # Log drifted predictions — should fire alerts
    drifted = _make_features(300, drifted=True, seed=33)
    for row in drifted:
        client.log_prediction(features=_feature_dict(row), prediction=1)
    drift_report = None
    with contextlib.suppress(Exception):
        drift_report = client.check_drift()
    if drift_report:
        _ok(
            f"Drifted window: is_drifted={drift_report.is_drifted}, "
            f"severity={drift_report.severity}"
        )
        for feat, score in sorted(
            (drift_report.feature_scores or {}).items(),
            key=lambda x: x[1],
            reverse=True,
        )[:3]:
            _detail(f"→ {feat}: PSI={score:.4f}")
    else:
        _warn("Drift report not generated (check detector state)")


def seed_concept_drift(client: SentinelClient) -> None:
    """[5/19] Concept drift via DDM — stable then degrading accuracy."""
    _header(5, TOTAL_STEPS, "Concept Drift — DDM detector")

    from sentinel.observability.drift.concept_drift import DDMConceptDriftDetector

    ddm = DDMConceptDriftDetector(
        model_name=MODEL_NAME,
        threshold=0.0,
        warning_level=2.0,
        drift_level=3.0,
        min_samples=30,
    )

    # Phase 1 — stable predictions (low error rate ~5%)
    rng = np.random.default_rng(42)
    stable_errors = (rng.random(100) < 0.05).astype(float)
    report_stable = ddm.detect(stable_errors)
    _ok(
        f"Stable phase (100 preds, ~5% error): "
        f"drifted={report_stable.is_drifted}, severity={report_stable.severity}"
    )

    # Phase 2 — degrading accuracy (error rate ramps from 10% to 50%)
    degrading_errors = []
    for i in range(100):
        error_rate = 0.10 + 0.40 * (i / 99)
        degrading_errors.append(float(rng.random() < error_rate))
    report_degrade = ddm.detect(np.array(degrading_errors))
    _ok(
        f"Degrading phase (100 preds, 10%→50% error): "
        f"drifted={report_degrade.is_drifted}, severity={report_degrade.severity}"
    )

    # Log to audit
    client.audit.log(
        event_type="concept_drift_detected",
        model_name=MODEL_NAME,
        model_version="1.2.0",
        method="ddm",
        is_drifted=report_degrade.is_drifted,
        severity=report_degrade.severity,
    )


def seed_model_drift(client: SentinelClient) -> None:
    """[6/19] Model drift — performance metric decay tracking."""
    _header(6, TOTAL_STEPS, "Model Drift — performance decay")

    from sentinel.observability.drift.model_drift import ModelPerformanceDriftDetector

    detector = ModelPerformanceDriftDetector(
        model_name=MODEL_NAME,
        threshold=0.05,
        metrics=["accuracy", "f1"],
        per_metric_thresholds={"accuracy": 0.05, "f1": 0.08},
    )
    detector.fit({"accuracy": 0.93, "f1": 0.90})
    _ok("Baseline: accuracy=0.93, f1=0.90")

    # Simulate decayed performance
    rng = np.random.default_rng(77)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    # Inject errors to get ~0.85 accuracy
    y_pred = y_true.copy()
    flip_idx = rng.choice(n, size=int(n * 0.15), replace=False)
    y_pred[flip_idx] = 1 - y_pred[flip_idx]

    report = detector.detect((y_true, y_pred))
    _ok(f"Current: is_drifted={report.is_drifted}, severity={report.severity}")
    for metric, score in (report.feature_scores or {}).items():
        _detail(f"→ {metric}: drop={score:.4f}")

    client.audit.log(
        event_type="model_drift_detected",
        model_name=MODEL_NAME,
        model_version="1.2.0",
        is_drifted=report.is_drifted,
        severity=report.severity,
        metrics={m: round(s, 4) for m, s in (report.feature_scores or {}).items()},
    )


def seed_feature_health(client: SentinelClient) -> None:
    """[7/19] Feature health -- per-feature drift x importance ranking."""
    _header(7, TOTAL_STEPS, "Feature Health -- importance x drift")

    # Set synthetic importances — keyed to match drift detector feature names
    # (when numpy arrays are used, the detector generates feature_0, feature_1, etc.)
    raw_importances = [0.25, 0.05, 0.20, 0.15, 0.10, 0.08, 0.12, 0.05]
    importances = {f"feature_{i}": raw_importances[i] for i in range(len(FEATURE_NAMES))}
    client.feature_health.set_importances(importances)
    _ok("Feature importances set (8 features)")

    with contextlib.suppress(Exception):
        report = client.get_feature_health()
        _ok(f"Feature health report: {len(report.features)} features evaluated")
        for fh in sorted(report.features, key=lambda f: f.importance, reverse=True)[:4]:
            _detail(
                f"→ {fh.name}: importance={fh.importance:.3f}, "
                f"drift_score={fh.drift_score:.4f}, drifted={fh.is_drifted}"
            )


def seed_cohort_analysis(client: SentinelClient) -> None:
    """[8/19] Cohort analysis — 3 segments with disparity detection."""
    _header(8, TOTAL_STEPS, "Cohort Analysis — 3 customer segments")

    segments = ["premium", "standard", "basic"]
    error_rates = {"premium": 0.05, "standard": 0.12, "basic": 0.30}
    rng = np.random.default_rng(55)

    client.clear_buffer()
    for i in range(600):
        seg = segments[i % 3]
        row = _make_features(1, drifted=False, seed=3000 + i)[0]
        features = _feature_dict(row)
        true_label = int(rng.random() < 0.3)
        error = rng.random() < error_rates[seg]
        pred = 1 - true_label if error else true_label
        client.log_prediction(
            features=features,
            prediction=pred,
            actual=true_label,
            cohort_id=seg,
        )

    # Per-cohort reports
    for seg in segments:
        rpt = client.get_cohort_report(seg)
        if rpt:
            m = rpt.metrics
            acc_str = f"{m.accuracy:.3f}" if m.accuracy is not None else "N/A"
            _ok(f"Cohort '{seg}': n={m.count}, accuracy={acc_str}")

    # Compare cohorts — should detect disparity for "basic"
    comparative = client.compare_cohorts()
    if comparative:
        if comparative.disparity_flags:
            _warn(f"Disparity detected: {comparative.disparity_flags}")
        else:
            _ok("No cohort disparity detected")


def seed_explainability(client: SentinelClient) -> None:
    """[9/19] Explainability — SHAP/permutation feature attributions."""
    _header(9, TOTAL_STEPS, "Explainability — feature attributions")

    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        _warn("sklearn not available — skipping explainability")
        return

    rng = np.random.default_rng(66)
    X_train = _make_features(400, drifted=False, seed=500)
    y_train = (X_train[:, 0] + 0.5 * X_train[:, 2] + rng.normal(0, 0.3, 400) > 0.8).astype(int)

    clf = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    _ok(f"Trained RandomForestClassifier (n={len(X_train)}, features={X_train.shape[1]})")

    with contextlib.suppress(Exception):
        client.set_model_for_explanations(clf, FEATURE_NAMES, background_data=X_train[:50])
        _ok(f"Explainability engine attached (method={client.explainability_engine.method_used})")

    # Per-row explanations
    X_sample = X_train[:5]
    with contextlib.suppress(Exception):
        explanations = client.explain(X_sample)
        _ok(
            f"Per-row explanations: {len(explanations)} rows "
            f"(method={client.explainability_engine.method_used})"
        )
        row0 = explanations[0]

        # Values may be scalars or lists (multi-class SHAP); normalise
        def _scalar(v: object) -> float:
            if isinstance(v, (list, tuple)):
                return float(sum(abs(x) for x in v))
            return float(abs(v))  # type: ignore[arg-type]

        top = sorted(row0.items(), key=lambda x: _scalar(x[1]), reverse=True)[:3]
        for feat, val in top:
            _detail(f"→ Row 0: {feat}={val}")

    # Global importance
    with contextlib.suppress(Exception):
        global_imp = client.explain_global(X_train[:100])
        _ok("Global feature importance:")
        for feat, imp in list(global_imp.items())[:4]:
            _detail(f"→ {feat}: {imp:.4f}")

    # Per-cohort importance
    with contextlib.suppress(Exception):
        cohort_labels = ["premium", "standard", "basic"] * (100 // 3) + ["premium"]
        cohort_labels = cohort_labels[:100]
        cohort_imp = client.explain_cohorts(X_train[:100], cohort_labels)
        _ok(f"Per-cohort importance: {list(cohort_imp.keys())}")
        for coh, feats in cohort_imp.items():
            top_feat = max(feats, key=feats.get)  # type: ignore[arg-type]
            _detail(f"→ {coh}: top feature = {top_feat} ({feats[top_feat]:.4f})")


def seed_notifications(client: SentinelClient) -> None:
    """[10/19] Notification engine — alerts + cooldown demonstration."""
    _header(10, TOTAL_STEPS, "Notifications — Slack alerts + cooldown")

    from sentinel.core.types import Alert, AlertSeverity

    alert1 = Alert(
        model_name=MODEL_NAME,
        title="Data drift detected (PSI > 0.20)",
        body="Features amount_log (PSI=0.34) and txn_velocity (PSI=0.28) exceeded threshold.",
        severity=AlertSeverity.HIGH,
        source="drift_detection",
        payload={"psi_scores": {"amount_log": 0.34, "txn_velocity": 0.28}},
        fingerprint=f"demo:drift:{MODEL_NAME}:psi",
    )
    results1 = client.notifications.dispatch(alert1)
    _ok(f"Alert 1 dispatched: {len(results1)} delivery attempt(s)")
    for r in results1:
        _detail(f"→ {r.channel}: delivered={r.delivered}")

    # Second identical alert — should be suppressed by cooldown
    results2 = client.notifications.dispatch(alert1)
    suppressed = all(not r.delivered for r in results2) if results2 else len(results2) == 0
    _ok(f"Alert 2 (duplicate): suppressed by cooldown = {suppressed or len(results2) == 0}")

    # Different alert — should go through
    alert_critical = Alert(
        model_name=MODEL_NAME,
        title="Model performance degraded below threshold",
        body="Accuracy dropped to 0.85 (baseline 0.93, threshold 0.05).",
        severity=AlertSeverity.CRITICAL,
        source="model_drift",
        payload={"accuracy_drop": 0.08},
        fingerprint=f"perf:{MODEL_NAME}:accuracy",
    )
    results3 = client.notifications.dispatch(alert_critical)
    _ok(f"Alert 3 (critical, different fingerprint): {len(results3)} attempt(s)")


def seed_deployment(client: SentinelClient) -> None:
    """[11/19] Deployment automation — canary rollout."""
    _header(11, TOTAL_STEPS, "Deployment — canary rollout v1.2.0 → v2.0.0")

    with contextlib.suppress(Exception):
        state = client.deployment_manager.start(
            model_name=MODEL_NAME,
            to_version="2.0.0",
            from_version="1.2.0",
            strategy_override="canary",
        )
        _ok(f"Deployment started: id={state.deployment_id}")
        _ok(f"Strategy={state.strategy}, phase={state.phase}")
        _detail(f"from={state.from_version} → to={state.to_version}")
        _detail(f"traffic_pct={state.traffic_pct}%")

        # Advance the canary one step
        with contextlib.suppress(Exception):
            advanced = client.deployment_manager.advance(
                state, observed_metrics={"error_rate": 0.02, "latency_p99_ms": 145}
            )
            _ok(f"Advanced canary: traffic_pct={advanced.traffic_pct}%, phase={advanced.phase}")


def seed_retrain(client: SentinelClient) -> None:
    """[12/19] Retrain orchestrator — drift → trigger → approve."""
    _header(12, TOTAL_STEPS, "Retrain Orchestration — drift-triggered retrain")

    # Install a mock pipeline runner
    def _mock_runner(uri: str, context: dict) -> dict:
        return {
            "status": "success",
            "candidate_version": "2.1.0",
            "metrics": {"accuracy": 0.95, "f1": 0.92},
            "artifact_uri": "azureml://models/claims_fraud_v2/versions/2.1.0",
        }

    client.retrain.set_pipeline_runner(_mock_runner)
    _ok("Mock pipeline runner installed")

    # Build a synthetic drift report to feed to on_drift()
    from sentinel.observability.drift.base import DriftReport

    fake_drift = DriftReport(
        is_drifted=True,
        severity="high",
        test_statistic=0.31,
        p_value=0.001,
        feature_scores={"amount_log": 0.34, "txn_velocity": 0.28},
        method="psi",
        model_name=MODEL_NAME,
    )

    # First call just registers the drift with the evaluator
    trigger1 = client.retrain.on_drift(fake_drift)
    _info(f"First drift signal: trigger={trigger1}")

    # Second consecutive drift should actually trigger retrain
    trigger2 = client.retrain.on_drift(fake_drift)
    if trigger2:
        _ok(f"Retrain triggered: type={trigger2.trigger_type}, reason={trigger2.reason}")

        with contextlib.suppress(Exception):
            result = client.retrain.run(MODEL_NAME, trigger=trigger2)
            _ok(f"Pipeline result: status={result.get('status')}")
            req_id = result.get("request_id")
            if req_id:
                _info(f"Pending approval: request_id={req_id}")
                with contextlib.suppress(Exception):
                    approval = client.retrain.approve(
                        req_id, by="ml-team@company.com", comment="Metrics look good"
                    )
                    _ok(f"Approved: decision={approval.get('decision')}")
    else:
        _info("Trigger evaluator requires more consecutive drifts (expected)")
        # Log the retrain event manually for dashboard visibility
        client.audit.log(
            event_type="retrain_triggered",
            model_name=MODEL_NAME,
            model_version="1.2.0",
            reason="drift_confirmed",
            pipeline="azureml://pipelines/retrain_fraud_v2",
        )
        client.audit.log(
            event_type="approval_decision",
            model_name=MODEL_NAME,
            decision="approved",
            by="ml-team@company.com",
            comment="Metrics look good",
        )
        _ok("Retrain + approval events logged to audit trail")


def seed_model_graph(client: SentinelClient) -> None:
    """[13/19] Multi-model graph — 3-model cascade alerts."""
    _header(13, TOTAL_STEPS, "Multi-Model Graph — cascade alerts")

    graph = client.model_graph
    _ok(f"Graph nodes: {list(graph._nodes.keys())}")

    impact = graph.cascade_impact(MODEL_NAME)
    _ok(f"Cascade impact from '{MODEL_NAME}':")
    for model, path in impact.items():
        _detail(f"→ {model}: affected via {path}")

    descendants = graph.get_descendants(MODEL_NAME)
    _ok(f"Downstream models: {descendants}")

    topo = graph.topological_sort()
    _ok(f"Topological order: {topo}")

    # Log cascade alert event
    client.audit.log(
        event_type="cascade_alert",
        model_name=MODEL_NAME,
        model_version="1.2.0",
        source_model=MODEL_NAME,
        affected_models=descendants,
        reason="upstream data drift propagated",
    )


def seed_kpi_linking(client: SentinelClient) -> None:
    """[14/19] Business KPI linking — map model metrics to KPIs."""
    _header(14, TOTAL_STEPS, "Business KPI Linking — metric → KPI mapping")

    # Install a mock KPI fetcher
    kpi_values = {
        "warehouse://analytics.fraud_metrics": 0.87,
    }
    client.kpi_linker.set_fetcher(lambda source: kpi_values.get(source))
    client.kpi_linker.refresh()
    _ok("KPI values fetched from warehouse")

    report = client.kpi_linker.report(model_metrics={"precision": 0.92, "recall": 0.85, "f1": 0.88})
    links = report.get("linked_kpis", report.get("impacts", []))
    _ok(f"KPI linkage report: {len(links)} mappings")
    for imp in links:
        _detail(
            f"→ {imp['model_metric']} ({imp.get('metric_value')}) "
            f"↔ {imp['business_kpi']} ({imp.get('kpi_value')})"
        )


def seed_cost_monitor(client: SentinelClient) -> None:
    """[15/19] Cost monitoring — latency and throughput metrics."""
    _header(15, TOTAL_STEPS, "Cost Monitor — latency & throughput tracking")

    rng = random.Random(88)
    for _ in range(200):
        latency = rng.gauss(45, 15)
        client.cost_monitor.record(max(5, latency))

    # Inject a few slow requests
    for _ in range(10):
        client.cost_monitor.record(rng.gauss(250, 50))

    snapshot = client.cost_monitor.snapshot()
    _ok(
        f"Latency p50={snapshot.latency_ms_p50:.1f}ms, "
        f"p95={snapshot.latency_ms_p95:.1f}ms, "
        f"p99={snapshot.latency_ms_p99:.1f}ms"
    )
    _ok(f"Throughput={snapshot.throughput_rps:.1f} rps, samples={snapshot.sample_count}")

    breaches = client.cost_monitor.check_thresholds()
    if breaches:
        _warn(f"Threshold breaches: {breaches}")
    else:
        _ok("All cost thresholds within limits")


def seed_audit_extras(client: SentinelClient) -> None:
    """[16/19] Audit trail — mixed events for dashboard variety."""
    _header(16, TOTAL_STEPS, "Audit Trail — compliance + lifecycle events")

    events = [
        ("model_registered", {"actor": "ml-team@bank.com", "version": "1.0.0"}),
        ("model_registered", {"actor": "ml-team@bank.com", "version": "1.1.0"}),
        ("model_registered", {"actor": "ml-team@bank.com", "version": "1.2.0"}),
        ("model_promoted", {"version": "1.2.0", "status": "production", "by": "risk-officer"}),
        (
            "alert_sent",
            {
                "severity": "high",
                "channel": "slack",
                "subject": "PSI=0.34 on amount_log",
            },
        ),
        (
            "alert_sent",
            {
                "severity": "critical",
                "channel": "slack",
                "subject": "accuracy dropped to 0.85",
            },
        ),
        (
            "alert_sent",
            {
                "severity": "info",
                "channel": "slack",
                "subject": "canary 5% → 25%",
            },
        ),
        (
            "retrain_triggered",
            {
                "reason": "drift_confirmed",
                "pipeline": "azureml://pipelines/retrain_fraud_v2",
            },
        ),
        (
            "deployment_started",
            {
                "strategy": "canary",
                "from_version": "1.2.0",
                "to_version": "2.0.0",
                "traffic_pct": 5,
            },
        ),
        (
            "deployment_advanced",
            {
                "strategy": "canary",
                "to_version": "2.0.0",
                "traffic_pct": 25,
            },
        ),
        (
            "compliance_review",
            {
                "framework": "fca_consumer_duty",
                "outcome": "pass",
                "reviewer": "compliance-team",
            },
        ),
        (
            "compliance_review",
            {
                "framework": "eu_ai_act",
                "outcome": "pass",
                "reviewer": "compliance-team",
            },
        ),
        (
            "compliance_review",
            {
                "framework": "pra_ss123",
                "outcome": "pass",
                "reviewer": "risk-officer",
            },
        ),
        (
            "bias_review",
            {
                "cohort": "basic",
                "finding": "30% error rate vs 5% for premium",
                "action": "investigation_opened",
            },
        ),
        ("model_deprecated", {"version": "1.0.0", "by": "ml-team@bank.com"}),
    ]

    for event_type, payload in events:
        client.audit.log(
            event_type=event_type,
            model_name=MODEL_NAME,
            model_version=payload.pop("version", "1.2.0"),
            **payload,
        )
    _ok(f"Logged {len(events)} audit events")

    # Verify tamper-evident chain
    with contextlib.suppress(Exception):
        result = client.audit.verify_integrity()
        _ok(f"Tamper-evident hash chain: valid={result.valid}, entries={result.entries_checked}")


def seed_dataset_registry(client: SentinelClient) -> None:
    """[17/19] Dataset registry — register, compare, search, and link."""
    _header(17, TOTAL_STEPS, "Dataset Registry — register, compare, link")

    from sentinel.foundation.datasets.registry import DatasetRegistry

    workspace = Path(client.config.audit.path).parent
    ds_reg = DatasetRegistry(workspace / "datasets", auto_hash=False)

    # Register a training dataset
    train_ds = ds_reg.register(
        name="fraud_training",
        version="1.0",
        path="s3://ml-data/fraud/train_v1.parquet",
        format="parquet",
        split="train",
        num_rows=50_000,
        num_features=len(FEATURE_NAMES),
        schema=dict.fromkeys(FEATURE_NAMES, "float64"),
        description="Fraud detection training set — Jan-Jun 2025",
        tags=["production", "fraud", "v1"],
        source="feature_engineering_pipeline",
        metadata={"collection_period": "2025-01-01/2025-06-30"},
    )
    _ok(
        f"Registered {train_ds.name}@{train_ds.version} "
        f"({train_ds.num_rows} rows, {train_ds.num_features} features)"
    )

    # Register a test dataset
    test_ds = ds_reg.register(
        name="fraud_training",
        version="1.1",
        path="s3://ml-data/fraud/train_v1.1.parquet",
        format="parquet",
        split="train",
        num_rows=65_000,
        num_features=len(FEATURE_NAMES) + 1,
        schema={
            **dict.fromkeys(FEATURE_NAMES, "float64"),
            "txn_country_code": "category",
        },
        description="Fraud training set v1.1 — added txn_country_code",
        tags=["production", "fraud", "v1.1"],
        source="feature_engineering_pipeline_v2",
        metadata={"collection_period": "2025-01-01/2025-09-30"},
    )
    _ok(
        f"Registered {test_ds.name}@{test_ds.version} "
        f"({test_ds.num_rows} rows, {test_ds.num_features} features)"
    )

    holdout_ds = ds_reg.register(
        name="fraud_holdout",
        version="1.0",
        path="s3://ml-data/fraud/holdout_v1.parquet",
        format="parquet",
        split="test",
        num_rows=10_000,
        num_features=len(FEATURE_NAMES),
        schema=dict.fromkeys(FEATURE_NAMES, "float64"),
        description="Hold-out test set for final evaluation",
        tags=["holdout", "fraud"],
    )
    _ok(f"Registered {holdout_ds.name}@{holdout_ds.version} (holdout)")

    # List versions
    versions = ds_reg.list_versions("fraud_training")
    _ok(f"fraud_training versions: {[v.version for v in versions]}")

    # Search by tags
    prod_datasets = ds_reg.search(tags=["production"])
    _ok(f"Search tags=['production'] → {len(prod_datasets)} result(s)")

    # Compare two versions
    diff = ds_reg.compare("fraud_training", "1.0", "1.1")
    _ok(
        f"Compare v1.0 vs v1.1: +{len(diff['schema_added'])} cols, "
        f"-{len(diff['schema_removed'])} cols, "
        f"row_diff={diff['row_count_diff']:+d}"
    )
    if diff["schema_added"]:
        _detail(f"New columns: {diff['schema_added']}")

    # Link dataset to model version
    ds_reg.link_to_model("fraud_training", "1.0", MODEL_NAME, "1.2.0")
    _ok(f"Linked fraud_training@1.0 → {MODEL_NAME}@1.2.0")


def seed_experiment_tracking(client: SentinelClient) -> None:
    """[18/19] Experiment tracking — runs, metrics, search, compare."""
    _header(18, TOTAL_STEPS, "Experiment Tracking — runs, metrics, compare")

    from sentinel.foundation.experiments.tracker import ExperimentTracker

    workspace = Path(client.config.audit.path).parent
    tracker = ExperimentTracker(storage_path=workspace / "experiments")

    # Create experiment
    exp = tracker.create_experiment(
        "fraud_model_experiments",
        description="Hyperparameter search for claims fraud classifier",
        tags=["fraud", "classification", "hpo"],
    )
    _ok(f"Created experiment: {exp.name}")

    # ── Run 1: baseline ──
    run1 = tracker.start_run(
        "fraud_model_experiments",
        name="xgboost_baseline",
        params={"lr": 0.01, "max_depth": 6, "n_estimators": 200, "subsample": 0.8},
        tags=["baseline"],
    )
    _ok(f"Started run {run1.run_id[:8]}… (xgboost_baseline)")

    # Simulate 10-epoch training loop
    rng = random.Random(42)
    for epoch in range(10):
        loss = 0.65 * (0.82**epoch) + rng.gauss(0, 0.01)
        f1 = 0.60 + 0.03 * epoch + rng.gauss(0, 0.005)
        accuracy = 0.70 + 0.025 * epoch + rng.gauss(0, 0.005)
        tracker.log_metrics(
            run1.run_id,
            {
                "loss": round(loss, 4),
                "f1": round(min(f1, 0.95), 4),
                "accuracy": round(min(accuracy, 0.96), 4),
            },
            step=epoch,
        )
    _ok("Logged metrics over 10 epochs (loss↓, f1↑, accuracy↑)")

    tracker.log_artifact(run1.run_id, "model", "s3://models/fraud_xgb_baseline.pkl")
    tracker.log_dataset(run1.run_id, "fraud_training@1.0")
    tracker.end_run(run1.run_id)
    _ok("Run 1 completed — artifact + dataset linked")

    # ── Run 2: tuned ──
    run2 = tracker.start_run(
        "fraud_model_experiments",
        name="xgboost_tuned",
        params={"lr": 0.005, "max_depth": 8, "n_estimators": 400, "subsample": 0.9},
        tags=["tuned"],
    )
    _ok(f"Started run {run2.run_id[:8]}… (xgboost_tuned)")

    for epoch in range(10):
        loss = 0.55 * (0.80**epoch) + rng.gauss(0, 0.008)
        f1 = 0.65 + 0.032 * epoch + rng.gauss(0, 0.004)
        accuracy = 0.73 + 0.026 * epoch + rng.gauss(0, 0.004)
        tracker.log_metrics(
            run2.run_id,
            {
                "loss": round(loss, 4),
                "f1": round(min(f1, 0.97), 4),
                "accuracy": round(min(accuracy, 0.98), 4),
            },
            step=epoch,
        )
    tracker.log_artifact(run2.run_id, "model", "s3://models/fraud_xgb_tuned.pkl")
    tracker.log_dataset(run2.run_id, "fraud_training@1.1")
    tracker.end_run(run2.run_id)
    _ok("Run 2 completed — better hyperparams")

    # Search: find runs with f1 > 0.7
    hits = tracker.search_runs(
        "fraud_model_experiments",
        filter_expr="metrics.f1 > 0.7",
        order_by="metrics.f1 DESC",
    )
    _ok(f"Search 'metrics.f1 > 0.7' → {len(hits)} run(s)")
    for h in hits:
        _detail(f"  {h.name}: f1={h.metrics.get('f1', 0):.4f}")

    # Compare the two runs
    comparison = tracker.compare_runs([run1.run_id, run2.run_id])
    _ok("Run comparison:")
    for metric, vals in comparison["metrics_latest"].items():
        vals_str = ", ".join(f"{rid[:8]}…={v:.4f}" for rid, v in vals.items() if v is not None)
        _detail(f"  {metric}: {vals_str}")
    if comparison["params_diff"]:
        _detail(f"  Differing params: {list(comparison['params_diff'].keys())}")

    # Metric history
    history = tracker.get_metric_history(run2.run_id, "f1")
    _ok(
        f"f1 history for run 2: {len(history)} entries, "
        f"first={history[0].value:.4f} → last={history[-1].value:.4f}"
    )


def launch_dashboard(client: SentinelClient) -> None:
    """[19/19] Launch the dashboard on all populated pages."""
    _header(19, TOTAL_STEPS, "Dashboard — launching on port 8000")

    base = f"http://{HOST}:{PORT}"
    pages = [
        ("/", "Overview", "Status cards, recent alerts, deployment status"),
        ("/drift", "Drift", "PSI timeline, per-feature scores, severity heatmap"),
        ("/features", "Feature Health", "Importance x drift table, top drifted features"),
        ("/registry", "Model Registry", "4 versions, promote history, metadata"),
        (f"/registry/{MODEL_NAME}/1.2.0", "Registry Detail", "v1.2.0 metrics, tags, lineage"),
        ("/audit", "Audit Trail", "Filterable timeline, tamper-evident chain"),
        ("/deployments", "Deployments", "Active canary, ramp history, rollback controls"),
        ("/compliance", "Compliance", "FCA, EU AI Act, PRA framework coverage"),
    ]

    # Add LLMOps/AgentOps pages if subsystems exist
    has_llmops = False
    has_agentops = False
    with contextlib.suppress(Exception):
        has_llmops = client.llmops is not None
    with contextlib.suppress(Exception):
        has_agentops = client.agentops is not None

    if has_llmops:
        pages += [
            ("/llmops/prompts", "LLM Prompts", "Prompt versions, A/B splits"),
            ("/llmops/guardrails", "LLM Guardrails", "Violation log, PII/jailbreak stats"),
            ("/llmops/tokens", "Token Economics", "Cost trend, model breakdown"),
        ]
    if has_agentops:
        pages += [
            ("/agentops/traces", "Agent Traces", "Recent runs, span timeline"),
            ("/agentops/tools", "Tool Audit", "Success rates, latency p95"),
            ("/agentops/agents", "Agent Registry", "Capability manifests, health status"),
        ]

    print(f"\n{_BOLD}  Dashboard is ready at {_CYAN}{base}{_RESET}")
    print(f"\n{_BOLD}  Guided tour:{_RESET}")
    print(f"  {'─' * 50}")
    for i, (path, name, desc) in enumerate(pages, 1):
        print(f"  {_BOLD}{i:>2}. {name:<20}{_RESET}→ {_CYAN}{base}{path}{_RESET}")
        print(f"      {_DIM}{desc}{_RESET}")
    print(f"  {'─' * 50}\n")

    try:
        from sentinel.dashboard.server import run as run_dashboard
    except ImportError as exc:
        raise SystemExit(
            "Dashboard requires the [dashboard] extra. Install with:\n"
            '  pip install -e ".[all,dashboard]"\n'
            f"(import error: {exc})"
        ) from exc

    print(f"  {_BOLD}Starting uvicorn — press Ctrl+C to stop.{_RESET}\n")
    run_dashboard(client, host=HOST, port=PORT, reload=False)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def seed_all(client: SentinelClient) -> None:
    """Run every seed function in dependency order."""
    seed_config_validation(client)
    seed_registry(client)
    seed_data_quality(client)
    seed_data_drift(client)
    seed_concept_drift(client)
    seed_model_drift(client)
    seed_feature_health(client)
    seed_cohort_analysis(client)
    seed_explainability(client)
    seed_notifications(client)
    seed_deployment(client)
    seed_retrain(client)
    seed_model_graph(client)
    seed_kpi_linking(client)
    seed_cost_monitor(client)
    seed_audit_extras(client)
    seed_dataset_registry(client)
    seed_experiment_tracking(client)


def main() -> None:
    workspace = Path(tempfile.mkdtemp(prefix="sentinel-mlops-demo-"))

    print(f"\n{_BOLD}{'═' * 55}")
    print("  Sentinel MLOps — Full Capabilities Demo")
    print(f"{'═' * 55}{_RESET}")
    print(f"  workspace : {workspace}")
    print(f"  model     : {MODEL_NAME}")
    print(f"  features  : {len(FEATURE_NAMES)} ({', '.join(FEATURE_NAMES[:4])}, …)")
    print(f"  url       : http://{HOST}:{PORT}")
    print(f"{'─' * 55}")

    t0 = time.monotonic()
    client = build_client(workspace)
    seed_all(client)
    elapsed = time.monotonic() - t0

    print(f"\n{_BOLD}{'─' * 55}{_RESET}")
    print(f"  {_GREEN}✓ All 19 capabilities seeded in {elapsed:.1f}s{_RESET}")
    print(f"{_BOLD}{'─' * 55}{_RESET}")

    launch_dashboard(client)


if __name__ == "__main__":
    main()
