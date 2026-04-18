"""Tests for deployment-target robustness fixes.

Covers:
* Gap 1 — Vertex AI exact version matching (``_version_matches``).
* Gap 2 — ``timeout_seconds`` parameter on every cloud target.
* Gap 3 — ``DeploymentManager.detect_stalled()`` for stuck deployments.
"""

from __future__ import annotations

import sys
import types
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from sentinel.action.deployment.strategies.base import (
    DeploymentPhase,
    DeploymentState,
)
from sentinel.config.schema import DeploymentConfig

# ────────────────────────────────────────────────────────────────────
# Helper: install fake cloud SDK modules
# ────────────────────────────────────────────────────────────────────

def _install_fake_gcp(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install a minimal fake ``google.cloud.aiplatform``."""
    aiplatform = MagicMock(name="aiplatform")
    ep = MagicMock(name="Endpoint")
    ep.gca_resource.deployed_models = []
    ep.gca_resource.traffic_split = {}
    aiplatform.Endpoint.list.return_value = [ep]

    google_mod = types.ModuleType("google")
    google_cloud_mod = types.ModuleType("google.cloud")
    google_cloud_mod.aiplatform = aiplatform  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.cloud", google_cloud_mod)
    monkeypatch.setitem(sys.modules, "google.cloud.aiplatform", aiplatform)
    return aiplatform


def _install_fake_azure_ml(monkeypatch: pytest.MonkeyPatch) -> None:
    ml_client = MagicMock(name="MLClient")
    ml_mod = types.ModuleType("azure.ai.ml")
    ml_mod.MLClient = MagicMock(return_value=ml_client)  # type: ignore[attr-defined]
    ai_mod = types.ModuleType("azure.ai")
    azure_mod = types.ModuleType("azure")
    identity_mod = types.ModuleType("azure.identity")
    identity_mod.DefaultAzureCredential = MagicMock()  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "azure", azure_mod)
    monkeypatch.setitem(sys.modules, "azure.ai", ai_mod)
    monkeypatch.setitem(sys.modules, "azure.ai.ml", ml_mod)
    monkeypatch.setitem(sys.modules, "azure.identity", identity_mod)


def _install_fake_azure_web(monkeypatch: pytest.MonkeyPatch) -> None:
    mgmt_client = MagicMock(name="WebSiteManagementClient")
    web_mod = types.ModuleType("azure.mgmt.web")
    web_mod.WebSiteManagementClient = MagicMock(return_value=mgmt_client)  # type: ignore[attr-defined]
    mgmt_mod = types.ModuleType("azure.mgmt")
    azure_mod = types.ModuleType("azure")
    identity_mod = types.ModuleType("azure.identity")
    identity_mod.DefaultAzureCredential = MagicMock()  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "azure", azure_mod)
    monkeypatch.setitem(sys.modules, "azure.mgmt", mgmt_mod)
    monkeypatch.setitem(sys.modules, "azure.mgmt.web", web_mod)
    monkeypatch.setitem(sys.modules, "azure.identity", identity_mod)


def _install_fake_boto3(monkeypatch: pytest.MonkeyPatch) -> None:
    sm_client = MagicMock(name="SageMakerClient")
    boto3_mod = types.ModuleType("boto3")
    boto3_mod.client = MagicMock(return_value=sm_client)  # type: ignore[attr-defined]
    botocore_config_mod = types.ModuleType("botocore.config")
    botocore_config_mod.Config = MagicMock(name="BotoConfig")  # type: ignore[attr-defined]
    botocore_mod = types.ModuleType("botocore")
    botocore_mod.config = botocore_config_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "boto3", boto3_mod)
    monkeypatch.setitem(sys.modules, "botocore", botocore_mod)
    monkeypatch.setitem(sys.modules, "botocore.config", botocore_config_mod)


def _install_fake_k8s(monkeypatch: pytest.MonkeyPatch) -> None:
    apps_v1 = MagicMock(name="AppsV1Api")
    client_mod = types.ModuleType("kubernetes.client")
    client_mod.AppsV1Api = MagicMock(return_value=apps_v1)  # type: ignore[attr-defined]
    config_mod = types.ModuleType("kubernetes.config")
    config_mod.load_kube_config = MagicMock()  # type: ignore[attr-defined]
    config_mod.load_incluster_config = MagicMock()  # type: ignore[attr-defined]
    kubernetes_mod = types.ModuleType("kubernetes")
    kubernetes_mod.client = client_mod  # type: ignore[attr-defined]
    kubernetes_mod.config = config_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "kubernetes", kubernetes_mod)
    monkeypatch.setitem(sys.modules, "kubernetes.client", client_mod)
    monkeypatch.setitem(sys.modules, "kubernetes.config", config_mod)


def _fresh(mod_path: str) -> Any:
    """Force-reimport a target module so it picks up patched sys.modules."""
    if mod_path in sys.modules:
        del sys.modules[mod_path]
    import importlib

    return importlib.import_module(mod_path)


# ────────────────────────────────────────────────────────────────────
# Gap 1: Vertex AI exact version matching
# ────────────────────────────────────────────────────────────────────


class TestVersionMatches:
    """Test ``VertexAIEndpointTarget._version_matches`` as a static method."""

    @pytest.fixture(autouse=True)
    def _load_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_gcp(monkeypatch)
        mod = _fresh("sentinel.action.deployment.targets.vertex_ai")
        self.cls = mod.VertexAIEndpointTarget

    def test_no_false_positive_substring(self) -> None:
        """'model-1.1' must NOT match version '1.11'."""
        assert self.cls._version_matches("model-1.1", "1.11") is False

    def test_exact_segment_match(self) -> None:
        """'model-1.1' SHOULD match version '1.1'."""
        assert self.cls._version_matches("model-1.1", "1.1") is True

    def test_version_at_end(self) -> None:
        """'fraud-2.3.1' matches version '2.3.1'."""
        assert self.cls._version_matches("fraud-2.3.1", "2.3.1") is True

    def test_version_with_underscores(self) -> None:
        """'fraud_v2.3.1' matches '2.3.1' (underscore delimiter)."""
        assert self.cls._version_matches("fraud_v2.3.1", "2.3.1") is True

    def test_version_only_display_name(self) -> None:
        """'2.3.1' matches '2.3.1' (exact equality path)."""
        assert self.cls._version_matches("2.3.1", "2.3.1") is True

    def test_empty_display_name(self) -> None:
        assert self.cls._version_matches("", "1.0") is False

    def test_empty_version(self) -> None:
        assert self.cls._version_matches("model-1.0", "") is False

    def test_no_match_at_all(self) -> None:
        assert self.cls._version_matches("model-3.0", "1.0") is False

    def test_version_not_prefix_match(self) -> None:
        """'21.1' should not match '1.1'."""
        assert self.cls._version_matches("model-21.1", "1.1") is False


# ────────────────────────────────────────────────────────────────────
# Gap 2: timeout_seconds accepted by each target
# ────────────────────────────────────────────────────────────────────


class TestTimeoutParam:
    """Each cloud target's constructor stores ``timeout_seconds``."""

    def test_vertex_ai_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_gcp(monkeypatch)
        mod = _fresh("sentinel.action.deployment.targets.vertex_ai")
        target = mod.VertexAIEndpointTarget(
            endpoint_name="ep", project="p", timeout_seconds=60,
        )
        assert target._timeout == 60

    def test_azure_ml_endpoint_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_azure_ml(monkeypatch)
        mod = _fresh("sentinel.action.deployment.targets.azure_ml_endpoint")
        target = mod.AzureMLEndpointTarget(
            endpoint_name="ep",
            subscription_id="s",
            resource_group="rg",
            workspace_name="ws",
            timeout_seconds=120,
        )
        assert target._timeout == 120

    def test_azure_app_service_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_azure_web(monkeypatch)
        mod = _fresh("sentinel.action.deployment.targets.azure_app_service")
        target = mod.AzureAppServiceTarget(
            subscription_id="s",
            resource_group="rg",
            site_name="app",
            timeout_seconds=180,
            health_check_timeout=15,
        )
        assert target._timeout == 180
        assert target._health_timeout == 15

    def test_sagemaker_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_boto3(monkeypatch)
        mod = _fresh("sentinel.action.deployment.targets.sagemaker")
        target = mod.SageMakerEndpointTarget(
            endpoint_name="ep", timeout_seconds=90,
        )
        assert target._timeout == 90

    def test_aks_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_k8s(monkeypatch)
        mod = _fresh("sentinel.action.deployment.targets.aks")
        target = mod.AKSDeploymentTarget(
            namespace="ml",
            service_name="svc",
            timeout_seconds=45,
        )
        assert target._timeout == 45

    def test_defaults_are_300(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All targets default to 300 s when ``timeout_seconds`` is omitted."""
        _install_fake_gcp(monkeypatch)
        mod = _fresh("sentinel.action.deployment.targets.vertex_ai")
        target = mod.VertexAIEndpointTarget(endpoint_name="ep", project="p")
        assert target._timeout == 300


# ────────────────────────────────────────────────────────────────────
# Gap 3: DeploymentManager.detect_stalled()
# ────────────────────────────────────────────────────────────────────


def _make_manager() -> Any:
    """Build a minimal ``DeploymentManager`` with a local target."""
    from sentinel.action.deployment.manager import DeploymentManager

    config = DeploymentConfig(strategy="direct", target="local")
    return DeploymentManager(config)


class TestDetectStalled:
    def test_returns_old_running_deployments(self) -> None:
        dm = _make_manager()
        old_state = DeploymentState(
            deployment_id="old-1",
            model_name="fraud",
            from_version=None,
            to_version="1.0",
            strategy="canary",
            phase=DeploymentPhase.RUNNING,
            updated_at=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        dm._active["old-1"] = old_state
        stalled = dm.detect_stalled(max_age_seconds=3600)
        assert len(stalled) == 1
        assert stalled[0].deployment_id == "old-1"

    def test_ignores_recent_running(self) -> None:
        dm = _make_manager()
        recent = DeploymentState(
            deployment_id="new-1",
            model_name="fraud",
            from_version=None,
            to_version="2.0",
            strategy="canary",
            phase=DeploymentPhase.RUNNING,
            updated_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        )
        dm._active["new-1"] = recent
        stalled = dm.detect_stalled(max_age_seconds=3600)
        assert stalled == []

    def test_ignores_promoted_even_if_old(self) -> None:
        dm = _make_manager()
        promoted = DeploymentState(
            deployment_id="done-1",
            model_name="fraud",
            from_version=None,
            to_version="1.0",
            strategy="canary",
            phase=DeploymentPhase.PROMOTED,
            updated_at=datetime.now(timezone.utc) - timedelta(hours=5),
        )
        dm._active["done-1"] = promoted
        stalled = dm.detect_stalled(max_age_seconds=3600)
        assert stalled == []

    def test_mixed_states(self) -> None:
        dm = _make_manager()
        old_running = DeploymentState(
            deployment_id="stale-1",
            model_name="fraud",
            from_version=None,
            to_version="1.0",
            strategy="direct",
            phase=DeploymentPhase.RUNNING,
            updated_at=datetime.now(timezone.utc) - timedelta(hours=3),
        )
        recent_running = DeploymentState(
            deployment_id="fresh-1",
            model_name="fraud",
            from_version=None,
            to_version="2.0",
            strategy="direct",
            phase=DeploymentPhase.RUNNING,
            updated_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        )
        old_promoted = DeploymentState(
            deployment_id="prom-1",
            model_name="fraud",
            from_version=None,
            to_version="0.9",
            strategy="direct",
            phase=DeploymentPhase.PROMOTED,
            updated_at=datetime.now(timezone.utc) - timedelta(hours=10),
        )
        dm._active["stale-1"] = old_running
        dm._active["fresh-1"] = recent_running
        dm._active["prom-1"] = old_promoted
        stalled = dm.detect_stalled(max_age_seconds=3600)
        assert len(stalled) == 1
        assert stalled[0].deployment_id == "stale-1"

    def test_empty_active(self) -> None:
        dm = _make_manager()
        assert dm.detect_stalled() == []
