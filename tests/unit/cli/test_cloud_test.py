"""Tests for the ``sentinel cloud test`` CLI subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from sentinel.cli.main import cli


def _write_minimal(path: Path) -> None:
    """Minimal config — local registry, local audit, local deploy target."""
    path.write_text(
        """version: "1.0"
model:
  name: cloud_test_model
  domain: tabular
drift:
  data:
    method: psi
    threshold: 0.2
    window: 7d
audit:
  storage: local
  path: ./audit/
registry:
  backend: local
  path: ./registry/
deployment:
  strategy: canary
  target: local
"""
    )


class TestCloudTestCommand:
    def test_all_backends_pass_on_minimal_config(self, tmp_path: Path) -> None:
        config_path = tmp_path / "sentinel.yaml"
        _write_minimal(config_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["cloud", "test", "--config", str(config_path)])
        assert result.exit_code == 0, result.output
        # Every backend should print an OK line
        assert "[OK] keyvault" in result.output
        assert "[OK] registry" in result.output
        assert "[OK] audit" in result.output
        assert "[OK] deploy" in result.output
        assert "All 4 backend(s) OK." in result.output

    def test_only_keyvault(self, tmp_path: Path) -> None:
        config_path = tmp_path / "sentinel.yaml"
        _write_minimal(config_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["cloud", "test", "--config", str(config_path), "--only", "keyvault"]
        )
        assert result.exit_code == 0
        assert "[OK] keyvault" in result.output
        assert "registry" not in result.output
        assert "audit" not in result.output
        assert "deploy" not in result.output

    def test_only_audit(self, tmp_path: Path) -> None:
        config_path = tmp_path / "sentinel.yaml"
        _write_minimal(config_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["cloud", "test", "--config", str(config_path), "--only", "audit"]
        )
        assert result.exit_code == 0
        assert "[OK] audit" in result.output
        # Key Vault is always skipped when --only is something else
        assert "keyvault" not in result.output

    def test_only_deploy(self, tmp_path: Path) -> None:
        config_path = tmp_path / "sentinel.yaml"
        _write_minimal(config_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["cloud", "test", "--config", str(config_path), "--only", "deploy"]
        )
        assert result.exit_code == 0
        assert "[OK] deploy" in result.output

    def test_invalid_backend_name_rejected(self, tmp_path: Path) -> None:
        config_path = tmp_path / "sentinel.yaml"
        _write_minimal(config_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["cloud", "test", "--config", str(config_path), "--only", "s3"])
        # click.Choice rejects unknown values with exit code 2
        assert result.exit_code != 0
        assert "Invalid value" in result.output or "invalid choice" in result.output.lower()

    def test_missing_config_file_exits_nonzero(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["cloud", "test", "--config", str(tmp_path / "does-not-exist.yaml")],
        )
        assert result.exit_code != 0
        assert "FAIL" in result.output or "keyvault" in result.output

    def test_registry_failure_reports_fail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the registry backend raises, the command exits 1 and the other checks still run."""
        config_path = tmp_path / "sentinel.yaml"
        _write_minimal(config_path)

        from sentinel.core.client import SentinelClient

        def _boom(_config: Any) -> Any:
            raise RuntimeError("simulated registry failure")

        monkeypatch.setattr(SentinelClient, "_build_registry_backend", staticmethod(_boom))
        runner = CliRunner()
        result = runner.invoke(cli, ["cloud", "test", "--config", str(config_path)])
        assert result.exit_code == 1
        assert "[FAIL] registry" in result.output
        assert "simulated registry failure" in result.output
        # Audit and deploy probes should still have run
        assert "[OK] audit" in result.output
        assert "[OK] deploy" in result.output
        assert "1 backend(s) failed" in result.output

    def test_deploy_failure_reports_fail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deploy probe failure surfaces in the summary but other checks still run."""
        config_path = tmp_path / "sentinel.yaml"
        _write_minimal(config_path)

        from sentinel.action.deployment.manager import DeploymentManager

        def _boom(_config: Any) -> Any:
            raise RuntimeError("simulated deploy failure")

        monkeypatch.setattr(DeploymentManager, "_build_target", staticmethod(_boom))
        runner = CliRunner()
        result = runner.invoke(cli, ["cloud", "test", "--config", str(config_path)])
        assert result.exit_code == 1
        assert "[FAIL] deploy" in result.output
        assert "[OK] registry" in result.output
        assert "[OK] audit" in result.output

    def test_elapsed_ms_printed(self, tmp_path: Path) -> None:
        config_path = tmp_path / "sentinel.yaml"
        _write_minimal(config_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["cloud", "test", "--config", str(config_path)])
        assert result.exit_code == 0
        # Each OK line ends with "(…ms)"
        assert "ms)" in result.output
