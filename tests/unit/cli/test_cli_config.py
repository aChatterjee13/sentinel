"""Tests for the ``sentinel config`` CLI subcommands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from sentinel.cli.main import cli


def _write_minimal(path: Path, *, with_secret: bool = True) -> None:
    secret_line = "      webhook_url: ${SENTINEL_TEST_HOOK}\n" if with_secret else ""
    path.write_text(
        f"""version: "1.0"
model:
  name: cli_test_model
  domain: tabular
drift:
  data:
    method: psi
    threshold: 0.2
    window: 7d
alerts:
  channels:
    - type: slack
{secret_line}      channel: "#test"
audit:
  storage: local
  path: ./audit/
"""
    )


class TestConfigValidate:
    def test_validate_lenient_passes_with_unset_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SENTINEL_TEST_HOOK", raising=False)
        path = tmp_path / "sentinel.yaml"
        _write_minimal(path)
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "validate", "--config", str(path)])
        assert result.exit_code == 0
        assert "OK" in result.output

    def test_validate_strict_fails_with_unset_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SENTINEL_TEST_HOOK", raising=False)
        path = tmp_path / "sentinel.yaml"
        _write_minimal(path)
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "validate", "--config", str(path), "--strict"])
        assert result.exit_code != 0
        assert "SENTINEL_TEST_HOOK" in result.output

    def test_validate_strict_passes_with_var_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SENTINEL_TEST_HOOK", "https://hooks.slack.com/X/Y/Z")
        path = tmp_path / "sentinel.yaml"
        _write_minimal(path)
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "validate", "--config", str(path), "--strict"])
        assert result.exit_code == 0

    def test_top_level_validate_alias(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SENTINEL_TEST_HOOK", "https://x")
        path = tmp_path / "sentinel.yaml"
        _write_minimal(path)
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--config", str(path)])
        assert result.exit_code == 0
        assert "OK" in result.output

    def test_validate_strict_fails_on_missing_file_reference(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SENTINEL_TEST_HOOK", "https://x")
        path = tmp_path / "sentinel.yaml"
        path.write_text(
            f"""version: "1.0"
model:
  name: cli_ref_test
  domain: tabular
  baseline_dataset: missing.parquet
audit:
  storage: local
  path: {tmp_path}
"""
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "validate", "--config", str(path), "--strict"])
        assert result.exit_code != 0
        assert "missing.parquet" in result.output


class TestConfigShow:
    def test_show_masks_secrets_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SENTINEL_TEST_HOOK", "https://hooks.slack.com/X/Y/Z")
        path = tmp_path / "sentinel.yaml"
        _write_minimal(path)
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "show", "--config", str(path)])
        assert result.exit_code == 0
        assert "<REDACTED>" in result.output
        assert "hooks.slack.com" not in result.output

    def test_show_unmask_reveals_secrets(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SENTINEL_TEST_HOOK", "https://hooks.slack.com/X/Y/Z")
        path = tmp_path / "sentinel.yaml"
        _write_minimal(path)
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "show", "--config", str(path), "--unmask"])
        assert result.exit_code == 0
        assert "hooks.slack.com" in result.output
        # The warning is emitted to stderr — combined output via CliRunner
        # mixes stdout and stderr by default.
        assert "warning" in result.output.lower()

    def test_show_yaml_format(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTINEL_TEST_HOOK", "https://x")
        path = tmp_path / "sentinel.yaml"
        _write_minimal(path)
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "show", "--config", str(path), "--format", "yaml"])
        assert result.exit_code == 0
        # Strip the structlog log line, then YAML-parse the rest.
        body_lines = [line for line in result.output.splitlines() if not line.startswith("2026")]
        body = "\n".join(body_lines)
        parsed = yaml.safe_load(body)
        assert parsed["model"]["name"] == "cli_test_model"

    def test_show_json_format(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SENTINEL_TEST_HOOK", "https://x")
        path = tmp_path / "sentinel.yaml"
        _write_minimal(path)
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "show", "--config", str(path), "--format", "json"])
        assert result.exit_code == 0
        # Find the first '{' to skip log noise.
        idx = result.output.find("{")
        assert idx >= 0
        parsed = json.loads(result.output[idx:])
        assert parsed["model"]["name"] == "cli_test_model"
        assert parsed["alerts"]["channels"][0]["webhook_url"] == "<REDACTED>"

    def test_show_invalid_config_exits_nonzero(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.yaml"
        path.write_text(
            """version: "1.0"
model:
  name: x
  type: not_a_type
"""
        )
        runner = CliRunner()
        result = runner.invoke(cli, ["config", "show", "--config", str(path)])
        assert result.exit_code != 0
