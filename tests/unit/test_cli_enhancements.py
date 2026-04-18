"""Tests for CLI --dry-run deploy and shell completion command."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from sentinel.cli.main import cli


def _write_minimal_config(path: Path) -> None:
    path.write_text(
        """\
version: "1.0"
model:
  name: cli_test_model
  domain: tabular
drift:
  data:
    method: psi
    threshold: 0.2
    window: 7d
alerts:
  channels: []
audit:
  storage: local
  path: ./audit/
"""
    )


class TestDeployDryRun:
    def _extract_json(self, output: str) -> dict:
        """Extract the JSON object from CLI output that may contain log lines."""
        # Find the first '{' and parse from there
        start = output.index("{")
        return json.loads(output[start:])

    def test_dry_run_outputs_json(self, tmp_path: Path) -> None:
        cfg = tmp_path / "sentinel.yaml"
        _write_minimal_config(cfg)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["deploy", "--config", str(cfg), "--version", "2.0", "--dry-run"],
        )
        assert result.exit_code == 0
        body = self._extract_json(result.output)
        assert body["dry_run"] is True
        assert body["version"] == "2.0"
        assert body["validation"] == "passed"

    def test_dry_run_with_strategy_and_traffic(self, tmp_path: Path) -> None:
        cfg = tmp_path / "sentinel.yaml"
        _write_minimal_config(cfg)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "deploy",
                "--config", str(cfg),
                "--version", "3.0",
                "--strategy", "direct",
                "--traffic", "10",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0
        body = self._extract_json(result.output)
        assert body["strategy"] == "direct"
        assert body["traffic_pct"] == 10

    def test_dry_run_does_not_deploy(self, tmp_path: Path) -> None:
        cfg = tmp_path / "sentinel.yaml"
        _write_minimal_config(cfg)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["deploy", "--config", str(cfg), "--version", "2.0", "--dry-run"],
        )
        assert result.exit_code == 0
        body = self._extract_json(result.output)
        assert "dry_run" in body


class TestShellCompletion:
    def test_bash_completion(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["completion", "bash"])
        assert result.exit_code == 0
        assert "_SENTINEL_COMPLETE" in result.output
        assert "bash_source" in result.output

    def test_zsh_completion(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["completion", "zsh"])
        assert result.exit_code == 0
        assert "_SENTINEL_COMPLETE" in result.output
        assert "zsh_source" in result.output

    def test_fish_completion(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["completion", "fish"])
        assert result.exit_code == 0
        assert "_SENTINEL_COMPLETE" in result.output
        assert "fish_source" in result.output

    def test_invalid_shell_rejected(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["completion", "powershell"])
        assert result.exit_code != 0
