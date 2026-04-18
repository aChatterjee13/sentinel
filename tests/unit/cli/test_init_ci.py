"""Tests for `sentinel init --ci azure-devops` — WS-F."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from sentinel.cli.main import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestInitWithCI:
    def test_generates_pipeline_file(self, runner: CliRunner, tmp_path: Path) -> None:
        out = str(tmp_path / "sentinel.yaml")
        result = runner.invoke(
            cli,
            ["init", "--name", "fraud_v2", "--out", out, "--ci", "azure-devops", "--force"],
        )
        assert result.exit_code == 0, result.output
        pipeline = tmp_path / "azure-pipelines.yml"
        assert pipeline.exists()

    def test_pipeline_is_valid_yaml(self, runner: CliRunner, tmp_path: Path) -> None:
        out = str(tmp_path / "sentinel.yaml")
        result = runner.invoke(
            cli,
            ["init", "--name", "test_model", "--out", out, "--ci", "azure-devops", "--force"],
        )
        assert result.exit_code == 0
        pipeline = tmp_path / "azure-pipelines.yml"
        data = yaml.safe_load(pipeline.read_text())
        assert isinstance(data, dict)
        assert "trigger" in data
        assert "stages" in data

    def test_model_name_in_pipeline(self, runner: CliRunner, tmp_path: Path) -> None:
        out = str(tmp_path / "sentinel.yaml")
        result = runner.invoke(
            cli,
            ["init", "--name", "my_fraud_model", "--out", out, "--ci", "azure-devops", "--force"],
        )
        assert result.exit_code == 0
        pipeline = tmp_path / "azure-pipelines.yml"
        content = pipeline.read_text()
        assert "my_fraud_model" in content

    def test_pipeline_contains_sentinel_commands(self, runner: CliRunner, tmp_path: Path) -> None:
        out = str(tmp_path / "sentinel.yaml")
        result = runner.invoke(
            cli,
            ["init", "--name", "test", "--out", out, "--ci", "azure-devops", "--force"],
        )
        assert result.exit_code == 0
        content = (tmp_path / "azure-pipelines.yml").read_text()
        assert "sentinel check" in content
        assert "sentinel config validate" in content
        assert "sentinel deploy" in content

    def test_no_pipeline_without_ci(self, runner: CliRunner, tmp_path: Path) -> None:
        out = str(tmp_path / "sentinel.yaml")
        result = runner.invoke(
            cli,
            ["init", "--name", "test", "--out", out, "--force"],
        )
        assert result.exit_code == 0
        pipeline = tmp_path / "azure-pipelines.yml"
        assert not pipeline.exists()

    def test_sentinel_yaml_still_created(self, runner: CliRunner, tmp_path: Path) -> None:
        out = str(tmp_path / "sentinel.yaml")
        result = runner.invoke(
            cli,
            ["init", "--name", "test", "--out", out, "--ci", "azure-devops", "--force"],
        )
        assert result.exit_code == 0
        assert (tmp_path / "sentinel.yaml").exists()

    def test_pipeline_has_stages(self, runner: CliRunner, tmp_path: Path) -> None:
        out = str(tmp_path / "sentinel.yaml")
        runner.invoke(
            cli,
            ["init", "--name", "test", "--out", out, "--ci", "azure-devops", "--force"],
        )
        data = yaml.safe_load((tmp_path / "azure-pipelines.yml").read_text())
        stage_names = [s["stage"] for s in data["stages"]]
        assert "Validate" in stage_names
        assert "DriftCheck" in stage_names
        assert "Deploy" in stage_names
