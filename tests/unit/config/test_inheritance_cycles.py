"""Tests for circular ``extends:`` inheritance detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinel.config.loader import load_config
from sentinel.core.exceptions import (
    ConfigCircularInheritanceError,
    ConfigNotFoundError,
)


class TestInheritanceCycles:
    def test_self_extends_self(self, tmp_path: Path) -> None:
        a = tmp_path / "self.yaml"
        a.write_text("extends: self.yaml\nmodel:\n  name: x\n")
        with pytest.raises(ConfigCircularInheritanceError) as exc_info:
            load_config(a)
        msg = str(exc_info.value)
        assert "self.yaml" in msg
        assert "→" in msg

    def test_two_file_cycle(self, tmp_path: Path) -> None:
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        a.write_text("extends: b.yaml\nmodel:\n  name: x\n")
        b.write_text("extends: a.yaml\nmodel:\n  name: y\n")
        with pytest.raises(ConfigCircularInheritanceError) as exc_info:
            load_config(a)
        msg = str(exc_info.value)
        assert "a.yaml" in msg and "b.yaml" in msg
        # Cycle chain shows both files.
        assert msg.count("a.yaml") >= 2 or msg.count("b.yaml") >= 2

    def test_three_file_cycle(self, tmp_path: Path) -> None:
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        c = tmp_path / "c.yaml"
        a.write_text("extends: b.yaml\nmodel:\n  name: a\n")
        b.write_text("extends: c.yaml\nmodel:\n  name: b\n")
        c.write_text("extends: a.yaml\nmodel:\n  name: c\n")
        with pytest.raises(ConfigCircularInheritanceError) as exc_info:
            load_config(a)
        msg = str(exc_info.value)
        for name in ("a.yaml", "b.yaml", "c.yaml"):
            assert name in msg

    def test_missing_parent_raises_not_found(self, tmp_path: Path) -> None:
        a = tmp_path / "a.yaml"
        a.write_text("extends: nonexistent.yaml\nmodel:\n  name: x\n")
        with pytest.raises(ConfigNotFoundError):
            load_config(a)

    def test_grandparent_chain_loads(self, tmp_path: Path) -> None:
        gp = tmp_path / "grandparent.yaml"
        gp.write_text(
            """
version: "1.0"
model:
  name: gp_model
  domain: tabular
drift:
  data:
    method: psi
    threshold: 0.5
"""
        )
        parent = tmp_path / "parent.yaml"
        parent.write_text(
            """
extends: grandparent.yaml
drift:
  data:
    threshold: 0.3
"""
        )
        child = tmp_path / "child.yaml"
        child.write_text(
            """
extends: parent.yaml
model:
  name: child_model
"""
        )
        cfg = load_config(child)
        assert cfg.model.name == "child_model"
        assert cfg.model.domain == "tabular"  # from grandparent
        assert cfg.drift.data.method == "psi"  # from grandparent
        assert cfg.drift.data.threshold == 0.3  # parent override
