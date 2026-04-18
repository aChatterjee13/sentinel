"""Shared fixtures and synthetic datasets for the Sentinel test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sentinel.config.schema import (
    AlertsConfig,
    AuditConfig,
    DataDriftConfig,
    DriftConfig,
    ModelConfig,
    SentinelConfig,
)

# ── Synthetic data ────────────────────────────────────────────────


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def stable_features(rng: np.random.Generator) -> np.ndarray:
    """A 2-D feature matrix sampled from a stable distribution."""
    return rng.normal(loc=0.0, scale=1.0, size=(500, 4))


@pytest.fixture
def drifted_features(rng: np.random.Generator) -> np.ndarray:
    """A 2-D feature matrix shifted in mean and variance."""
    return rng.normal(loc=2.0, scale=2.0, size=(500, 4))


@pytest.fixture
def slight_drift(rng: np.random.Generator) -> np.ndarray:
    """A 2-D feature matrix with mild drift (PSI ~0.1-0.2)."""
    return rng.normal(loc=0.3, scale=1.1, size=(500, 4))


@pytest.fixture
def categorical_data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Categorical features as integer codes."""
    ref = rng.integers(0, 5, size=(300, 2))
    cur = rng.integers(0, 5, size=(300, 2))
    return ref.astype(float), cur.astype(float)


# ── Configs ───────────────────────────────────────────────────────


@pytest.fixture
def minimal_config() -> SentinelConfig:
    """A minimal valid SentinelConfig — useful for client construction."""
    return SentinelConfig(
        model=ModelConfig(name="test_model", domain="tabular"),
        drift=DriftConfig(
            data=DataDriftConfig(method="psi", threshold=0.2, window="7d"),
        ),
        alerts=AlertsConfig(),
        audit=AuditConfig(storage="local"),
    )


@pytest.fixture
def tmp_audit_dir(tmp_path: Path) -> Path:
    audit = tmp_path / "audit"
    audit.mkdir()
    return audit


@pytest.fixture
def example_yaml(tmp_path: Path) -> Path:
    """A complete YAML config written to a temp file for loader tests."""
    yaml_text = """
version: "1.0"
model:
  name: test_model_yaml
  type: classification
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
    path = tmp_path / "sentinel.yaml"
    path.write_text(yaml_text.strip())
    return path
