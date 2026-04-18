"""Tests for model artifact storage in local backend — WS-A."""

from __future__ import annotations

from pathlib import Path

import pytest

from sentinel.core.exceptions import ModelNotFoundError
from sentinel.foundation.registry.backends.local import LocalRegistryBackend
from sentinel.foundation.registry.model_registry import ModelRegistry


class TestLocalBackendArtifacts:
    def test_save_and_has_artifact(self, tmp_path: Path) -> None:
        backend = LocalRegistryBackend(root=tmp_path / "registry")

        # Create a fake artifact file
        artifact = tmp_path / "model.pkl"
        artifact.write_bytes(b"fake-model-bytes")

        # Save metadata first
        backend.save("fraud", "1.0.0", {"name": "fraud", "version": "1.0.0"})

        # Save artifact
        uri = backend.save_artifact("fraud", "1.0.0", artifact)
        assert "model.pkl" in uri
        assert backend.has_artifact("fraud", "1.0.0")

    def test_has_artifact_false_when_none(self, tmp_path: Path) -> None:
        backend = LocalRegistryBackend(root=tmp_path / "registry")
        backend.save("fraud", "1.0.0", {"name": "fraud", "version": "1.0.0"})
        assert not backend.has_artifact("fraud", "1.0.0")

    def test_has_artifact_false_when_missing(self, tmp_path: Path) -> None:
        backend = LocalRegistryBackend(root=tmp_path / "registry")
        assert not backend.has_artifact("fraud", "1.0.0")

    def test_load_artifact(self, tmp_path: Path) -> None:
        backend = LocalRegistryBackend(root=tmp_path / "registry")
        backend.save("fraud", "1.0.0", {"name": "fraud", "version": "1.0.0"})

        artifact = tmp_path / "model.pkl"
        artifact.write_bytes(b"fake-model-bytes")
        backend.save_artifact("fraud", "1.0.0", artifact)

        dest = tmp_path / "loaded"
        loaded = backend.load_artifact("fraud", "1.0.0", dest)
        assert loaded.exists()
        assert loaded.read_bytes() == b"fake-model-bytes"

    def test_load_artifact_not_found(self, tmp_path: Path) -> None:
        backend = LocalRegistryBackend(root=tmp_path / "registry")
        backend.save("fraud", "1.0.0", {"name": "fraud", "version": "1.0.0"})
        with pytest.raises(ModelNotFoundError, match="no artifact"):
            backend.load_artifact("fraud", "1.0.0", tmp_path / "dest")


class TestModelRegistryWithArtifact:
    def test_register_with_artifact_roundtrip(self, tmp_path: Path) -> None:
        backend = LocalRegistryBackend(root=tmp_path / "registry")
        registry = ModelRegistry(backend=backend)

        model_obj = {"weights": [1.0, 2.0, 3.0]}
        mv = registry.register_with_artifact(
            "fraud",
            "1.0.0",
            model=model_obj,
            serializer_name="pickle",
            framework="custom",
        )
        assert mv.version == "1.0.0"
        assert mv.artifact_uri is not None
        assert backend.has_artifact("fraud", "1.0.0")

    def test_register_with_artifact_auto_serializer(self, tmp_path: Path) -> None:
        backend = LocalRegistryBackend(root=tmp_path / "registry")
        registry = ModelRegistry(backend=backend)

        model_obj = {"simple": True}
        mv = registry.register_with_artifact(
            "fraud",
            "2.0.0",
            model=model_obj,
            serializer_name="auto",
        )
        assert mv.version == "2.0.0"
        assert backend.has_artifact("fraud", "2.0.0")
