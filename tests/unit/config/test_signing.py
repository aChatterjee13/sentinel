"""Tests for the detached HMAC config signing module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sentinel.config.loader import ConfigLoader
from sentinel.config.signing import (
    SIGNATURE_SUFFIX,
    ConfigSignature,
    canonicalise_config,
    read_signature_file,
    sign_config,
    verify_config,
    write_signature_file,
)
from sentinel.core.exceptions import ConfigSignatureError
from sentinel.foundation.audit.keystore import EnvKeystore

_KEY_A = b"this-is-a-strong-32-byte-key!!!!"
_KEY_B = b"a-different-32-byte-signing-key!"


# ── canonicalise_config ─────────────────────────────────────────────


class TestCanonicaliseConfig:
    def test_key_order_is_irrelevant(self) -> None:
        a = {"alpha": 1, "beta": 2, "gamma": {"x": 1, "y": 2}}
        b = {"gamma": {"y": 2, "x": 1}, "beta": 2, "alpha": 1}
        assert canonicalise_config(a) == canonicalise_config(b)

    def test_volatile_source_keys_are_stripped(self) -> None:
        with_meta = {
            "model": {"name": "demo", "__source__": "/some/path.yaml"},
            "__sources__": {"path": "x"},
        }
        without_meta = {"model": {"name": "demo"}}
        assert canonicalise_config(with_meta) == canonicalise_config(without_meta)

    def test_handles_nested_lists(self) -> None:
        data = {"items": [{"b": 2, "a": 1}, {"d": 4, "c": 3}]}
        canonical = canonicalise_config(data)
        # Lists keep order; only dict keys are sorted.
        assert b'"items":[{"a":1,"b":2},{"c":3,"d":4}]' in canonical

    def test_path_objects_are_serialised(self) -> None:
        data = {"path": Path("/tmp/foo")}
        canonical = canonicalise_config(data)
        assert b'"/tmp/foo"' in canonical

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError):
            canonicalise_config({"bad": object()})


# ── sign / verify ───────────────────────────────────────────────────


class TestSignAndVerify:
    def test_round_trip(self) -> None:
        data = {"version": "1.0", "model": {"name": "demo"}}
        sig = sign_config(data, _KEY_A)
        assert verify_config(data, sig, _KEY_A)

    def test_tampered_payload_fails(self) -> None:
        data = {"version": "1.0", "model": {"name": "demo"}}
        sig = sign_config(data, _KEY_A)
        tampered = {"version": "1.0", "model": {"name": "evil"}}
        assert not verify_config(tampered, sig, _KEY_A)

    def test_wrong_key_fails(self) -> None:
        data = {"version": "1.0", "model": {"name": "demo"}}
        sig = sign_config(data, _KEY_A)
        assert not verify_config(data, sig, _KEY_B)

    def test_signature_metadata_populated(self) -> None:
        data = {"version": "1.0"}
        sig = sign_config(data, _KEY_A)
        assert sig.algorithm == "hmac-sha256"
        assert len(sig.signature) == 64  # 32-byte HMAC as hex
        assert len(sig.digest) == 64  # SHA-256 as hex
        assert len(sig.key_fingerprint) == 8
        assert sig.signed_at  # ISO timestamp

    def test_unknown_algorithm_fails_closed(self) -> None:
        data = {"version": "1.0"}
        sig = sign_config(data, _KEY_A)
        forged = ConfigSignature(**{**sig.model_dump(), "algorithm": "md5"})
        assert not verify_config(data, forged, _KEY_A)

    def test_signature_is_independent_of_volatile_metadata(self) -> None:
        with_meta = {"model": {"name": "demo", "__source__": "/etc/x.yaml"}}
        without_meta = {"model": {"name": "demo"}}
        sig = sign_config(with_meta, _KEY_A)
        # Verifier sees the meta-stripped version — should still pass.
        assert verify_config(without_meta, sig, _KEY_A)


# ── sidecar file IO ─────────────────────────────────────────────────


class TestSidecarFiles:
    def test_write_creates_sidecar_with_suffix(self, tmp_path: Path) -> None:
        config_path = tmp_path / "sentinel.yaml"
        config_path.write_text("version: '1.0'\n")
        sig = sign_config({"version": "1.0"}, _KEY_A)
        sig_path = write_signature_file(config_path, sig)
        assert sig_path == config_path.with_name("sentinel.yaml" + SIGNATURE_SUFFIX)
        assert sig_path.exists()

    def test_write_accepts_explicit_sig_path(self, tmp_path: Path) -> None:
        sig = sign_config({"version": "1.0"}, _KEY_A)
        out = tmp_path / "custom.sig"
        sig_path = write_signature_file(out, sig)
        assert sig_path == out
        assert out.exists()

    def test_round_trip_via_disk(self, tmp_path: Path) -> None:
        config_path = tmp_path / "sentinel.yaml"
        config_path.write_text("version: '1.0'\n")
        sig = sign_config({"version": "1.0"}, _KEY_A)
        write_signature_file(config_path, sig)

        loaded = read_signature_file(config_path)
        assert loaded.signature == sig.signature
        assert loaded.digest == sig.digest
        assert loaded.key_fingerprint == sig.key_fingerprint

    def test_read_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigSignatureError, match="not found"):
            read_signature_file(tmp_path / "nope.yaml")

    def test_read_invalid_json_raises(self, tmp_path: Path) -> None:
        sig_path = tmp_path / ("sentinel.yaml" + SIGNATURE_SUFFIX)
        sig_path.write_text("not-json")
        with pytest.raises(ConfigSignatureError, match="not valid JSON"):
            read_signature_file(sig_path)

    def test_read_malformed_signature_raises(self, tmp_path: Path) -> None:
        sig_path = tmp_path / ("sentinel.yaml" + SIGNATURE_SUFFIX)
        sig_path.write_text(json.dumps({"hello": "world"}))
        with pytest.raises(ConfigSignatureError, match="malformed"):
            read_signature_file(sig_path)


# ── ConfigLoader.verify_signature integration ───────────────────────


def _write_minimal_config(path: Path) -> None:
    path.write_text("version: '1.0'\nmodel:\n  name: signed_demo\n  domain: tabular\n")


class TestLoaderSignatureVerification:
    def test_verify_signature_requires_keystore(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "sentinel.yaml"
        _write_minimal_config(cfg_path)
        with pytest.raises(ConfigSignatureError, match="requires"):
            ConfigLoader(cfg_path, verify_signature=True)

    def test_loader_verifies_a_valid_signature(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg_path = tmp_path / "sentinel.yaml"
        _write_minimal_config(cfg_path)
        # Sign once with a non-verifying loader, then re-load with
        # verification turned on.
        signing_loader = ConfigLoader(cfg_path)
        signing_loader.load()
        sig = sign_config(signing_loader.resolved_payload or {}, _KEY_A)
        write_signature_file(cfg_path, sig)

        monkeypatch.setenv("SENTINEL_CONFIG_KEY", _KEY_A.decode())
        verify_loader = ConfigLoader(
            cfg_path,
            verify_signature=True,
            signature_keystore=EnvKeystore("SENTINEL_CONFIG_KEY"),
        )
        cfg = verify_loader.load()
        assert cfg.model.name == "signed_demo"
        assert verify_loader.signature_verified is True

    def test_loader_rejects_tampered_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg_path = tmp_path / "sentinel.yaml"
        _write_minimal_config(cfg_path)
        signing_loader = ConfigLoader(cfg_path)
        signing_loader.load()
        sig = sign_config(signing_loader.resolved_payload or {}, _KEY_A)
        write_signature_file(cfg_path, sig)

        # Edit the config to flip a value after signing.
        cfg_path.write_text(cfg_path.read_text().replace("signed_demo", "tampered"))

        monkeypatch.setenv("SENTINEL_CONFIG_KEY", _KEY_A.decode())
        verify_loader = ConfigLoader(
            cfg_path,
            verify_signature=True,
            signature_keystore=EnvKeystore("SENTINEL_CONFIG_KEY"),
        )
        with pytest.raises(ConfigSignatureError, match="does not match"):
            verify_loader.load()

    def test_loader_rejects_wrong_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg_path = tmp_path / "sentinel.yaml"
        _write_minimal_config(cfg_path)
        signing_loader = ConfigLoader(cfg_path)
        signing_loader.load()
        sig = sign_config(signing_loader.resolved_payload or {}, _KEY_A)
        write_signature_file(cfg_path, sig)

        monkeypatch.setenv("SENTINEL_CONFIG_KEY", _KEY_B.decode())
        verify_loader = ConfigLoader(
            cfg_path,
            verify_signature=True,
            signature_keystore=EnvKeystore("SENTINEL_CONFIG_KEY"),
        )
        with pytest.raises(ConfigSignatureError):
            verify_loader.load()

    def test_loader_rejects_missing_signature_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg_path = tmp_path / "sentinel.yaml"
        _write_minimal_config(cfg_path)
        monkeypatch.setenv("SENTINEL_CONFIG_KEY", _KEY_A.decode())
        verify_loader = ConfigLoader(
            cfg_path,
            verify_signature=True,
            signature_keystore=EnvKeystore("SENTINEL_CONFIG_KEY"),
        )
        with pytest.raises(ConfigSignatureError, match="not found"):
            verify_loader.load()
