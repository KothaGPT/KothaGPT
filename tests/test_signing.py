"""WS-13: Artifact signing — hash + HMAC sign/verify round trip."""

from __future__ import annotations

import json
from pathlib import Path

from services.security.signing import sign_artifact, verify_artifact


def test_sign_artifact_writes_sidecar_with_sha256_and_hmac(tmp_path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"binary-payload-1234")
    sig = sign_artifact(artifact)
    sig_path = Path(str(artifact) + ".sig")
    assert sig_path.is_file()
    meta = json.loads(sig_path.read_text(encoding="utf-8"))
    assert "sha256" in meta and "hmac" in meta
    assert meta["sha256"] == __import__("hashlib").sha256(b"binary-payload-1234").hexdigest()
    assert sig == meta["hmac"]


def test_verify_artifact_round_trip(tmp_path):
    artifact = tmp_path / "ckpt.pt"
    artifact.write_bytes(b"checkpoint-bytes")
    sign_artifact(artifact)
    assert verify_artifact(artifact) is True


def test_verify_artifact_fails_without_sidecar(tmp_path):
    artifact = tmp_path / "unsigned.bin"
    artifact.write_bytes(b"no-signature")
    assert verify_artifact(artifact) is False


def test_verify_artifact_fails_on_tampered_data(tmp_path):
    artifact = tmp_path / "tampered.bin"
    artifact.write_bytes(b"original")
    sign_artifact(artifact)
    artifact.write_bytes(b"tampered!")
    assert verify_artifact(artifact) is False


def test_verify_artifact_fails_on_custom_key(tmp_path):
    artifact = tmp_path / "custom.bin"
    artifact.write_bytes(b"custom-key-payload")
    sign_artifact(artifact, key=b"alpha-key")
    assert verify_artifact(artifact, key=b"beta-key") is False


def test_sign_and_verify_use_default_key(tmp_path):
    artifact = tmp_path / "default.bin"
    artifact.write_bytes(b"default-key-payload")
    sign_artifact(artifact)
    assert verify_artifact(artifact) is True
