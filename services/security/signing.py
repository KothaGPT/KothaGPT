"""WS-13 Artifact signing — hash + HMAC stub."""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path


def sign_artifact(path: str | Path, key: bytes = b"test-key") -> str:
    data = Path(path).read_bytes()
    sig = hmac.new(key, data, hashlib.sha256).hexdigest()
    sig_path = Path(str(path) + ".sig")
    sig_path.write_text(json.dumps({"sha256": hashlib.sha256(data).hexdigest(), "hmac": sig}) + "\n", encoding="utf-8")
    return sig

def verify_artifact(path: str | Path, key: bytes = b"test-key") -> bool:
    p = Path(path)
    sig_path = Path(str(p) + ".sig")
    if not sig_path.is_file():
        return False
    try:
        meta = json.loads(sig_path.read_text(encoding="utf-8"))
        data = p.read_bytes()
        expected = hmac.new(key, data, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, meta.get("hmac", ""))
    except Exception:  # noqa: BLE001
        return False
