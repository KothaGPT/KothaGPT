"""WS-7 Encryption — at-rest inventory + envelope helper."""

from __future__ import annotations

import base64
import hashlib
import os

# Inventory per docs/infra-plan.md: every storage class encrypted
INVENTORY = {
    "api_tls": {"at_rest": "TLS 1.3", "key": "KMS:api-tls"},
    "postgres": {"at_rest": "pgcrypto + volume SSE", "key": "KMS:pg"},
    "redis": {"at_rest": "volume SSE", "key": "KMS:redis"},
    "qdrant": {"at_rest": "volume SSE", "key": "KMS:qdrant"},
    "object_storage": {"at_rest": "SSE-S3", "key": "KMS:s3"},
    "artifacts": {"at_rest": "SSE", "key": "KMS:artifacts"},
}

def inventory() -> dict[str, dict[str, str]]:
    return INVENTORY

def is_encrypted(service: str) -> bool:
    return service in INVENTORY and "SSE" in INVENTORY[service]["at_rest"] or "TLS" in INVENTORY[service]["at_rest"]

def envelope_encrypt(plaintext: bytes, key: bytes | None = None) -> tuple[bytes, str]:
    # Stub: real would use KMS; here simple base64 + hash for inventory check
    k = key or os.urandom(16)
    # Fake encrypt: base64 + hmac
    ct = base64.b64encode(plaintext)
    digest = hashlib.sha256(ct + k).hexdigest()[:16]
    return ct, digest

def envelope_decrypt(ciphertext: bytes, key: bytes, digest: str) -> bytes:
    # Verify
    if hashlib.sha256(ciphertext + key).hexdigest()[:16] != digest:
        raise ValueError("decrypt failed: digest mismatch (key rotation?)")
    return base64.b64decode(ciphertext)
