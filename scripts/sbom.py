"""WS-12 SBOM generator + scan stub."""

from __future__ import annotations

import json
import pathlib

LOCKS = {
    "python": "uv.lock",
    "node": "pnpm-lock.yaml",
    "rust": "Cargo.lock",
    "go": "go.mod",
}

def generate_sbom() -> dict:
    sbom = {}
    for lang, lock in LOCKS.items():
        p = pathlib.Path(lock)
        if p.exists():
            sbom[lang] = {"lock": lock, "exists": True, "sha256": __import__("hashlib").sha256(p.read_bytes()).hexdigest()[:12]}
        else:
            # also check packages/* for go/rust
            found = list(pathlib.Path("packages").rglob(lock)) if pathlib.Path("packages").exists() else []
            sbom[lang] = {"lock": lock, "exists": bool(found), "count": len(found)}
    return sbom

def check() -> bool:
    sbom = generate_sbom()
    # CI gate: all lockfiles must exist
    for lang in ["python", "node", "rust", "go"]:
        if not sbom.get(lang, {}).get("exists"):
            print(f"SBOM missing {lang} {LOCKS[lang]}")
            return False
    print("SBOM OK", json.dumps(sbom, indent=2))
    return True

if __name__ == "__main__":
    import sys
    sys.exit(0 if check() else 1)
