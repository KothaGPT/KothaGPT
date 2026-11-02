"""Fetch legally usable Bangla corpora into data/raw/.

Usage:
    python data/scripts/fetch_corpora.py [--manifest corpora.json] [--yes]

The manifest is a JSON list of entries:
    {
      "name": "bn_wikipedia",
      "url": "https://example.com/file.jsonl.gz",
      "license": "CC-BY-SA-4.0",
      "license_url": "https://...",
      "description": "Bengali Wikipedia dump"
    }

Only download data whose license you have verified. See data/README.md for a
list of recommended open Bangla datasets. No dataset is downloaded unless it is
listed in the manifest.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


def _load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("datasets", [])
    if not isinstance(data, list):
        raise TypeError("manifest must be a JSON list (or {datasets: [...]})")
    required = {"name", "url", "license"}
    for entry in data:
        missing = required - set(entry)
        if missing:
            raise ValueError(f"manifest entry missing fields: {sorted(missing)}")
    return data


def _confirm() -> bool:
    answer = input("Download these datasets? [y/N] ").strip().lower()
    return answer in ("y", "yes")


def fetch_manifest(manifest_path: Path, raw_dir: Path, assume_yes: bool = False) -> list[dict]:
    entries = _load_manifest(manifest_path)
    if not entries:
        print("Manifest is empty; nothing to fetch.")
        return []
    print(f"Found {len(entries)} dataset(s) to fetch.")
    if not assume_yes and not _confirm():
        print("Aborted.")
        sys.exit(1)

    results: list[dict] = []
    raw_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        name = entry["name"]
        url = entry["url"]
        print(f"[{name}] downloading {url} ...")
        target = raw_dir / name
        target.mkdir(parents=True, exist_ok=True)
        out_file = target / url.split("/")[-1] or "data.bin"
        urllib.request.urlretrieve(url, out_file)
        print(f"[{name}] saved to {out_file}")
        results.append(
            {
                "name": name,
                "url": url,
                "license": entry["license"],
                "license_url": entry.get("license_url"),
                "path": str(out_file),
            }
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).parent / "corpora.manifest.json"),
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--yes", action="store_true", help="skip confirmation")
    args = parser.parse_args(argv)

    results = fetch_manifest(Path(args.manifest), Path(args.raw_dir), assume_yes=args.yes)
    if results:
        print(f"\nFetched {len(results)} dataset(s).")
        for r in results:
            print(f"  {r['name']}: {r['path']} ({r['license']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
