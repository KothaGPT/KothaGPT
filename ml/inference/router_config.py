from __future__ import annotations

from pathlib import Path

import yaml

__all__ = ["ROUTER_CONFIG_PATH", "load_router_config"]

ROUTER_CONFIG_PATH = "ml/inference/router_config.yaml"


def load_router_config(path: str | Path | None = None) -> dict:
    """Load the WS-5 routing configuration from ``router_config.yaml``.

    Returns a dict with ``tiers``, ``task_class_tiers``, ``escalation_chain``,
    ``tier_quality``, ``complexity`` and ``cost`` blocks. Missing file raises
    FileNotFoundError so callers can fall back to the in-code defaults.
    """
    config_path = Path(path) if path else Path(ROUTER_CONFIG_PATH)
    if not config_path.is_file():
        raise FileNotFoundError(f"router config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise TypeError(f"router config {config_path} must be a mapping")
    return data