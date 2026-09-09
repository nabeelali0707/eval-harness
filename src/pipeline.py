"""Config-driven pipeline orchestration."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(question: str, config: dict[str, Any]) -> dict[str, Any]:
    """Run one question through the configured pipeline."""
    raise NotImplementedError
