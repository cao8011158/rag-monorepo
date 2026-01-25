from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_settings(path: Path) -> Dict[str, Any]:
    """Load YAML config into a plain dict."""
    raw = Path(path).read_text(encoding="utf-8")
    cfg = yaml.safe_load(raw)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping/dict.")
    for k in ("input", "outputs", "stores"):
        if k not in cfg:
            raise ValueError(f"Missing required top-level key: {k}")
    return cfg
