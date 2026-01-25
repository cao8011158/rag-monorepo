from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from qr_pipeline.stores.base import Store
from qr_pipeline.stores.filesystem import FilesystemStore


def build_store_registry(cfg: Dict[str, Any]) -> Dict[str, Store]:
    """Build store instances from cfg['stores'].

    Example:
      stores:
        fs_local:
          kind: filesystem
          root: data
    """
    stores_cfg = cfg.get("stores")
    if not isinstance(stores_cfg, dict):
        raise ValueError("cfg['stores'] must be a dict")
    out: Dict[str, Store] = {}
    for name, sc in stores_cfg.items():
        if not isinstance(sc, dict):
            raise ValueError(f"stores.{name} must be a dict")
        kind = sc.get("kind")
        if kind == "filesystem":
            root = sc.get("root", "data")
            out[name] = FilesystemStore(Path(root))
        else:
            raise ValueError(f"Unsupported store kind: {kind!r} (store={name})")
    return out
