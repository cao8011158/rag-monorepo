from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from reranker_training.stores.base import Store
from reranker_training.filesystem import FilesystemStore
from reranker_training.object_storage import S3Store


def build_store_registry(cfg: Dict[str, Any]) -> Dict[str, Store]:
    stores_cfg = cfg.get("stores", {})
    if not isinstance(stores_cfg, dict):
        raise ValueError("cfg.stores must be a mapping")

    reg: Dict[str, Store] = {}
    for name, sc in stores_cfg.items():
        if not isinstance(sc, dict):
            raise ValueError(f"Store '{name}' must be a mapping")

        kind = sc.get("kind")
        if kind == "filesystem":
            root = sc.get("root", ".")
            reg[name] = FilesystemStore(root=Path(root).resolve())
        elif kind == "object_storage":
            driver = sc.get("driver")
            if driver != "s3":
                raise ValueError(f"Unsupported object_storage driver: {driver}")
            bucket = sc.get("bucket")
            prefix = sc.get("prefix", "")
            if not bucket:
                raise ValueError("S3 store requires 'bucket'")
            reg[name] = S3Store(bucket=bucket, prefix=prefix)
        else:
            raise ValueError(f"Unsupported store kind: {kind}")

    return reg
