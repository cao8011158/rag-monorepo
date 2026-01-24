# src/ce_pipeline/settings.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Set

import hashlib
import json

import yaml


SettingsDict = Dict[str, Any]


# =========================
# Public API
# =========================

def load_settings(path: str | Path) -> SettingsDict:
    """
    Load pipeline.yaml -> nested dict settings (ALL DYNAMIC).

    Guarantees:
    - defaults are applied (so required nested maps exist)
    - validation is executed (ValueError with clear messages)
    - runtime metadata is attached into settings["_meta"]

    Returns:
        settings: Dict[str, Any]
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError("pipeline config root must be a mapping (YAML dict)")

    settings = apply_defaults(raw)
    validate_settings(settings)

    settings.setdefault("_meta", {})
    settings["_meta"]["config_path"] = str(path)
    settings["_meta"]["config_hash"] = hash_settings(settings, exclude_keys={"_meta"})
    return settings


def apply_defaults(raw: SettingsDict) -> SettingsDict:
    """
    Apply defaults to the raw YAML dict, producing a normalized settings dict
    that always contains required nested objects.
    """
    s: SettingsDict = _deep_copy_dict(raw)

    # ---- input ----
    s.setdefault("input", {})
    _must_be_mapping(s["input"], "input")
    s["input"].setdefault("input_store", "")
    s["input"].setdefault("input_path", "")

    # ---- outputs ----
    s.setdefault("outputs", {})
    _must_be_mapping(s["outputs"], "outputs")
    # outputs entries are not defaulted here; validate will check required keys

    # ---- stores ----
    s.setdefault("stores", {})
    _must_be_mapping(s["stores"], "stores")

    # ---- chunking (TOP-LEVEL) ----
    s.setdefault("chunking", {})
    _must_be_mapping(s["chunking"], "chunking")
    s["chunking"].setdefault("window_chars", 1200)
    s["chunking"].setdefault("overlap_chars", 200)
    s["chunking"].setdefault("min_chunk_chars", 200)

    # ---- processing (dedup only) ----
    s.setdefault("processing", {})
    _must_be_mapping(s["processing"], "processing")

    s["processing"].setdefault("dedup", {})
    _must_be_mapping(s["processing"]["dedup"], "processing.dedup")

    s["processing"]["dedup"].setdefault("exact_dedup", {})
    _must_be_mapping(s["processing"]["dedup"]["exact_dedup"], "processing.dedup.exact_dedup")
    s["processing"]["dedup"]["exact_dedup"].setdefault("hash_field", "chunk_text_hash")

    s["processing"]["dedup"].setdefault("semantic_dedup", {})
    _must_be_mapping(s["processing"]["dedup"]["semantic_dedup"], "processing.dedup.semantic_dedup")
    sd = s["processing"]["dedup"]["semantic_dedup"]
    sd.setdefault("enable", False)
    sd.setdefault("threshold", 0.95)
    sd.setdefault("topk", 20)
    sd.setdefault("hnsw_m", 32)
    sd.setdefault("ef_construction", 200)
    sd.setdefault("ef_search", 128)
    sd.setdefault("normalize", True)

    # ---- embedding ----
    s.setdefault("embedding", {})
    _must_be_mapping(s["embedding"], "embedding")

    s["embedding"].setdefault("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    s["embedding"].setdefault("batch_size", 64)
    s["embedding"].setdefault("normalize_embeddings", True)
    s["embedding"].setdefault("device", None)

    s["embedding"].setdefault("instructions", {})
    _must_be_mapping(s["embedding"]["instructions"], "embedding.instructions")
    s["embedding"]["instructions"].setdefault("passage", "passage: ")
    s["embedding"]["instructions"].setdefault("query", "query: ")

    # ---- indexing ----
    s.setdefault("indexing", {})
    _must_be_mapping(s["indexing"], "indexing")

    s["indexing"].setdefault("bm25", {})
    _must_be_mapping(s["indexing"]["bm25"], "indexing.bm25")
    s["indexing"]["bm25"].setdefault("enabled", True)

    s["indexing"].setdefault("vector", {})
    _must_be_mapping(s["indexing"]["vector"], "indexing.vector")
    s["indexing"]["vector"].setdefault("enabled", True)
    s["indexing"]["vector"].setdefault("faiss_index", "FlatIP")

    return s


def validate_settings(s: SettingsDict) -> None:
    """
    Validate normalized settings dict (after apply_defaults).
    Raises ValueError with explicit messages (same spirit as your dataclass validate()).
    """
    # ---- input ----
    if not s["input"]["input_store"]:
        raise ValueError("input.input_store is required")
    if not s["input"]["input_path"]:
        raise ValueError("input.input_path is required")

    # ---- chunking (top-level) ----
    c = s["chunking"]
    window = _as_int(c["window_chars"], "chunking.window_chars")
    overlap = _as_int(c["overlap_chars"], "chunking.overlap_chars")
    min_chars = _as_int(c["min_chunk_chars"], "chunking.min_chunk_chars")

    if window <= 0:
        raise ValueError("chunking.window_chars must be > 0")
    if overlap < 0:
        raise ValueError("chunking.overlap_chars must be >= 0")
    if overlap >= window:
        raise ValueError("chunking.overlap_chars must be < window_chars")
    if min_chars <= 0:
        raise ValueError("chunking.min_chunk_chars must be > 0")

    # ---- dedup ----
    d = s["processing"]["dedup"]
    hash_field = str(d["exact_dedup"].get("hash_field", "") or "")
    if not hash_field:
        raise ValueError("processing.dedup.exact_dedup.hash_field is required")

    sd = d["semantic_dedup"]
    if bool(sd.get("enable", False)):
        thr = _as_float(sd.get("threshold", 0.0), "processing.dedup.semantic_dedup.threshold")
        if not (0.0 <= thr <= 1.0):
            raise ValueError("processing.dedup.semantic_dedup.threshold must be in [0, 1]")

        for k in ("topk", "hnsw_m", "ef_construction", "ef_search"):
            v = _as_int(sd.get(k), f"processing.dedup.semantic_dedup.{k}")
            if v <= 0:
                raise ValueError(f"processing.dedup.semantic_dedup.{k} must be > 0")

    # ---- embedding ----
    e = s["embedding"]
    if not e["model_name"]:
        raise ValueError("embedding.model_name is required")

    batch = _as_int(e.get("batch_size"), "embedding.batch_size")
    if batch <= 0:
        raise ValueError("embedding.batch_size must be > 0")

    instr = e["instructions"]
    if not instr.get("passage"):
        raise ValueError("embedding.instructions.passage is required")
    if not instr.get("query"):
        raise ValueError("embedding.instructions.query is required")

    # ---- indexing ----
    v = s["indexing"]["vector"]
    if bool(v.get("enabled", True)):
        faiss_index = str(v.get("faiss_index", ""))
        if faiss_index not in ("FlatIP", "FlatL2"):
            raise ValueError(f"indexing.vector.faiss_index unsupported: {faiss_index}")

    # ---- outputs ----
    outputs = s["outputs"]
    if "chunks" not in outputs:
        raise ValueError("outputs.chunks is required")

    if bool(s["indexing"]["bm25"].get("enabled", True)) and "bm25_index" not in outputs:
        raise ValueError("outputs.bm25_index is required when bm25 is enabled")

    if bool(s["indexing"]["vector"].get("enabled", True)) and "vector_index" not in outputs:
        raise ValueError("outputs.vector_index is required when vector index is enabled")

    # ---- stores ----
    if not s["stores"]:
        raise ValueError("stores is required")

    # ensure referenced stores exist
    referenced = {s["input"]["input_store"]} | {block.get("store", "") for block in outputs.values() if isinstance(block, dict)}
    referenced.discard("")  # ignore empty
    missing = [name for name in referenced if name not in s["stores"]]
    if missing:
        raise ValueError(f"stores missing definitions for: {missing}")


def hash_settings(s: SettingsDict, *, exclude_keys: Optional[Set[str]] = None) -> str:
    """
    Stable hash for settings dict (used as config fingerprint).
    """
    exclude_keys = exclude_keys or set()
    filtered = {k: v for k, v in s.items() if k not in exclude_keys}
    blob = json.dumps(filtered, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# =========================
# Internal helpers
# =========================

def _deep_copy_dict(d: SettingsDict) -> SettingsDict:
    # json roundtrip is good enough here (yaml-safe types), and keeps it simple
    return json.loads(json.dumps(d, ensure_ascii=False))


def _must_be_mapping(v: Any, path: str) -> None:
    if not isinstance(v, dict):
        raise ValueError(f"{path} must be a mapping (YAML dict)")


def _as_int(v: Any, path: str) -> int:
    try:
        return int(v)
    except Exception as e:
        raise ValueError(f"{path} must be int-like, got {v!r}") from e


def _as_float(v: Any, path: str) -> float:
    try:
        return float(v)
    except Exception as e:
        raise ValueError(f"{path} must be float-like, got {v!r}") from e
