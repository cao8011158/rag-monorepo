# src/rq_pipeline/settings.py
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
    Load YAML -> normalized nested dict settings.

    Guarantees:
    - defaults are applied (so required nested maps exist)
    - validation is executed (ValueError with clear messages)
    - runtime metadata is attached into settings["_meta"]
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping (YAML dict)")

    s = apply_defaults(raw)
    validate_settings(s)

    s.setdefault("_meta", {})
    s["_meta"]["config_path"] = str(path)
    s["_meta"]["config_hash"] = hash_settings(s, exclude_keys={"_meta"})
    return s


def apply_defaults(raw: SettingsDict) -> SettingsDict:
    """
    Apply defaults aligned to your current `configs/rag.yaml`.

    Supported top-level blocks:
      - service.host / service.port
      - stores.*
      - inputs.ce_artifacts.{chunks, vector_index, bm25_index}
      - models.gemini_api.model_name
      - models.reranker.*
      - models.embedding.*
      - retrieval.*
      - generation.*

    Backward-compat:
      - If older configs include outputs/query_generation/pair_construction/models.llm, we keep them,
        but they are NOT required for rag.yaml to validate.
    """
    s: SettingsDict = _deep_copy_dict(raw)

    # ---- service ----
    s.setdefault("service", {})
    _must_be_mapping(s["service"], "service")
    svc = s["service"]
    svc.setdefault("host", "0.0.0.0")
    svc.setdefault("port", 8000)

    # ---- stores ----
    s.setdefault("stores", {})
    _must_be_mapping(s["stores"], "stores")

    # ---- inputs ----
    s.setdefault("inputs", {})
    _must_be_mapping(s["inputs"], "inputs")

    s["inputs"].setdefault("ce_artifacts", {})
    _must_be_mapping(s["inputs"]["ce_artifacts"], "inputs.ce_artifacts")
    ca = s["inputs"]["ce_artifacts"]

    ca.setdefault("chunks", {})
    _must_be_mapping(ca["chunks"], "inputs.ce_artifacts.chunks")
    ca["chunks"].setdefault("store", "")
    ca["chunks"].setdefault("base", "")
    ca["chunks"].setdefault("chunks_file", "chunks.jsonl")

    ca.setdefault("vector_index", {})
    _must_be_mapping(ca["vector_index"], "inputs.ce_artifacts.vector_index")
    ca["vector_index"].setdefault("store", "")
    ca["vector_index"].setdefault("base", "")
    ca["vector_index"].setdefault("faiss_index", "faiss.index")
    ca["vector_index"].setdefault("id_map", "id_map.jsonl")

    ca.setdefault("bm25_index", {})
    _must_be_mapping(ca["bm25_index"], "inputs.ce_artifacts.bm25_index")
    ca["bm25_index"].setdefault("store", "")
    ca["bm25_index"].setdefault("base", "")
    ca["bm25_index"].setdefault("bm25_pkl", "bm25.pkl")

    # ---- models ----
    s.setdefault("models", {})
    _must_be_mapping(s["models"], "models")
    m = s["models"]

    # gemini_api (your rag.yaml uses this instead of models.llm)
    m.setdefault("gemini_api", {})
    _must_be_mapping(m["gemini_api"], "models.gemini_api")
    ga = m["gemini_api"]
    ga.setdefault("model_name", "")

    # reranker
    m.setdefault("reranker", {})
    _must_be_mapping(m["reranker"], "models.reranker")
    rr = m["reranker"]
    rr.setdefault("provider", "hf_transformers")
    rr.setdefault("model_name", "")
    rr.setdefault("device", "cpu")  # cpu/cuda
    rr.setdefault("cache_dir", None)
    rr.setdefault("batch_size", 16)
    rr.setdefault("max_length", 512)
    rr.setdefault("fp16", False)

    # embedding
    m.setdefault("embedding", {})
    _must_be_mapping(m["embedding"], "models.embedding")
    emb = m["embedding"]
    emb.setdefault("model_name", "intfloat/e5-base-v2")
    emb.setdefault("device", None)  # cpu/cuda/None
    emb.setdefault("batch_size", 64)
    emb.setdefault("cache_dir", None)
    emb.setdefault("normalize_embeddings", True)
    emb.setdefault("instructions", {})
    _must_be_mapping(emb["instructions"], "models.embedding.instructions")
    emb["instructions"].setdefault("passage", "passage: ")
    emb["instructions"].setdefault("query", "query: ")

    # Optional/back-compat: models.llm (kept if present, but not required)
    if "llm" in m:
        if m["llm"] is None:
            m["llm"] = {}
        _must_be_mapping(m["llm"], "models.llm")
        llm = m["llm"]
        llm.setdefault("provider", "hf_transformers")
        llm.setdefault("model_name", "")
        llm.setdefault("device", "cpu")
        llm.setdefault("cache_dir", None)
        llm.setdefault("max_new_tokens", 128)
        llm.setdefault("temperature", 0.7)
        llm.setdefault("top_p", 0.9)

    # ---- retrieval ----
    s.setdefault("retrieval", {})
    _must_be_mapping(s["retrieval"], "retrieval")
    r = s["retrieval"]
    r.setdefault("mode", "hybrid")  # dense | bm25 | hybrid
    r.setdefault("top_k", 30)

    r.setdefault("dense", {})
    _must_be_mapping(r["dense"], "retrieval.dense")
    r["dense"].setdefault("top_k", 30)
    r["dense"].setdefault("normalize_query", True)

    r.setdefault("bm25", {})
    _must_be_mapping(r["bm25"], "retrieval.bm25")
    r["bm25"].setdefault("top_k", 60)

    r.setdefault("hybrid_fusion", {})
    _must_be_mapping(r["hybrid_fusion"], "retrieval.hybrid_fusion")
    hf = r["hybrid_fusion"]
    hf.setdefault("method", "rrf")  # rrf | linear
    hf.setdefault("rrf_k", 60)
    hf.setdefault("w_dense", 0.5)
    hf.setdefault("w_bm25", 0.5)

    # ---- generation ----
    s.setdefault("generation", {})
    _must_be_mapping(s["generation"], "generation")
    g = s["generation"]
    g.setdefault("max_context_token", 20000)
    g.setdefault("max_docs", 2)

    # ---- optional/back-compat blocks (do not enforce) ----
    if "outputs" in s and s["outputs"] is None:
        s["outputs"] = {}
    if "outputs" in s:
        _must_be_mapping(s["outputs"], "outputs")
        out = s["outputs"]
        out.setdefault("store", "")
        out.setdefault("base", "")
        out.setdefault("files", {})
        _must_be_mapping(out["files"], "outputs.files")
        of = out["files"]
        of.setdefault("queries_in_domain", "queries/in_domain.jsonl")
        of.setdefault("queries_out_domain", "queries/out_domain.jsonl")
        of.setdefault("pairs", "pairs/query_pack.jsonl")
        of.setdefault("stats", "run_stats.json")
        of.setdefault("errors", "errors.jsonl")

    return s


def validate_settings(s: SettingsDict) -> None:
    """
    Validate normalized settings dict (after apply_defaults).
    Raises ValueError with explicit messages.

    NOTE:
    - This validator is aligned to `configs/rag.yaml`.
    - Legacy blocks (outputs/models.llm/query_generation/...) are not required.
    """
    # ---- service ----
    svc = s.get("service")
    if not isinstance(svc, dict):
        raise ValueError("service must be a mapping (YAML dict)")
    _require_nonempty_str(svc.get("host"), "service.host")
    _as_int(svc.get("port"), "service.port", min_value=1, max_value=65535)

    # ---- stores ----
    if not s.get("stores"):
        raise ValueError("stores is required")
    _must_be_mapping(s["stores"], "stores")

    # ---- inputs required ----
    inp = s.get("inputs")
    if not isinstance(inp, dict):
        raise ValueError("inputs must be a mapping (YAML dict)")

    ca = inp.get("ce_artifacts", {})
    if not isinstance(ca, dict):
        raise ValueError("inputs.ce_artifacts must be a mapping (YAML dict)")

    ch = ca.get("chunks")
    if not isinstance(ch, dict):
        raise ValueError("inputs.ce_artifacts.chunks must be a mapping (YAML dict)")
    _require_nonempty_str(ch.get("store"), "inputs.ce_artifacts.chunks.store")
    _require_nonempty_str(ch.get("base"), "inputs.ce_artifacts.chunks.base")
    _require_nonempty_str(ch.get("chunks_file"), "inputs.ce_artifacts.chunks.chunks_file")

    vi = ca.get("vector_index")
    if not isinstance(vi, dict):
        raise ValueError("inputs.ce_artifacts.vector_index must be a mapping (YAML dict)")
    _require_nonempty_str(vi.get("store"), "inputs.ce_artifacts.vector_index.store")
    _require_nonempty_str(vi.get("base"), "inputs.ce_artifacts.vector_index.base")
    _require_nonempty_str(vi.get("faiss_index"), "inputs.ce_artifacts.vector_index.faiss_index")
    _require_nonempty_str(vi.get("id_map"), "inputs.ce_artifacts.vector_index.id_map")

    bi = ca.get("bm25_index")
    if not isinstance(bi, dict):
        raise ValueError("inputs.ce_artifacts.bm25_index must be a mapping (YAML dict)")
    _require_nonempty_str(bi.get("store"), "inputs.ce_artifacts.bm25_index.store")
    _require_nonempty_str(bi.get("base"), "inputs.ce_artifacts.bm25_index.base")
    _require_nonempty_str(bi.get("bm25_pkl"), "inputs.ce_artifacts.bm25_index.bm25_pkl")

    # ---- models required for rag.yaml ----
    m = s.get("models")
    if not isinstance(m, dict):
        raise ValueError("models must be a mapping (YAML dict)")

    ga = m.get("gemini_api")
    if not isinstance(ga, dict):
        raise ValueError("models.gemini_api must be a mapping (YAML dict)")
    _require_nonempty_str(ga.get("model_name"), "models.gemini_api.model_name")

    rr = m.get("reranker")
    if not isinstance(rr, dict):
        raise ValueError("models.reranker must be a mapping (YAML dict)")
    _require_nonempty_str(rr.get("provider"), "models.reranker.provider")
    _require_nonempty_str(rr.get("model_name"), "models.reranker.model_name")
    _validate_enum(str(rr.get("device")), {"cpu", "cuda"}, "models.reranker.device")
    _as_int(rr.get("batch_size"), "models.reranker.batch_size", min_value=1)
    _as_int(rr.get("max_length"), "models.reranker.max_length", min_value=1)
    if not isinstance(rr.get("fp16"), bool):
        raise ValueError("models.reranker.fp16 must be boolean")

    emb = m.get("embedding")
    if not isinstance(emb, dict):
        raise ValueError("models.embedding must be a mapping (YAML dict)")
    _require_nonempty_str(emb.get("model_name"), "models.embedding.model_name")
    if emb.get("device") is not None:
        _validate_enum(str(emb.get("device")), {"cpu", "cuda"}, "models.embedding.device")
    _as_int(emb.get("batch_size"), "models.embedding.batch_size", min_value=1)
    if not isinstance(emb.get("normalize_embeddings"), bool):
        raise ValueError("models.embedding.normalize_embeddings must be boolean")
    if not isinstance(emb.get("instructions"), dict):
        raise ValueError("models.embedding.instructions must be a mapping (YAML dict)")
    _require_nonempty_str(emb["instructions"].get("passage"), "models.embedding.instructions.passage")
    _require_nonempty_str(emb["instructions"].get("query"), "models.embedding.instructions.query")

    # Optional/back-compat: models.llm (validate only if user provided)
    if "llm" in m:
        llm = m.get("llm")
        if not isinstance(llm, dict):
            raise ValueError("models.llm must be a mapping (YAML dict) when provided")
        # If provided, require minimal correctness
        _require_nonempty_str(llm.get("provider"), "models.llm.provider")
        _require_nonempty_str(llm.get("model_name"), "models.llm.model_name")
        _validate_enum(str(llm.get("device")), {"cpu", "cuda"}, "models.llm.device")
        _as_int(llm.get("max_new_tokens"), "models.llm.max_new_tokens", min_value=1)
        _as_float(llm.get("temperature"), "models.llm.temperature", min_value=0.0)
        _as_float(llm.get("top_p"), "models.llm.top_p", min_value=0.0, max_value=1.0)

    # ---- retrieval ----
    r = s.get("retrieval")
    if not isinstance(r, dict):
        raise ValueError("retrieval must be a mapping (YAML dict)")
    _validate_enum(str(r.get("mode")), {"dense", "bm25", "hybrid"}, "retrieval.mode")
    _as_int(r.get("top_k"), "retrieval.top_k", min_value=1)

    if not isinstance(r.get("dense"), dict):
        raise ValueError("retrieval.dense must be a mapping (YAML dict)")
    _as_int(r["dense"].get("top_k"), "retrieval.dense.top_k", min_value=1)
    if "normalize_query" in r["dense"] and not isinstance(r["dense"].get("normalize_query"), bool):
        raise ValueError("retrieval.dense.normalize_query must be boolean")

    if not isinstance(r.get("bm25"), dict):
        raise ValueError("retrieval.bm25 must be a mapping (YAML dict)")
    _as_int(r["bm25"].get("top_k"), "retrieval.bm25.top_k", min_value=1)

    if not isinstance(r.get("hybrid_fusion"), dict):
        raise ValueError("retrieval.hybrid_fusion must be a mapping (YAML dict)")
    hf = r["hybrid_fusion"]
    _validate_enum(str(hf.get("method")), {"rrf", "linear"}, "retrieval.hybrid_fusion.method")
    if str(hf.get("method")) == "rrf":
        _as_int(hf.get("rrf_k"), "retrieval.hybrid_fusion.rrf_k", min_value=1)
    else:
        _as_float(hf.get("w_dense"), "retrieval.hybrid_fusion.w_dense", min_value=0.0, max_value=1.0)
        _as_float(hf.get("w_bm25"), "retrieval.hybrid_fusion.w_bm25", min_value=0.0, max_value=1.0)

    # ---- generation ----
    g = s.get("generation")
    if not isinstance(g, dict):
        raise ValueError("generation must be a mapping (YAML dict)")
    _as_int(g.get("max_context_token"), "generation.max_context_token", min_value=1)
    _as_int(g.get("max_docs"), "generation.max_docs", min_value=1)

    # ---- referenced stores exist ----
    referenced: Set[str] = set()
    referenced.add(str(ch.get("store", "") or ""))
    referenced.add(str(vi.get("store", "") or ""))
    referenced.add(str(bi.get("store", "") or ""))
    referenced.discard("")
    missing = [name for name in sorted(referenced) if name not in s["stores"]]
    if missing:
        raise ValueError(f"stores missing definitions for: {missing}")

    # Optional: outputs validation only if user provided non-empty outputs.store/base
    if "outputs" in s and isinstance(s["outputs"], dict):
        out = s["outputs"]
        has_any = bool(str(out.get("store") or "").strip() or str(out.get("base") or "").strip())
        if has_any:
            _require_nonempty_str(out.get("store"), "outputs.store")
            _require_nonempty_str(out.get("base"), "outputs.base")
            if not isinstance(out.get("files"), dict):
                raise ValueError("outputs.files must be a mapping (YAML dict)")
            of = out["files"]
            _require_nonempty_str(of.get("pairs"), "outputs.files.pairs")


def hash_settings(s: SettingsDict, *, exclude_keys: Optional[Set[str]] = None) -> str:
    """Stable hash for settings dict (used as config fingerprint)."""
    exclude_keys = exclude_keys or set()
    filtered = {k: v for k, v in s.items() if k not in exclude_keys}
    blob = json.dumps(filtered, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# =========================
# Internal helpers
# =========================
def _deep_copy_dict(d: SettingsDict) -> SettingsDict:
    return json.loads(json.dumps(d, ensure_ascii=False))


def _must_be_mapping(v: Any, path: str) -> None:
    if not isinstance(v, dict):
        raise ValueError(f"{path} must be a mapping (YAML dict)")


def _require_nonempty_str(v: Any, path: str) -> str:
    vv = str(v or "")
    if not vv.strip():
        raise ValueError(f"{path} is required")
    return vv


def _validate_enum(v: str, allowed: Set[str], path: str) -> None:
    if v not in allowed:
        raise ValueError(f"{path} must be one of {sorted(allowed)}, got {v!r}")


def _as_int(v: Any, path: str, *, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    try:
        x = int(v)
    except Exception as e:
        raise ValueError(f"{path} must be int-like, got {v!r}") from e
    if min_value is not None and x < min_value:
        raise ValueError(f"{path} must be >= {min_value}, got {x}")
    if max_value is not None and x > max_value:
        raise ValueError(f"{path} must be <= {max_value}, got {x}")
    return x


def _as_float(v: Any, path: str, *, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    try:
        x = float(v)
    except Exception as e:
        raise ValueError(f"{path} must be float-like, got {v!r}") from e
    if min_value is not None and x < min_value:
        raise ValueError(f"{path} must be >= {min_value}, got {x}")
    if max_value is not None and x > max_value:
        raise ValueError(f"{path} must be <= {max_value}, got {x}")
    return x
