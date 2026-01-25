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
    Load rq-pipeline YAML -> normalized nested dict settings.

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

    s = apply_defaults(raw)
    validate_settings(s)

    s.setdefault("_meta", {})
    s["_meta"]["config_path"] = str(path)
    s["_meta"]["config_hash"] = hash_settings(s, exclude_keys={"_meta"})
    return s


def apply_defaults(raw: SettingsDict) -> SettingsDict:
    """
    Apply defaults to raw YAML, ensuring required nested objects exist.
    """
    s: SettingsDict = _deep_copy_dict(raw)

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

    # ---- outputs ----
    s.setdefault("outputs", {})
    _must_be_mapping(s["outputs"], "outputs")
    s["outputs"].setdefault("store", "")
    s["outputs"].setdefault("base", "")
    s["outputs"].setdefault("files", {})
    _must_be_mapping(s["outputs"]["files"], "outputs.files")

    of = s["outputs"]["files"]
    of.setdefault("queries_in_domain", "queries/in_domain.jsonl")
    of.setdefault("queries_out_domain", "queries/out_domain.jsonl")
    of.setdefault("candidates", "retrieval_candidates.jsonl")
    of.setdefault("pairs", "pairs.pairwise.train.jsonl")
    of.setdefault("stats", "run_stats.json")
    of.setdefault("errors", "errors.jsonl")

    # ---- models ----
    s.setdefault("models", {})
    _must_be_mapping(s["models"], "models")

    # LLM
    s["models"].setdefault("llm", {})
    _must_be_mapping(s["models"]["llm"], "models.llm")
    llm = s["models"]["llm"]
    llm.setdefault("provider", "hf_transformers")
    llm.setdefault("model_name", "")
    llm.setdefault("device", "cpu")  # cpu/cuda
    llm.setdefault("cache_dir", None)
    llm.setdefault("max_new_tokens", 128)
    llm.setdefault("temperature", 0.7)
    llm.setdefault("top_p", 0.9)

    # Embedder (for dense query embedding)
    s["models"].setdefault("embedding", {})
    _must_be_mapping(s["models"]["embedding"], "models.embedding")
    emb = s["models"]["embedding"]
    emb.setdefault("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    emb.setdefault("device", None)  # "cpu" / "cuda" / None(auto)
    emb.setdefault("batch_size", 64)
    emb.setdefault("cache_dir", None)
    emb.setdefault("normalize_embeddings", True)
    emb.setdefault("instructions", {})
    _must_be_mapping(emb["instructions"], "models.embedding.instructions")
    emb["instructions"].setdefault("passage", "passage: ")
    emb["instructions"].setdefault("query", "query: ")

    # ---- query_generation ----
    s.setdefault("query_generation", {})
    _must_be_mapping(s["query_generation"], "query_generation")

    qg = s["query_generation"]
    qg.setdefault("target_num_queries", 2000)

    qg.setdefault("sampling", {})
    _must_be_mapping(qg["sampling"], "query_generation.sampling")
    qg["sampling"].setdefault("seed", 42)
    qg["sampling"].setdefault("strategy", "uniform_random")
    qg["sampling"].setdefault("max_chunks_considered", 6500)
    qg["sampling"].setdefault("per_doc_cap", None)  # only for specific strategies

    qg.setdefault("prompt", {})
    _must_be_mapping(qg["prompt"], "query_generation.prompt")
    p = qg["prompt"]
    p.setdefault("language", "en")
    p.setdefault("style", "information-seeking")
    p.setdefault("num_queries_per_chunk", 1)
    p.setdefault("max_chunk_chars", 1800)
    p.setdefault("diversify", True)
    p.setdefault("diversity_hints", "")
    p.setdefault("avoid_near_duplicates", True)

    qg.setdefault("postprocess", {})
    _must_be_mapping(qg["postprocess"], "query_generation.postprocess")
    qg["postprocess"].setdefault("min_query_chars", 8)
    qg["postprocess"].setdefault("max_query_chars", 200)

    qg.setdefault("normalize", {})
    _must_be_mapping(qg["normalize"], "query_generation.normalize")
    qg["normalize"].setdefault("lower", True)
    qg["normalize"].setdefault("strip", True)
    qg["normalize"].setdefault("collapse_whitespace", True)

    # ---- processing ----
    s.setdefault("processing", {})
    _must_be_mapping(s["processing"], "processing")

    s["processing"].setdefault("dedup", {})
    _must_be_mapping(s["processing"]["dedup"], "processing.dedup")

    s["processing"]["dedup"].setdefault("semantic_dedup", {})
    _must_be_mapping(s["processing"]["dedup"]["semantic_dedup"], "processing.dedup.semantic_dedup")
    sd = s["processing"]["dedup"]["semantic_dedup"]
    sd.setdefault("enable", False)
    sd.setdefault("threshold", 0.95)
    sd.setdefault("topk", 50)
    sd.setdefault("hnsw_m", 32)
    sd.setdefault("ef_construction", 200)
    sd.setdefault("ef_search", 128)
    sd.setdefault("normalize", True)

    sd.setdefault("min_text_chars", 20)
    sd.setdefault("keep_strategy", "longest")  # longest | first
    sd.setdefault("max_remove_ratio", 0.5)

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

    # ---- pair_construction ----
    s.setdefault("pair_construction", {})
    _must_be_mapping(s["pair_construction"], "pair_construction")
    pc = s["pair_construction"]

    pc.setdefault("positive", {})
    _must_be_mapping(pc["positive"], "pair_construction.positive")
    pc["positive"].setdefault("strategy", "source_chunk")
    pc["positive"].setdefault("min_cosine", 0.85)  # only when threshold_match

    pc.setdefault("hard_negatives", {})
    _must_be_mapping(pc["hard_negatives"], "pair_construction.hard_negatives")
    hn = pc["hard_negatives"]
    hn.setdefault("num_per_query", 6)
    hn.setdefault("strategy", "top_rank_excluding_pos")

    hn.setdefault("filters", {})
    _must_be_mapping(hn["filters"], "pair_construction.hard_negatives.filters")
    flt = hn["filters"]
    flt.setdefault("enable_similarity_filter", True)
    flt.setdefault("max_cosine_with_positive", 0.92)
    flt.setdefault("enable_text_hash_dedup", True)

    # ---- run (optional but useful) ----
    s.setdefault("run", {})
    _must_be_mapping(s["run"], "run")
    s["run"].setdefault("fail_fast", False)
    s["run"].setdefault("log_level", "info")
    s["run"].setdefault("write_manifest", True)

    return s


def validate_settings(s: SettingsDict) -> None:
    """
    Validate normalized settings dict (after apply_defaults).
    Raises ValueError with explicit messages.
    """
    # ---- stores ----
    if not s.get("stores"):
        raise ValueError("stores is required")
    _must_be_mapping(s["stores"], "stores")

    # ---- inputs required ----
    inp = s["inputs"]
    ca = inp.get("ce_artifacts", {})
    if not isinstance(ca, dict):
        raise ValueError("inputs.ce_artifacts must be a mapping (YAML dict)")

    # chunks
    ch = ca["chunks"]
    _require_nonempty_str(ch.get("store"), "inputs.ce_artifacts.chunks.store")
    _require_nonempty_str(ch.get("base"), "inputs.ce_artifacts.chunks.base")
    _require_nonempty_str(ch.get("chunks_file"), "inputs.ce_artifacts.chunks.chunks_file")

    # vector
    vi = ca["vector_index"]
    _require_nonempty_str(vi.get("store"), "inputs.ce_artifacts.vector_index.store")
    _require_nonempty_str(vi.get("base"), "inputs.ce_artifacts.vector_index.base")
    _require_nonempty_str(vi.get("faiss_index"), "inputs.ce_artifacts.vector_index.faiss_index")
    _require_nonempty_str(vi.get("id_map"), "inputs.ce_artifacts.vector_index.id_map")

    # bm25
    bi = ca["bm25_index"]
    _require_nonempty_str(bi.get("store"), "inputs.ce_artifacts.bm25_index.store")
    _require_nonempty_str(bi.get("base"), "inputs.ce_artifacts.bm25_index.base")
    _require_nonempty_str(bi.get("bm25_pkl"), "inputs.ce_artifacts.bm25_index.bm25_pkl")

    # ---- outputs required ----
    out = s["outputs"]
    _require_nonempty_str(out.get("store"), "outputs.store")
    _require_nonempty_str(out.get("base"), "outputs.base")
    if not isinstance(out.get("files"), dict):
        raise ValueError("outputs.files must be a mapping (YAML dict)")

    # ---- models ----
    llm = s["models"]["llm"]
    _require_nonempty_str(llm.get("provider"), "models.llm.provider")
    _require_nonempty_str(llm.get("model_name"), "models.llm.model_name")
    _validate_enum(str(llm.get("device")), {"cpu", "cuda"}, "models.llm.device")
    _as_int(llm.get("max_new_tokens"), "models.llm.max_new_tokens", min_value=1)
    _as_float(llm.get("temperature"), "models.llm.temperature", min_value=0.0)
    _as_float(llm.get("top_p"), "models.llm.top_p", min_value=0.0, max_value=1.0)

    emb = s["models"]["embedding"]
    _require_nonempty_str(emb.get("model_name"), "models.embedding.model_name")
    if emb.get("device") is not None:
        _validate_enum(str(emb.get("device")), {"cpu", "cuda"}, "models.embedding.device")
    _as_int(emb.get("batch_size"), "models.embedding.batch_size", min_value=1)
    if not isinstance(emb.get("instructions"), dict):
        raise ValueError("models.embedding.instructions must be a mapping (YAML dict)")
    _require_nonempty_str(emb["instructions"].get("passage"), "models.embedding.instructions.passage")
    _require_nonempty_str(emb["instructions"].get("query"), "models.embedding.instructions.query")

    # ---- query_generation ----
    qg = s["query_generation"]
    _as_int(qg.get("target_num_queries"), "query_generation.target_num_queries", min_value=1)

    sp = qg["sampling"]
    _validate_enum(
        str(sp.get("strategy")),
        {"uniform_random", "stratified_by_doc", "per_doc_cap"},
        "query_generation.sampling.strategy",
    )
    _as_int(sp.get("max_chunks_considered"), "query_generation.sampling.max_chunks_considered", min_value=1)

    if str(sp.get("strategy")) in {"stratified_by_doc", "per_doc_cap"}:
        if sp.get("per_doc_cap") is None:
            raise ValueError("query_generation.sampling.per_doc_cap is required when strategy is stratified_by_doc/per_doc_cap")
        _as_int(sp.get("per_doc_cap"), "query_generation.sampling.per_doc_cap", min_value=1)

    pr = qg["prompt"]
    _require_nonempty_str(pr.get("language"), "query_generation.prompt.language")
    _require_nonempty_str(pr.get("style"), "query_generation.prompt.style")
    _as_int(pr.get("num_queries_per_chunk"), "query_generation.prompt.num_queries_per_chunk", min_value=1)
    _as_int(pr.get("max_chunk_chars"), "query_generation.prompt.max_chunk_chars", min_value=1)

    pp = qg["postprocess"]
    min_q = _as_int(pp.get("min_query_chars"), "query_generation.postprocess.min_query_chars", min_value=1)
    max_q = _as_int(pp.get("max_query_chars"), "query_generation.postprocess.max_query_chars", min_value=1)
    if min_q >= max_q:
        raise ValueError("query_generation.postprocess.min_query_chars must be < max_query_chars")

    # ---- semantic dedup ----
    sd = s["processing"]["dedup"]["semantic_dedup"]
    if bool(sd.get("enable", False)):
        thr = _as_float(sd.get("threshold"), "processing.dedup.semantic_dedup.threshold", min_value=0.0, max_value=1.0)
        _as_int(sd.get("topk"), "processing.dedup.semantic_dedup.topk", min_value=1)
        _as_int(sd.get("hnsw_m"), "processing.dedup.semantic_dedup.hnsw_m", min_value=1)
        _as_int(sd.get("ef_construction"), "processing.dedup.semantic_dedup.ef_construction", min_value=1)
        _as_int(sd.get("ef_search"), "processing.dedup.semantic_dedup.ef_search", min_value=1)
        _as_int(sd.get("min_text_chars"), "processing.dedup.semantic_dedup.min_text_chars", min_value=0)
        _validate_enum(str(sd.get("keep_strategy")), {"longest", "first"}, "processing.dedup.semantic_dedup.keep_strategy")
        _as_float(sd.get("max_remove_ratio"), "processing.dedup.semantic_dedup.max_remove_ratio", min_value=0.0, max_value=1.0)
        # threshold already validated in thr; keep thr variable to silence linters
        _ = thr

    # ---- retrieval ----
    r = s["retrieval"]
    _validate_enum(str(r.get("mode")), {"dense", "bm25", "hybrid"}, "retrieval.mode")
    _as_int(r.get("top_k"), "retrieval.top_k", min_value=1)

    _as_int(r["dense"].get("top_k"), "retrieval.dense.top_k", min_value=1)
    if not isinstance(r["dense"].get("normalize_query"), bool):
        raise ValueError("retrieval.dense.normalize_query must be boolean")

    _as_int(r["bm25"].get("top_k"), "retrieval.bm25.top_k", min_value=1)

    hf = r["hybrid_fusion"]
    _validate_enum(str(hf.get("method")), {"rrf", "linear"}, "retrieval.hybrid_fusion.method")
    if str(hf.get("method")) == "rrf":
        _as_int(hf.get("rrf_k"), "retrieval.hybrid_fusion.rrf_k", min_value=1)
    else:
        _as_float(hf.get("w_dense"), "retrieval.hybrid_fusion.w_dense", min_value=0.0, max_value=1.0)
        _as_float(hf.get("w_bm25"), "retrieval.hybrid_fusion.w_bm25", min_value=0.0, max_value=1.0)

    # ---- pair construction ----
    pc = s["pair_construction"]

    pos = pc["positive"]
    _validate_enum(str(pos.get("strategy")), {"source_chunk", "best_ranked", "threshold_match"}, "pair_construction.positive.strategy")
    if str(pos.get("strategy")) == "threshold_match":
        _as_float(pos.get("min_cosine"), "pair_construction.positive.min_cosine", min_value=0.0, max_value=1.0)

    hn = pc["hard_negatives"]
    _as_int(hn.get("num_per_query"), "pair_construction.hard_negatives.num_per_query", min_value=1)
    _validate_enum(
        str(hn.get("strategy")),
        {"top_rank_excluding_pos", "score_band", "mixed"},
        "pair_construction.hard_negatives.strategy",
    )

    flt = hn["filters"]
    if not isinstance(flt.get("enable_similarity_filter"), bool):
        raise ValueError("pair_construction.hard_negatives.filters.enable_similarity_filter must be boolean")
    if flt.get("enable_similarity_filter"):
        _as_float(
            flt.get("max_cosine_with_positive"),
            "pair_construction.hard_negatives.filters.max_cosine_with_positive",
            min_value=0.0,
            max_value=1.0,
        )
    if not isinstance(flt.get("enable_text_hash_dedup"), bool):
        raise ValueError("pair_construction.hard_negatives.filters.enable_text_hash_dedup must be boolean")

    # ---- referenced stores exist ----
    referenced: Set[str] = set()
    referenced.add(str(out.get("store", "") or ""))

    # inputs stores
    referenced.add(str(ch.get("store", "") or ""))
    referenced.add(str(vi.get("store", "") or ""))
    referenced.add(str(bi.get("store", "") or ""))

    referenced.discard("")
    missing = [name for name in sorted(referenced) if name not in s["stores"]]
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
    # yaml-safe types -> json roundtrip keeps it simple
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
