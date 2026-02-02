# src/ce_pipeline/pipeline/run.py
from __future__ import annotations

from typing import Any, Dict
import json

from ce_pipeline.pipeline.chunking_stage import run_chunking_stage
from ce_pipeline.pipeline.embedding_stage import run_embedding_stage, EmbeddingStageResult
from ce_pipeline.pipeline.indexing_stage import run_indexing_stage
from ce_pipeline.stores.registry import build_store_registry


def run_pipeline(
    cfg: Dict[str, Any],
    *,
    fail_fast: bool = False,
    best_effort_indexing: bool = True,
    run_bm25: bool = True,
) -> Dict[str, Any]:
    """
    Orchestrate full pipeline in order:
      1) chunking_stage
      2) embedding_stage
      3) indexing_stage (bm25, optional)

    Returns a summary dict for logging / CLI / CI.
    """

    # -------------------------
    # 1) Chunking
    # -------------------------
    run_chunking_stage(cfg)

    # -------------------------
    # 2) Embedding
    # -------------------------
    emb_res: EmbeddingStageResult = run_embedding_stage(cfg)

    # ---- read embedding meta.json ----
    semantic_dedup_info = None
    timing_info = None

    try:
        stores = build_store_registry(cfg)
        vec_out = cfg["outputs"]["vector_index"]
        vec_store = stores[vec_out["store"]]

        meta_path = emb_res.meta_path  # e.g. ce_out/indexes/vector/meta.json
        if vec_store.exists(meta_path):
            raw = vec_store.read_bytes(meta_path)
            meta = json.loads(raw.decode("utf-8"))

            semantic_dedup_info = meta.get("semantic_dedup")
            timing_info = meta.get("_timing_sec")
    except Exception as e:
        semantic_dedup_info = {"error": f"Failed to read meta.json: {e}"}

    # -------------------------
    # 3) BM25 (optional)
    # -------------------------
    bm25_res = None
    if run_bm25:
        bm25_res = run_indexing_stage(cfg, best_effort=best_effort_indexing)

    # -------------------------
    # Summary
    # -------------------------
    return {
        "chunking": chunk_res,
        "embedding": {
            "vector_base": emb_res.vector_base,
            "chunks_path": emb_res.chunks_path,
            "faiss_index_path": emb_res.faiss_index_path,
            "id_map_path": emb_res.id_map_path,
            "meta_path": emb_res.meta_path,
            "total_chunks_in": emb_res.total_chunks_in,
            "total_vectors_out": emb_res.total_vectors_out,
            "dim": emb_res.dim,
            "semantic_dedup": semantic_dedup_info,
            "timing_sec": timing_info,
        },
        "bm25": bm25_res,
    }
