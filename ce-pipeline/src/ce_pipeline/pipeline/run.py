# src/ce_pipeline/pipeline/run.py
from __future__ import annotations

from typing import Any, Dict
import argparse
import json

from ce_pipeline.pipeline.embedding_stage import run_embedding_stage, EmbeddingStageResult
from ce_pipeline.pipeline.indexing_stage import run_indexing_stage
from ce_pipeline.stores.registry import build_store_registry
from ce_pipeline.settings import load_settings


def run_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrate pipeline:
      1) embedding_stage
      2) indexing_stage (bm25 + vector)
    """

    # -------------------------
    # 1) Embedding
    # -------------------------
    emb_res: EmbeddingStageResult = run_embedding_stage(cfg)

    # ---- read embedding meta.json ----
    semantic_dedup_info = None
    timing_info = None

    try:
        stores = build_store_registry(cfg)
        vec_out = cfg["outputs"]["vector_index"]
        vec_store = stores[vec_out["store"]]

        meta_path = emb_res.meta_path
        if vec_store.exists(meta_path):
            raw = vec_store.read_bytes(meta_path)
            meta = json.loads(raw.decode("utf-8"))

            semantic_dedup_info = meta.get("semantic_dedup")
            timing_info = meta.get("_timing_sec")
    except Exception as e:
        semantic_dedup_info = {"error": f"Failed to read meta.json: {e}"}

    # -------------------------
    # 2) Indexing (BM25 / vector)
    # -------------------------
    bm25_res = run_indexing_stage(cfg)

    # -------------------------
    # Summary
    # -------------------------
    return {
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ce_pipeline embedding + indexing pipeline"
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to pipeline config yaml",
    )

    args = parser.parse_args()

    # ✅ 使用你项目的 settings 系统
    cfg = load_settings(args.config)

    summary = run_pipeline(cfg)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
