# src/ce_pipeline/pipeline/embedding_stage.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import time

import numpy as np
import orjson
import faiss  # type: ignore

from ce_pipeline.stores.base import Store
from ce_pipeline.stores.registry import build_store_registry
from ce_pipeline.io.jsonl import read_jsonl
from ce_pipeline.embedding import DualInstructEmbedder

from ce_pipeline.processing.near_dedup import near_dedup_by_ann_faiss
from ce_pipeline.indexing.vector import build_faiss_index


# 固定产物命名（写到 outputs.vector_index.base 下）
FAISS_INDEX_NAME = "faiss.index"
ID_MAP_NAME = "id_map.jsonl"
META_NAME = "meta.json"


@dataclass(frozen=True)
class EmbeddingStageResult:
    vector_base: str
    chunks_path: str
    faiss_index_path: str
    id_map_path: str
    meta_path: str
    total_chunks_in: int
    total_vectors_out: int
    dim: int


def run_embedding_stage(s: Dict[str, Any]) -> EmbeddingStageResult:
    """
    Embedding Stage: chunks.jsonl -> embeddings -> near_dedup -> faiss.index + id_map

    Reads:
      - chunks: outputs.chunks.{store, base}/chunks.jsonl

    Writes (outputs.vector_index.base):
      - faiss.index          (FAISS index bytes)
      - id_map.jsonl         (faiss_id -> chunk_id)
      - meta.json            (run metadata)

    Required chunk fields (from chunks.jsonl):
      - chunk_id: str
      - chunk_text: str
    """
    t0 = time.time()

    # -------------------------
    # Resolve stores and paths
    # -------------------------
    stores = build_store_registry(s)

    chunks_out = s["outputs"]["chunks"]
    vec_out = s["outputs"]["vector_index"]

    chunks_store: Store = stores[chunks_out["store"]]
    vec_store: Store = stores[vec_out["store"]]

    chunks_base = str(chunks_out["base"])
    vec_base = str(vec_out["base"])

    chunks_path = _join_posix(chunks_base, "chunks.jsonl")
    faiss_index_path = _join_posix(vec_base, FAISS_INDEX_NAME)
    id_map_path = _join_posix(vec_base, ID_MAP_NAME)
    meta_path = _join_posix(vec_base, META_NAME)

    if not chunks_store.exists(chunks_path):
        raise FileNotFoundError(
            f"chunks.jsonl not found: store='{chunks_out['store']}', path='{chunks_path}'"
        )

    # -------------------------
    # Build embedder from settings
    # -------------------------
    emb_cfg = s["embedding"]
    instr = (emb_cfg.get("instructions") or {})  # {"passage": "...", "query": "..."}

    embedder = DualInstructEmbedder(
        model_name=str(emb_cfg["model_name"]),
        passage_instruction=str(instr.get("passage", "") or ""),
        query_instruction=str(instr.get("query", "") or ""),
        batch_size=int(emb_cfg.get("batch_size", 64)),
        normalize_embeddings=bool(emb_cfg.get("normalize_embeddings", True)),
        device=emb_cfg.get("device"),
    )

    # -------------------------
    # Load chunks
    # -------------------------
    chunk_ids: List[str] = []
    texts: List[str] = []

    total_in = 0
    for row in read_jsonl(chunks_store, chunks_path):
        total_in += 1

        cid = row.get("chunk_id")
        txt = row.get("chunk_text")

        if not isinstance(cid, str) or not cid:
            raise KeyError(f"Row #{total_in}: missing/invalid 'chunk_id'")
        if not isinstance(txt, str) or not txt:
            raise KeyError(f"Row #{total_in}: missing/invalid 'chunk_text'")

        chunk_ids.append(cid)
        texts.append(txt)

    if not texts:
        # 仍然写出空产物，保证可复现
        vec_store.write_bytes(faiss_index_path, b"")
        _write_jsonl(vec_store, id_map_path, [])
        _write_json(
            vec_store,
            meta_path,
            {
                "created_at_unix": int(time.time()),
                "chunks_path": chunks_path,
                "total_chunks_in": total_in,
                "total_vectors_out": 0,
                "dim": 0,
                "_meta": s.get("_meta", {}),
                "_timing_sec": {"embedding_stage": round(time.time() - t0, 4)},
            },
        )
        return EmbeddingStageResult(
            vector_base=vec_base,
            chunks_path=chunks_path,
            faiss_index_path=faiss_index_path,
            id_map_path=id_map_path,
            meta_path=meta_path,
            total_chunks_in=total_in,
            total_vectors_out=0,
            dim=0,
        )

    # -------------------------
    # Encode (passage embeddings)
    # -------------------------
    vecs = embedder.encode_passages(texts)  # np.ndarray float32 [N, D]
    if not isinstance(vecs, np.ndarray) or vecs.ndim != 2:
        raise RuntimeError(f"Embedder returned invalid embeddings: {type(vecs)} / {getattr(vecs, 'shape', None)}")

    n, dim = int(vecs.shape[0]), int(vecs.shape[1])
    if n != len(chunk_ids):
        raise RuntimeError(f"Embedding count mismatch: vecs={n}, chunk_ids={len(chunk_ids)}")

    # -------------------------
    # near_dedup 
    # -------------------------
    sd = (((s.get("processing") or {}).get("dedup") or {}).get("semantic_dedup") or {})
    enable_sd = bool(sd.get("enable", False))

    kept_indices = list(range(n))
    removed_mask = np.zeros((n,), dtype=bool)

    if enable_sd and n > 1:
        res = near_dedup_by_ann_faiss(
            vecs,
            threshold=float(sd.get("threshold", 0.95)),
            topk=int(sd.get("topk", 20)),
            hnsw_m=int(sd.get("hnsw_m", 32)),
            ef_construction=int(sd.get("ef_construction", 200)),
            ef_search=int(sd.get("ef_search", 128)),
            normalize=bool(sd.get("normalize", True)),
        )
        kept_indices = res.kept_indices
        removed_mask = res.removed_mask

    kept_vecs = vecs[kept_indices]
    kept_ids = [chunk_ids[i] for i in kept_indices]

    # -------------------------
    # Build FAISS index
    # -------------------------
    index_type = str(((s.get("indexing") or {}).get("vector") or {}).get("faiss_index", "FlatIP"))
    index = build_faiss_index(kept_vecs, index_type=index_type)

    # Serialize index -> bytes
    index_bytes = faiss.serialize_index(index)

    # -------------------------
    # Write outputs
    # -------------------------
    vec_store.write_bytes(faiss_index_path, index_bytes)

    # id_map.jsonl: faiss_id -> chunk_id
    id_rows = ({"faiss_id": i, "chunk_id": kept_ids[i]} for i in range(len(kept_ids)))
    _write_jsonl(vec_store, id_map_path, id_rows)

    meta = {
        "created_at_unix": int(time.time()),
        "chunks_path": chunks_path,
        "total_chunks_in": total_in,
        "total_vectors_out": int(kept_vecs.shape[0]),
        "dim": dim,
        "faiss_index_type": index_type,
        "semantic_dedup": {
            "enabled": enable_sd,
            "threshold": float(sd.get("threshold", 0.95)) if enable_sd else None,
            "topk": int(sd.get("topk", 20)) if enable_sd else None,
            "hnsw_m": int(sd.get("hnsw_m", 32)) if enable_sd else None,
            "ef_construction": int(sd.get("ef_construction", 200)) if enable_sd else None,
            "ef_search": int(sd.get("ef_search", 128)) if enable_sd else None,
            "normalize": bool(sd.get("normalize", True)) if enable_sd else None,
            "num_removed": int(removed_mask.sum()) if enable_sd else 0,
        },
        "_meta": s.get("_meta", {}),
        "_timing_sec": {"embedding_stage": round(time.time() - t0, 4)},
    }
    _write_json(vec_store, meta_path, meta)

    return EmbeddingStageResult(
        vector_base=vec_base,
        chunks_path=chunks_path,
        faiss_index_path=faiss_index_path,
        id_map_path=id_map_path,
        meta_path=meta_path,
        total_chunks_in=total_in,
        total_vectors_out=int(kept_vecs.shape[0]),
        dim=dim,
    )


# -------------------------
# Helpers
# -------------------------
def _join_posix(base: str, name: str) -> str:
    base = base.rstrip("/")
    name = name.lstrip("/")
    return f"{base}/{name}" if base else name


def _write_json(store: Store, path: str, obj: Dict[str, Any]) -> None:
    store.write_bytes(path, orjson.dumps(obj, option=orjson.OPT_INDENT_2) + b"\n")


def _write_jsonl(store: Store, path: str, rows) -> None:
    """
    覆盖式写入 JSONL（最终产物建议覆盖写）
    rows: Iterable[dict]
    """
    buf = bytearray()
    for r in rows:
        if not isinstance(r, dict):
            raise TypeError("write_jsonl expects dict rows")
        buf.extend(orjson.dumps(r))
        buf.extend(b"\n")
    store.write_bytes(path, bytes(buf))