# hybrid retrieval , 对单条query, 同时使用fassi 和 BM25 , 从 chunks.jsonl 提取top-k档案 , 使用 RRF  进行融合, 
# 输入为string (单条query)  输出为list[docmument](RRF 融合后的 top-k档案列)
# src/qr_pipeline/llm/retrieval.py
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import faiss  # type: ignore

from qr_pipeline.stores.registry import build_store_registry
from qr_pipeline.io.jsonl import read_jsonl
from qr_pipeline.processing.embedder import DualInstructEmbedder


def _posix_join(*parts: str) -> str:
    return "/".join([p.strip("/").replace("\\", "/") for p in parts if p])


def _simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _deserialize_faiss_index(b: bytes):
    """
    Robust FAISS index deserialization.

    Newer faiss python bindings (e.g. faiss-cpu 1.13.x) expect a 1D numpy uint8 array
    for deserialize_index. Passing raw bytes will crash with:
      AttributeError: 'bytes' object has no attribute 'shape'
    """
    # Preferred path: deserialize from np.uint8 view (works for cpu/gpu bindings)
    try:
        arr = np.frombuffer(b, dtype=np.uint8)
        return faiss.deserialize_index(arr)
    except Exception:
        # Fallback: some environments support reading from file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".index", delete=True) as f:
            f.write(b)
            f.flush()
            return faiss.read_index(f.name)


def _rrf_fuse(
    dense_ranked: List[Tuple[str, int, float]],
    bm25_ranked: List[Tuple[str, int, float]],
    *,
    rrf_k: int,
) -> Dict[str, float]:
    fused: Dict[str, float] = {}

    for doc_key, rank, _ in dense_ranked:
        fused[doc_key] = fused.get(doc_key, 0.0) + 1.0 / (rrf_k + rank)

    for doc_key, rank, _ in bm25_ranked:
        fused[doc_key] = fused.get(doc_key, 0.0) + 1.0 / (rrf_k + rank)

    return fused


def _linear_fuse(
    dense_scores: Dict[str, float],
    bm25_scores: Dict[str, float],
    *,
    w_dense: float,
    w_bm25: float,
) -> Dict[str, float]:
    keys = set(dense_scores.keys()) | set(bm25_scores.keys())
    fused: Dict[str, float] = {}
    for k in keys:
        fused[k] = w_dense * _safe_float(dense_scores.get(k, 0.0)) + w_bm25 * _safe_float(bm25_scores.get(k, 0.0))
    return fused


@dataclass
class RetrievalArtifacts:
    chunks_rows: List[Dict[str, Any]]
    id_map_rows: List[Dict[str, Any]]
    faiss_index: Any  # can be None when mode=bm25
    bm25_obj: Any     # can be None when mode=dense

    idx_to_chunk: List[Dict[str, Any]]
    idx_to_key: List[str]
    key_to_idx: Dict[str, int]


class HybridRetriever:
    """
    retrieve(query: str) -> List[Dict[str, Any]]

    每条输出:
    {
        "key": chunk_id,
        "chunk_text": "...",
        "rrf_score": float,
        "dense": {"rank": int|None, "score": float|None},
        "bm25":  {"rank": int|None, "score": float|None},
    }
    """

    def __init__(
        self,
        *,
        artifacts: RetrievalArtifacts,
        embedder: DualInstructEmbedder,
        mode: str,
        top_k: int,
        dense_top_k: int,
        bm25_top_k: int,
        fusion_method: str,
        rrf_k: int = 60,
        w_dense: float = 0.5,
        w_bm25: float = 0.5,
    ) -> None:
        self.a = artifacts
        self.embedder = embedder

        self.mode = str(mode)
        self.top_k = int(top_k)
        self.dense_top_k = int(dense_top_k)
        self.bm25_top_k = int(bm25_top_k)

        self.fusion_method = str(fusion_method)
        self.rrf_k = int(rrf_k)
        self.w_dense = float(w_dense)
        self.w_bm25 = float(w_bm25)

    # -----------------------------
    # Factory from settings
    # -----------------------------
    @staticmethod
    def from_settings(s: Dict[str, Any]) -> "HybridRetriever":
        stores = build_store_registry(s)
        ce = s["inputs"]["ce_artifacts"]

        chunks_cfg = ce["chunks"]
        vec_cfg = ce["vector_index"]
        bm25_cfg = ce["bm25_index"]

        chunks_store = stores[chunks_cfg["store"]]
        vec_store = stores[vec_cfg["store"]]
        bm25_store = stores[bm25_cfg["store"]]

        chunks_path = _posix_join(chunks_cfg["base"], chunks_cfg["chunks_file"])
        faiss_path = _posix_join(vec_cfg["base"], vec_cfg["faiss_index"])
        id_map_path = _posix_join(vec_cfg["base"], vec_cfg["id_map"])
        bm25_path = _posix_join(bm25_cfg["base"], bm25_cfg["bm25_pkl"])

        # Read lightweight metadata first
        chunks_rows = list(read_jsonl(chunks_store, chunks_path))
        id_map_rows = list(read_jsonl(vec_store, id_map_path))

        # Build idx_to_key from id_map (vector row alignment)
        idx_to_key: List[str] = []
        for row in id_map_rows:
            key = row["chunk_id"]
            idx_to_key.append(str(key))

        # IMPORTANT: validate mismatch BEFORE loading faiss/bm25
        if len(chunks_rows) != len(idx_to_key):
            raise ValueError("chunks.jsonl and id_map.jsonl length mismatch")

        idx_to_chunk = chunks_rows
        key_to_idx = {k: i for i, k in enumerate(idx_to_key)}

        # Load embedder config
        emb_cfg = s["embedding"]
        instr = emb_cfg["instructions"]
        embedder = DualInstructEmbedder(
            model_name=emb_cfg["model_name"],
            passage_instruction=instr["passage"],
            query_instruction=instr["query"],
            batch_size=int(emb_cfg.get("batch_size", 64)),
            normalize_embeddings=bool(emb_cfg.get("normalize_embeddings", True)),
            device=emb_cfg.get("device"),
        )

        # Retrieval config
        r = s["retrieval"]
        mode = str(r["mode"])
        top_k = int(r["top_k"])
        dense_top_k = int(r["dense"]["top_k"])
        bm25_top_k = int(r["bm25"]["top_k"])

        fusion = r["hybrid_fusion"]
        method = str(fusion["method"])
        rrf_k = int(fusion.get("rrf_k", 60))
        w_dense = float(fusion.get("w_dense", 0.5))
        w_bm25 = float(fusion.get("w_bm25", 0.5))

        # Lazy-load heavy artifacts based on mode
        faiss_index: Optional[Any] = None
        bm25_obj: Optional[Any] = None

        if mode in {"dense", "hybrid"}:
            faiss_bytes = vec_store.read_bytes(faiss_path)
            faiss_index = _deserialize_faiss_index(faiss_bytes)

        if mode in {"bm25", "hybrid"}:
            bm25_bytes = bm25_store.read_bytes(bm25_path)
            bm25_obj = pickle.loads(bm25_bytes)

        artifacts = RetrievalArtifacts(
            chunks_rows=chunks_rows,
            id_map_rows=id_map_rows,
            faiss_index=faiss_index,
            bm25_obj=bm25_obj,
            idx_to_chunk=idx_to_chunk,
            idx_to_key=idx_to_key,
            key_to_idx=key_to_idx,
        )

        return HybridRetriever(
            artifacts=artifacts,
            embedder=embedder,
            mode=mode,
            top_k=top_k,
            dense_top_k=dense_top_k,
            bm25_top_k=bm25_top_k,
            fusion_method=method,
            rrf_k=rrf_k,
            w_dense=w_dense,
            w_bm25=w_bm25,
        )

    # -----------------------------
    # Public API
    # -----------------------------
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        dense_ranked, dense_scores = [], {}
        bm25_ranked, bm25_scores = [], {}

        if self.mode in {"dense", "hybrid"}:
            dense_ranked, dense_scores = self._dense_search(query, self.dense_top_k)

        if self.mode in {"bm25", "hybrid"}:
            bm25_ranked, bm25_scores = self._bm25_search(query, self.bm25_top_k)

        if self.mode == "dense":
            fused_scores = dense_scores
        elif self.mode == "bm25":
            fused_scores = bm25_scores
        else:
            if self.fusion_method == "rrf":
                fused_scores = _rrf_fuse(dense_ranked, bm25_ranked, rrf_k=self.rrf_k)
            else:
                fused_scores = _linear_fuse(dense_scores, bm25_scores, w_dense=self.w_dense, w_bm25=self.w_bm25)

        final_keys = sorted(fused_scores, key=lambda k: fused_scores[k], reverse=True)[: self.top_k]

        dense_rank_map = {k: r for (k, r, _) in dense_ranked}
        dense_score_map = {k: s for (k, _, s) in dense_ranked}
        bm25_rank_map = {k: r for (k, r, _) in bm25_ranked}
        bm25_score_map = {k: s for (k, _, s) in bm25_ranked}

        results: List[Dict[str, Any]] = []
        for k in final_keys:
            idx = self.a.key_to_idx[k]
            chunk_row = self.a.idx_to_chunk[idx]
            chunk_text = chunk_row["chunk_text"]  # ✅ 只取文本字段

            results.append(
                {
                    "key": k,
                    "chunk_text": chunk_text,
                    "rrf_score": float(fused_scores[k]),
                    "dense": {
                        "rank": dense_rank_map.get(k),
                        "score": dense_score_map.get(k),
                    },
                    "bm25": {
                        "rank": bm25_rank_map.get(k),
                        "score": bm25_score_map.get(k),
                    },
                }
            )

        return results

    # -----------------------------
    # Internal search
    # -----------------------------
    def _dense_search(self, query: str, top_k: int):
        if self.a.faiss_index is None:
            raise RuntimeError("Dense search requested but FAISS index is not loaded (mode mismatch).")

        qv = self.embedder.encode_queries([query]).astype(np.float32)
        distances, indices = self.a.faiss_index.search(qv, int(top_k))

        ranked = []
        score_map = {}

        for j, idx in enumerate(indices[0], start=1):
            key = self.a.idx_to_key[int(idx)]
            score = float(distances[0][j - 1])
            ranked.append((key, j, score))
            score_map[key] = score

        return ranked, score_map

    def _bm25_search(self, query: str, top_k: int):
        if self.a.bm25_obj is None:
            raise RuntimeError("BM25 search requested but BM25 object is not loaded (mode mismatch).")

        toks = _simple_tokenize(query)
        scores = np.asarray(self.a.bm25_obj.get_scores(toks), dtype=np.float32)

        k = min(int(top_k), len(scores))
        if k <= 0:
            return [], {}

        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx_sorted = top_idx[np.argsort(-scores[top_idx])]

        ranked = []
        score_map = {}

        for rank, idx in enumerate(top_idx_sorted, start=1):
            key = self.a.idx_to_key[int(idx)]
            score = float(scores[int(idx)])
            ranked.append((key, rank, score))
            score_map[key] = score

        return ranked, score_map
