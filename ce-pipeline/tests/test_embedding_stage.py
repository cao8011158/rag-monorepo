# tests/test_embedding_stage.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

pytest.importorskip("faiss")  # 没装 faiss 就跳过整文件
import faiss  # type: ignore


def _write_chunks_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    import orjson

    path.parent.mkdir(parents=True, exist_ok=True)
    data = b"".join(orjson.dumps(r) + b"\n" for r in rows)
    path.write_bytes(data)


def _load_faiss_index_from_bytes(b: bytes) -> faiss.Index:
    """
    兼容 faiss 的 deserialize_index 输入类型：
    - 一些版本要求 np.uint8 1D array，而不是 python bytes
    """
    arr = np.frombuffer(b, dtype=np.uint8)
    return faiss.deserialize_index(arr)


class FakeEmbedder:
    """
    代替 DualInstructEmbedder：
    - 相同 chunk_text -> 返回完全一样的向量（保证 near_dedup 会删掉后出现的重复）
    - 不同文本 -> 不同向量
    """
    def __init__(
        self,
        model_name: str,
        passage_instruction: str,
        query_instruction: str,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        device: str | None = None,
    ) -> None:
        self.normalize_embeddings = bool(normalize_embeddings)

    def encode_passages(self, passages: List[str]) -> np.ndarray:
        if not passages:
            return np.zeros((0, 0), dtype=np.float32)

        # 8维向量足够测试
        out = np.zeros((len(passages), 8), dtype=np.float32)

        for i, t in enumerate(passages):
            # 用稳定hash构造向量；相同文本 -> 相同向量
            h = abs(hash(t))
            vec = np.array(
                [
                    (h % 97) / 97.0,
                    (h % 193) / 193.0,
                    (h % 389) / 389.0,
                    (h % 769) / 769.0,
                    ((h // 3) % 97) / 97.0,
                    ((h // 7) % 193) / 193.0,
                    ((h // 11) % 389) / 389.0,
                    ((h // 13) % 769) / 769.0,
                ],
                dtype=np.float32,
            )
            out[i] = vec

        if self.normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.maximum(norms, 1e-12)

        return out

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        # 本 stage 不用 query embedding，提供一个一致实现即可
        return self.encode_passages(queries)


def _base_settings(tmp_root: Path) -> Dict[str, Any]:
    """
    构造最小可运行 settings dict（不走 load_settings）
    """
    return {
        "stores": {
            "fs_local": {
                "kind": "filesystem",
                "root": str(tmp_root),
            }
        },
        "outputs": {
            "chunks": {"store": "fs_local", "base": "ce_out/chunks"},
            "vector_index": {"store": "fs_local", "base": "ce_out/indexes/vector"},
        },
        "embedding": {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "device": None,
            "batch_size": 64,
            "normalize_embeddings": True,
            "instructions": {"passage": "passage: ", "query": "query: "},
        },
        "processing": {
            "dedup": {
                "semantic_dedup": {
                    "enable": False,
                    "threshold": 0.95,
                    "topk": 20,
                    "hnsw_m": 32,
                    "ef_construction": 200,
                    "ef_search": 64,
                    "normalize": True,
                }
            }
        },
        "indexing": {"vector": {"faiss_index": "FlatIP"}},
        "_meta": {"config_path": "configs/pipeline.yaml", "config_hash": "x" * 64},
    }


def _read_jsonl_lines(path: Path) -> List[dict]:
    import orjson

    lines = []
    for raw in path.read_bytes().splitlines():
        if raw.strip():
            lines.append(orjson.loads(raw))
    return lines


def test_embedding_stage_no_dedup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # patch embedder
    import ce_pipeline.pipeline.embedding_stage as st

    monkeypatch.setattr(st, "DualInstructEmbedder", FakeEmbedder)

    s = _base_settings(tmp_path)
    s["processing"]["dedup"]["semantic_dedup"]["enable"] = False

    chunks_file = tmp_path / "ce_out/chunks/chunks.jsonl"
    rows = [
        {"chunk_id": "c1", "chunk_text": "hello"},
        {"chunk_id": "c2", "chunk_text": "world"},
        {"chunk_id": "c3", "chunk_text": "hello world"},
    ]
    _write_chunks_jsonl(chunks_file, rows)

    res = st.run_embedding_stage(s)

    # outputs exist
    faiss_bytes = (tmp_path / res.faiss_index_path).read_bytes()
    assert len(faiss_bytes) > 0

    idx = _load_faiss_index_from_bytes(faiss_bytes)
    assert idx.ntotal == 3

    id_map = _read_jsonl_lines(tmp_path / res.id_map_path)
    assert len(id_map) == 3
    assert [r["chunk_id"] for r in id_map] == ["c1", "c2", "c3"]
    assert [r["faiss_id"] for r in id_map] == [0, 1, 2]


def test_embedding_stage_with_near_dedup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # patch embedder
    import ce_pipeline.pipeline.embedding_stage as st

    monkeypatch.setattr(st, "DualInstructEmbedder", FakeEmbedder)

    s = _base_settings(tmp_path)
    sd = s["processing"]["dedup"]["semantic_dedup"]
    sd["enable"] = True
    sd["threshold"] = 0.999  # 相同向量必删
    sd["topk"] = 3
    sd["ef_search"] = 64
    sd["normalize"] = True

    chunks_file = tmp_path / "ce_out/chunks/chunks.jsonl"
    rows = [
        {"chunk_id": "c1", "chunk_text": "DUP"},
        {"chunk_id": "c2", "chunk_text": "unique"},
        {"chunk_id": "c3", "chunk_text": "DUP"},   # duplicate of c1 => should be removed
        {"chunk_id": "c4", "chunk_text": "unique2"},
    ]
    _write_chunks_jsonl(chunks_file, rows)

    res = st.run_embedding_stage(s)

    faiss_bytes = (tmp_path / res.faiss_index_path).read_bytes()
    idx = _load_faiss_index_from_bytes(faiss_bytes)

    # 期待删除 1 条
    assert idx.ntotal == 3

    id_map = _read_jsonl_lines(tmp_path / res.id_map_path)
    kept_ids = [r["chunk_id"] for r in id_map]
    assert kept_ids == ["c1", "c2", "c4"]  # c3 被删

    # meta.json 校验
    import orjson

    meta = orjson.loads((tmp_path / res.meta_path).read_bytes())
    assert meta["total_chunks_in"] == 4
    assert meta["total_vectors_out"] == 3
    assert meta["semantic_dedup"]["enabled"] is True
    assert meta["semantic_dedup"]["num_removed"] == 1


def test_embedding_stage_empty_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ce_pipeline.pipeline.embedding_stage as st

    monkeypatch.setattr(st, "DualInstructEmbedder", FakeEmbedder)

    s = _base_settings(tmp_path)

    chunks_file = tmp_path / "ce_out/chunks/chunks.jsonl"
    _write_chunks_jsonl(chunks_file, [])

    res = st.run_embedding_stage(s)

    # faiss.index 应该存在但为空 bytes（你的实现是 write b""）
    faiss_bytes = (tmp_path / res.faiss_index_path).read_bytes()
    assert faiss_bytes == b""

    id_map = _read_jsonl_lines(tmp_path / res.id_map_path)
    assert id_map == []

    import orjson

    meta = orjson.loads((tmp_path / res.meta_path).read_bytes())
    assert meta["total_chunks_in"] == 0
    assert meta["total_vectors_out"] == 0
    assert meta["dim"] == 0
