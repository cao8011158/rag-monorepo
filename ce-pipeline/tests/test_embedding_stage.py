from __future__ import annotations

from typing import Any, Dict, List
from pathlib import Path

import numpy as np
import pytest

from ce_pipeline.stores.registry import build_store_registry
from ce_pipeline.io.jsonl import read_jsonl
from ce_pipeline.pipeline.embedding_stage import run_embedding_stage


def _write_jsonl_text(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import orjson

    buf = bytearray()
    for r in rows:
        buf.extend(orjson.dumps(r))
        buf.extend(b"\n")
    path.write_bytes(bytes(buf))


def _base_settings(tmp_path: Path) -> Dict[str, Any]:
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    return {
        "stores": {
            "fs_local": {
                "kind": "filesystem",
                "root": str(data_root),
            }
        },
        "outputs": {
            "chunks": {"store": "fs_local", "base": "ce_out/chunks"},
            "vector_index": {"store": "fs_local", "base": "ce_out/indexes/vector"},
            "bm25_index": {"store": "fs_local", "base": "ce_out/indexes/bm25"},
        },
        "embedding": {
            # 这里随便写都行，因为我们会 monkeypatch __post_init__
            "model_name": "dummy",
            "instructions": {"passage": "", "query": ""},
            "batch_size": 64,
            "normalize_embeddings": True,
            "device": None,
        },
        "indexing": {"vector": {"faiss_index": "FlatIP"}},
        "processing": {
            "dedup": {
                "semantic_dedup": {
                    "enable": False,
                    "threshold": 0.95,
                    "topk": 20,
                    "hnsw_m": 32,
                    "ef_construction": 200,
                    "ef_search": 128,
                    "normalize": True,
                }
            }
        },
        "_meta": {"run_date": "test"},
    }


def _chunks_file_on_disk(tmp_path: Path) -> Path:
    return tmp_path / "data" / "ce_out" / "chunks" / "chunks.jsonl"


def _read_chunks_ids_via_store(s: Dict[str, Any]) -> List[str]:
    stores = build_store_registry(s)
    chunks_out = s["outputs"]["chunks"]
    store = stores[chunks_out["store"]]
    chunks_path = f"{chunks_out['base'].rstrip('/')}/chunks.jsonl"
    return [row["chunk_id"] for row in read_jsonl(store, chunks_path)]


def _patch_embedder(monkeypatch: pytest.MonkeyPatch, *, vecs: np.ndarray) -> None:
    """
    ✅ 关键补丁：
    - patch __post_init__：避免 HF 下载
    - patch encode_passages：直接返回我们指定的 embeddings
    """
    import ce_pipeline.pipeline.embedding_stage as es

    def _fake_post_init(self) -> None:
        self._model = object()

    def _fake_encode_passages(self, texts: List[str]) -> np.ndarray:
        assert len(texts) == int(vecs.shape[0])
        return vecs

    monkeypatch.setattr(es.DualInstructEmbedder, "__post_init__", _fake_post_init, raising=True)
    monkeypatch.setattr(es.DualInstructEmbedder, "encode_passages", _fake_encode_passages, raising=True)


def test_embedding_stage_no_prune_when_semantic_dedup_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = _base_settings(tmp_path)
    s["processing"]["dedup"]["semantic_dedup"]["enable"] = False

    _write_jsonl_text(
        _chunks_file_on_disk(tmp_path),
        [
            {"chunk_id": "c1", "chunk_text": "a"},
            {"chunk_id": "c2", "chunk_text": "b"},
            {"chunk_id": "c3", "chunk_text": "c"},
        ],
    )

    _patch_embedder(
        monkeypatch,
        vecs=np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
    )

    res = run_embedding_stage(s)

    ids_after = _read_chunks_ids_via_store(s)
    assert ids_after == ["c1", "c2", "c3"]
    assert res.total_chunks_in == 3
    assert res.total_vectors_out == 3
    assert res.dim == 2


def test_embedding_stage_prunes_chunks_when_semantic_dedup_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = _base_settings(tmp_path)
    s["processing"]["dedup"]["semantic_dedup"]["enable"] = True

    _write_jsonl_text(
        _chunks_file_on_disk(tmp_path),
        [
            {"chunk_id": "c1", "chunk_text": "a"},
            {"chunk_id": "c2", "chunk_text": "b"},
            {"chunk_id": "c3", "chunk_text": "c"},
        ],
    )

    _patch_embedder(
        monkeypatch,
        vecs=np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )

    # ✅ 不 mock near_dedup_and_prune_chunks，让它真实覆盖 chunks.jsonl
    # 只 mock near_dedup_by_ann_faiss 来固定删除 index=1
    import ce_pipeline.processing.near_dedup as nd

    def _fake_near_dedup_by_ann_faiss(*args, **kwargs):
        removed_mask = np.asarray([False, True, False], dtype=bool)
        return nd.ANNDedupResult(
            kept_indices=[0, 2],
            removed_mask=removed_mask,
            num_kept=2,
            num_removed=1,
        )

    monkeypatch.setattr(nd, "near_dedup_by_ann_faiss", _fake_near_dedup_by_ann_faiss, raising=True)

    out = run_embedding_stage(s)

    # ✅ 1) chunks.jsonl 被真实覆写删行
    ids_after = _read_chunks_ids_via_store(s)
    assert ids_after == ["c1", "c3"]

    # ✅ 2) vectors_out 与删行后保持一致
    assert out.total_chunks_in == 3
    assert out.total_vectors_out == 2
    assert out.dim == 2

    # ✅ 3) id_map.jsonl 也对齐
    stores = build_store_registry(s)
    vec_out = s["outputs"]["vector_index"]
    vec_store = stores[vec_out["store"]]
    id_map_path = f"{vec_out['base'].rstrip('/')}/id_map.jsonl"

    id_rows = list(read_jsonl(vec_store, id_map_path))
    assert [r["chunk_id"] for r in id_rows] == ["c1", "c3"]
    assert [r["faiss_id"] for r in id_rows] == [0, 1]
