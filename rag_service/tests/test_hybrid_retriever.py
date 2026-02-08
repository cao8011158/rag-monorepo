import json
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pytest


import rag_service.common.retriever as R


# -----------------------
# Helpers
# -----------------------
def _write_jsonl(path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class FakeStore:
    """Minimal store for this unit test (filesystem-backed)."""

    def __init__(self, root):
        self.root = root

    def read_bytes(self, rel_path: str) -> bytes:
        p = self.root / rel_path
        return p.read_bytes()


def _fake_read_jsonl(store: FakeStore, rel_path: str):
    p = store.root / rel_path
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


class FakeEmbedder:
    """Mock DualInstructEmbedder: encode_queries returns deterministic vectors."""

    def __init__(self, dim: int = 3):
        self.dim = dim

    def encode_queries(self, qs: List[str]) -> np.ndarray:
        out = []
        for q in qs:
            q = (q or "").lower()
            if "cmu" in q or "carnegie" in q:
                out.append([1.0, 0.0, 0.0])
            elif "pittsburgh" in q:
                out.append([0.0, 1.0, 0.0])
            else:
                out.append([0.0, 0.0, 1.0])
        return np.asarray(out, dtype=np.float32)


class FakeBM25:
    """Mock bm25 object that supports get_scores(token_list)."""

    def __init__(self, scores: np.ndarray):
        self._scores = np.asarray(scores, dtype=np.float32)

    def get_scores(self, toks: List[str]) -> np.ndarray:
        # token list is ignored in this pure unit test
        return self._scores


def _base_settings(tmp_root) -> Dict[str, Any]:
    # 与你贴的 configs/rag.yaml 对齐（只保留本模块需要的字段）
    return {
        "stores": {
            "fs_local": {"kind": "filesystem", "root": str(tmp_root)},
        },
        "inputs": {
            "ce_artifacts": {
                "chunks": {"store": "fs_local", "base": "ce_out/chunks", "chunks_file": "chunks.jsonl"},
                "vector_index": {
                    "store": "fs_local",
                    "base": "ce_out/indexes/vector",
                    "faiss_index": "faiss.index",
                    "id_map": "id_map.jsonl",
                },
                "bm25_index": {"store": "fs_local", "base": "ce_out/indexes/bm25", "bm25_pkl": "bm25.pkl"},
            }
        },
        "models": {
            "embedding": {
                "model_name": "intfloat/e5-base-v2",
                "device": "cpu",
                "batch_size": 64,
                "cache_dir": str(tmp_root / ".hf_cache"),
                "normalize_embeddings": True,
                "instructions": {"passage": "passage: ", "query": "query: "},
            }
        },
        "retrieval": {
            "mode": "hybrid",
            "top_k": 30,
            "dense": {"top_k": 30},
            "bm25": {"top_k": 60},
            "hybrid_fusion": {"method": "rrf", "rrf_k": 60},
        },
    }


def _prepare_artifacts(tmp_path):
    # Minimal ChunkDoc rows: chunk_text is required by retrieve()
    chunks_rows = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "chunk_index": 0,
            "chunk_text": "Carnegie Mellon University (CMU) is in Pittsburgh.",
            "chunk_text_hash": "h1",
            "url": "https://example.com/1",
            "title": "Doc1",
        },
        {
            "chunk_id": "c2",
            "doc_id": "d2",
            "chunk_index": 0,
            "chunk_text": "Pittsburgh is a city in Pennsylvania.",
            "chunk_text_hash": "h2",
            "url": "https://example.com/2",
            "title": "Doc2",
        },
    ]
    id_map_rows = [
        {"faiss_id": 0, "chunk_id": "c1"},
        {"faiss_id": 1, "chunk_id": "c2"},
    ]

    chunks_path = tmp_path / "ce_out/chunks/chunks.jsonl"
    id_map_path = tmp_path / "ce_out/indexes/vector/id_map.jsonl"
    _write_jsonl(chunks_path, chunks_rows)
    _write_jsonl(id_map_path, id_map_rows)

    # Dummy bytes for faiss + bm25; we patch loader to ignore them.
    (tmp_path / "ce_out/indexes/vector").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ce_out/indexes/bm25").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ce_out/indexes/vector/faiss.index").write_bytes(b"FAISS_DUMMY")
    (tmp_path / "ce_out/indexes/bm25/bm25.pkl").write_bytes(b"BM25_DUMMY")


# -----------------------
# Fixtures
# -----------------------
@pytest.fixture()
def env(tmp_path, monkeypatch):
    """
    Patch:
      - build_store_registry -> FakeStore dict
      - read_jsonl -> filesystem jsonl reader
      - DualInstructEmbedder -> FakeEmbedder
      - load_bm25_portable + tokenize_query -> mocks
      - _deserialize_faiss_index -> real faiss index returned
    """
    _prepare_artifacts(tmp_path)
    settings = _base_settings(tmp_path)

    # stores
    stores = {"fs_local": FakeStore(tmp_path)}
    monkeypatch.setattr(R, "build_store_registry", lambda s: stores)

    # jsonl
    monkeypatch.setattr(R, "read_jsonl", _fake_read_jsonl)

    # embedder: replace constructor call with our fake
    monkeypatch.setattr(R, "DualInstructEmbedder", lambda **kwargs: FakeEmbedder(dim=3))

    # bm25
    doc_ids = ["c1", "c2"]
    # Make c1 score higher than c2 for any query (pure unit)
    bm25_scores = np.array([10.0, 3.0], dtype=np.float32)
    monkeypatch.setattr(
        R,
        "load_bm25_portable",
        lambda b: (FakeBM25(bm25_scores), doc_ids, "whoosh_stemming_v1"),
    )
    monkeypatch.setattr(R, "tokenize_query", lambda tok_id, q: ["dummy"])

    # faiss: return a real faiss index object (requires faiss-cpu/gpu installed)
    import faiss  # type: ignore

    emb = np.asarray(
        [
            [1.0, 0.0, 0.0],  # c1
            [0.0, 1.0, 0.0],  # c2
        ],
        dtype=np.float32,
    )
    index = faiss.IndexFlatIP(3)
    index.add(emb)

    monkeypatch.setattr(R, "_deserialize_faiss_index", lambda b: index)

    return settings


# -----------------------
# Tests: factory / alignment
# -----------------------
def test_from_settings_ok(env):
    hr = R.HybridRetriever.from_settings(env)
    assert hr.mode == "hybrid"
    assert hr.a.idx_to_key == ["c1", "c2"]
    assert hr.a.key_to_idx["c1"] == 0
    assert hr.a.key_to_idx["c2"] == 1
    assert hr.a.bm25_tokenizer_id == "whoosh_stemming_v1"


def test_chunks_id_map_length_mismatch_raises(tmp_path, monkeypatch):
    # Prepare mismatched files
    chunks_rows = [
        {"chunk_id": "c1", "doc_id": "d1", "chunk_index": 0, "chunk_text": "x", "chunk_text_hash": "h1"},
        {"chunk_id": "c2", "doc_id": "d2", "chunk_index": 0, "chunk_text": "y", "chunk_text_hash": "h2"},
    ]
    id_map_rows = [{"faiss_id": 0, "chunk_id": "c1"}]  # mismatch: only 1

    _write_jsonl(tmp_path / "ce_out/chunks/chunks.jsonl", chunks_rows)
    _write_jsonl(tmp_path / "ce_out/indexes/vector/id_map.jsonl", id_map_rows)
    (tmp_path / "ce_out/indexes/vector").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ce_out/indexes/bm25").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ce_out/indexes/vector/faiss.index").write_bytes(b"FAISS_DUMMY")
    (tmp_path / "ce_out/indexes/bm25/bm25.pkl").write_bytes(b"BM25_DUMMY")

    settings = _base_settings(tmp_path)
    stores = {"fs_local": FakeStore(tmp_path)}

    monkeypatch.setattr(R, "build_store_registry", lambda s: stores)
    monkeypatch.setattr(R, "read_jsonl", _fake_read_jsonl)
    monkeypatch.setattr(R, "DualInstructEmbedder", lambda **kwargs: FakeEmbedder(dim=3))
    monkeypatch.setattr(R, "_deserialize_faiss_index", lambda b: object())
    monkeypatch.setattr(R, "load_bm25_portable", lambda b: (FakeBM25([1.0]), ["c1"], "whoosh_stemming_v1"))
    monkeypatch.setattr(R, "tokenize_query", lambda tok_id, q: ["dummy"])

    with pytest.raises(ValueError, match="length mismatch"):
        R.HybridRetriever.from_settings(settings)


def test_bm25_doc_ids_unknown_chunk_raises(tmp_path, monkeypatch):
    # normal chunks/id_map
    _prepare_artifacts(tmp_path)
    settings = _base_settings(tmp_path)

    stores = {"fs_local": FakeStore(tmp_path)}
    monkeypatch.setattr(R, "build_store_registry", lambda s: stores)
    monkeypatch.setattr(R, "read_jsonl", _fake_read_jsonl)
    monkeypatch.setattr(R, "DualInstructEmbedder", lambda **kwargs: FakeEmbedder(dim=3))
    monkeypatch.setattr(R, "_deserialize_faiss_index", lambda b: object())

    # bm25 doc_ids includes unknown id
    monkeypatch.setattr(
        R,
        "load_bm25_portable",
        lambda b: (FakeBM25([1.0, 2.0]), ["c1", "UNKNOWN_CHUNK"], "whoosh_stemming_v1"),
    )
    monkeypatch.setattr(R, "tokenize_query", lambda tok_id, q: ["dummy"])

    with pytest.raises(ValueError, match="unknown chunk_ids"):
        R.HybridRetriever.from_settings(settings)


# -----------------------
# Tests: retrieval behaviors
# -----------------------
def test_dense_search_prefers_cmu(env):
    hr = R.HybridRetriever.from_settings(env)
    hr.mode = "dense"

    rows = hr.retrieve("CMU")
    assert len(rows) > 0
    assert rows[0]["key"] == "c1"
    assert "chunk_text" in rows[0]
    assert "dense" in rows[0] and rows[0]["dense"]["rank"] == 1


def test_bm25_search_prefers_c1(env):
    hr = R.HybridRetriever.from_settings(env)
    hr.mode = "bm25"

    rows = hr.retrieve("anything")
    assert len(rows) > 0
    assert rows[0]["key"] == "c1"
    assert rows[0]["bm25"]["rank"] == 1
    assert rows[0]["bm25"]["score"] is not None


def test_hybrid_rrf_contains_both_channels(env):
    hr = R.HybridRetriever.from_settings(env)
    hr.mode = "hybrid"
    hr.fusion_method = "rrf"
    hr.top_k = 2

    rows = hr.retrieve("CMU")
    assert [r["key"] for r in rows] == ["c1", "c2"]

    # schema checks
    r0 = rows[0]
    assert set(r0.keys()) == {"key", "chunk_text", "rrf_score", "dense", "bm25"}
    assert "rank" in r0["dense"] and "score" in r0["dense"]
    assert "rank" in r0["bm25"] and "score" in r0["bm25"]
    assert isinstance(r0["rrf_score"], float)


def test_dense_mode_without_faiss_raises(env):
    hr = R.HybridRetriever.from_settings(env)
    hr.mode = "dense"
    hr.a.faiss_index = None
    with pytest.raises(RuntimeError, match="FAISS index is not loaded"):
        hr.retrieve("CMU")


def test_bm25_mode_without_bm25_raises(env):
    hr = R.HybridRetriever.from_settings(env)
    hr.mode = "bm25"
    hr.a.bm25_obj = None
    with pytest.raises(RuntimeError, match="BM25 artifacts are not loaded"):
        hr.retrieve("CMU")
