# tests/test_retrieval.py
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import pytest


# -----------------------------
# A minimal BM25 object (pickle-able)
# -----------------------------
class SimpleBM25:
    """
    Pickle-able object with get_scores(tokens) -> list[float] length == N
    Using simple token-overlap scoring for deterministic tests.
    """

    def __init__(self, corpus_tokens: List[List[str]]) -> None:
        self.corpus_tokens = corpus_tokens

    def get_scores(self, q_tokens: List[str]) -> List[float]:
        q = set(q_tokens)
        out: List[float] = []
        for doc_toks in self.corpus_tokens:
            out.append(float(len(q & set(doc_toks))))
        return out


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@pytest.fixture()
def faiss_mod():
    # If faiss not installed, skip these tests
    return pytest.importorskip("faiss")


# -----------------------------
# Fake embedder for tests (NO downloads, fixed dim=4)
# -----------------------------
class FakeDualInstructEmbedder:
    """
    A drop-in replacement for DualInstructEmbedder used by retrieval.py.

    Goal:
      - Always output float32 embeddings with dim=4 (to match our test FAISS index)
      - Deterministic: same input -> same vector
      - Simple semantics:
          query about pittsburgh -> close to c0=[1,0,0,0]
          query about CMU/carnegie mellon -> close to c1=[0.7,0.3,0,0]
          query about tokyo/japan -> close to c2=[0,0,1,0]
    """

    dim = 4

    def __init__(
        self,
        model_name: str,
        passage_instruction: str,
        query_instruction: str,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        device: str | None = None,
        **_: Any,
    ) -> None:
        self.model_name = model_name
        self.passage_instruction = passage_instruction
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device

    def _tokenize(self, s: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9]+", s.lower())

    def _vec_for_text(self, text: str) -> np.ndarray:
        toks = set(self._tokenize(text))

        # match test corpus intent
        if {"carnegie", "mellon", "university", "cmu"} & toks:
            v = np.array([0.7, 0.3, 0.0, 0.0], dtype=np.float32)  # like c1
        elif {"tokyo", "japan", "capital"} & toks:
            v = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)  # like c2
        else:
            v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # like c0

        if self.normalize_embeddings:
            n = float(np.linalg.norm(v))
            if n > 0:
                v = v / n
        return v

    def embed_query(self, text: str) -> np.ndarray:
        # return shape [D]
        return self._vec_for_text(text)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        # return shape [N, D]
        out = np.stack([self._vec_for_text(t) for t in texts], axis=0).astype(np.float32)
        return out

    def embed_passage(self, text: str) -> np.ndarray:
        return self._vec_for_text(text)

    def embed_passages(self, texts: Sequence[str]) -> np.ndarray:
        out = np.stack([self._vec_for_text(t) for t in texts], axis=0).astype(np.float32)
        return out

    # common alias in sentence-transformers style
    def encode(
        self,
        sentences: Union[str, Sequence[str]],
        **__: Any,
    ) -> np.ndarray:
        if isinstance(sentences, str):
            # sentence-transformers returns 1D for single str
            return self._vec_for_text(sentences)
        return self.embed_queries(list(sentences))


def _patch_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Patch the retrieval module's DualInstructEmbedder symbol so that:
      HybridRetriever.from_settings(...) uses FakeDualInstructEmbedder.
    """
    import qr_pipeline.llm.retrieval as r

    monkeypatch.setattr(r, "DualInstructEmbedder", FakeDualInstructEmbedder, raising=True)


@pytest.fixture()
def ce_artifacts_on_disk(tmp_path: Path, faiss_mod):
    """
    Create CE artifacts on disk under:
      <tmp_path>/ce_out/...

    Uses a SMALL deterministic FAISS index with dim=4.
    Our FakeDualInstructEmbedder outputs dim=4 too.
    """
    base_dir = tmp_path / "ce_out"

    chunks_path = base_dir / "chunks" / "chunks.jsonl"
    id_map_path = base_dir / "indexes" / "vector" / "id_map.jsonl"
    faiss_path = base_dir / "indexes" / "vector" / "faiss.index"
    bm25_path = base_dir / "indexes" / "bm25" / "bm25.pkl"

    # chunks.jsonl
    _write_jsonl(
        chunks_path,
        [
            {"chunk_id": "c0", "chunk_text": "Pittsburgh is a city in Pennsylvania."},
            {"chunk_id": "c1", "chunk_text": "Carnegie Mellon University is in Pittsburgh."},
            {"chunk_id": "c2", "chunk_text": "Tokyo is the capital of Japan."},
        ],
    )

    # id_map.jsonl (MUST align with vectors & chunks length)
    _write_jsonl(
        id_map_path,
        [
            {"chunk_id": "c0"},
            {"chunk_id": "c1"},
            {"chunk_id": "c2"},
        ],
    )

    # faiss.index (dim=4, IP)
    xb = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # c0
            [0.7, 0.3, 0.0, 0.0],  # c1
            [0.0, 0.0, 1.0, 0.0],  # c2
        ],
        dtype=np.float32,
    )
    # normalize to match FakeDualInstructEmbedder(normalize_embeddings=True)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    index = faiss_mod.IndexFlatIP(4)
    index.add(xb)
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    faiss_path.write_bytes(faiss_mod.serialize_index(index))

    # bm25.pkl (pickle)
    bm25_obj = SimpleBM25(
        [
            ["pittsburgh", "city", "pennsylvania"],
            ["carnegie", "mellon", "university", "pittsburgh"],
            ["tokyo", "capital", "japan"],
        ]
    )
    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    bm25_path.write_bytes(pickle.dumps(bm25_obj))

    return {"tmp_root": tmp_path, "base_dir": base_dir}


def _make_settings_like_your_yaml(tmp_root: Path) -> Dict[str, Any]:
    """
    Build settings dict that matches your YAML-like structure.

    NOTE:
      The "embedding.model_name" here is irrelevant in route B because we monkeypatch
      DualInstructEmbedder -> FakeDualInstructEmbedder (no downloads, dim=4).
    """
    return {
        "stores": {
            "fs_local": {
                "kind": "filesystem",
                "root": str(tmp_root),  # ✅ use tmp dir in tests
            }
        },
        "inputs": {
            "ce_artifacts": {
                "chunks": {
                    "store": "fs_local",
                    "base": "ce_out/chunks",
                    "chunks_file": "chunks.jsonl",
                },
                "vector_index": {
                    "store": "fs_local",
                    "base": "ce_out/indexes/vector",
                    "faiss_index": "faiss.index",
                    "id_map": "id_map.jsonl",
                },
                "bm25_index": {
                    "store": "fs_local",
                    "base": "ce_out/indexes/bm25",
                    "bm25_pkl": "bm25.pkl",
                },
            }
        },
        "embedding": {
            "model_name": "intfloat/e5-base-v2",  # doesn't matter in route B
            "instructions": {"passage": "passage:", "query": "query:"},
            "batch_size": 4,
            "normalize_embeddings": True,
            "device": "cpu",
        },
        "retrieval": {
            "mode": "hybrid",  # dense | bm25 | hybrid
            "top_k": 2,
            "dense": {"top_k": 2},
            "bm25": {"top_k": 2},
            "hybrid_fusion": {
                "method": "rrf",  # rrf | linear
                "rrf_k": 60,
                "w_dense": 0.5,
                "w_bm25": 0.5,
            },
        },
    }


def test_integration_from_settings_and_retrieve_hybrid_rrf(ce_artifacts_on_disk, monkeypatch: pytest.MonkeyPatch):
    _patch_embedder(monkeypatch)

    import qr_pipeline.llm.retrieval as r

    s = _make_settings_like_your_yaml(ce_artifacts_on_disk["tmp_root"])
    retriever = r.HybridRetriever.from_settings(s)

    out = retriever.retrieve("Where is Carnegie Mellon University?")
    assert isinstance(out, list)
    assert len(out) == 2

    for item in out:
        # ✅ schema as your retrieval.py docstring
        assert set(item.keys()) == {"key", "chunk_text", "rrf_score", "dense", "bm25"}
        assert isinstance(item["key"], str)
        assert isinstance(item["chunk_text"], str)
        assert isinstance(item["rrf_score"], float)
        assert "rank" in item["dense"] and "score" in item["dense"]
        assert "rank" in item["bm25"] and "score" in item["bm25"]

    # rrf_score should be descending
    assert out[0]["rrf_score"] >= out[1]["rrf_score"]


def test_integration_dense_only(ce_artifacts_on_disk, monkeypatch: pytest.MonkeyPatch):
    _patch_embedder(monkeypatch)

    import qr_pipeline.llm.retrieval as r

    s = _make_settings_like_your_yaml(ce_artifacts_on_disk["tmp_root"])
    s["retrieval"]["mode"] = "dense"

    retriever = r.HybridRetriever.from_settings(s)
    out = retriever.retrieve("pittsburgh")
    assert len(out) == 2

    for item in out:
        # in dense-only, bm25 wasn't executed -> should remain None
        assert item["bm25"]["rank"] is None
        assert item["bm25"]["score"] is None


def test_integration_bm25_only(ce_artifacts_on_disk, monkeypatch: pytest.MonkeyPatch):
    # not strictly necessary, but harmless & keeps consistency
    _patch_embedder(monkeypatch)

    import qr_pipeline.llm.retrieval as r

    s = _make_settings_like_your_yaml(ce_artifacts_on_disk["tmp_root"])
    s["retrieval"]["mode"] = "bm25"

    retriever = r.HybridRetriever.from_settings(s)
    out = retriever.retrieve("tokyo japan")
    assert len(out) == 2

    for item in out:
        # in bm25-only, dense wasn't executed -> should remain None
        assert item["dense"]["rank"] is None
        assert item["dense"]["score"] is None


def test_integration_length_mismatch_raises(tmp_path: Path, faiss_mod, monkeypatch: pytest.MonkeyPatch):
    """
    If chunks.jsonl length != id_map.jsonl length, retrieval.from_settings should raise ValueError.
    """
    _patch_embedder(monkeypatch)

    import qr_pipeline.llm.retrieval as r

    base_dir = tmp_path / "ce_out"
    chunks_path = base_dir / "chunks" / "chunks.jsonl"
    vec_dir = base_dir / "indexes" / "vector"
    bm25_dir = base_dir / "indexes" / "bm25"

    # chunks: 3
    _write_jsonl(
        chunks_path,
        [
            {"chunk_id": "c0", "chunk_text": "A"},
            {"chunk_id": "c1", "chunk_text": "B"},
            {"chunk_id": "c2", "chunk_text": "C"},
        ],
    )

    # id_map: 2 (mismatch)
    _write_jsonl(vec_dir / "id_map.jsonl", [{"chunk_id": "c0"}, {"chunk_id": "c1"}])

    # faiss vectors: 3 (valid file, dim=4)
    xb = np.array([[1, 0, 0, 0], [0.7, 0.3, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    index = faiss_mod.IndexFlatIP(4)
    index.add(xb)
    (vec_dir / "faiss.index").parent.mkdir(parents=True, exist_ok=True)
    (vec_dir / "faiss.index").write_bytes(faiss_mod.serialize_index(index))

    bm25_obj = SimpleBM25([["a"], ["b"], ["c"]])
    (bm25_dir / "bm25.pkl").parent.mkdir(parents=True, exist_ok=True)
    (bm25_dir / "bm25.pkl").write_bytes(pickle.dumps(bm25_obj))

    s = _make_settings_like_your_yaml(tmp_path)

    with pytest.raises(ValueError, match="length mismatch"):
        _ = r.HybridRetriever.from_settings(s)
