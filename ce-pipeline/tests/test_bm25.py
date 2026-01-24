# tests/test_bm25.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List

import pytest

from ce_pipeline.indexing.bm25 import (
    _simple_tokenize,
    build_bm25_index,
    save_bm25,
    load_bm25,
    BM25Artifact,
)


class InMemoryStore:
    """
    Minimal Store double for tests.

    The production Store likely supports filesystem / s3 / etc.
    Here we only need write_bytes/read_bytes to test bm25 artifact persistence.
    """
    def __init__(self) -> None:
        self._blob: Dict[str, bytes] = {}

    def write_bytes(self, path: str, data: bytes) -> None:
        self._blob[path] = data

    def read_bytes(self, path: str) -> bytes:
        if path not in self._blob:
            raise FileNotFoundError(path)
        return self._blob[path]


def test_simple_tokenize_lower_split_and_strip() -> None:
    assert _simple_tokenize("Hello   WORLD") == ["hello", "world"]
    assert _simple_tokenize("") == []
    assert _simple_tokenize("   ") == []
    assert _simple_tokenize(None) == []  # type: ignore[arg-type]


def test_build_bm25_index_scores_matching_doc_higher() -> None:
    texts = [
        "carnegie mellon university is in pittsburgh",
        "pittsburgh is a city in pennsylvania",
        "cmu is a top research university",
        "this document talks about cooking recipes",
        "how to bake a cake at home",
    ]
    bm25 = build_bm25_index(texts)

    q = _simple_tokenize("pittsburgh university")
    scores = bm25.get_scores(q)

    # doc0 should score higher than doc1
    assert scores[0] > scores[1]


def test_save_and_load_bm25_roundtrip_preserves_doc_ids_and_scores() -> None:
    store = InMemoryStore()
    path = "ce_out/indexes/bm25/index.pkl"

    texts = [
        "cmu is in pittsburgh",
        "pittsburgh has many bridges",
        "machine learning research at cmu",
    ]
    doc_ids = ["chunk_0", "chunk_1", "chunk_2"]
    bm25 = build_bm25_index(texts)
    save_bm25(store, path, bm25, doc_ids)

    artifact = load_bm25(store, path)
    assert isinstance(artifact, BM25Artifact)
    assert artifact.doc_ids == doc_ids

    q = _simple_tokenize("cmu pittsburgh research")
    scores_before = bm25.get_scores(q)
    scores_after = artifact.bm25.get_scores(q)

    # Functional equivalence: scores should match after reload
    assert scores_after == pytest.approx(scores_before, rel=1e-9, abs=1e-12)


def test_load_missing_path_raises() -> None:
    store = InMemoryStore()
    with pytest.raises(FileNotFoundError):
        load_bm25(store, "not-exist.pkl")
