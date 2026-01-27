# tests/test_pairing.py
from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import pytest

from qr_pipeline.llm.pairing import build_pairs_for_query


class FakeLLM:
    """A minimal LLM stub that returns a fixed JSON string."""
    def __init__(self, output: str):
        self._out = output
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self._out


class StableFakeEmbedder:
    """
    Deterministic & stable embedder for tests.

    Key properties:
    - If text exists in vec_map, use that vector.
    - Otherwise, generate a near-orthogonal one-hot vector based on a stable hash of the text.
      => avoids accidental high cosine similarities that would wrongly trigger cosine filters/dedup.
    - Always normalizes vectors so dot == cosine.
    """
    def __init__(self, vec_map: Dict[str, np.ndarray] | None = None, dim: int = 64):
        self.vec_map = vec_map or {}
        self.dim = dim

    @staticmethod
    def _stable_hash(s: str) -> int:
        # stable across runs: use sha1 instead of Python hash()
        import hashlib
        h = hashlib.sha1((s or "").encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    def encode_passages(self, texts: List[str]) -> np.ndarray:
        out: List[np.ndarray] = []
        for t in texts:
            if t in self.vec_map:
                v = self.vec_map[t].astype(np.float32).copy()
            else:
                idx = self._stable_hash(t) % self.dim
                v = np.zeros(self.dim, dtype=np.float32)
                v[idx] = 1.0

            n = float(np.linalg.norm(v))
            if n > 0:
                v = v / n
            out.append(v)

        return np.stack(out, axis=0)


def _doc(did: str, text: str) -> Dict[str, str]:
    return {"doc_id": did, "text": text}


# -----------------------------
# Tests
# -----------------------------
def test_prompt_uses_short_labels_C1_C2_not_long_doc_ids() -> None:
    """
    This test checks prompt formatting only:
    - prompt should use C1/C2 labels
    - should NOT contain long raw doc_ids
    It also expects source + 1 extra positive => 2 samples.
    """
    source = _doc("SOURCE_LONG_SHA256_ABC" * 2, "source sentence. more text.")
    cands = [
        _doc("LONGID_1_" * 10, "doc one text"),
        _doc("LONGID_2_" * 10, "doc two text"),
    ]

    llm = FakeLLM(
        json.dumps(
            {
                "positives": [{"doc_id": "C1", "evidence": "doc one text"}],
                "negatives": ["C2"],
            }
        )
    )

    # Use stable embedder; set threshold to 1.0 to ensure no accidental dedup/filter impacts this prompt test.
    embedder = StableFakeEmbedder(dim=64)

    samples, stats = build_pairs_for_query(
        query_text="q",
        source_doc=source,
        candidate_docs=[source] + cands,
        llm=llm,
        embedder=embedder,
        require_evidence=True,
        include_one_shot=False,
        cosine_threshold=1.0,  # keep this test focused on prompt+label behavior
    )

    assert llm.last_prompt is not None
    assert "doc_id: C1" in llm.last_prompt
    assert "doc_id: C2" in llm.last_prompt
    assert "LONGID_1_" not in llm.last_prompt
    assert "LONGID_2_" not in llm.last_prompt

    # source positive + 1 extra positive
    assert stats["num_samples"] == 2
    assert len(samples) == 2


def test_positive_cosine_dedup_happens_before_truncation() -> None:
    """
    RRF order: D2 (near-dup), D3 (near-dup), D4 (distinct).
    LLM marks all 3 as positives.
    Correct behavior: cosine-dedup first, then truncate extras => keep D2 and D4 (drop D3).
    """
    source = _doc("S", "SOURCE")
    d2 = _doc("D2", "A")
    d3 = _doc("D3", "A_DUP")
    d4 = _doc("D4", "B")

    llm = FakeLLM(
        json.dumps(
            {
                "positives": [
                    {"doc_id": "C1", "evidence": "A"},
                    {"doc_id": "C2", "evidence": "A_DUP"},
                    {"doc_id": "C3", "evidence": "B"},
                ],
                "negatives": [],
            }
        )
    )

    # Make A and A_DUP extremely similar, B orthogonal
    vA = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    vAdup = np.array([0.999, 0.001, 0.0], dtype=np.float32)
    vB = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    vS = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    embedder = StableFakeEmbedder(
        vec_map={"A": vA, "A_DUP": vAdup, "B": vB, "SOURCE": vS},
        dim=3,
    )

    samples, stats = build_pairs_for_query(
        query_text="q",
        source_doc=source,
        candidate_docs=[source, d2, d3, d4],
        llm=llm,
        embedder=embedder,
        require_evidence=True,
        cosine_threshold=0.92,
        max_extra_positives=2,
        include_one_shot=False,
    )

    assert stats["num_extra_pos_final"] == 2
    assert stats["num_pos_kept_final"] == 3  # source + 2 extras

    positives = [s["positive"]["doc_id"] for s in samples]
    assert "S" in positives
    assert "D2" in positives
    assert "D4" in positives
    assert "D3" not in positives


def test_require_evidence_filters_bad_positive() -> None:
    source = _doc("S", "SOURCE")
    d2 = _doc("D2", "Alpha content here")
    d3 = _doc("D3", "Beta content here")

    # evidence for C1 is NOT in D2 => should be dropped
    llm = FakeLLM(
        json.dumps(
            {
                "positives": [
                    {"doc_id": "C1", "evidence": "NOT IN DOC"},
                    {"doc_id": "C2", "evidence": "Beta content here"},
                ],
                "negatives": [],
            }
        )
    )

    embedder = StableFakeEmbedder(dim=64)

    samples, stats = build_pairs_for_query(
        query_text="q",
        source_doc=source,
        candidate_docs=[source, d2, d3],
        llm=llm,
        embedder=embedder,
        require_evidence=True,
        cosine_threshold=1.0,  # keep focus on evidence filter
        max_extra_positives=2,
        include_one_shot=False,
    )

    assert stats["invalid_evidence"] >= 1
    assert stats["num_extra_pos_final"] == 1

    positives = [s["positive"]["doc_id"] for s in samples]
    assert "D3" in positives
    assert "D2" not in positives


def test_negatives_are_capped_to_6_and_keep_rrf_order() -> None:
    source = _doc("S", "SOURCE")
    cands = [_doc(f"D{i}", f"NEG{i}") for i in range(1, 11)]

    llm = FakeLLM(
        json.dumps(
            {
                "positives": [],
                "negatives": [f"C{i}" for i in range(1, 11)],
            }
        )
    )

    embedder = StableFakeEmbedder(dim=128)

    samples, stats = build_pairs_for_query(
        query_text="q",
        source_doc=source,
        candidate_docs=[source] + cands,
        llm=llm,
        embedder=embedder,
        require_evidence=False,
        num_hard_negatives=6,
        cosine_threshold=1.0,  # avoid cosine dropping; this test checks cap+order
        include_one_shot=False,
    )

    assert stats["num_samples"] == 1
    assert samples[0]["positive"]["doc_id"] == "S"

    negs = samples[0]["negatives"]
    assert len(negs) == 6
    assert [d["doc_id"] for d in negs] == ["D1", "D2", "D3", "D4", "D5", "D6"]


def test_llm_negative_strict_pool_is_used() -> None:
    source = _doc("S", "SOURCE")
    d1 = _doc("D1", "t1")
    d2 = _doc("D2", "t2")
    d3 = _doc("D3", "t3")

    # LLM says negatives are only C3
    llm = FakeLLM(
        json.dumps(
            {
                "positives": [],
                "negatives": ["C3"],
            }
        )
    )

    embedder = StableFakeEmbedder(dim=64)

    samples, stats = build_pairs_for_query(
        query_text="q",
        source_doc=source,
        candidate_docs=[source, d1, d2, d3],
        llm=llm,
        embedder=embedder,
        require_evidence=False,
        num_hard_negatives=6,
        cosine_threshold=1.0,  # keep focus on strict pool selection
        include_one_shot=False,
    )

    negs = samples[0]["negatives"]
    assert [d["doc_id"] for d in negs] == ["D3"]
    assert stats["num_neg_pool"] == 1
