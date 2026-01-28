# tests/test_pairing_build_pairs_for_query.py
import json
import numpy as np
import pytest

# 改成你工程的真实路径
from qr_pipeline.llm.pairing import build_pairs_for_query


# -----------------------
# Fakes
# -----------------------
class FakeLLM:
    """
    llm.generate(prompt)->str
    We ignore prompt content and return pre-set json.
    """
    def __init__(self, payload):
        self.payload = payload
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return json.dumps(self.payload, ensure_ascii=False)


class FakeEmbedder:
    """
    embedder.encode_passages(list[str])->np.ndarray
    We return deterministic unit vectors based on text -> pre-defined mapping.
    This lets us control cosine similarities precisely.
    """
    def __init__(self, vec_map):
        # vec_map: dict[str, np.ndarray] each vector will be normalized
        self.vec_map = {}
        for k, v in vec_map.items():
            v = np.asarray(v, dtype=np.float32)
            n = float(np.linalg.norm(v)) or 1.0
            self.vec_map[k] = (v / n).astype(np.float32)

    def encode_passages(self, texts):
        out = []
        for t in texts:
            if t not in self.vec_map:
                # fallback: unique-ish vector from hash, normalized
                h = abs(hash(t)) % 997
                v = np.array([h, h + 1, h + 2, h + 3], dtype=np.float32)
                v = v / (np.linalg.norm(v) or 1.0)
                out.append(v.astype(np.float32))
            else:
                out.append(self.vec_map[t])
        return np.vstack(out).astype(np.float32)


# -----------------------
# Helpers
# -----------------------
def mk_chunk(chunk_id: str, text: str, *, doc_id: str | None = None):
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id or chunk_id,
        "chunk_index": 0,
        "chunk_text": text,
        "chunk_text_hash": "h_" + chunk_id,
    }


# -----------------------
# Tests
# -----------------------
def test_autofill_source_chunk_ids_and_basic_pack():
    source = mk_chunk("S", "SOURCE TEXT")
    c1 = mk_chunk("A", "A TEXT")
    c2 = mk_chunk("B", "B TEXT")

    # LLM returns C2, but we will set max_extra_positives=1 to keep at most 1 extra
    llm = FakeLLM({"positives": [{"doc_id": "C2", "evidence": "x"}]})

    # Make SOURCE and B not too similar so B can be kept as extra positive
    embedder = FakeEmbedder({
        "SOURCE TEXT": [1, 0, 0, 0],
        "B TEXT":      [0, 1, 0, 0],
        "A TEXT":      [0, 0, 1, 0],
    })

    query = {"query_text": "what is x"}  # deliberately no source_chunk_ids

    pack, stats = build_pairs_for_query(
        query=query,
        source_doc=source,
        candidate_docs=[c1, c2, source],  # includes source; function should exclude it for LLM
        llm=llm,
        embedder=embedder,
        max_extra_positives=1,
        cosine_threshold=0.92,
        num_hard_negatives=10,
        enable_text_hash_dedup=True,
        include_one_shot=False,
    )

    assert "source_chunk_ids" in pack["query"]
    assert pack["query"]["source_chunk_ids"] == ["S"]

    # positives = [source] + [B] (extra)
    assert [d["chunk_id"] for d in pack["positives"]] == ["S", "B"]

    # negatives should be whatever remains (A), because B is positive and source removed from pool
    assert [d["chunk_id"] for d in pack["negatives"]] == ["A"]

    assert stats["num_samples"] == 1
    assert stats["num_candidates_in"] == 3
    assert stats["num_candidates_for_llm"] == 2  # source excluded


def test_llm_invalid_labels_dedup_and_sorted_by_rrf_order():
    source = mk_chunk("S", "SOURCE")
    a = mk_chunk("A", "DOC A")
    b = mk_chunk("B", "DOC B")
    c = mk_chunk("C", "DOC C")

    # candidate_docs (excluding source) in RRF order: A, B, C -> labels: C1=A, C2=B, C3=C
    # LLM outputs duplicates, invalid label, and out-of-order labels; code should:
    # - drop invalid ("C999")
    # - dedup keep first occurrence
    # - sort by numeric suffix => C1 then C3
    llm = FakeLLM({"positives": [{"doc_id": "C3"}, {"doc_id": "C1"}, {"doc_id": "C3"}, {"doc_id": "C999"}]})

    # make all dissimilar so cos dedup doesn't remove extras
    embedder = FakeEmbedder({
        "SOURCE": [1, 0, 0, 0],
        "DOC A":  [0, 1, 0, 0],
        "DOC B":  [0, 0, 1, 0],
        "DOC C":  [0, 0, 0, 1],
    })

    query = {"query_text": "q", "source_chunk_ids": ["S"]}

    pack, stats = build_pairs_for_query(
        query=query,
        source_doc=source,
        candidate_docs=[a, b, c, source],
        llm=llm,
        embedder=embedder,
        max_extra_positives=10,
        cosine_threshold=0.92,
        num_hard_negatives=10,
        enable_text_hash_dedup=False,
        include_one_shot=False,
    )

    # expected positives: source + (A from C1) + (C from C3) in RRF order
    assert [d["chunk_id"] for d in pack["positives"]] == ["S", "A", "C"]

    assert stats["invalid_label"] == 1
    assert stats["llm_pos_valid_total"] == 2


def test_positive_cosine_dedup_happens_before_truncation():
    """
    Ensure: LLM returns 2 extras; one is too similar to source and removed by cosine dedup,
    then truncation applies to remaining extras.
    """
    source = mk_chunk("S", "T_source")
    a = mk_chunk("A", "T_similar_to_source")
    b = mk_chunk("B", "T_far")

    llm = FakeLLM({"positives": [{"doc_id": "C1"}, {"doc_id": "C2"}]})  # C1=A, C2=B

    # Make A very similar to source: cosine=1.0, B orthogonal
    embedder = FakeEmbedder({
        "T_source":            [1, 0, 0, 0],
        "T_similar_to_source": [1, 0, 0, 0],
        "T_far":               [0, 1, 0, 0],
    })

    query = {"query_text": "q", "source_chunk_ids": ["S"]}

    pack, stats = build_pairs_for_query(
        query=query,
        source_doc=source,
        candidate_docs=[a, b, source],
        llm=llm,
        embedder=embedder,
        max_extra_positives=1,  # allow 1 extra after dedup
        cosine_threshold=0.92,
        num_hard_negatives=10,
        enable_text_hash_dedup=False,
        include_one_shot=False,
    )

    # A removed by cos dedup; B kept; then truncation keeps B (1 extra)
    assert [d["chunk_id"] for d in pack["positives"]] == ["S", "B"]
    assert stats["num_pos_after_cos_dedup"] == 2
    assert stats["num_extra_pos_final"] == 1


def test_negative_cosine_filter_and_hash_dedup_and_cap():
    source = mk_chunk("S", "P0")
    # candidates excluding source: A, B, C, D
    a = mk_chunk("A", "NEG_NEAR_POS")    # should be dropped by cosine filter
    b = mk_chunk("B", "NEG_DUP")         # duplicate (by normalized text) with C
    c = mk_chunk("C", "  neg_dup  ")     # same after _norm_text -> "neg_dup"
    d = mk_chunk("D", "NEG_OK")          # should remain

    # LLM marks none as extra positives => only source is positive
    llm = FakeLLM({"positives": []})

    # cosine filter: make A similar to positive (source) -> drop
    # others far -> keep
    embedder = FakeEmbedder({
        "P0":           [1, 0, 0, 0],
        "NEG_NEAR_POS": [1, 0, 0, 0],  # cosine 1 with P0
        "NEG_DUP":      [0, 1, 0, 0],
        "  neg_dup  ":  [0, 1, 0, 0],  # same vector, but hash dedup is text-based anyway
        "NEG_OK":       [0, 0, 1, 0],
    })

    query = {"query_text": "q", "source_chunk_ids": ["S"]}

    pack, stats = build_pairs_for_query(
        query=query,
        source_doc=source,
        candidate_docs=[a, b, c, d, source],
        llm=llm,
        embedder=embedder,
        max_extra_positives=2,
        cosine_threshold=0.92,
        num_hard_negatives=1,       # cap to 1 to test truncation after filters
        enable_text_hash_dedup=True,
        include_one_shot=False,
    )

    # A filtered out by cosine; B and C dedup by hash; remaining order should be B then D
    # cap to 1 => only B stays
    assert [x["chunk_id"] for x in pack["negatives"]] == ["B"]
    assert stats["num_neg_after_cos_filter"] == 3  # B,C,D (A dropped)
    assert stats["num_neg_after_hash_dedup"] == 2  # B,D (C deduped)
    assert stats["num_neg_final"] == 1


def test_errors_when_missing_required_methods():
    source = mk_chunk("S", "S")
    query = {"query_text": "q"}

    class BadLLM:  # no generate
        pass

    class BadEmbedder:  # no encode_passages
        pass

    with pytest.raises(TypeError):
        build_pairs_for_query(
            query=query,
            source_doc=source,
            candidate_docs=[source],
            llm=BadLLM(),
            embedder=FakeEmbedder({"S": [1, 0, 0, 0]}),
        )

    with pytest.raises(TypeError):
        build_pairs_for_query(
            query=query,
            source_doc=source,
            candidate_docs=[source],
            llm=FakeLLM({"positives": []}),
            embedder=BadEmbedder(),
        )


def test_error_when_query_text_empty():
    source = mk_chunk("S", "S")
    llm = FakeLLM({"positives": []})
    embedder = FakeEmbedder({"S": [1, 0, 0, 0]})

    with pytest.raises(ValueError):
        build_pairs_for_query(
            query={"query_text": "   "},
            source_doc=source,
            candidate_docs=[source],
            llm=llm,
            embedder=embedder,
        )
