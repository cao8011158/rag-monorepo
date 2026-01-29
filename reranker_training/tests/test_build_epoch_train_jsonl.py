import pytest
from reranker_training.data.build_epoch_train_jsonl import build_epoch_pairs


def make_chunk(cid: str) -> dict:
    return {
        "chunk_id": cid,
        "doc_id": "doc1",
        "chunk_index": 0,
        "chunk_text": f"text of {cid}",
        "chunk_text_hash": cid + "_hash",
    }


def make_query_pack(num_pos: int, num_neg: int):
    return {
        "query": {
            "query_text": "what is AI?",
            "source_chunk_ids": ["src1"],
        },
        "positives": [make_chunk(f"p{i}") for i in range(num_pos)],
        "negatives": [make_chunk(f"n{i}") for i in range(num_neg)],
        "meta": {},
    }


def test_basic_generation():
    qp = make_query_pack(num_pos=2, num_neg=4)

    all_chunk_ids = ["x1", "x2", "x3", "x4", "x5"]
    chunk_map = {cid: make_chunk(cid) for cid in all_chunk_ids}

    rows = build_epoch_pairs(
        query_packs=[qp],
        all_chunk_ids=all_chunk_ids,
        chunk_map=chunk_map,
        epoch=1,
        seed=42,
        hard_negative_per_positive=2,
        random_negative_per_positive=1,
        fail_fast=True,
    )

    # 每个 positive: 2 hard + 1 random = 3
    assert len(rows) == 2 * 3

    for r in rows:
        assert "query" in r
        assert "positive" in r
        assert "negative" in r

        assert "chunk_text" in r["positive"]
        assert "chunk_text" in r["negative"]
        assert "query_text" in r["query"]


def test_fail_fast_raises():
    qp = make_query_pack(num_pos=0, num_neg=3)

    with pytest.raises(ValueError):
        build_epoch_pairs(
            query_packs=[qp],
            all_chunk_ids=["x1"],
            chunk_map={"x1": make_chunk("x1")},
            epoch=1,
            seed=0,
            hard_negative_per_positive=2,
            random_negative_per_positive=1,
            fail_fast=True,
        )


def test_best_effort_skips():
    qp = make_query_pack(num_pos=0, num_neg=3)

    rows = build_epoch_pairs(
        query_packs=[qp],
        all_chunk_ids=["x1"],
        chunk_map={"x1": make_chunk("x1")},
        epoch=1,
        seed=0,
        hard_negative_per_positive=2,
        random_negative_per_positive=1,
        fail_fast=False,
    )

    assert rows == []


def test_hn_less_than_p_allows_repeat():
    # h_n < p → k=1 allow repeat
    qp = make_query_pack(num_pos=3, num_neg=2)

    rows = build_epoch_pairs(
        query_packs=[qp],
        all_chunk_ids=["x1", "x2", "x3"],
        chunk_map={cid: make_chunk(cid) for cid in ["x1", "x2", "x3"]},
        epoch=1,
        seed=0,
        hard_negative_per_positive=4,
        random_negative_per_positive=0,
        fail_fast=True,
    )

    # 每个 positive 1 个 hard negative
    assert len(rows) == 3


def test_random_negative_exclusion():
    qp = make_query_pack(num_pos=1, num_neg=1)

    all_chunk_ids = ["src1", "p0", "n0", "x1"]
    chunk_map = {cid: make_chunk(cid) for cid in all_chunk_ids}

    rows = build_epoch_pairs(
        query_packs=[qp],
        all_chunk_ids=all_chunk_ids,
        chunk_map=chunk_map,
        epoch=1,
        seed=123,
        hard_negative_per_positive=1,
        random_negative_per_positive=1,
        fail_fast=True,
    )

    neg = rows[-1]["negative"]["chunk_id"]

    # random negative 不能是 source / positive / hard negative
    assert neg not in {"src1", "p0", "n0"}


def test_reproducibility_same_epoch():
    qp = make_query_pack(1, 3)
    all_chunk_ids = ["x1", "x2", "x3"]
    chunk_map = {cid: make_chunk(cid) for cid in all_chunk_ids}

    r1 = build_epoch_pairs(
        query_packs=[qp],
        all_chunk_ids=all_chunk_ids,
        chunk_map=chunk_map,
        epoch=1,
        seed=42,
        hard_negative_per_positive=2,
        random_negative_per_positive=1,
        fail_fast=True,
    )

    r2 = build_epoch_pairs(
        query_packs=[qp],
        all_chunk_ids=all_chunk_ids,
        chunk_map=chunk_map,
        epoch=1,
        seed=42,
        hard_negative_per_positive=2,
        random_negative_per_positive=1,
        fail_fast=True,
    )

    assert r1 == r2
