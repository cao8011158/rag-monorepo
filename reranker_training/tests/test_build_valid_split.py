# tests/test_build_valid_split.py
from __future__ import annotations

import random
from typing import Any, Dict, List

import pytest

from reranker_training.io.jsonl import read_jsonl, write_jsonl
from reranker_training.stores.registry import build_store_registry
from reranker_training.data.build_valid_split import build_fixed_valid_jsonl_from_candidates


def _make_settings(tmp_root: str) -> Dict[str, Any]:
    """
    Build a minimal settings dict compatible with build_store_registry + data func.
    """
    return {
        "stores": {
            "fs_local": {
                "kind": "filesystem",
                "root": tmp_root,  # tmp_path as string
            }
        },
        "inputs": {
            "files": {
                "store": "fs_local",
                "base": "reranker_t_out",
                "candidates": "pairs.pairwise.jsonl",
                "valid_path": "data/processed/valid.jsonl",
            }
        },
        # train ratio
        "data_split": 0.85,
    }


def _candidate(
    qid: str,
    qtext: str,
    pos_id: str,
    pos_text: str,
    negs: List[Dict[str, str]],
    *,
    domain: str = "IN",
) -> Dict[str, Any]:
    return {
        "query_id": qid,
        "query_text": qtext,
        "positive": {"doc_id": pos_id, "text": pos_text},
        "negatives": negs,
        "source_chunk": "source chunk text",
        "meta": {"llm_model": "fake", "prompt_style": "x", "domain": domain},
    }


def _expected_valid_qids(all_qids: List[str], *, train_ratio: float, seed: int) -> List[str]:
    """
    Mirror the selection rule from _stable_sample_qids_for_valid:
    - unique qids, sorted
    - shuffle with seed
    - take n_valid = round(n_total*(1-train_ratio)) (min 1 if n_total>0)
    """
    qids_sorted = sorted(set(all_qids))
    n_total = len(qids_sorted)
    n_valid = int(round(n_total * (1.0 - train_ratio)))
    if n_total > 0 and n_valid == 0:
        n_valid = 1
    rng = random.Random(seed)
    rng.shuffle(qids_sorted)
    return sorted(qids_sorted[:n_valid])


def test_build_valid_writes_fixed_file_and_expands_pairs(tmp_path: pytest.TempPathFactory) -> None:
    s = _make_settings(str(tmp_path))

    stores = build_store_registry(s)
    store = stores["fs_local"]

    base = s["inputs"]["files"]["base"]
    candidates_path = f"{base}/{s['inputs']['files']['candidates']}"

    # Create candidates:
    # - q1 appears twice (two positives)
    # - q2 appears once
    # - q3 appears once
    # -> unique qids = 3; valid fraction ~ 15% -> round(0.45)=0 -> min 1 => 1 valid qid
    rows = [
        _candidate(
            "q1",
            "what is x",
            "p1",
            "pos text 1",
            [
                {"doc_id": "n1", "text": "neg 1"},
                {"doc_id": "n2", "text": "neg 2"},
                {"doc_id": "p1", "text": "SHOULD_BE_DROPPED_same_as_pos"},  # will be filtered
            ],
        ),
        _candidate(
            "q1",
            "what is x",
            "p2",
            "pos text 2",
            [
                {"doc_id": "n3", "text": "neg 3"},
                {"doc_id": "n4", "text": "neg 4"},
            ],
        ),
        _candidate(
            "q2",
            "what is y",
            "p3",
            "pos text 3",
            [
                {"doc_id": "n5", "text": "neg 5"},
                {"doc_id": "n6", "text": "neg 6"},
            ],
        ),
        _candidate(
            "q3",
            "what is z",
            "p4",
            "pos text 4",
            [
                {"doc_id": "n7", "text": "neg 7"},
            ],
        ),
    ]
    write_jsonl(store, candidates_path, rows)

    seed = 42
    stats = build_fixed_valid_jsonl_from_candidates(s, seed=seed)

    assert stats["num_candidate_rows"] == 4
    assert stats["num_unique_qids"] == 3
    assert stats["num_valid_qids"] == 1  # min-1 rule on small sets

    # Figure out which qid should be chosen by the deterministic sampling rule
    all_qids = [r["query_id"] for r in rows]
    exp_valid_qids = _expected_valid_qids(all_qids, train_ratio=float(s["data_split"]), seed=seed)
    assert len(exp_valid_qids) == 1
    chosen_qid = exp_valid_qids[0]

    valid_path = stats["valid_path"]
    out_rows = list(read_jsonl(store, valid_path))

    # Only rows belonging to chosen_qid appear.
    assert all(r["query_id"] == chosen_qid for r in out_rows)

    # Expanded format must contain 'negative' (single) instead of 'negatives' list
    assert all("negative" in r and "negatives" not in r for r in out_rows)

    # Ensure neg==pos was filtered (doc_id equal)
    for r in out_rows:
        assert r["negative"]["doc_id"] != r["positive"]["doc_id"]

    # Expected expansion count:
    # Count negatives across candidate rows for the chosen qid, excluding neg==pos
    expected_count = 0
    for r in rows:
        if r["query_id"] != chosen_qid:
            continue
        pos_id = r["positive"]["doc_id"]
        for neg in r["negatives"]:
            if neg["doc_id"] == pos_id:
                continue
            expected_count += 1

    assert len(out_rows) == expected_count
    assert stats["num_valid_rows"] == expected_count


def test_build_valid_is_deterministic_given_seed(tmp_path: pytest.TempPathFactory) -> None:
    s = _make_settings(str(tmp_path))
    stores = build_store_registry(s)
    store = stores["fs_local"]

    base = s["inputs"]["files"]["base"]
    candidates_path = f"{base}/{s['inputs']['files']['candidates']}"

    # 10 unique qids, each one row with 2 negatives => easy to check
    rows = []
    for i in range(10):
        qid = f"q{i}"
        rows.append(
            _candidate(
                qid,
                f"query {i}",
                f"p{i}",
                f"pos {i}",
                [{"doc_id": f"n{i}a", "text": "na"}, {"doc_id": f"n{i}b", "text": "nb"}],
            )
        )
    write_jsonl(store, candidates_path, rows)

    seed = 123
    stats1 = build_fixed_valid_jsonl_from_candidates(s, seed=seed)
    out1 = list(read_jsonl(store, stats1["valid_path"]))

    # Re-run (overwrite) with same seed should produce identical qid set and identical length
    stats2 = build_fixed_valid_jsonl_from_candidates(s, seed=seed)
    out2 = list(read_jsonl(store, stats2["valid_path"]))

    assert stats1["num_valid_qids"] == stats2["num_valid_qids"]
    assert len(out1) == len(out2)

    qids1 = sorted(set(r["query_id"] for r in out1))
    qids2 = sorted(set(r["query_id"] for r in out2))
    assert qids1 == qids2

    # With 10 qids and train_ratio 0.85 => valid ratio 0.15 => round(1.5)=2 valid qids
