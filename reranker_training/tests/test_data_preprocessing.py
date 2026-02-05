# tests/test_data_preprocessing.py
from __future__ import annotations

import types
import pytest
import torch

# 你这段代码所在模块路径（按你截图推断）
# src/reranker_training/data/data_preprocessing.py
from reranker_training.data import data_preprocessing as dp


# ----------------------------
# Tokenizer stubs (no HF needed)
# ----------------------------

class RecordingTokenizer:
    """
    Minimal tokenizer stub:
    - __call__(q, d, ...) returns dict-like encoding
    - pad(list_of_encodings, return_tensors="pt") returns padded tensors
    Also records last __call__ kwargs for assertions.
    """

    def __init__(self, include_token_type_ids: bool = True):
        self.include_token_type_ids = include_token_type_ids
        self.last_call = None

    def __call__(self, q, d, *, max_length, truncation, padding):
        # record args for testing
        self.last_call = {
            "q": q,
            "d": d,
            "max_length": max_length,
            "truncation": truncation,
            "padding": padding,
        }

        # fake "tokenization": length depends on doc length (bounded)
        # create deterministic, variable-length sequences to test dynamic padding
        doc_len = max(1, min(8, len(str(d)) % 8 + 1))
        input_ids = list(range(1, doc_len + 1))
        attention_mask = [1] * doc_len

        enc = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.include_token_type_ids:
            enc["token_type_ids"] = [0] * doc_len
        return enc

    def pad(self, encodings, *, return_tensors="pt"):
        # encodings: List[Dict[str, List[int]]]
        max_len = max(len(e["input_ids"]) for e in encodings) if encodings else 0

        def pad_1d(xs, pad_value=0):
            return xs + [pad_value] * (max_len - len(xs))

        input_ids = [pad_1d(e["input_ids"], 0) for e in encodings]
        attn = [pad_1d(e["attention_mask"], 0) for e in encodings]

        out = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

        # only include token_type_ids if present in the first encoding
        if encodings and "token_type_ids" in encodings[0]:
            tti = [pad_1d(e["token_type_ids"], 0) for e in encodings]
            out["token_type_ids"] = torch.tensor(tti, dtype=torch.long)

        return out


# ----------------------------
# Helper: monkeypatch read_jsonl
# ----------------------------

def _patch_read_jsonl(monkeypatch, rows, capture: dict | None = None):
    """
    Patch dp.read_jsonl to yield `rows`.
    Optionally capture "store/path/on_error" used by the caller.
    """
    def fake_read_jsonl(store, path, on_error=None):
        if capture is not None:
            capture["store"] = store
            capture["path"] = path
            capture["on_error"] = on_error
        for r in rows:
            yield r

    monkeypatch.setattr(dp, "read_jsonl", fake_read_jsonl, raising=True)


# ----------------------------
# load_pairs_for_epoch tests
# ----------------------------

def test_load_pairs_for_epoch_happy(monkeypatch):
    cap = {}
    rows = [
        {
            "query": {"query_text": "q1"},
            "positive": {"title": "T", "chunk_text": "pos chunk"},
            "negative": {"chunk_text": "neg chunk"},
        },
        {
            "query": {"query_text": "q2"},
            "positive": {"chunk_text": "pos2"},
            "negative": {"title": "N", "chunk_text": "neg2"},
        },
    ]
    _patch_read_jsonl(monkeypatch, rows, capture=cap)

    store = object()
    items = dp.load_pairs_for_epoch(
        store=store,
        base="outputs",
        train_pair_path_tpl="processed/train_pair_epoch_{epoch}.jsonl",
        epoch=3,
    )

    assert cap["store"] is store
    assert cap["path"] == "outputs/processed/train_pair_epoch_3.jsonl"
    assert len(items) == 2
    assert items[0].query_text == "q1"
    assert items[0].pos_text == "T\npos chunk"  # title + '\n' + chunk
    assert items[0].neg_text == "neg chunk"
    assert items[1].neg_text == "N\nneg2"


def test_load_pairs_for_epoch_empty_raises(monkeypatch):
    _patch_read_jsonl(monkeypatch, rows=[])
    with pytest.raises(ValueError, match="Loaded 0 pairs"):
        dp.load_pairs_for_epoch(
            store=object(),
            base="b",
            train_pair_path_tpl="x_{epoch}.jsonl",
            epoch=0,
        )


def test_load_pairs_for_epoch_invalid_schema_raises(monkeypatch):
    rows = [
        {"query": "not a dict", "positive": {"chunk_text": "p"}, "negative": {"chunk_text": "n"}}
    ]
    _patch_read_jsonl(monkeypatch, rows=rows)

    with pytest.raises(ValueError, match="query.*must be dict"):
        dp.load_pairs_for_epoch(
            store=object(),
            base="b",
            train_pair_path_tpl="x_{epoch}.jsonl",
            epoch=1,
        )


def test_load_pairs_for_epoch_missing_query_text_raises(monkeypatch):
    rows = [
        {"query": {"query_text": ""}, "positive": {"chunk_text": "p"}, "negative": {"chunk_text": "n"}}
    ]
    _patch_read_jsonl(monkeypatch, rows=rows)

    with pytest.raises(ValueError, match="query.query_text is missing/empty"):
        dp.load_pairs_for_epoch(
            store=object(),
            base="b",
            train_pair_path_tpl="x_{epoch}.jsonl",
            epoch=1,
        )


# ----------------------------
# Dataset tests
# ----------------------------

def test_dataset_getitem_calls_tokenizer_with_expected_kwargs():
    tok = RecordingTokenizer(include_token_type_ids=True)
    items = [
        dp.PairwiseItem(query_text="q", pos_text="pos", neg_text="neg"),
    ]
    ds = dp.CrossEncoderPairwiseDataset(items=items, tokenizer=tok, max_length=128, pad_to_max_length=False)

    out = ds[0]
    assert "pos" in out and "neg" in out

    # last call corresponds to neg (since __getitem__ encodes pos then neg)
    assert tok.last_call["max_length"] == 128
    assert tok.last_call["truncation"] == "only_second"
    assert tok.last_call["padding"] is False


def test_dataset_pad_to_max_length_sets_padding_max_length():
    tok = RecordingTokenizer(include_token_type_ids=False)
    items = [dp.PairwiseItem(query_text="q", pos_text="pos", neg_text="neg")]
    ds = dp.CrossEncoderPairwiseDataset(items=items, tokenizer=tok, max_length=64, pad_to_max_length=True)

    _ = ds[0]
    assert tok.last_call["padding"] == "max_length"


# ----------------------------
# Collator tests
# ----------------------------

def test_pairwise_collator_outputs_expected_keys_with_token_type_ids():
    tok = RecordingTokenizer(include_token_type_ids=True)
    collate = dp.PairwiseCollator(tok)

    features = [
        {
            "pos": {"input_ids": [1, 2], "attention_mask": [1, 1], "token_type_ids": [0, 0]},
            "neg": {"input_ids": [3], "attention_mask": [1], "token_type_ids": [0]},
        },
        {
            "pos": {"input_ids": [4], "attention_mask": [1], "token_type_ids": [0]},
            "neg": {"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "token_type_ids": [0, 0, 0]},
        },
    ]

    batch = collate(features)

    # required
    assert set(batch.keys()) >= {
        "pos_input_ids",
        "pos_attention_mask",
        "neg_input_ids",
        "neg_attention_mask",
    }
    # optional present
    assert "pos_token_type_ids" in batch
    assert "neg_token_type_ids" in batch

    assert batch["pos_input_ids"].dtype == torch.long
    assert batch["neg_input_ids"].shape[0] == 2  # batch size


def test_pairwise_collator_omits_token_type_ids_when_not_present():
    tok = RecordingTokenizer(include_token_type_ids=False)
    collate = dp.PairwiseCollator(tok)

    features = [
        {"pos": {"input_ids": [1], "attention_mask": [1]}, "neg": {"input_ids": [2, 3], "attention_mask": [1, 1]}},
    ]

    batch = collate(features)
    assert "pos_token_type_ids" not in batch
    assert "neg_token_type_ids" not in batch


# ----------------------------
# load_valid_query_packs tests
# ----------------------------

def test_load_valid_query_packs_skips_no_positive(monkeypatch):
    rows = [
        {
            "query": {"query_text": "q"},
            "positives": [],
            "negatives": [{"chunk_text": "n"}],
        }
    ]
    _patch_read_jsonl(monkeypatch, rows=rows)

    with pytest.raises(ValueError, match="Loaded 0 valid query packs"):
        dp.load_valid_query_packs(store=object(), base="b", valid_path="v.jsonl")


def test_load_valid_query_packs_applies_negative_cap_and_is_deterministic(monkeypatch):
    cap = {}
    rows = [
        {
            "query": {"query_text": "q"},
            "positives": [{"title": "P", "chunk_text": "pos"}],
            "negatives": [
                {"chunk_text": "n0"},
                {"chunk_text": "n1"},
                {"chunk_text": "n2"},
                {"chunk_text": "n3"},
                {"chunk_text": "n4"},
            ],
        }
    ]
    _patch_read_jsonl(monkeypatch, rows=rows, capture=cap)

    packs1 = dp.load_valid_query_packs(
        store=object(),
        base="outputs",
        valid_path="processed/valid.jsonl",
        max_negatives_per_query=2,
        seed=123,
    )
    packs2 = dp.load_valid_query_packs(
        store=object(),
        base="outputs",
        valid_path="processed/valid.jsonl",
        max_negatives_per_query=2,
        seed=123,
    )

    assert cap["path"] == "outputs/processed/valid.jsonl"
    assert len(packs1) == 1
    p1 = packs1[0]
    assert p1.query_text == "q"
    assert len(p1.doc_texts) == 1 + 2  # 1 pos + capped negs
    assert sum(p1.labels) == 1
    assert p1.doc_texts[0] == "P\npos"  # title included

    # deterministic sampling given same seed
    assert packs1[0].doc_texts == packs2[0].doc_texts
    assert packs1[0].labels == packs2[0].labels
