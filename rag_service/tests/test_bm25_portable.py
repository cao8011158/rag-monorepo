import pickle
import pytest


# 按你的 src-layout import
from rag_service.common.bm25_portable import (
    load_bm25_portable,
    tokenize_query,
    _TOKENIZER_ID_WHOOSH,
)


# -------------------------
# helpers
# -------------------------

def _payload_bytes():
    payload = {
        "tokenizer": _TOKENIZER_ID_WHOOSH,
        "corpus_tokens": [
            ["cmu", "school"],
            ["pittsburgh", "city"],
        ],
        "doc_ids": ["a", "b"],
    }
    return pickle.dumps(payload)


# -------------------------
# smoke tests
# -------------------------

def test_load_smoke():
    bm25, ids, tok = load_bm25_portable(_payload_bytes())

    assert tok == _TOKENIZER_ID_WHOOSH
    assert ids == ["a", "b"]

    # minimal sanity call
    scores = bm25.get_scores(["cmu"])
    assert len(scores) == 2


def test_tokenize_smoke():
    tokens = tokenize_query(_TOKENIZER_ID_WHOOSH, "Running runs")

    assert isinstance(tokens, list)
    assert len(tokens) > 0


def test_bad_tokenizer_id():
    with pytest.raises(KeyError):
        tokenize_query("bad", "text")
