from __future__ import annotations

import pickle
from typing import List, Tuple, Dict, Any

from rank_bm25 import BM25Okapi
from whoosh.analysis import StemmingAnalyzer

from ..stores.base import Store


# -----------------------------
# Tokenizer (Whoosh)
# -----------------------------
# StemmingAnalyzer = RegexTokenizer + LowercaseFilter + StopFilter + StemFilter
_ANALYZER = StemmingAnalyzer()
_TOKENIZER_ID = "whoosh_stemming_v1"


def _whoosh_tokenize(text: str) -> List[str]:
    # Each token yielded has .text
    return [t.text for t in _ANALYZER(text or "")]


# -----------------------------
# Build
# -----------------------------
def build_bm25_tokens(texts: List[str]) -> List[List[str]]:
    return [_whoosh_tokenize(t) for t in texts]


# -----------------------------
# Save (Portable)
# -----------------------------
def save_bm25(
    store: Store,
    path: str,
    corpus_tokens: List[List[str]],
    doc_ids: List[str],
) -> None:
    payload: Dict[str, Any] = {
        "version": 1,
        "tokenizer": _TOKENIZER_ID,
        "doc_ids": doc_ids,
        "corpus_tokens": corpus_tokens,
    }
    store.write_bytes(path, pickle.dumps(payload))


# -----------------------------
# Load (Rebuild Object)
# -----------------------------
def load_bm25(store: Store, path: str) -> Tuple[BM25Okapi, List[str]]:
    payload = pickle.loads(store.read_bytes(path))

    tok = payload.get("tokenizer")
    if tok is not None and tok != _TOKENIZER_ID:
        raise ValueError(f"BM25 tokenizer mismatch: expected={_TOKENIZER_ID} got={tok}")

    bm25 = BM25Okapi(payload["corpus_tokens"])
    return bm25, payload["doc_ids"]
