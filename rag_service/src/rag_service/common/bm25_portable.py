# rag_server/retrieval/bm25_portable.py
from __future__ import annotations

import pickle
from typing import Any, Dict, List, Tuple, Callable

from rank_bm25 import BM25Okapi
from whoosh.analysis import StemmingAnalyzer



TokenizerFn = Callable[[str], List[str]]

_TOKENIZER_ID_WHOOSH = "whoosh_stemming_v1"


def _whoosh_tokenize(text: str) -> List[str]:
    if StemmingAnalyzer is None:
        raise RuntimeError(
            "BM25 tokenizer requires whoosh, but whoosh is not installed. "
            "Install whoosh or rebuild bm25 with a different tokenizer."
        )
    analyzer = StemmingAnalyzer()
    return [t.text for t in analyzer(text or "")]


TOKENIZERS: Dict[str, TokenizerFn] = {
    _TOKENIZER_ID_WHOOSH: _whoosh_tokenize,
}


def load_bm25_portable(payload_bytes: bytes) -> Tuple[BM25Okapi, List[str], str]:
    payload = pickle.loads(payload_bytes)

    tok_id = payload.get("tokenizer")
    if not tok_id:
        raise ValueError("BM25 payload missing tokenizer id")

    if tok_id not in TOKENIZERS:
        raise ValueError(f"Unknown BM25 tokenizer id: {tok_id}")

    bm25 = BM25Okapi(payload["corpus_tokens"])
    doc_ids = [str(x) for x in payload["doc_ids"]]
    return bm25, doc_ids, tok_id


def tokenize_query(tok_id: str, text: str) -> List[str]:
    return TOKENIZERS[tok_id](text)
