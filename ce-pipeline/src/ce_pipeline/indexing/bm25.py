from __future__ import annotations
import pickle
from dataclasses import dataclass
from typing import List, Tuple

from rank_bm25 import BM25Okapi
from ..stores.base import Store


def _simple_tokenize(text: str) -> List[str]:
    # 仍然保留简单版本；建议后面换成更强的 tokenizer
    return [t for t in (text or "").lower().split() if t]


@dataclass
class BM25Artifact:
    bm25: BM25Okapi
    doc_ids: List[str]  # 与 corpus 顺序一致（例如 chunk_id）


def build_bm25_index(texts: List[str]) -> BM25Okapi:
    corpus = [_simple_tokenize(t) for t in texts]
    return BM25Okapi(corpus)


def save_bm25(store: Store, path: str, bm25: BM25Okapi, doc_ids: List[str]) -> None:
    artifact = BM25Artifact(bm25=bm25, doc_ids=doc_ids)
    store.write_bytes(path, pickle.dumps(artifact))


def load_bm25(store: Store, path: str) -> BM25Artifact:
    data = store.read_bytes(path)
    return pickle.loads(data)

def get_scores(self, tokenized_query):
    return self.bm25.get_scores(tokenized_query)