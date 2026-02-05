from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from typing import Any, Dict, List, Sequence
from torch.utils.data import Dataset
from dataclasses import dataclass

from typing import Any, List

# IO
from reranker_training.io.jsonl import read_jsonl



# ============================================================
# Utils
# ============================================================

def _safe_str(x: Any) -> str:
    return x if isinstance(x, str) else ""


def _posix_join(base: str, name: str) -> str:
    base = str(base).rstrip("/")
    name = str(name).lstrip("/")
    return f"{base}/{name}" if base else name


def _query_to_text(q: Dict[str, Any]) -> str:
    qt = _safe_str(q.get("query_text")).strip()
    if not qt:
        raise ValueError("Pair/Pack row: query.query_text is missing/empty")
    return qt


def _chunkdoc_to_text(doc: Dict[str, Any]) -> str:
    """
    ChunkDoc -> text for cross-encoder.
    Uses: optional title + '\\n' + chunk_text
    """
    title = _safe_str(doc.get("title")).strip()
    chunk_text = _safe_str(doc.get("chunk_text")).strip()
    if not chunk_text:
        raise ValueError("Pair/Pack row: ChunkDoc.chunk_text is missing/empty")
    return (title + "\n" + chunk_text).strip() if title else chunk_text


def _fail(msg: str) -> None:
    raise ValueError(msg)


# ============================================================
# Train Pair schema -> training item
# 
# Reads:
#   outputs.files.base + outputs.files.train_pair_path.format(epoch=epoch)
# 
# Pair schema per line:
# {
#   "query": Query,
#   "positive": ChunkDoc,
#   "negative": ChunkDoc,
# }
# output:
#   List[PairwiseItem]
# ============================================================

@dataclass
class PairwiseItem:
    query_text: str
    pos_text: str
    neg_text: str

def load_pairs_for_epoch(
    *,
    store: Any,
    base: str,
    train_pair_path_tpl: str,
    epoch: int,
) -> List[PairwiseItem]:
    """
    
    """
    rel = str(train_pair_path_tpl).format(epoch=int(epoch))
    path = _posix_join(base, rel)

    items: List[PairwiseItem] = []
    for row in read_jsonl(store, path, on_error=None):  # fail-fast
        if not isinstance(row, dict):
            continue

        q = row.get("query")
        p = row.get("positive")
        n = row.get("negative")

        if not isinstance(q, dict):
            _fail(f"Pair row invalid: 'query' must be dict. keys={list(row.keys())}")
        if not isinstance(p, dict):
            _fail(f"Pair row invalid: 'positive' must be dict. keys={list(row.keys())}")
        if not isinstance(n, dict):
            _fail(f"Pair row invalid: 'negative' must be dict. keys={list(row.keys())}")

        items.append(
            PairwiseItem(
                query_text=_query_to_text(q),
                pos_text=_chunkdoc_to_text(p),
                neg_text=_chunkdoc_to_text(n),
            )
        )

    if not items:
        raise ValueError(f"Loaded 0 pairs from {path}")
    return items


# ============================================================
#        Dataset 
# Pairwise dataset for cross-encoder reranker training.
# Each dataset item represents ONE training pair constructed from:
#     query_text
#     pos_text  (relevant document chunk)
#     neg_text  (non-relevant document chunk)

# ------------------------------------------------------------
# Output schema per __getitem__:
# ------------------------------------------------------------
# {
#     "pos": TokenizerEncoding(q, d+)
#     "neg": TokenizerEncoding(q, d-)
# }

# Where each encoding contains:
#     input_ids
#     attention_mask
#     token_type_ids (optional, model-dependent)

# ------------------------------------------------------------
# Truncation policy  truncation="only_second"
# ------------------------------------------------------------

# ------------------------------------------------------------
# Padding strategy  pad_to_max_length = False  (default)
# ------------------------------------------------------------

# ============================================================

class CrossEncoderPairwiseDataset(Dataset):
    """
    Each example returns:
      pos: tokenized (q, d+)
      neg: tokenized (q, d-)

    Truncation policy:
      truncation="only_second" => keep query, truncate doc
    """

    def __init__(
        self,
        *,
        items: Sequence[PairwiseItem],
        tokenizer: Any,
        max_length: int,
        pad_to_max_length: bool = False,
    ) -> None:
        self.items = list(items)
        self.tok = tokenizer
        self.max_length = int(max_length)
        self.pad_to_max_length = bool(pad_to_max_length)

    def __len__(self) -> int:
        return len(self.items)

    def _enc(self, q: str, d: str) -> Dict[str, Any]:
        padding = "max_length" if self.pad_to_max_length else False
        return self.tok(
            q,
            d,
            max_length=self.max_length,
            truncation="only_second",  # keep query, cut doc
            padding=padding,           # False => dynamic padding in collator
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        return {
            "pos": self._enc(it.query_text, it.pos_text),
            "neg": self._enc(it.query_text, it.neg_text),
        }
    

# ============================================================
# Collator for pairwise cross-encoder reranker training.

#     This component receives a list of tokenized samples produced by
#     CrossEncoderPairwiseDataset and converts them into a padded batch
#     of PyTorch tensors suitable for model forward pass.

#     ------------------------------------------------------------
#     Input format (from Dataset)
#     ------------------------------------------------------------
#     features: List[
#         {
#             "pos": TokenizerEncoding(q, d+)
#             "neg": TokenizerEncoding(q, d-)
#         }
#     ------------------------------------------------------------
#     Output format (Trainer-ready)
#     ------------------------------------------------------------
#     {
#         pos_input_ids
#         pos_attention_mask
#         neg_input_ids
#         neg_attention_mask
#         pos_token_type_ids (optional)
#         neg_token_type_ids (optional)
#     }
#     ------------------------------------------------------------
#     Dynamic Padding
#     ------------------------------------------------------------
#     tokenizer.pad(...)    Pads sequences ONLY to the longest sequence length in this batch.

# ============================================================

class PairwiseCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tok = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        pos = self.tok.pad([f["pos"] for f in features], return_tensors="pt")
        neg = self.tok.pad([f["neg"] for f in features], return_tensors="pt")

        batch: Dict[str, Any] = {
            "pos_input_ids": pos["input_ids"],
            "pos_attention_mask": pos["attention_mask"],
            "neg_input_ids": neg["input_ids"],
            "neg_attention_mask": neg["attention_mask"],
        }

        # optional (some models/tokenizers produce this)
        if "token_type_ids" in pos:
            batch["pos_token_type_ids"] = pos["token_type_ids"]
        if "token_type_ids" in neg:
            batch["neg_token_type_ids"] = neg["token_type_ids"]

        return batch
    

# ============================================================
# Valid QueryPack schema -> eval pack
# ============================================================

@dataclass
class EvalPack:
    query_text: str
    doc_texts: List[str]   # candidates = positives + negatives
    labels: List[int]      # 1 for positive, 0 for negative


def load_valid_query_packs(
    *,
    store: Any,
    base: str,
    valid_path: str,
    max_negatives_per_query: Optional[int] = None,
    seed: int = 42,
) -> List[EvalPack]:
    """
    Reads:
      outputs.files.base + outputs.files.valid_path

    QueryPack per line:
    {
      "query": Query,
      "positives": List[ChunkDoc],
      "negatives": List[ChunkDoc],
      ...
    }

    Optional: max_negatives_per_query to cap very large packs for faster eval.
    """
    path = _posix_join(base, valid_path)
    packs: List[EvalPack] = []

    rnd = random.Random(int(seed))

    for row in read_jsonl(store, path, on_error=None):  # fail-fast
        if not isinstance(row, dict):
            continue

        q = row.get("query")
        pos = row.get("positives") or []
        neg = row.get("negatives") or []

        if not isinstance(q, dict):
            _fail("Valid QueryPack row: query must be dict")
        if not isinstance(pos, list) or not isinstance(neg, list):
            _fail("Valid QueryPack row: positives/negatives must be lists")
        if not isinstance(neg, list) or not isinstance(neg, list):
            _fail("Valid QueryPack row: positives/negatives must be lists")

        query_text = _query_to_text(q)

        # optional negative cap (random sample to keep unbiased-ish)
        neg_docs: List[Dict[str, Any]] = [d for d in neg if isinstance(d, dict)]
        if isinstance(max_negatives_per_query, int) and max_negatives_per_query > 0:
            if len(neg_docs) > max_negatives_per_query:
                neg_docs = rnd.sample(neg_docs, max_negatives_per_query)

        doc_texts: List[str] = []
        labels: List[int] = []

        for d in pos:
            if isinstance(d, dict):
                doc_texts.append(_chunkdoc_to_text(d))
                labels.append(1)

        for d in neg_docs:
            if isinstance(d, dict):
                doc_texts.append(_chunkdoc_to_text(d))
                labels.append(0)

        # Need at least 1 positive for NDCG/MRR to be meaningful
        if sum(labels) == 0 or len(doc_texts) == 0:
            continue

        packs.append(EvalPack(query_text=query_text, doc_texts=doc_texts, labels=labels))

    if not packs:
        raise ValueError(f"Loaded 0 valid query packs from {path}")
    return packs