from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .io_utils import read_jsonl


@dataclass
class PairSample:
    query_text: str
    pos_text: str
    neg_texts: List[str]  # pool


def load_pair_samples(path: str) -> List[PairSample]:
    out: List[PairSample] = []
    for obj in read_jsonl(path):
        q = str(obj["query_text"])
        pos = str(obj["positive"]["text"])
        negs = [str(x["text"]) for x in (obj.get("negatives") or [])]
        if not negs:
            # skip if no negatives (can't train pairwise)
            continue
        out.append(PairSample(query_text=q, pos_text=pos, neg_texts=negs))
    return out


class PairwiseResampleDataset(Dataset):
    """
    Each epoch, call .set_epoch(epoch) to change negative sampling seed.
    Returns one training item per query (or more if you want to upsample).
    Item: tokenized (q,pos) and (q,neg) pair.
    """

    def __init__(
        self,
        samples: List[PairSample],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        negs_per_query_per_epoch: int = 1,
        mix_random_negs: bool = True,
        random_neg_ratio: float = 0.25,
        seed: int = 42,
    ) -> None:
        self.samples = samples
        self.tok = tokenizer
        self.max_length = max_length
        self.negs_per_query_per_epoch = max(1, int(negs_per_query_per_epoch))
        self.mix_random_negs = bool(mix_random_negs)
        self.random_neg_ratio = float(random_neg_ratio)
        self.base_seed = int(seed)
        self.epoch = 0

        # Pre-build index to allow stable length
        self._index: List[Tuple[int, int]] = []
        for i in range(len(self.samples)):
            for j in range(self.negs_per_query_per_epoch):
                self._index.append((i, j))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._index)

    def _sample_neg(self, neg_pool: List[str], rng: random.Random) -> str:
        if len(neg_pool) == 1:
            return neg_pool[0]
        # If you already have hard negatives in the pool, uniform sampling is OK.
        # Optionally mix random negatives by sampling from the tail more often.
        if self.mix_random_negs and rng.random() < self.random_neg_ratio:
            # sample from the entire pool (random-ish)
            return rng.choice(neg_pool)
        # prefer harder: sample more from the front (if your pool is ranked hard->easy)
        # Without rank info, just uniform:
        return rng.choice(neg_pool)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        qi, rep = self._index[idx]
        s = self.samples[qi]

        rng = random.Random(self.base_seed + self.epoch * 1000003 + qi * 97 + rep)
        neg = self._sample_neg(s.neg_texts, rng)

        # Encode (query, doc) for cross-encoder
        pos_enc = self.tok(
            s.query_text,
            s.pos_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        neg_enc = self.tok(
            s.query_text,
            neg,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {
            "pos": pos_enc,
            "neg": neg_enc,
        }


def collate_pairwise(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """
    Collate pos/neg pairs into padded tensors.
    Output:
      input_ids_pos, attention_mask_pos, input_ids_neg, attention_mask_neg
    """
    def _pad(key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids = [b[key]["input_ids"] for b in batch]
        attn = [b[key]["attention_mask"] for b in batch]
        maxlen = max(len(x) for x in input_ids)

        ids_t = torch.full((len(batch), maxlen), pad_token_id, dtype=torch.long)
        attn_t = torch.zeros((len(batch), maxlen), dtype=torch.long)
        for i, (ids, am) in enumerate(zip(input_ids, attn)):
            ids_t[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn_t[i, : len(am)] = torch.tensor(am, dtype=torch.long)
        return ids_t, attn_t

    ids_pos, attn_pos = _pad("pos")
    ids_neg, attn_neg = _pad("neg")
    return {
        "input_ids_pos": ids_pos,
        "attention_mask_pos": attn_pos,
        "input_ids_neg": ids_neg,
        "attention_mask_neg": attn_neg,
    }
