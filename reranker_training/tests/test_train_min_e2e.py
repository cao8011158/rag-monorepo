# tests/test_train_min_real_slow.py
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Union

import pytest
import torch

from transformers import TrainingArguments

from reranker_training.data.data_preprocessing import (
    CrossEncoderPairwiseDataset,
    PairwiseCollator,
    PairwiseItem,
)
from reranker_training.trainer import PairwiseTrainerWithRankingEval, EvalPack
from reranker_training.modeling import CrossEncoderReranker


# -------------------------
# Tiny tokenizer
# - supports single encode: tokenizer(str, str, ...)
# - supports batch encode : tokenizer(List[str], List[str], padding=True, return_tensors="pt")
# - supports pad([...], return_tensors="pt")
# -------------------------
class TinyTokenizer:
    def __init__(self, *, with_token_type_ids: bool) -> None:
        self.with_token_type_ids = bool(with_token_type_ids)
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self._vocab: Dict[str, int] = {"<pad>": 0, "<eos>": 1, "[CLS]": 2, "[SEP]": 3}

    def _id(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = len(self._vocab)
        return self._vocab[token]

    def _encode_one(self, q: str, d: str, *, max_length: int) -> Dict[str, Any]:
        q_tokens = [t for t in (q or "").split() if t]
        d_tokens = [t for t in (d or "").split() if t]

        ids = [self._id("[CLS]")]
        ids += [self._id(t) for t in q_tokens]
        ids.append(self._id("[SEP]"))
        ids += [self._id(t) for t in d_tokens]
        ids.append(self._id("[SEP]"))

        ids = ids[: int(max_length)]
        attn = [1] * len(ids)

        out: Dict[str, Any] = {"input_ids": ids, "attention_mask": attn}

        if self.with_token_type_ids:
            tt = [0] * len(ids)
            # after first [SEP], mark doc part as 1
            try:
                sep_idx = ids.index(self._id("[SEP]"))
            except ValueError:
                sep_idx = len(ids)
            for i in range(sep_idx + 1, len(tt)):
                tt[i] = 1
            out["token_type_ids"] = tt

        return out

    def __call__(
        self,
        q: Union[str, Sequence[str]],
        d: Union[str, Sequence[str]],
        *,
        max_length: int = 32,
        truncation: str = "only_second",
        padding: bool | str = False,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Any]:
        # --- normalize to batch ---
        if isinstance(q, (list, tuple)):
            qs = list(q)
        else:
            qs = [q]
        if isinstance(d, (list, tuple)):
            ds = list(d)
        else:
            ds = [d]

        if len(qs) != len(ds):
            raise ValueError(f"Expected len(q)==len(d) for batch encode, got {len(qs)} vs {len(ds)}")

        rows = [self._encode_one(qs[i], ds[i], max_length=max_length) for i in range(len(qs))]

        # --- padding semantics ---
        # score_query_docs uses padding=True, return_tensors="pt"
        if padding is True:
            enc = self.pad(rows, return_tensors="pt")
            if return_tensors == "pt":
                return enc
            # if caller wanted python lists (not used in this project path)
            return {k: v.tolist() for k, v in enc.items()}

        if padding == "max_length":
            pad_id = self._id("<pad>")
            for r in rows:
                while len(r["input_ids"]) < int(max_length):
                    r["input_ids"].append(pad_id)
                    r["attention_mask"].append(0)
                    if "token_type_ids" in r:
                        r["token_type_ids"].append(0)

        # --- return tensors or python objects ---
        if return_tensors == "pt":
            # For simplicity, always return a padded batch if tensors requested.
            return self.pad(rows, return_tensors="pt")

        # HF-like: if single example, return single dict of lists
        if len(rows) == 1:
            return rows[0]

        # otherwise return batch lists (rarely used)
        out: Dict[str, Any] = {
            "input_ids": [r["input_ids"] for r in rows],
            "attention_mask": [r["attention_mask"] for r in rows],
        }
        if self.with_token_type_ids:
            out["token_type_ids"] = [r["token_type_ids"] for r in rows]
        return out

    def pad(self, features: List[Dict[str, Any]], *, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self._id("<pad>")

        def _pad_list(x: List[int], pad_value: int) -> List[int]:
            return x + [pad_value] * (max_len - len(x))

        batch: Dict[str, torch.Tensor] = {}
        batch["input_ids"] = torch.tensor([_pad_list(f["input_ids"], pad_id) for f in features], dtype=torch.long)
        batch["attention_mask"] = torch.tensor([_pad_list(f["attention_mask"], 0) for f in features], dtype=torch.long)

        if self.with_token_type_ids and "token_type_ids" in features[0]:
            batch["token_type_ids"] = torch.tensor([_pad_list(f["token_type_ids"], 0) for f in features], dtype=torch.long)

        return batch


# -------------------------
# Tiny HF-like base model that returns .logits
# -------------------------
class TinySeqClsModel(torch.nn.Module):
    """
    Accepts input_ids/attention_mask(/token_type_ids) and returns an object with .logits
    logits shape: [B, 1]
    """

    def __init__(self, vocab_size: int, hidden_size: int = 32) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, hidden_size)
        self.proj = torch.nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ):
        x = self.emb(input_ids)  # [B, L, H]
        mask = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (x * mask).sum(dim=1) / denom  # [B, H]
        logits = self.proj(pooled)  # [B, 1]

        class _Out:
            def __init__(self, logits: torch.Tensor) -> None:
                self.logits = logits

        return _Out(logits)


@pytest.mark.slow
def test_min_real_train_numeric_health_and_eval(tmp_path):
    """
    Mini E2E:
      - real Trainer.train() for a few steps
      - loss finite
      - params update
      - evaluate() override returns eval_ndcg@10 / eval_mrr@10
    """
    tok = TinyTokenizer(with_token_type_ids=True)

    items = [
        PairwiseItem(query_text="q1", pos_text="good doc", neg_text="bad doc"),
        PairwiseItem(query_text="q1", pos_text="good doc 2", neg_text="bad doc 2"),
        PairwiseItem(query_text="q2", pos_text="relevant", neg_text="irrelevant"),
        PairwiseItem(query_text="q2", pos_text="relevant 2", neg_text="irrelevant 2"),
    ]

    train_ds = CrossEncoderPairwiseDataset(items=items, tokenizer=tok, max_length=32)
    collator = PairwiseCollator(tok)

    valid_packs = [
        EvalPack(query_text="q1", doc_texts=["good doc", "bad doc"], labels=[1, 0]),
        EvalPack(query_text="q2", doc_texts=["irrelevant", "relevant"], labels=[0, 1]),
    ]

    base = TinySeqClsModel(vocab_size=2000, hidden_size=32)
    model = CrossEncoderReranker(base)

    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    out_dir = tmp_path / "out"
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        max_steps=5,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=2,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        bf16=False,
        fp16=False,
    )

    trainer = PairwiseTrainerWithRankingEval(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=train_ds,  # not used; evaluate overridden
        data_collator=collator,
        processing_class=tok,
        valid_packs=valid_packs,
        max_length=32,
        ndcg_k=10,
        mrr_k=10,
        infer_batch_size=8,
        callbacks=[],
    )

    trainer.train()

    # losses are finite
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    assert len(losses) > 0
    assert all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in losses)

    # parameters updated
    after = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    assert set(before.keys()) == set(after.keys())
    changed = any(not torch.allclose(before[n], after[n]) for n in before.keys())
    assert changed, "Expected at least one trainable parameter to change after training."

    # evaluate override yields ranking metrics
    metrics = trainer.evaluate()
    assert "eval_ndcg@10" in metrics
    assert "eval_mrr@10" in metrics
    assert "eval_num_queries" in metrics
    assert 0.0 <= float(metrics["eval_ndcg@10"]) <= 1.0
    assert 0.0 <= float(metrics["eval_mrr@10"]) <= 1.0
