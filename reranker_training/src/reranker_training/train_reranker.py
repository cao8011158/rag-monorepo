from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)

from reranker_training.settings import load_settings
from reranker_training.stores.registry import build_store_registry
from reranker_training.io.jsonl import read_jsonl


# -----------------------------
# Model wrapper (you already have)
# -----------------------------
from reranker_training.models.cross_encoder import CrossEncoderReranker  # adjust import to your path


# -----------------------------
# Utils
# -----------------------------
def _safe_str(x: Any) -> str:
    return x if isinstance(x, str) else ""


def _posix_join(base: str, name: str) -> str:
    base = str(base).rstrip("/")
    name = str(name).lstrip("/")
    return f"{base}/{name}" if base else name


# -----------------------------
# Build chunk_id -> text lookup
# -----------------------------
def build_chunk_text_lookup(
    *,
    store: Any,
    chunks_path: str,
) -> Dict[str, str]:
    """
    Reads chunks.jsonl and returns mapping:
      chunk_id -> (title + "\\n" + chunk_text) or chunk_text
    """
    lookup: Dict[str, str] = {}
    for row in read_jsonl(store, chunks_path, on_error=None):  # fail-fast by default
        cid = row.get("chunk_id")
        txt = row.get("chunk_text")
        title = row.get("title")

        if not isinstance(cid, str) or not cid:
            raise KeyError("chunks.jsonl: missing/invalid chunk_id")
        if not isinstance(txt, str) or not txt.strip():
            raise KeyError(f"chunks.jsonl: missing/invalid chunk_text for chunk_id={cid}")

        title_s = _safe_str(title).strip()
        if title_s:
            full = (title_s + "\n" + txt).strip()
        else:
            full = txt.strip()

        lookup[cid] = full
    return lookup


# -----------------------------
# Pairwise items (q, d+, d-)
# -----------------------------
@dataclass
class PairwiseItem:
    query_text: str
    pos_text: str
    neg_text: str


def build_pairwise_items_from_querypack(
    *,
    store: Any,
    querypack_path: str,
    chunk_lookup: Dict[str, str],
    hard_negative_per_positive: int,
    random_negative_per_positive: int,
    seed: int,
) -> List[PairwiseItem]:
    """
    Expects each row in query_pack.jsonl contains:
      - query_text or query_text_norm
      - positives: [chunk_id, ...]
      - negatives: [chunk_id, ...]   (you said negatives can be empty without error)
    Creates pairwise training items by sampling negatives for each positive.
    """
    rng = random.Random(int(seed))
    items: List[PairwiseItem] = []

    for qp in read_jsonl(store, querypack_path, on_error=None):
        q = _safe_str(qp.get("query_text_norm") or qp.get("query_text")).strip()
        if not q:
            raise KeyError("query_pack: missing query_text/query_text_norm")

        pos_ids = qp.get("positives") or []
        neg_ids = qp.get("negatives") or []

        if not isinstance(pos_ids, list):
            raise TypeError("query_pack: positives must be a list")
        if not isinstance(neg_ids, list):
            raise TypeError("query_pack: negatives must be a list")

        # You requested: positives empty => error
        if len(pos_ids) == 0:
            raise ValueError("query_pack: positives is empty (this should error by your rule)")

        pos_ids = [x for x in pos_ids if isinstance(x, str) and x in chunk_lookup]
        neg_ids = [x for x in neg_ids if isinstance(x, str) and x in chunk_lookup]

        # Build a pool for random negatives:
        # - use given negatives first (if any)
        # - else fallback to sampling from global chunk_lookup excluding positives
        global_candidates = [cid for cid in chunk_lookup.keys() if cid not in set(pos_ids)]
        if len(global_candidates) == 0:
            raise ValueError("No available chunks to sample random negatives (all are positives?)")

        for pid in pos_ids:
            pos_text = chunk_lookup[pid]

            # hard negatives: sample from neg_ids (if provided)
            if hard_negative_per_positive > 0 and len(neg_ids) > 0:
                hard_k = min(int(hard_negative_per_positive), len(neg_ids))
                hard_samples = rng.sample(neg_ids, k=hard_k)
                for nid in hard_samples:
                    items.append(PairwiseItem(query_text=q, pos_text=pos_text, neg_text=chunk_lookup[nid]))

            # random negatives: sample from global (exclude positives; also avoid sampling pid)
            if random_negative_per_positive > 0:
                r_k = int(random_negative_per_positive)
                # ensure no overlap with positives by id
                # sample without replacement if possible
                if len(global_candidates) >= r_k:
                    rand_samples = rng.sample(global_candidates, k=r_k)
                else:
                    # fallback: allow repeats if pool too small
                    rand_samples = [rng.choice(global_candidates) for _ in range(r_k)]

                for nid in rand_samples:
                    items.append(PairwiseItem(query_text=q, pos_text=pos_text, neg_text=chunk_lookup[nid]))

    if len(items) == 0:
        raise ValueError("Built 0 pairwise items. Check query_pack content and sampling params.")
    return items


# -----------------------------
# Dataset + Collator (pairwise)
# -----------------------------
class CrossEncoderPairwiseDataset(Dataset):
    """
    Each example returns:
      pos: tokenized (q, d+)
      neg: tokenized (q, d-)
    Truncation policy: keep query, truncate doc.
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
            truncation="only_second",   # ✅ keep query, cut doc
            padding=padding,            # False => dynamic padding in collator
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        return {
            "pos": self._enc(it.query_text, it.pos_text),
            "neg": self._enc(it.query_text, it.neg_text),
        }


class PairwiseCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tok = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # dynamic pad to the longest in batch
        pos = self.tok.pad([f["pos"] for f in features], return_tensors="pt")
        neg = self.tok.pad([f["neg"] for f in features], return_tensors="pt")

        batch: Dict[str, Any] = {
            "pos_input_ids": pos["input_ids"],
            "pos_attention_mask": pos["attention_mask"],
            "neg_input_ids": neg["input_ids"],
            "neg_attention_mask": neg["attention_mask"],
        }
        # include token_type_ids if exists
        if "token_type_ids" in pos:
            batch["pos_token_type_ids"] = pos["token_type_ids"]
        if "token_type_ids" in neg:
            batch["neg_token_type_ids"] = neg["token_type_ids"]
        return batch


# -----------------------------
# Pairwise loss Trainer
# -----------------------------
class PairwiseTrainer(Trainer):
    """
    Computes pairwise logistic loss using s_pos - s_neg.
    """

    def compute_loss(self, model: nn.Module, inputs: Dict[str, Any], return_outputs: bool = False):
        # unpack
        pos_ids = inputs["pos_input_ids"]
        pos_mask = inputs["pos_attention_mask"]
        neg_ids = inputs["neg_input_ids"]
        neg_mask = inputs["neg_attention_mask"]

        # optional token_type_ids
        pos_tti = inputs.get("pos_token_type_ids", None)
        neg_tti = inputs.get("neg_token_type_ids", None)

        # forward
        if pos_tti is None:
            s_pos = model(pos_ids, pos_mask)
            s_neg = model(neg_ids, neg_mask)
        else:
            # If you update CrossEncoderReranker.forward to accept token_type_ids, pass them here.
            # For now, simplest: ignore token_type_ids (most modern models don't need it).
            s_pos = model(pos_ids, pos_mask)
            s_neg = model(neg_ids, neg_mask)

        # pairwise logistic loss: -log(sigmoid(s_pos - s_neg))
        diff = s_pos - s_neg
        loss = torch.nn.functional.softplus(-diff).mean()

        if return_outputs:
            return loss, {"s_pos": s_pos.detach(), "s_neg": s_neg.detach()}
        return loss


# -----------------------------
# LoRA / QLoRA injection
# -----------------------------
def maybe_wrap_with_lora(model: nn.Module, s: Dict[str, Any]) -> nn.Module:
    lora_cfg = s.get("lora", {}) or {}
    enabled = bool(lora_cfg.get("enabled", False))
    if not enabled:
        return model

    from peft import LoraConfig, get_peft_model, TaskType

    r = int(lora_cfg.get("r", 16))
    alpha = int(lora_cfg.get("alpha", 32))
    dropout = float(lora_cfg.get("dropout", 0.05))
    target_modules = lora_cfg.get("target_modules") or ["q_proj", "k_proj", "v_proj", "o_proj"]

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,   # sequence classification head
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    return model


def build_quantization_config_if_needed(s: Dict[str, Any]) -> Optional[BitsAndBytesConfig]:
    lora_cfg = s.get("lora", {}) or {}
    if not bool(lora_cfg.get("enabled", False)):
        return None
    if not bool(lora_cfg.get("qlora_4bit", False)):
        return None

    # QLoRA 4bit quantization for base model weights
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if bool(s.get("bf16", True)) else torch.float16,
    )


# -----------------------------
# Train entry
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    s = load_settings(args.config)
    stores = build_store_registry(s)

    model_name = str(s["model_name"])
    max_length = int(s.get("max_length", 512))

    # Inputs
    pairs_cfg = s["inputs"]["pairs"]
    chunks_cfg = s["inputs"]["chunks"]

    pairs_store = stores[pairs_cfg["store"]]
    chunks_store = stores[chunks_cfg["store"]]

    pairs_path = _posix_join(pairs_cfg["base"], pairs_cfg["pairs"])
    chunks_path = _posix_join(chunks_cfg["base"], chunks_cfg["chunks_file"])

    # Training params
    tr = s.get("Training", {}) or {}
    seed = int(tr.get("seed", 42))
    num_epochs = int(tr.get("num_epochs", 1))
    hard_k = int(tr.get("hard_negative_per_positive", 0))
    rand_k = int(tr.get("random_negative_per_positive", 1))

    output_dir = str(tr.get("output_dir", "runs/reranker"))
    lr = float(tr.get("lr", 2e-5))
    wd = float(tr.get("weight_decay", 0.01))
    warmup_ratio = float(tr.get("warmup_ratio", 0.0))
    bsz_train = int(tr.get("per_device_train_batch_size", 8))
    bsz_eval = int(tr.get("per_device_eval_batch_size", 16))
    grad_accum = int(tr.get("grad_accum_steps", 1))
    log_steps = int(tr.get("log_every_steps", 50))
    eval_steps = int(tr.get("eval_every_steps", 200))
    save_steps = int(tr.get("save_every_steps", 200))
    max_steps = tr.get("max_steps", None)
    max_steps = int(max_steps) if isinstance(max_steps, int) else None

    # Determinism
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build text lookup
    chunk_lookup = build_chunk_text_lookup(store=chunks_store, chunks_path=chunks_path)

    # Build pairwise items
    items = build_pairwise_items_from_querypack(
        store=pairs_store,
        querypack_path=pairs_path,
        chunk_lookup=chunk_lookup,
        hard_negative_per_positive=hard_k,
        random_negative_per_positive=rand_k,
        seed=seed,
    )

    # Split train/valid
    split = float(s.get("data_split", 0.85))
    n_train = int(len(items) * split)
    train_items = items[:n_train]
    valid_items = items[n_train:] if n_train < len(items) else items[: max(1, len(items) // 20)]

    train_ds = CrossEncoderPairwiseDataset(items=train_items, tokenizer=tokenizer, max_length=max_length)
    valid_ds = CrossEncoderPairwiseDataset(items=valid_items, tokenizer=tokenizer, max_length=max_length)
    collator = PairwiseCollator(tokenizer)

    # Build model (with optional QLoRA)
    quant_cfg = build_quantization_config_if_needed(s)

    # If you want QLoRA: you should load the HF model with quant config.
    # Since your CrossEncoderReranker currently loads internally, simplest is:
    # - for QLoRA: you move loading into this script OR modify CrossEncoderReranker to accept quant_cfg.
    #
    # Here we do the practical approach: if qlora_4bit enabled, load base model in this script.
    if quant_cfg is not None:
        from transformers import AutoModelForSequenceClassification

        base = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            quantization_config=quant_cfg,
            device_map="auto",
        )

        # wrap so forward returns [B] score
        class _Wrapper(nn.Module):
            def __init__(self, m: nn.Module) -> None:
                super().__init__()
                self.model = m

            def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits
                if logits.dim() == 2 and logits.size(-1) == 1:
                    return logits.squeeze(-1)
                if logits.dim() == 2 and logits.size(-1) > 1:
                    return logits[:, -1]
                return logits

        model: nn.Module = _Wrapper(base)
    else:
        model = CrossEncoderReranker(model_name)

    # Apply LoRA (works for both normal + quantized base)
    model = maybe_wrap_with_lora(model, s)

    # Precision flags
    bf16 = bool(s.get("bf16", False))
    fp16 = bool(s.get("fp16", False))

    # Logging (TensorBoard)
    logging_dir = os.path.join(output_dir, "runs")

    args_tr = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        weight_decay=wd,
        warmup_ratio=warmup_ratio,
        per_device_train_batch_size=bsz_train,
        per_device_eval_batch_size=bsz_eval,
        gradient_accumulation_steps=grad_accum,
        logging_steps=log_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        bf16=bf16,
        fp16=fp16,
        max_steps=max_steps if max_steps is not None else -1,
        report_to=["tensorboard"],
        logging_dir=logging_dir,
        remove_unused_columns=False,  # IMPORTANT for custom batch keys
    )

    trainer = PairwiseTrainer(
        model=model,
        args=args_tr,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)

    # Optional: print how to view TensorBoard
    print(f"[OK] Training done. Output: {output_dir}")
    print(f"TensorBoard logs: {logging_dir}")
    print("To view (colab): %load_ext tensorboard; %tensorboard --logdir <logging_dir>")


if __name__ == "__main__":
    main()
