from __future__ import annotations

import argparse
import os
import random
from typing import Any, Dict, List

import torch
import torch.nn as nn

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, TaskType, get_peft_model

from reranker_training.settings import load_settings
from reranker_training.stores.registry import build_store_registry
from reranker_training.data.data_preprocessing import (
    CrossEncoderPairwiseDataset,
    PairwiseCollator,
    load_pairs_for_epoch,
    load_valid_query_packs,
)
from reranker_training.trainer import PairwiseTrainerWithRankingEval


# ============================================================
# Utils
# ============================================================

def _posix_join(base: str, name: str) -> str:
    base = str(base).rstrip("/")
    name = str(name).lstrip("/")
    return f"{base}/{name}" if base else name


# ============================================================
# LoRA helpers
# ============================================================

def _apply_lora_once(
    base_model: nn.Module,
    *,
    target_modules: List[str],
    r: int,
    alpha: int,
    dropout: float,
) -> nn.Module:
    cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=list(target_modules),
        bias="none",
    )
    return get_peft_model(base_model, cfg)


# ============================================================
# Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    s = load_settings(args.config)
    stores = build_store_registry(s)

    model_name = str(s["model_name"])
    max_length = int(s.get("max_length", 512))

    # outputs.files = where you put epoch pair jsonl + valid query_pack jsonl
    out_cfg = s["outputs"]["files"]
    out_store = stores[out_cfg["store"]]
    out_base = str(out_cfg["base"]).rstrip("/")
    train_pair_tpl = str(out_cfg["train_pair_path"])  # "processed/train_pair_epoch_{epoch}.jsonl"
    valid_path = str(out_cfg["valid_path"])           # "processed/valid_query_pack.jsonl"

    tr = s.get("training", {}) or {}
    seed = int(tr.get("seed", 42))
    num_epochs = int(tr.get("num_epochs", 0))
    output_dir = str(tr.get("output_dir", "runs/reranker"))
    lr = float(tr.get("lr", 2e-5))
    wd = float(tr.get("weight_decay", 0.01))
    warmup_ratio = float(tr.get("warmup_ratio", 0.0))
    bsz_train = int(tr.get("per_device_train_batch_size", 8))
    bsz_eval = int(tr.get("per_device_eval_batch_size", 16))
    grad_accum = int(tr.get("grad_accum_steps", 0))
    log_steps = int(tr.get("log_every_steps", 50))
    eval_steps = int(tr.get("eval_every_steps", 200))
    save_steps = int(tr.get("save_every_steps", 200))
    max_steps = tr.get("max_steps", -1)
    max_steps = int(max_steps) if isinstance(max_steps, int) else -1
    num_workers = int(tr.get("num_workers", 0))

    bf16 = bool(s.get("bf16", False))
    fp16 = bool(s.get("fp16", False))

    # eval config
    ev = s.get("eval", {}) or {}
    ndcg_k = int(ev.get("ndcg_k", 10))
    mrr_k = int(ev.get("mrr_k", 10))
    infer_bsz = int(ev.get("infer_batch_size", 32))
    max_negs = ev.get("max_negatives_per_query", None)
    max_negs = int(max_negs) if isinstance(max_negs, int) else None

    # seeds
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------------------------
    # TRAIN: init dataset with epoch=0 only
    # (later epochs will be swapped in callback)
    # -------------------------
    ep0_items = load_pairs_for_epoch(
        store=out_store,
        base=out_base,
        train_pair_path_tpl=train_pair_tpl,
        epoch=0,
    )
    if not ep0_items:
        raise ValueError("No training pairs loaded for epoch=0")

    print(f"[DATA] train epoch=0 pairs={len(ep0_items)}")

    # optional: shuffle epoch0
    rnd0 = random.Random(seed)
    rnd0.shuffle(ep0_items)

    train_ds = CrossEncoderPairwiseDataset(items=ep0_items, tokenizer=tokenizer, max_length=max_length)
    collator = PairwiseCollator(tokenizer)

    # -------------------------
    # VALID: load fixed QueryPacks from valid_path
    # -------------------------
    valid_packs = load_valid_query_packs(
        store=out_store,
        base=out_base,
        valid_path=valid_path,
        max_negatives_per_query=max_negs,
        seed=seed,
    )
    print(f"[DATA] valid query_packs={len(valid_packs)} from {_posix_join(out_base, valid_path)}")

    # -------------------------
    # MODEL: base -> LoRA -> wrapper
    # -------------------------
    base_model = AutoModelForSequenceClassification.from_pretrained(model_name)

    lora_cfg = s.get("lora", {}) or {}
    r = int(lora_cfg.get("r", 16))
    alpha = int(lora_cfg.get("alpha", 32))
    dropout = float(lora_cfg.get("dropout", 0.05))

    targets = lora_cfg.get("target_modules") or ["query", "key", "value", "dense"]
    if not isinstance(targets, list) or not all(isinstance(x, str) and x.strip() for x in targets):
        raise ValueError("lora.target_modules must be a list of non-empty strings")

    base_model = _apply_lora_once(
        base_model,
        target_modules=[x.strip() for x in targets],
        r=r,
        alpha=alpha,
        dropout=dropout,
    )

    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    if trainable <= 0:
        raise RuntimeError(f"LoRA injected 0 trainable params. target_modules={targets}")
    print(f"[LoRA] target_modules={targets}. trainable_params={trainable}")

    model = base_model

    # tensorboard logging dir
    logging_dir = os.path.join(output_dir, "runs")

    args_tr = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=float(num_epochs),
        learning_rate=lr,
        weight_decay=wd,
        warmup_ratio=warmup_ratio,
        per_device_train_batch_size=bsz_train,
        per_device_eval_batch_size=bsz_eval,
        gradient_accumulation_steps=grad_accum,
        logging_steps=log_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        bf16=bf16,
        fp16=fp16,
        max_steps=-1,  
        report_to=["tensorboard"],
        logging_dir=logging_dir,
        remove_unused_columns=False,          # keep custom keys
        dataloader_num_workers=num_workers,
        load_best_model_at_end=True,      # ⭐⭐⭐关键
        metric_for_best_model=f"eval_ndcg@{ndcg_k}",
        greater_is_better=True,
    )

    # -------------------------
    # Callback: swap dataset items at each epoch begin
    # -------------------------
    class _SwapEpochDatasetCallback(TrainerCallback):

        def __init__(self) -> None:
            super().__init__()
            self._epoch = 0

        def on_epoch_begin(self, args, state, control, **kwargs):

            # epoch0 已经预加载
            if self._epoch == 0:
                print(f"[DATA] using dataset epoch=0 pairs={len(train_ds.items)}")
                self._epoch += 1
                return control

            e = self._epoch

            if e >= num_epochs:
                return control

            ep_items = load_pairs_for_epoch(
                store=out_store,
                base=out_base,
                train_pair_path_tpl=train_pair_tpl,
                epoch=e,  # ⭐ base0 对齐
            )

            rnd = random.Random(seed + e)
            rnd.shuffle(ep_items)

            train_ds.items = ep_items

            print(f"[DATA] swapped to dataset epoch={e} pairs={len(ep_items)}")

            self._epoch += 1
            return control

    trainer = PairwiseTrainerWithRankingEval(
        model=model,
        args=args_tr,
        train_dataset=train_ds,
        eval_dataset=train_ds,  # not used; evaluate() overridden
        data_collator=collator,
        processing_class=tokenizer,
        valid_packs=valid_packs,
        max_length=max_length,
        ndcg_k=ndcg_k,
        mrr_k=mrr_k,
        infer_batch_size=infer_bsz,
        callbacks=[_SwapEpochDatasetCallback()],
    )

    trainer.train()

    # 保存最终（best）模型（LoRA adapter）
    trainer.save_model(output_dir)

    # 保存 tokenizer（推理必需）
    tokenizer.save_pretrained(output_dir)

    print(f"[OK] Training done. Output: {output_dir}")
    print(f"TensorBoard logs: {logging_dir}")
    print("To view (colab): %load_ext tensorboard; %tensorboard --logdir <logging_dir>")


if __name__ == "__main__":
    main()
