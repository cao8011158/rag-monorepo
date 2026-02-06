from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model

from reranker_training.data.data_preprocessing import (
    CrossEncoderPairwiseDataset,
    PairwiseCollator,
    load_pairs_for_epoch,
)
from reranker_training.modeling import CrossEncoderReranker
from reranker_training.trainer import PairwiseTrainerWithRankingEval


# -------------------------------------------------
# EvalPack (import yours if available)
# -------------------------------------------------
@dataclass
class EvalPack:
    query_text: str
    doc_texts: List[str]
    labels: List[int]


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# -------------------------------------------------
# REAL MODEL SLOW INTEGRATION TEST
# -------------------------------------------------
@pytest.mark.slow
def test_train_two_steps_real_model(tmp_path: Path):

    # -------------------------
    # 1️⃣ Synthetic pairs
    # -------------------------
    pairs_path = tmp_path / "pairs.jsonl"

    def chunk(cid, text):
        return {
            "chunk_id": cid,
            "doc_id": "doc",
            "chunk_index": 0,
            "chunk_text": text,
            "chunk_text_hash": cid,
        }

    _write_jsonl(
        pairs_path,
        [
            {
                "query": {"query_text": "apple", "source_chunk_ids": ["x"]},
                "positive": chunk("p1", "apple apple apple"),
                "negative": chunk("n1", "car car car"),
                "meta": {"epoch": 1, "type": "hard_negative"},
            },
            {
                "query": {"query_text": "banana", "source_chunk_ids": ["y"]},
                "positive": chunk("p2", "banana banana banana"),
                "negative": chunk("n2", "truck truck truck"),
                "meta": {"epoch": 1, "type": "random_negative"},
            },
        ],
    )

    items = load_pairs_for_epoch(path=str(pairs_path), epoch=1)

    # -------------------------
    # 2️⃣ Eval packs
    # -------------------------
    valid_packs = [
        EvalPack(
            query_text="apple",
            doc_texts=["apple apple apple", "car car car"],
            labels=[1, 0],
        )
    ]

    # -------------------------
    # 3️⃣ Model + LoRA
    # -------------------------
    model_name = "BAAI/bge-reranker-v2-m3"

    tok = AutoTokenizer.from_pretrained(model_name)

    base = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["query", "key", "value", "dense"],
        bias="none",
    )

    base = get_peft_model(base, lora_cfg)
    model = CrossEncoderReranker(base)

    # -------------------------
    # 4️⃣ Dataset
    # -------------------------
    ds = CrossEncoderPairwiseDataset(items=items, tokenizer=tok, max_length=128)
    collator = PairwiseCollator(tok)

    # -------------------------
    # 5️⃣ Trainer args (NEW API)
    # -------------------------
    args = TrainingArguments(
        output_dir=str(tmp_path / "out"),
        per_device_train_batch_size=1,
        max_steps=2,
        logging_steps=1,
        save_steps=2,
        eval_strategy="steps",
        eval_steps=1,
        remove_unused_columns=False,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    trainer = PairwiseTrainerWithRankingEval(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
        processing_class=tok,
        valid_packs=valid_packs,
        max_length=128,
    )

    trainer.train()

    # -------------------------
    # 6️⃣ Assertions
    # -------------------------
    assert trainer.state.global_step == 2
    assert (Path(args.output_dir) / "checkpoint-2").exists()

    metrics = trainer.evaluate()
    assert "eval_" in "".join(metrics.keys())
