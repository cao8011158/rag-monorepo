from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments

from reranker_training.data.data_preprocessing import (
    CrossEncoderPairwiseDataset,
    PairwiseCollator,
    load_pairs_for_epoch,
)
from reranker_training.modeling import CrossEncoderReranker
from reranker_training.trainer import PairwiseTrainerWithRankingEval


# -----------------------------
# Minimal EvalPack (only if your project doesn't already export it)
# -----------------------------
@dataclass
class EvalPack:
    query_text: str
    doc_texts: List[str]
    labels: List[int]


# -----------------------------
# Local filesystem Store adapter matching YOUR io/jsonl.py
# Requires: read_bytes / write_bytes / exists
# -----------------------------
class LocalStore:
    def read_bytes(self, path: str) -> bytes:
        return Path(path).read_bytes()

    def write_bytes(self, path: str, data: bytes) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def exists(self, path: str) -> bool:
        return Path(path).exists()


def _write_jsonl_local(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write JSONL using stdlib json; compatible with your read_jsonl (orjson loads)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@pytest.mark.smoke
def test_train_smoke_two_steps_with_eval_transformers5(tmp_path: Path) -> None:
    # -----------------------------
    # 1) Write synthetic pair file (YOUR schema)
    #
    # load_pairs_for_epoch does:
    #   rel = train_pair_path_tpl.format(epoch=epoch)
    #   path = base/rel
    #
    # So we must create the file exactly at: tmp_path / rel
    # -----------------------------
    base = tmp_path
    train_pair_path_tpl = "pairs_epoch_{epoch}.jsonl"
    pairs_path = base / "pairs_epoch_1.jsonl"

    def mk_chunk(chunk_id: str, text: str) -> Dict[str, Any]:
        return {
            "chunk_id": chunk_id,
            "doc_id": "doc1",
            "chunk_index": 0,
            "chunk_text": text,
            "chunk_text_hash": f"hash-{chunk_id}",
            "url": "https://example.com",
            "title": "t",
            "source": "s",
        }

    _write_jsonl_local(
        pairs_path,
        [
            {
                "query": {
                    "query_text": "apple",
                    "source_chunk_ids": ["c1"],
                    "query_id": "q1",
                    "domain": "test",
                },
                "positive": mk_chunk("c-pos-1", "apple apple apple"),
                "negative": mk_chunk("c-neg-1", "car car car"),
                "meta": {"epoch": 1, "type": "hard_negative"},
            },
            {
                "query": {
                    "query_text": "banana",
                    "source_chunk_ids": ["c2"],
                    "query_id": "q2",
                    "domain": "test",
                },
                "positive": mk_chunk("c-pos-2", "banana banana banana"),
                "negative": mk_chunk("c-neg-2", "truck truck truck"),
                "meta": {"epoch": 1, "type": "random_negative"},
            },
        ],
    )

    # -----------------------------
    # 2) Load items via your REAL loader signature
    # -----------------------------
    store = LocalStore()
    items = load_pairs_for_epoch(
        store=store,
        base=str(base),
        train_pair_path_tpl=train_pair_path_tpl,
        epoch=1,
    )
    assert len(items) == 2

    # -----------------------------
    # 3) Minimal valid packs (evaluate must run)
    # -----------------------------
    valid_packs = [
        EvalPack(
            query_text="apple",
            doc_texts=["apple apple apple", "car car car", "banana banana banana"],
            labels=[1, 0, 0],
        )
    ]

    # -----------------------------
    # 4) Tiny model for fast smoke
    # -----------------------------
    model_name = "hf-internal-testing/tiny-random-bert"
    tok = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            ignore_mismatched_sizes=True,   # ⭐⭐⭐ 关键新增
    )
    model = CrossEncoderReranker(base_model)

    # -----------------------------
    # 5) Dataset / Collator
    # -----------------------------
    train_ds = CrossEncoderPairwiseDataset(items=items, tokenizer=tok, max_length=64)
    collator = PairwiseCollator(tok)

    # -----------------------------
    # 6) Transformers 5 TrainingArguments
    # -----------------------------
    args = TrainingArguments(
        output_dir=str(tmp_path / "out"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        max_steps=2,
        logging_steps=1,
        save_strategy="steps",
        save_steps=2,
        eval_strategy="steps",
        eval_steps=1,
        report_to=[],
        remove_unused_columns=False,
        fp16=False,
        bf16=False,
    )

    # -----------------------------
    # 7) Trainer (Transformers 5: processing_class)
    # PairwiseTrainerWithRankingEval takes valid_packs/max_length as explicit kwargs
    # -----------------------------
    trainer = PairwiseTrainerWithRankingEval(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=train_ds,
        data_collator=collator,
        processing_class=tok,
        valid_packs=valid_packs,
        max_length=64,
        ndcg_k=3,
        mrr_k=3,
        infer_batch_size=2,
    )

    out = trainer.train()

    # -----------------------------
    # 8) Assertions
    # -----------------------------
    assert trainer.state.global_step == 2
    assert torch.isfinite(torch.tensor(float(out.training_loss)))

    # checkpoint existence (robust)
    out_dir = Path(args.output_dir)
    ckpts = sorted([p for p in out_dir.glob("checkpoint-*") if p.is_dir()])
    assert ckpts, f"no checkpoint-* found under {out_dir}"

    metrics = trainer.evaluate()
    assert isinstance(metrics, dict)
    assert any(k.startswith("eval_") for k in metrics.keys()), metrics
