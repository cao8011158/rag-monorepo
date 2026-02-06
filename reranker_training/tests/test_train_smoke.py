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


# If your project already exports EvalPack, import it instead of redefining.
@dataclass
class EvalPack:
    query_text: str
    doc_texts: List[str]
    labels: List[int]


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@pytest.mark.smoke
def test_train_smoke_two_steps_with_eval_new_transformers_api(tmp_path: Path) -> None:
    # -----------------------------
    # 1) Synthetic pairs.jsonl (YOUR schema)
    # -----------------------------
    pairs_path = tmp_path / "pairs.jsonl"

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

    _write_jsonl(
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
    # 2) Load items through your real loader
    #    Assumes signature like: load_pairs_for_epoch(path=..., epoch=..., seed=..., shuffle=...)
    #    If yours differs, adjust ONLY this call.
    # -----------------------------
    items = load_pairs_for_epoch(path=str(pairs_path), epoch=1, seed=42, shuffle=True)

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
    base = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model = CrossEncoderReranker(base)

    # -----------------------------
    # 5) Dataset / Collator
    # -----------------------------
    train_ds = CrossEncoderPairwiseDataset(items=items, tokenizer=tok, max_length=64)
    collator = PairwiseCollator(tok)

    # -----------------------------
    # 6) NEW TrainingArguments API: eval_strategy (not evaluation_strategy)
    #    remove_unused_columns=False is critical for custom batch keys
    # -----------------------------
    args = TrainingArguments(
        output_dir=str(tmp_path / "out"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        max_steps=2,
        logging_steps=1,
        save_steps=2,
        eval_strategy="steps",
        eval_steps=1,
        report_to=[],
        remove_unused_columns=False,
        fp16=False,
        bf16=False,
    )

    # -----------------------------
    # 7) NEW Trainer API: processing_class (not tokenizer)
    # -----------------------------
    trainer = PairwiseTrainerWithRankingEval(
        model=model,
        args=args,
        train_dataset=train_ds,
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
    # 8) Assertions: ran + saved + eval callable + finite loss
    # -----------------------------
    assert trainer.state.global_step == 2
    assert torch.isfinite(torch.tensor(out.training_loss))

    ckpt = Path(args.output_dir) / "checkpoint-2"
    assert ckpt.exists(), f"checkpoint not found: {ckpt}"

    metrics = trainer.evaluate()
    assert isinstance(metrics, dict)
    # depending on your implementation, keys may be eval_ndcg / eval_mrr etc.
    assert any(k.startswith("eval_") for k in metrics.keys()), metrics

