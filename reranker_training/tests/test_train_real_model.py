from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model

from reranker_training.data.data_preprocessing import (
    CrossEncoderPairwiseDataset,
    PairwiseCollator,
    load_pairs_for_epoch,
)
from reranker_training.modeling import CrossEncoderReranker
from reranker_training.trainer import PairwiseTrainerWithRankingEval


# -------------------------------------------------
# EvalPack (use your project's EvalPack if you have one)
# -------------------------------------------------
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


# -------------------------------------------------
# Minimal Local Store for tests
# (adapts to common store interfaces)
# -------------------------------------------------
class MinimalLocalStore:
    """
    A tiny filesystem-backed store for tests.

    Your load_pairs_for_epoch(store=..., base=..., ...) likely does one of:
      - store.open(rel_path, "r")
      - store.read_text(rel_path)
      - store.read_jsonl(rel_path)
      - store.get_bytes(rel_path) / store.read_bytes(rel_path)

    This class provides several common methods so your loader can use it
    without importing your project's real store class.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root is not None else Path(".")

    def _to_path(self, p: str) -> Path:
        # loader is passing absolute joined path (base/rel) OR rel path; handle both
        pp = Path(p)
        return pp if pp.is_absolute() else (self.root / pp)

    def open(self, path: str, mode: str = "r", encoding: str = "utf-8"):
        p = self._to_path(path)
        if "b" in mode:
            return p.open(mode)
        return p.open(mode, encoding=encoding)

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        return self._to_path(path).read_text(encoding=encoding)

    def read_bytes(self, path: str) -> bytes:
        return self._to_path(path).read_bytes()

    def read_jsonl(self, path: str) -> List[Dict[str, Any]]:
        p = self._to_path(path)
        rows: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows


@pytest.mark.slow
def test_train_two_steps_real_model_transformers5(tmp_path: Path) -> None:
    # -------------------------
    # 0) IMPORTANT PRECONDITION (in your library code):
    #    PairwiseTrainer.compute_loss must accept TF5 extra kwargs, e.g.:
    #      def compute_loss(..., **kwargs):
    #    otherwise Trainer will throw:
    #      unexpected keyword argument 'num_items_in_batch'
    # -------------------------

    # -------------------------
    # 1) Write synthetic pairs file at EXACT location your loader expects
    #    rel = train_pair_path_tpl.format(epoch=epoch)
    #    path = base/rel
    # -------------------------
    base = tmp_path
    train_pair_path_tpl = "pairs_epoch_{epoch}.jsonl"
    pairs_path = base / "pairs_epoch_1.jsonl"

    def mk_chunk(chunk_id: str, text: str) -> Dict[str, Any]:
        return {
            "chunk_id": chunk_id,
            "doc_id": "doc",
            "chunk_index": 0,
            "chunk_text": text,
            "chunk_text_hash": f"hash-{chunk_id}",
            # optional metadata fields
            "url": "https://example.com",
            "title": "t",
            "source": "s",
        }

    _write_jsonl(
        pairs_path,
        [
            {
                "query": {"query_text": "apple", "source_chunk_ids": ["x"], "query_id": "q1", "domain": "test"},
                "positive": mk_chunk("p1", "apple apple apple"),
                "negative": mk_chunk("n1", "car car car"),
                "meta": {"epoch": 1, "type": "hard_negative"},
            },
            {
                "query": {"query_text": "banana", "source_chunk_ids": ["y"], "query_id": "q2", "domain": "test"},
                "positive": mk_chunk("p2", "banana banana banana"),
                "negative": mk_chunk("n2", "truck truck truck"),
                "meta": {"epoch": 1, "type": "random_negative"},
            },
        ],
    )

    # -------------------------
    # 2) Load items using your REAL signature
    # -------------------------
    store = MinimalLocalStore()
    items = load_pairs_for_epoch(
        store=store,
        base=str(base),
        train_pair_path_tpl=train_pair_path_tpl,
        epoch=1,
    )
    assert len(items) == 2

    # -------------------------
    # 3) Eval packs (your overridden evaluate() uses these)
    # -------------------------
    valid_packs = [
        EvalPack(
            query_text="apple",
            doc_texts=["apple apple apple", "car car car"],
            labels=[1, 0],
        )
    ]

    # -------------------------
    # 4) Real model + LoRA
    # -------------------------
    model_name = "BAAI/bge-reranker-v2-m3"
    tok = AutoTokenizer.from_pretrained(model_name)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        ignore_mismatched_sizes=True,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["query", "key", "value", "dense"],  # you confirmed it matches
        bias="none",
    )
    base_model = get_peft_model(base_model, lora_cfg)
    model = CrossEncoderReranker(base_model)

    # -------------------------
    # 5) Dataset / Collator
    # -------------------------
    ds = CrossEncoderPairwiseDataset(items=items, tokenizer=tok, max_length=128)
    collator = PairwiseCollator(tok)

    # -------------------------
    # 6) Transformers 5 TrainingArguments
    #    IMPORTANT: eval_strategy != "no" => Trainer requires eval_dataset != None
    # -------------------------
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
        remove_unused_columns=False,
        report_to=[],
        fp16=torch.cuda.is_available(),
        bf16=False,
    )

    # -------------------------
    # 7) Trainer (Transformers 5: processing_class)
    # -------------------------
    trainer = PairwiseTrainerWithRankingEval(
        model=model,
        args=args,
        train_dataset=ds,
        eval_dataset=ds,          # ✅ required by TF5 init check
        data_collator=collator,
        processing_class=tok,     # ✅ TF5 new arg name
        valid_packs=valid_packs,
        max_length=128,
        ndcg_k=10,
        mrr_k=10,
        infer_batch_size=2,
    )

    # -------------------------
    # 8) Train + assert steps/checkpoint
    # -------------------------
    trainer.train()
    assert trainer.state.global_step == 2
    assert (Path(args.output_dir) / "checkpoint-2").exists()

    # -------------------------
    # 9) Evaluate (your overridden evaluate() prefixes eval_)
    # -------------------------
    metrics = trainer.evaluate()
    assert any(k.startswith("eval_") for k in metrics.keys())
