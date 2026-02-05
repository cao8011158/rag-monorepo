import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest
import torch
import torch.nn as nn
from transformers import TrainingArguments

# ✅ 按你项目结构的正确导入路径
from reranker_training.trainer import (
    PairwiseTrainer,
    PairwiseTrainerWithRankingEval,
    ndcg_at_k,
    mrr_at_k,
    score_query_docs,
    evaluate_ranking,
)

from reranker_training.data.data_preprocessing import EvalPack


# -----------------------------
# Helpers: dummy tokenizer/model
# -----------------------------
class DummyTokenizer:
    """
    A minimal tokenizer stub that returns tensors shaped like HF tokenizer output.
    Supports optional token_type_ids.
    """

    def __init__(self, *, with_token_type_ids: bool):
        self.with_token_type_ids = with_token_type_ids

    def __call__(
        self,
        queries: List[str],
        docs: List[str],
        max_length: int,
        truncation: str,
        padding: bool,
        return_tensors: str,
    ) -> Dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        bsz = len(docs)
        # fake lengths
        L = min(int(max_length), 8)
        input_ids = torch.arange(bsz * L).view(bsz, L)
        attention_mask = torch.ones(bsz, L, dtype=torch.long)

        out = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.with_token_type_ids:
            out["token_type_ids"] = torch.zeros(bsz, L, dtype=torch.long)
        return out


class DummyCrossEncoder(nn.Module):
    """
    A tiny "cross-encoder" stub that returns a single score per row: shape [B].
    It accepts (input_ids, attention_mask) or (input_ids, attention_mask, token_type_ids).
    Score is deterministic so we can assert ordering/metrics.
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # score = sum(input_ids * attention_mask) per sample
        x = (input_ids * attention_mask).sum(dim=1).float()
        return x  # [B]


class DummyCrossEncoderNoTT(nn.Module):
    """Same but does NOT accept token_type_ids."""

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = (input_ids * attention_mask).sum(dim=1).float()
        return x


# --------------------------------
# Pure metric function unit tests
# --------------------------------
def test_mrr_at_k_basic():
    rels = [0, 0, 1, 0]
    assert mrr_at_k(rels, 10) == pytest.approx(1.0 / 3.0)

    rels2 = [1, 0, 0]
    assert mrr_at_k(rels2, 2) == pytest.approx(1.0)

    rels3 = [0, 0, 0]
    assert mrr_at_k(rels3, 10) == pytest.approx(0.0)


def test_ndcg_at_k_basic():
    # if best doc is relevant and appears first => ndcg = 1
    rels = [1, 0, 0]
    assert ndcg_at_k(rels, 10) == pytest.approx(1.0)

    # relevant at rank 2 => dcg = 1/log2(3)
    rels2 = [0, 1, 0]
    expected = (1.0 / math.log2(2 + 1)) / 1.0
    assert ndcg_at_k(rels2, 10) == pytest.approx(expected)

    # no positives => 0
    rels3 = [0, 0, 0]
    assert ndcg_at_k(rels3, 10) == pytest.approx(0.0)


# ----------------------------------------
# score_query_docs() behavior unit tests
# ----------------------------------------
@pytest.mark.parametrize("with_token_type_ids", [False, True])
def test_score_query_docs_batching_and_device(with_token_type_ids: bool):
    device = torch.device("cpu")
    tok = DummyTokenizer(with_token_type_ids=with_token_type_ids)

    # choose model consistent with tokenizer output
    model = DummyCrossEncoder() if with_token_type_ids else DummyCrossEncoderNoTT()
    model.to(device)

    query = "q"
    doc_texts = ["d1", "d2", "d3"]  # len < batch_size case covered if batch_size large
    scores = score_query_docs(
        model=model,
        tokenizer=tok,
        query_text=query,
        doc_texts=doc_texts,
        max_length=16,
        device=device,
        batch_size=32,  # doc_texts < batch_size
    )

    assert isinstance(scores, list)
    assert len(scores) == len(doc_texts)
    assert all(isinstance(x, float) for x in scores)

    # check deterministic monotonicity due to arange input_ids:
    # each sample sum increases with sample index, so scores strictly increasing
    assert scores[0] < scores[1] < scores[2]


def test_score_query_docs_multiple_batches():
    device = torch.device("cpu")
    tok = DummyTokenizer(with_token_type_ids=False)
    model = DummyCrossEncoderNoTT().to(device)

    doc_texts = [f"d{i}" for i in range(10)]
    scores = score_query_docs(
        model=model,
        tokenizer=tok,
        query_text="q",
        doc_texts=doc_texts,
        max_length=16,
        device=device,
        batch_size=4,  # force 3 batches: 4,4,2
    )
    assert len(scores) == 10
    assert scores[0] < scores[-1]


# ----------------------------------------
# evaluate_ranking() logic unit tests
# ----------------------------------------
def test_evaluate_ranking_metrics_keys_and_ranges():
    device = torch.device("cpu")
    tok = DummyTokenizer(with_token_type_ids=False)
    model = DummyCrossEncoderNoTT().to(device)

    # Construct a pack where label aligns with increasing score
    # Because scores increase with doc index, the highest score is last doc.
    pack = EvalPack(
        query_text="q",
        doc_texts=["d0", "d1", "d2", "d3"],
        labels=[0, 0, 0, 1],  # relevant doc is last => should rank first after sorting => ndcg==1
    )

    metrics = evaluate_ranking(
        packs=[pack],
        model=model,
        tokenizer=tok,
        max_length=16,
        device=device,
        ndcg_k=10,
        mrr_k=10,
        infer_batch_size=32,
    )

    assert "ndcg@10" in metrics
    assert "mrr@10" in metrics
    assert "num_queries" in metrics
    assert metrics["num_queries"] == pytest.approx(1.0)

    assert 0.0 <= metrics["ndcg@10"] <= 1.0
    assert 0.0 <= metrics["mrr@10"] <= 1.0

    assert metrics["ndcg@10"] == pytest.approx(1.0)
    assert metrics["mrr@10"] == pytest.approx(1.0)


# ----------------------------------------
# PairwiseTrainer.compute_loss() unit tests
# ----------------------------------------
class DummyPairwiseModel(nn.Module):
    """
    PairwiseTrainer expects model(...) returns shape [B] scores.
    We'll return simple deterministic score from input_ids.
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return (input_ids * attention_mask).sum(dim=1).float()


def test_pairwise_trainer_compute_loss_decreases_when_pos_better(tmp_path):
    # We call compute_loss directly; no need to run full training loop.
    tr_args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        report_to=[],
    )
    trainer = PairwiseTrainer(
        model=DummyPairwiseModel(),
        args=tr_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        data_collator=None,
    )

    # Build a fake batch: pos_ids should yield higher score than neg_ids
    pos_ids = torch.tensor([[10, 10], [10, 10]])
    neg_ids = torch.tensor([[1, 1], [1, 1]])
    mask = torch.ones_like(pos_ids)

    inputs = {
        "pos_input_ids": pos_ids,
        "pos_attention_mask": mask,
        "neg_input_ids": neg_ids,
        "neg_attention_mask": mask,
    }

    loss_good = trainer.compute_loss(trainer.model, inputs).item()

    # Now flip: make pos worse than neg => loss should be larger
    inputs_bad = {
        "pos_input_ids": neg_ids,
        "pos_attention_mask": mask,
        "neg_input_ids": pos_ids,
        "neg_attention_mask": mask,
    }
    loss_bad = trainer.compute_loss(trainer.model, inputs_bad).item()

    assert loss_good < loss_bad


# -----------------------------------------------------
# PairwiseTrainerWithRankingEval.evaluate() unit tests
# -----------------------------------------------------
def test_pairwise_trainer_with_ranking_eval_prefix_and_log_called(monkeypatch, tmp_path):
    # Setup trainer
    tr_args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_eval_batch_size=2,
        report_to=[],
    )

    model = DummyCrossEncoderNoTT()
    tok = DummyTokenizer(with_token_type_ids=False)

    packs = [
        EvalPack(query_text="q", doc_texts=["d0", "d1"], labels=[0, 1]),
        EvalPack(query_text="q2", doc_texts=["d0", "d1", "d2"], labels=[0, 0, 1]),
    ]

    t = PairwiseTrainerWithRankingEval(
        model=model,
        args=tr_args,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=tok,
        data_collator=None,
        valid_packs=packs,
        max_length=16,
        ndcg_k=10,
        mrr_k=10,
        infer_batch_size=32,
    )

    # spy on log()
    logged = {}

    def _log_spy(metrics: Dict[str, float]):
        logged.update(metrics)

    monkeypatch.setattr(t, "log", _log_spy)

    metrics = t.evaluate()

    # evaluate() returns eval_* keys
    assert all(k.startswith("eval_") for k in metrics.keys())
    assert "eval_ndcg@10" in metrics
    assert "eval_mrr@10" in metrics
    assert "eval_num_queries" in metrics

    # ensure log called with same metrics
    assert logged == metrics
