
import math
from typing import Any, Dict, List

import torch
import torch.nn as nn
from transformers import Trainer
from reranker_training.data.data_preprocessing import (
    EvalPack
  )



#   @dataclass
#   class EvalPack:
#       query_text: str
#       doc_texts: List[str]   # candidates = positives + negatives
#       labels: List[int]      # 1 for positive, 0 for negative

#   @dataclass
#   class PairwiseItem:
#       query_text: str
#       pos_text: str
#       neg_text: str
    
# ============================================================
#  PairwiseTrainer (TRAINING)
#
# ▶ OVERRIDE: Trainer.compute_loss()
# ▶ USE     : Pairwise ranking loss
# ============================================================

class PairwiseTrainer(Trainer):
    """
    Pairwise logistic loss using s_pos - s_neg:
      loss = softplus(-(s_pos - s_neg))
    """

    def compute_loss(self, model: nn.Module, inputs: Dict[str, Any], return_outputs: bool = False,  **kwargs):
        pos_ids = inputs["pos_input_ids"]
        pos_mask = inputs["pos_attention_mask"]
        neg_ids = inputs["neg_input_ids"]
        neg_mask = inputs["neg_attention_mask"]

        pos_tt = inputs.get("pos_token_type_ids", None)
        neg_tt = inputs.get("neg_token_type_ids", None)

        s_pos = model(pos_ids, pos_mask, pos_tt) if pos_tt is not None else model(pos_ids, pos_mask)
        s_neg = model(neg_ids, neg_mask, neg_tt) if neg_tt is not None else model(neg_ids, neg_mask)

        diff = s_pos - s_neg
        loss = torch.nn.functional.softplus(-diff).mean()

        if return_outputs:
            return loss, {"s_pos": s_pos.detach(), "s_neg": s_neg.detach()}
        return loss


# ============================================================
# Ranking metrics (eval)
# ============================================================

def ndcg_at_k(rels_sorted: List[int], k: int) -> float:
    k = min(int(k), len(rels_sorted))
    if k <= 0:
        return 0.0

    def dcg(rels: List[int]) -> float:
        s = 0.0
        for i, rel in enumerate(rels[:k], start=1):
            if rel:
                s += 1.0 / math.log2(i + 1)
        return s

    dcg_val = dcg(rels_sorted)
    ideal = sorted(rels_sorted, reverse=True)
    idcg_val = dcg(ideal)
    return (dcg_val / idcg_val) if idcg_val > 0 else 0.0


def mrr_at_k(rels_sorted: List[int], k: int) -> float:
    k = min(int(k), len(rels_sorted))
    for i, rel in enumerate(rels_sorted[:k], start=1):
        if rel:
            return 1.0 / float(i)
    return 0.0


# ============================================================
# Evaluation Flow (Bottom → Top)
# ------------------------------------------------------------
# 1️⃣ score_query_docs()
#     - 最底层：模型推理打分
#     - 对单个 query + 多个 docs：
#        tokenize → batch forward → 得到 relevance scores
#
# 2️⃣ evaluate_ranking()
#     - 中层：排序评估逻辑
#     - 对每个 query pack：
#           调用 score_query_docs()
#           根据 scores 排序 docs
#           计算 NDCG@k / MRR@k
#
# 3️⃣ PairwiseTrainerWithRankingEval.evaluate()
#     - 最外层：Trainer 生命周期入口
#     - 在 validation 时触发：
#           调用 evaluate_ranking()
#           将指标写入 HF logging / dashboard
#
# Overall Call Stack:
#
#   Trainer.evaluate()
#        ↓
#   evaluate_ranking()
#        ↓
#   score_query_docs()
#
# ============================================================


@torch.no_grad()
def score_query_docs(
    *,
    model: nn.Module,
    tokenizer: Any,
    query_text: str,
    doc_texts: List[str],
    max_length: int,
    device: torch.device,
    batch_size: int = 32,
) -> List[float]:
    model.eval()
    scores: List[float] = []

    for i in range(0, len(doc_texts), int(batch_size)):
        batch_docs = doc_texts[i : i + int(batch_size)]
        enc = tokenizer(
            [query_text] * len(batch_docs),
            batch_docs,
            max_length=int(max_length),
            truncation="only_second",
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        tt = enc.get("token_type_ids", None)
        s = model(enc["input_ids"], enc["attention_mask"], tt) if tt is not None else model(enc["input_ids"], enc["attention_mask"])
        scores.extend(s.detach().float().cpu().tolist())

    return scores


def evaluate_ranking(
    *,
    packs: List[EvalPack],
    model: nn.Module,
    tokenizer: Any,
    max_length: int,
    device: torch.device,
    ndcg_k: int = 10,
    mrr_k: int = 10,
    infer_batch_size: int = 32,
) -> Dict[str, float]:
    ndcgs: List[float] = []
    mrrs: List[float] = []

    for p in packs:
        scores = score_query_docs(
            model=model,
            tokenizer=tokenizer,
            query_text=p.query_text,
            doc_texts=p.doc_texts,
            max_length=max_length,
            device=device,
            batch_size=infer_batch_size,
        )

        # sort by score desc
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        rels_sorted = [p.labels[i] for i in idx]

        ndcgs.append(ndcg_at_k(rels_sorted, ndcg_k))
        mrrs.append(mrr_at_k(rels_sorted, mrr_k))

    nq = max(1, len(packs))
    return {
        f"ndcg@{int(ndcg_k)}": float(sum(ndcgs) / nq),
        f"mrr@{int(mrr_k)}": float(sum(mrrs) / nq),
        "num_queries": float(len(packs)),
    }


class PairwiseTrainerWithRankingEval(PairwiseTrainer):
    """
    Overrides evaluate() to compute ranking metrics on QueryPacks:
      - NDCG@k (main)
      - MRR@k (aux)
    """

    def __init__(
        self,
        *args,
        valid_packs: List[EvalPack],
        max_length: int,
        ndcg_k: int = 10,
        mrr_k: int = 10,
        infer_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.valid_packs = valid_packs
        self.eval_max_length = int(max_length)
        self.ndcg_k = int(ndcg_k)
        self.mrr_k = int(mrr_k)
        self.infer_batch_size = int(infer_batch_size)

    def evaluate(self, *args, **kwargs):
        # ✅ Trainer 决定的真实 device（支持单卡/多卡/accelerate）
        device = self.args.device
        if hasattr(self, "accelerator") and getattr(self.accelerator, "device", None) is not None:
            device = self.accelerator.device

        metrics = evaluate_ranking(
            packs=self.valid_packs,
            model=self.model,
            tokenizer=self.processing_class,
            max_length=self.eval_max_length,
            device=device,
            ndcg_k=self.ndcg_k,
            mrr_k=self.mrr_k,
            infer_batch_size=self.infer_batch_size,
        )

        metrics = {("eval_" + k): v for k, v in metrics.items()}
        self.log(metrics)
        return metrics