# rag_service/nodes/reranker_node.py
from __future__ import annotations
from typing import Dict, Any, List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

def create_reranker_runnable(settings) -> RunnableLambda:
    cfg = settings["reranker"]  # 你的 config 段

    model_name = cfg["model_name"]
    device = cfg.get("device", "cpu")
    cache_dir = cfg.get("cache_dir")
    batch_size = int(cfg.get("batch_size", 16))
    max_length = int(cfg.get("max_length", 512))

    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
    model.eval()
    model.to(device)

    @torch.inference_mode()
    def _rerank(inp: Dict[str, Any]) -> List[Document]:
        query: str = inp["query"]
        docs: List[Document] = inp["docs"]

        pairs = [(query, d.page_content) for d in docs]

        scores: List[float] = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            enc = tok(
                [q for q, _ in batch],
                [t for _, t in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            scores.extend(logits.detach().float().view(-1).cpu().tolist())


        reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        out: List[Document] = []
        for d, s in reranked:
            d.metadata = dict(d.metadata or {})
            d.metadata["rerank_score"] = float(s)
            out.append(d)
        return out

    return RunnableLambda(_rerank)
