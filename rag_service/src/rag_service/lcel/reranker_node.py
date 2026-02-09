# rag_service/nodes/reranker_node.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda


def _get_required(cfg: Dict[str, Any], key: str) -> Any:
    if key not in cfg:
        raise KeyError(f"Missing required config key: {key}")
    return cfg[key]


def create_reranker_runnable(settings: Dict[str, Any]) -> RunnableLambda:
    cfg = settings["models"]["reranker"]

    provider = str(cfg.get("provider", "hf_transformers"))
    device = str(cfg.get("device", "cpu"))
    cache_dir = cfg.get("cache_dir")
    batch_size = int(cfg.get("batch_size", 16))
    max_length = int(cfg.get("max_length", 512))

    # Optional knobs
    use_fast = bool(cfg.get("use_fast_tokenizer", True))

    if provider == "hf_transformers":
        model_name = str(_get_required(cfg, "model_name"))

        tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=use_fast)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)

    elif provider == "hf_peft":
        # base + adapter (LoRA)
        base_model = str(_get_required(cfg, "base_model"))
        adapter_path = str(_get_required(cfg, "adapter_path"))

        # Import peft lazily so that unit tests can mock it if needed.
        try:
            from peft import PeftModel
        except Exception as e:
            raise RuntimeError(
                "provider=hf_peft requires `peft` installed. "
                "Install peft (and accelerate) then retry."
            ) from e

        # Tokenizer: usually from base_model (stable).
        tok = AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir, use_fast=use_fast)

        base = AutoModelForSequenceClassification.from_pretrained(base_model, cache_dir=cache_dir)
        model = PeftModel.from_pretrained(base, adapter_path)

        # (optional) If you later want merged weights:
        # model = model.merge_and_unload()

    else:
        raise ValueError(f"Unknown reranker provider: {provider}")

    model.eval()
    model.to(device)

    @torch.inference_mode()
    def _rerank(inp: Dict[str, Any]) -> List[Document]:
        query: str = (inp.get("query") or "").strip()
        docs: List[Document] = inp.get("docs") or []
        if not query or not docs:
            return docs

        pairs: List[Tuple[str, str]] = [(query, (d.page_content or "")) for d in docs]

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
            # Cross-encoder rerankers typically output shape [B, 1] or [B]
            scores.extend(logits.detach().float().view(-1).cpu().tolist())

        reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        out: List[Document] = []
        for d, s in reranked:
            d.metadata = dict(d.metadata or {})
            d.metadata["rerank_score"] = float(s)
            out.append(d)
        return out

    return RunnableLambda(_rerank)
