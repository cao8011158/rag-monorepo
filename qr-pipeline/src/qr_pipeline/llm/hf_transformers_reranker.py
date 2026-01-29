# src/qr_pipeline/llm/hf_transformers_reranker.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class HFTransformersReranker:
    """
    Minimal cross-encoder reranker wrapper.

    API:
      - load(): load tokenizer + model
      - score(query, docs): return list[float] scores (higher => more relevant)

    Notes:
      - Scores are *not* probabilities (not guaranteed 0-1).
      - Uses AutoModelForSequenceClassification logits as relevance scores.
    """

    model_name: str
    device: str = "cpu"  # "cpu" / "cuda"
    cache_dir: Optional[str] = None
    batch_size: int = 32
    max_length: int = 512
    fp16: bool = True

    _tokenizer: Any = None
    _model: Any = None

    def load(self) -> None:
        if not self.model_name or not str(self.model_name).strip():
            raise ValueError("HFTransformersReranker.model_name is required")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_fast=True,
            trust_remote_code=True,  # bge-reranker-v2-m3 often needs this
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        dev = str(self.device or "cpu")
        if dev.startswith("cuda") and not torch.cuda.is_available():
            dev = "cpu"
        self.device = dev

        self._model.to(self.device)
        self._model.eval()

        # optional half precision on cuda
        if self.fp16 and self.device.startswith("cuda"):
            try:
                self._model.half()
            except Exception:
                # Some models may not support .half(); ignore.
                pass

    def _ensure_loaded(self) -> None:
        if self._tokenizer is None or self._model is None:
            raise RuntimeError("Reranker not loaded. Call .load() first.")

    @torch.no_grad()
    def score(self, query: str, docs: Sequence[str]) -> List[float]:
        """
        Score each doc for a single query.
        Returns: scores in the same order as docs.
        """
        self._ensure_loaded()

        q = (query or "").strip()
        if not q:
            raise ValueError("query must be non-empty")

        texts = [(q, (d or "").strip()) for d in docs]
        if not texts:
            return []

        scores: List[float] = []
        bs = max(1, int(self.batch_size))

        # Inference in batches
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            # tokenizer supports pair inputs: (query, doc)
            enc = self._tokenizer(
                [x[0] for x in batch],
                [x[1] for x in batch],
                padding=True,
                truncation=True,
                max_length=int(self.max_length),
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            out = self._model(**enc)
            logits = out.logits

            # logits shape can be [B, 1] or [B, 2]
            if logits.dim() == 2 and logits.size(-1) == 1:
                batch_scores = logits.squeeze(-1)
            elif logits.dim() == 2 and logits.size(-1) >= 2:
                # take "positive" logit as relevance
                batch_scores = logits[:, -1]
            else:
                batch_scores = logits.reshape(-1)

            scores.extend([float(x) for x in batch_scores.detach().float().cpu().tolist()])

        return scores

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> List[float]:
        """
        Score multiple (query, doc) pairs (for advanced use).
        """
        self._ensure_loaded()
        if not pairs:
            return []
        scores: List[float] = []
        bs = max(1, int(self.batch_size))

        for i in range(0, len(pairs), bs):
            batch = pairs[i : i + bs]
            enc = self._tokenizer(
                [q for q, _ in batch],
                [d for _, d in batch],
                padding=True,
                truncation=True,
                max_length=int(self.max_length),
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            out = self._model(**enc)
            logits = out.logits

            if logits.dim() == 2 and logits.size(-1) == 1:
                batch_scores = logits.squeeze(-1)
            elif logits.dim() == 2 and logits.size(-1) >= 2:
                batch_scores = logits[:, -1]
            else:
                batch_scores = logits.reshape(-1)

            scores.extend([float(x) for x in batch_scores.detach().float().cpu().tolist()])
        return scores
