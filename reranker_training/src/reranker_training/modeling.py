from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification


class CrossEncoderReranker(nn.Module):
    """
    Wraps a (query, doc) cross-encoder returning a single relevance logit.
    Many reranker checkpoints already are sequence classification with 1 logit.
    """

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits
        if logits.dim() == 2 and logits.size(-1) == 1:
            return logits.squeeze(-1)
        if logits.dim() == 2 and logits.size(-1) > 1:
            # If model outputs multiple labels, take the last logit as "relevance" fallback.
            return logits[:, -1]
        return logits
