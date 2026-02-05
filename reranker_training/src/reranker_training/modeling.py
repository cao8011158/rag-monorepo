import inspect
import torch
import torch.nn as nn
from typing import Optional


class CrossEncoderReranker(nn.Module):
    """
    Wraps a (query, doc) cross-encoder returning a single relevance logit.
    - Handles models that do / do not accept token_type_ids (BERT vs RoBERTa/DeBERTa/etc.)
    - Always returns a 1D tensor of shape [B] when possible.
    """

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.model = base

        # Detect whether underlying forward() accepts token_type_ids.
        self._accepts_token_type_ids = self._detect_accepts_token_type_ids(base)

    @staticmethod
    def _detect_accepts_token_type_ids(model: nn.Module) -> bool:
        """
        Best-effort detection:
        - Try to inspect forward() signature for token_type_ids or **kwargs.
        - If inspection fails, default to False (safer).
        """
        try:
            sig = inspect.signature(model.forward)
            params = sig.parameters

            # If explicitly has token_type_ids
            if "token_type_ids" in params:
                return True

            # If has **kwargs, it likely can accept token_type_ids (but not guaranteed).
            for p in params.values():
                if p.kind == inspect.Parameter.VAR_KEYWORD:
                    return True

            return False
        except Exception:
            # Safer default: don't pass token_type_ids
            return False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Only pass token_type_ids if:
        # 1) caller provided it
        # 2) underlying model appears to accept it
        if token_type_ids is not None and self._accepts_token_type_ids:
            try:
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
            except TypeError:
                # Some models have **kwargs detection but still reject token_type_ids.
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = out.logits

        # Normalize logits to a single relevance score per example.
        if logits.dim() == 2 and logits.size(-1) == 1:
            return logits.squeeze(-1)
        if logits.dim() == 2 and logits.size(-1) > 1:
            # fallback: take last logit as relevance
            return logits[:, -1]
        return logits
