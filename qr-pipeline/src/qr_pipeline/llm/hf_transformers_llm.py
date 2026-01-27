from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class HFTransformersLLM:
    model_name: str
    device: str = "cpu"  # "cpu" / "cuda"
    cache_dir: Optional[str] = None
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9

    _tok: Any = None
    _model: Any = None

    def load(self) -> None:
        tok_kwargs: Dict[str, Any] = {}
        if self.cache_dir:
            tok_kwargs["cache_dir"] = self.cache_dir

        self._tok = AutoTokenizer.from_pretrained(self.model_name, **tok_kwargs)

        model_kwargs: Dict[str, Any] = {}
        if self.cache_dir:
            model_kwargs["cache_dir"] = self.cache_dir

        # 让 GPU 上更省显存/更快（能用就用）
        if self.device == "cuda" and torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16

        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

        if self.device == "cuda" and torch.cuda.is_available():
            self._model = self._model.to("cuda")
        else:
            self._model = self._model.to("cpu")

        self._model.eval()

    def generate(self, prompt: str) -> str:
        """
        Returns generated text string.  
        """
        if self._model is None or self._tok is None:
            self.load()

        dev = next(self._model.parameters()).device

        inputs = self._tok(prompt, return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        do_sample = (self.temperature is not None and float(self.temperature) > 0.0)

        with torch.no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=int(self.max_new_tokens),
                do_sample=do_sample,
                temperature=float(self.temperature) if do_sample else None,
                top_p=float(self.top_p) if do_sample else None,
                pad_token_id=self._tok.eos_token_id,
                eos_token_id=self._tok.eos_token_id,
            )

        # 只取新生成部分（避免把 prompt 原样吐回来太长）
        gen_ids = out_ids[0][inputs["input_ids"].shape[1] :]
        text = self._tok.decode(gen_ids, skip_special_tokens=True)
        return text.strip()
