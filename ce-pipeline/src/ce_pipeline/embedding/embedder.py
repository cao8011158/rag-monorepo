from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class DualInstructEmbedder:
    model_name: str
    passage_instruction: str
    query_instruction: str
    batch_size: int = 64
    normalize_embeddings: bool = True
    device: Optional[str] = None  # "cpu" / "cuda" / None(auto)

    def __post_init__(self) -> None:
        try:
            if self.device:
                self._model = SentenceTransformer(self.model_name, device=self.device)
            else:
                self._model = SentenceTransformer(self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{self.model_name}': {e}") from e

    def _encode(self, texts: List[str], instruction: str) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        inputs = [f"{instruction}\n{t}" for t in texts]

        vec = self._model.encode(
            inputs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        if vec.dtype != np.float32:
            vec = vec.astype(np.float32, copy=False)
        return vec

    def encode_passages(self, passages: List[str]) -> np.ndarray:
        return self._encode(passages, self.passage_instruction)

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        return self._encode(queries, self.query_instruction)
