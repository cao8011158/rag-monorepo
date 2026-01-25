from __future__ import annotations

from .embedder import DualInstructEmbedder



# near dedup (ANN + cosine, embeddings → mask/indices)
from .near_dedup import near_dedup_by_ann_faiss, ANNDedupResult

__all__ = [


    # near/semantic dedup
    "near_dedup_by_ann_faiss",
    "ANNDedupResult",

    # embedding
    "DualInstructEmbedder",
]