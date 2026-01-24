from __future__ import annotations

# noise (split modules)
from .noise_trim import trim_noise_edges
from .noise_filter import is_noise_chunk


from .repair import repair_boundary_truncation

# exact dedup (hash-based, JSONL → JSONL)
from .exact_dedup import exact_dedup_jsonl_by_hash_meta

# near dedup (ANN + cosine, embeddings → mask/indices)
from .near_dedup import near_dedup_by_ann_faiss, ANNDedupResult

__all__ = [
    # noise handling
    "trim_noise_edges",
    "is_noise_chunk",

    # repair

    "repair_boundary_truncation",

    # exact dedup
    "exact_dedup_jsonl_by_hash_meta",

    # near/semantic dedup
    "near_dedup_by_ann_faiss",
    "ANNDedupResult",
]