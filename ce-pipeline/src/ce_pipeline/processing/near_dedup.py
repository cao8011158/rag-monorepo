from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class ANNDedupResult:
    kept_indices: List[int]         # 保留下来的向量/chunk 行号索引（按输入顺序）
    removed_mask: np.ndarray        # shape [N], True=被删（重复）
    num_kept: int
    num_removed: int


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("emb must be 2D [N, D]")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def near_dedup_by_ann_faiss(
    emb: np.ndarray,
    *,
    threshold: float = 0.95,
    topk: int = 20,
    hnsw_m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 64,
    normalize: bool = True,
) -> ANNDedupResult:
    """
    Near-duplicate removal via ANN (FAISS HNSW) + local cosine verification.

    Notes:
    - If normalize=True: cosine similarity == inner product on L2-normalized vectors.
    - Keeps the first occurrence (lower index) and removes later duplicates.
    """
    if emb.ndim != 2:
        raise ValueError("emb must be 2D [N, D]")
    n, d = emb.shape
    if n == 0:
        return ANNDedupResult([], np.zeros((0,), dtype=bool), 0, 0)
    if topk < 2:
        raise ValueError("topk must be >= 2 (needs self + neighbors)")

    # Optional cosine-friendly normalization
    x = emb.astype(np.float32, copy=False)
    if normalize:
        x = _l2_normalize(x)

    # Import faiss lazily so project can keep it optional
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "FAISS is required for near_dedup_by_ann_faiss. "
            "Install faiss-cpu or faiss-gpu."
        ) from e

    # Build temporary ANN index (HNSW)
    index = faiss.IndexHNSWFlat(d, hnsw_m)
    index.hnsw.efConstruction = int(ef_construction)
    index.hnsw.efSearch = int(ef_search)
    index.add(x)

    # Query ANN: for each vector, get topk nearest (includes itself)
    sims, nbrs = index.search(x, int(topk))  # sims: [N, topk] (inner product if normalized)

    removed = np.zeros(n, dtype=bool)
    kept: List[int] = []

    for i in range(n):
        if removed[i]:
            continue
        kept.append(i)

        # Examine neighbors as duplicate candidates; keep earliest (i), remove later ones
        for pos in range(1, topk):  # pos=0 is usually self
            j = int(nbrs[i, pos])
            if j < 0 or j == i:
                continue
            if j < i:
                # ensure "keep first occurrence": do not delete earlier items
                continue
            if removed[j]:
                continue

            # Local exact verification (cosine) using dot on normalized vectors
            # If not normalized, you could normalize on-the-fly or compute cosine explicitly.
            sim_ij = float(np.dot(x[i], x[j])) if normalize else float(
                np.dot(x[i], x[j]) / (np.linalg.norm(x[i]) * np.linalg.norm(x[j]) + 1e-12)
            )

            if sim_ij >= threshold:
                removed[j] = True

    return ANNDedupResult(
        kept_indices=kept,
        removed_mask=removed,
        num_kept=len(kept),
        num_removed=int(removed.sum()),
    )
