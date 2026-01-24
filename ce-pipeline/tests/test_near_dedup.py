from __future__ import annotations

from typing import Callable, Any

import numpy as np
import pytest

import ce_pipeline.processing.near_dedup as near_mod


def _resolve_near_func() -> Callable[..., Any]:
    """
    Try to locate the near/semantic dedup function from src/ce_pipeline/processing/near_dedup.py

    Adjust here if your function name differs.
    """
    candidates = [
        "near_dedup_by_ann_faiss",
        "near_dedup_by_cosine",
        "semantic_dedup",
        "run_near_dedup",
    ]
    for name in candidates:
        fn = getattr(near_mod, name, None)
        if callable(fn):
            return fn

    raise AssertionError(
        "Cannot find a near/semantic dedup function in ce_pipeline.processing.near_dedup.\n"
        f"Tried: {candidates}\n"
        "Fix: rename your function to one of these, OR edit _resolve_near_func() in this test."
    )


def _has_faiss() -> bool:
    try:
        import faiss  # noqa: F401
        return True
    except Exception:
        return False


def test_near_dedup_input_must_be_2d() -> None:
    fn = _resolve_near_func()
    with pytest.raises(ValueError):
        fn(np.array([1.0, 2.0], dtype=np.float32))


def test_near_dedup_removes_later_duplicate_when_embeddings_identical() -> None:
    """
    Works for both:
    - ANN-based dedup implementations
    - brute cosine-based dedup implementations

    Expected: keep first occurrence and remove later duplicates.
    """
    fn = _resolve_near_func()

    # 0 and 1 identical; 2 orthogonal
    emb = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],  # dup of 0
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    # If function is ANN/FAISS-based and faiss isn't installed, skip.
    if fn.__name__ in ("near_dedup_by_ann_faiss", "semantic_dedup") and not _has_faiss():
        pytest.skip("FAISS not installed; skipping ANN near-dedup test.")
        return

    # Try calling with common signatures:
    # 1) returns kept_indices list
    # 2) returns a result object with kept_indices / removed_mask
    # 3) returns indices list under other name
    res = None
    try:
        res = fn(emb, threshold=0.999, topk=3, normalize=True)
    except TypeError:
        # maybe cosine-only function signature: (emb, threshold=...)
        res = fn(emb, threshold=0.999)

    # Interpret output
    if isinstance(res, list):
        kept = res
        assert kept == [0, 2]
    else:
        kept = getattr(res, "kept_indices", None)
        removed_mask = getattr(res, "removed_mask", None)
        if kept is None:
            # fallback: some impl returns (kept, removed_mask)
            if isinstance(res, tuple) and len(res) >= 1 and isinstance(res[0], list):
                kept = res[0]
            else:
                raise AssertionError(
                    "Near dedup returned an unsupported result type. "
                    "Expected list[int] or an object with kept_indices."
                )

        assert kept == [0, 2]

        if removed_mask is not None:
            assert removed_mask.tolist() == [False, True, False]


def test_near_dedup_no_removal_when_threshold_too_high() -> None:
    fn = _resolve_near_func()

    emb = np.array(
        [
            [1.0, 0.0],
            [0.999, 0.01],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    if fn.__name__ in ("near_dedup_by_ann_faiss", "semantic_dedup") and not _has_faiss():
        pytest.skip("FAISS not installed; skipping ANN near-dedup test.")
        return

    try:
        res = fn(emb, threshold=0.99999, topk=3, normalize=True)
    except TypeError:
        res = fn(emb, threshold=0.99999)

    if isinstance(res, list):
        assert res == [0, 1, 2]
    else:
        kept = getattr(res, "kept_indices", None)
        if kept is None and isinstance(res, tuple) and len(res) >= 1 and isinstance(res[0], list):
            kept = res[0]
        assert kept == [0, 1, 2]
