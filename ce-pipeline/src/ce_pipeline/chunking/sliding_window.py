from __future__ import annotations
from typing import List, Tuple


def sliding_window_chunks(text: str, window_chars: int, overlap_chars: int) -> List[Tuple[int, str]]:
    if window_chars <= 0:
        raise ValueError("window_chars must be > 0")
    if overlap_chars < 0 or overlap_chars >= window_chars:
        raise ValueError("overlap_chars must be >=0 and < window_chars")

    t = (text or "").strip()
    if not t:
        return []

    step = window_chars - overlap_chars
    out: List[Tuple[int, str]] = []
    i = 0
    idx = 0
    n = len(t)
    while i < n:
        chunk = t[i : i + window_chars].strip()
        if chunk:
            out.append((idx, chunk))
            idx += 1
        i += step
    return out
