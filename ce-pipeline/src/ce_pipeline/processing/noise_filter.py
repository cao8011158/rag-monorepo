from __future__ import annotations

"""
noise_filter.py

Chunk-level noise decision function.

Purpose:
- Decide whether a chunk should be dropped AFTER edge trimming.
- Length-aware rules reduce false positives for sliding window chunking.

Public API:
- is_noise_chunk(text: str, min_chunk_chars: int = 200) -> bool

Important:
- This function expects already-trimmed text (use trim_noise_edges first).
"""

import re
from typing import List


_NOISE_PATTERNS_STRONG: List[str] = [
    r"accept\s+cookies",
    r"cookie\s+settings",
    r"manage\s+cookies",
    r"consent\s+preferences?",
    r"do\s+not\s+sell\s+my\s+personal\s+information",
    r"your\s+privacy\s+choices?",
    r"advertis(e|ing|ement)",
    r"sponsored\s+content",
    r"subscribe\s+to",
    r"sign\s+up\s+for",
    r"newsletter",
]

_NOISE_PATTERNS_WEAK: List[str] = [
    r"cookie\s+policy",
    r"privacy\s+policy",
    r"terms\s+of\s+service",
    r"terms\s+and\s+conditions",
    r"all\s+rights\s+reserved",
    r"copyright\s*(?:©\s*)?\d{4}",
    r"related\s+articles",
    r"related\s+posts",
    r"more\s+from\s+this\s+(?:author|category|section)",
]

_NOISE_RE_STRONG = re.compile("|".join(f"(?:{p})" for p in _NOISE_PATTERNS_STRONG), re.IGNORECASE)
_NOISE_RE_WEAK = re.compile("|".join(f"(?:{p})" for p in _NOISE_PATTERNS_WEAK), re.IGNORECASE)

_SEP_CHARS = ["|", "•", "»", "›"]


def is_noise_chunk(text: str, *, min_chunk_chars: int = 200) -> bool:
    """
    Decide whether a trimmed chunk should be dropped.

    Rules (in order):
    1) empty => drop
    2) too short (< min_chunk_chars) => drop
    3) short (<250) + separator-dense => drop (nav/breadcrumb)
    4) strong noise match + not long (<600) => drop
    5) weak noise match + short (<250) => drop
    """
    t = (text or "").strip()
    if not t:
        return True

    if len(t) < min_chunk_chars:
        return True

    if len(t) < 250:
        seps = sum(t.count(ch) for ch in _SEP_CHARS)
        if seps >= 4:
            return True

    # Strong noise: drop unless the chunk is long (avoid footer contamination false positives)
    if _NOISE_RE_STRONG.search(t) and len(t) < 600:
        return True

    # Weak noise: only drop if short (avoid legit content discussing privacy/terms)
    if _NOISE_RE_WEAK.search(t) and len(t) < 250:
        return True

    return False
