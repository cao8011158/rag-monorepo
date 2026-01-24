from __future__ import annotations

"""
noise_trim.py

Edge-trimming for "flatten text + sliding window" chunks.

Purpose:
- Sliding window often mixes useful content with footer/cookie/related blocks
  at the *edges* of a chunk.
- We trim obvious boilerplate from the head/tail only (NOT mid-chunk),
  to reduce false positives.

Public API:
- trim_noise_edges(text: str) -> str
"""

import re
from typing import List


# -------------------------
# Noise keyword patterns
# -------------------------

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
_EDGE_SPLIT_RE = re.compile(r"\n{2,}|(?:\s*\|\s*)|(?:\s*[•»›]\s*)", re.IGNORECASE)


def trim_noise_edges(text: str) -> str:
    """
    Trim boilerplate-like segments from chunk *edges* (head/tail).

    Notes:
    - Edge-only trimming is intentional: mid-chunk keyword removal is risky.
    - We split the chunk into "parts" by paragraph breaks and common nav separators,
      then pop noise-like parts from the beginning and end.
    """
    t = (text or "").strip()
    if not t:
        return ""

    parts = [p.strip() for p in _EDGE_SPLIT_RE.split(t) if p.strip()]
    if not parts:
        return ""

    while parts and _part_is_edge_noise(parts[-1]):
        parts.pop()

    while parts and _part_is_edge_noise(parts[0]):
        parts.pop(0)

    return "\n\n".join(parts).strip()


def _part_is_edge_noise(part: str) -> bool:
    """
    Decide whether a *single edge part* looks like boilerplate.
    Used internally by trim_noise_edges().
    """
    p = (part or "").strip()
    if not p:
        return True

    # Short multi-line blocks often represent nav/footer
    if len(p) < 80 and p.count("\n") >= 2:
        return True

    # Strong patterns at edges are safe to trim
    if _NOISE_RE_STRONG.search(p):
        return True

    # Weak patterns at edges: trim only if short (avoid legit legal/editorial)
    if _NOISE_RE_WEAK.search(p) and len(p) < 220:
        return True

    # Separator-heavy short parts -> likely breadcrumb/nav lists
    if len(p) < 220:
        seps = sum(p.count(ch) for ch in _SEP_CHARS)
        if seps >= 3:
            return True

    return False
