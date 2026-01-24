from __future__ import annotations
import re

_END_PUNCT = re.compile(r"[\.!?…。！？]\s*$")
_START_LOWER = re.compile(r"^[a-z]")
_DASH_END = re.compile(r"[-–—]\s*$")

def _looks_truncated_end(t: str) -> bool:
    t = t.strip()
    if not t:
        return False
    if _END_PUNCT.search(t):
        return False
    if _DASH_END.search(t):
        return True
    if t.endswith((",", ":", ";")):
        return True
    if len(t) > 60 and not _END_PUNCT.search(t[-3:]):
        return True
    return False

def _looks_truncated_start(t: str) -> bool:
    t = t.lstrip()
    if not t:
        return False
    if _START_LOWER.search(t):
        return True
    if t[0] in ",;:)]}":
        return True
    return False

def repair_boundary_truncation(prev_text: str | None, cur_text: str, next_text: str | None) -> str:
    cur = (cur_text or "").strip()
    if not cur:
        return cur

    if prev_text and _looks_truncated_start(cur):
        tail = prev_text.strip()[-300:].strip()
        if tail:
            cur = tail + " " + cur

    if next_text and _looks_truncated_end(cur):
        head = next_text.strip()[:300].strip()
        if head:
            cur = cur + " " + head

    return cur.strip()
