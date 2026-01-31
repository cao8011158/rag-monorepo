# src/ce_pipeline/processing/repair.py
from __future__ import annotations

from typing import Iterable, Iterator, Optional, Tuple, Any

from syntok import segmenter


# -----------------------------
# Token / sentence iteration helpers
# -----------------------------
def _iter_sentence_token_lists(sent: Any) -> Iterator[list[Any]]:
    """
    syntok segmenter.process(s) output shape varies by version.

    We want to iterate sentences as `list[Token]`.

    Possible shapes we've seen:
      A) sent is an iterable of Token:
           sent = [Token, Token, ...]   or generator(Token, ...)
      B) sent is an iterable of sentence token lists:
           sent = [[Token, Token], [Token, ...], ...]

    This helper yields `list[Token]` for each sentence.
    """
    # Try to iterate `sent` once
    try:
        it = iter(sent)
    except TypeError:
        return

    # Peek first element to decide shape
    try:
        first = next(it)
    except StopIteration:
        return

    def _is_token(x: Any) -> bool:
        return hasattr(x, "offset") and hasattr(x, "value")

    if _is_token(first):
        # Shape A: tokens directly
        toks = [first]
        for x in it:
            toks.append(x)
        yield toks
    else:
        # Shape B: nested sentence lists/iters (each element is an iterable of Token)
        # first is expected to be iterable of Token
        try:
            s_it = iter(first)
        except TypeError:
            # Unknown shape; give up safely
            return

        toks1: list[Any] = []
        for t in s_it:
            toks1.append(t)
        if toks1:
            yield toks1

        for sub in it:
            try:
                sub_it = iter(sub)
            except TypeError:
                continue
            toks: list[Any] = []
            for t in sub_it:
                toks.append(t)
            if toks:
                yield toks


# -----------------------------
# Syntok boundary detector
# -----------------------------
def _last_sentence_end_pos_syntok(text: str, start: int, end: int) -> Optional[int]:
    """
    Return the last sentence end position (inclusive char index) within text[start:end),
    based on syntok sentence segmentation.

    We compute a sentence end position using the last token's absolute offset + len(token.value)-1.

    If no sentence end falls in [start, end), return None.
    """
    s = text or ""
    n = len(s)
    start = max(0, min(start, n))
    end = max(0, min(end, n))
    if start >= end:
        return None

    last_pos: Optional[int] = None

    for sent in segmenter.process(s):
        # sent may represent tokens directly, or a list of sentence token lists
        for toks in _iter_sentence_token_lists(sent):
            if not toks:
                continue
            last_tok = toks[-1]
            pos = int(last_tok.offset) + len(last_tok.value) - 1
            if start <= pos < end:
                if last_pos is None or pos > last_pos:
                    last_pos = pos

    return last_pos


def _collapse_space_join(a: str, b: str) -> str:
    a = (a or "").rstrip()
    b = (b or "").lstrip()
    if not a:
        return b
    if not b:
        return a
    return a + " " + b


def _overlap_matches(cur: str, nxt: str, overlap: int) -> bool:
    if overlap <= 0:
        return True
    if len(cur) < overlap or len(nxt) < overlap:
        return False
    return cur[-overlap:] == nxt[:overlap]


# -----------------------------
# Public API
# -----------------------------
def repair_boundary_by_sentence_syntok(
    cur_text: str,
    next_text: str | None,
    *,
    overlap: int = 120,
    back_search: int = 100,
    forward_search: int = 100,
    min_cur_len: int = 400,
) -> Tuple[str, str | None, str]:
    """
    Sentence-aligned boundary repair using syntok.

    Rules (your understanding):
      Rule 1: Find the last sentence end inside cur's overlap window (last `overlap` chars).
      Rule 2: If not found, search backward `back_search` chars BEFORE overlap_start in cur.
      Rule 3: If still not found, look forward into next (first `overlap+forward_search` chars),
              find a sentence end, move that prefix from next to cur (to close the sentence).

    Strong mode (preferred): if overlap matches exactly, we use a "deduped probe" (continuous text)
    to search boundaries across the boundary safely.

    If overlap verification fails:
      - ONLY run Rule 1.
      - After trimming cur, let `removed = cur[pos1+1:]`.
        Find `removed` EXACTLY in next; if found at idx => drop next[:idx], else abandon repair.
      - If Rule 1 can't find a boundary => abandon repair.

    Returns (cur_fixed, next_fixed, carry)
      - carry is only non-empty for Rule 2 in strong mode.
    """
    cur = (cur_text or "").strip()
    nxt = (next_text or "").strip() if next_text is not None else None

    if not cur:
        return cur, nxt, ""
    if not nxt:
        return cur, nxt, ""

    overlap_start = max(0, len(cur) - overlap)

    # -----------------------------
    # Weak mode: overlap mismatch -> only Rule 1 + exact removed alignment in next
    # -----------------------------
    if not _overlap_matches(cur, nxt, overlap):
        pos1 = _last_sentence_end_pos_syntok(cur, overlap_start, len(cur))
        if pos1 is None:
            return cur, nxt, ""

        new_cur = cur[: pos1 + 1].rstrip()
        if len(new_cur) < min_cur_len:
            return cur, nxt, ""

        removed = cur[pos1 + 1 :].strip()
        if not removed:
            # Nothing removed => nothing to align; keep next unchanged
            return new_cur, nxt, ""

        idx = nxt.find(removed)
        if idx < 0:
            # Can't find exact removed segment in next -> abandon repair
            return cur, nxt, ""

        new_next = nxt[idx:].lstrip()
        return new_cur, new_next, ""

    # -----------------------------
    # Strong mode: overlap verified -> Rule 1/2/3 using a deduped continuous probe
    # -----------------------------
    # Build continuous boundary probe with no duplicated overlap:
    #   cur_tail = last(back_search + overlap) chars from cur
    #   next_head = first(overlap + forward_search) chars from nxt
    #   next_after = next_head[overlap:]  (dedup overlap)
    cur_tail_len = min(len(cur), back_search + overlap)
    cur_tail = cur[len(cur) - cur_tail_len :]
    next_head = nxt[: min(len(nxt), overlap + forward_search)]
    next_after = next_head[overlap:] if len(next_head) >= overlap else ""

    probe = cur_tail + next_after

    # Index mapping:
    cur_tail_start_in_cur = len(cur) - len(cur_tail)
    next_after_start_in_nxt = overlap  # because next_after is nxt[overlap: overlap+forward_search]

    def _probe_pos_to_cur_pos(p: int) -> int:
        return cur_tail_start_in_cur + p

    def _probe_pos_to_nxt_pos(p: int) -> int:
        # p is in probe; next_after begins at probe index len(cur_tail)
        return next_after_start_in_nxt + (p - len(cur_tail))

    # ---- Rule 1: overlap window in cur (cur_tail's last `overlap`) ----
    overlap_start_in_probe = max(0, len(cur_tail) - overlap)
    pos1_probe = _last_sentence_end_pos_syntok(probe, overlap_start_in_probe, len(cur_tail))
    if pos1_probe is not None:
        pos1_cur = _probe_pos_to_cur_pos(pos1_probe)
        new_cur = cur[: pos1_cur + 1].rstrip()
        if len(new_cur) < min_cur_len:
            return cur, nxt, ""

        # Best-effort de-dup on next: if next's first overlap contains a sentence end,
        # drop everything up to that end (inclusive).
        pos_next = _last_sentence_end_pos_syntok(nxt, 0, min(len(nxt), overlap))
        if pos_next is not None:
            new_next = nxt[pos_next + 1 :].lstrip()
        else:
            new_next = nxt

        return new_cur, new_next, ""

    # ---- Rule 2: back_search window before overlap (still on cur side) ----
    back_region_end = overlap_start_in_probe
    back_region_start = max(0, back_region_end - back_search)
    pos2_probe = _last_sentence_end_pos_syntok(probe, back_region_start, back_region_end)
    if pos2_probe is not None:
        pos2_cur = _probe_pos_to_cur_pos(pos2_probe)
        new_cur = cur[: pos2_cur + 1].rstrip()
        if len(new_cur) < min_cur_len:
            return cur, nxt, ""

        overlap_start_in_cur = max(0, len(cur) - overlap)
        carry = cur[pos2_cur + 1 : overlap_start_in_cur].strip()
        new_next = _collapse_space_join(carry, nxt) if carry else nxt
        return new_cur, new_next, carry

    # ---- Rule 3: forward_search in next (after removing overlap) ----
    # Search inside next_after region of probe: [len(cur_tail), len(cur_tail)+len(next_after))
    next_region_start = len(cur_tail)
    next_region_end = len(cur_tail) + len(next_after)
    if next_region_end <= next_region_start:
        return cur, nxt, ""

    pos3_probe = _last_sentence_end_pos_syntok(probe, next_region_start, next_region_end)
    if pos3_probe is None:
        return cur, nxt, ""

    pos3_nxt = _probe_pos_to_nxt_pos(pos3_probe)
    carry = nxt[: pos3_nxt + 1].strip()
    if not carry:
        return cur, nxt, ""

    new_cur = _collapse_space_join(cur, carry)
    if len(new_cur) < min_cur_len:
        return cur, nxt, ""

    new_next = nxt[pos3_nxt + 1 :].lstrip()
    return new_cur, new_next, carry
