# tests/test_boundary_repair_syntok.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List

import pytest

from ce_pipeline.processing.repair import _last_sentence_end_pos_syntok


def _idx(s: str, sub: str) -> int:
    """Return index of first occurrence; assert exists."""
    i = s.find(sub)
    assert i >= 0
    return i


def test_last_sentence_end_pos_basic_windows() -> None:
    """
    Returns the last sentence end (inclusive char index) within [start, end).
    """
    text = "Hello world. This is second sentence! Third one?"
    pos_dot = _idx(text, ".")
    pos_bang = _idx(text, "!")
    pos_q = _idx(text, "?")

    # Full window: last end is '?'
    assert _last_sentence_end_pos_syntok(text, 0, len(text)) == pos_q

    # Exclude the last sentence end: last end becomes '!'
    assert _last_sentence_end_pos_syntok(text, 0, pos_q) == pos_bang

    # Window only up to before '!': last end becomes '.'
    assert _last_sentence_end_pos_syntok(text, 0, pos_bang) == pos_dot

    # Window that only covers the tail containing '?'
    assert _last_sentence_end_pos_syntok(text, pos_bang + 1, len(text)) == pos_q


def test_last_sentence_end_pos_none_when_no_end_in_window() -> None:
    """
    If no sentence end falls in [start, end), return None.
    """
    text = "Hello world. This is second sentence! Third one?"
    pos_dot = _idx(text, ".")
    pos_bang = _idx(text, "!")
    pos_q = _idx(text, "?")

    # A window in the middle that excludes any sentence-ending punctuation
    start = pos_dot + 2
    end = min(pos_bang, start + 10)
    assert _last_sentence_end_pos_syntok(text, start, end) is None

    # Another window inside the last sentence but excluding '?'
    start2 = pos_bang + 2
    end2 = pos_q  # exclude '?'
    assert _last_sentence_end_pos_syntok(text, start2, end2) is None


# --- Iterator/generator compatibility test (the bug you fixed) ---

@dataclass
class _Tok:
    offset: int
    value: str


def test_last_sentence_end_pos_sent_is_iterator(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Ensure no TypeError when `sent` is an iterator/generator (not subscriptable).
    We monkeypatch segmenter.process to yield sentences as token iterators.
    """
    # IMPORTANT: patch the segmenter object in the module where the function is defined
    import ce_pipeline.processing.repair as repair_mod

    # Any string is fine, offsets are absolute indices into this string
    text = "X" * 200

    # Sentence 1 ends at pos=9: token "abc." at offset 6, len 4 => 6+4-1=9
    # Sentence 2 ends at pos=49: token "zzz!" at offset 46, len 4 => 49
    sent1: List[_Tok] = [_Tok(0, "Hello"), _Tok(6, "abc.")]
    sent2: List[_Tok] = [_Tok(20, "More"), _Tok(46, "zzz!")]

    def fake_process(_: str) -> Iterable[Iterator[_Tok]]:
        yield iter(sent1)  # iterator, not list
        yield iter(sent2)

    monkeypatch.setattr(repair_mod.segmenter, "process", fake_process)

    assert _last_sentence_end_pos_syntok(text, 0, 80) == 49
    assert _last_sentence_end_pos_syntok(text, 0, 20) == 9
    assert _last_sentence_end_pos_syntok(text, 10, 45) is None
