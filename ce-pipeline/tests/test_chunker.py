# tests/test_chunker.py
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Tuple

import pytest

import ce_pipeline.chunking.chunker as chunker


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _expected_chunk_id(doc_id: str, chunk_index: int, chunk_text: str) -> str:
    raw = f"{doc_id}::{chunk_index}::{chunk_text}"
    return _sha256_hex(raw)[:24]


def _base_doc(text: str) -> Dict[str, Any]:
    return {
        "doc_id": "doc-1",
        "url": "https://example.com/a",
        "title": "Title A",
        "text": text,
        "source": "example",
        "content_hash": "ch_aaa",
        "content_type": "text/html",
        "fetched_at": "2026-01-22T00:00:00Z",
        "run_date": "2026-01-22",
    }


def test_reads_top_level_chunking_knobs_and_passes_to_sliding_window_and_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    called: Dict[str, Any] = {}

    def fake_sliding_window_chunks(text: str, *, window_chars: int, overlap_chars: int) -> List[Tuple[int, str]]:
        called["text"] = text
        called["window_chars"] = window_chars
        called["overlap_chars"] = overlap_chars
        # single chunk => repair should NOT be called
        return [(0, "AAA")]

    monkeypatch.setattr(chunker, "sliding_window_chunks", fake_sliding_window_chunks)

    def fake_repair(cur_text: str, next_text: str | None, *, overlap: int, back_search: int, forward_search: int, min_cur_len: int):
        # Should not be called because only 1 kept chunk
        raise AssertionError("repair should not be called for single-chunk docs")

    monkeypatch.setattr(chunker, "repair_boundary_by_sentence_syntok", fake_repair)

    cfg = {"chunking": {"window_chars": 999, "overlap_chars": 111, "min_chunk_chars": 1}}
    out = chunker.chunk_doc(_base_doc("hello world"), cfg)

    assert called["window_chars"] == 999
    assert called["overlap_chars"] == 111
    assert len(out) == 1
    assert out[0]["chunk_index"] == 0
    assert out[0][chunker.TEXT_FIELD] == "AAA"


def test_pairwise_boundary_repair_updates_adjacent_chunks_without_index_text_misalignment(monkeypatch: pytest.MonkeyPatch) -> None:
    # Create three windows: idx 0/1/2
    def fake_sliding_window_chunks(text: str, *, window_chars: int, overlap_chars: int) -> List[Tuple[int, str]]:
        return [(0, "AAA"), (1, "BBB"), (2, "CCC")]

    monkeypatch.setattr(chunker, "sliding_window_chunks", fake_sliding_window_chunks)

    # Pairwise repair semantics:
    # (AAA, BBB) -> (AAA1, BBB1)
    # (BBB1, CCC) -> (BBB2, CCC2)
    # Final expected texts: [AAA1, BBB2, CCC2]
    # Additionally, simulate a "bad" middle result once: return empty for BBB in first call,
    # which chunker should treat as "do not update" (it keeps previous text).
    calls: List[Tuple[str, str | None]] = []

    def fake_repair(
        cur_text: str,
        next_text: str | None,
        *,
        overlap: int,
        back_search: int,
        forward_search: int,
        min_cur_len: int,
    ) -> Tuple[str, str | None, str]:
        calls.append((cur_text, next_text))
        assert overlap == 0  # we'll set overlap_chars=0 in cfg below
        assert back_search == 50
        assert forward_search == 50
        assert min_cur_len == 1

        # First pair: (AAA, BBB)
        if cur_text == "AAA" and next_text == "BBB":
            # return empty next_fixed to simulate "no update" for next in this step
            return ("AAA1", "", "")
        # Second pair: (BBB, CCC) OR (BBB?, CCC) depending on whether BBB updated
        if (cur_text in ("BBB", "BBB1")) and next_text == "CCC":
            return ("BBB2", "CCC2", "")
        # Fallback: no changes
        return (cur_text, next_text, "")

    monkeypatch.setattr(chunker, "repair_boundary_by_sentence_syntok", fake_repair)

    cfg = {"chunking": {"window_chars": 10, "overlap_chars": 0, "min_chunk_chars": 1}}
    out = chunker.chunk_doc(_base_doc("x"), cfg)

    # Index alignment must be preserved
    assert [r["chunk_index"] for r in out] == [0, 1, 2]

    # Text outcome: first chunk updated; middle chunk updated in second pass; last updated
    assert [r[chunker.TEXT_FIELD] for r in out] == ["AAA1", "BBB2", "CCC2"]

    # chunk_id corresponds to correct (index,text) pair
    assert out[0]["chunk_id"] == _expected_chunk_id("doc-1", 0, "AAA1")
    assert out[1]["chunk_id"] == _expected_chunk_id("doc-1", 1, "BBB2")
    assert out[2]["chunk_id"] == _expected_chunk_id("doc-1", 2, "CCC2")

    # Ensure repair was called exactly twice: (0,1) and (1,2)
    assert calls == [("AAA", "BBB"), ("BBB", "CCC")]


def test_skip_bad_doc_missing_doc_id(monkeypatch: pytest.MonkeyPatch) -> None:
    # Even if downstream functions exist, a bad doc should be skipped early.
    monkeypatch.setattr(chunker, "sliding_window_chunks", lambda *a, **k: [(0, "AAA")])

    # Repair shouldn't matter here; doc gets skipped before reaching it.
    monkeypatch.setattr(
        chunker,
        "repair_boundary_by_sentence_syntok",
        lambda *a, **k: ("", None, ""),
    )

    doc = _base_doc("hello")
    doc["doc_id"] = ""  # bad
    out = chunker.chunk_doc(doc, {"chunking": {"min_chunk_chars": 1}})
    assert out == []


def test_skip_bad_doc_non_string_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chunker, "sliding_window_chunks", lambda *a, **k: [(0, "AAA")])
    monkeypatch.setattr(
        chunker,
        "repair_boundary_by_sentence_syntok",
        lambda *a, **k: ("", None, ""),
    )

    doc = _base_doc("hello")
    doc["text"] = {"not": "a string"}  # bad
    out = chunker.chunk_doc(doc, {"chunking": {"min_chunk_chars": 1}})
    assert out == []
