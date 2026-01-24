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


def test_reads_top_level_chunking_knobs_and_passes_to_sliding_window(monkeypatch: pytest.MonkeyPatch) -> None:
    called: Dict[str, Any] = {}

    def fake_sliding_window_chunks(text: str, *, window_chars: int, overlap_chars: int) -> List[Tuple[int, str]]:
        called["text"] = text
        called["window_chars"] = window_chars
        called["overlap_chars"] = overlap_chars
        return [(0, "AAA")]

    monkeypatch.setattr(chunker, "sliding_window_chunks", fake_sliding_window_chunks)
    monkeypatch.setattr(chunker, "trim_noise_edges", lambda s: s)
    monkeypatch.setattr(chunker, "is_noise_chunk", lambda s, *, min_chunk_chars: False)
    monkeypatch.setattr(chunker, "repair_boundary_truncation", lambda prev, cur, nxt: cur)

    cfg = {"chunking": {"window_chars": 999, "overlap_chars": 111, "min_chunk_chars": 1}}
    out = chunker.chunk_doc(_base_doc("hello world"), cfg)

    assert called["window_chars"] == 999
    assert called["overlap_chars"] == 111
    assert len(out) == 1
    assert out[0]["chunk_index"] == 0


def test_boundary_repair_drops_middle_chunk_without_index_text_misalignment(monkeypatch: pytest.MonkeyPatch) -> None:
    # Create three windows: idx 0/1/2
    def fake_sliding_window_chunks(text: str, *, window_chars: int, overlap_chars: int) -> List[Tuple[int, str]]:
        return [(0, "AAA"), (1, "BBB"), (2, "CCC")]

    # No-op trim/noise
    monkeypatch.setattr(chunker, "sliding_window_chunks", fake_sliding_window_chunks)
    monkeypatch.setattr(chunker, "trim_noise_edges", lambda s: s)
    monkeypatch.setattr(chunker, "is_noise_chunk", lambda s, *, min_chunk_chars: False)

    # Repair makes the middle chunk empty -> should drop (idx=1) ONLY,
    # and should keep (0,"AAA") and (2,"CCC") aligned.
    def fake_repair(prev: str | None, cur: str, nxt: str | None) -> str:
        if cur == "BBB":
            return ""
        return cur

    monkeypatch.setattr(chunker, "repair_boundary_truncation", fake_repair)

    cfg = {"chunking": {"window_chars": 10, "overlap_chars": 0, "min_chunk_chars": 1}}
    out = chunker.chunk_doc(_base_doc("x"), cfg)

    assert [r["chunk_index"] for r in out] == [0, 2]
    assert [r[chunker.TEXT_FIELD] for r in out] == ["AAA", "CCC"]

    # Ensure chunk_id corresponds to correct (index,text) pair (no misalignment)
    assert out[0]["chunk_id"] == _expected_chunk_id("doc-1", 0, "AAA")
    assert out[1]["chunk_id"] == _expected_chunk_id("doc-1", 2, "CCC")


def test_skip_bad_doc_missing_doc_id(monkeypatch: pytest.MonkeyPatch) -> None:
    # Even if downstream functions exist, a bad doc should be skipped early.
    monkeypatch.setattr(chunker, "sliding_window_chunks", lambda *a, **k: [(0, "AAA")])
    monkeypatch.setattr(chunker, "trim_noise_edges", lambda s: s)
    monkeypatch.setattr(chunker, "is_noise_chunk", lambda s, *, min_chunk_chars: False)
    monkeypatch.setattr(chunker, "repair_boundary_truncation", lambda prev, cur, nxt: cur)

    doc = _base_doc("hello")
    doc["doc_id"] = ""  # bad
    out = chunker.chunk_doc(doc, {"chunking": {"min_chunk_chars": 1}})
    assert out == []


def test_skip_bad_doc_non_string_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chunker, "sliding_window_chunks", lambda *a, **k: [(0, "AAA")])
    monkeypatch.setattr(chunker, "trim_noise_edges", lambda s: s)
    monkeypatch.setattr(chunker, "is_noise_chunk", lambda s, *, min_chunk_chars: False)
    monkeypatch.setattr(chunker, "repair_boundary_truncation", lambda prev, cur, nxt: cur)

    doc = _base_doc("hello")
    doc["text"] = {"not": "a string"}  # bad
    out = chunker.chunk_doc(doc, {"chunking": {"min_chunk_chars": 1}})
    assert out == []
