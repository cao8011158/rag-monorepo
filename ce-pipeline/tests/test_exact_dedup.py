from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Callable

import pytest

import ce_pipeline.processing.exact_dedup as exact_mod


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _resolve_exact_func() -> Callable[..., int]:
    """
    Try to locate the exact dedup function from src/ce_pipeline/processing/exact_dedup.py

    Adjust here if your function name differs.
    """
    candidates = [
        "exact_dedup_jsonl_by_hash_meta",
        "exact_dedup_by_hash_meta",
        "exact_dedup_jsonl",
        "run_exact_dedup",
    ]
    for name in candidates:
        fn = getattr(exact_mod, name, None)
        if callable(fn):
            return fn

    raise AssertionError(
        "Cannot find an exact dedup function in ce_pipeline.processing.exact_dedup.\n"
        f"Tried: {candidates}\n"
        "Fix: rename your function to one of these, OR edit _resolve_exact_func() in this test."
    )


def test_exact_dedup_keeps_first_occurrence_and_preserves_order(tmp_path: Path) -> None:
    fn = _resolve_exact_func()

    in_path = tmp_path / "chunks.jsonl"
    out_path = tmp_path / "chunks.dedup.jsonl"

    rows = [
        {"chunk_id": "c0", "chunk_text_hash": "h1", "Text": "A"},
        {"chunk_id": "c1", "chunk_text_hash": "h2", "Text": "B"},
        {"chunk_id": "c2", "chunk_text_hash": "h1", "Text": "A (dup)"},
        {"chunk_id": "c3", "chunk_text_hash": "h3", "Text": "C"},
        {"chunk_id": "c4", "chunk_text_hash": "h2", "Text": "B (dup)"},
    ]
    _write_jsonl(in_path, rows)

    kept = fn(str(in_path), str(out_path))
    assert kept == 3

    out = _read_jsonl(out_path)
    assert [r["chunk_id"] for r in out] == ["c0", "c1", "c3"]
    assert [r["chunk_text_hash"] for r in out] == ["h1", "h2", "h3"]


def test_exact_dedup_raises_on_missing_hash_field(tmp_path: Path) -> None:
    fn = _resolve_exact_func()

    in_path = tmp_path / "chunks.jsonl"
    out_path = tmp_path / "chunks.dedup.jsonl"

    rows = [
        {"chunk_id": "c0", "chunk_text_hash": "h1", "Text": "A"},
        {"chunk_id": "c1", "Text": "B"},  # missing chunk_text_hash
    ]
    _write_jsonl(in_path, rows)

    with pytest.raises(KeyError):
        fn(str(in_path), str(out_path))


def test_exact_dedup_supports_custom_hash_field_if_exposed(tmp_path: Path) -> None:
    """
    If your exact dedup supports a custom hash field parameter,
    this test validates it. If not, we skip.
    """
    fn = _resolve_exact_func()

    in_path = tmp_path / "chunks.jsonl"
    out_path = tmp_path / "chunks.dedup.jsonl"

    rows = [
        {"chunk_id": "c0", "content_fingerprint": "x1", "Text": "A"},
        {"chunk_id": "c1", "content_fingerprint": "x1", "Text": "A (dup)"},
        {"chunk_id": "c2", "content_fingerprint": "x2", "Text": "B"},
    ]
    _write_jsonl(in_path, rows)

    # If your function signature includes hash_field=..., run it; otherwise skip.
    try:
        kept = fn(str(in_path), str(out_path), hash_field="content_fingerprint")
    except TypeError:
        pytest.skip("exact dedup function does not accept hash_field parameter.")
        return

    assert kept == 2
    out = _read_jsonl(out_path)
    assert [r["chunk_id"] for r in out] == ["c0", "c2"]


def test_exact_dedup_raises_on_bad_json_line(tmp_path: Path) -> None:
    fn = _resolve_exact_func()

    in_path = tmp_path / "chunks.jsonl"
    out_path = tmp_path / "chunks.dedup.jsonl"

    with open(in_path, "w", encoding="utf-8") as f:
        f.write('{"chunk_id":"c0","chunk_text_hash":"h1","Text":"A"}\n')
        f.write("{BAD JSON}\n")

    with pytest.raises(ValueError):
        fn(str(in_path), str(out_path))
