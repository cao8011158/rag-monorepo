# tests/test_settings.py
from __future__ import annotations

from pathlib import Path
import pytest

from ce_pipeline.settings import load_settings


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "pipeline.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def test_load_settings_success_minimal(tmp_path: Path) -> None:
    # With current defaults:
    # - bm25.enabled=True
    # - vector.enabled=True
    # so outputs must include chunks + bm25_index + vector_index.
    p = _write_yaml(
        tmp_path,
        """
input:
  input_store: fs_local
  input_path: cleaned/latest/documents.jsonl

outputs:
  chunks:
    store: fs_local
    base: ce_out/chunks
  bm25_index:
    store: fs_local
    base: ce_out/indexes/bm25
  vector_index:
    store: fs_local
    base: ce_out/indexes/vector

stores:
  fs_local:
    kind: filesystem
    root: .
""",
    )

    s = load_settings(p)

    assert s["input"]["input_path"] == "cleaned/latest/documents.jsonl"
    assert s["outputs"]["chunks"]["base"] == "ce_out/chunks"

    assert "_meta" in s
    assert "config_hash" in s["_meta"]
    assert isinstance(s["_meta"]["config_hash"], str)
    assert len(s["_meta"]["config_hash"]) == 64


def test_missing_input_store_raises(tmp_path: Path) -> None:
    p = _write_yaml(
        tmp_path,
        """
input:
  input_store: ""
  input_path: cleaned/latest/documents.jsonl

outputs:
  chunks:
    store: fs_local
    base: ce_out/chunks
  bm25_index:
    store: fs_local
    base: ce_out/indexes/bm25
  vector_index:
    store: fs_local
    base: ce_out/indexes/vector

stores:
  fs_local:
    kind: filesystem
    root: .
""",
    )
    with pytest.raises(ValueError, match=r"input\.input_store is required"):
        load_settings(p)


def test_chunking_overlap_must_be_lt_window(tmp_path: Path) -> None:
    # This error happens before outputs checks, so outputs can be minimal.
    p = _write_yaml(
        tmp_path,
        """
input:
  input_store: fs_local
  input_path: cleaned/latest/documents.jsonl

chunking:
  window_chars: 200
  overlap_chars: 200
  min_chunk_chars: 10

outputs:
  chunks:
    store: fs_local
    base: ce_out/chunks

stores:
  fs_local:
    kind: filesystem
    root: .
""",
    )
    with pytest.raises(ValueError, match=r"chunking\.overlap_chars must be < window_chars"):
        load_settings(p)


def test_semantic_dedup_threshold_range(tmp_path: Path) -> None:
    # This error happens during dedup validation before outputs checks.
    p = _write_yaml(
        tmp_path,
        """
input:
  input_store: fs_local
  input_path: cleaned/latest/documents.jsonl

processing:
  dedup:
    semantic_dedup:
      enable: true
      threshold: 1.5

outputs:
  chunks:
    store: fs_local
    base: ce_out/chunks

stores:
  fs_local:
    kind: filesystem
    root: .
""",
    )
    with pytest.raises(ValueError, match=r"processing\.dedup\.semantic_dedup\.threshold must be in \[0, 1\]"):
        load_settings(p)


def test_outputs_require_chunks(tmp_path: Path) -> None:
    # outputs check begins with: if "chunks" not in outputs -> raise
    # so this will raise outputs.chunks is required regardless of indexing defaults.
    p = _write_yaml(
        tmp_path,
        """
input:
  input_store: fs_local
  input_path: cleaned/latest/documents.jsonl

outputs:
  bm25_index:
    store: fs_local
    base: ce_out/indexes/bm25

stores:
  fs_local:
    kind: filesystem
    root: .
""",
    )
    with pytest.raises(ValueError, match=r"outputs\.chunks is required"):
        load_settings(p)


def test_outputs_require_bm25_index_when_enabled(tmp_path: Path) -> None:
    # bm25 is enabled (explicitly true), vector defaults to enabled True,
    # so we provide vector_index but omit bm25_index to trigger the intended error.
    p = _write_yaml(
        tmp_path,
        """
input:
  input_store: fs_local
  input_path: cleaned/latest/documents.jsonl

indexing:
  bm25:
    enabled: true

outputs:
  chunks:
    store: fs_local
    base: ce_out/chunks
  vector_index:
    store: fs_local
    base: ce_out/indexes/vector

stores:
  fs_local:
    kind: filesystem
    root: .
""",
    )
    with pytest.raises(ValueError, match=r"outputs\.bm25_index is required when bm25 is enabled"):
        load_settings(p)


def test_outputs_require_vector_index_when_enabled(tmp_path: Path) -> None:
    # vector enabled (explicitly true), bm25 defaults to enabled True,
    # so we provide bm25_index but omit vector_index to trigger intended error.
    p = _write_yaml(
        tmp_path,
        """
input:
  input_store: fs_local
  input_path: cleaned/latest/documents.jsonl

indexing:
  vector:
    enabled: true

outputs:
  chunks:
    store: fs_local
    base: ce_out/chunks
  bm25_index:
    store: fs_local
    base: ce_out/indexes/bm25

stores:
  fs_local:
    kind: filesystem
    root: .
""",
    )
    with pytest.raises(ValueError, match=r"outputs\.vector_index is required when vector index is enabled"):
        load_settings(p)


def test_stores_must_include_referenced(tmp_path: Path) -> None:
    # Choose B: explicitly disable vector to avoid requiring outputs.vector_index,
    # so we can reach the stores-missing validation.
    p = _write_yaml(
        tmp_path,
        """
input:
  input_store: fs_local
  input_path: cleaned/latest/documents.jsonl

indexing:
  vector:
    enabled: false

outputs:
  chunks:
    store: fs_local
    base: ce_out/chunks
  bm25_index:
    store: bm25_store
    base: ce_out/indexes/bm25

stores:
  fs_local:
    kind: filesystem
    root: .
""",
    )
    with pytest.raises(ValueError, match=r"stores missing definitions for:"):
        load_settings(p)
