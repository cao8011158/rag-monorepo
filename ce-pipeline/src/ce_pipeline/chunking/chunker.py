# src/ce_pipeline/chunking/chunker.py
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Tuple

from ce_pipeline.chunking.sliding_window import sliding_window_chunks
from ce_pipeline.processing import (
    trim_noise_edges,
    is_noise_chunk,
    repair_boundary_truncation,
)


TEXT_FIELD = "chunk_text"


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _stable_chunk_id(doc_id: str, chunk_index: int, chunk_text: str) -> str:
    """
    chunk_id = sha256(doc_id + "::" + chunk_index + "::" + chunk_text)[:24]
    """
    raw = f"{doc_id}::{chunk_index}::{chunk_text}"
    return _sha256_hex(raw)[:24]


def _chunk_text_hash(chunk_text: str) -> str:
    """
    Stable content hash for exact dedup (full sha256 hex).
    """
    return _sha256_hex(chunk_text)


def _read_chunking_knobs(cfg: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Read TOP-LEVEL chunking knobs per Settings API:
      window_chars = cfg["chunking"]["window_chars"]
      overlap_chars = cfg["chunking"]["overlap_chars"]
      min_chunk_chars = cfg["chunking"]["min_chunk_chars"]

    Returns: (window_chars, overlap_chars, min_chunk_chars)

    This function is defensive: missing keys fall back to defaults.
    """
    c = (cfg or {}).get("chunking", {}) if isinstance(cfg, dict) else {}
    if not isinstance(c, dict):
        c = {}

    window_chars = int(c.get("window_chars", 1200))
    overlap_chars = int(c.get("overlap_chars", 200))
    min_chunk_chars = int(c.get("min_chunk_chars", 200))

    # sanity checks (avoid pathological chunking configs)
    if window_chars <= 0:
        raise ValueError("chunking.window_chars must be > 0")
    if overlap_chars < 0:
        raise ValueError("chunking.overlap_chars must be >= 0")
    if overlap_chars >= window_chars:
        raise ValueError("chunking.overlap_chars must be < chunking.window_chars")
    if min_chunk_chars <= 0:
        raise ValueError("chunking.min_chunk_chars must be > 0")

    return window_chars, overlap_chars, min_chunk_chars


def chunk_doc(doc: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert ONE cleaned document record to a list of ChunkRecord JSON objects.

    Best-effort policy:
      - If the document is invalid (missing doc_id/text), return [] (skip).
      - Chunking config errors (invalid knobs) raise ValueError (fail fast),
        because that is a pipeline configuration bug, not data quality.

    Input schema (doc):
      {
        "doc_id": str,          # REQUIRED (stable identity)
        "url": str,
        "title": str,
        "text": str,            # REQUIRED (non-empty)
        "source": str,
        "content_hash": str,
        "content_type": str,
        "fetched_at": str,k_
        "run_date": str
      }

    Output: List[ChunkRecord]
      {
        "chunk_id": str,                 # sha256(doc_id::chunk_index::chunk_text)[:24]
        "doc_id": str,
        "chunk_index": int,              # document-internal window id
        "chunk_text": str,           # chunk text content (TEXT_FIELD)
        "chunk_text_hash": str,          # sha256(chunk_text) full hex
        "url": str,
        "title": str,
        "source": str,
        "content_hash": str,
        "content_type": str,
        "fetched_at": str,
        "run_date": str,
      }
    """
    # -----------------------
    # 0) Validate + normalize doc (skip bad docs)
    # -----------------------
    if not isinstance(doc, dict):
        return []

    doc_id = str(doc.get("doc_id", "") or "").strip()
    if not doc_id:
        # Without doc_id, chunk_id becomes unstable / collision-prone.
        return []

    text = doc.get("text")
    if not isinstance(text, str):
        return []
    text = text.strip()
    if not text:
        return []

    # -----------------------
    # 1) Read chunking knobs (TOP-LEVEL per Settings API)
    # -----------------------
    window_chars, overlap_chars, min_chunk_chars = _read_chunking_knobs(cfg)

    # -----------------------
    # 2) Other doc fields (best-effort, optional)
    # -----------------------
    url = str(doc.get("url", "") or "")
    title = str(doc.get("title", "") or "")
    source = doc.get("source", "") or ""
    content_hash = doc.get("content_hash", "") or ""
    content_type = doc.get("content_type", "") or ""
    fetched_at = doc.get("fetched_at", "") or ""
    run_date = doc.get("run_date", "") or ""

    # -----------------------
    # 3) Split into raw windows
    # -----------------------
    parts: List[Tuple[int, str]] = sliding_window_chunks(
        text, window_chars=window_chars, overlap_chars=overlap_chars
    )

    # -----------------------
    # 4) trim -> minlen -> noise_filter (KEEP PAIRS)
    # -----------------------
    kept: List[Tuple[int, str]] = []
    for idx, raw_chunk in parts:
        c = (raw_chunk or "").strip()
        if not c:
            continue

        # ALWAYS do edge-only trim first
        c = trim_noise_edges(c).strip()
        if not c:
            continue

        # min length gate
        if len(c) < min_chunk_chars:
            continue

        # ALWAYS do noise filter
        if is_noise_chunk(c, min_chunk_chars=min_chunk_chars):
            continue

        kept.append((int(idx), c))

    # -----------------------
    # 5) boundary repair (PRESERVE idx-text alignment)
    # -----------------------
    if kept:
        repaired_pairs: List[Tuple[int, str]] = []
        for i, (idx, cur) in enumerate(kept):
            prev_t = kept[i - 1][1] if i > 0 else None
            next_t = kept[i + 1][1] if i + 1 < len(kept) else None
            new_t = (repair_boundary_truncation(prev_t, cur, next_t) or "").strip()
            if new_t:
                repaired_pairs.append((idx, new_t))
        kept = repaired_pairs

    # -----------------------
    # 6) emit ChunkRecords
    # -----------------------
    out: List[Dict[str, Any]] = []
    for chunk_index, chunk_text in kept:
        cid = _stable_chunk_id(doc_id, chunk_index, chunk_text)
        th = _chunk_text_hash(chunk_text)

        row: Dict[str, Any] = {
            "chunk_id": cid,
            "doc_id": doc_id,
            "chunk_index": int(chunk_index),
            "chunk_text_hash": th,
            "url": url,
            "title": title,
            "source": source,
            "content_hash": content_hash,
            "content_type": content_type,
            "fetched_at": fetched_at,
            "run_date": run_date,
        }
        row[TEXT_FIELD] = chunk_text
        out.append(row)

    return out
