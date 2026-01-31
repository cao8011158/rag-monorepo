# src/ce_pipeline/chunking/chunker.py
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Tuple

from ce_pipeline.chunking.sliding_window import sliding_window_chunks
from ce_pipeline.processing.repair import repair_boundary_by_sentence_syntok

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

    Defensive defaults if missing.
    """
    c = (cfg or {}).get("chunking", {}) if isinstance(cfg, dict) else {}
    if not isinstance(c, dict):
        c = {}

    window_chars = int(c.get("window_chars", 1200))
    overlap_chars = int(c.get("overlap_chars", 200))
    min_chunk_chars = int(c.get("min_chunk_chars", 200))

    # sanity checks
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
      - Chunking config errors raise ValueError (fail fast).

    Input schema (doc):
      {
        "doc_id": str,          # REQUIRED
        "url": str,
        "title": str,
        "text": str,            # REQUIRED
        "source": str,
        "content_hash": str,
        "content_type": str,
        "fetched_at": str,
        "run_date": str
      }

    Output: List[ChunkRecord]
      {
        "chunk_id": str,              # sha256(doc_id::chunk_index::chunk_text)[:24]
        "doc_id": str,
        "chunk_index": int,
        "chunk_text": str,            # TEXT_FIELD
        "chunk_text_hash": str,       # sha256(chunk_text) full hex
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
        return []

    text = doc.get("text")
    if not isinstance(text, str):
        return []
    text = text.strip()
    if not text:
        return []

    # -----------------------
    # 1) Read chunking knobs
    # -----------------------
    window_chars, overlap_chars, min_chunk_chars = _read_chunking_knobs(cfg)

    # -----------------------
    # 2) Other doc fields (best-effort, optional)
    # -----------------------
    url = str(doc.get("url", "") or "")
    title = str(doc.get("title", "") or "")
    source = str(doc.get("source", "") or "")
    content_hash = str(doc.get("content_hash", "") or "")
    content_type = str(doc.get("content_type", "") or "")
    fetched_at = str(doc.get("fetched_at", "") or "")
    run_date = str(doc.get("run_date", "") or "")

    # -----------------------
    # 3) Split into raw windows
    # -----------------------
    parts: List[Tuple[int, str]] = sliding_window_chunks(
        text, window_chars=window_chars, overlap_chars=overlap_chars
    )

    # -----------------------
    # 4) strip -> minlen (KEEP PAIRS)
    #    (删除 trim_noise_edges / is_noise_chunk)
    # -----------------------
    kept: List[Tuple[int, str]] = []
    for idx, raw_chunk in parts:
        c = (raw_chunk or "").strip()
        if not c:
            continue
        if len(c) < min_chunk_chars:
            continue
        kept.append((int(idx), c))

    # -----------------------
    # 5) boundary repair (sliding pairwise):
    #    (c0,c1)->(c1,c2)->...; i 用 cur_fixed, i+1 用 next_fixed
    #    overlap = overlap_chars; back/forward 固定 50
    # -----------------------
    if len(kept) >= 2:
        idxs = [p[0] for p in kept]
        texts = [p[1] for p in kept]

        for i in range(len(texts) - 1):
            cur = texts[i]
            nxt = texts[i + 1]

            cur_fixed, nxt_fixed, _carry = repair_boundary_by_sentence_syntok(
                cur,
                nxt,
                overlap=overlap_chars,
                back_search=100,
                forward_search=100,
                # 用 min_chunk_chars 约束“修复后不能太短”，避免产出 < min_chunk_chars 的 chunk
                min_cur_len=min_chunk_chars,
            )

            # 仍然做一下基本规范化（不改变你的步骤语义，只是防止意外空白）
            cur_fixed = (cur_fixed or "").strip()
            nxt_fixed = (nxt_fixed or "").strip() if nxt_fixed is not None else ""

            # 由于 repair 内部已经用 min_cur_len 保护，这里不额外 drop；
            # 只要非空就更新
            if cur_fixed:
                texts[i] = cur_fixed
            if nxt_fixed:
                texts[i + 1] = nxt_fixed

        kept = list(zip(idxs, texts))

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
