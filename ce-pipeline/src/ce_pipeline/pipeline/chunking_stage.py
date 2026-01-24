# src/ce_pipeline/pipeline/chunking_stage.py
from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, Optional, Callable, List, Tuple
import os
import traceback

from ce_pipeline.stores.base import Store
from ce_pipeline.stores.registry import build_store_registry
from ce_pipeline.io.jsonl import read_jsonl, append_jsonl
from ce_pipeline.chunking.chunker import chunk_doc

# 你已经实现好的 exact dedup（文件路径版）
from ce_pipeline.processing.exact_dedup import exact_dedup_jsonl_by_hash_meta


def _pjoin(*parts: str) -> str:
    """Join logical paths using POSIX style."""
    return "/".join([p.strip("/").replace("\\", "/") for p in parts if p is not None and str(p) != ""])


def _default_error_payload(
    *,
    stage: str,
    path: str,
    line_no: Optional[int] = None,
    error: str,
    raw: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "stage": stage,
        "path": path,
        "error": error,
    }
    if line_no is not None:
        payload["line_no"] = line_no
    if raw is not None:
        payload["raw"] = raw
    if extra:
        payload.update(extra)
    return payload


def run_chunking_stage(
    cfg: Dict[str, Any],
    *,
    fail_fast: bool = False,
) -> Dict[str, Any]:
    """
    Chunking Stage

    Steps:
    1) Read cleaned documents.jsonl from input store/path
    2) For each doc -> chunk_doc(doc, cfg) -> list[ChunkRecord]
    3) Stream-append all chunks into outputs.chunks.base/chunks.raw.jsonl
    4) Run exact_dedup on chunks.raw.jsonl -> outputs.chunks.base/chunks.jsonl

    Args:
        cfg: settings dict loaded from YAML (dynamic dict)
        fail_fast:
            - True: any bad JSONL line / bad doc / chunking error raises immediately
            - False: best-effort; errors are appended into errors.*.jsonl and processing continues

    Returns:
        summary dict: counts and artifact paths
    """
    stores = build_store_registry(cfg)

    # ---- input ----
    in_cfg = cfg["input"]
    in_store_name: str = in_cfg["input_store"]
    in_path: str = in_cfg["input_path"]
    in_store: Store = stores[in_store_name]

    # ---- output (chunks artifact) ----
    out_cfg = cfg["outputs"]["chunks"]
    out_store_name: str = out_cfg["store"]
    out_base: str = out_cfg["base"]
    out_store: Store = stores[out_store_name]

    # artifacts under base
    chunks_raw_rel = _pjoin(out_base, "chunks.raw.jsonl")
    chunks_final_rel = _pjoin(out_base, "chunks.jsonl")
    err_docs_rel = _pjoin(out_base, "errors.documents.jsonl")
    err_chunking_rel = _pjoin(out_base, "errors.chunking.jsonl")

    # reset outputs (overwrite to empty) so each run is deterministic
    out_store.write_bytes(chunks_raw_rel, b"")
    out_store.write_bytes(err_docs_rel, b"")
    out_store.write_bytes(err_chunking_rel, b"")

    def _log_doc_error(payload: Dict[str, Any]) -> None:
        append_jsonl(out_store, err_docs_rel, [payload])

    def _log_chunking_error(payload: Dict[str, Any]) -> None:
        append_jsonl(out_store, err_chunking_rel, [payload])

    # ---- read documents.jsonl ----
    on_read_error: Optional[Callable[[Dict[str, Any]], None]]
    if fail_fast:
        on_read_error = None
    else:
        # read_jsonl 的 on_error payload 结构由你 IO 文档定义
        def _on_read_error(payload: Dict[str, Any]) -> None:
            _log_doc_error(payload)
        on_read_error = _on_read_error

    docs_iter = read_jsonl(in_store, in_path, on_error=on_read_error)

    docs_total = 0
    docs_ok = 0
    chunks_total = 0
    docs_skipped = 0
    chunk_errors = 0

    # ---- chunking loop (streaming append) ----
    for doc in docs_iter:
        docs_total += 1

        # 基础“坏文档”策略：缺字段直接跳过（或 fail-fast）
        doc_id = str(doc.get("doc_id", "") or "")
        text = doc.get("text", None)

        if not doc_id or not isinstance(text, str) or not text.strip():
            if fail_fast:
                raise ValueError(f"Bad document record: doc_id/text invalid (doc_id={doc_id!r})")
            docs_skipped += 1
            _log_chunking_error(
                _default_error_payload(
                    stage="chunking_stage.validate_doc",
                    path=in_path,
                    error="Bad document record: missing/invalid doc_id or text",
                    extra={"doc_id": doc_id, "has_text": isinstance(text, str)},
                )
            )
            continue

        try:
            chunks: List[Dict[str, Any]] = chunk_doc(doc, cfg)
            if chunks:
                append_jsonl(out_store, chunks_raw_rel, chunks)
                chunks_total += len(chunks)
            docs_ok += 1
        except Exception as e:
            if fail_fast:
                raise
            chunk_errors += 1
            _log_chunking_error(
                _default_error_payload(
                    stage="chunking_stage.chunk_doc",
                    path=in_path,
                    error=f"{type(e).__name__}: {e}",
                    extra={
                        "doc_id": doc_id,
                        "traceback": traceback.format_exc(limit=20),
                    },
                )
            )
            continue

    # ---- exact dedup (reuse your existing exact_dedup.py) ----
    # 注意：你的 exact_dedup_jsonl_by_hash_meta 只接受“文件系统路径”
    # 所以这里要求 out_store 是 filesystem store（root 能映射到本地路径）
    # 我们通过常见字段/root 的方式尽量适配；不行就报错提示。
    out_root = getattr(out_store, "root", None)  # FilesystemStore(root=Path("."))
    if out_root is None:
        raise NotImplementedError(
            "exact_dedup_jsonl_by_hash_meta requires a filesystem-backed store "
            "(out_store should expose .root). For non-filesystem stores, implement a Store-based dedup stage."
        )

    # 把 logical path 转为本地路径：root + logical_rel
    # root 可能是 pathlib.Path
    root_str = str(out_root)
    raw_fs_path = os.path.abspath(os.path.join(root_str, chunks_raw_rel))
    final_fs_path = os.path.abspath(os.path.join(root_str, chunks_final_rel))

    hash_field = (
        cfg.get("processing", {})
          .get("dedup", {})
          .get("exact_dedup", {})
          .get("hash_field", "chunk_text_hash")
    )

    kept = exact_dedup_jsonl_by_hash_meta(
        raw_fs_path,
        final_fs_path,
        hash_field=hash_field,
        encoding="utf-8",
    )

    return {
        "input": {"store": in_store_name, "path": in_path},
        "output": {
            "store": out_store_name,
            "base": out_base,
            "chunks_raw": chunks_raw_rel,
            "chunks_final": chunks_final_rel,
            "errors_documents": err_docs_rel,
            "errors_chunking": err_chunking_rel,
        },
        "counts": {
            "docs_total": docs_total,
            "docs_ok": docs_ok,
            "docs_skipped": docs_skipped,
            "chunk_errors": chunk_errors,
            "chunks_total_raw": chunks_total,
            "chunks_kept_after_exact_dedup": kept,
        },
        "dedup": {"hash_field": hash_field},
    }
