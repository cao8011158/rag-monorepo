# src/ce_pipeline/pipeline/indexing_stage.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional, Callable
import os

from ce_pipeline.stores.base import Store
from ce_pipeline.stores.registry import build_store_registry
from ce_pipeline.io.jsonl import read_jsonl  # 你已实现：fail-fast / best-effort
from ce_pipeline.indexing.bm25 import build_bm25_index, save_bm25


def _posix_join(*parts: str) -> str:
    # Store 的逻辑路径统一按 POSIX 风格
    return "/".join([p.strip("/").replace("\\", "/") for p in parts if p is not None and str(p) != ""])


def _default_error_logger(store: Store, error_path: str) -> Callable[[Dict[str, Any]], None]:
    """
    把 read_jsonl 的坏行错误写到 errors.jsonl（一行一个 JSON 对象）
    注意：这里直接用 append_bytes，避免依赖你是否实现了 append_jsonl。
    """
    def _log(err: Dict[str, Any]) -> None:
        import json
        line = (json.dumps(err, ensure_ascii=False) + "\n").encode("utf-8")
        store.append_bytes(error_path, line)

    return _log


def run_indexing_stage(cfg: Dict[str, Any], *, best_effort: bool = True) -> Dict[str, Any]:
    """
    从 outputs.chunks.base/chunks.jsonl 读取 chunk_id + text，
    构建 BM25 并保存到 outputs.bm25_index.base/bm25.pkl

    参数
    - cfg: load_settings() 返回的动态 dict
    - best_effort:
        True  => read_jsonl 使用 on_error 记录坏行并跳过（不中断）
        False => fail-fast（任何坏行直接抛异常）

    返回
    - 运行统计信息 dict（便于 CLI 打印/日志）
    """
    # ---- build stores ----
    stores = build_store_registry(cfg)

    # ---- resolve inputs (chunks.jsonl) ----
    out_chunks = cfg["outputs"]["chunks"]
    chunks_store_name: str = out_chunks["store"]
    chunks_base: str = out_chunks["base"]

    chunks_store: Store = stores[chunks_store_name]
    chunks_rel = _posix_join(chunks_base, "chunks.jsonl")

    # ---- resolve outputs (bm25.pkl) ----
    out_bm25 = cfg["outputs"]["bm25_index"]
    bm25_store_name: str = out_bm25["store"]
    bm25_base: str = out_bm25["base"]

    bm25_store: Store = stores[bm25_store_name]
    bm25_rel = _posix_join(bm25_base, "bm25.pkl")

    # （可选）错误日志：默认写到 bm25 输出目录下
    errors_rel = _posix_join(bm25_base, "errors.indexing.read_chunks.jsonl")

    if not chunks_store.exists(chunks_rel):
        raise FileNotFoundError(f"chunks.jsonl not found: store='{chunks_store_name}', path='{chunks_rel}'")

    # ---- read chunks.jsonl -> (doc_ids, texts) ----
    doc_ids: List[str] = []
    texts: List[str] = []

    on_error = _default_error_logger(bm25_store, errors_rel) if best_effort else None

    for row in read_jsonl(chunks_store, chunks_rel, on_error=on_error):
        # 你要求：提取 chunk_id 和 text
        cid = row.get("chunk_id")
        txt = row.get("chunk_text")
        title = row.get("title")

        if not cid or not isinstance(cid, str):
            # 这里属于“坏数据”，在 best_effort 下直接跳过并记一条
            if best_effort and on_error is not None:
                on_error(
                    {
                        "stage": "indexing_stage.extract_fields",
                        "path": chunks_rel,
                        "line_no": row.get("_line_no", None),  # 如果你没提供，就为 None
                        "error": "MissingOrInvalidField: chunk_id",
                        "line_preview": str({k: row.get(k) for k in ("chunk_id", "text", "chunk_text")})[:200],
                    }
                )
                continue
            raise KeyError("Missing or invalid 'chunk_id' in chunk row")

        if not isinstance(txt, str) or not txt.strip():
            if best_effort and on_error is not None:
                on_error(
                    {
                        "stage": "indexing_stage.extract_fields",
                        "path": chunks_rel,
                        "line_no": row.get("_line_no", None),
                        "error": "MissingOrInvalidField: text",
                        "line_preview": str({k: row.get(k) for k in ("chunk_id", "text", "chunk_text")})[:200],
                    }
                )
                continue
            raise KeyError("Missing or invalid 'text' in chunk row")
        if isinstance(title, str) and title.strip():
            full_text = f"{title} {txt}"
        else:
            full_text = txt

        doc_ids.append(cid)
        texts.append(full_text)

    if not texts:
        raise ValueError(f"No valid chunks found in {chunks_rel}; cannot build BM25 index.")

    # ---- build + save bm25 ----
    bm25 = build_bm25_index(texts)
    save_bm25(bm25_store, bm25_rel, bm25=bm25, doc_ids=doc_ids)

    return {
        "stage": "indexing_stage",
        "chunks_store": chunks_store_name,
        "chunks_path": chunks_rel,
        "bm25_store": bm25_store_name,
        "bm25_path": bm25_rel,
        "errors_path": errors_rel if best_effort else None,
        "num_chunks_indexed": len(texts),
    }
