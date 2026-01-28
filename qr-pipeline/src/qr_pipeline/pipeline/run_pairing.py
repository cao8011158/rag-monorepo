# src/qr_pipeline/pipeline/run_pairing.py
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, TypedDict, cast

from qr_pipeline.settings import load_settings
from qr_pipeline.stores.registry import build_store_registry
from qr_pipeline.io.jsonl import read_jsonl, write_jsonl, append_jsonl
from qr_pipeline.llm.retrieval import HybridRetriever
from qr_pipeline.llm.pairing import build_pairs_for_query


# -----------------------------
# Types (match new pairing API)
# -----------------------------
class ChunkDoc(TypedDict, total=False):
    chunk_id: str
    doc_id: str
    chunk_index: int
    chunk_text: str
    chunk_text_hash: str
    # optional passthrough fields:
    url: str
    title: str
    source: str
    content_hash: str
    content_type: str
    fetched_at: str
    run_date: str


class Query(TypedDict, total=False):
    query_text: str
    source_chunk_ids: List[str]
    query_id: str
    domain: str


class QueryPack(TypedDict, total=False):
    query: Query
    positives: List[ChunkDoc]
    negatives: List[ChunkDoc]
    meta: Dict[str, Any]


@dataclass
class RunPairingResult:
    num_queries_total: int
    num_queries_processed: int
    num_packs_written: int
    num_errors: int
    timing: Dict[str, float]


# -----------------------------
# Helpers
# -----------------------------
def _join_posix(base: str, rel: str) -> str:
    base = (base or "").strip().strip("/")
    rel = (rel or "").strip().lstrip("/")
    return f"{base}/{rel}" if base else rel


def _must_get(d: Dict[str, Any], path: Sequence[str]) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"Missing config key: {'.'.join(path)} (stuck at '{k}')")
        cur = cur[k]
    return cur


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _now() -> float:
    return time.perf_counter()


def _norm_text_for_hash(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _chunk_text_hash_from_text(text: str) -> str:
    return _sha1_hex(_norm_text_for_hash(text))


def _read_query_rows(
    store: Any,
    path: str,
    *,
    on_error: Optional[Callable[[Dict[str, Any]], None]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in read_jsonl(store, path, on_error=on_error):
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _row_to_query(row: Dict[str, Any]) -> Query:
    # JSONL: query_text_norm -> pairing Query: query_text
    qtext = _safe_str(row.get("query_text_norm") or row.get("query_text")).strip()

    src_ids = [
        _safe_str(x).strip()
        for x in (row.get("source_chunk_ids") or [])
        if _safe_str(x).strip()
    ]

    q: Query = {"query_text": qtext, "source_chunk_ids": src_ids}

    qid = _safe_str(row.get("query_id")).strip()
    if qid:
        q["query_id"] = qid

    domain = _safe_str(row.get("domain")).strip()
    if domain:
        q["domain"] = domain

    return q


def _collect_needed_source_chunk_ids(query_rows: Sequence[Dict[str, Any]]) -> List[str]:
    """
    只加载 pairing 需要的 source_doc（单个 ChunkDoc）。
    这里按约定取 source_chunk_ids[0] 作为 source_doc 的 chunk_id；
    但 QueryPack.query.source_chunk_ids 会保留 query 行里完整列表。
    """
    need: List[str] = []
    seen: set[str] = set()
    for row in query_rows:
        ids = row.get("source_chunk_ids") or []
        if not ids:
            continue
        cid = _safe_str(ids[0]).strip()
        if cid and cid not in seen:
            seen.add(cid)
            need.append(cid)
    return need


def _coerce_chunkdoc_from_chunks_row(row: Dict[str, Any]) -> ChunkDoc:
    chunk_id = _safe_str(row.get("chunk_id") or row.get("key")).strip()
    chunk_text = _safe_str(row.get("chunk_text")).strip()

    doc: ChunkDoc = {
        "chunk_id": chunk_id,
        "doc_id": _safe_str(row.get("doc_id") or chunk_id).strip() or chunk_id,
        "chunk_index": int(row.get("chunk_index") or 0),
        "chunk_text": chunk_text,
        "chunk_text_hash": _safe_str(row.get("chunk_text_hash")).strip()
        or _chunk_text_hash_from_text(chunk_text),
    }

    for k in ["url", "title", "source", "content_hash", "content_type", "fetched_at", "run_date"]:
        if k in row and row.get(k) is not None:
            doc[k] = _safe_str(row.get(k))
    return doc


def _build_source_chunk_map(
    *,
    chunks_store: Any,
    chunks_jsonl_path: str,
    needed_ids: Sequence[str],
    on_error: Optional[Callable[[Dict[str, Any]], None]],
) -> Dict[str, ChunkDoc]:
    need_set = set(needed_ids)
    out: Dict[str, ChunkDoc] = {}

    for row in read_jsonl(chunks_store, chunks_jsonl_path, on_error=on_error):
        if not isinstance(row, dict):
            continue
        cid = _safe_str(row.get("chunk_id")).strip()
        if not cid or cid not in need_set or cid in out:
            continue
        out[cid] = _coerce_chunkdoc_from_chunks_row(row)
        if len(out) >= len(need_set):
            break

    return out


def _coerce_chunkdoc_from_retrieval_item(it: Dict[str, Any]) -> ChunkDoc:
    chunk_id = _safe_str(it.get("key")).strip()
    chunk_text = _safe_str(it.get("chunk_text")).strip()

    doc: ChunkDoc = {
        "chunk_id": chunk_id,
        "doc_id": _safe_str(it.get("doc_id") or chunk_id).strip() or chunk_id,
        "chunk_index": int(it.get("chunk_index") or 0),
        "chunk_text": chunk_text,
        "chunk_text_hash": _safe_str(it.get("chunk_text_hash")).strip()
        or _chunk_text_hash_from_text(chunk_text),
    }

    for k in ["url", "title", "source", "content_hash", "content_type", "fetched_at", "run_date"]:
        if k in it and it.get(k) is not None:
            doc[k] = _safe_str(it.get(k))
    return doc


def _convert_retrieval_items_to_chunkdocs(items: Sequence[Dict[str, Any]]) -> List[ChunkDoc]:
    out: List[ChunkDoc] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if not _safe_str(it.get("key")).strip():
            continue
        out.append(_coerce_chunkdoc_from_retrieval_item(it))
    return out


def _flush_buffer(
    *,
    out_store: Any,
    pairs_path: str,
    buf: List[QueryPack],
) -> int:
    if not buf:
        return 0
    append_jsonl(out_store, pairs_path, cast(Iterable[Dict[str, Any]], buf))
    n = len(buf)
    buf.clear()
    return n


# -----------------------------
# Main callable API
# -----------------------------
def run_pairing(
    config_path: str,
    *,
    llm: Any,
    embedder: Any,
    buffer_size: int = 15,
    best_effort: bool = True,
) -> RunPairingResult:
    """
    只从 outputs.files.queries_in_domain 读取 query。
    每个 query 输出 1 条 QueryPack，写入 outputs.files.pairs。
    缓冲 buffer_size 条后 append，一次 flush 打印一次状态。
    """
    t0 = _now()
    settings: Dict[str, Any] = load_settings(config_path)
    stores = build_store_registry(settings)

    # -------- outputs --------
    out_store_name = _must_get(settings, ["outputs", "store"])
    out_base = _must_get(settings, ["outputs", "base"])
    out_files = _must_get(settings, ["outputs", "files"])
    out_store = stores[out_store_name]

    queries_in_rel = _must_get(out_files, ["queries_in_domain"])
    pairs_rel = _must_get(out_files, ["pairs"])
    stats_rel = _must_get(out_files, ["stats"])
    errors_rel = _must_get(out_files, ["errors"])

    queries_path = _join_posix(out_base, queries_in_rel)
    pairs_path = _join_posix(out_base, pairs_rel)
    stats_path = _join_posix(out_base, stats_rel)
    errors_path = _join_posix(out_base, errors_rel)

    # -------- inputs: ce chunks --------
    chunks_store_name = _must_get(settings, ["inputs", "ce_artifacts", "chunks", "store"])
    chunks_base = _must_get(settings, ["inputs", "ce_artifacts", "chunks", "base"])
    chunks_file = _must_get(settings, ["inputs", "ce_artifacts", "chunks", "chunks_file"])
    chunks_store = stores[chunks_store_name]
    chunks_path = _join_posix(chunks_base, chunks_file)

    # -------- error logger --------
    def _log_err(payload: Dict[str, Any]) -> None:
        try:
            append_jsonl(out_store, errors_path, [payload])
        except Exception:
            pass

    on_error = _log_err if best_effort else None

    # -------- build retriever (once) --------
    t_retr0 = _now()
    retriever = HybridRetriever.from_settings(settings)
    t_retr1 = _now()

    # -------- read queries (in_domain only) --------
    t_q0 = _now()
    query_rows = _read_query_rows(out_store, queries_path, on_error=on_error)
    t_q1 = _now()

    # -------- build source map (only needed ids) --------
    needed_ids = _collect_needed_source_chunk_ids(query_rows)
    t_s0 = _now()
    source_map = _build_source_chunk_map(
        chunks_store=chunks_store,
        chunks_jsonl_path=chunks_path,
        needed_ids=needed_ids,
        on_error=on_error,
    )
    t_s1 = _now()

    # -------- reset outputs --------
    try:
        write_jsonl(out_store, pairs_path, [])
    except Exception:
        pass
    try:
        write_jsonl(out_store, errors_path, [])
    except Exception:
        pass

    num_processed = 0
    num_written = 0
    num_errors = 0
    buf: List[QueryPack] = []
    last_flush_t = _now()

    t_loop0 = _now()
    for idx, row in enumerate(query_rows, start=1):
        try:
            query_obj = _row_to_query(row)
            qtext = _safe_str(query_obj.get("query_text")).strip()
            if not qtext:
                continue

            src_ids_full = row.get("source_chunk_ids") or []
            if not src_ids_full:
                raise ValueError("query row missing source_chunk_ids")

            src_id0 = _safe_str(src_ids_full[0]).strip()
            if not src_id0:
                raise ValueError("source_chunk_ids[0] is empty")

            if src_id0 not in source_map:
                raise KeyError(f"source_chunk_id not found in chunks.jsonl: {src_id0}")

            source_doc = source_map[src_id0]

            # retrieval -> candidate ChunkDocs (RRF order preserved)
            retrieved_items: List[Dict[str, Any]] = retriever.retrieve(qtext)
            candidate_docs = _convert_retrieval_items_to_chunkdocs(retrieved_items)

            # pairing (new API): returns (pack, stats)
            pack, stats = build_pairs_for_query(
                query=query_obj,
                source_doc=source_doc,
                candidate_docs=candidate_docs,
                llm=llm,
                embedder=embedder,
            )

            # ensure stats is present in pack.meta.stats
            if isinstance(pack, dict):
                meta = pack.get("meta") if isinstance(pack.get("meta"), dict) else {}
                if not isinstance(meta.get("stats"), dict):
                    meta["stats"] = stats
                pack["meta"] = meta

            buf.append(cast(QueryPack, pack))
            num_processed += 1

            if len(buf) >= int(buffer_size):
                n = _flush_buffer(out_store=out_store, pairs_path=pairs_path, buf=buf)
                num_written += n
                now = _now()
                print(
                    f"[run_pairing] flushed={n} processed={num_processed}/{len(query_rows)} "
                    f"written={num_written} errors={num_errors} since_last_flush_sec={now - last_flush_t:.2f}"
                )
                last_flush_t = now

        except Exception as e:
            num_errors += 1
            payload = {
                "stage": "run_pairing",
                "error": f"{type(e).__name__}: {e}",
                "query_id": _safe_str(row.get("query_id")),
                "query_text_norm": _safe_str(row.get("query_text_norm"))[:300],
            }
            if best_effort:
                _log_err(payload)
            else:
                raise

    # final flush
    if buf:
        n = _flush_buffer(out_store=out_store, pairs_path=pairs_path, buf=buf)
        num_written += n
        print(
            f"[run_pairing] flushed_final={n} processed={num_processed}/{len(query_rows)} "
            f"written={num_written} errors={num_errors}"
        )

    t_loop1 = _now()

    timing = {
        "build_retriever_sec": float(t_retr1 - t_retr0),
        "read_queries_sec": float(t_q1 - t_q0),
        "build_source_map_sec": float(t_s1 - t_s0),
        "process_queries_sec": float(t_loop1 - t_loop0),
        "total_sec": float(_now() - t0),
    }

    stats_obj: Dict[str, Any] = {
        "stage": "run_pairing",
        "num_queries_total": len(query_rows),
        "num_queries_processed": num_processed,
        "num_packs_written": num_written,
        "num_errors": num_errors,
        "timing": timing,
        "inputs": {"queries_path": queries_path, "chunks_path": chunks_path},
        "outputs": {"pairs_path": pairs_path, "errors_path": errors_path, "stats_path": stats_path},
        "meta": settings.get("_meta", {}),
    }

    out_store.write_text(stats_path, json.dumps(stats_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    return RunPairingResult(
        num_queries_total=len(query_rows),
        num_queries_processed=num_processed,
        num_packs_written=num_written,
        num_errors=num_errors,
        timing=timing,
    )


# -----------------------------
# CLI
# -----------------------------
def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Run pairing (new API): queries_in_domain -> QueryPack jsonl.")
    p.add_argument("--config", required=True, help="Path to pipeline yaml, e.g. configs/pipeline.yaml")
    p.add_argument("--buffer", type=int, default=15, help="Flush buffer size (QueryPack rows)")
    p.add_argument("--fail_fast", action="store_true", help="Fail fast instead of best-effort.")
    args = p.parse_args(list(argv) if argv is not None else None)

    settings = load_settings(args.config)

    # LLM config: models.llm
    models_cfg = settings.get("models", {}) if isinstance(settings.get("models", {}), dict) else {}
    llm_cfg = models_cfg.get("llm", {}) if isinstance(models_cfg.get("llm", {}), dict) else {}

    from qr_pipeline.llm.hf_transformers_llm import HFTransformersLLM

    llm = HFTransformersLLM(
        model_name=_safe_str(llm_cfg.get("model_name", "")),
        device=_safe_str(llm_cfg.get("device", "cpu")) or "cpu",
        cache_dir=llm_cfg.get("cache_dir"),
        max_new_tokens=int(llm_cfg.get("max_new_tokens", 128) or 128),
        temperature=float(llm_cfg.get("temperature", 0.7) or 0.7),
        top_p=float(llm_cfg.get("top_p", 0.9) or 0.9),
    )
    llm.load()

    # Embedder config: embedding
    from qr_pipeline.processing.embedder import DualInstructEmbedder

    emb_cfg = settings.get("embedding", {}) if isinstance(settings.get("embedding", {}), dict) else {}
    instr_cfg = emb_cfg.get("instructions", {}) if isinstance(emb_cfg.get("instructions", {}), dict) else {}

    embedder = DualInstructEmbedder(
        model_name=_safe_str(emb_cfg.get("model_name", "")),
        passage_instruction=_safe_str(instr_cfg.get("passage", "")),
        query_instruction=_safe_str(instr_cfg.get("query", "")),
        batch_size=int(emb_cfg.get("batch_size", 64) or 64),
        normalize_embeddings=bool(emb_cfg.get("normalize_embeddings", True)),
        device=emb_cfg.get("device"),  # None / "cpu" / "cuda"
    )

    res = run_pairing(
        args.config,
        llm=llm,
        embedder=embedder,
        buffer_size=int(args.buffer),
        best_effort=(not args.fail_fast),
    )

    print(
        f"[run_pairing] processed={res.num_queries_processed}/{res.num_queries_total}, "
        f"packs_written={res.num_packs_written}, errors={res.num_errors}, total_sec={res.timing.get('total_sec'):.2f}"
    )


if __name__ == "__main__":
    main()
