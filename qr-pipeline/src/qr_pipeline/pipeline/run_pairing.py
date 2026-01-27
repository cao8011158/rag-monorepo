# src/qr_pipeline/pipeline/run_pairing.py
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, TypedDict

from qr_pipeline.settings import load_settings
from qr_pipeline.stores.registry import build_store_registry
from qr_pipeline.io.jsonl import read_jsonl, write_jsonl, append_jsonl
from qr_pipeline.llm.pairing import build_pairs_for_query
from qr_pipeline.llm.retrieval import HybridRetriever


# -----------------------------
# Types
# -----------------------------
class Document(TypedDict):
    doc_id: str
    text: str


class QueryRow(TypedDict, total=False):
    query_id: str
    query_text_norm: str
    source_chunk_ids: List[str]
    llm_model: str
    prompt_style: str
    domain: str  # "in" / "out"


@dataclass
class RunPairingResult:
    num_queries_total: int
    num_queries_processed: int
    num_pairs_written: int
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


def _read_queries_in_domain(
    store: Any,
    path: str,
    *,
    on_error: Optional[Callable[[Dict[str, Any]], None]],
) -> List[QueryRow]:
    rows: List[QueryRow] = []
    for row in read_jsonl(store, path, on_error=on_error):
        if isinstance(row, dict):
            rows.append(row)  # type: ignore[assignment]
    return rows


def _collect_needed_source_chunk_ids(queries: Sequence[QueryRow]) -> List[str]:
    need: List[str] = []
    seen: set[str] = set()
    for q in queries:
        ids = q.get("source_chunk_ids") or []
        if not ids:
            continue
        cid = _safe_str(ids[0]).strip()
        if cid and cid not in seen:
            seen.add(cid)
            need.append(cid)
    return need


def _build_source_chunk_map(
    *,
    chunks_store: Any,
    chunks_jsonl_path: str,
    needed_ids: Sequence[str],
    on_error: Optional[Callable[[Dict[str, Any]], None]],
) -> Dict[str, Document]:
    need_set = set(needed_ids)
    out: Dict[str, Document] = {}

    for row in read_jsonl(chunks_store, chunks_jsonl_path, on_error=on_error):
        if not isinstance(row, dict):
            continue
        cid = _safe_str(row.get("chunk_id")).strip()
        if cid in need_set and cid not in out:
            out[cid] = {"doc_id": cid, "text": _safe_str(row.get("chunk_text"))}
            if len(out) >= len(need_set):
                break

    return out


def _convert_retrieval_items_to_docs(items: Sequence[Dict[str, Any]]) -> List[Document]:
    docs: List[Document] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        key = _safe_str(it.get("key")).strip()
        txt = _safe_str(it.get("chunk_text"))
        if key:
            docs.append({"doc_id": key, "text": txt})
    return docs


def _expand_samples_to_pairwise_rows(
    *,
    query_row: QueryRow,
    samples: Sequence[Dict[str, Any]],
) -> Iterator[Dict[str, Any]]:
    """
    PairSample:
      { query_text, positive:{doc_id,text}, negatives:[{doc_id,text}...], source_chunk:str }

    展开为 pairwise:
      {query, positive, negative} 每个 negative 一行
    """
    qid = _safe_str(query_row.get("query_id"))
    qtext = _safe_str(query_row.get("query_text_norm"))

    for s in samples:
        pos = s.get("positive") or {}
        negs = s.get("negatives") or []
        source_chunk = _safe_str(s.get("source_chunk"))

        pos_doc_id = _safe_str(pos.get("doc_id"))
        pos_text = _safe_str(pos.get("text"))

        for neg in negs:
            neg_doc_id = _safe_str((neg or {}).get("doc_id"))
            neg_text = _safe_str((neg or {}).get("text"))

            yield {
                "query_id": qid,
                "query_text": qtext,
                "positive": {"doc_id": pos_doc_id, "text": pos_text},
                "negative": {"doc_id": neg_doc_id, "text": neg_text},
                "source_chunk": source_chunk,
                "meta": {
                    "llm_model": _safe_str(query_row.get("llm_model")),
                    "prompt_style": _safe_str(query_row.get("prompt_style")),
                    "domain": _safe_str(query_row.get("domain")),
                },
            }


def _pack_samples_as_candidates_rows(
    *,
    query_row: QueryRow,
    samples: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    将 PairSample（未展开）保存为 candidates 行：
      {
        query_id, query_text,
        positive:{...},
        negatives:[...],
        source_chunk,
        meta:{...}
      }
    """
    qid = _safe_str(query_row.get("query_id"))
    qtext = _safe_str(query_row.get("query_text_norm"))

    out: List[Dict[str, Any]] = []
    for s in samples:
        out.append(
            {
                "query_id": qid,
                "query_text": qtext,
                "positive": s.get("positive") or {"doc_id": "", "text": ""},
                "negatives": s.get("negatives") or [],
                "source_chunk": _safe_str(s.get("source_chunk")),
                "meta": {
                    "llm_model": _safe_str(query_row.get("llm_model")),
                    "prompt_style": _safe_str(query_row.get("prompt_style")),
                    "domain": _safe_str(query_row.get("domain")),
                },
            }
        )
    return out


# -----------------------------
# Main callable API
# -----------------------------
def run_pairing(
    config_path: str,
    *,
    llm: Any,
    embedder: Any,
    best_effort: bool = True,
) -> RunPairingResult:
    """
    可被其他程序 import 调用的入口。
    - llm: 必须有 generate(prompt)->str
    - embedder: 你的 DualInstructEmbedder（encode_passages / encode_queries）
    - best_effort=True: 单条 query 出错不影响整体；错误写入 errors.jsonl
    """
    t0 = _now()
    settings: Dict[str, Any] = load_settings(config_path)

    # ✅ 一次性创建检索器
    t_retr = _now()
    retriever = HybridRetriever.from_settings(settings)
    t_retr2 = _now()

    stores = build_store_registry(settings)

    # -------- outputs --------
    out_store_name = _must_get(settings, ["outputs", "store"])
    out_base = _must_get(settings, ["outputs", "base"])
    out_files = _must_get(settings, ["outputs", "files"])

    out_store = stores[out_store_name]

    queries_in_rel = _must_get(out_files, ["queries_in_domain"])  # queries/in_domain.jsonl

    # ✅ 新增：保存未展开 PairSample 的 candidates
    candidates_rel = _must_get(out_files, ["candidates"])         # e.g. pairs/pairs.candidates.jsonl

    pairs_rel = _must_get(out_files, ["pairs"])                   # pairs/pairs.pairwise_train.jsonl
    stats_rel = _must_get(out_files, ["stats"])                   # run_stats.json
    errors_rel = _must_get(out_files, ["errors"])                 # errors.jsonl

    queries_in_path = _join_posix(out_base, queries_in_rel)
    candidates_path = _join_posix(out_base, candidates_rel)
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

    # -------- read queries --------
    t_readq = _now()
    queries = _read_queries_in_domain(out_store, queries_in_path, on_error=on_error)
    t_readq2 = _now()

    # -------- build source chunk map (only needed ids) --------
    needed_ids = _collect_needed_source_chunk_ids(queries)
    t_src = _now()
    source_map = _build_source_chunk_map(
        chunks_store=chunks_store,
        chunks_jsonl_path=chunks_path,
        needed_ids=needed_ids,
        on_error=on_error,
    )
    t_src2 = _now()

    # 覆盖清空 outputs（如果 write_jsonl 允许空列表）
    try:
        write_jsonl(out_store, candidates_path, [])
    except Exception:
        pass
    try:
        write_jsonl(out_store, pairs_path, [])
    except Exception:
        pass

    num_pairs_written = 0
    num_candidates_written = 0
    num_processed = 0
    num_errors = 0

    t_loop = _now()
    for q in queries:
        try:
            qtext = _safe_str(q.get("query_text_norm")).strip()
            if not qtext:
                continue

            src_ids = q.get("source_chunk_ids") or []
            if not src_ids:
                raise ValueError("query row missing source_chunk_ids")

            src_id = _safe_str(src_ids[0]).strip()
            if not src_id:
                raise ValueError("source_chunk_ids[0] is empty")

            if src_id not in source_map:
                raise KeyError(f"source_chunk_id not found in chunks.jsonl: {src_id}")

            source_doc: Document = source_map[src_id]

            # ✅ 检索
            retrieved: List[Dict[str, Any]] = retriever.retrieve(qtext)
            candidate_docs: List[Document] = _convert_retrieval_items_to_docs(retrieved)

            # ✅ pairing（返回 PairSample 列表）
            samples, _stats = build_pairs_for_query(
                query_text=qtext,
                source_doc=source_doc,
                candidate_docs=candidate_docs,
                llm=llm,
                embedder=embedder,
            )

            # ✅ 新增：保存未展开 PairSample 到 candidates_path
            cand_rows = _pack_samples_as_candidates_rows(query_row=q, samples=samples)
            if cand_rows:
                append_jsonl(out_store, candidates_path, cand_rows)
                num_candidates_written += len(cand_rows)

            # ✅ pairwise 写入（每个 negative 一行）
            batch = list(_expand_samples_to_pairwise_rows(query_row=q, samples=samples))
            if batch:
                append_jsonl(out_store, pairs_path, batch)
                num_pairs_written += len(batch)

            num_processed += 1

        except Exception as e:
            num_errors += 1
            payload = {
                "stage": "run_pairing",
                "error": f"{type(e).__name__}: {e}",
                "query_id": _safe_str(q.get("query_id")),
                "query_text_norm": _safe_str(q.get("query_text_norm"))[:300],
            }
            if best_effort:
                _log_err(payload)
            else:
                raise

    t_loop2 = _now()

    timing = {
        "build_retriever_sec": float(t_retr2 - t_retr),
        "read_queries_sec": float(t_readq2 - t_readq),
        "build_source_map_sec": float(t_src2 - t_src),
        "process_queries_sec": float(t_loop2 - t_loop),
        "total_sec": float(_now() - t0),
    }

    stats_obj: Dict[str, Any] = {
        "stage": "run_pairing",
        "num_queries_total": len(queries),
        "num_queries_processed": num_processed,
        "num_candidates_written": num_candidates_written,  # ✅ 新增
        "num_pairs_written": num_pairs_written,
        "num_errors": num_errors,
        "timing": timing,
        "inputs": {
            "queries_in_path": queries_in_path,
            "chunks_path": chunks_path,
        },
        "outputs": {
            "candidates_path": candidates_path,  # ✅ 新增
            "pairs_path": pairs_path,
            "errors_path": errors_path,
            "stats_path": stats_path,
        },
        "meta": settings.get("_meta", {}),
    }

    out_store.write_text(stats_path, json.dumps(stats_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    return RunPairingResult(
        num_queries_total=len(queries),
        num_queries_processed=num_processed,
        num_pairs_written=num_pairs_written,
        num_errors=num_errors,
        timing=timing,
    )


# -----------------------------
# CLI
# -----------------------------
def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Run pairing: query -> retrieval -> LLM pairing -> pairwise jsonl.")
    p.add_argument("--config", required=True, help="Path to pipeline yaml, e.g. configs/pipeline.yaml")
    p.add_argument("--fail_fast", action="store_true", help="Fail fast instead of best-effort.")
    args = p.parse_args(list(argv) if argv is not None else None)

    settings = load_settings(args.config)

    # ✅ 你的 config 结构是 models.llm / embedding
    models_cfg = settings.get("models", {}) if isinstance(settings.get("models", {}), dict) else {}
    llm_cfg = models_cfg.get("llm", {}) if isinstance(models_cfg.get("llm", {}), dict) else {}

    # --- LLM: qr_pipeline/llm/hf_transformers_llm.py ---
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

    # --- Embedder: qr_pipeline/processing/embedder.py ---
    from qr_pipeline.processing.embedder import DualInstructEmbedder

    emb_cfg = settings.get("embedding", {}) if isinstance(settings.get("embedding", {}), dict) else {}
    instr_cfg = emb_cfg.get("instructions", {}) if isinstance(emb_cfg.get("instructions", {}), dict) else {}

    embedder = DualInstructEmbedder(
        model_name=_safe_str(emb_cfg.get("model_name", "")),
        passage_instruction=_safe_str(instr_cfg.get("passage", "")),
        query_instruction=_safe_str(instr_cfg.get("query", "")),
        batch_size=int(emb_cfg.get("batch_size", 64) or 64),
        normalize_embeddings=bool(emb_cfg.get("normalize_embeddings", True)),
        device=emb_cfg.get("device"),  # "cpu" / "cuda" / None(auto)
    )

    res = run_pairing(
        args.config,
        llm=llm,
        embedder=embedder,
        best_effort=(not args.fail_fast),
    )

    print(
        f"[run_pairing] queries={res.num_queries_processed}/{res.num_queries_total}, "
        f"pairs={res.num_pairs_written}, errors={res.num_errors}, total_sec={res.timing.get('total_sec'):.2f}"
    )


if __name__ == "__main__":
    main()
