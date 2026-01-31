# src/qr_pipeline/pipeline/query_generation.py
# chunks.jsonl -> Gemini generate queries -> normalize -> hash (real query_id)
# -> exact dedup by real query_id
# -> domain classify in BATCH (in_ids/out_ids, 1-based)
# -> write queries/in_domain.jsonl and queries/out_domain.jsonl

from __future__ import annotations

import hashlib
import random
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from qr_pipeline.io.jsonl import append_jsonl, read_jsonl
from qr_pipeline.stores.registry import build_store_registry

# NEW: Gemini wrappers
from qr_pipeline.llm.query_generation_gemini import run_query_generation
from qr_pipeline.llm.query_classification_gemini import run_query_classification


# -----------------------------
# Helpers
# -----------------------------

def _posix_join(*parts: str) -> str:
    return "/".join([p.strip("/").replace("\\", "/") for p in parts if p is not None and str(p) != ""])


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _make_id(norm_text: str, n: int = 24) -> str:
    return _sha256_hex(norm_text)[:n]


def _normalize_text(text: str, *, lower: bool, strip: bool, collapse_whitespace: bool) -> str:
    x = text or ""
    if strip:
        x = x.strip()
    if collapse_whitespace:
        x = re.sub(r"\s+", " ", x)
    if lower:
        x = x.lower()
    return x


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _now_ms() -> int:
    return int(time.time() * 1000)


def _get_llm_model_name(cfg: Dict[str, Any]) -> str:
    """
    Best-effort extract model name for logging/stats.
    Common layout in your earlier configs: cfg["models"]["gemini_api"]["model_name"].
    """
    try:
        return str(cfg["models"]["gemini_api"]["model_name"])
    except Exception:
        return "gemini"


# -----------------------------
# Sampling
# -----------------------------

def _sample_chunks(rows: List[Dict[str, Any]], sampling_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    seed = _safe_int(sampling_cfg.get("seed", 42), 42)
    strategy = str(sampling_cfg.get("strategy", "uniform_random"))
    cap = _safe_int(sampling_cfg.get("max_chunks_considered", len(rows)), len(rows))

    rng = random.Random(seed)

    if strategy == "uniform_random":
        pool = rows[:]
        rng.shuffle(pool)
        return pool[: min(cap, len(pool))]

    raise NotImplementedError(f"sampling.strategy not implemented: {strategy}")


# -----------------------------
# Domain classification helpers (NEW)
# -----------------------------

def _dedup_ints_keep_order(xs: List[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for x in xs:
        if isinstance(x, int) and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _coerce_in_out_ids(
    *,
    in_ids: Any,
    out_ids: Any,
    n: int,
) -> Tuple[List[int], List[int]]:
    """
    Policy (match your old behavior, conservative):
      - Keep only ints in [1..n]
      - Dedup keep order
      - If a label appears in both, OUT wins (i.e., remove from IN)
      - Missing labels => OUT handled by downstream (anything not IN is OUT)
    """
    in_list = in_ids if isinstance(in_ids, list) else []
    out_list = out_ids if isinstance(out_ids, list) else []

    in_clean = []
    for x in in_list:
        try:
            xi = int(x)
        except Exception:
            continue
        if 1 <= xi <= n:
            in_clean.append(xi)

    out_clean = []
    for x in out_list:
        try:
            xi = int(x)
        except Exception:
            continue
        if 1 <= xi <= n:
            out_clean.append(xi)

    in_clean = _dedup_ints_keep_order(in_clean)
    out_clean = _dedup_ints_keep_order(out_clean)

    out_set = set(out_clean)
    # Conflict -> OUT wins
    in_clean = [i for i in in_clean if i not in out_set]

    return in_clean, out_clean


def _classify_batch_conservative(
    queries_norm: List[str],
    cfg: Dict[str, Any],
) -> Tuple[List[int], List[int], Optional[str]]:
    """
    Calls run_query_classification(queries, cfg) and enforces conservative policies:
      - exception / malformed => all OUT
      - missing => OUT
      - conflict => OUT wins
    Returns: (in_ids, out_ids, err_msg_or_none)
    """
    n = len(queries_norm)
    if n == 0:
        return [], [], None

    try:
        res = run_query_classification(queries_norm, cfg)
        in_ids_raw = res.get("in_ids", [])
        out_ids_raw = res.get("out_ids", [])
        in_ids, out_ids = _coerce_in_out_ids(in_ids=in_ids_raw, out_ids=out_ids_raw, n=n)

        # Missing labels => OUT (implicit). For convenience we can fill out_ids,
        # but downstream can treat "not IN" as OUT anyway.
        # We'll keep explicit out_ids as-is.
        return in_ids, out_ids, None

    except Exception as e:
        # 解析失败 -> 整批 OUT（保守）
        return [], list(range(1, n + 1)), f"{type(e).__name__}: {e}"


# -----------------------------
# Main
# -----------------------------

def run_query_generation_pipeline(s: Dict[str, Any]) -> Dict[str, Any]:
    """
    chunks.jsonl -> generate queries -> normalize -> hash
    -> domain classify (batch)
    -> append in/out domain jsonl
    -> stats checkpoint each batch
    Dedup via seen_ids only (方案A：重复忽略 source_chunk_ids)
    """

    import json

    stores = build_store_registry(s)
    llm_model_name = _get_llm_model_name(s)

    # ---- inputs ----
    in_cfg = s["inputs"]["ce_artifacts"]["chunks"]
    in_store = stores[str(in_cfg["store"])]
    chunks_path = _posix_join(str(in_cfg["base"]), str(in_cfg["chunks_file"]))

    # ---- outputs ----
    out_cfg = s["outputs"]
    out_store = stores[str(out_cfg["store"])]
    out_base = str(out_cfg["base"])
    files = out_cfg["files"]

    errors_path = _posix_join(out_base, str(files["errors"]))
    in_domain_path = _posix_join(out_base, str(files["queries_in_domain"]))
    out_domain_path = _posix_join(out_base, str(files["queries_out_domain"]))
    stats_path = _posix_join(out_base, str(files["stats"]))

    # ---- config ----
    qg = s["query_generation"]
    target_in_domain = _safe_int(qg.get("target_num_queries", 2000), 2000)
    batch_size = _safe_int(qg.get("domain_batch_size", 15), 15)

    sampling_cfg = qg.get("sampling", {}) or {}
    prompt_cfg = qg.get("prompt", {}) or {}
    post_cfg = qg.get("postprocess", {}) or {}
    norm_cfg = qg.get("normalize", {}) or {}

    min_chars = _safe_int(post_cfg.get("min_query_chars", 8), 8)
    max_chars = _safe_int(post_cfg.get("max_query_chars", 160), 160)

    lower = bool(norm_cfg.get("lower", True))
    strip = bool(norm_cfg.get("strip", True))
    collapse_whitespace = bool(norm_cfg.get("collapse_whitespace", True))

    prompt_style = str(prompt_cfg.get("style", "information-seeking"))
    avoid_near_duplicates = bool(prompt_cfg.get("avoid_near_duplicates", True))

    max_chunk_chars = _safe_int(prompt_cfg.get("max_chunk_chars", 1800), 1800)

    # ---- counters ----
    read_errors = 0
    total_llm_calls_gen = 0
    total_llm_calls_domain = 0
    total_raw_queries = 0
    total_after_post = 0
    total_in_unique = 0
    total_out_unique = 0
    total_domain_batches = 0
    total_domain_items_classified = 0
    domain_fail_batches = 0

    # ---- load chunks ----
    def _on_read_error(payload: Dict[str, Any]) -> None:
        nonlocal read_errors
        read_errors += 1
        payload["ts_ms"] = _now_ms()
        append_jsonl(out_store, errors_path, [payload])

    rows = list(read_jsonl(in_store, chunks_path, on_error=_on_read_error))
    sampled = _sample_chunks(rows, sampling_cfg)

    # ---- seen_ids (resume) ----
    seen_ids: set[str] = set()

    for path in (in_domain_path, out_domain_path):
        try:
            for r in read_jsonl(out_store, path):
                seen_ids.add(str(r["query_id"]))
        except Exception:
            pass

    total_in_unique = sum(1 for _ in read_jsonl(out_store, in_domain_path)) if out_store.exists(in_domain_path) else 0
    total_out_unique = sum(1 for _ in read_jsonl(out_store, out_domain_path)) if out_store.exists(out_domain_path) else 0

    print(f"[INIT] loaded seen_ids={len(seen_ids)} in={total_in_unique} out={total_out_unique}")

    batch_buf: List[Dict[str, Any]] = []

    def _print_state() -> None:
        print(
            f"[STATE] in={total_in_unique} out={total_out_unique} "
            f"batch={total_domain_batches} "
            f"llm_gen={total_llm_calls_gen} "
            f"llm_domain={total_llm_calls_domain} "
            f"seen={len(seen_ids)} "
            f"domain_fail_batches={domain_fail_batches}"
        )

    def _write_stats_checkpoint() -> None:
        snap = {
            "ts_ms": _now_ms(),
            "counters": {
                "read_errors": read_errors,
                "llm_calls_generate_queries": total_llm_calls_gen,
                "llm_calls_domain_classify_batches": total_llm_calls_domain,
                "domain_classify_batches": total_domain_batches,
                "domain_classify_items": total_domain_items_classified,
                "domain_fail_batches": domain_fail_batches,
                "raw_queries_parsed": total_raw_queries,
                "kept_after_postprocess": total_after_post,
            },
            "outputs": {
                "num_queries_in_domain_unique_written": total_in_unique,
                "num_queries_out_domain_unique_written": total_out_unique,
            },
            "meta": {
                "llm_model": llm_model_name,
                "prompt_style": prompt_style,
                "target_num_in_domain": target_in_domain,
            },
        }
        out_store.write_text(stats_path, json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")

    def _flush_domain_batch() -> None:
        nonlocal total_llm_calls_domain, total_domain_batches
        nonlocal total_domain_items_classified, total_in_unique, total_out_unique, domain_fail_batches

        if not batch_buf:
            return

        queries_norm = [x["q_norm"] for x in batch_buf]

        # NEW: call your classifier
        in_ids, out_ids, err = _classify_batch_conservative(queries_norm, s)
        total_llm_calls_domain += 1
        total_domain_batches += 1

        if err is not None:
            domain_fail_batches += 1
            append_jsonl(out_store, errors_path, [{
                "ts_ms": _now_ms(),
                "stage": "domain_classification",
                "error": err,
                "batch_size": len(batch_buf),
            }])

        in_set = set(in_ids)
        out_set = set(out_ids)  # may be full [1..n] in failure case

        new_in_rows = []
        new_out_rows = []

        for idx, x in enumerate(batch_buf, start=1):  # 1-based for ids
            real_id = x["real_id"]
            q_norm = x["q_norm"]
            chunk_id = x["chunk_id"]

            # Policy:
            # - conflict -> OUT wins (we enforce by checking OUT first)
            # - missing -> OUT (anything not explicitly IN is OUT)
            label = "OUT"
            if idx in out_set:
                label = "OUT"
            elif idx in in_set:
                label = "IN"
            else:
                label = "OUT"

            total_domain_items_classified += 1

            row_obj = {
                "query_id": real_id,
                "query_text_norm": q_norm,
                "source_chunk_ids": [chunk_id],
                "llm_model": llm_model_name,
                "prompt_style": prompt_style,
                "domain": "in" if label == "IN" else "out",
            }

            if label == "IN":
                new_in_rows.append(row_obj)
                total_in_unique += 1
            else:
                new_out_rows.append(row_obj)
                total_out_unique += 1

        _print_state()

        if new_in_rows:
            append_jsonl(out_store, in_domain_path, new_in_rows)
        if new_out_rows:
            append_jsonl(out_store, out_domain_path, new_out_rows)

        _write_stats_checkpoint()
        batch_buf.clear()

    # ---------------- Loop ----------------
    for row in sampled:
        if total_in_unique >= target_in_domain:
            break

        chunk_id = row.get("chunk_id")
        chunk_text = row.get("chunk_text", "")
        if not chunk_id or not chunk_text:
            continue

        passage = (chunk_text or "")[:max_chunk_chars]

        # NEW: call your generator (returns List[str])
        try:
            queries = run_query_generation(passage, s)
            total_llm_calls_gen += 1
        except Exception as e:
            total_llm_calls_gen += 1
            append_jsonl(out_store, errors_path, [{
                "ts_ms": _now_ms(),
                "stage": "query_generation",
                "error": f"{type(e).__name__}: {e}",
                "chunk_id": chunk_id,
            }])
            continue

        if not isinstance(queries, list):
            # defensive: treat as generation failure -> skip chunk
            append_jsonl(out_store, errors_path, [{
                "ts_ms": _now_ms(),
                "stage": "query_generation",
                "error": f"Unexpected return type: {type(queries).__name__}",
                "chunk_id": chunk_id,
            }])
            continue

        total_raw_queries += len(queries)

        seen_in_chunk: set[str] = set()

        for q in queries:
            if total_in_unique >= target_in_domain:
                break

            if not isinstance(q, str):
                continue

            q = q.strip()
            if len(q) < min_chars:
                continue
            if len(q) > max_chars:
                q = q[:max_chars].strip()

            q_norm = _normalize_text(q, lower=lower, strip=strip, collapse_whitespace=collapse_whitespace)
            real_id = _make_id(q_norm)

            # chunk-local de-dup
            if avoid_near_duplicates and real_id in seen_in_chunk:
                continue
            seen_in_chunk.add(real_id)

            # global de-dup (resume)
            if real_id in seen_ids:
                continue
            seen_ids.add(real_id)

            total_after_post += 1
            batch_buf.append({"real_id": real_id, "q_norm": q_norm, "chunk_id": chunk_id})

            if len(batch_buf) >= batch_size:
                _flush_domain_batch()

    _flush_domain_batch()
    _write_stats_checkpoint()

    return {"status": "done"}


def main(config_path: str) -> None:
    """
    保持你原来的入口风格：加载 settings -> 跑 pipeline。
    注意：不要在这里覆盖掉 stats checkpoint（原来 main() 结尾覆盖 stats 的问题，这里去掉了）。
    """
    from qr_pipeline.settings import load_settings

    s = load_settings(config_path)
    run_query_generation_pipeline(s)
