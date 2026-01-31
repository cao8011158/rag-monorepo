# src/qr_pipeline/query_generation.py
# chunks.jsonl -> Gemini generate queries (structured) -> postprocess -> normalize -> hash(query_id)
# -> exact dedup by query_id (resume supported) -> Gemini domain classify in BATCH (structured in_ids/out_ids)
# -> write queries/in_domain.jsonl and queries/out_domain.jsonl + stats checkpoints

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, ValidationError
from google import genai

from qr_pipeline.io.jsonl import append_jsonl, read_jsonl
from qr_pipeline.stores.registry import build_store_registry


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


def _get_required(cfg: Dict[str, Any], path: str) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing required config key: {path}")
        cur = cur[part]
    return cur


# -----------------------------
# Structured output schemas (Gemini)
# -----------------------------

class QueryOutput(BaseModel):
    queries: List[str] = Field(description="A list of search queries.")


class ClassifyOutput(BaseModel):
    in_ids: List[int] = Field(description="1-based indices of IN queries, e.g. [1,3,4].")
    out_ids: List[int] = Field(description="1-based indices of OUT queries, e.g. [2,5].")


# -----------------------------
# Prompt builders
# -----------------------------

def _build_query_prompt(chunk_text: str, p: Dict[str, Any]) -> str:
    """
    Matches your config format:
      style: "information-seeking"
      num_queries_per_chunk: 3
      max_chunk_chars: 2100
      diversify: true
      diversity_hints: | ...
      avoid_near_duplicates: true
    """
    style = str(p.get("style", "information-seeking"))
    n = _safe_int(p.get("num_queries_per_chunk", 1), 1)
    diversify = bool(p.get("diversify", True))
    hints = (p.get("diversity_hints") or "").strip()
    passage = (chunk_text or "")

    lines: List[str] = []
    lines.append(f"Task: generate {n} {style} queries that a user might ask to find the information in the passage.")
    if diversify and hints:
        lines.append("Diversify query types using these hints:")
        lines.append(hints)
    lines.append("")
    lines.append("Output requirements:")
    lines.append("- Return JSON only (no markdown).")
    lines.append("- Field: queries (array of strings).")
    lines.append("- No extra fields.")
    lines.append("")
    lines.append("Passage:")
    lines.append(passage)
    return "\n".join(lines)


def _build_domain_prompt_for_batch(queries: List[str]) -> str:
    """
    Keep your preferred long prompt, then append Input lines:
      1: ...
      2: ...
    Output is structured by JSON schema (in_ids/out_ids).
    """
    base = """Task :
You will receive N user queries, classify each query into exactly one category:
IN: primarily about Pittsburgh or Carnegie Mellon University (CMU), and explicitly mentions "Pittsburgh" or "Carnegie Mellon University" or "CMU".
OUT: everything else.
Domain definition (IN):
IN queries explicitly focus on Pittsburgh or CMU, including their history, geography, culture, population, economy, campus life, traditions, trivia, and current events.
Decision rules:
Choose IN only if the query is primarily about Pittsburgh or CMU AND explicitly mentions one of: "Pittsburgh", "Carnegie Mellon University", "CMU".
Otherwise choose OUT (even if the topic could be related).
If uncertain, choose OUT.

Example :
Input:
1:what year was carnegie mellon university founded
2:best universities for computer science in the us
3:major sports teams in pittsburgh
4:how did the whiskey rebellion impact pittsburgh's development
5:what was the french and indian war

Output:
{"in_ids":[1,3,4],"out_ids":[2,5]}

Now classify the following queries:
"""

    lines: List[str] = [base.rstrip(), "", "Input:"]
    for i, q in enumerate(queries, start=1):
        qq = (q or "").strip().replace("\n", " ")
        lines.append(f"{i}: {qq}")
    return "\n".join(lines)


# -----------------------------
# Post-process / validation
# -----------------------------

def _normalize_queries_list(queries: List[Any]) -> List[str]:
    cleaned: List[str] = []
    for q in queries:
        if isinstance(q, str):
            s = q.strip()
            if s:
                cleaned.append(s)
    # exact-string dedup preserving order
    return list(dict.fromkeys(cleaned))


def _dedup_ints_keep_order(xs: List[int]) -> List[int]:
    seen: Set[int] = set()
    out: List[int] = []
    for x in xs:
        if isinstance(x, int) and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _validate_ids(
    *,
    in_ids: List[int],
    out_ids: List[int],
    n: int,
    fill_missing_as_out: bool = True,
) -> ClassifyOutput:
    """
    Enforce:
    - 1 <= id <= n
    - dedup
    - no overlaps (IN wins)
    - optionally fill missing as OUT
    """
    in_ids = _dedup_ints_keep_order([i for i in in_ids if 1 <= i <= n])
    out_ids = _dedup_ints_keep_order([i for i in out_ids if 1 <= i <= n])

    in_set = set(in_ids)
    out_ids = [i for i in out_ids if i not in in_set]
    out_set = set(out_ids)

    if fill_missing_as_out:
        for i in range(1, n + 1):
            if i not in in_set and i not in out_set:
                out_ids.append(i)

    # stable order
    in_ids = [i for i in range(1, n + 1) if i in set(in_ids)]
    out_ids = [i for i in range(1, n + 1) if i in set(out_ids)]
    return ClassifyOutput(in_ids=in_ids, out_ids=out_ids)


# -----------------------------
# Gemini calls (structured output)
# -----------------------------

def _gemini_generate_queries_once(
    *,
    model_name: str,
    prompt: str,
) -> List[str]:
    client = genai.Client()
    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={
            "system_instruction": "You are a helpful assistant that generates search queries.",
            "response_mime_type": "application/json",
            "response_json_schema": QueryOutput.model_json_schema(),
        },
    )
    parsed = QueryOutput.model_validate_json(resp.text)
    return _normalize_queries_list(parsed.queries)


def _gemini_classify_once(
    *,
    model_name: str,
    prompt: str,
) -> ClassifyOutput:
    client = genai.Client()
    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config={
            "system_instruction": "You are a strict classifier.",
            "response_mime_type": "application/json",
            "response_json_schema": ClassifyOutput.model_json_schema(),
        },
    )
    parsed = ClassifyOutput.model_validate_json(resp.text)
    return parsed


def _with_retry(
    *,
    fn_name: str,
    fn,
    max_retries: int,
    backoff_sec: float,
) -> Any:
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except (ValidationError, ValueError) as e:
            # schema / format problems: treat as non-retry (you can change this policy if desired)
            raise RuntimeError(f"{fn_name}: invalid structured output format: {e}") from e
        except Exception as e:
            last_err = e
            print(f"[WARN] {fn_name} failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(backoff_sec * attempt)
                continue
            break
    raise RuntimeError(f"{fn_name} failed after {max_retries} retries: {last_err}")


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
# Main
# -----------------------------

def run_query_generation(s: Dict[str, Any]) -> Dict[str, Any]:
    """
    chunks.jsonl -> Gemini gen -> postprocess -> normalize -> hash -> dedup -> Gemini classify (batch)
    -> append in/out domain jsonl -> stats checkpoint each batch
    Dedup via seen_ids only (重复忽略 source_chunk_ids)
    Target stop: IN unique == target_num_queries
    """
    stores = build_store_registry(s)

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
    max_chars = _safe_int(post_cfg.get("max_query_chars", 200), 200)

    lower = bool(norm_cfg.get("lower", True))
    strip = bool(norm_cfg.get("strip", True))
    collapse_whitespace = bool(norm_cfg.get("collapse_whitespace", True))

    prompt_style = str(prompt_cfg.get("style", "information-seeking"))
    avoid_near_duplicates = bool(prompt_cfg.get("avoid_near_duplicates", True))
    max_chunk_chars = _safe_int(prompt_cfg.get("max_chunk_chars", 2100), 2100)

    # ---- Gemini model ----
    gemini_model_name = str(_get_required(s, "models.gemini_api.model_name"))

    # ---- retry settings (optional config; defaults here) ----
    max_retries = _safe_int(qg.get("gemini_max_retries", 3), 3)
    backoff_sec = float(qg.get("gemini_backoff_sec", 1.5) or 1.5)

    # ---- counters ----
    read_errors = 0
    total_gemini_calls_gen = 0
    total_gemini_calls_domain = 0
    total_raw_queries = 0
    total_after_post = 0
    total_in_unique = 0
    total_out_unique = 0
    total_domain_batches = 0
    total_domain_items_classified = 0

    # ---- error writer ----
    def _log_error(stage: str, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        payload["ts_ms"] = _now_ms()
        payload["stage"] = stage
        append_jsonl(out_store, errors_path, [payload])

    def _on_read_error(payload: Dict[str, Any]) -> None:
        nonlocal read_errors
        read_errors += 1
        _log_error("read_chunks", payload)

    # ---- load chunks ----
    rows = list(read_jsonl(in_store, chunks_path, on_error=_on_read_error))
    sampled = _sample_chunks(rows, sampling_cfg)

    # ---- seen_ids (resume) ----
    seen_ids: Set[str] = set()
    for path in (in_domain_path, out_domain_path):
        try:
            for r in read_jsonl(out_store, path):
                seen_ids.add(str(r["query_id"]))
        except Exception:
            pass

    total_in_unique = sum(1 for _ in read_jsonl(out_store, in_domain_path)) if out_store.exists(in_domain_path) else 0
    total_out_unique = sum(1 for _ in read_jsonl(out_store, out_domain_path)) if out_store.exists(out_domain_path) else 0

    print(f"[INIT] loaded seen_ids={len(seen_ids)} in={total_in_unique} out={total_out_unique}")

    # ---- stats checkpoint ----
    def _write_stats_checkpoint(extra: Optional[Dict[str, Any]] = None) -> None:
        snap = {
            "ts_ms": _now_ms(),
            "counters": {
                "read_errors": read_errors,
                "gemini_calls_generate_queries": total_gemini_calls_gen,
                "gemini_calls_domain_classify_batches": total_gemini_calls_domain,
                "domain_classify_batches": total_domain_batches,
                "domain_classify_items": total_domain_items_classified,
                "raw_queries_parsed": total_raw_queries,
                "kept_after_postprocess": total_after_post,
            },
            "outputs": {
                "num_queries_in_domain_unique_written": total_in_unique,
                "num_queries_out_domain_unique_written": total_out_unique,
            },
            "meta": {
                "gemini_model": gemini_model_name,
                "prompt_style": prompt_style,
                "target_num_in_domain": target_in_domain,
            },
        }
        if extra:
            snap["extra"] = extra
        out_store.write_text(stats_path, json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")

    def _print_state() -> None:
        print(
            f"[STATE] in={total_in_unique} out={total_out_unique} "
            f"batch={total_domain_batches} "
            f"gem_gen={total_gemini_calls_gen} "
            f"gem_domain={total_gemini_calls_domain} "
            f"seen={len(seen_ids)}"
        )

    # ---- batch buffer ----
    # store both q_text (for classification) and q_norm (for storage/id)
    batch_buf: List[Dict[str, Any]] = []

    def _flush_domain_batch() -> None:
        nonlocal total_gemini_calls_domain, total_domain_batches
        nonlocal total_domain_items_classified, total_in_unique, total_out_unique

        if not batch_buf:
            return

        # build classification input (use original cleaned query text)
        qs_for_classify = [str(x["q_text"]) for x in batch_buf]
        prompt = _build_domain_prompt_for_batch(qs_for_classify)

        def _call() -> ClassifyOutput:
            return _gemini_classify_once(model_name=gemini_model_name, prompt=prompt)

        try:
            parsed = _with_retry(
                fn_name="gemini_classify",
                fn=_call,
                max_retries=max_retries,
                backoff_sec=backoff_sec,
            )
            total_gemini_calls_domain += 1
            total_domain_batches += 1
        except Exception as e:
            # if classification totally fails, log and mark all as OUT (conservative)
            _log_error(
                "domain_classify",
                {
                    "error": str(e),
                    "batch_size": len(batch_buf),
                },
            )
            parsed = ClassifyOutput(in_ids=[], out_ids=list(range(1, len(batch_buf) + 1)))
            total_gemini_calls_domain += 1
            total_domain_batches += 1

        final = _validate_ids(
            in_ids=parsed.in_ids,
            out_ids=parsed.out_ids,
            n=len(batch_buf),
            fill_missing_as_out=True,
        )
        in_set = set(final.in_ids)  # 1-based indices

        new_in_rows: List[Dict[str, Any]] = []
        new_out_rows: List[Dict[str, Any]] = []

        for idx, x in enumerate(batch_buf, start=1):
            real_id = str(x["real_id"])
            q_norm = str(x["q_norm"])
            chunk_id = str(x["chunk_id"])

            label_in = idx in in_set
            total_domain_items_classified += 1

            row_obj = {
                "query_id": real_id,
                "query_text_norm": q_norm,
                "source_chunk_ids": [chunk_id],
                "llm_model": gemini_model_name,
                "prompt_style": prompt_style,
                "domain": "in" if label_in else "out",
            }

            if label_in:
                new_in_rows.append(row_obj)
                total_in_unique += 1
            else:
                new_out_rows.append(row_obj)
                total_out_unique += 1

        if new_in_rows:
            append_jsonl(out_store, in_domain_path, new_in_rows)
        if new_out_rows:
            append_jsonl(out_store, out_domain_path, new_out_rows)

        _print_state()
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
        gen_prompt = _build_query_prompt(passage, prompt_cfg)

        def _call_gen() -> List[str]:
            return _gemini_generate_queries_once(model_name=gemini_model_name, prompt=gen_prompt)

        try:
            queries = _with_retry(
                fn_name="gemini_generate_queries",
                fn=_call_gen,
                max_retries=max_retries,
                backoff_sec=backoff_sec,
            )
            total_gemini_calls_gen += 1
        except Exception as e:
            _log_error(
                "query_generate",
                {
                    "error": str(e),
                    "chunk_id": str(chunk_id),
                },
            )
            total_gemini_calls_gen += 1
            continue

        total_raw_queries += len(queries)

        # per-chunk near-duplicate control (by real_id)
        seen_in_chunk: Set[str] = set()

        for q in queries:
            if total_in_unique >= target_in_domain:
                break

            q_text = (q or "").strip()
            if len(q_text) < min_chars:
                continue
            if len(q_text) > max_chars:
                q_text = q_text[:max_chars].strip()

            q_norm = _normalize_text(q_text, lower=lower, strip=strip, collapse_whitespace=collapse_whitespace)
            real_id = _make_id(q_norm)

            if avoid_near_duplicates and real_id in seen_in_chunk:
                continue
            seen_in_chunk.add(real_id)

            # global dedup (resume supported)
            if real_id in seen_ids:
                continue
            seen_ids.add(real_id)
            total_after_post += 1

            batch_buf.append(
                {
                    "real_id": real_id,
                    "q_norm": q_norm,
                    "q_text": q_text,      # for classification
                    "chunk_id": chunk_id,
                }
            )

            if len(batch_buf) >= batch_size:
                _flush_domain_batch()

    _flush_domain_batch()
    _write_stats_checkpoint(extra={"status": "done"})
    return {"status": "done"}


def main(config_path: str) -> None:
    from qr_pipeline.settings import load_settings

    s = load_settings(config_path)
    stats = run_query_generation(s)

    out_cfg = s["outputs"]
    stores = build_store_registry(s)
    out_store = stores[str(out_cfg["store"])]
    out_base = str(out_cfg["base"])
    stats_path = _posix_join(out_base, str(out_cfg["files"]["stats"]))

    out_store.write_text(stats_path, json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
