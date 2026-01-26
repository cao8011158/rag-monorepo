# src/qr_pipeline/query_generation.py
# chunks.jsonl -> LLM generate queries -> normalize -> hash (qid) -> exact dedup
# -> BUFFER -> batch domain classify (15 per call) -> write in/out domain jsonl
# Early stop ONLY based on IN-domain unique queries count.

from __future__ import annotations

import hashlib
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

from qr_pipeline.stores.registry import build_store_registry
from qr_pipeline.io.jsonl import read_jsonl, write_jsonl, append_jsonl


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
    x = text
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


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _now_ms() -> int:
    return int(time.time() * 1000)


# -----------------------------
# LLM wrapper (hf_transformers)
# -----------------------------
@dataclass
class HFTransformersLLM:
    model_name: str
    device: str = "cpu"  # "cpu" / "cuda"
    cache_dir: Optional[str] = None
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9

    def __post_init__(self) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Missing dependency for hf_transformers LLM. Please install transformers + torch."
            ) from e

        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        self._tok = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir, use_fast=True)

        device_map: Optional[str] = "auto" if str(self.device).lower() == "cuda" else None

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            device_map=device_map,
        )
        self._model.eval()

    def generate(self, prompt: str) -> str:
        import torch  # type: ignore

        inputs = self._tok(prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        do_sample = (self.temperature is not None) and (self.temperature > 0.0)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=int(self.max_new_tokens),
                do_sample=bool(do_sample),
                temperature=float(self.temperature) if do_sample else None,
                top_p=float(self.top_p) if do_sample else None,
                pad_token_id=self._tok.eos_token_id,
            )

        text = self._tok.decode(out[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt) :]
        return text.strip()


def _build_llm_from_settings(s: Dict[str, Any]) -> Tuple[Any, str]:
    llm_cfg = s["models"]["llm"]
    provider = str(llm_cfg.get("provider", "hf_transformers"))
    model_name = str(llm_cfg["model_name"])

    if provider == "hf_transformers":
        llm = HFTransformersLLM(
            model_name=model_name,
            device=str(llm_cfg.get("device", "cpu")),
            cache_dir=llm_cfg.get("cache_dir"),
            max_new_tokens=_safe_int(llm_cfg.get("max_new_tokens", 128), 128),
            temperature=_safe_float(llm_cfg.get("temperature", 0.7), 0.7),
            top_p=_safe_float(llm_cfg.get("top_p", 0.9), 0.9),
        )
        return llm, model_name

    raise NotImplementedError(f"LLM provider not supported yet: {provider}")


# -----------------------------
# Prompt building & parsing
# -----------------------------
def _build_query_prompt(chunk_text: str, p: Dict[str, Any]) -> str:
    lang = str(p.get("language", "en"))
    style = str(p.get("style", "information-seeking"))
    n = _safe_int(p.get("num_queries_per_chunk", 1), 1)

    diversify = bool(p.get("diversify", False))
    hints = p.get("diversity_hints", []) or []
    hints_str = ", ".join([str(x) for x in hints]) if hints else ""

    if lang.lower().startswith("en"):
        lines: List[str] = []
        lines.append("You are a helpful assistant that generates search queries.")
        lines.append(f"Task: generate {n} {style} queries that a user might ask to find the information in the passage.")
        if diversify and hints_str:
            lines.append(f"Diversify query types using these hints: {hints_str}.")
        lines.append("Output format requirements:")
        lines.append("- Output ONLY the queries.")
        lines.append("- One query per line.")
        lines.append("- No numbering, no bullets, no explanations.")
        lines.append("")
        lines.append("Passage:")
        lines.append(chunk_text)
        lines.append("")
        lines.append("Queries:")
        return "\n".join(lines)

    return _build_query_prompt(chunk_text, {**p, "language": "en"})


DEFAULT_DOMAIN_DESC = "question about Pittsburgh and CMU, including history, culture, trivia, and upcoming events"


def _build_domain_batch_prompt(items: List[Dict[str, Any]], domain_desc: str) -> str:
    """
    Batch classify, output IDs only for robustness.
    Input items: [{"qid":..., "query_text_norm":...}, ...]
    Output format:
      IN:
      Q1
      Q3
      OUT:
      Q2
    """
    lines: List[str] = []
    lines.append("You are a strict text classifier.")
    lines.append(f'Domain: "{domain_desc}".')
    lines.append("Task: classify each query as IN (within Domain) or OUT (outside Domain).")
    lines.append("")
    lines.append("CRITICAL rules:")
    lines.append("- Do NOT rewrite queries.")
    lines.append("- Output ONLY IDs (Q1, Q2, ...) under IN/OUT.")
    lines.append("- Every provided ID must appear exactly once in either IN or OUT.")
    lines.append("- Output MUST have exactly two sections in this order: IN: then OUT:.")
    lines.append("")
    lines.append("Queries:")
    for i, it in enumerate(items, start=1):
        q = str(it.get("query_text_norm", ""))
        lines.append(f"Q{i}: {q}")
    lines.append("")
    lines.append("Answer:")
    lines.append("IN:")
    lines.append("OUT:")
    return "\n".join(lines)


def _parse_queries(raw: str) -> List[str]:
    out: List[str] = []
    for line in raw.splitlines():
        x = line.strip()
        if not x:
            continue
        x = re.sub(r"^\s*[\-\*\u2022]\s+", "", x)
        x = re.sub(r"^\s*\d+[\.\)\:]\s+", "", x)
        x = x.strip()
        if x:
            out.append(x)
    return out


def _parse_in_out_ids(raw: str) -> Tuple[List[int], List[int]]:
    """
    Parse batch classifier output. Accepts:
      IN:
      Q1
      Q3
      OUT:
      Q2
    Returns 0-based indices.
    """
    text = raw.strip()
    if not text:
        return [], []

    in_ids: List[int] = []
    out_ids: List[int] = []
    mode: Optional[str] = None

    for line in text.splitlines():
        t = line.strip()
        if not t:
            continue
        if re.fullmatch(r"IN\s*:\s*", t, flags=re.IGNORECASE):
            mode = "IN"
            continue
        if re.fullmatch(r"OUT\s*:\s*", t, flags=re.IGNORECASE):
            mode = "OUT"
            continue

        m = re.search(r"\bQ(\d+)\b", t, flags=re.IGNORECASE)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if idx < 0:
            continue
        if mode == "IN":
            in_ids.append(idx)
        elif mode == "OUT":
            out_ids.append(idx)

    return in_ids, out_ids


def _unique_preserve_order(xs: List[int]) -> List[int]:
    seen: set[int] = set()
    out: List[int] = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


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
    chunks.jsonl -> generate queries -> normalize -> hash
    -> BUFFER -> batch LLM domain classify (15 per call)
    -> write in/out domain jsonl (paths from config)
    Early stop ONLY based on IN-domain unique queries count.
    """
    stores = build_store_registry(s)

    # ---- inputs: chunks ----
    in_cfg = s["inputs"]["ce_artifacts"]["chunks"]
    in_store = stores[str(in_cfg["store"])]
    chunks_path = _posix_join(str(in_cfg["base"]), str(in_cfg["chunks_file"]))

    # ---- outputs (ALL from config) ----
    out_cfg = s["outputs"]
    out_store = stores[str(out_cfg["store"])]
    out_base = str(out_cfg["base"])
    files = out_cfg["files"]

    errors_path = _posix_join(out_base, str(files["errors"]))
    in_domain_path = _posix_join(out_base, str(files["queries_in_domain"]))
    out_domain_path = _posix_join(out_base, str(files["queries_out_domain"]))

    # ---- config ----
    qg = s["query_generation"]
    target_in_domain = _safe_int(qg.get("target_num_queries", 2000), 2000)

    sampling_cfg = qg.get("sampling", {}) or {}
    prompt_cfg = qg.get("prompt", {}) or {}
    post_cfg = qg.get("postprocess", {}) or {}
    norm_cfg = qg.get("normalize", {}) or {}

    # domain classify batch config (default 15)
    dom_cfg = qg.get("domain_classify", {}) or {}
    batch_size = _safe_int(dom_cfg.get("batch_size", 15), 15)
    domain_desc = str(dom_cfg.get("domain_desc", DEFAULT_DOMAIN_DESC))

    min_chars = _safe_int(post_cfg.get("min_query_chars", 8), 8)
    max_chars = _safe_int(post_cfg.get("max_query_chars", 160), 160)

    lower = bool(norm_cfg.get("lower", True))
    strip = bool(norm_cfg.get("strip", True))
    collapse_whitespace = bool(norm_cfg.get("collapse_whitespace", True))

    prompt_style = str(prompt_cfg.get("style", "information-seeking"))
    avoid_near_duplicates = bool(prompt_cfg.get("avoid_near_duplicates", True))

    # ---- LLM ----
    llm, llm_model_name = _build_llm_from_settings(s)

    # ---- read chunks ----
    read_errors = 0

    def _on_read_error(payload: Dict[str, Any]) -> None:
        nonlocal read_errors
        read_errors += 1
        payload = dict(payload)
        payload["ts_ms"] = _now_ms()
        append_jsonl(out_store, errors_path, [payload])

    rows = list(read_jsonl(in_store, chunks_path, on_error=_on_read_error))
    sampled = _sample_chunks(rows, sampling_cfg)

    # ---- separate dedup maps ----
    in_map: Dict[str, Dict[str, Any]] = {}
    out_map: Dict[str, Dict[str, Any]] = {}

    # ---- counters ----
    total_llm_calls_gen = 0
    total_llm_calls_domain_batch = 0
    total_raw_queries = 0
    total_after_post = 0
    total_in_unique = 0
    total_out_unique = 0

    # ---- buffer for batch domain classify ----
    # each item: {"qid":..., "query_text_norm":..., "chunk_id":...}
    buffer_items: List[Dict[str, Any]] = []

    def _flush_buffer() -> None:
        nonlocal total_llm_calls_domain_batch, total_in_unique, total_out_unique

        if not buffer_items:
            return

        dom_prompt = _build_domain_batch_prompt(buffer_items, domain_desc)
        try:
            dom_raw = llm.generate(dom_prompt)
            total_llm_calls_domain_batch += 1
        except Exception as e:
            append_jsonl(
                out_store,
                errors_path,
                [{
                    "ts_ms": _now_ms(),
                    "stage": "llm_domain_classify_batch",
                    "batch_size": int(len(buffer_items)),
                    "error": f"{type(e).__name__}: {e}",
                }],
            )
            buffer_items.clear()
            return

        in_ids, out_ids = _parse_in_out_ids(dom_raw)
        in_ids = _unique_preserve_order(in_ids)
        out_ids = _unique_preserve_order(out_ids)

        k = len(buffer_items)
        all_ids = set(range(k))
        got = set(in_ids) | set(out_ids)
        missing = sorted(all_ids - got)

        # 兜底：缺失的一律 OUT（更保守）
        out_ids = _unique_preserve_order(out_ids + missing)

        # 再兜底：如果某个 idx 同时出现在 IN 和 OUT，以 IN 为准，把它从 OUT 去掉
        in_set = set(in_ids)
        out_ids = [i for i in out_ids if i not in in_set]

        # 应用分类结果
        def _apply(idx: int, label: str) -> None:
            nonlocal total_in_unique, total_out_unique
            if idx < 0 or idx >= k:
                return
            it = buffer_items[idx]
            qid = str(it["qid"])
            q_norm = str(it["query_text_norm"])
            chunk_id = it.get("chunk_id")

            if label == "IN":
                if qid not in in_map:
                    in_map[qid] = {
                        "query_id": qid,
                        "query_text_norm": q_norm,
                        "source_chunk_ids": [chunk_id] if chunk_id else [],
                        "llm_model": llm_model_name,
                        "prompt_style": prompt_style,
                        "domain": "in",
                    }
                    total_in_unique += 1
                else:
                    if chunk_id:
                        srcs = in_map[qid]["source_chunk_ids"]
                        if chunk_id not in srcs:
                            srcs.append(chunk_id)
            else:
                if qid not in out_map:
                    out_map[qid] = {
                        "query_id": qid,
                        "query_text_norm": q_norm,
                        "source_chunk_ids": [chunk_id] if chunk_id else [],
                        "llm_model": llm_model_name,
                        "prompt_style": prompt_style,
                        "domain": "out",
                    }
                    total_out_unique += 1
                else:
                    if chunk_id:
                        srcs = out_map[qid]["source_chunk_ids"]
                        if chunk_id not in srcs:
                            srcs.append(chunk_id)

        for i in in_ids:
            _apply(i, "IN")
        for i in out_ids:
            _apply(i, "OUT")

        # 记录一下“模型漏分”的情况，方便你观察是否随 batch 变大而变差
        if missing:
            append_jsonl(
                out_store,
                errors_path,
                [{
                    "ts_ms": _now_ms(),
                    "stage": "parse_domain_batch_label",
                    "batch_size": int(k),
                    "missing_ids": [f"Q{i+1}" for i in missing][:50],
                    "llm_output_preview": dom_raw[:400],
                    "error": "Batch classifier output missing some IDs; defaulted them to OUT.",
                }],
            )

        buffer_items.clear()

    for row in sampled:
        if total_in_unique >= target_in_domain:
            break

        chunk_id = row.get("chunk_id")
        chunk_text = row.get("chunk_text", "")

        if not chunk_id or not isinstance(chunk_text, str) or not chunk_text.strip():
            append_jsonl(
                out_store,
                errors_path,
                [{
                    "ts_ms": _now_ms(),
                    "stage": "query_generation",
                    "error": "Missing chunk_id or empty chunk_text",
                    "row_preview": {k: row.get(k) for k in ("chunk_id", "doc_id", "chunk_index")},
                }],
            )
            continue

        max_chunk_chars = _safe_int(prompt_cfg.get("max_chunk_chars", 1800), 1800)
        passage = chunk_text[:max_chunk_chars]
        prompt = _build_query_prompt(passage, prompt_cfg)

        try:
            raw = llm.generate(prompt)
            total_llm_calls_gen += 1
        except Exception as e:
            append_jsonl(
                out_store,
                errors_path,
                [{
                    "ts_ms": _now_ms(),
                    "stage": "llm_generate_queries",
                    "chunk_id": chunk_id,
                    "error": f"{type(e).__name__}: {e}",
                }],
            )
            continue

        queries = _parse_queries(raw)
        total_raw_queries += len(queries)

        # chunk 内去重（用 qid）
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
            if len(q_norm) < min_chars:
                continue

            qid = _make_id(q_norm)

            if avoid_near_duplicates and (qid in seen_in_chunk):
                continue
            seen_in_chunk.add(qid)

            # 全局 exact dedup：如果已经被分类进 in/out，就没必要再进 buffer
            if (qid in in_map) or (qid in out_map):
                continue

            total_after_post += 1

            buffer_items.append(
                {
                    "qid": qid,
                    "query_text_norm": q_norm,
                    "chunk_id": chunk_id,
                }
            )

            # 满 15 就 flush 一次
            if len(buffer_items) >= batch_size:
                _flush_buffer()

                # flush 后可能已经达标
                if total_in_unique >= target_in_domain:
                    break

    # 最后把不足 15 的尾巴也 flush 掉
    if total_in_unique < target_in_domain:
        _flush_buffer()

    # ---- write outputs deterministically ----
    def _iter_sorted(m: Dict[str, Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        for qid in sorted(m.keys()):
            yield m[qid]

    write_jsonl(out_store, in_domain_path, _iter_sorted(in_map))
    write_jsonl(out_store, out_domain_path, _iter_sorted(out_map))

    stats = {
        "ts_ms": _now_ms(),
        "inputs": {
            "chunks_path": chunks_path,
            "num_chunks_total_read": int(len(rows)),
            "num_chunks_sampled": int(len(sampled)),
        },
        "outputs": {
            "queries_in_domain_path": in_domain_path,
            "queries_out_domain_path": out_domain_path,
            "errors_path": errors_path,
            "num_queries_in_domain_unique_written": int(total_in_unique),
            "num_queries_out_domain_unique_written": int(total_out_unique),
        },
        "counters": {
            "read_errors": int(read_errors),
            "llm_calls_generate_queries": int(total_llm_calls_gen),
            "llm_calls_domain_classify_batch": int(total_llm_calls_domain_batch),
            "raw_queries_parsed": int(total_raw_queries),
            "kept_after_postprocess": int(total_after_post),
            "domain_batch_size": int(batch_size),
        },
        "meta": {
            "llm_model": llm_model_name,
            "prompt_style": prompt_style,
            "target_num_in_domain": int(target_in_domain),
            "domain_desc": domain_desc,
            "normalize": {
                "lower": bool(lower),
                "strip": bool(strip),
                "collapse_whitespace": bool(collapse_whitespace),
            },
        },
    }

    return stats


def main(config_path: str) -> None:
    from qr_pipeline.settings import load_settings

    s = load_settings(config_path)
    stats = run_query_generation(s)

    out_cfg = s["outputs"]
    stores = build_store_registry(s)
    out_store = stores[str(out_cfg["store"])]
    out_base = str(out_cfg["base"])
    stats_path = _posix_join(out_base, str(out_cfg["files"]["stats"]))

    import json
    out_store.write_text(stats_path, json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
