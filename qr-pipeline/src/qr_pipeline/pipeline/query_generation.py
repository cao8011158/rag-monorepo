# src/qr_pipeline/query_generation.py
#  通过store 提取 chunks_Jsonl, 使用llm 生成 query , 将每个query 的text 进行 normalization,  然后使用hash 生成 query ID,  根据ID 进行exact dedup
#  之后对 in_domain完整的query_jsonl 进行储存.    
# queries_in_domain: queries/in_domain.jsonl (去重后) 
# queries_out_domain: queries/out_domain.jsonl  (去重后)

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


def _build_domain_prompt(query_norm: str) -> str:
    # 强约束：只输出 IN 或 OUT，便于解析
    lines: List[str] = []
    lines.append("You are a strict text classifier.")
    lines.append(
        'Determine whether the query belongs to the domain: "Pittsburgh and CMU, including history, culture, trivia, and upcoming events".'
    )
    lines.append("Output exactly one token: IN or OUT.")
    lines.append("")
    lines.append("Query:")
    lines.append(query_norm)
    lines.append("")
    lines.append("Answer:")
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


def _parse_domain_label(raw: str) -> Optional[str]:
    t = raw.strip().upper()
    if t == "IN":
        return "IN"
    if t == "OUT":
        return "OUT"
    if re.search(r"\bIN\b", t):
        return "IN"
    if re.search(r"\bOUT\b", t):
        return "OUT"
    return None


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
    -> LLM domain classify -> write in/out domain jsonl (paths from config)
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

    # 必须存在这些 key（你 config 要加上）
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

    min_chars = _safe_int(post_cfg.get("min_query_chars", 8), 8)
    max_chars = _safe_int(post_cfg.get("max_query_chars", 160), 160)

    lower = bool(norm_cfg.get("lower", True))
    strip = bool(norm_cfg.get("strip", True))
    collapse_whitespace = bool(norm_cfg.get("collapse_whitespace", True))

    prompt_style = str(prompt_cfg.get("style", "information-seeking"))
    avoid_near_duplicates = bool(prompt_cfg.get("avoid_near_duplicates", True))

    # ---- LLM ----
    llm, llm_model_name = _build_llm_from_settings(s)

    # ---- read chunks (best-effort) ----
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
    total_llm_calls_domain = 0
    total_raw_queries = 0
    total_after_post = 0
    total_in_unique = 0
    total_out_unique = 0

    for row in sampled:
        # ✅ 早停只看 IN-domain
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

            total_after_post += 1

            # ---- domain classify ----
            dom_prompt = _build_domain_prompt(q_norm)
            try:
                dom_raw = llm.generate(dom_prompt)
                total_llm_calls_domain += 1
            except Exception as e:
                append_jsonl(
                    out_store,
                    errors_path,
                    [{
                        "ts_ms": _now_ms(),
                        "stage": "llm_domain_classify",
                        "chunk_id": chunk_id,
                        "query_id": qid,
                        "query_text_norm": q_norm,
                        "error": f"{type(e).__name__}: {e}",
                    }],
                )
                continue

            label = _parse_domain_label(dom_raw)
            if label is None:
                append_jsonl(
                    out_store,
                    errors_path,
                    [{
                        "ts_ms": _now_ms(),
                        "stage": "parse_domain_label",
                        "chunk_id": chunk_id,
                        "query_id": qid,
                        "query_text_norm": q_norm,
                        "llm_output_preview": dom_raw[:200],
                        "error": "Could not parse domain label (expected IN/OUT).",
                    }],
                )
                continue

            if label == "IN":
                if qid not in in_map:
                    in_map[qid] = {
                        "query_id": qid,
                        "query_text_norm": q_norm,
                        "source_chunk_ids": [chunk_id],
                        "llm_model": llm_model_name,
                        "prompt_style": prompt_style,
                        "domain": "in",
                    }
                    total_in_unique += 1
                else:
                    srcs = in_map[qid]["source_chunk_ids"]
                    if chunk_id not in srcs:
                        srcs.append(chunk_id)
            else:
                if qid not in out_map:
                    out_map[qid] = {
                        "query_id": qid,
                        "query_text_norm": q_norm,
                        "source_chunk_ids": [chunk_id],
                        "llm_model": llm_model_name,
                        "prompt_style": prompt_style,
                        "domain": "out",
                    }
                    total_out_unique += 1
                else:
                    srcs = out_map[qid]["source_chunk_ids"]
                    if chunk_id not in srcs:
                        srcs.append(chunk_id)

    # ---- write outputs deterministically ----
    def _iter_sorted(m: Dict[str, Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        for qid in sorted(m.keys()):
            yield m[qid]

    write_jsonl(out_store, in_domain_path, _iter_sorted(in_map))
    write_jsonl(out_store, out_domain_path, _iter_sorted(out_map))

    # ---- stats (建议字段与 config 对齐) ----
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
            "llm_calls_domain_classify": int(total_llm_calls_domain),
            "raw_queries_parsed": int(total_raw_queries),
            "kept_after_postprocess": int(total_after_post),
        },

        "meta": {
            "llm_model": llm_model_name,
            "prompt_style": prompt_style,
            "target_num_in_domain": int(target_in_domain),
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

    # run_stats.json 路径来自 config
    out_cfg = s["outputs"]
    stores = build_store_registry(s)
    out_store = stores[str(out_cfg["store"])]
    out_base = str(out_cfg["base"])
    stats_path = _posix_join(out_base, str(out_cfg["files"]["stats"]))

    import json
    out_store.write_text(stats_path, json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
