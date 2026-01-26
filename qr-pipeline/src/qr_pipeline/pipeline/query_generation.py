# src/qr_pipeline/query_generation.py
# 通过 store 提取 chunks_jsonl, 使用 LLM 一段式生成 query，并按 domain 分成 IN/OUT 两组输出
# 对每个 query 做 normalization -> hash id -> exact dedup
# queries_in_domain:  queries/in_domain.jsonl (去重后)
# queries_out_domain: queries/out_domain.jsonl (去重后)

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
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

    def __post_init__(self) -> None:
        try:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as e:
            raise RuntimeError("Missing dependency for hf_transformers LLM. Please install transformers + torch.") from e

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
        # always move inputs to model device (works for cpu/cuda)
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
        # strip prompt prefix if echoed
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
            max_new_tokens=_safe_int(llm_cfg.get("max_new_tokens", 256), 256),
            temperature=_safe_float(llm_cfg.get("temperature", 0.7), 0.7),
            top_p=_safe_float(llm_cfg.get("top_p", 0.9), 0.9),
        )
        return llm, model_name

    raise NotImplementedError(f"LLM provider not supported yet: {provider}")


# -----------------------------
# Prompt building & parsing (ONE-SHOT)
# -----------------------------
def _build_queries_with_domain_prompt(chunk_text: str, p: Dict[str, Any]) -> str:
    """
    One-shot: generate queries AND classify into IN/OUT domain in a single response.

    Output format (STRICT):
    IN:
    <one query per line>
    OUT:
    <one query per line>
    """
    lang = str(p.get("language", "en"))
    style = str(p.get("style", "information-seeking"))
    n = _safe_int(p.get("num_queries_per_chunk", 1), 1)

    diversify = bool(p.get("diversify", False))
    hints = p.get("diversity_hints", []) or []
    hints_str = ", ".join([str(x) for x in hints]) if hints else ""

    domain_desc = 'question about Pittsburgh and CMU, including history, culture, trivia, and upcoming events'

    if lang.lower().startswith("en"):
        lines: List[str] = []
        lines.append("You are a helpful assistant that generates search queries and classifies them by domain relevance.")
        lines.append(f'Domain: "{domain_desc}".')
        lines.append(
            f"Task: generate exactly {n} {style} queries in total (IN + OUT combined = {n}) that a user might ask to find information in the passage."
        )
        if diversify and hints_str:
            lines.append(f"Diversify query types using these hints: {hints_str}.")
        lines.append("")

        lines.append("CRITICAL output rules:")
        lines.append("1) Output MUST have exactly two sections, in this exact order and with these exact headers:")
        lines.append("IN:")
        lines.append("(one query per line, no numbering, no bullets, no explanations)")
        lines.append("OUT:")
        lines.append("(one query per line, no numbering, no bullets, no explanations)")
        lines.append("2) Queries under IN must be clearly within the Domain. Queries under OUT must be outside the Domain.")
        lines.append("3) The total number of queries across IN and OUT must be exactly "
                    f"{n}. Either section may be empty.")
        lines.append("4) Do NOT include anything other than the two sections.")
        lines.append("")

        lines.append("Example (format only, do NOT reuse its content):")
        lines.append("Passage:")
        lines.append(
            "Carnegie Mellon University (CMU) was founded in 1900 in Pittsburgh, Pennsylvania. "
            "It is known for its strong programs in computer science and robotics and works closely "
            "with local industry and research institutions in the city."
        )
        lines.append("")
        lines.append("Answer:")
        lines.append("IN:")
        lines.append("When was Carnegie Mellon University founded?")
        lines.append("What is Carnegie Mellon University known for?")
        lines.append("OUT:")
        lines.append("How do I apply for a driving license in Pennsylvania?")
        lines.append("")

        lines.append("Passage:")
        lines.append(chunk_text)
        lines.append("")
        lines.append("Answer:")

        return "\n".join(lines)


    # fallback
    return _build_queries_with_domain_prompt(chunk_text, {**p, "language": "en"})


def _strip_query_line(x: str) -> str:
    """Remove bullets/numbering similar to the old parser; keep the text-only query."""
    y = x.strip()
    if not y:
        return ""
    # remove bullets
    y = re.sub(r"^\s*[\-\*\u2022]\s+", "", y)
    # remove leading numbering like "1." "2)" "3:"
    y = re.sub(r"^\s*\d+[\.\)\:]\s+", "", y)
    return y.strip()


def _parse_in_out_queries(raw: str) -> Tuple[List[str], List[str]]:
    """
    Parse one-shot output:
    IN:
    ...
    OUT:
    ...

    Be tolerant to extra whitespace and different casing.
    """
    text = raw.strip()
    if not text:
        return [], []

    lines = text.splitlines()

    in_lines: List[str] = []
    out_lines: List[str] = []
    mode: Optional[str] = None  # "IN" | "OUT"

    for line in lines:
        t = line.strip()
        if not t:
            continue

        # header detection (case-insensitive)
        if re.fullmatch(r"IN\s*:\s*", t, flags=re.IGNORECASE):
            mode = "IN"
            continue
        if re.fullmatch(r"OUT\s*:\s*", t, flags=re.IGNORECASE):
            mode = "OUT"
            continue

        if mode == "IN":
            q = _strip_query_line(t)
            if q:
                in_lines.append(q)
        elif mode == "OUT":
            q = _strip_query_line(t)
            if q:
                out_lines.append(q)
        else:
            # ignore anything before headers (or model mistakes)
            continue

    return in_lines, out_lines


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
    chunks.jsonl -> (ONE-SHOT) generate queries + IN/OUT grouping
    -> normalize -> hash -> exact dedup
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
        p2 = dict(payload)
        p2["ts_ms"] = _now_ms()
        append_jsonl(out_store, errors_path, [p2])

    rows = list(read_jsonl(in_store, chunks_path, on_error=_on_read_error))
    sampled = _sample_chunks(rows, sampling_cfg)

    # ---- separate dedup maps ----
    in_map: Dict[str, Dict[str, Any]] = {}
    out_map: Dict[str, Dict[str, Any]] = {}

    # ---- counters ----
    total_llm_calls_gen = 0
    total_raw_in = 0
    total_raw_out = 0
    total_after_post = 0
    total_in_unique = 0
    total_out_unique = 0
    parse_failures = 0

    for row in sampled:
        # ✅ early-stop ONLY based on IN-domain
        if total_in_unique >= target_in_domain:
            break

        chunk_id = row.get("chunk_id")
        chunk_text = row.get("chunk_text", "")

        if not chunk_id or not isinstance(chunk_text, str) or not chunk_text.strip():
            append_jsonl(
                out_store,
                errors_path,
                [
                    {
                        "ts_ms": _now_ms(),
                        "stage": "query_generation",
                        "error": "Missing chunk_id or empty chunk_text",
                        "row_preview": {k: row.get(k) for k in ("chunk_id", "doc_id", "chunk_index")},
                    }
                ],
            )
            continue

        max_chunk_chars = _safe_int(prompt_cfg.get("max_chunk_chars", 1800), 1800)
        passage = chunk_text[:max_chunk_chars]

        prompt = _build_queries_with_domain_prompt(passage, prompt_cfg)

        try:
            raw = llm.generate(prompt)
            total_llm_calls_gen += 1
        except Exception as e:
            append_jsonl(
                out_store,
                errors_path,
                [
                    {
                        "ts_ms": _now_ms(),
                        "stage": "llm_generate_and_classify_queries",
                        "chunk_id": chunk_id,
                        "error": f"{type(e).__name__}: {e}",
                    }
                ],
            )
            continue

        in_qs, out_qs = _parse_in_out_queries(raw)
        total_raw_in += len(in_qs)
        total_raw_out += len(out_qs)

        # if both empty, treat as parse failure (optional, but helpful)
        if len(in_qs) == 0 and len(out_qs) == 0:
            parse_failures += 1
            append_jsonl(
                out_store,
                errors_path,
                [
                    {
                        "ts_ms": _now_ms(),
                        "stage": "parse_in_out_queries",
                        "chunk_id": chunk_id,
                        "error": "Could not parse any queries under IN:/OUT: headers.",
                        "llm_output_preview": raw[:400],
                    }
                ],
            )
            continue

        # chunk-local dedup uses qid (across both IN/OUT lists)
        seen_in_chunk: set[str] = set()

        def _consume_queries(qs: List[str], label: str) -> None:
            nonlocal total_after_post, total_in_unique, total_out_unique
            for q in qs:
                # ✅ stop only depends on IN unique count
                if total_in_unique >= target_in_domain:
                    return

                if not isinstance(q, str):
                    continue

                qq = q.strip()
                if len(qq) < min_chars:
                    continue
                if len(qq) > max_chars:
                    qq = qq[:max_chars].strip()

                q_norm = _normalize_text(qq, lower=lower, strip=strip, collapse_whitespace=collapse_whitespace)
                if len(q_norm) < min_chars:
                    continue

                qid = _make_id(q_norm)

                if avoid_near_duplicates and (qid in seen_in_chunk):
                    continue
                seen_in_chunk.add(qid)

                total_after_post += 1

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

        # prefer to consume IN first (since early-stop is based on IN)
        _consume_queries(in_qs, "IN")
        # you may choose to skip OUT when already reached target
        if total_in_unique < target_in_domain:
            _consume_queries(out_qs, "OUT")

    # ---- write outputs deterministically ----
    def _iter_sorted(m: Dict[str, Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        for qid in sorted(m.keys()):
            yield m[qid]

    write_jsonl(out_store, in_domain_path, _iter_sorted(in_map))
    write_jsonl(out_store, out_domain_path, _iter_sorted(out_map))

    # ---- stats ----
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
            "llm_calls_generate_and_classify": int(total_llm_calls_gen),
            "raw_queries_in_parsed": int(total_raw_in),
            "raw_queries_out_parsed": int(total_raw_out),
            "kept_after_postprocess": int(total_after_post),
            "parse_failures": int(parse_failures),
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
            "one_shot": {
                "output_format": "IN:/OUT: sections",
                "domain": 'question about Pittsburgh and CMU, including history, culture, trivia, and upcoming events',
            },
        },
    }

    return stats


def main(config_path: str) -> None:
    from qr_pipeline.settings import load_settings

    s = load_settings(config_path)
    stats = run_query_generation(s)

    # run_stats.json path from config
    out_cfg = s["outputs"]
    stores = build_store_registry(s)
    out_store = stores[str(out_cfg["store"])]
    out_base = str(out_cfg["base"])
    stats_path = _posix_join(out_base, str(out_cfg["files"]["stats"]))

    import json

    out_store.write_text(stats_path, json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
