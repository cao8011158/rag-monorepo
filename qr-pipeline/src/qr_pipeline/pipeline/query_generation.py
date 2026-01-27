# src/qr_pipeline/query_generation.py
#  chunks.jsonl -> LLM generate queries -> normalize -> hash (real query_id)
#  -> exact dedup by real query_id
#  -> domain classify in BATCH (Q1..Qn labels + one-shot example, ID-only output)
#  -> write queries/in_domain.jsonl and queries/out_domain.jsonl

from __future__ import annotations

import hashlib
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

from qr_pipeline.io.jsonl import append_jsonl, read_jsonl, write_jsonl
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
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore # noqa: F401
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


def _build_domain_batch_prompt_qstyle(items: List[Tuple[str, str]]) -> Tuple[str, Dict[str, str]]:
    """
    items: List[(real_query_id, query_text_norm)]
    prompt uses short labels Q1..Qn to reduce context burden.
    Returns:
      prompt_text
      qlabel_to_real_id: {"Q1": real_id1, ...}
    """
    qlabel_to_real_id: Dict[str, str] = {}
    lines: List[str] = []
    lines.append("You are a strict classifier.")
    lines.append("")
    lines.append("Domain:")
    lines.append(
        "Questions that explicitly focus on Pittsburgh or Carnegie Mellon University (CMU). "
        "This includes their history, geography, culture, population, economy, campus life, traditions, trivia, and current events.")
    lines.append("")
    lines.append("Decision rules:")
    lines.append(
    'Choose IN only if the query is primarily about Pittsburgh or CMU '
    'and explicitly mentions Pittsburgh or Carnegie Mellon University (CMU).'
    )
    lines.append('Choose OUT for all other queries. even if the topic could be related.') 
    lines.append("- If uncertain, choose OUT.")
    lines.append("")
    lines.append("Output format (strict):")
    lines.append("IN:")
    lines.append("Q<number>")
    lines.append("OUT:")
    lines.append("Q<number>")
    lines.append("")
    lines.append("One-shot example:")
    lines.append("Input:")
    lines.append("Q1\twhat year was carnegie mellon university founded")
    lines.append("Q2\tbest universities for computer science in the us")
    lines.append("Q3\tmajor sports teams in pittsburgh")
    lines.append("Q4\thow did the whiskey rebellion impact pittsburgh's development")
    lines.append("Q5\twhat was the french and indian war")
    lines.append("")
    lines.append("Output:")
    lines.append("IN:")
    lines.append("Q1")
    lines.append("Q3")
    lines.append("Q4")
    lines.append("OUT:")
    lines.append("Q2")
    lines.append("Q5")
    lines.append("")
    lines.append("Now classify the following queries:")
    lines.append("Input:")


    for i, (real_id, qnorm) in enumerate(items, start=1):
        qlabel = f"Q{i}"
        qlabel_to_real_id[qlabel] = real_id
        lines.append(f"{qlabel}\t{qnorm}")

    lines.append("")
    lines.append("Output:")
    return "\n".join(lines), qlabel_to_real_id


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


def _parse_domain_batch_qstyle(raw: str, qlabel_to_real_id: Dict[str, str]) -> Tuple[Dict[str, str], List[str]]:
    """
    Parse output:
      IN:
      Q1
      Q3
      OUT:
      Q2
    Return:
      labels_by_real_id: {real_id: "IN"/"OUT"}
      errors: [...]
    Policy:
      - unknown Q labels are errors
      - missing labels -> OUT (conservative)
      - if a label appears in both, OUT wins (conservative)
    """
    errors: List[str] = []
    labels_by_real_id: Dict[str, str] = {}

    text = (raw or "").strip()
    lines = text.splitlines()

    in_idx: Optional[int] = None
    out_idx: Optional[int] = None
    for i, ln in enumerate(lines):
        t = ln.strip().upper()
        if in_idx is None and t == "IN:":
            in_idx = i
            continue
        if t == "OUT:":
            out_idx = i
            break

    if in_idx is None or out_idx is None or out_idx <= in_idx:
        return labels_by_real_id, ["Missing or malformed IN:/OUT: sections."]

    def _collect(section_lines: List[str]) -> List[str]:
        got: List[str] = []
        for ln in section_lines:
            x = ln.strip()
            if not x:
                continue
            x = re.sub(r"^\s*[\-\*\u2022]\s+", "", x)
            x = re.sub(r"^\s*\d+[\.\)\:]\s+", "", x)
            x = x.strip()
            if x:
                got.append(x)
        return got

    in_q = _collect(lines[in_idx + 1 : out_idx])
    out_q = _collect(lines[out_idx + 1 :])

    # Apply IN first, then OUT overrides if conflict
    for q in in_q:
        if q not in qlabel_to_real_id:
            errors.append(f"Unknown label: {q}")
            continue
        real_id = qlabel_to_real_id[q]
        labels_by_real_id[real_id] = "IN"

    for q in out_q:
        if q not in qlabel_to_real_id:
            errors.append(f"Unknown label: {q}")
            continue
        real_id = qlabel_to_real_id[q]
        labels_by_real_id[real_id] = "OUT"

    # Missing -> OUT
    for qlabel, real_id in qlabel_to_real_id.items():
        if real_id not in labels_by_real_id:
            labels_by_real_id[real_id] = "OUT"

    return labels_by_real_id, errors


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
    -> LLM domain classify (BATCH, Q1..Qn labels) -> write in/out domain jsonl (paths from config)
    Early stop ONLY based on IN-domain unique queries count.
    """
    stores = build_store_registry(s)

    # ---- inputs: chunks ----
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

    # ---- LLM ----
    llm, llm_model_name = _build_llm_from_settings(s)

    # ---- counters (MUST be defined BEFORE inner function uses nonlocal) ----
    read_errors = 0

    total_llm_calls_gen = 0
    total_llm_calls_domain = 0  # batch calls
    total_raw_queries = 0
    total_after_post = 0
    total_in_unique = 0
    total_out_unique = 0

    total_domain_batches = 0
    total_domain_items_classified = 0

    # ---- read chunks (best-effort) ----
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

    # ---- batch buffer ----
    # each: {"real_id": qid_hash, "q_norm": ..., "chunk_id": ...}
    batch_buf: List[Dict[str, Any]] = []

    def _flush_domain_batch() -> None:
        nonlocal total_llm_calls_domain, total_domain_batches
        nonlocal total_domain_items_classified, total_in_unique, total_out_unique

        if not batch_buf:
            return

        items: List[Tuple[str, str]] = [(x["real_id"], x["q_norm"]) for x in batch_buf]
        prompt, qlabel_to_real_id = _build_domain_batch_prompt_qstyle(items)

        try:
            raw = llm.generate(prompt)
            total_llm_calls_domain += 1
            total_domain_batches += 1
        except Exception as e:
            append_jsonl(
                out_store,
                errors_path,
                [{
                    "ts_ms": _now_ms(),
                    "stage": "llm_domain_classify_batch",
                    "error": f"{type(e).__name__}: {e}",
                    "batch_size": int(len(batch_buf)),
                    "real_ids_preview": [x["real_id"] for x in batch_buf[:10]],
                }],
            )
            batch_buf.clear()
            return

        labels_by_real_id, perr = _parse_domain_batch_qstyle(raw, qlabel_to_real_id)
        if perr:
            append_jsonl(
                out_store,
                errors_path,
                [{
                    "ts_ms": _now_ms(),
                    "stage": "parse_domain_batch",
                    "batch_size": int(len(batch_buf)),
                    "errors": perr,
                    "llm_output_preview": raw[:500],
                }],
            )

        for x in batch_buf:
            real_id = x["real_id"]
            q_norm = x["q_norm"]
            chunk_id = x["chunk_id"]

            label = labels_by_real_id.get(real_id, "OUT")
            total_domain_items_classified += 1

            if label == "IN":
                if real_id not in in_map:
                    in_map[real_id] = {
                        "query_id": real_id,
                        "query_text_norm": q_norm,
                        "source_chunk_ids": [chunk_id],
                        "llm_model": llm_model_name,
                        "prompt_style": prompt_style,
                        "domain": "in",
                    }
                    total_in_unique += 1
                else:
                    srcs = in_map[real_id]["source_chunk_ids"]
                    if chunk_id not in srcs:
                        srcs.append(chunk_id)
            else:
                if real_id not in out_map:
                    out_map[real_id] = {
                        "query_id": real_id,
                        "query_text_norm": q_norm,
                        "source_chunk_ids": [chunk_id],
                        "llm_model": llm_model_name,
                        "prompt_style": prompt_style,
                        "domain": "out",
                    }
                    total_out_unique += 1
                else:
                    srcs = out_map[real_id]["source_chunk_ids"]
                    if chunk_id not in srcs:
                        srcs.append(chunk_id)

        batch_buf.clear()

    # -----------------------------
    # Loop
    # -----------------------------
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

            real_id = _make_id(q_norm)

            if avoid_near_duplicates and (real_id in seen_in_chunk):
                continue
            seen_in_chunk.add(real_id)

            total_after_post += 1

            # already classified? just append source
            if real_id in in_map:
                srcs = in_map[real_id]["source_chunk_ids"]
                if chunk_id not in srcs:
                    srcs.append(chunk_id)
                continue
            if real_id in out_map:
                srcs = out_map[real_id]["source_chunk_ids"]
                if chunk_id not in srcs:
                    srcs.append(chunk_id)
                continue

            batch_buf.append({"real_id": real_id, "q_norm": q_norm, "chunk_id": chunk_id})

            if len(batch_buf) >= batch_size:
                _flush_domain_batch()
                if total_in_unique >= target_in_domain:
                    break

    # flush remaining
    _flush_domain_batch()

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
            "llm_calls_domain_classify_batches": int(total_llm_calls_domain),
            "domain_classify_batches": int(total_domain_batches),
            "domain_classify_items": int(total_domain_items_classified),
            "domain_batch_size": int(batch_size),
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

    out_cfg = s["outputs"]
    stores = build_store_registry(s)
    out_store = stores[str(out_cfg["store"])]
    out_base = str(out_cfg["base"])
    stats_path = _posix_join(out_base, str(out_cfg["files"]["stats"]))

    import json

    out_store.write_text(stats_path, json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
