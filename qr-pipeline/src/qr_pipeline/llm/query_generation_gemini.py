from __future__ import annotations

import time
from typing import Any, Dict, List

from pydantic import BaseModel, Field, ValidationError
from google import genai


# =========================
# Structured output schema
# =========================

class QueryOutput(BaseModel):
    queries: List[str] = Field(description="A list of search queries.")


# =========================
# Utils
# =========================

def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _get_required(cfg: Dict[str, Any], path: str) -> Any:
    """
    Fetch nested keys using dotted path, e.g. "models.gemini_api.model_name".
    Raises KeyError with clear message if missing.
    """
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing required config key: {path}")
        cur = cur[part]
    return cur


# =========================
# Prompt builder (adapted from your function)
# =========================

def build_query_prompt(chunk_text: str, p: Dict[str, Any]) -> str:
    """
    p matches your config format:
      style: "information-seeking"
      num_queries_per_chunk: 3
      max_chunk_chars: 2100
      diversify: true
      diversity_hints: str
      avoid_near_duplicates: true
    """
    style = str(p.get("style", "information-seeking"))
    n = _safe_int(p.get("num_queries_per_chunk", 1), 1)
    diversify = bool(p.get("diversify", True))
    hints = p.get("diversity_hints") or ""
    passage = (chunk_text or "")

    lines: List[str] = []
    lines.append(
        f"Task: generate {n} {style} queries that a user might ask to find the information in the passage."
    )
    if diversify and hints:
        lines.append("Diversify query types using these hints:")
        lines.append(hints.strip())
    lines.append("")
    lines.append("Passage:")
    lines.append(passage)
    return "\n".join(lines)


# =========================
# Post-process & validation
# =========================

def _normalize_queries(queries: List[Any], n: int) -> List[str]:
    cleaned: List[str] = []
    for q in queries:
        if isinstance(q, str):
            s = q.strip()
            if s:
                cleaned.append(s)
    # Exact-string dedup while preserving order
    deduped = list(dict.fromkeys(cleaned))
    return deduped


# =========================
# Core Gemini call (new SDK, structured output)
# =========================

def generate_queries_with_gemini(
    *,
    chunk_text: str,
    prompt_cfg: Dict[str, Any],
    model_name: str,
    max_retries: int = 3,
    backoff_sec: float = 1.5,
) -> Dict[str, Any]:
    """
    Returns: {"queries": [...]}

    Level 2:
    - structured output (JSON schema)
    - retry with backoff
    - strict count + dedup validation
    """
    n = _safe_int(prompt_cfg.get("num_queries_per_chunk", 1), 1)
    prompt = build_query_prompt(chunk_text, prompt_cfg)

    client = genai.Client()

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "system_instruction": "You are a helpful assistant that generates search queries.",
                    "response_mime_type": "application/json",
                    "response_json_schema": QueryOutput.model_json_schema(),
                },
            )
            parsed = QueryOutput.model_validate_json(response.text)
            qs = _normalize_queries(parsed.queries, n)
            return {"queries": qs}

        except (ValidationError, ValueError) as e:
            # 不可重试：格式/schema问题
            raise RuntimeError(f"Invalid output format: {e}") from e

        except Exception as e:
            # 其他异常：这里才是“可能可重试”的范围
            last_err = e
            print(f"[WARN] failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(backoff_sec * attempt)
                continue
            break

    raise RuntimeError(f"Gemini query generation failed after {max_retries} retries: {last_err}")


# =========================
# Public API (your pipeline calls this)
# =========================

def run_query_generation(chunk_text: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    cfg is already-parsed dict (by your own config tool).

    Expected keys (as in your screenshots):
      cfg["models"]["gemini_api"]["model_name"]
      cfg["query_generation"]["prompt"]

    Returns:
      {"queries": [...]}
    """
    model_name = _get_required(cfg, "models.gemini_api.model_name")
    prompt_cfg = _get_required(cfg, "query_generation.prompt")

    if not isinstance(prompt_cfg, dict):
        raise TypeError("query_generation.prompt must be a dict")

    return generate_queries_with_gemini(
        chunk_text=chunk_text,
        prompt_cfg=prompt_cfg,
        model_name=str(model_name),
    )
