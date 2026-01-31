from __future__ import annotations

import time
from typing import Any, Dict, List, Set

from pydantic import BaseModel, Field, ValidationError
from google import genai


# =========================
# Structured output schema
# =========================

class ClassifyOutput(BaseModel):
    in_ids: List[int] = Field(description="1-based indices of IN queries, e.g. [1,3,4].")
    out_ids: List[int] = Field(description="1-based indices of OUT queries, e.g. [2,5].")


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
    - optionally fill missing as OUT (strict fallback)
    """
    in_ids = _dedup_ints_keep_order([i for i in in_ids if 1 <= i <= n])
    out_ids = _dedup_ints_keep_order([i for i in out_ids if 1 <= i <= n])

    in_set = set(in_ids)
    out_ids = [i for i in out_ids if i not in in_set]  # remove overlaps (IN wins)
    out_set = set(out_ids)

    if fill_missing_as_out:
        for i in range(1, n + 1):
            if i not in in_set and i not in out_set:
                out_ids.append(i)

    # final stable order
    in_ids = [i for i in range(1, n + 1) if i in set(in_ids)]
    out_ids = [i for i in range(1, n + 1) if i in set(out_ids)]
    return ClassifyOutput(in_ids=in_ids, out_ids=out_ids)


# =========================
# Core Gemini call (structured output)
# =========================

def classify_queries_with_gemini(
    *,
    queries: List[str],
    model_name: str,
    max_retries: int = 3,
    backoff_sec: float = 1.5,
    fill_missing_as_out: bool = True,
) -> Dict[str, Any]:
    """
    Returns:
      {"in_ids": [...], "out_ids": [...]}

    Level 2:
    - structured output (JSON schema)
    - retry with backoff
    - strict ID validation
    """

    if not isinstance(queries, list) or not all(isinstance(x, str) for x in queries):
        raise TypeError("queries must be a List[str]")

    n = len(queries)
    if n == 0:
        return {"in_ids": [], "out_ids": []}

    # ✅ 不搞“函数式改写 prompt”，只把 Q1..Qn 追加到你给的 prompt string 后面
    prompt = """Task :
You will receive N user queries, for each query, assign exactly one of the following two labels:
- IN
- OUT
IN: queries about or related to Pittsburgh or Carnegie Mellon University (CMU), including their history, geography, culture, population, economy, campus life, traditions, trivia, and current events.
OUT: everything else.
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

    lines: List[str] = [prompt.rstrip(), "", "Input:"]
    for i, q in enumerate(queries, start=1):
        qq = (q or "").strip().replace("\n", " ")
        lines.append(f"{i}: {qq}")
    prompt = "\n".join(lines)

    client = genai.Client()

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "system_instruction": "You are a strict classifier.",
                    "response_mime_type": "application/json",
                    "response_json_schema": ClassifyOutput.model_json_schema(),
                },
            )

            # ✅ 结构化输出：按 schema 解析 JSON
            parsed = ClassifyOutput.model_validate_json(response.text)

            # ✅ 严格校验 + 可选补全
            final = _validate_ids(
                in_ids=parsed.in_ids,
                out_ids=parsed.out_ids,
                n=n,
                fill_missing_as_out=fill_missing_as_out,
            )

            return {"in_ids": final.in_ids, "out_ids": final.out_ids}

        except (ValidationError, ValueError) as e:
            # 不可重试：schema/格式不符合
            raise RuntimeError(f"Invalid structured output format: {e}") from e

        except Exception as e:
            # 其他异常：可重试（网络/限流/临时错误）
            last_err = e
            print(f"[WARN] failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(backoff_sec * attempt)
                continue
            break

    raise RuntimeError(f"Gemini classification failed after {max_retries} retries: {last_err}")


# =========================
# Public API (pipeline entry)
# =========================

def run_query_classification(
    queries: List[str],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Expected keys:
      cfg["models"]["gemini_api"]["model_name"]

    Returns:
      {"in_ids": [...], "out_ids": [...]}
    """
    model_name = _get_required(cfg, "models.gemini_api.model_name")
    return classify_queries_with_gemini(
        queries=queries,
        model_name=str(model_name),
    )
