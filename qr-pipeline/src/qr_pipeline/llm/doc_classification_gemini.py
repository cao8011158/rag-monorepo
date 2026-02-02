from __future__ import annotations

import time
from typing import Any, Dict, List, Set

from pydantic import BaseModel, Field, ValidationError
from google import genai


# =========================
# Structured output schema
# =========================

class ClassifyPNOutput(BaseModel):
    positive_ids: List[int] = Field(
        description=(
            "0-based indices of documents that are POSITIVE (answerable/supported) for the query. "
            "Return [] if none is a valid positive."
        )
    )


# =========================
# Utils
# =========================

def _get_required(cfg: Dict[str, Any], path: str) -> Any:
    """Fetch nested keys using dotted path."""
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


def _validate_positive_ids(
    *,
    positive_ids: List[int],
    n: int,
) -> ClassifyPNOutput:
    """
    Enforce:
    - 0 <= id < n
    - dedup
    - stable order (by document order)
    """
    positive_ids = _dedup_ints_keep_order(
        [i for i in positive_ids if 0 <= i < n]
    )
    positive_ids = [i for i in range(n) if i in set(positive_ids)]
    return ClassifyPNOutput(positive_ids=positive_ids)


# =========================
# Prompt builder
# =========================

def _build_prompt(
    *,
    query: str,
    documents: List[str],
    max_doc_chars: int = 2000,
) -> str:
    q = (query or "").strip().replace("\n", " ")
    if not q:
        raise ValueError("query must be a non-empty string")

    lines: List[str] = []
    lines.append("Task:")
    lines.append("You will receive ONE query and N candidate documents.")
    lines.append("Select ONLY the documents that are POSITIVE (Answerable / Supported).")
    lines.append("")
    lines.append("Definition of POSITIVE (must satisfy ALL):")
    lines.append("1) Directness: the document contains the core answer (key entity/number/definition/claim), not just background.")
    lines.append("2) Sufficiency: the document alone can answer the query without needing other documents.")
    lines.append("3) Evidence: the answer must be explicitly supported by a quoted span in the document.")
    lines.append("")
    lines.append("If uncertain, do NOT mark it positive.")
    lines.append("")
    lines.append("Query:")
    lines.append(q)
    lines.append("")
    lines.append("Documents:")

    for i, doc in enumerate(documents):
        text = (doc or "").strip().replace("\n", " ")
        if len(text) > max_doc_chars:
            text = text[:max_doc_chars] + "..."
        lines.append(f"{i}: {text}")

    return "\n".join(lines)


# =========================
# Core Gemini call
# =========================

def classify_query_docs_with_gemini(
    *,
    query: str,
    documents: List[str],
    model_name: str,
    max_retries: int = 3,
    backoff_sec: float = 1.5,
) -> Dict[str, Any]:
    """
    Returns:
      {"positive_ids": [0, 2, ...]}   # 0-based indices
    """

    if not isinstance(documents, list) or not all(isinstance(x, str) for x in documents):
        raise TypeError("documents must be List[str]")

    n = len(documents)
    if n == 0:
        return {"positive_ids": []}

    prompt = _build_prompt(query=query, documents=documents)

    client = genai.Client()

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "system_instruction": "You are a strict answerability classifier.",
                    "response_mime_type": "application/json",
                    "response_json_schema": ClassifyPNOutput.model_json_schema(),
                },
            )

            parsed = ClassifyPNOutput.model_validate_json(response.text)

            final = _validate_positive_ids(
                positive_ids=parsed.positive_ids,
                n=n,
            )

            return {"positive_ids": final.positive_ids}

        except (ValidationError, ValueError) as e:
            raise RuntimeError(f"Invalid structured output format: {e}") from e

        except Exception as e:
            last_err = e
            print(f"[WARN] Gemini failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(backoff_sec * attempt)
                continue
            break

    raise RuntimeError(f"Gemini classification failed after {max_retries} retries: {last_err}")


# =========================
# Public API (pipeline entry)
# =========================

def run_gemini_classification_PN(
    query: str,
    documents: List[str],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Expected keys:
      cfg["models"]["gemini_api"]["model_name"]

    Returns:
      {"positive_ids": [0, 2, ...]}
    """
    model_name = _get_required(cfg, "models.gemini_api.model_name")

    return classify_query_docs_with_gemini(
        query=query,
        documents=documents,
        model_name=str(model_name),
    )
