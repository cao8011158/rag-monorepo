from __future__ import annotations

import time
import re
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ValidationError
from google import genai


# =========================
# Structured output schema
# =========================

class QAOutput(BaseModel):
    question: str = Field(description="Must be exactly the original input query_text.")
    answer: str = Field(
        description=(
            "A concise answer grounded ONLY in the given chunk_text. "
            "If the chunk does not contain enough information to answer, return an empty string."
        )
    )


# =========================
# Utils
# =========================

def _get_required(cfg: Dict[str, Any], path: str) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing required config key: {path}")
        cur = cur[part]
    return cur


def _clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _validate_qa(
    *,
    original_query: str,
    parsed: QAOutput,
    max_answer_tokens_hint: int = 80,  # soft hint; Gemini may not be exact
) -> QAOutput:
    # Enforce question is EXACT original query (after strip normalization)
    oq = (original_query or "").strip()
    pq = (parsed.question or "").strip()
    if pq != oq:
        # hard fail: you explicitly want question == original query
        raise ValueError(f"Invalid output: question must equal original query.\nGot: {pq}\nExpected: {oq}")

    # Basic sanity cleanup (do NOT over-edit content)
    ans = (parsed.answer or "").strip()

    # Optional: prevent obvious "I don't know" style (you can relax this)
    # If it's not answerable, require empty string.
    lowered = ans.lower()
    if any(x in lowered for x in ["not enough information", "cannot be determined", "i don't know", "i cannot", "unknown"]):
        ans = ""

    # Keep answer reasonably short in practice (soft clamp by characters as a safety net).
    # 80 tokens ~ often <= ~500-700 chars depending on language; use conservative clamp.
    if len(ans) > 900:
        ans = ans[:900].rstrip()

    return QAOutput(question=oq, answer=ans)


# =========================
# Core Gemini call (structured output)
# =========================

def generate_grounded_qa_with_gemini(
    *,
    query_text: str,
    chunk_text: str,
    model_name: str,
    max_retries: int = 3,
    backoff_sec: float = 1.5,
) -> Dict[str, str]:
    """
    Input:
      query_text: original user query (used as question, unchanged)
      chunk_text: the source chunk; answer must be grounded ONLY in this text

    Output:
      {"question": <original query_text>, "answer": <40-80 token concise answer or "">}

    Notes:
      - structured output via JSON schema
      - retry/backoff for transient errors
      - strict validation: question must equal original query
    """
    q = (query_text or "").strip()
    c = (chunk_text or "").strip()

    if not q:
        return {"question": "", "answer": ""}

    # Keep chunk reasonably sized; no heavy truncation needed.
    # Still protect against extreme cases.
    if len(c) > 6000:
        c = c[:6000]

    prompt = f"""Task:
You will receive:
- query_text: the user question
- chunk_text: a source passage

Rules (STRICT):
1) You MUST set output.question exactly equal to query_text (verbatim).
2) You MUST answer using ONLY facts stated in chunk_text. Do NOT use outside knowledge.
3) If chunk_text does not contain enough information to answer query_text, set output.answer to "" (empty string).
4) output.answer must be a concise summary-style answer, ideally 40–80 tokens. No extra commentary.

query_text:
{q}

chunk_text:
\"\"\"{c}\"\"\"

"""

    client = genai.Client()

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "system_instruction": (
                        "You are a careful data generator for reranker training. "
                        "You must be strictly grounded in the provided chunk."
                    ),
                    "response_mime_type": "application/json",
                    "response_json_schema": QAOutput.model_json_schema(),
                },
            )

            parsed = QAOutput.model_validate_json(resp.text)
            final = _validate_qa(original_query=q, parsed=parsed)

            return {"question": final.question, "answer": final.answer}

        except (ValidationError, ValueError) as e:
            # schema mismatch or strict grounding validation fail -> treat as format error
            raise RuntimeError(f"Invalid structured output format: {e}") from e

        except Exception as e:
            last_err = e
            print(f"[WARN] failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(backoff_sec * attempt)
                continue
            break

    raise RuntimeError(f"Gemini QA generation failed after {max_retries} retries: {last_err}")


# =========================
# Public API (pipeline entry)
# =========================

def run_grounded_qa_generation(
    *,
    query_text: str,
    chunk_text: str,
    cfg: Dict[str, Any],
) -> Dict[str, str]:
    """
    Expected keys:
      cfg["models"]["gemini_api"]["model_name"]

    Returns:
      {"question": <original query_text>, "answer": <concise grounded answer or "">}
    """
    model_name = str(_get_required(cfg, "models.gemini_api.model_name"))
    return generate_grounded_qa_with_gemini(
        query_text=query_text,
        chunk_text=chunk_text,
        model_name=model_name,
    )
