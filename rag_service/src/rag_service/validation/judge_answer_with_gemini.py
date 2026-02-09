from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field, ValidationError
from google import genai



class JudgeResult(TypedDict, total=False):
    is_correct: bool
    score: float                 # 0.0 - 1.0
    reason: str                  # short justification
    missing_points: List[str]    # what gold has but output missed
    extra_claims: List[str]      # claims not supported by gold (or by chunk if provided)
    is_grounded: Optional[bool]  # only if check_grounding=True and chunk_text provided


# =========================
# Structured output schema
# =========================

class JudgeOutput(BaseModel):
    is_correct: bool = Field(description="Whether MODEL_OUTPUT matches GOLD_ANSWER for the QUESTION.")
    score: float = Field(description="0.0-1.0 correctness score. 1.0 fully correct; 0.5 partially; 0.0 incorrect.")
    reason: str = Field(description="Short justification (1-3 sentences).")
    missing_points: List[str] = Field(description="Key facts in GOLD_ANSWER that MODEL_OUTPUT missed.")
    extra_claims: List[str] = Field(description="Claims in MODEL_OUTPUT that contradict GOLD_ANSWER or add unsupported specifics.")
    # 仅在 check_grounding=True 且提供 chunk_text 时才有意义；否则我们会清空/移除
    is_grounded: Optional[bool] = Field(
        default=None,
        description="If grounding check enabled: true only if every factual claim is supported by EVIDENCE; false if any unsupported; null if no factual claims."
    )


def _clamp_float(x: Any, lo: float, hi: float, default: float) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _finalize_judge_output(
    parsed: JudgeOutput,
    *,
    want_grounding: bool,
) -> JudgeOutput:
    """
    Extra strictness:
    - clamp score to [0,1]
    - if grounding not wanted, force is_grounded=None
    """
    parsed.score = _clamp_float(parsed.score, 0.0, 1.0, 0.0)
    if not want_grounding:
        parsed.is_grounded = None
    return parsed


def judge_answer_with_gemini(
    *,
    question: str,
    gold_answer: str,
    model_output: str,
    model_name: str,
    chunk_text: Optional[str] = None,
    check_grounding: bool = False,
    max_retries: int = 6,
    backoff_sec: float = 1.5,
) -> JudgeResult:
    """
    Gemini judge (structured output):
    - correctness: compare model_output vs gold_answer under question
    - optionally grounding: whether model_output is supported by chunk_text

    Returns strict JSON parsed via Pydantic schema.
    """
    q = (question or "").strip()
    gold = (gold_answer or "").strip()
    out = (model_output or "").strip()
    ctx = (chunk_text or "").strip()

    # Minimal guard: if no gold, can't judge
    if not q or not gold:
        return {
            "is_correct": False,
            "score": 0.0,
            "reason": "Missing question or gold_answer.",
            "missing_points": [],
            "extra_claims": [],
        }

    want_grounding = bool(check_grounding and ctx)

    grounding_line = ""
    grounding_rule = ""
    if want_grounding:
        grounding_line = "\nEVIDENCE (chunk_text):\n" + ctx
        grounding_rule = (
            "\nAdditionally, judge whether the model_output is supported by EVIDENCE. "
            "Set is_grounded=true only if every factual claim in model_output is supported by EVIDENCE; "
            "false if any unsupported factual claim exists; null if model_output has no factual claims."
        )

    prompt = f"""You are a strict evaluator.

TASK:
Given a QUESTION, a GOLD_ANSWER (the reference), and a MODEL_OUTPUT, decide if MODEL_OUTPUT is correct.

SCORING:
- is_correct: true if MODEL_OUTPUT matches GOLD_ANSWER in meaning for the QUESTION (paraphrases ok).
- score: 1.0 for fully correct; 0.5 for partially correct; 0.0 for incorrect.
- missing_points: key facts present in GOLD_ANSWER that MODEL_OUTPUT missed.
- extra_claims: claims in MODEL_OUTPUT that contradict GOLD_ANSWER or add unsupported specifics beyond GOLD_ANSWER.
- reason: short justification (1-3 sentences). Do not be verbose.{grounding_rule}

IMPORTANT:
- Judge factual correctness, not writing quality.
- If MODEL_OUTPUT says "I don't know"/refuses while GOLD_ANSWER is answerable, that is incorrect (score 0).
- If GOLD_ANSWER is short, do NOT penalize MODEL_OUTPUT for extra correct detail unless it introduces unsupported specifics or contradictions.

QUESTION:
{q}

GOLD_ANSWER:
{gold}

MODEL_OUTPUT:
{out}
{grounding_line}
"""

    client = genai.Client()

    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "system_instruction": "You are a strict evaluator.",
                    "response_mime_type": "application/json",
                    "response_json_schema": JudgeOutput.model_json_schema(),
                },
            )

            # ✅ structured output: 按 schema 解析 JSON
            parsed = JudgeOutput.model_validate_json(resp.text)

            # ✅ 额外严格化（score clamp / grounding 字段控制）
            parsed = _finalize_judge_output(parsed, want_grounding=want_grounding)

            data = parsed.model_dump()

            # 不需要 grounding 时，输出里就不要带 is_grounded（保持你原逻辑“干净”）
            if not want_grounding:
                data.pop("is_grounded", None)

            return data  # type: ignore[return-value]

        except (ValidationError, ValueError) as e:
            # 不可重试：schema/格式不符合（structured output 没按要求来）
            raise RuntimeError(f"Invalid structured output format: {e}") from e

        except Exception as e:
            # 可重试：网络/限流/临时错误
            last_err = e
            print(f"[WARN] failed (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(backoff_sec * attempt)
                continue
            break

    # 全部重试失败：给出保守失败
    raise RuntimeError(f"Gemini judge failed after {max_retries} retries: {last_err}")


def run_answer_judge(
    *,
    qa_pair: Dict[str, Any],
    llm_output: str,
    cfg: Dict[str, Any],
    check_grounding: bool = False,
) -> JudgeResult:
    """
    Expected cfg keys:
      cfg["models"]["gemini_api"]["model_name"]

    qa_pair expected keys:
      - question
      - answer (gold)
      - optional: chunk_text (if you want grounding check)

    Returns JudgeResult.
    """
    model_name = str(_get_required(cfg, "models.gemini_api.model_name"))

    question = str(qa_pair.get("question", "") or "")
    gold_answer = str(qa_pair.get("answer", "") or "")

    chunk_text = None
    if check_grounding:
        chunk_text = str(qa_pair.get("chunk_text", "") or "")

    return judge_answer_with_gemini(
        question=question,
        gold_answer=gold_answer,
        model_output=str(llm_output or ""),
        model_name=model_name,
        chunk_text=chunk_text,
        check_grounding=check_grounding,
    )
