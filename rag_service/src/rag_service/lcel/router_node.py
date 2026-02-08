# rag_service/nodes/router_node.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, List

from pydantic import BaseModel, Field, ValidationError

from langchain_core.runnables import RunnableLambda

# Google Gemini (new SDK)
from google import genai


# =========================
# Structured output schema
# =========================

class RouteOutput(BaseModel):
    """
    Router decision for a single query.
    """
    use_rag: bool = Field(description="Whether to route this query to RAG.")
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in [0,1]. If uncertain, keep low.",
    )
    reason: str = Field(default="", description="Short reason for the decision.")


# =========================
# Utils (keep your style)
# =========================

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


def _get_optional(cfg: Dict[str, Any], path: str, default: Any) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _norm(s: str) -> str:
    return (s or "").strip()


def _contains_any(hay: str, needles: List[str]) -> bool:
    h = hay.lower()
    for n in needles:
        if n and n.lower() in h:
            return True
    return False


# =========================
# Prompt builder
# =========================

_DEFAULT_TOPIC_DEF = (
    "IN: primarily about Pittsburgh or Carnegie Mellon University (CMU), including their history, "
    "geography, culture, population, economy, campus life, traditions, trivia, and current events.\n"
    "OUT: everything else.\n"
    "If uncertain, choose OUT (use_rag=false)."
)

_DEFAULT_EXAMPLES = (
    "Examples:\n"
    'Query: "what year was carnegie mellon university founded"\n'
    'Output: {"use_rag": true, "confidence": 0.95, "reason": "CMU history"}\n\n'
    'Query: "major sports teams in pittsburgh"\n'
    'Output: {"use_rag": true, "confidence": 0.85, "reason": "Pittsburgh local topic"}\n\n'
    'Query: "best universities for computer science in the us"\n'
    'Output: {"use_rag": false, "confidence": 0.75, "reason": "General topic, not CMU/Pittsburgh-specific"}\n'
)

def _build_router_prompt(
    *,
    topic_definition: str,
    examples: str,
    query: str,
) -> str:
    # Keep it simple and strict: single query → single JSON object.
    return "\n".join([
        "Task:",
        "Decides whether a query should be answered using a CMU/Pittsburgh RAG system.",
        "",
        "Rules:",
        _norm(topic_definition),
        "",
        _norm(examples),
        "",
        "Now decide for the following query.",
        f'Query: "{_norm(query).replace(chr(10), " ")}"',
    ])


# =========================
# Core Gemini call
# =========================

@dataclass(frozen=True)
class RouterConfig:
    model_name: str
    max_retries: int = 3
    backoff_sec: float = 1.5

    # behavior knobs
    threshold: float = 0.5  # if confidence >= threshold and use_rag true, route to rag
    force_out_on_low_conf: bool = True

    # prompt knobs
    topic_definition: str = _DEFAULT_TOPIC_DEF
    examples: str = _DEFAULT_EXAMPLES

    # fast path (optional)
    keyword_override_enabled: bool = True
    keyword_hits: tuple[str, ...] = (
        "cmu",
        "carnegie mellon",
        "carnegie-mellon",
        "pittsburgh",
        "allegheny county",
        "412",
        "tepper",
        "scs",
        "cfa",
        "dietrich",
        "mellon college",
    )


def _call_gemini_router_once(
    *,
    client: genai.Client,
    cfg: RouterConfig,
    query: str,
) -> RouteOutput:
    prompt = _build_router_prompt(
        topic_definition=cfg.topic_definition,
        examples=cfg.examples,
        query=query,
    )

    response = client.models.generate_content(
        model=cfg.model_name,
        contents=prompt,
        config={
            "system_instruction": "You are a strict router.",
            "response_mime_type": "application/json",
            "response_json_schema": RouteOutput.model_json_schema(),
        },
    )

    # Gemini returns structured JSON as text; validate strictly.
    parsed = RouteOutput.model_validate_json(response.text)

    # Normalize/guardrails
    conf = float(parsed.confidence if parsed.confidence is not None else 0.5)
    conf = max(0.0, min(1.0, conf))

    use_rag = bool(parsed.use_rag)

    # Optional policy: low-confidence → OUT (safer for routing)
    if cfg.force_out_on_low_conf and conf < cfg.threshold:
        use_rag = False
        reason = parsed.reason or "Low confidence; defaulting to direct LLM."
    else:
        reason = parsed.reason or ("RAG topic match" if use_rag else "Out of domain")

    return RouteOutput(use_rag=use_rag, confidence=conf, reason=reason)


def route_query_with_gemini(
    *,
    query: str,
    client: genai.Client,
    cfg: RouterConfig,
) -> Dict[str, Any]:
    """
    Public callable: single query → {"use_rag": bool, "confidence": float, "reason": str}

    Includes:
    - optional keyword fast-path
    - retry with exponential-ish backoff
    - strict schema validation
    """
    q = _norm(query)
    if not q:
        return {"use_rag": False, "confidence": 0.0, "reason": "Empty query."}

    # Fast keyword override (optional)
    if cfg.keyword_override_enabled and _contains_any(q, list(cfg.keyword_hits)):
        return {"use_rag": True, "confidence": 0.99, "reason": "Keyword match for CMU/Pittsburgh."}

    last_err: Optional[Exception] = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            out = _call_gemini_router_once(client=client, cfg=cfg, query=q)
            return {"use_rag": out.use_rag, "confidence": out.confidence, "reason": out.reason}
        except (ValidationError, ValueError) as e:
            # Schema/format errors usually indicate model didn't comply; you can retry once,
            # but by default treat as hard fail to surface quickly.
            raise RuntimeError(f"Invalid structured output format: {e}") from e
        except Exception as e:
            last_err = e
            if attempt < cfg.max_retries:
                time.sleep(cfg.backoff_sec * attempt)
                continue
            break

    raise RuntimeError(f"Gemini routing failed after {cfg.max_retries} retries: {last_err}") from last_err


# =========================
# Node factory (LCEL adapter)
# =========================

def create_router_runnable(settings: Dict[str, Any]) -> RunnableLambda:
    """
    Build a router Runnable that takes a query (str) and returns:
      {"use_rag": bool, "confidence": float, "reason": str}

    Expected settings keys (minimal):
      settings["models"]["gemini_api"]["model_name"]

    Optional (recommended) knobs (you can add to your config later):
      settings["router"]["max_retries"]
      settings["router"]["backoff_sec"]
      settings["router"]["threshold"]
      settings["router"]["force_out_on_low_conf"]
      settings["router"]["topic_definition"]
      settings["router"]["examples"]
      settings["router"]["keyword_override_enabled"]
      settings["router"]["keyword_hits"]
    """
    model_name = str(_get_required(settings, "models.gemini_api.model_name"))

    cfg = RouterConfig(
        model_name=model_name,
        max_retries=int(_get_optional(settings, "router.max_retries", 3)),
        backoff_sec=float(_get_optional(settings, "router.backoff_sec", 1.5)),
        threshold=float(_get_optional(settings, "router.threshold", 0.5)),
        force_out_on_low_conf=bool(_get_optional(settings, "router.force_out_on_low_conf", False)),
        topic_definition=str(_get_optional(settings, "router.topic_definition", _DEFAULT_TOPIC_DEF)),
        examples=str(_get_optional(settings, "router.examples", _DEFAULT_EXAMPLES)),
        keyword_override_enabled=bool(_get_optional(settings, "router.keyword_override_enabled", True)),
        keyword_hits=tuple(_get_optional(settings, "router.keyword_hits", list(RouterConfig.keyword_hits))),
    )

    client = genai.Client()

    return RunnableLambda(lambda q: route_query_with_gemini(query=str(q), client=client, cfg=cfg))
