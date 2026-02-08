from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from google import genai


# -------------------------
# Utils
# -------------------------

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


def _approx_truncate_text(text: str, max_chars: int) -> str:
    """
    A lightweight guardrail: approximate token limit using char budget.
    (Rough heuristic: 1 token ~= 3~4 chars in English; Chinese differs.
     We'll just keep a conservative char budget derived from max_context_token.)
    """
    if not text:
        return ""
    t = text.strip()
    return t if len(t) <= max_chars else (t[:max_chars].rstrip() + "…")


def _build_context_block(
    docs: List[Document],
    *,
    max_docs: int,
    max_context_token: int,
) -> str:
    """
    Build context string from top docs.
    - User requested: NO chunk_id in prompt.
    - Apply a coarse length budget to reduce over-long prompts.
    """
    # Very rough: assume 1 token ~= 4 chars. Give 80% budget to context.
    max_context_chars = int(max_context_token * 4 * 0.8)

    parts: List[str] = []
    used_chars = 0

    for d in docs[:max_docs]:
        body = (d.page_content or "").strip()

        # Plain header WITHOUT any chunk_id
        header = "Document:\n"

        remain = max_context_chars - used_chars
        if remain <= 0:
            break

        piece = header + _approx_truncate_text(body, max_chars=max(0, remain - len(header))) + "\n"
        parts.append(piece)
        used_chars += len(piece)

    return "\n".join(parts).strip()


# -------------------------
# Node factory
# -------------------------

def create_rag_answer_runnable(settings: Dict[str, Any]) -> RunnableLambda:
    """
    RAG answer generator:
      input : {"query": str, "docs": List[Document]}
      output: {"answer": str, "mode": "rag"}

    Config used:
      settings["models"]["gemini_api"]["model_name"]
      settings["generation"]["max_context_token"] (default 20000)
      settings["generation"]["max_docs"]          (default 2)
    """
    model_name = str(_get_required(settings, "models.gemini_api.model_name"))

    max_context_token = int(_get_optional(settings, "generation.max_context_token", 20000))
    max_docs = int(_get_optional(settings, "generation.max_docs", 2))

    client = genai.Client()

    def _answer(inp: Dict[str, Any]) -> Dict[str, Any]:
        query = str(inp.get("query", "") or "").strip()
        docs: List[Document] = inp.get("docs") or []

        if not query:
            return {"answer": "", "mode": "rag"}

        # Build prompt
        if docs:
            context = _build_context_block(
                docs,
                max_docs=max_docs,
                max_context_token=max_context_token,
            )
            prompt = "\n".join([
                "Instructions:",
                "- Use the context as the primary source of truth.",
                "- If the context does not contain enough information, say you don't know or explain the limitation.",
                "- Do NOT invent facts that are not supported by the context.",
                "- Keep the answer concise and directly address the question.",
                "",
                "Context:",
                context,
                "",
                "Question:",
                query,
                "",
                "Answer:",
            ])
            system_instruction = "You are a helpful assistant answering the user's question using the provided context."
            contents = prompt
        else:
            # Router chose RAG but retrieval got nothing (or filtering removed all)
            system_instruction = (
                "You are a helpful assistant. "
                "No documents were retrieved. Answer using general knowledge, "
                "and be explicit if you are unsure."
            )
            contents = query

        resp = client.models.generate_content(
            model=model_name,
            contents=contents,
            config={"system_instruction": system_instruction},
        )

        return {"answer": (resp.text or "").strip(), "mode": "rag"}

    return RunnableLambda(_answer)
