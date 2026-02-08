# rag_service/lcel/direct_answer_node.py
from __future__ import annotations
from typing import Any, Dict

from langchain_core.runnables import RunnableLambda
from google import genai

def _get_required(cfg: Dict[str, Any], path: str) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing required config key: {path}")
        cur = cur[part]
    return cur

def create_direct_answer_runnable(settings: Dict[str, Any]) -> RunnableLambda:
    model_name = str(_get_required(settings, "models.gemini_api.model_name"))
    client = genai.Client()

    def _answer(query: str) -> Dict[str, Any]:
        q = (query or "").strip()
        if not q:
            return {"answer": "", "mode": "direct"}

        resp = client.models.generate_content(
            model=model_name,
            contents=q,
            config={
                "system_instruction": (
                    "You are a helpful assistant. "
                    "Answer directly using general knowledge. "
                ),
            },
        )
        return {"answer": (resp.text or "").strip(), "mode": "direct"}

    return RunnableLambda(_answer)
