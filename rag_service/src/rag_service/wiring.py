# rag_service/wiring.py
from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from rag_service.lcel.router_node import create_router_runnable
from rag_service.lcel.retriever_node import create_retriever_runnable
from rag_service.lcel.reranker_node import create_reranker_runnable
from rag_service.lcel.direct_answer_node import create_direct_answer_runnable
from rag_service.lcel.rag_answer_node import create_rag_answer_runnable


def build_app_chain(settings: Dict[str, Any]) -> RunnableLambda:
    """
    Build the main app chain:
      input : query (str)
      output: {"answer": str, "mode": "direct"|"rag", ...optional debug...}

    Flow:
      decision = router(query)
      if not decision["use_rag"]:
          return direct(query)
      docs = retriever(query) -> List[Document]
      docs = reranker({"query": query, "docs": docs}) -> List[Document]
      return rag({"query": query, "docs": docs})
    """
    router = create_router_runnable(settings)
    retriever = create_retriever_runnable(settings)
    reranker = create_reranker_runnable(settings)
    direct = create_direct_answer_runnable(settings)
    rag = create_rag_answer_runnable(settings)

    def _run(query: str) -> Dict[str, Any]:
        q = (query or "").strip()
        if not q:
            return {"answer": "", "mode": "direct"}

        decision: Dict[str, Any] = router.invoke(q)
        use_rag = bool(decision.get("use_rag", False))

        if not use_rag:
            out = direct.invoke(q)  # {"answer": ..., "mode": "direct"}
            # 可选：把 router 信息带回去，方便 debug/观测
            out["router"] = decision
            return out

        # ---- RAG path ----
        docs: List[Document] = retriever.invoke(q)

        # reranker 输入是 dict；输出仍是 List[Document]
        docs = reranker.invoke({"query": q, "docs": docs})

        out = rag.invoke({"query": q, "docs": docs})  # {"answer": ..., "mode": "rag"}
        out["router"] = decision
        return out

    return RunnableLambda(_run)
