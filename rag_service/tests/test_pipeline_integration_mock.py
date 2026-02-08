# tests/test_pipeline_integration_mock.py
from __future__ import annotations

from typing import Any, Dict, List

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from rag_service.settings import load_settings
import rag_service.wiring as wiring


class CallLog:
    """Simple call recorder for verifying the pipeline path."""
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def add(self, name: str, payload: Any) -> None:
        self.calls.append({"name": name, "payload": payload})

    def names(self) -> List[str]:
        return [c["name"] for c in self.calls]

    def count(self, name: str) -> int:
        return sum(1 for c in self.calls if c["name"] == name)


def _doc(text: str, chunk_id: str, rrf_score: float) -> Document:
    return Document(
        page_content=text,
        metadata={"chunk_id": chunk_id, "rrf_score": rrf_score},
    )


@pytest.mark.integration
def test_pipeline_routes_to_direct_when_router_says_no(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = load_settings("configs/rag.yaml")
    log = CallLog()

    # -------------------------
    # Mock nodes
    # -------------------------
    def mock_create_router_runnable(_settings: Dict[str, Any]) -> RunnableLambda:
        def _router(q: str) -> Dict[str, Any]:
            log.add("router", q)
            return {"use_rag": False, "confidence": 0.9, "reason": "Out-of-domain (mock)"}
        return RunnableLambda(_router)

    def mock_create_retriever_runnable(_settings: Dict[str, Any]) -> RunnableLambda:
        def _retr(q: str) -> List[Document]:
            log.add("retriever", q)
            return [_doc("dummy", "c1", 0.1)]
        return RunnableLambda(_retr)

    def mock_create_reranker_runnable(_settings: Dict[str, Any]) -> RunnableLambda:
        def _rer(inp: Dict[str, Any]) -> List[Document]:
            log.add("reranker", inp)
            return inp["docs"]
        return RunnableLambda(_rer)

    def mock_create_direct_answer_runnable(_settings: Dict[str, Any]) -> RunnableLambda:
        def _direct(q: str) -> Dict[str, Any]:
            log.add("direct", q)
            return {"answer": f"[DIRECT MOCK] {q}", "mode": "direct"}
        return RunnableLambda(_direct)

    def mock_create_rag_answer_runnable(_settings: Dict[str, Any]) -> RunnableLambda:
        def _rag(inp: Dict[str, Any]) -> Dict[str, Any]:
            log.add("rag_answer", inp)
            return {"answer": "[RAG MOCK]", "mode": "rag"}
        return RunnableLambda(_rag)

    # Patch the factory functions that wiring.build_app_chain uses
    monkeypatch.setattr(wiring, "create_router_runnable", mock_create_router_runnable)
    monkeypatch.setattr(wiring, "create_retriever_runnable", mock_create_retriever_runnable)
    monkeypatch.setattr(wiring, "create_reranker_runnable", mock_create_reranker_runnable)
    monkeypatch.setattr(wiring, "create_direct_answer_runnable", mock_create_direct_answer_runnable)
    monkeypatch.setattr(wiring, "create_rag_answer_runnable", mock_create_rag_answer_runnable)

    chain = wiring.build_app_chain(settings)

    out = chain.invoke("Explain quantum entanglement")

    # -------------------------
    # Assertions
    # -------------------------
    assert out["mode"] == "direct"
    assert out["answer"].startswith("[DIRECT MOCK]")
    assert "router" in out and out["router"]["use_rag"] is False

    # Only router + direct should have been called
    assert log.names() == ["router", "direct"]
    assert log.count("retriever") == 0
    assert log.count("reranker") == 0
    assert log.count("rag_answer") == 0


@pytest.mark.integration
def test_pipeline_routes_to_rag_and_calls_retrieval_and_rerank(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = load_settings("configs/rag.yaml")
    log = CallLog()

    # -------------------------
    # Mock nodes
    # -------------------------
    def mock_create_router_runnable(_settings: Dict[str, Any]) -> RunnableLambda:
        def _router(q: str) -> Dict[str, Any]:
            log.add("router", q)
            return {"use_rag": True, "confidence": 0.95, "reason": "In-domain (mock)"}
        return RunnableLambda(_router)

    def mock_create_retriever_runnable(_settings: Dict[str, Any]) -> RunnableLambda:
        def _retr(q: str) -> List[Document]:
            log.add("retriever", q)
            # Two docs with different rrf_score
            return [
                _doc("CMU is ...", "c1", 0.40),
                _doc("Pittsburgh is ...", "c2", 0.90),
            ]
        return RunnableLambda(_retr)

    def mock_create_reranker_runnable(_settings: Dict[str, Any]) -> RunnableLambda:
        def _rer(inp: Dict[str, Any]) -> List[Document]:
            log.add("reranker", inp)

            assert isinstance(inp, dict)
            assert "query" in inp and isinstance(inp["query"], str)
            assert "docs" in inp and isinstance(inp["docs"], list)

            # Add rerank_score and sort descending by rerank_score
            docs = inp["docs"]
            # pretend reranker prefers c1 over c2
            scores = {"c1": 9.9, "c2": 1.2}

            out_docs: List[Document] = []
            for d in docs:
                md = dict(d.metadata or {})
                md["rerank_score"] = float(scores.get(md.get("chunk_id", ""), 0.0))
                out_docs.append(Document(page_content=d.page_content, metadata=md))

            out_docs.sort(key=lambda d: float(d.metadata.get("rerank_score", 0.0)), reverse=True)
            return out_docs

        return RunnableLambda(_rer)

    def mock_create_direct_answer_runnable(_settings: Dict[str, Any]) -> RunnableLambda:
        def _direct(q: str) -> Dict[str, Any]:
            log.add("direct", q)
            return {"answer": f"[DIRECT MOCK] {q}", "mode": "direct"}
        return RunnableLambda(_direct)

    def mock_create_rag_answer_runnable(_settings: Dict[str, Any]) -> RunnableLambda:
        def _rag(inp: Dict[str, Any]) -> Dict[str, Any]:
            log.add("rag_answer", inp)

            assert "query" in inp and isinstance(inp["query"], str)
            assert "docs" in inp and isinstance(inp["docs"], list)
            docs: List[Document] = inp["docs"]
            assert len(docs) > 0

            # Ensure reranker_score exists on docs
            assert "rerank_score" in (docs[0].metadata or {})

            # Produce deterministic answer
            top_id = (docs[0].metadata or {}).get("chunk_id", "unknown")
            return {"answer": f"[RAG MOCK] top={top_id}", "mode": "rag"}

        return RunnableLambda(_rag)

    # Patch the factory functions that wiring.build_app_chain uses
    monkeypatch.setattr(wiring, "create_router_runnable", mock_create_router_runnable)
    monkeypatch.setattr(wiring, "create_retriever_runnable", mock_create_retriever_runnable)
    monkeypatch.setattr(wiring, "create_reranker_runnable", mock_create_reranker_runnable)
    monkeypatch.setattr(wiring, "create_direct_answer_runnable", mock_create_direct_answer_runnable)
    monkeypatch.setattr(wiring, "create_rag_answer_runnable", mock_create_rag_answer_runnable)

    chain = wiring.build_app_chain(settings)

    out = chain.invoke("When was Carnegie Mellon University founded?")

    # -------------------------
    # Assertions
    # -------------------------
    assert out["mode"] == "rag"
    assert out["answer"].startswith("[RAG MOCK]")
    assert "router" in out and out["router"]["use_rag"] is True

    # Ensure full rag path called in order
    assert log.names() == ["router", "retriever", "reranker", "rag_answer"]
    assert log.count("direct") == 0
