import types
from dataclasses import dataclass
from typing import Any, Dict, List

import pytest


# -------------------------
# Minimal Document stand-in (avoid requiring langchain in tests)
# -------------------------
@dataclass
class FakeDocument:
    page_content: str
    metadata: Dict[str, Any]


# -------------------------
# Fake Gemini client
# -------------------------
class FakeGeminiModels:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []
        self.next_text = "stubbed answer"

    def generate_content(self, *, model: str, contents: str, config: Dict[str, Any]):
        # record call args
        self.calls.append(
            {
                "model": model,
                "contents": contents,
                "config": config,
            }
        )
        # emulate response object with .text
        return types.SimpleNamespace(text=self.next_text)


class FakeGeminiClient:
    def __init__(self):
        self.models = FakeGeminiModels()


# -------------------------
# Helpers
# -------------------------
def _mk_settings(
    *,
    model_name: str = "gemini-2.0-flash",
    max_context_token: int = 20000,
    max_docs: int = 2,
) -> Dict[str, Any]:
    return {
        "models": {"gemini_api": {"model_name": model_name}},
        "generation": {
            "max_context_token": max_context_token,
            "max_docs": max_docs,
        },
    }


@pytest.fixture
def patch_genai_client(monkeypatch):
    """
    Patch rag_answer_node.genai.Client() -> FakeGeminiClient
    Returns the fake client instance so tests can inspect calls.
    """
    # NOTE: change this import path to your actual module path
    import rag_service.lcel.rag_answer_node  as mod

    fake = FakeGeminiClient()
    monkeypatch.setattr(mod.genai, "Client", lambda: fake)
    return fake


# -------------------------
# Tests
# -------------------------
def test_empty_query_returns_empty_answer_and_no_call(patch_genai_client):
    import rag_service.lcel.rag_answer_node  as mod

    runnable = mod.create_rag_answer_runnable(_mk_settings())
    out = runnable.invoke({"query": "   ", "docs": []})

    assert out == {"answer": "", "mode": "rag"}
    assert patch_genai_client.models.calls == []


def test_no_docs_uses_query_as_contents_and_sets_no_docs_system_instruction(patch_genai_client):
    import rag_service.lcel.rag_answer_node  as mod

    runnable = mod.create_rag_answer_runnable(_mk_settings(model_name="gemini-x"))
    out = runnable.invoke({"query": "What is CMU?", "docs": []})

    assert out["mode"] == "rag"
    assert out["answer"] == "stubbed answer"

    calls = patch_genai_client.models.calls
    assert len(calls) == 1
    call = calls[0]
    assert call["model"] == "gemini-x"
    assert call["contents"] == "What is CMU?"
    assert "system_instruction" in call["config"]
    assert "No documents were retrieved" in call["config"]["system_instruction"]


def test_with_docs_builds_prompt_contains_context_and_question(patch_genai_client):
    import rag_service.lcel.rag_answer_node  as mod

    docs = [
        FakeDocument(page_content="Carnegie Mellon University is in Pittsburgh.", metadata={"chunk_id": "c1"}),
        FakeDocument(page_content="It is a private research university.", metadata={"chunk_id": "c2"}),
    ]

    runnable = mod.create_rag_answer_runnable(_mk_settings(max_docs=2))
    out = runnable.invoke({"query": "Where is CMU?", "docs": docs})

    assert out["mode"] == "rag"
    assert out["answer"] == "stubbed answer"

    call = patch_genai_client.models.calls[0]
    contents = call["contents"]

    # prompt structure sanity
    assert "Instructions:" in contents
    assert "Context:" in contents
    assert "Question:" in contents
    assert "Answer:" in contents

    # context includes doc text
    assert "Carnegie Mellon University is in Pittsburgh." in contents
    assert "It is a private research university." in contents

    # question included
    assert "Where is CMU?" in contents

    # docs branch system instruction
    assert "using the provided context" in call["config"]["system_instruction"]


def test_with_docs_does_not_leak_chunk_id_into_prompt(patch_genai_client):
    import rag_service.lcel.rag_answer_node  as mod

    docs = [
        FakeDocument(
            page_content="Some content here.",
            metadata={"chunk_id": "SHOULD_NOT_APPEAR", "rrf_score": 0.9},
        ),
    ]
    runnable = mod.create_rag_answer_runnable(_mk_settings(max_docs=1))
    runnable.invoke({"query": "Q", "docs": docs})

    contents = patch_genai_client.models.calls[0]["contents"]
    assert "SHOULD_NOT_APPEAR" not in contents
    assert "chunk_id" not in contents  # coarse guard


def test_max_docs_limits_number_of_documents_in_context(patch_genai_client):
    import rag_service.lcel.rag_answer_node  as mod

    docs = [
        FakeDocument(page_content="Doc1", metadata={}),
        FakeDocument(page_content="Doc2", metadata={}),
        FakeDocument(page_content="Doc3", metadata={}),
    ]
    runnable = mod.create_rag_answer_runnable(_mk_settings(max_docs=2))
    runnable.invoke({"query": "Q", "docs": docs})

    contents = patch_genai_client.models.calls[0]["contents"]
    assert "Doc1" in contents
    assert "Doc2" in contents
    assert "Doc3" not in contents  # should be excluded


def test_context_is_truncated_when_max_context_token_small(patch_genai_client):
    import rag_service.lcel.rag_answer_node as mod

    long_text = "A" * 5000
    docs = [FakeDocument(page_content=long_text, metadata={})]

    # Small token budget => small char budget => truncation expected
    runnable = mod.create_rag_answer_runnable(_mk_settings(max_context_token=50, max_docs=1))
    runnable.invoke({"query": "Q", "docs": docs})

    contents = patch_genai_client.models.calls[0]["contents"]
    # Should contain ellipsis char added by _approx_truncate_text
    assert "…" in contents


def test_missing_required_config_key_raises_clear_error():
    import rag_service.lcel.rag_answer_node as mod

    settings = {"models": {"gemini_api": {}}}  # missing model_name
    with pytest.raises(KeyError) as e:
        mod.create_rag_answer_runnable(settings)

    assert "Missing required config key: models.gemini_api.model_name" in str(e.value)
