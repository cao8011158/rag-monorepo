import sys
import types
import importlib
import pytest
import torch
from langchain_core.documents import Document


# -------------------------
# Stubs (no transformers)
# -------------------------

class _CallCounter:
    def __init__(self):
        self.tokenizer_calls = 0
        self.model_calls = 0
        self.seen_batch_sizes = []  # record actual batch sizes


class _FakeTokenizer:
    """
    A tokenizer stub that converts (query, doc) pairs into a tensor feature:
      overlap_count = |tokens(query) ∩ tokens(doc)|
    We encode that as input_ids[:, 0] = overlap_count
    """
    def __init__(self, counter: _CallCounter):
        self.counter = counter

    def __call__(
        self,
        queries,
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ):
        assert isinstance(queries, list)
        assert isinstance(texts, list)
        assert len(queries) == len(texts)

        self.counter.tokenizer_calls += 1
        self.counter.seen_batch_sizes.append(len(queries))

        feats = []
        for q, t in zip(queries, texts):
            q_tokens = set((q or "").lower().split())
            t_tokens = set((t or "").lower().split())
            overlap = len(q_tokens & t_tokens)
            feats.append(overlap)

        # Shape: (batch, 1)
        input_ids = torch.tensor(feats, dtype=torch.long).unsqueeze(-1)
        attention_mask = torch.ones_like(input_ids)

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FakeModel(torch.nn.Module):
    """
    A model stub: returns logits equal to overlap_count as float.
    """
    def __init__(self, counter: _CallCounter):
        super().__init__()
        self.counter = counter

    def eval(self):
        return self

    def to(self, device):
        return self

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        self.counter.model_calls += 1
        # input_ids shape: (batch, 1); use it as score
        logits = input_ids.squeeze(-1).to(torch.float32)
        return types.SimpleNamespace(logits=logits)


def _install_transformers_stub(counter: _CallCounter):
    """
    Install a minimal 'transformers' module stub into sys.modules
    BEFORE importing rag_service.nodes.reranker_node.
    """
    fake_transformers = types.ModuleType("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_name, cache_dir=None, use_fast=True):
            return _FakeTokenizer(counter)

    class AutoModelForSequenceClassification:
        @staticmethod
        def from_pretrained(model_name, cache_dir=None):
            return _FakeModel(counter)

    fake_transformers.AutoTokenizer = AutoTokenizer
    fake_transformers.AutoModelForSequenceClassification = AutoModelForSequenceClassification

    sys.modules["transformers"] = fake_transformers


def _import_reranker_node_with_stubs(counter: _CallCounter):
    """
    Import (or reload) reranker_node after transformers stub is installed.
    """
    _install_transformers_stub(counter)

    # IMPORTANT: if the module was already imported in this pytest session,
    # reload it so it picks up our stubbed transformers.
    mod_name = "rag_service.lcel.reranker_node"
    if mod_name in sys.modules:
        return importlib.reload(sys.modules[mod_name])
    return importlib.import_module(mod_name)


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def counter():
    return _CallCounter()


@pytest.fixture
def reranker_module(counter):
    return _import_reranker_node_with_stubs(counter)


@pytest.fixture
def settings():
    return {
        "reranker": {
            "model_name": "stubbed/reranker",
            "device": "cpu",
            "cache_dir": None,
            "batch_size": 2,
            "max_length": 128,
        }
    }


# -------------------------
# A — API behavior
# -------------------------

def test_create_reranker_runnable_returns_runnable(reranker_module, settings):
    rr = reranker_module.create_reranker_runnable(settings)
    # RunnableLambda has invoke()
    assert hasattr(rr, "invoke")


def test_rerank_requires_query_and_docs_keys(reranker_module, settings):
    rr = reranker_module.create_reranker_runnable(settings)

    with pytest.raises(KeyError):
        rr.invoke({"docs": []})

    with pytest.raises(KeyError):
        rr.invoke({"query": "x"})


def test_empty_docs_returns_empty_list(reranker_module, settings):
    rr = reranker_module.create_reranker_runnable(settings)
    out = rr.invoke({"query": "cmu", "docs": []})
    assert out == []


# -------------------------
# B — Sorting correctness
# -------------------------

def test_sorted_by_score_desc(reranker_module, settings):
    rr = reranker_module.create_reranker_runnable(settings)

    docs = [
        Document(page_content="banana yellow fruit", metadata={"chunk_id": "a"}),
        Document(page_content="cmu pittsburgh university", metadata={"chunk_id": "b"}),
        Document(page_content="cmu university in pittsburgh", metadata={"chunk_id": "c"}),
    ]
    query = "cmu university pittsburgh"

    out = rr.invoke({"query": query, "docs": docs})
    scores = [d.metadata["rerank_score"] for d in out]

    assert scores == sorted(scores, reverse=True)
    # strongest overlap should be first
    assert out[0].metadata["chunk_id"] in {"b", "c"}


# -------------------------
# C — Metadata mutation behavior
# -------------------------

def test_metadata_preserved_and_rerank_score_added(reranker_module, settings):
    rr = reranker_module.create_reranker_runnable(settings)

    meta0 = {"chunk_id": "x", "rrf_score": 0.72}
    old_meta_ref = meta0  # keep reference to ensure it isn't mutated in-place

    d0 = Document(page_content="cmu pittsburgh", metadata=meta0)
    d1 = Document(page_content="banana")

    out = rr.invoke({"query": "cmu", "docs": [d0, d1]})

    # rerank_score exists
    assert "rerank_score" in out[0].metadata
    assert "rerank_score" in out[1].metadata

    # original keys preserved
    # (d0 metadata should still contain chunk_id and rrf_score)
    assert d0.metadata["chunk_id"] == "x"
    assert d0.metadata["rrf_score"] == 0.72

    # IMPORTANT: the original dict reference should NOT be mutated in-place
    # code does: d.metadata = dict(d.metadata or {})
    assert old_meta_ref is not d0.metadata
    assert "rerank_score" not in old_meta_ref  # old dict remains clean


# -------------------------
# D — Batching correctness
# -------------------------

def test_batching_calls(counter, reranker_module, settings):
    # batch_size = 2 from fixture
    rr = reranker_module.create_reranker_runnable(settings)

    docs = [Document(page_content=f"cmu {i}") for i in range(5)]
    rr.invoke({"query": "cmu", "docs": docs})

    # 5 docs with batch_size=2 => 3 batches: 2,2,1
    assert counter.tokenizer_calls == 3
    assert counter.model_calls == 3
    assert counter.seen_batch_sizes == [2, 2, 1]
