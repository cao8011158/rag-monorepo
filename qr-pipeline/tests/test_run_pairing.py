import json
import pytest
from typing import Any, Dict, List, Optional


# -----------------------
# In-memory Fake Store
# -----------------------
class FakeStore:
    def __init__(self):
        self.files: Dict[str, bytes] = {}

    def exists(self, path: str) -> bool:
        return path in self.files

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        return self.files[path].decode(encoding)

    def write_text(self, path: str, content: str, encoding: str = "utf-8") -> None:
        self.files[path] = content.encode(encoding)

    def read_bytes(self, path: str) -> bytes:
        return self.files[path]

    def write_bytes(self, path: str, content: bytes) -> None:
        self.files[path] = content

    def append_bytes(self, path: str, content: bytes) -> None:
        if path not in self.files:
            self.files[path] = content
        else:
            self.files[path] += content

    def list(self, prefix: str):
        for k in sorted(self.files.keys()):
            if k.startswith(prefix):
                yield k


# -----------------------
# JSONL helpers for tests
# -----------------------
def write_jsonl_raw(store: FakeStore, path: str, rows: List[Dict[str, Any]]) -> None:
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    store.write_text(path, data, encoding="utf-8")


def read_jsonl_raw(store: FakeStore, path: str) -> List[Dict[str, Any]]:
    if not store.exists(path):
        return []
    lines = store.read_text(path, encoding="utf-8").splitlines()
    out = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        out.append(json.loads(ln))
    return out


# -----------------------
# Fakes: Retriever / Pairing
# -----------------------
class FakeRetriever:
    def __init__(self, items: List[Dict[str, Any]]):
        self._items = items
        self.calls: List[str] = []

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        self.calls.append(query)
        return list(self._items)


def fake_build_pairs_for_query(
    *,
    query: Dict[str, Any],
    source_doc: Dict[str, Any],
    candidate_docs: List[Dict[str, Any]],
    llm: Any,
    embedder: Any,
    **kwargs: Any,
):
    # 生成一个最小 QueryPack：positives=source_doc, negatives=前2个候选（排除 source）
    src_id = source_doc["chunk_id"]
    negs = [c for c in candidate_docs if c.get("chunk_id") != src_id][:2]
    pack = {
        "query": query,
        "positives": [source_doc],
        "negatives": negs,
        "meta": {"stats": {"num_samples": 1, "num_candidates_in": len(candidate_docs)}},
    }
    stats = dict(pack["meta"]["stats"])
    return pack, stats


# -----------------------
# Minimal settings + store registry
# -----------------------
def minimal_settings() -> Dict[str, Any]:
    return {
        "stores": {"fs_local": {"kind": "filesystem", "root": "data"}},
        "inputs": {
            "ce_artifacts": {
                "chunks": {"store": "fs_local", "base": "ce_out/chunks", "chunks_file": "chunks.jsonl"}
            }
        },
        "outputs": {
            "store": "fs_local",
            "base": "rq_out",
            "files": {
                "queries_in_domain": "queries/in_domain.jsonl",
                "pairs": "pairs/pairs.pairwise.train.jsonl",
                "stats": "stats/run_pairing.stats.json",
                "errors": "logs/run_pairing.errors.jsonl",
            },
        },
        "embedding": {
            "model_name": "fake",
            "instructions": {"passage": "passage:", "query": "query:"},
            "batch_size": 8,
            "normalize_embeddings": True,
            "device": None,
        },
        "models": {"llm": {"model_name": "fake", "device": "cpu"}},
        "_meta": {"config_path": "configs/fake.yaml", "config_hash": "0" * 64},
    }


def fake_build_store_registry(settings: Dict[str, Any], store: FakeStore):
    # build_store_registry normally returns name->Store
    return {"fs_local": store}


# -----------------------
# Tests
# -----------------------
def seed_queries(store: FakeStore, path: str, n: int):
    rows = []
    for i in range(n):
        rows.append(
            {
                "query_id": f"q{i:03d}",
                "query_text_norm": f"query text {i}",
                "source_chunk_ids": [f"chunk_{i}", f"chunk_extra_{i}"],  # <- 完整列表应被保留
                "domain": "in",
            }
        )
    write_jsonl_raw(store, path, rows)


def seed_chunks(store: FakeStore, path: str, n: int):
    rows = []
    for i in range(n):
        rows.append(
            {
                "chunk_id": f"chunk_{i}",
                "chunk_text": f"chunk text {i}",
                "doc_id": f"doc_{i}",
                "chunk_index": i,
                "chunk_text_hash": f"h{i}",
            }
        )
    write_jsonl_raw(store, path, rows)


def test_run_pairing_flush_and_source_ids(monkeypatch, capsys):
    """
    验证：
    - 只读 queries_in_domain
    - 每条输出 1 个 QueryPack
    - buffer_size=15 会 flush 两次（15 + 5）
    - QueryPack.query.source_chunk_ids 保留完整列表
    """
    store = FakeStore()
    s = minimal_settings()

    queries_path = "rq_out/queries/in_domain.jsonl"
    pairs_path = "rq_out/pairs/pairs.pairwise.train.jsonl"
    chunks_path = "ce_out/chunks/chunks.jsonl"

    seed_queries(store, queries_path, n=20)
    seed_chunks(store, chunks_path, n=25)

    # monkeypatch settings/load/store registry/jsonl io
    import qr_pipeline.pipeline.run_pairing as rp

    monkeypatch.setattr(rp, "load_settings", lambda _: s)
    monkeypatch.setattr(rp, "build_store_registry", lambda settings: fake_build_store_registry(settings, store))

    # patch jsonl io to use our FakeStore raw
    monkeypatch.setattr(rp, "write_jsonl", lambda st, p, rows: write_jsonl_raw(st, p, list(rows)))
    monkeypatch.setattr(rp, "append_jsonl", lambda st, p, rows: write_jsonl_raw(st, p, read_jsonl_raw(st, p) + list(rows)))

    def _read_jsonl(st, p, on_error=None):
        for row in read_jsonl_raw(st, p):
            yield row

    monkeypatch.setattr(rp, "read_jsonl", _read_jsonl)

    # patch retriever + pairing
    fake_retriever = FakeRetriever(
        items=[
            {"key": "cand_1", "chunk_text": "cand text 1"},
            {"key": "cand_2", "chunk_text": "cand text 2"},
            {"key": "cand_3", "chunk_text": "cand text 3"},
        ]
    )
    monkeypatch.setattr(rp, "HybridRetriever", type("X", (), {"from_settings": staticmethod(lambda settings: fake_retriever)}))
    monkeypatch.setattr(rp, "build_pairs_for_query", fake_build_pairs_for_query)

    class DummyLLM:
        def generate(self, prompt: str) -> str:
            return "{}"

    class DummyEmbedder:
        def encode_passages(self, texts):
            return None

    res = rp.run_pairing(
        "configs/fake.yaml",
        llm=DummyLLM(),
        embedder=DummyEmbedder(),
        buffer_size=15,
        best_effort=True,
    )

    # check result counts
    assert res.num_queries_total == 20
    assert res.num_queries_processed == 20
    assert res.num_packs_written == 20
    assert res.num_errors == 0

    # check pairs jsonl lines
    packs = read_jsonl_raw(store, pairs_path)
    assert len(packs) == 20

    # check source_chunk_ids preserved fully
    first = packs[0]
    assert first["query"]["source_chunk_ids"] == ["chunk_0", "chunk_extra_0"]

    # check flush prints at least twice
    out = capsys.readouterr().out
    assert "flushed=15" in out
    assert "flushed_final=5" in out


def test_run_pairing_best_effort_logs_error(monkeypatch):
    """
    验证 best_effort=True 时，单条 query 出错会写入 errors.jsonl 并继续。
    """
    store = FakeStore()
    s = minimal_settings()

    queries_path = "rq_out/queries/in_domain.jsonl"
    errors_path = "rq_out/logs/run_pairing.errors.jsonl"
    pairs_path = "rq_out/pairs/pairs.pairwise.train.jsonl"
    chunks_path = "ce_out/chunks/chunks.jsonl"

    # 3条 query，其中第2条 source_chunk_ids 为空 -> 应报错并记录
    write_jsonl_raw(
        store,
        queries_path,
        [
            {"query_id": "q0", "query_text_norm": "ok", "source_chunk_ids": ["chunk_0"], "domain": "in"},
            {"query_id": "q1", "query_text_norm": "bad", "source_chunk_ids": [], "domain": "in"},
            {"query_id": "q2", "query_text_norm": "ok2", "source_chunk_ids": ["chunk_2"], "domain": "in"},
        ],
    )
    seed_chunks(store, chunks_path, n=5)

    import qr_pipeline.pipeline.run_pairing as rp

    monkeypatch.setattr(rp, "load_settings", lambda _: s)
    monkeypatch.setattr(rp, "build_store_registry", lambda settings: fake_build_store_registry(settings, store))

    monkeypatch.setattr(rp, "write_jsonl", lambda st, p, rows: write_jsonl_raw(st, p, list(rows)))
    monkeypatch.setattr(rp, "append_jsonl", lambda st, p, rows: write_jsonl_raw(st, p, read_jsonl_raw(st, p) + list(rows)))

    def _read_jsonl(st, p, on_error=None):
        for row in read_jsonl_raw(st, p):
            yield row

    monkeypatch.setattr(rp, "read_jsonl", _read_jsonl)

    fake_retriever = FakeRetriever(items=[{"key": "cand_1", "chunk_text": "cand text 1"}])
    monkeypatch.setattr(rp, "HybridRetriever", type("X", (), {"from_settings": staticmethod(lambda settings: fake_retriever)}))
    monkeypatch.setattr(rp, "build_pairs_for_query", fake_build_pairs_for_query)

    class DummyLLM:
        def generate(self, prompt: str) -> str:
            return "{}"

    class DummyEmbedder:
        def encode_passages(self, texts):
            return None

    res = rp.run_pairing(
        "configs/fake.yaml",
        llm=DummyLLM(),
        embedder=DummyEmbedder(),
        buffer_size=15,
        best_effort=True,
    )

    assert res.num_queries_total == 3
    assert res.num_queries_processed == 2
    assert res.num_packs_written == 2
    assert res.num_errors == 1

    packs = read_jsonl_raw(store, pairs_path)
    assert len(packs) == 2

    errs = read_jsonl_raw(store, errors_path)
    assert len(errs) == 1
    assert errs[0]["query_id"] == "q1"
    assert errs[0]["stage"] == "run_pairing"
