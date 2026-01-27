from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Iterator, List, Optional

import pytest


# -----------------------------
# In-memory FakeStore
# -----------------------------
class FakeStore:
    def __init__(self) -> None:
        # store logical path -> text
        self.text: Dict[str, str] = {}
        # appended jsonl lines (for errors/pairs)
        self.appended: Dict[str, List[Dict[str, Any]]] = {}

    def exists(self, path: str) -> bool:
        return path in self.text or path in self.appended

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        if path not in self.text:
            raise FileNotFoundError(path)
        return self.text[path]

    def write_text(self, path: str, content: str, encoding: str = "utf-8") -> None:
        self.text[path] = content

    def append_jsonl_rows(self, path: str, rows: List[Dict[str, Any]]) -> None:
        self.appended.setdefault(path, []).extend(rows)


# -----------------------------
# Fake JSONL IO (monkeypatched)
# -----------------------------
def fake_read_jsonl(store: FakeStore, path: str, *, on_error=None) -> Iterator[Dict[str, Any]]:
    raw = store.read_text(path)
    for i, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj
        except Exception as e:
            if on_error is None:
                raise
            on_error(
                {
                    "stage": "read_jsonl",
                    "path": path,
                    "line_no": i,
                    "error": f"{type(e).__name__}: {e}",
                    "line_preview": line[:200],
                }
            )


def fake_write_jsonl(store: FakeStore, path: str, rows: Iterable[Dict[str, Any]]) -> None:
    # overwrite as jsonl
    out_lines: List[str] = []
    for r in rows:
        out_lines.append(json.dumps(r, ensure_ascii=False))
    store.write_text(path, "\n".join(out_lines) + ("\n" if out_lines else ""))


def fake_append_jsonl(store: FakeStore, path: str, rows: Iterable[Dict[str, Any]]) -> None:
    store.append_jsonl_rows(path, list(rows))


# -----------------------------
# Fakes for retriever & pairing
# -----------------------------
class FakeRetriever:
    def __init__(self, items: List[Dict[str, Any]]) -> None:
        self._items = items
        self.queries: List[str] = []

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        self.queries.append(query)
        return list(self._items)


class FakeHybridRetriever:
    @classmethod
    def from_settings(cls, settings: Dict[str, Any]) -> FakeRetriever:
        # default fake items; can be overridden by monkeypatching attribute if needed
        items = [
            {"key": "CAND1", "chunk_text": "cand one", "rrf_score": 1.0, "dense": {}, "bm25": {}},
            {"key": "CAND2", "chunk_text": "cand two", "rrf_score": 0.9, "dense": {}, "bm25": {}},
        ]
        return FakeRetriever(items)


def fake_build_pairs_for_query(**kwargs):
    # minimal deterministic PairSample output
    query_text = kwargs["query_text"]
    source_doc = kwargs["source_doc"]
    # produce one positive sample with two negatives
    samples = [
        {
            "query_text": query_text,
            "positive": source_doc,
            "negatives": [
                {"doc_id": "CAND1", "text": "cand one"},
                {"doc_id": "CAND2", "text": "cand two"},
            ],
            "source_chunk": source_doc["text"],
        }
    ]
    stats = {"ok": True}
    return samples, stats


# -----------------------------
# Shared fake settings/config
# -----------------------------
def make_settings() -> Dict[str, Any]:
    return {
        "stores": {
            "fs_local": {"kind": "filesystem", "root": "data"},
        },
        "inputs": {
            "ce_artifacts": {
                "chunks": {
                    "store": "fs_local",
                    "base": "ce_out/chunks",
                    "chunks_file": "chunks.jsonl",
                }
            }
        },
        "outputs": {
            "store": "fs_local",
            "base": "rq_out",
            "files": {
                "queries_in_domain": "queries/in_domain.jsonl",
                "pairs": "pairs/pairs.pairwise_train.jsonl",
                "stats": "run_stats.json",
                "errors": "errors.jsonl",
            },
        },
        "_meta": {"config_path": "configs/pipeline.yaml", "config_hash": "x" * 64},
    }


# -----------------------------
# Tests
# -----------------------------
def test_run_pairing_happy_path_writes_pairs_and_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    import qr_pipeline.pipeline.run_pairing as rp

    settings = make_settings()
    store = FakeStore()

    # prepare inputs
    # queries_in_domain
    store.write_text(
        "rq_out/queries/in_domain.jsonl",
        "\n".join(
            [
                json.dumps(
                    {
                        "query_id": "q1",
                        "query_text_norm": "when was cmu founded",
                        "source_chunk_ids": ["SRC1"],
                        "domain": "in",
                    }
                )
            ]
        )
        + "\n",
    )
    # chunks.jsonl
    store.write_text(
        "ce_out/chunks/chunks.jsonl",
        "\n".join(
            [
                json.dumps({"chunk_id": "SRC1", "chunk_text": "CMU was founded in 1900."}),
                json.dumps({"chunk_id": "OTHER", "chunk_text": "other text"}),
            ]
        )
        + "\n",
    )

    # monkeypatch settings/stores/io/retriever/pairing
    monkeypatch.setattr(rp, "load_settings", lambda _: settings)
    monkeypatch.setattr(rp, "build_store_registry", lambda _s: {"fs_local": store})
    monkeypatch.setattr(rp, "read_jsonl", fake_read_jsonl)
    monkeypatch.setattr(rp, "write_jsonl", fake_write_jsonl)
    monkeypatch.setattr(rp, "append_jsonl", fake_append_jsonl)
    monkeypatch.setattr(rp, "HybridRetriever", FakeHybridRetriever)
    monkeypatch.setattr(rp, "build_pairs_for_query", fake_build_pairs_for_query)

    # dummy llm/embedder (pairing is monkeypatched, so unused)
    llm = object()
    embedder = object()

    res = rp.run_pairing("configs/pipeline.yaml", llm=llm, embedder=embedder, best_effort=True)

    assert res.num_queries_total == 1
    assert res.num_queries_processed == 1
    assert res.num_errors == 0
    # fake_build_pairs_for_query returns 1 sample with 2 negatives -> 2 pairwise rows
    assert res.num_pairs_written == 2

    # pairs should be appended
    pairs_path = "rq_out/pairs/pairs.pairwise_train.jsonl"
    assert pairs_path in store.appended
    assert len(store.appended[pairs_path]) == 2
    assert store.appended[pairs_path][0]["query_id"] == "q1"

    # stats should be written
    stats_path = "rq_out/run_stats.json"
    assert stats_path in store.text
    stats_obj = json.loads(store.text[stats_path])
    assert stats_obj["num_pairs_written"] == 2


def test_run_pairing_best_effort_logs_error_and_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    import qr_pipeline.pipeline.run_pairing as rp

    settings = make_settings()
    store = FakeStore()

    # two queries: first ok, second missing source chunk
    store.write_text(
        "rq_out/queries/in_domain.jsonl",
        "\n".join(
            [
                json.dumps({"query_id": "q1", "query_text_norm": "q1", "source_chunk_ids": ["SRC1"], "domain": "in"}),
                json.dumps({"query_id": "q2", "query_text_norm": "q2", "source_chunk_ids": ["MISSING"], "domain": "in"}),
            ]
        )
        + "\n",
    )
    store.write_text(
        "ce_out/chunks/chunks.jsonl",
        json.dumps({"chunk_id": "SRC1", "chunk_text": "src one"}) + "\n",
    )

    monkeypatch.setattr(rp, "load_settings", lambda _: settings)
    monkeypatch.setattr(rp, "build_store_registry", lambda _s: {"fs_local": store})
    monkeypatch.setattr(rp, "read_jsonl", fake_read_jsonl)
    monkeypatch.setattr(rp, "write_jsonl", fake_write_jsonl)
    monkeypatch.setattr(rp, "append_jsonl", fake_append_jsonl)
    monkeypatch.setattr(rp, "HybridRetriever", FakeHybridRetriever)
    monkeypatch.setattr(rp, "build_pairs_for_query", fake_build_pairs_for_query)

    res = rp.run_pairing("configs/pipeline.yaml", llm=object(), embedder=object(), best_effort=True)

    assert res.num_queries_total == 2
    # q1 processed; q2 fails but best_effort continues (q2 not counted as processed)
    assert res.num_queries_processed == 1
    assert res.num_errors == 1

    errors_path = "rq_out/errors.jsonl"
    assert errors_path in store.appended
    assert len(store.appended[errors_path]) == 1
    assert store.appended[errors_path][0]["query_id"] == "q2"


def test_run_pairing_fail_fast_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import qr_pipeline.pipeline.run_pairing as rp

    settings = make_settings()
    store = FakeStore()

    # missing source chunk => should raise when best_effort=False
    store.write_text(
        "rq_out/queries/in_domain.jsonl",
        json.dumps({"query_id": "q1", "query_text_norm": "q1", "source_chunk_ids": ["MISSING"], "domain": "in"}) + "\n",
    )
    store.write_text("ce_out/chunks/chunks.jsonl", "")  # empty

    monkeypatch.setattr(rp, "load_settings", lambda _: settings)
    monkeypatch.setattr(rp, "build_store_registry", lambda _s: {"fs_local": store})
    monkeypatch.setattr(rp, "read_jsonl", fake_read_jsonl)
    monkeypatch.setattr(rp, "write_jsonl", fake_write_jsonl)
    monkeypatch.setattr(rp, "append_jsonl", fake_append_jsonl)
    monkeypatch.setattr(rp, "HybridRetriever", FakeHybridRetriever)
    monkeypatch.setattr(rp, "build_pairs_for_query", fake_build_pairs_for_query)

    with pytest.raises(Exception):
        rp.run_pairing("configs/pipeline.yaml", llm=object(), embedder=object(), best_effort=False)
