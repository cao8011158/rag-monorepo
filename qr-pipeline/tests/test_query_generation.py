from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest


# -----------------------------
# In-memory Store + JSONL helpers
# -----------------------------

class MemStore:
    """
    Minimal store interface used by the pipeline:
      - exists(path)
      - write_text(path, text, encoding)
    JSONL read/append are monkeypatched at module-level, not methods here.
    """
    def __init__(self) -> None:
        self.text_files: Dict[str, str] = {}
        self.jsonl_files: Dict[str, List[Dict[str, Any]]] = {}

    def exists(self, path: str) -> bool:
        return path in self.text_files or path in self.jsonl_files

    def write_text(self, path: str, text: str, encoding: str = "utf-8") -> None:
        self.text_files[path] = text


def _mem_read_jsonl(store: MemStore, path: str, on_error=None):
    # emulate generator behavior
    rows = store.jsonl_files.get(path, [])
    for r in rows:
        yield r


def _mem_append_jsonl(store: MemStore, path: str, rows: List[Dict[str, Any]]) -> None:
    store.jsonl_files.setdefault(path, [])
    store.jsonl_files[path].extend(rows)


# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture()
def base_settings() -> Dict[str, Any]:
    """
    Minimal settings dict required by run_query_generation_pipeline().
    """
    return {
        "stores": {
            "fs_local": {"kind": "filesystem", "root": "/dev/null"}
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
                "queries_out_domain": "queries/out_domain.jsonl",
                "stats": "run_stats.json",
                "errors": "errors.jsonl",
            },
        },
        "models": {
            "gemini_api": {
                "model_name": "gemini-test-model"
            }
        },
        "query_generation": {
            "target_num_queries": 2000,
            "domain_batch_size": 3,
            "sampling": {
                "strategy": "uniform_random",
                "seed": 123,
                "max_chunks_considered": 100,
            },
            "prompt": {
                "style": "information-seeking",
                "avoid_near_duplicates": True,
                "max_chunk_chars": 1800,
            },
            "postprocess": {
                "min_query_chars": 1,
                "max_query_chars": 200,
            },
            "normalize": {
                "lower": True,
                "strip": True,
                "collapse_whitespace": True,
            },
        },
    }


@pytest.fixture()
def mem_stores():
    """
    Create one in-store (chunks) and one out-store (outputs).
    """
    in_store = MemStore()
    out_store = MemStore()
    return in_store, out_store


def _paths(s: Dict[str, Any]) -> Dict[str, str]:
    base = s["outputs"]["base"]
    files = s["outputs"]["files"]
    return {
        "chunks_path": "ce_out/chunks/chunks.jsonl",
        "in_path": f"{base}/{files['queries_in_domain']}",
        "out_path": f"{base}/{files['queries_out_domain']}",
        "err_path": f"{base}/{files['errors']}",
        "stats_path": f"{base}/{files['stats']}",
    }


# -----------------------------
# Tests
# -----------------------------

def test_generation_classification_writes_and_dedups(monkeypatch, base_settings, mem_stores):
    """
    - generator returns duplicates across chunks
    - pipeline should normalize/hash + global dedup
    - classification by 1-based indices -> in/out files
    """
    in_store, out_store = mem_stores
    p = _paths(base_settings)

    # seed chunks input
    in_store.jsonl_files[p["chunks_path"]] = [
        {"chunk_id": "c1", "chunk_text": "Chunk 1 text"},
        {"chunk_id": "c2", "chunk_text": "Chunk 2 text"},
    ]

    # import module under test
    import qr_pipeline.pipeline.query_generation as mod

    # monkeypatch store registry
    monkeypatch.setattr(mod, "build_store_registry", lambda s: {"fs_local": (in_store if True else None)})  # placeholder


    # build_store_registry must return both stores; pipeline grabs store by key twice
    def _fake_build_store_registry(s):
        # in_store used for reading chunks; out_store for writing outputs
        # pipeline uses same store key for both, so we return the same object
        # but we want separate in/out. easiest: return out_store and route read_jsonl by store object identity.
        # Instead: patch read_jsonl/append_jsonl to ignore store and use path prefix to choose.
        return {"fs_local": out_store}

    monkeypatch.setattr(mod, "build_store_registry", _fake_build_store_registry)

    # patch jsonl ops to use in_store for chunks_path, out_store otherwise
    def _read_jsonl_router(store, path, on_error=None):
        if path == p["chunks_path"]:
            yield from _mem_read_jsonl(in_store, path, on_error=on_error)
        else:
            yield from _mem_read_jsonl(out_store, path, on_error=on_error)

    def _append_jsonl_router(store, path, rows):
        _mem_append_jsonl(out_store, path, rows)

    monkeypatch.setattr(mod, "read_jsonl", _read_jsonl_router)
    monkeypatch.setattr(mod, "append_jsonl", _append_jsonl_router)

    # patch generator: duplicates across chunks + extra spaces/case to test normalize
    def fake_gen(chunk_text: str, cfg: Dict[str, Any]) -> List[str]:
        if "Chunk 1" in chunk_text:
            return ["Hello  Pittsburgh", "DUPLICATE QUERY", "short"]
        return ["duplicate   query", "Another question", "hello pittsburgh"]

    monkeypatch.setattr(mod, "run_query_generation", fake_gen)

    # patch classifier: IN = first item only (Q1), OUT = rest
    def fake_cls(queries: List[str], cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {"in_ids": [1], "out_ids": [2, 3]}

    monkeypatch.setattr(mod, "run_query_classification", fake_cls)

    # run
    mod.run_query_generation_pipeline(base_settings)

    # assert outputs
    in_rows = out_store.jsonl_files.get(p["in_path"], [])
    out_rows = out_store.jsonl_files.get(p["out_path"], [])

    # global dedup: "DUPLICATE QUERY" and "duplicate query" should normalize to same and appear once total
    all_norm = [r["query_text_norm"] for r in in_rows + out_rows]
    assert all_norm.count("duplicate query") == 1

    # classifier put 1 item per batch into IN
    assert len(in_rows) >= 1
    assert all(r["domain"] == "in" for r in in_rows)
    assert all(r["domain"] == "out" for r in out_rows)

    # each row contains required fields
    sample = (in_rows + out_rows)[0]
    assert "query_id" in sample
    assert "query_text_norm" in sample
    assert "source_chunk_ids" in sample and isinstance(sample["source_chunk_ids"], list)

    # stats written
    assert p["stats_path"] in out_store.text_files
    snap = json.loads(out_store.text_files[p["stats_path"]])
    assert snap["meta"]["llm_model"] == "gemini-test-model"


def test_classification_conflict_out_wins(monkeypatch, base_settings, mem_stores):
    """
    If classifier returns same index in both in_ids and out_ids, OUT should win.
    """
    in_store, out_store = mem_stores
    p = _paths(base_settings)

    in_store.jsonl_files[p["chunks_path"]] = [
        {"chunk_id": "c1", "chunk_text": "Chunk 1 text"},
    ]

    import qr_pipeline.pipeline.query_generation as mod

    monkeypatch.setattr(mod, "build_store_registry", lambda s: {"fs_local": out_store})

    def _read_jsonl_router(store, path, on_error=None):
        if path == p["chunks_path"]:
            yield from _mem_read_jsonl(in_store, path, on_error=on_error)
        else:
            yield from _mem_read_jsonl(out_store, path, on_error=on_error)

    monkeypatch.setattr(mod, "read_jsonl", _read_jsonl_router)
    monkeypatch.setattr(mod, "append_jsonl", lambda store, path, rows: _mem_append_jsonl(out_store, path, rows))

    # generator returns 3 queries -> one batch
    monkeypatch.setattr(mod, "run_query_generation", lambda passage, cfg: ["q1", "q2", "q3"])

    # conflict: put 1 in both
    monkeypatch.setattr(mod, "run_query_classification", lambda qs, cfg: {"in_ids": [1], "out_ids": [1]})

    mod.run_query_generation_pipeline(base_settings)

    in_rows = out_store.jsonl_files.get(p["in_path"], [])
    out_rows = out_store.jsonl_files.get(p["out_path"], [])

    # conflict should lead to OUT, so IN should be empty (or at least q1 not in IN)
    assert len(in_rows) == 0
    assert len(out_rows) == 3


def test_classification_exception_all_out_and_error_logged(monkeypatch, base_settings, mem_stores):
    """
    If run_query_classification raises (e.g. ValidationError), entire batch should be OUT,
    and errors.jsonl should record the failure.
    """
    in_store, out_store = mem_stores
    p = _paths(base_settings)

    in_store.jsonl_files[p["chunks_path"]] = [
        {"chunk_id": "c1", "chunk_text": "Chunk 1 text"},
    ]

    import qr_pipeline.pipeline.query_generation as mod

    monkeypatch.setattr(mod, "build_store_registry", lambda s: {"fs_local": out_store})

    def _read_jsonl_router(store, path, on_error=None):
        if path == p["chunks_path"]:
            yield from _mem_read_jsonl(in_store, path, on_error=on_error)
        else:
            yield from _mem_read_jsonl(out_store, path, on_error=on_error)

    monkeypatch.setattr(mod, "read_jsonl", _read_jsonl_router)
    monkeypatch.setattr(mod, "append_jsonl", lambda store, path, rows: _mem_append_jsonl(out_store, path, rows))

    monkeypatch.setattr(mod, "run_query_generation", lambda passage, cfg: ["q1", "q2", "q3"])

    def boom(qs, cfg):
        raise ValueError("schema validation failed")

    monkeypatch.setattr(mod, "run_query_classification", boom)

    mod.run_query_generation_pipeline(base_settings)

    # all should be OUT
    in_rows = out_store.jsonl_files.get(p["in_path"], [])
    out_rows = out_store.jsonl_files.get(p["out_path"], [])
    assert len(in_rows) == 0
    assert len(out_rows) == 3

    # error logged
    err_rows = out_store.jsonl_files.get(p["err_path"], [])
    assert any(r.get("stage") == "domain_classification" for r in err_rows)
    assert any("schema validation failed" in (r.get("error") or "") for r in err_rows)


def test_resume_seen_ids_skips_existing(monkeypatch, base_settings, mem_stores):
    """
    If in/out domain files already contain query_id, pipeline should skip it (global dedup resume).
    """
    in_store, out_store = mem_stores
    p = _paths(base_settings)

    # one chunk
    in_store.jsonl_files[p["chunks_path"]] = [
        {"chunk_id": "c1", "chunk_text": "Chunk 1 text"},
    ]

    import qr_pipeline.pipeline.query_generation as mod

    monkeypatch.setattr(mod, "build_store_registry", lambda s: {"fs_local": out_store})

    def _read_jsonl_router(store, path, on_error=None):
        if path == p["chunks_path"]:
            yield from _mem_read_jsonl(in_store, path, on_error=on_error)
        else:
            yield from _mem_read_jsonl(out_store, path, on_error=on_error)

    monkeypatch.setattr(mod, "read_jsonl", _read_jsonl_router)
    monkeypatch.setattr(mod, "append_jsonl", lambda store, path, rows: _mem_append_jsonl(out_store, path, rows))

    # generator: produces one query; we will precompute its query_id by using module helpers
    monkeypatch.setattr(mod, "run_query_generation", lambda passage, cfg: ["Hello Pittsburgh"])

    # classifier says it is IN
    monkeypatch.setattr(mod, "run_query_classification", lambda qs, cfg: {"in_ids": [1], "out_ids": []})

    # Prepopulate in_domain with same query_id so resume dedup will skip
    q_norm = mod._normalize_text("Hello Pittsburgh", lower=True, strip=True, collapse_whitespace=True)
    qid = mod._make_id(q_norm)
    out_store.jsonl_files[p["in_path"]] = [{
        "query_id": qid,
        "query_text_norm": q_norm,
        "source_chunk_ids": ["old_chunk"],
        "llm_model": "gemini-test-model",
        "prompt_style": "information-seeking",
        "domain": "in",
    }]

    mod.run_query_generation_pipeline(base_settings)

    # should still have only 1 row (no duplicate append)
    in_rows = out_store.jsonl_files.get(p["in_path"], [])
    assert len(in_rows) == 1
