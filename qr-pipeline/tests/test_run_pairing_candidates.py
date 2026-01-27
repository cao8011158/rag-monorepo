# tests/test_run_pairing_candidates.py
from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest


class DummyStore:
    """
    最小 out_store：run_pairing 最后会调用 out_store.write_text(stats_path, ...)
    """
    def __init__(self) -> None:
        self.written_text: Dict[str, str] = {}

    def write_text(self, path: str, text: str, encoding: str = "utf-8") -> None:
        self.written_text[path] = text


class DummyRetriever:
    def __init__(self, items: List[Dict[str, Any]]) -> None:
        self._items = items

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        return list(self._items)


def test_run_pairing_writes_candidates_and_pairwise(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    验证：
    - candidates_path 被写入的是未展开 PairSample 结构（包含 negatives list）
    - pairs_path 被写入的是 pairwise 结构（每个 negative 一行）
    - stats.json 里包含 candidates_path & num_candidates_written
    """
    import qr_pipeline.pipeline.run_pairing as rp

    # -----------------------
    # 1) 假 settings
    # -----------------------
    settings: Dict[str, Any] = {
        "outputs": {
            "store": "out",
            "base": "rq_out",
            "files": {
                "queries_in_domain": "queries/in_domain.jsonl",
                "candidates": "pairs/pairs.candidates.jsonl",   # ✅ 新增输出
                "pairs": "pairs/pairs.pairwise_train.jsonl",
                "stats": "run_stats.json",
                "errors": "errors.jsonl",
            },
        },
        "inputs": {
            "ce_artifacts": {
                "chunks": {
                    "store": "chunks",
                    "base": "ce_out",
                    "chunks_file": "chunks.jsonl",
                }
            }
        },
        "models": {"llm": {"model_name": "fake"}},
        "embedding": {"model_name": "fake-emb", "instructions": {"passage": "", "query": ""}},
        "_meta": {"test": True},
    }
    monkeypatch.setattr(rp, "load_settings", lambda _: settings)

    # -----------------------
    # 2) 假 stores registry
    # -----------------------
    out_store = DummyStore()
    chunks_store = object()  # run_pairing 不直接用 chunks_store 的方法（只通过 read_jsonl）
    monkeypatch.setattr(
        rp,
        "build_store_registry",
        lambda _settings: {"out": out_store, "chunks": chunks_store},
    )

    # -----------------------
    # 3) 假 retriever
    # -----------------------
    retrieved_items = [
        {"key": "C1", "chunk_text": "candidate doc one"},
        {"key": "C2", "chunk_text": "candidate doc two"},
    ]
    monkeypatch.setattr(
        rp.HybridRetriever,
        "from_settings",
        lambda _settings: DummyRetriever(retrieved_items),
    )

    # -----------------------
    # 4) 假 read_jsonl：分别喂 queries 和 chunks
    # -----------------------
    def fake_read_jsonl(store: Any, path: str, on_error: Any = None):
        # queries/in_domain.jsonl
        if path == "rq_out/queries/in_domain.jsonl":
            yield {
                "query_id": "Q_0001",
                "query_text_norm": "what is x",
                "source_chunk_ids": ["CH_1"],
                "llm_model": "Qwen2.5-7B-Instruct",
                "prompt_style": "information-seeking",
                "domain": "in",
            }
            return

        # ce_out/chunks.jsonl
        if path == "ce_out/chunks.jsonl":
            yield {"chunk_id": "CH_1", "chunk_text": "source chunk text"}
            return

        raise AssertionError(f"Unexpected read_jsonl path: {path}")

    monkeypatch.setattr(rp, "read_jsonl", fake_read_jsonl)

    # -----------------------
    # 5) 假 build_pairs_for_query：返回 1 个 PairSample（含 2 个 negatives）
    # -----------------------
    samples = [
        {
            "query_text": "what is x",
            "positive": {"doc_id": "CH_1", "text": "source chunk text"},
            "negatives": [
                {"doc_id": "C1", "text": "candidate doc one"},
                {"doc_id": "C2", "text": "candidate doc two"},
            ],
            "source_chunk": "source chunk text",
        }
    ]
    monkeypatch.setattr(
        rp,
        "build_pairs_for_query",
        lambda **kwargs: (samples, {"ok": True}),
    )

    # -----------------------
    # 6) 捕获 write_jsonl / append_jsonl 的写入
    # -----------------------
    cleared_paths: List[str] = []
    appended: Dict[str, List[Dict[str, Any]]] = {}

    def fake_write_jsonl(store: Any, path: str, rows: List[Dict[str, Any]]):
        # run_pairing 会尝试清空 candidates_path 和 pairs_path
        if rows == []:
            cleared_paths.append(path)

    def fake_append_jsonl(store: Any, path: str, rows: List[Dict[str, Any]]):
        appended.setdefault(path, []).extend(rows)

    monkeypatch.setattr(rp, "write_jsonl", fake_write_jsonl)
    monkeypatch.setattr(rp, "append_jsonl", fake_append_jsonl)

    # -----------------------
    # 7) 执行
    # -----------------------
    res = rp.run_pairing("dummy.yaml", llm=object(), embedder=object(), best_effort=True)

    # -----------------------
    # 8) 断言：清空行为
    # -----------------------
    assert "rq_out/pairs/pairs.candidates.jsonl" in cleared_paths
    assert "rq_out/pairs/pairs.pairwise_train.jsonl" in cleared_paths

    # -----------------------
    # 9) 断言：candidates 写入的是未展开结构（negatives 是 list）
    # -----------------------
    cand_path = "rq_out/pairs/pairs.candidates.jsonl"
    assert cand_path in appended
    assert len(appended[cand_path]) == 1

    cand_row = appended[cand_path][0]
    assert cand_row["query_id"] == "Q_0001"
    assert cand_row["query_text"] == "what is x"
    assert "negatives" in cand_row
    assert isinstance(cand_row["negatives"], list)
    assert len(cand_row["negatives"]) == 2
    assert "negative" not in cand_row  # ✅ 关键：未展开前不应该出现 single negative

    # meta 也应被补齐
    assert cand_row["meta"]["llm_model"] == "Qwen2.5-7B-Instruct"
    assert cand_row["meta"]["prompt_style"] == "information-seeking"
    assert cand_row["meta"]["domain"] == "in"

    # -----------------------
    # 10) 断言：pairwise 写入的是展开结构（每个 negative 一行）
    # -----------------------
    pair_path = "rq_out/pairs/pairs.pairwise_train.jsonl"
    assert pair_path in appended
    pair_rows = appended[pair_path]
    assert len(pair_rows) == 2  # 2 negatives => 2 行

    for r in pair_rows:
        assert r["query_id"] == "Q_0001"
        assert r["query_text"] == "what is x"
        assert "positive" in r and "negative" in r
        assert isinstance(r["negative"], dict)
        assert "negatives" not in r  # ✅ 展开后不应再有 negatives list
        assert r["meta"]["domain"] == "in"

    # -----------------------
    # 11) 断言：stats.json 包含 candidates_path / num_candidates_written
    # -----------------------
    stats_path = "rq_out/run_stats.json"
    assert stats_path in out_store.written_text

    stats_obj = json.loads(out_store.written_text[stats_path])
    assert stats_obj["outputs"]["candidates_path"] == cand_path
    assert stats_obj["num_candidates_written"] == 1
    assert stats_obj["num_pairs_written"] == 2
    assert stats_obj["num_queries_processed"] == 1
    assert res.num_pairs_written == 2
    assert res.num_queries_processed == 1
