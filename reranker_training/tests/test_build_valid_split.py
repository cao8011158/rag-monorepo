from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

import pytest

# 你的模块路径（截图显示 build_valid_split.py 在 src/reranker_training/ 下）
import reranker_training.build_valid_split as m


# -----------------------
# Fake Store (内存版)
# -----------------------
class FakeStore:
    def __init__(self) -> None:
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
# JSONL raw helpers
# -----------------------
def _write_jsonl_raw(store: FakeStore, path: str, rows: List[Dict[str, Any]]) -> None:
    text = ""
    if rows:
        text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    store.write_text(path, text, encoding="utf-8")


def _read_jsonl_raw(store: FakeStore, path: str) -> List[Dict[str, Any]]:
    if not store.exists(path):
        return []
    text = store.read_text(path, encoding="utf-8")
    out: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


# -----------------------
# Monkeypatch: store registry + jsonl io
# -----------------------
@pytest.fixture()
def fake_env(monkeypatch: pytest.MonkeyPatch) -> FakeStore:
    store = FakeStore()

    # patch build_store_registry(cfg) -> {"fs_local": store}
    monkeypatch.setattr(m, "build_store_registry", lambda cfg: {"fs_local": store})

    # patch read_jsonl / write_jsonl / append_jsonl to run against FakeStore text
    def _read_jsonl(store_obj: FakeStore, path: str, *, on_error=None):
        # on_error ignored in tests
        for row in _read_jsonl_raw(store_obj, path):
            yield row

    def _write_jsonl(store_obj: FakeStore, path: str, rows: Iterable[Dict[str, Any]]) -> None:
        _write_jsonl_raw(store_obj, path, list(rows))

    def _append_jsonl(store_obj: FakeStore, path: str, rows: Iterable[Dict[str, Any]]) -> None:
        existing = store_obj.read_text(path, encoding="utf-8") if store_obj.exists(path) else ""
        # ensure existing ends with newline if not empty
        if existing and not existing.endswith("\n"):
            existing += "\n"
        add_lines = [json.dumps(r, ensure_ascii=False) for r in rows]
        add_text = ("\n".join(add_lines) + ("\n" if add_lines else ""))
        store_obj.write_text(path, existing + add_text, encoding="utf-8")

    monkeypatch.setattr(m, "read_jsonl", _read_jsonl)
    monkeypatch.setattr(m, "write_jsonl", _write_jsonl)
    monkeypatch.setattr(m, "append_jsonl", _append_jsonl)

    return store


# -----------------------
# Test data
# -----------------------
def _settings(base: str = "reranker_t_out") -> Dict[str, Any]:
    # 与你截图一致：settings["inputs"]["files"] 下包含 train_path/valid_path/candidates
    return {
        "data_split": 0.85,  # train ratio
        "stores": {"fs_local": {"kind": "filesystem", "root": "/dev/null"}},
        "inputs": {
            "files": {
                "store": "fs_local",
                "base": base,
                "train_path": "data/processed/train.jsonl",
                "valid_path": "data/processed/valid.jsonl",
                "candidates": "pairs.pairwise.jsonl",
                "stats": "run_stats.json",
                "errors": "errors.jsonl",
            }
        },
    }


def _candidates_rows() -> List[Dict[str, Any]]:
    # 4 个 query_id，其中 q1/q4 各有 2 条（模拟一个 query 多个 positive）
    # negatives 用少量就够验证逻辑
    return [
        {
            "query_id": "q1",
            "query_text": "q1 text",
            "positive": {"doc_id": "p1a", "text": "pos1a"},
            "negatives": [{"doc_id": "n1", "text": "neg1"}, {"doc_id": "n2", "text": "neg2"}],
            "source_chunk": "c1",
            "meta": {"domain": "in"},
        },
        {
            "query_id": "q1",
            "query_text": "q1 text",
            "positive": {"doc_id": "p1b", "text": "pos1b"},
            "negatives": [{"doc_id": "n3", "text": "neg3"}],
            "source_chunk": "c1",
            "meta": {"domain": "in"},
        },
        {
            "query_id": "q2",
            "query_text": "q2 text",
            "positive": {"doc_id": "p2", "text": "pos2"},
            "negatives": [{"doc_id": "n4", "text": "neg4"}],
            "source_chunk": "c2",
            "meta": {"domain": "out"},
        },
        {
            "query_id": "q3",
            "query_text": "q3 text",
            "positive": {"doc_id": "p3", "text": "pos3"},
            "negatives": [{"doc_id": "n5", "text": "neg5"}, {"doc_id": "n6", "text": "neg6"}],
            "source_chunk": "c3",
            "meta": {"domain": "out"},
        },
        {
            "query_id": "q4",
            "query_text": "q4 text",
            "positive": {"doc_id": "p4a", "text": "pos4a"},
            "negatives": [{"doc_id": "n7", "text": "neg7"}],
            "source_chunk": "c4",
            "meta": {"domain": "in"},
        },
        {
            "query_id": "q4",
            "query_text": "q4 text",
            "positive": {"doc_id": "p4b", "text": "pos4b"},
            # 故意制造 3 种应跳过的 negative：
            # 1) neg==pos
            # 2) neg 缺 doc_id
            # 3) neg 缺 text
            "negatives": [
                {"doc_id": "p4b", "text": "pos4b"},
                {"text": "missing doc_id"},
                {"doc_id": "n8"},
            ],
            "source_chunk": "c4",
            "meta": {"domain": "in"},
        },
        {
            # pos 不合法：应被 valid 展平跳过（但如果该 qid 在 train，train 是原样保留）
            "query_id": "q_bad_pos",
            "query_text": "bad pos",
            "positive": {"doc_id": "", "text": ""},  # invalid
            "negatives": [{"doc_id": "n9", "text": "neg9"}],
            "source_chunk": "c_bad",
            "meta": {},
        },
    ]


def _paths_from_settings(s: Dict[str, Any]) -> Dict[str, str]:
    base = s["inputs"]["files"].get("base", "")
    return {
        "candidates_path": m._pjoin(base, s["inputs"]["files"]["candidates"]),
        "train_path": m._pjoin(base, s["inputs"]["files"]["train_path"]),
        "valid_path": m._pjoin(base, s["inputs"]["files"]["valid_path"]),
    }


# -----------------------
# Tests
# -----------------------
def test_missing_train_path_raises_keyerror(fake_env: FakeStore) -> None:
    s = _settings()
    del s["inputs"]["files"]["train_path"]

    # candidates file exists to ensure it fails specifically due to missing key
    p = _paths_from_settings({**_settings(), "inputs": {"files": {**_settings()["inputs"]["files"], "train_path": "x"}}})
    _write_jsonl_raw(fake_env, p["candidates_path"], _candidates_rows())

    with pytest.raises(KeyError):
        m.build_train_and_fixed_valid_jsonl_from_candidates(s, seed=42, batch_size=3)


def test_build_train_and_valid_outputs(fake_env: FakeStore) -> None:
    s = _settings(base="reranker_t_out")
    paths = _paths_from_settings(s)

    rows = _candidates_rows()
    _write_jsonl_raw(fake_env, paths["candidates_path"], rows)

    seed = 42
    stats = m.build_train_and_fixed_valid_jsonl_from_candidates(s, seed=seed, batch_size=2)

    # ---- stats should include train/valid ----
    assert stats["candidates_path"] == paths["candidates_path"]
    assert stats["train_path"] == paths["train_path"]
    assert stats["valid_path"] == paths["valid_path"]
    assert stats["seed"] == seed
    assert stats["train_ratio"] == float(s["data_split"])
    assert stats["num_candidate_rows"] == len(rows)

    # ---- compute expected valid_qids with same helper (stable) ----
    all_qids = [str(r.get("query_id", "")).strip() for r in rows if str(r.get("query_id", "")).strip()]
    valid_qids = set(m._stable_sample_qids_for_valid(all_qids, train_ratio=float(s["data_split"]), seed=seed))

    # ---- train.jsonl: 原样保留（不展平、结构与 candidates 行一致） ----
    train_rows = _read_jsonl_raw(fake_env, paths["train_path"])
    expected_train_rows = [r for r in rows if r["query_id"] not in valid_qids]
    assert train_rows == expected_train_rows
    assert stats["num_train_rows"] == len(expected_train_rows)

    # ---- valid.jsonl: 展平，并应用过滤规则 ----
    valid_rows = _read_jsonl_raw(fake_env, paths["valid_path"])

    expected_valid_rows: List[Dict[str, Any]] = []
    for r in rows:
        if r["query_id"] not in valid_qids:
            continue

        pos = r.get("positive") or {}
        negs = r.get("negatives") or []
        if not isinstance(pos, dict) or not pos.get("doc_id") or not pos.get("text"):
            continue
        if not isinstance(negs, list) or len(negs) == 0:
            continue

        pos_doc_id = str(pos.get("doc_id", "")).strip()
        pos_text = str(pos.get("text", ""))

        for i, neg in enumerate(negs):
            if not isinstance(neg, dict):
                continue
            neg_doc_id = str(neg.get("doc_id", "")).strip()
            neg_text = neg.get("text", "")

            # 过滤规则（与你确认的一致）
            if not neg_doc_id or not neg_text:
                continue
            if neg_doc_id == pos_doc_id:
                continue

            expected_valid_rows.append(
                {
                    "query_id": r["query_id"],
                    "query_text": str(r.get("query_text", "") or ""),
                    "positive": {"doc_id": pos_doc_id, "text": pos_text},
                    "negative": {"doc_id": neg_doc_id, "text": str(neg_text)},
                    "source_chunk": str(r.get("source_chunk", "") or ""),
                    "meta": r.get("meta") or {},
                    "meta_pair": {"neg_rank_in_row": i},
                }
            )

    assert valid_rows == expected_valid_rows
    assert stats["num_valid_rows"] == len(expected_valid_rows)

    # split sanity: train_qids 和 valid_qids 互斥（按 query_id）
    train_qids = {r["query_id"] for r in train_rows}
    valid_qids_in_rows = {r["query_id"] for r in valid_rows}
    assert train_qids.isdisjoint(valid_qids_in_rows)


def test_outputs_are_overwritten(fake_env: FakeStore) -> None:
    s = _settings(base="reranker_t_out")
    paths = _paths_from_settings(s)

    _write_jsonl_raw(fake_env, paths["candidates_path"], _candidates_rows())

    # 预写垃圾数据，确保函数运行后输出被覆盖/重建
    _write_jsonl_raw(fake_env, paths["train_path"], [{"bad": 1}])
    _write_jsonl_raw(fake_env, paths["valid_path"], [{"bad": 2}])

    m.build_train_and_fixed_valid_jsonl_from_candidates(s, seed=42, batch_size=3)

    train_rows = _read_jsonl_raw(fake_env, paths["train_path"])
    valid_rows = _read_jsonl_raw(fake_env, paths["valid_path"])

    assert train_rows != [{"bad": 1}]
    assert valid_rows != [{"bad": 2}]

    # 结构断言
    if train_rows:
        assert "negatives" in train_rows[0]  # 原始结构
    if valid_rows:
        assert "negative" in valid_rows[0]   # 展平结构
        assert "meta_pair" in valid_rows[0]
