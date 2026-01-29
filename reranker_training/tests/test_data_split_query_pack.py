from __future__ import annotations

from typing import Dict, Any, List
import importlib
import orjson
import pytest


# =========================
# Fake Store (in-memory)
# =========================
class FakeStore:
    def __init__(self) -> None:
        self.files: Dict[str, bytes] = {}

    def exists(self, path: str) -> bool:
        return path in self.files

    def read_bytes(self, path: str) -> bytes:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    def write_bytes(self, path: str, content: bytes) -> None:
        self.files[path] = content


def _write_jsonl_bytes(rows: List[Dict[str, Any]]) -> bytes:
    return b"".join(orjson.dumps(r) + b"\n" for r in rows)


def _read_jsonl_bytes(b: bytes) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in b.splitlines():
        if not line.strip():
            continue
        obj = orjson.loads(line)
        assert isinstance(obj, dict)
        out.append(obj)
    return out


def make_settings() -> Dict[str, Any]:
    # 只提供函数会用到的字段即可
    return {
        "data_split": 0.5,  # train_ratio = 0.5 => valid_ratio=0.5
        "inputs": {
            "pairs": {
                "store": "fs_local",
                "base": "rq_out/pairs",
                "pairs": "query_pack.jsonl",
            }
        },
        "outputs": {
            "files": {
                "store": "fs_local",
                "base": "reranker_out",
                "train_path": "processed/train_query_pack.jsonl",
                "valid_path": "processed/valid_query_pack.jsonl",
            }
        },
        "stores": {},
    }


def make_query_pack(qid: str, qtext: str, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "query": {
            "query_id": qid,
            "query_text": qtext,
            "source_chunk_ids": ["c1", "c2"],
        },
        "positives": [
            {
                "chunk_id": "p1",
                "doc_id": "d1",
                "chunk_index": 0,
                "chunk_text": "pos",
                "chunk_text_hash": "h1",
            }
        ],
        "negatives": [
            {
                "chunk_id": "n1",
                "doc_id": "d2",
                "chunk_index": 0,
                "chunk_text": "neg",
                "chunk_text_hash": "h2",
            }
        ],
        "meta": {"stats": {"x": 1}},
    }
    if extra:
        row.update(extra)  # 用于验证“未知字段也原样保留”
    return row


# 你的模块路径（已根据你项目结构确定）
MODULE_UNDER_TEST = "reranker_training.data.data_split"


@pytest.fixture
def fake_store_and_patch_registry(monkeypatch):
    mod = importlib.import_module(MODULE_UNDER_TEST)

    store = FakeStore()
    registry = {"fs_local": store}

    # 关键：让被测函数拿到我们的 FakeStore
    monkeypatch.setattr(mod, "build_store_registry", lambda settings: registry)

    return store


def test_outputs_exist_and_rows_are_preserved(fake_store_and_patch_registry):
    mod = importlib.import_module(MODULE_UNDER_TEST)
    store = fake_store_and_patch_registry

    settings = make_settings()

    # 4 rows, with q1 duplicated => unique qids: q1,q2,q3
    rows_in = [
        make_query_pack("q1", "t1", extra={"custom_field": {"a": 1}}),
        make_query_pack("q1", "t1-second-row"),
        make_query_pack("q2", "t2"),
        make_query_pack("q3", "t3"),
    ]
    store.write_bytes("rq_out/pairs/query_pack.jsonl", _write_jsonl_bytes(rows_in))

    stats = mod.build_train_and_valid_query_pack_jsonl_from_pairs(
        settings, seed=123, batch_size=2, fail_fast=True
    )

    train_path = "reranker_out/processed/train_query_pack.jsonl"
    valid_path = "reranker_out/processed/valid_query_pack.jsonl"

    assert store.exists(train_path)
    assert store.exists(valid_path)

    train_rows = _read_jsonl_bytes(store.read_bytes(train_path))
    valid_rows = _read_jsonl_bytes(store.read_bytes(valid_path))

    # 输出总行数应等于输入总行数（所有行都合法）
    assert len(train_rows) + len(valid_rows) == len(rows_in)

    # 输出行必须“原样存在于输入集合中”（不改字段）
    in_set = {orjson.dumps(r) for r in rows_in}
    out_set = {orjson.dumps(r) for r in (train_rows + valid_rows)}
    assert out_set == in_set

    # stats sanity
    assert stats["num_input_rows"] == 4
    assert stats["num_unique_qids"] == 3
    assert stats["num_train_rows"] + stats["num_valid_rows"] == 4


def test_fail_fast_true_raises_on_bad_row(fake_store_and_patch_registry):
    mod = importlib.import_module(MODULE_UNDER_TEST)
    store = fake_store_and_patch_registry
    settings = make_settings()

    good = make_query_pack("q1", "t1")

    # bad: missing query_id
    bad = {
        "query": {"query_text": "hello"},
        "positives": [],
        "negatives": [],
        "meta": {},
    }

    store.write_bytes("rq_out/pairs/query_pack.jsonl", _write_jsonl_bytes([good, bad]))

    with pytest.raises(ValueError):
        mod.build_train_and_valid_query_pack_jsonl_from_pairs(settings, seed=1, fail_fast=True)


def test_fail_fast_false_skips_bad_rows(fake_store_and_patch_registry):
    mod = importlib.import_module(MODULE_UNDER_TEST)
    store = fake_store_and_patch_registry
    settings = make_settings()

    bad1 = {"query": {"query_id": "   ", "query_text": "x"}, "positives": [], "negatives": [], "meta": {}}
    bad2 = {"query": {"query_id": "q2", "query_text": "   "}, "positives": [], "negatives": [], "meta": {}}
    good1 = make_query_pack("q1", "t1")
    good2 = make_query_pack("q2", "t2")

    store.write_bytes("rq_out/pairs/query_pack.jsonl", _write_jsonl_bytes([bad1, good1, bad2, good2]))

    stats = mod.build_train_and_valid_query_pack_jsonl_from_pairs(settings, seed=7, fail_fast=False)

    train_rows = _read_jsonl_bytes(store.read_bytes("reranker_out/processed/train_query_pack.jsonl"))
    valid_rows = _read_jsonl_bytes(store.read_bytes("reranker_out/processed/valid_query_pack.jsonl"))
    all_out = train_rows + valid_rows

    # 只有两条 good survive
    assert len(all_out) == 2
    assert {r["query"]["query_id"] for r in all_out} == {"q1", "q2"}

    assert stats["skipped_rows"] >= 2
    assert stats["bad_rows"] >= 2


def test_all_bad_rows_outputs_empty_files(fake_store_and_patch_registry):
    mod = importlib.import_module(MODULE_UNDER_TEST)
    store = fake_store_and_patch_registry
    settings = make_settings()

    bad1 = {"query": {}, "positives": [], "negatives": [], "meta": {}}
    bad2 = {"query": {"query_id": ""}, "positives": [], "negatives": [], "meta": {}}

    store.write_bytes("rq_out/pairs/query_pack.jsonl", _write_jsonl_bytes([bad1, bad2]))

    stats = mod.build_train_and_valid_query_pack_jsonl_from_pairs(settings, fail_fast=False)

    train_b = store.read_bytes("reranker_out/processed/train_query_pack.jsonl")
    valid_b = store.read_bytes("reranker_out/processed/valid_query_pack.jsonl")

    # 你的 write_jsonl(rows=[]) 会写空文件 (b"")
    assert train_b == b""
    assert valid_b == b""

    assert stats["num_unique_qids"] == 0
    assert stats["num_train_rows"] == 0
    assert stats["num_valid_rows"] == 0
