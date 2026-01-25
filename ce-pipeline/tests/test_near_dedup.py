from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from ce_pipeline.io.jsonl import read_jsonl, write_jsonl
from ce_pipeline.stores.registry import build_store_registry


def _base_settings(tmp_path: Path) -> Dict[str, Any]:
    """
    生成最小 settings：
    - stores.fs_local.root 指到 pytest tmp_path/data
    - outputs.chunks.base = ce_out/chunks
    - semantic_dedup.enable 可在测试里改
    """
    return {
        "stores": {
            "fs_local": {
                "kind": "filesystem",
                "root": str(tmp_path / "data"),
            }
        },
        "outputs": {
            "chunks": {
                "store": "fs_local",
                "base": "ce_out/chunks",
            }
        },
        "processing": {
            "dedup": {
                "semantic_dedup": {
                    "enable": True,
                    "threshold": 0.95,
                    "topk": 20,
                    "hnsw_m": 32,
                    "ef_construction": 200,
                    "ef_search": 64,
                    "normalize": True,
                }
            }
        },
    }


def _read_all(store, posix_path: str) -> List[Dict[str, Any]]:
    return list(read_jsonl(store, posix_path, on_error=None))


def test_near_dedup_prune_raises_on_count_mismatch_and_does_not_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ✅ 新逻辑：chunks 行数 != emb 行数 => 直接抛错终止，并且不覆盖写回 chunks.jsonl
    """
    import ce_pipeline.processing.near_dedup as nd

    s = _base_settings(tmp_path)
    stores = build_store_registry(s)
    store = stores["fs_local"]

    chunks_path = "ce_out/chunks/chunks.jsonl"

    # chunks：3 行
    rows = [
        {"chunk_id": "c1", "chunk_text": "a"},
        {"chunk_id": "c2", "chunk_text": "b"},
        {"chunk_id": "c3", "chunk_text": "c"},
    ]
    write_jsonl(store, chunks_path, rows)

    # emb：2 行（故意 mismatch）
    emb = np.random.randn(2, 4).astype(np.float32)

    # patch near_dedup_by_ann_faiss：返回 removed_mask 长度=2（与 emb 对齐）
    fake_res = nd.ANNDedupResult(
        kept_indices=[0, 1],
        removed_mask=np.array([False, False], dtype=bool),
        num_kept=2,
        num_removed=0,
    )
    monkeypatch.setattr(nd, "near_dedup_by_ann_faiss", lambda *args, **kwargs: fake_res)

    before = _read_all(store, chunks_path)

    with pytest.raises(ValueError) as ei:
        nd.near_dedup_and_prune_chunks(
            s=s,
            emb=emb,
            chunks_filename="chunks.jsonl",
            on_read_error=None,  # fail-fast 读（更容易暴露问题）
        )

    msg = str(ei.value).lower()
    assert "mismatch" in msg
    # 下面两条取决于你 ValueError 文案；如果你没写这两个字段，可以删掉
    assert "chunks_rows=3" in msg
    assert "emb_rows=2" in msg

    after = _read_all(store, chunks_path)
    assert after == before, "mismatch 时必须不写回（保持 chunks.jsonl 原样）"


def test_near_dedup_prune_overwrites_when_count_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ✅ chunks 行数 == emb 行数 => 允许覆盖写回，按 removed_mask 删除重复行
    """
    import ce_pipeline.processing.near_dedup as nd

    s = _base_settings(tmp_path)
    stores = build_store_registry(s)
    store = stores["fs_local"]

    chunks_path = "ce_out/chunks/chunks.jsonl"

    rows = [
        {"chunk_id": "c1", "chunk_text": "a"},
        {"chunk_id": "c2", "chunk_text": "b"},
        {"chunk_id": "c3", "chunk_text": "c"},
    ]
    write_jsonl(store, chunks_path, rows)

    emb = np.random.randn(3, 4).astype(np.float32)

    # 删除 idx=1
    fake_res = nd.ANNDedupResult(
        kept_indices=[0, 2],
        removed_mask=np.array([False, True, False], dtype=bool),
        num_kept=2,
        num_removed=1,
    )
    monkeypatch.setattr(nd, "near_dedup_by_ann_faiss", lambda *args, **kwargs: fake_res)

    res = nd.near_dedup_and_prune_chunks(
        s=s,
        emb=emb,
        chunks_filename="chunks.jsonl",
        on_read_error=None,
    )

    assert res.num_removed == 1
    assert res.num_kept == 2

    out = _read_all(store, chunks_path)
    assert out == [rows[0], rows[2]]


def test_near_dedup_prune_noop_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    ✅ enable=false => 不应修改 chunks.jsonl，并返回“全保留”
    """
    import ce_pipeline.processing.near_dedup as nd

    s = _base_settings(tmp_path)
    s["processing"]["dedup"]["semantic_dedup"]["enable"] = False

    stores = build_store_registry(s)
    store = stores["fs_local"]

    chunks_path = "ce_out/chunks/chunks.jsonl"

    rows = [
        {"chunk_id": "c1", "chunk_text": "a"},
        {"chunk_id": "c2", "chunk_text": "b"},
    ]
    write_jsonl(store, chunks_path, rows)

    emb = np.random.randn(2, 4).astype(np.float32)

    # enable=false 时不应调用 near_dedup_by_ann_faiss
    called = {"v": False}

    def _fake(*args, **kwargs):
        called["v"] = True
        raise AssertionError("near_dedup_by_ann_faiss 不应在 enable=false 时被调用")

    monkeypatch.setattr(nd, "near_dedup_by_ann_faiss", _fake)

    before = _read_all(store, chunks_path)
    res = nd.near_dedup_and_prune_chunks(
        s=s,
        emb=emb,
        chunks_filename="chunks.jsonl",
        on_read_error=None,
    )
    after = _read_all(store, chunks_path)

    assert called["v"] is False
    assert before == after
    assert res.kept_indices == [0, 1]
    assert res.num_kept == 2
    assert res.num_removed == 0
    assert res.removed_mask.shape == (2,)
    assert bool(res.removed_mask.any()) is False
