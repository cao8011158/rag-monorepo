from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from ce_pipeline.pipeline.indexing_stage import run_indexing_stage
from ce_pipeline.indexing.bm25 import load_bm25


def _make_cfg(tmp_root: Path) -> Dict[str, Any]:
    """
    最小 cfg（动态 dict），只包含 indexing_stage 需要的字段：
    - stores.fs_local
    - outputs.chunks
    - outputs.bm25_index
    """
    return {
        "stores": {
            "fs_local": {
                "kind": "filesystem",
                "root": str(tmp_root),
            }
        },
        "outputs": {
            "chunks": {
                "store": "fs_local",
                "base": "ce_out/chunks",
            },
            "bm25_index": {
                "store": "fs_local",
                "base": "ce_out/indexes/bm25",
            },
        },
    }


def _write_chunks_jsonl(path: Path, lines: List[bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for line in lines:
            if not line.endswith(b"\n"):
                line += b"\n"
            f.write(line)


def test_indexing_stage_default_is_best_effort_and_writes_bm25(tmp_path: Path) -> None:
    """
    你的设计：stage 默认 best_effort=True
    因此 chunks.jsonl 中存在坏行时：
    - 不应抛异常
    - 应生成 bm25.pkl
    - 应生成 errors.indexing.read_chunks.jsonl（至少包含 1 条错误记录）
    """
    cfg = _make_cfg(tmp_path)

    chunks_file = tmp_path / "ce_out" / "chunks" / "chunks.jsonl"
    good1 = json.dumps({"chunk_id": "c1", "text": "hello world"}).encode("utf-8")
    bad_json = b"{ this is bad json"
    good2 = json.dumps({"chunk_id": "c2", "text": "another chunk"}).encode("utf-8")
    _write_chunks_jsonl(chunks_file, [good1, bad_json, good2])

    stats = run_indexing_stage(cfg)  # 默认 best_effort=True
    assert stats["stage"] == "indexing_stage"
    assert stats["num_chunks_indexed"] == 2

    bm25_path = tmp_path / "ce_out" / "indexes" / "bm25" / "bm25.pkl"
    assert bm25_path.exists()

    errors_path = tmp_path / "ce_out" / "indexes" / "bm25" / "errors.indexing.read_chunks.jsonl"
    assert errors_path.exists()

    # errors 文件至少应该有一行（bad_json）
    err_lines = errors_path.read_text(encoding="utf-8").splitlines()
    assert len([ln for ln in err_lines if ln.strip()]) >= 1

    # errors 行应为 JSON 对象（stage=read_jsonl）
    first_err = json.loads(err_lines[0])
    assert first_err["stage"] == "read_jsonl"


def test_indexing_stage_strict_fail_fast_on_bad_json(tmp_path: Path) -> None:
    """
    strict 模式：显式 best_effort=False
    一旦 chunks.jsonl 中出现坏 JSON 行，read_jsonl 应直接抛异常，中断 stage。
    """
    cfg = _make_cfg(tmp_path)

    chunks_file = tmp_path / "ce_out" / "chunks" / "chunks.jsonl"
    good1 = json.dumps({"chunk_id": "c1", "text": "hello world"}).encode("utf-8")
    bad_json = b"{ this is bad json"
    good2 = json.dumps({"chunk_id": "c2", "text": "another chunk"}).encode("utf-8")
    _write_chunks_jsonl(chunks_file, [good1, bad_json, good2])

    with pytest.raises(Exception):
        run_indexing_stage(cfg, best_effort=False)


def test_indexing_stage_best_effort_skips_missing_fields_and_preserves_doc_ids_order(tmp_path: Path) -> None:
    """
    best-effort：坏行（bad json / 缺字段）会被跳过，但有效 chunk 会被收集进 bm25
    并且 doc_ids 顺序应与有效行出现顺序一致。
    """
    cfg = _make_cfg(tmp_path)

    chunks_file = tmp_path / "ce_out" / "chunks" / "chunks.jsonl"

    good1 = json.dumps({"chunk_id": "c1", "text": "hello world"}).encode("utf-8")
    bad_json = b"{ this is bad json"
    missing_chunk_id = json.dumps({"text": "missing id"}).encode("utf-8")
    missing_text = json.dumps({"chunk_id": "cX"}).encode("utf-8")
    good2 = json.dumps({"chunk_id": "c2", "text": "another chunk"}).encode("utf-8")

    _write_chunks_jsonl(chunks_file, [good1, bad_json, missing_chunk_id, missing_text, good2])

    stats = run_indexing_stage(cfg, best_effort=True)
    assert stats["num_chunks_indexed"] == 2

    # 读取 bm25 artifact 验证 doc_ids
    store = _fs_store_from_cfg(cfg)
    artifact = load_bm25(store, "ce_out/indexes/bm25/bm25.pkl")
    assert artifact.doc_ids == ["c1", "c2"]
    assert artifact.bm25 is not None


def test_indexing_stage_missing_chunks_file_raises(tmp_path: Path) -> None:
    """
    chunks.jsonl 不存在时应抛 FileNotFoundError（无论 best_effort 与否都应如此）
    """
    cfg = _make_cfg(tmp_path)

    with pytest.raises(FileNotFoundError):
        run_indexing_stage(cfg)  # 默认 best-effort=True 也应抛


# --------- helper: construct filesystem store for loading artifact ---------
def _fs_store_from_cfg(cfg: Dict[str, Any]):
    """
    测试用：构造 FilesystemStore 以读取 bm25.pkl
    按你的实际路径修改 import（只需改这一行）。
    """
    from ce_pipeline.stores.filesystem import FilesystemStore  # ← 若你的文件名不同请改这里

    root = Path(cfg["stores"]["fs_local"]["root"])
    return FilesystemStore(root=root)
