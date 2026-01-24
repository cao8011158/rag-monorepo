# tests/test_chunking_stage.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from ce_pipeline.pipeline.chunking_stage import run_chunking_stage


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl_lines(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _base_cfg(tmp_path: Path) -> Dict[str, Any]:
    """
    Minimal dynamic settings dict to satisfy:
      - build_store_registry(cfg)
      - run_chunking_stage(cfg)
    """
    return {
        "input": {
            "input_store": "fs_local",
            "input_path": "cleaned/latest/documents.jsonl",
        },
        "outputs": {
            "chunks": {
                "store": "fs_local",
                "base": "ce_out/chunks",
            }
        },
        "stores": {
            "fs_local": {
                "kind": "filesystem",
                "root": str(tmp_path),
            }
        },
        "chunking": {
            # stage 不强依赖这些值，但 chunk_doc 签名需要 cfg
            "window_chars": 1200,
            "overlap_chars": 200,
            "min_chunk_chars": 200,
        },
        "processing": {
            "dedup": {
                "exact_dedup": {
                    "hash_field": "chunk_text_hash",
                }
            }
        },
    }


def test_chunking_stage_happy_path_and_exact_dedup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    - documents -> chunks.raw.jsonl
    - exact_dedup(chunks.raw) -> chunks.jsonl
    - 验证去重逻辑：chunk_text_hash 相同的重复 chunk 被移除
    """
    cfg = _base_cfg(tmp_path)

    # prepare input documents.jsonl
    in_file = tmp_path / "cleaned/latest/documents.jsonl"
    _write_jsonl(
        in_file,
        [
            {"doc_id": "d1", "text": "hello", "url": "u1"},
            {"doc_id": "d2", "text": "world", "url": "u2"},
        ],
    )

    # monkeypatch deterministic chunk_doc
    def fake_chunk_doc(doc: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        # d1 和 d2 各产生 2 个 chunk，其中第 1 个 chunk 的 hash 故意相同用于测试 exact dedup
        common_hash = "HASH_COMMON"
        if doc["doc_id"] == "d1":
            return [
                {
                    "chunk_id": "c_d1_0",
                    "doc_id": "d1",
                    "chunk_index": 0,
                    "text": "same chunk",
                    "chunk_text_hash": common_hash,
                },
                {
                    "chunk_id": "c_d1_1",
                    "doc_id": "d1",
                    "chunk_index": 1,
                    "text": "unique d1",
                    "chunk_text_hash": "HASH_D1",
                },
            ]
        if doc["doc_id"] == "d2":
            return [
                {
                    "chunk_id": "c_d2_0",
                    "doc_id": "d2",
                    "chunk_index": 0,
                    "text": "same chunk",
                    "chunk_text_hash": common_hash,  # duplicate
                },
                {
                    "chunk_id": "c_d2_1",
                    "doc_id": "d2",
                    "chunk_index": 1,
                    "text": "unique d2",
                    "chunk_text_hash": "HASH_D2",
                },
            ]
        return []

    monkeypatch.setattr("ce_pipeline.pipeline.chunking_stage.chunk_doc", fake_chunk_doc)

    summary = run_chunking_stage(cfg, fail_fast=False)

    # artifact paths
    raw_rel = summary["output"]["chunks_raw"]
    final_rel = summary["output"]["chunks_final"]
    raw_file = tmp_path / raw_rel
    final_file = tmp_path / final_rel

    raw_rows = _read_jsonl_lines(raw_file)
    final_rows = _read_jsonl_lines(final_file)

    assert len(raw_rows) == 4, "raw 应包含全部 chunk"
    assert len(final_rows) == 3, "final 应去掉 1 个重复 hash 的 chunk"

    # 验证保持 first occurrence：保留 d1 的 common chunk，丢弃 d2 的 common chunk
    kept_common = [r for r in final_rows if r["chunk_text_hash"] == "HASH_COMMON"]
    assert len(kept_common) == 1
    assert kept_common[0]["doc_id"] == "d1"

    # counts
    assert summary["counts"]["docs_total"] == 2
    assert summary["counts"]["docs_ok"] == 2
    assert summary["counts"]["chunks_total_raw"] == 4
    assert summary["counts"]["chunks_kept_after_exact_dedup"] == 3


def test_chunking_stage_best_effort_skips_bad_docs_and_logs_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    fail_fast=False 时：
    - 坏文档（缺 doc_id / text）跳过
    - errors.chunking.jsonl 写入错误
    - 迭代不中断，仍会处理后续好文档
    """
    cfg = _base_cfg(tmp_path)

    # input: 1 bad doc + 1 good doc
    in_file = tmp_path / "cleaned/latest/documents.jsonl"
    _write_jsonl(
        in_file,
        [
            {"doc_id": "", "text": "bad"},           # bad: empty doc_id
            {"doc_id": "d_ok", "text": "good one"},  # ok
        ],
    )

    def fake_chunk_doc(doc: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {
                "chunk_id": f"c_{doc['doc_id']}_0",
                "doc_id": doc["doc_id"],
                "chunk_index": 0,
                "text": "only",
                "chunk_text_hash": f"H_{doc['doc_id']}",
            }
        ]

    monkeypatch.setattr("ce_pipeline.pipeline.chunking_stage.chunk_doc", fake_chunk_doc)

    summary = run_chunking_stage(cfg, fail_fast=False)

    # final chunks should include only good doc chunks
    final_file = tmp_path / summary["output"]["chunks_final"]
    final_rows = _read_jsonl_lines(final_file)
    assert len(final_rows) == 1
    assert final_rows[0]["doc_id"] == "d_ok"

    # chunking errors should have 1 row
    err_chunking_file = tmp_path / summary["output"]["errors_chunking"]
    err_rows = _read_jsonl_lines(err_chunking_file)
    assert len(err_rows) == 1
    assert err_rows[0]["stage"] == "chunking_stage.validate_doc"

    # counts
    assert summary["counts"]["docs_total"] == 2
    assert summary["counts"]["docs_ok"] == 1
    assert summary["counts"]["docs_skipped"] == 1
    assert summary["counts"]["chunk_errors"] == 0


def test_chunking_stage_fail_fast_raises_on_bad_doc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    fail_fast=True 时：
    - 遇到坏文档立即抛异常
    """
    cfg = _base_cfg(tmp_path)

    in_file = tmp_path / "cleaned/latest/documents.jsonl"
    _write_jsonl(
        in_file,
        [
            {"doc_id": "", "text": "bad"},  # bad
            {"doc_id": "d2", "text": "ok"},
        ],
    )

    # 即便 fake_chunk_doc 不会被调用，仍补上以防未来逻辑变化
    def fake_chunk_doc(doc: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []

    monkeypatch.setattr("ce_pipeline.pipeline.chunking_stage.chunk_doc", fake_chunk_doc)

    with pytest.raises(ValueError):
        run_chunking_stage(cfg, fail_fast=True)
