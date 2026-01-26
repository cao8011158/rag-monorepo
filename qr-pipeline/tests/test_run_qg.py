# tests/test_run_qg.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np
import pytest


# -----------------------------
# Minimal in-memory Store
# -----------------------------
class MemoryStore:
    def __init__(self) -> None:
        self._fs: Dict[str, bytes] = {}

    def exists(self, path: str) -> bool:
        return path in self._fs

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        if path not in self._fs:
            raise FileNotFoundError(path)
        return self._fs[path].decode(encoding)

    def write_text(self, path: str, content: str, encoding: str = "utf-8") -> None:
        self._fs[path] = content.encode(encoding)

    def read_bytes(self, path: str) -> bytes:
        if path not in self._fs:
            raise FileNotFoundError(path)
        return self._fs[path]

    def write_bytes(self, path: str, content: bytes) -> None:
        self._fs[path] = content

    def append_bytes(self, path: str, content: bytes) -> None:
        if path in self._fs and len(self._fs[path]) > 0 and not self._fs[path].endswith(b"\n"):
            self._fs[path] += b"\n"
        self._fs[path] = self._fs.get(path, b"") + content

    def list(self, prefix: str) -> Iterable[str]:
        p = prefix.rstrip("/") + "/"
        for k in sorted(self._fs.keys()):
            if k == prefix or k.startswith(p):
                yield k


# -----------------------------
# Local helper to read JSONL from MemoryStore
# -----------------------------
def _read_jsonl_text(store: MemoryStore, path: str) -> List[Dict[str, Any]]:
    text = store.read_text(path)
    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


# -----------------------------
# Fake ANN result + fake components
# -----------------------------
@dataclass
class FakeANNDedupResult:
    kept_indices: List[int]
    removed_mask: np.ndarray
    num_kept: int
    num_removed: int


class FakeEmbedder:
    """
    encode_queries 返回简单二维向量，确保可控重复：
      - 相同文本 => 相同向量
    """
    def __init__(
        self,
        model_name: str,
        passage_instruction: str,
        query_instruction: str,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.passage_instruction = passage_instruction
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device

    def encode_queries(self, texts: List[str]) -> np.ndarray:
        vecs: List[List[float]] = []
        for t in texts:
            # 把文本 hash 到一个小空间，保证重复文本同向量
            h = abs(hash(t)) % 1000
            vecs.append([float(h), float(h) / 10.0])
        arr = np.array(vecs, dtype=np.float32)
        return arr


def fake_near_dedup_by_ann_faiss(
    emb: np.ndarray,
    *,
    threshold: float = 0.95,
    topk: int = 20,
    hnsw_m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 64,
    normalize: bool = True,
) -> FakeANNDedupResult:
    """
    一个确定性“近重复”模拟：
      - 向量完全相等 => 重复
      - 保留首次出现
    """
    if emb.ndim != 2:
        raise ValueError("emb must be 2D")
    n = emb.shape[0]
    removed = np.zeros((n,), dtype=bool)
    seen = {}
    for i in range(n):
        key = tuple(emb[i].tolist())
        if key in seen:
            removed[i] = True
        else:
            seen[key] = i
    kept = [i for i in range(n) if not removed[i]]
    return FakeANNDedupResult(
        kept_indices=kept,
        removed_mask=removed,
        num_kept=len(kept),
        num_removed=int(removed.sum()),
    )


# -----------------------------
# Tests
# -----------------------------
def test_run_pipeline_semantic_dedup_enabled_writes_back_filtered(monkeypatch: pytest.MonkeyPatch) -> None:
    import qr_pipeline.pipeline.run_qg as rqg
    from qr_pipeline.io.jsonl import write_jsonl

    store = MemoryStore()

    # fake settings
    s: Dict[str, Any] = {
        "stores": {
            "fs_local": {"driver": "memory"}  # 仅占位；实际 build_store_registry 会被 monkeypatch
        },
        "outputs": {
            "store": "fs_local",
            "base": "qr_out",
            "files": {
                "errors": "errors.jsonl",
                "queries_in_domain": "queries/in_domain.jsonl",
                "queries_out_domain": "queries/out_domain.jsonl",
                "stats": "run_stats.json",
            },
        },
        "models": {
            "embedding": {
                "model_name": "fake-embedder",
                "batch_size": 16,
                "normalize_embeddings": True,
                "device": None,
                "instructions": {"passage": "passage:", "query": "query:"},
            }
        },
        "processing": {
            "dedup": {
                "semantic_dedup": {
                    "enable": True,
                    "threshold": 0.95,
                    "topk": 10,
                    "hnsw_m": 16,
                    "ef_construction": 100,
                    "ef_search": 32,
                    "normalize": True,
                    "min_text_chars": 0,
                }
            }
        },
    }

    # patch load_settings / build_store_registry
    monkeypatch.setattr(rqg, "load_settings", lambda _: s)
    monkeypatch.setattr(rqg, "build_store_registry", lambda _cfg: {"fs_local": store})

    # patch embedder + ann
    monkeypatch.setattr(rqg, "DualInstructEmbedder", FakeEmbedder)
    monkeypatch.setattr(rqg, "near_dedup_by_ann_faiss", fake_near_dedup_by_ann_faiss)

    # patch run_query_generation: 写出两份 queries 文件（包含重复 query_text_norm）
    def fake_run_query_generation(_settings: Dict[str, Any]) -> Dict[str, Any]:
        out_base = _settings["outputs"]["base"]
        files = _settings["outputs"]["files"]
        in_path = rqg._posix_join(out_base, files["queries_in_domain"])
        out_path = rqg._posix_join(out_base, files["queries_out_domain"])

        in_rows = [
            {"query_id": "a1", "query_text_norm": "what is cmu", "domain": "in"},
            {"query_id": "a2", "query_text_norm": "what is cmu", "domain": "in"},  # duplicate by text
            {"query_id": "a3", "query_text_norm": "pittsburgh history", "domain": "in"},
        ]
        out_rows = [
            {"query_id": "b1", "query_text_norm": "best pizza", "domain": "out"},
            {"query_id": "b2", "query_text_norm": "best pizza", "domain": "out"},  # duplicate by text
        ]
        write_jsonl(store, in_path, in_rows)
        write_jsonl(store, out_path, out_rows)

        return {
            "ts_ms": 1,
            "inputs": {"chunks_path": "ce_out/chunks/chunks.jsonl", "num_chunks_total_read": 10, "num_chunks_sampled": 10},
            "outputs": {
                "queries_in_domain_path": in_path,
                "queries_out_domain_path": out_path,
                "errors_path": rqg._posix_join(out_base, files["errors"]),
                "num_queries_in_domain_unique_written": 3,
                "num_queries_out_domain_unique_written": 2,
            },
            "counters": {"llm_calls_generate_queries": 10, "llm_calls_domain_classify": 10},
            "meta": {"llm_model": "fake", "prompt_style": "info"},
        }

    monkeypatch.setattr(rqg, "run_query_generation", fake_run_query_generation)

    # run
    final_stats = rqg.run_pipeline("configs/pipeline.yaml")

    in_path = final_stats["outputs"]["queries_in_domain_path"]
    out_path = final_stats["outputs"]["queries_out_domain_path"]
    stats_path = final_stats["outputs"]["stats_path"]

    assert store.exists(in_path)
    assert store.exists(out_path)
    assert store.exists(stats_path)

    # after semantic dedup, duplicates should be removed: in 3 -> 2, out 2 -> 1
    in_rows_after = _read_jsonl_text(store, in_path)
    out_rows_after = _read_jsonl_text(store, out_path)

    assert len(in_rows_after) == 2
    assert len(out_rows_after) == 1

    # semantic_dedup stats should reflect removals
    sd = final_stats["semantic_dedup"]
    assert sd["enabled"] is True
    assert sd["in_domain"]["num_removed"] == 1
    assert sd["out_domain"]["num_removed"] == 1


def test_run_pipeline_semantic_dedup_disabled_does_not_touch_files(monkeypatch: pytest.MonkeyPatch) -> None:
    import qr_pipeline.pipeline.run_qg as rqg
    from qr_pipeline.io.jsonl import write_jsonl

    store = MemoryStore()

    s: Dict[str, Any] = {
        "stores": {"fs_local": {"driver": "memory"}},
        "outputs": {
            "store": "fs_local",
            "base": "qr_out",
            "files": {
                "errors": "errors.jsonl",
                "queries_in_domain": "queries/in_domain.jsonl",
                "queries_out_domain": "queries/out_domain.jsonl",
                "stats": "run_stats.json",
            },
        },
        "models": {
            "embedding": {
                "model_name": "fake-embedder",
                "batch_size": 16,
                "normalize_embeddings": True,
                "device": None,
                "instructions": {"passage": "passage:", "query": "query:"},
            }
        },
        "processing": {"dedup": {"semantic_dedup": {"enable": False}}},
    }

    monkeypatch.setattr(rqg, "load_settings", lambda _: s)
    monkeypatch.setattr(rqg, "build_store_registry", lambda _cfg: {"fs_local": store})

    # run_query_generation 写出文件（包含重复），但 dedup disabled 不会删
    def fake_run_query_generation(_settings: Dict[str, Any]) -> Dict[str, Any]:
        out_base = _settings["outputs"]["base"]
        files = _settings["outputs"]["files"]
        in_path = rqg._posix_join(out_base, files["queries_in_domain"])
        out_path = rqg._posix_join(out_base, files["queries_out_domain"])

        in_rows = [
            {"query_id": "a1", "query_text_norm": "dup", "domain": "in"},
            {"query_id": "a2", "query_text_norm": "dup", "domain": "in"},
        ]
        out_rows = [
            {"query_id": "b1", "query_text_norm": "dup2", "domain": "out"},
            {"query_id": "b2", "query_text_norm": "dup2", "domain": "out"},
        ]
        write_jsonl(store, in_path, in_rows)
        write_jsonl(store, out_path, out_rows)

        return {
            "ts_ms": 1,
            "inputs": {},
            "outputs": {
                "queries_in_domain_path": in_path,
                "queries_out_domain_path": out_path,
                "errors_path": rqg._posix_join(out_base, files["errors"]),
                "num_queries_in_domain_unique_written": 2,
                "num_queries_out_domain_unique_written": 2,
            },
            "counters": {},
            "meta": {},
        }

    monkeypatch.setattr(rqg, "run_query_generation", fake_run_query_generation)

    final_stats = rqg.run_pipeline("configs/pipeline.yaml")

    in_path = final_stats["outputs"]["queries_in_domain_path"]
    out_path = final_stats["outputs"]["queries_out_domain_path"]

    # unchanged counts
    assert len(_read_jsonl_text(store, in_path)) == 2
    assert len(_read_jsonl_text(store, out_path)) == 2
    assert final_stats["semantic_dedup"]["enabled"] is False


def test_main_prints_all_sections(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    import qr_pipeline.pipeline.run_qg as rqg

    # patch run_pipeline to avoid real work
    fake_stats = {
        "ts_ms": 123,
        "config_path": "configs/pipeline.yaml",
        "outputs": {
            "queries_in_domain_path": "qr_out/queries/in_domain.jsonl",
            "queries_out_domain_path": "qr_out/queries/out_domain.jsonl",
            "errors_path": "qr_out/errors.jsonl",
            "stats_path": "qr_out/run_stats.json",
        },
        "query_generation": {
            "ts_ms": 1,
            "inputs": {"chunks_path": "ce_out/chunks/chunks.jsonl", "num_chunks_total_read": 10, "num_chunks_sampled": 10},
            "outputs": {"num_queries_in_domain_unique_written": 3, "num_queries_out_domain_unique_written": 2},
            "counters": {"llm_calls_generate_queries": 10, "llm_calls_domain_classify": 10},
            "meta": {"llm_model": "fake", "prompt_style": "info"},
        },
        "semantic_dedup": {
            "enabled": True,
            "in_domain": {"file_path": "a", "num_total": 3, "num_kept": 2, "num_removed": 1, "ann": {"threshold": 0.95}},
            "out_domain": {"file_path": "b", "num_total": 2, "num_kept": 1, "num_removed": 1, "ann": {"threshold": 0.95}},
        },
    }

    monkeypatch.setattr(rqg, "run_pipeline", lambda _cfg: fake_stats)

    # simulate argv: qr-qg --config configs/pipeline.yaml
    monkeypatch.setattr("sys.argv", ["qr-qg", "--config", "configs/pipeline.yaml"])

    rqg.main()
    out = capsys.readouterr().out

    # key section markers should appear
    assert "QR PIPELINE: RUN_QG" in out
    assert "[OUTPUTS]" in out
    assert "[QUERY_GENERATION]" in out
    assert "[SEMANTIC_DEDUP]" in out
    assert "DONE" in out
