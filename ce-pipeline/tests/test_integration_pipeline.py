from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pytest

from ce_pipeline.pipeline.run import run_pipeline


# -------------------------
# Fake Embedder (避免下载模型/网络不稳定)
# -------------------------
class _FakeDualInstructEmbedder:
    def __init__(
        self,
        model_name: str,
        passage_instruction: str,
        query_instruction: str,
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        device=None,
    ) -> None:
        self.model_name = model_name
        self.passage_instruction = passage_instruction
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        # 生成稳定、可复现的向量：每条文本 -> [D=8] float32
        # 让向量不全一样，避免 near_dedup 把它们全删掉（即使 enable 了）
        dim = 8
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            # 简单 hash -> 数值
            h = sum((ord(c) for c in t[:200])) % 997
            out[i, 0] = float(i + 1)
            out[i, 1] = float(h)
            out[i, 2:] = 0.1
        # normalize 可选（你的 near_dedup/FAISS FlatIP 常配合 normalize）
        if self.normalize_embeddings and len(texts) > 0:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out = out / norms
        return out


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


@pytest.mark.integration
def test_pipeline_end_to_end_tmp_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    真正的“集成测试”：
    - 不使用你项目真实的 data/ce_out
    - 使用 tmp_path 作为 fs_local.root
    - 从 fixtures 复制最小 documents.jsonl
    - 跑完整 pipeline：chunking -> embedding -> bm25
    - 断言产物存在 + chunks schema 正确
    """

    # 1) monkeypatch embedder，避免下载模型
    import ce_pipeline.pipeline.embedding_stage as embedding_stage_mod
    monkeypatch.setattr(embedding_stage_mod, "DualInstructEmbedder", _FakeDualInstructEmbedder)

    # 2) 准备输入：tmp_path/cleaned/latest/documents.jsonl
    fixture_path = Path(__file__).parent / "fixtures" / "documents.jsonl"
    assert fixture_path.exists(), f"Fixture not found: {fixture_path}"

    in_rel = Path("cleaned/latest/documents.jsonl")
    in_abs = tmp_path / in_rel
    in_abs.parent.mkdir(parents=True, exist_ok=True)
    in_abs.write_bytes(fixture_path.read_bytes())

    # 3) 组装 cfg（动态 dict），root 指向 tmp_path
    cfg = {
        "input": {
            "input_store": "fs_local",
            "input_path": str(in_rel).replace("\\", "/"),
        },
        "outputs": {
            "chunks": {"store": "fs_local", "base": "ce_out/chunks"},
            "vector_index": {"store": "fs_local", "base": "ce_out/indexes/vector"},
            "bm25_index": {"store": "fs_local", "base": "ce_out/indexes/bm25"},
        },
        "stores": {
            "fs_local": {
                "kind": "filesystem",
                "root": str(tmp_path),  # ✅ 关键：独立环境
            }
        },
        # chunking knobs（尽量让三条短文也能出 chunk）
        "chunking": {
            "window_chars": 400,
            "overlap_chars": 80,
            "min_chunk_chars": 80,
        },
        # embedding config（会被 fake embedder 接收，但不下载）
        "embedding": {
            "model_name": "fake-model",
            "batch_size": 16,
            "normalize_embeddings": True,
            "instructions": {"passage": "passage: ", "query": "query: "},
            "device": None,
        },
        # dedup config：exact 依赖 chunk_text_hash；semantic 可开可关
        "processing": {
            "dedup": {
                "exact_dedup": {"hash_field": "chunk_text_hash"},
                "semantic_dedup": {
                    "enable": True,
                    "threshold": 0.95,
                    "topk": 10,
                    "hnsw_m": 16,
                    "ef_construction": 100,
                    "ef_search": 64,
                    "normalize": True,
                },
            }
        },
        # indexing vector config（embedding_stage 会读取）
        "indexing": {"vector": {"faiss_index": "FlatIP"}},
    }

    # 4) 跑 pipeline（默认：chunking best-effort, embedding strict, indexing best-effort）
    res = run_pipeline(cfg)

    # 5) 断言产物存在（都在 tmp_path 下）
    chunks_jsonl = tmp_path / "ce_out/chunks/chunks.jsonl"
    chunks_raw = tmp_path / "ce_out/chunks/chunks.raw.jsonl"
    faiss_index = tmp_path / "ce_out/indexes/vector/faiss.index"
    id_map = tmp_path / "ce_out/indexes/vector/id_map.jsonl"
    meta = tmp_path / "ce_out/indexes/vector/meta.json"
    bm25 = tmp_path / "ce_out/indexes/bm25/bm25.pkl"

    assert chunks_raw.exists(), "chunks.raw.jsonl not created"
    assert chunks_jsonl.exists(), "chunks.jsonl not created"
    assert faiss_index.exists(), "faiss.index not created"
    assert id_map.exists(), "id_map.jsonl not created"
    assert meta.exists(), "meta.json not created"
    assert bm25.exists(), "bm25.pkl not created"

    # 6) 断言 chunks schema（你方案A后应该是 chunk_text）
    rows = _read_jsonl(chunks_jsonl)
    assert len(rows) > 0, "chunks.jsonl is empty"

    for r in rows[:5]:
        assert isinstance(r.get("chunk_id"), str) and r["chunk_id"], "missing chunk_id"
        assert isinstance(r.get("chunk_text"), str) and r["chunk_text"].strip(), "missing chunk_text"
        assert isinstance(r.get("chunk_text_hash"), str) and r["chunk_text_hash"], "missing chunk_text_hash"

    # 7) 可选：检查 summary 返回结构存在（不做强约束）
    assert "chunking" in res and "embedding" in res, "pipeline summary missing keys"
