from __future__ import annotations

import numpy as np
import pytest

# 根据你的结构：src/ce_pipeline/indexing/vector.py
# 你在 indexing/__init__.py 里如果也导出了 build_faiss_index，也可以从 ce_pipeline.indexing import build_faiss_index
from ce_pipeline.indexing.vector import build_faiss_index


faiss = pytest.importorskip("faiss")  # 如果没装 faiss，自动 skip 这些测试


def test_build_faiss_index_flatip_adds_vectors_and_sets_dim() -> None:
    emb = np.random.rand(10, 8).astype(np.float32)
    index = build_faiss_index(emb, index_type="FlatIP")

    # 类型检查：IndexFlatIP / IndexFlat (不同 faiss 版本表现略有差异，但 IndexFlatIP 是最常见)
    assert isinstance(index, faiss.Index)
    assert index.d == 8
    assert index.ntotal == 10


def test_build_faiss_index_flatl2_adds_vectors_and_sets_dim() -> None:
    emb = np.random.rand(7, 5).astype(np.float32)
    index = build_faiss_index(emb, index_type="FlatL2")

    assert isinstance(index, faiss.Index)
    assert index.d == 5
    assert index.ntotal == 7


def test_build_faiss_index_converts_float64_to_float32() -> None:
    emb64 = np.random.rand(6, 4).astype(np.float64)
    index = build_faiss_index(emb64, index_type="FlatIP")

    # 通过 reconstruct 检查：faiss 内部保存的是 float32
    v0 = np.zeros((4,), dtype=np.float32)
    index.reconstruct(0, v0)
    assert v0.dtype == np.float32
    assert index.ntotal == 6


def test_build_faiss_index_rejects_unsupported_index_type() -> None:
    emb = np.random.rand(3, 2).astype(np.float32)
    with pytest.raises(ValueError, match="Unsupported faiss index_type"):
        build_faiss_index(emb, index_type="IVF")  # 你当前没支持 IVF/HNSW 等


def test_build_faiss_index_allows_empty_matrix_with_known_dim() -> None:
    # 允许 N=0，但必须给出 D（即 shape=(0, D)）
    emb = np.empty((0, 12), dtype=np.float32)
    index = build_faiss_index(emb, index_type="FlatIP")

    assert index.d == 12
    assert index.ntotal == 0


def test_build_faiss_index_raises_on_wrong_shape_1d() -> None:
    # 错误形状：没有第二维，会在 emb.shape[1] 处失败
    emb = np.random.rand(8).astype(np.float32)
    with pytest.raises(Exception):
        build_faiss_index(emb, index_type="FlatIP")


def test_build_faiss_index_handles_non_contiguous_array() -> None:
    # 非连续内存的 view（工程中很常见，比如切片）
    base = np.random.rand(20, 6).astype(np.float32)
    emb = base[::2]  # shape=(10,6) but not necessarily contiguous
    assert emb.shape == (10, 6)

    index = build_faiss_index(emb, index_type="FlatIP")
    assert index.d == 6
    assert index.ntotal == 10
