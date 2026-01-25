from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from qr_pipeline.stores.base import Store
from qr_pipeline.stores.registry import build_store_registry
from qr_pipeline.io.jsonl import read_jsonl, write_jsonl


@dataclass
class ANNDedupResult:
    kept_indices: List[int]         # 保留下来的向量/chunk 行号索引（按输入顺序）
    removed_mask: np.ndarray        # shape [N], True=被删（重复）
    num_kept: int
    num_removed: int


def _posix_join(*parts: str) -> str:
    return "/".join([p.strip("/").replace("\\", "/") for p in parts if p is not None and str(p) != ""])


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("emb must be 2D [N, D]")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def _default_log_error_factory(store: Store, path: str) -> Callable[[Dict[str, Any]], None]:
    """
    默认错误上报：追加到 ce_out/logs/errors.read_jsonl.jsonl 之类的路径
    你也可以在上层传入自定义 on_error。
    """
    # 这里不给你强行固定日志目录；用 chunks base 的同级 logs 也行
    # 为了简单：写到 "logs/read_jsonl.errors.jsonl"
    err_path = "logs/read_jsonl.errors.jsonl"

    def _on_error(payload: Dict[str, Any]) -> None:
        # 你的 append_jsonl/append_bytes 可能也可用；这里假设你实现了 append_jsonl
        # 如果你只有 append_bytes，也可以改成 bytes 追加
        try:
            from qr_pipeline.io.jsonl import append_jsonl  # 如果你实现了这个
            append_jsonl(store, err_path, [payload])
        except Exception:
            # 兜底：忽略日志写入失败，不影响主流程
            pass

    return _on_error


def near_dedup_by_ann_faiss(
    emb: np.ndarray,
    *,
    threshold: float = 0.95,
    topk: int = 20,
    hnsw_m: int = 32,
    ef_construction: int = 200,
    ef_search: int = 64,
    normalize: bool = True,
) -> ANNDedupResult:
    """
    Near-duplicate removal via ANN (FAISS HNSW) + local cosine verification.

    Notes:
    - If normalize=True: cosine similarity == inner product on L2-normalized vectors.
    - Keeps the first occurrence (lower index) and removes later duplicates.
    """
    if emb.ndim != 2:
        raise ValueError("emb must be 2D [N, D]")
    n, d = emb.shape
    if n == 0:
        return ANNDedupResult([], np.zeros((0,), dtype=bool), 0, 0)
    if topk < 2:
        raise ValueError("topk must be >= 2 (needs self + neighbors)")

    x = emb.astype(np.float32, copy=False)
    if normalize:
        x = _l2_normalize(x)

    try:
        import faiss  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "FAISS is required for near_dedup_by_ann_faiss. Install faiss-cpu or faiss-gpu."
        ) from e

    index = faiss.IndexHNSWFlat(d, int(hnsw_m))
    index.hnsw.efConstruction = int(ef_construction)
    index.hnsw.efSearch = int(ef_search)
    index.add(x)

    sims, nbrs = index.search(x, int(topk))

    removed = np.zeros(n, dtype=bool)
    kept: List[int] = []

    for i in range(n):
        if removed[i]:
            continue
        kept.append(i)

        for pos in range(1, topk):
            j = int(nbrs[i, pos])
            if j < 0 or j == i:
                continue
            if j < i:
                continue
            if removed[j]:
                continue

            sim_ij = float(np.dot(x[i], x[j])) if normalize else float(
                np.dot(x[i], x[j]) / (np.linalg.norm(x[i]) * np.linalg.norm(x[j]) + 1e-12)
            )

            if sim_ij >= float(threshold):
                removed[j] = True

    return ANNDedupResult(
        kept_indices=kept,
        removed_mask=removed,
        num_kept=len(kept),
        num_removed=int(removed.sum()),
    )


def near_dedup_and_prune_chunks(
    *,
    s: Dict[str, Any],
    emb: np.ndarray,
    chunks_filename: str = "chunks.jsonl",
    # 允许覆盖/注入（比如你想用 stage 的统一 logger）
    on_read_error: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> ANNDedupResult:
    """
    1) 对 emb 做 near-dedup（FAISS ANN）
    2) 根据 removed_mask，直接从 outputs.chunks.base 下的 chunks.jsonl “删除”对应行
       （实现方式：过滤后覆盖写回 write_jsonl）

    要求：
    - emb 的行顺序 == chunks.jsonl 的行顺序（第 i 个 embedding 对应第 i 行 chunk）
    """
    # ---------- 1) 从 settings 取出 dedup 参数 ----------
    sd = s["processing"]["dedup"]["semantic_dedup"]
    enable = bool(sd.get("enable", False))
    if not enable:
        # 不启用就不改 chunks 文件，直接返回“全保留”
        n = int(emb.shape[0])
        return ANNDedupResult(
            kept_indices=list(range(n)),
            removed_mask=np.zeros((n,), dtype=bool),
            num_kept=n,
            num_removed=0,
        )

    threshold = float(sd.get("threshold", 0.95))
    topk = int(sd.get("topk", 20))
    hnsw_m = int(sd.get("hnsw_m", 32))
    ef_construction = int(sd.get("ef_construction", 200))
    ef_search = int(sd.get("ef_search", 64))
    normalize = bool(sd.get("normalize", True))

    # ---------- 2) 做近重复判定 ----------
    res = near_dedup_by_ann_faiss(
        emb,
        threshold=threshold,
        topk=topk,
        hnsw_m=hnsw_m,
        ef_construction=ef_construction,
        ef_search=ef_search,
        normalize=normalize,
    )

    # ---------- 3) 用 Store + outputs.chunks.base 定位 chunks.jsonl ----------
    stores = build_store_registry(s)
    out_cfg = s["outputs"]["chunks"]
    store_name: str = out_cfg["store"]
    base: str = out_cfg["base"]

    store: Store = stores[store_name]
    chunks_path = _posix_join(base, chunks_filename)

    # ---------- 4) 读取 chunks.jsonl（best-effort / fail-fast 由 on_read_error 决定） ----------
    if on_read_error is None:
        # 如果你希望默认 best-effort，就给一个默认 logger；否则传 None 就是 fail-fast
        on_read_error = _default_log_error_factory(store, chunks_path)

    # ---------- 5) 覆盖写回：过滤掉 removed_mask=True 的行 ----------
    n = int(res.removed_mask.shape[0])

    # 先完整读取，做严格对齐校验；不一致就直接报错并终止（不写回）
    rows: List[Dict[str, Any]] = list(read_jsonl(store, chunks_path, on_error=on_read_error))
    m = len(rows)

    if m != n:
        raise ValueError(
            f"[near_dedup_and_prune_chunks] chunks/emb count mismatch, aborting overwrite: "
            f"chunks_rows={m}, emb_rows={n}. "
            f"Fix upstream alignment (or run fail-fast / disable best-effort) before pruning."
        )

    # 数量一致才允许过滤并覆盖写回
    def _filtered_rows() -> Iterator[Dict[str, Any]]:
        for idx, row in enumerate(rows):
            if not bool(res.removed_mask[idx]):
                yield row

    write_jsonl(store, chunks_path, _filtered_rows())
    return res
