from __future__ import annotations

from typing import Any, Dict, List, Sequence, Set, Optional
import posixpath
import random

from reranker_training.stores.registry import build_store_registry
from reranker_training.io.jsonl import read_jsonl, append_jsonl, write_jsonl


def _pjoin(*parts: str) -> str:
    parts = [p for p in parts if p]
    if not parts:
        return ""
    return posixpath.normpath(posixpath.join(*parts))


def _stable_sample_qids_for_valid(
    qids: Sequence[str],
    *,
    train_ratio: float,
    seed: int,
) -> List[str]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")

    qids_sorted = sorted(set(qids))
    n_total = len(qids_sorted)
    n_valid = int(round(n_total * (1.0 - train_ratio)))

    if n_total > 0 and n_valid == 0:
        n_valid = 1

    rng = random.Random(seed)
    rng.shuffle(qids_sorted)
    return sorted(qids_sorted[:n_valid])


def build_train_and_fixed_valid_jsonl_from_candidates(
    settings: Dict[str, Any],
    *,
    seed: int = 42,
    batch_size: int = 512,
) -> Dict[str, Any]:
    """
    Build BOTH:
    - train.jsonl: original candidates rows (NOT expanded), excluding valid_qids
    - valid.jsonl: expanded 1-1-1 rows, for valid_qids only

    Requirements from your spec:
    - split unit: query_id
    - train_path must exist in config (else raise)
    - train rows keep original schema identical to candidates.jsonl
    - valid rows expanded per negative
    - write stats including train + valid
    """

    train_ratio = float(settings.get("data_split", 0.85))

    files_cfg = settings["inputs"]["files"]
    store_name: str = files_cfg["store"]
    base: str = files_cfg.get("base", "")

    candidates_rel: str = files_cfg["candidates"]
    valid_rel: str = files_cfg["valid_path"]

    # 强制要求 train_path 存在
    if "train_path" not in files_cfg:
        raise KeyError('Missing required config: settings["inputs"]["files"]["train_path"]')
    train_rel: str = files_cfg["train_path"]

    candidates_path = _pjoin(base, candidates_rel)
    valid_path = _pjoin(base, valid_rel)
    train_path = _pjoin(base, train_rel)

    stores = build_store_registry(settings)
    store = stores[store_name]

    # -------------------------
    # Pass 1: collect qids
    # -------------------------
    all_qids: List[str] = []
    n_candidate_rows = 0

    for row in read_jsonl(store, candidates_path):
        n_candidate_rows += 1
        qid = str(row.get("query_id", "")).strip()
        if qid:
            all_qids.append(qid)

    # empty input => write empty outputs deterministically
    if not all_qids:
        write_jsonl(store, train_path, [])
        write_jsonl(store, valid_path, [])
        return {
            "candidates_path": candidates_path,
            "train_path": train_path,
            "valid_path": valid_path,
            "num_candidate_rows": n_candidate_rows,
            "num_unique_qids": 0,
            "num_train_qids": 0,
            "num_train_rows": 0,
            "num_valid_qids": 0,
            "num_valid_candidate_rows": 0,
            "num_valid_rows": 0,
            "seed": seed,
            "train_ratio": train_ratio,
            "batch_size": batch_size,
        }

    valid_qids_list = _stable_sample_qids_for_valid(all_qids, train_ratio=train_ratio, seed=seed)
    valid_qids: Set[str] = set(valid_qids_list)

    # -------------------------
    # Pass 2: streaming write
    # -------------------------
    # 为了“覆盖式”输出：先清空文件（写空列表），后续用 append_jsonl 分批追加
    write_jsonl(store, train_path, [])
    write_jsonl(store, valid_path, [])

    train_buf: List[Dict[str, Any]] = []
    valid_buf: List[Dict[str, Any]] = []

    n_train_rows = 0
    n_valid_candidate_rows = 0
    n_valid_rows = 0

    def flush_train() -> None:
        nonlocal train_buf
        if train_buf:
            append_jsonl(store, train_path, train_buf)
            train_buf = []

    def flush_valid() -> None:
        nonlocal valid_buf
        if valid_buf:
            append_jsonl(store, valid_path, valid_buf)
            valid_buf = []

    for row in read_jsonl(store, candidates_path):
        qid = str(row.get("query_id", "")).strip()
        if not qid:
            continue

        if qid not in valid_qids:
            # TRAIN: 原样写入（不展平）
            train_buf.append(row)
            n_train_rows += 1
            if len(train_buf) >= batch_size:
                flush_train()
            continue

        # VALID: 展平
        n_valid_candidate_rows += 1

        qtext = str(row.get("query_text", "") or "")
        pos = row.get("positive") or {}
        negs = row.get("negatives") or []
        source_chunk = str(row.get("source_chunk", "") or "")
        meta = row.get("meta") or {}

        # basic guards
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

            if not neg_doc_id or not neg_text:
                continue
            if neg_doc_id == pos_doc_id:
                continue

            valid_buf.append(
                {
                    "query_id": qid,
                    "query_text": qtext,
                    "positive": {"doc_id": pos_doc_id, "text": pos_text},
                    "negative": {"doc_id": neg_doc_id, "text": str(neg_text)},
                    "source_chunk": source_chunk,
                    "meta": meta,
                    "meta_pair": {"neg_rank_in_row": i},
                }
            )
            n_valid_rows += 1

            if len(valid_buf) >= batch_size:
                flush_valid()

    flush_train()
    flush_valid()

    # 统计 train unique qids（只要你想要这个统计；这里用一次 pass 2 的“缓存集合”会更快，
    # 但为了不再扫一遍文件，我们可以 pass 2 顺便维护集合）
    # 这里简单起见，用 set(all_qids) - valid_qids 作为 train_qids 统计（按 query_id split 是成立的）
    all_unique_qids = set(all_qids)
    train_qids = all_unique_qids - valid_qids

    return {
        "candidates_path": candidates_path,
        "train_path": train_path,
        "valid_path": valid_path,
        "num_candidate_rows": n_candidate_rows,
        "num_unique_qids": len(all_unique_qids),
        "num_train_qids": len(train_qids),
        "num_train_rows": n_train_rows,  # not expanded
        "num_valid_qids": len(valid_qids),
        "num_valid_candidate_rows": n_valid_candidate_rows,
        "num_valid_rows": n_valid_rows,  # expanded
        "seed": seed,
        "train_ratio": train_ratio,
        "batch_size": batch_size,
    }
