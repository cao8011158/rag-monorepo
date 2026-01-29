from __future__ import annotations

from typing import Any, Dict, List, Sequence, Set, Optional, Callable, Iterable, Iterator
import posixpath
import random
import argparse
import json
from reranker_training.settings import load_settings
from reranker_training.stores.registry import build_store_registry
from reranker_training.io.jsonl import read_jsonl, write_jsonl, append_jsonl


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
    """
    Stable split by query_id.

    - de-dup + sort => deterministic base order
    - shuffle with seeded RNG
    - take first n_valid, then sort => deterministic output set order
    """
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


def build_train_and_valid_query_pack_jsonl_from_pairs(
    settings: Dict[str, Any],
    *,
    seed: int = 42,
    batch_size: int = 512,
    fail_fast: bool = True,
) -> Dict[str, Any]:
    """
    Build BOTH (QueryPack in / QueryPack out; no schema changes):

    - train_query_pack.jsonl: write QueryPack rows as-is, excluding valid_qids
    - valid_query_pack.jsonl: write QueryPack rows as-is, only valid_qids

    Split unit: query.query_id (required for splitting)
      - if missing/invalid:
          - fail_fast=True  -> raise ValueError
          - fail_fast=False -> skip that row

    Input:
      settings["inputs"]["pairs"]: {store, base, pairs}
    Output:
      settings["outputs"]["files"]: {store, base, train_path, valid_path}

    Note:
      - This function does NOT modify any field names or contents.
      - It only routes rows into train/valid based on query_id split.
    """
    # keep same behavior as old code: settings.get("data_split", 0.85) means train_ratio
    train_ratio = float(settings.get("data_split", 0.85))

    in_cfg = settings["inputs"]["pairs"]
    out_cfg = settings["outputs"]["files"]

    in_store_name: str = in_cfg["store"]
    in_base: str = in_cfg.get("base", "")
    pairs_rel: str = in_cfg["pairs"]

    out_store_name: str = out_cfg["store"]
    out_base: str = out_cfg.get("base", "")
    train_rel: str = out_cfg["train_path"]
    valid_rel: str = out_cfg["valid_path"]

    pairs_path = _pjoin(in_base, pairs_rel)
    train_path = _pjoin(out_base, train_rel)
    valid_path = _pjoin(out_base, valid_rel)

    stores = build_store_registry(settings)
    in_store = stores[in_store_name]
    out_store = stores[out_store_name]

    # -------------------------
    # Error collector (only used when fail_fast=False)
    # -------------------------
    skipped_rows = 0
    bad_rows = 0
    jsonl_errors = 0

    def _on_jsonl_error(payload: Dict[str, Any]) -> None:
        # read_jsonl parsing errors
        nonlocal jsonl_errors
        jsonl_errors += 1

    def _fail_or_skip(msg: str) -> None:
        nonlocal bad_rows
        bad_rows += 1
        if fail_fast:
            raise ValueError(msg)

    def _extract_query_id_or_none(row: Dict[str, Any]) -> Optional[str]:
        """
        Validate minimal QueryPack structure enough for split:
          row must be dict
          row["query"] must be dict
          row["query"]["query_id"] must be non-empty str
        If invalid:
          - fail_fast=True  -> raise
          - fail_fast=False -> return None (caller will skip)
        """
        if not isinstance(row, dict):
            _fail_or_skip("QueryPack row must be an object/dict")
            return None

        q = row.get("query")
        if not isinstance(q, dict):
            _fail_or_skip("QueryPack row missing required object field: 'query'")
            return None

        qid = q.get("query_id")
        if qid is None:
            _fail_or_skip("QueryPack row missing required split key: query.query_id")
            return None

        qid_s = str(qid).strip()
        if not qid_s:
            _fail_or_skip("QueryPack row has empty split key: query.query_id")
            return None

        # Optional (but common-sense) quick checks; do NOT modify data:
        # - query_text should exist for training usefulness
        qt = q.get("query_text")
        if qt is None or not str(qt).strip():
            _fail_or_skip("QueryPack row missing required field: query.query_text")
            return None

        return qid_s

    # -------------------------
    # Pass 1: collect qids
    # -------------------------
    all_qids: List[str] = []
    n_input_rows = 0

    on_error = None if fail_fast else _on_jsonl_error
    for row in read_jsonl(in_store, pairs_path, on_error=on_error):
        n_input_rows += 1
        try:
            qid = _extract_query_id_or_none(row)
        except ValueError:
            # fail_fast=True: already raised
            raise
        if not qid:
            skipped_rows += 1
            continue
        all_qids.append(qid)

    # empty/degenerate input => deterministic empty outputs
    if not all_qids:
        write_jsonl(out_store, train_path, [])
        write_jsonl(out_store, valid_path, [])
        return {
            "pairs_path": pairs_path,
            "train_path": train_path,
            "valid_path": valid_path,
            "num_input_rows": n_input_rows,
            "num_unique_qids": 0,
            "num_train_qids": 0,
            "num_train_rows": 0,
            "num_valid_qids": 0,
            "num_valid_rows": 0,
            "seed": seed,
            "train_ratio": train_ratio,
            "batch_size": batch_size,
            "fail_fast": fail_fast,
            "skipped_rows": skipped_rows,
            "bad_rows": bad_rows,
            "jsonl_errors": jsonl_errors,
        }

    valid_qids_list = _stable_sample_qids_for_valid(all_qids, train_ratio=train_ratio, seed=seed)
    valid_qids: Set[str] = set(valid_qids_list)

    # -------------------------
    # Pass 2: streaming write (as-is)
    # -------------------------
    write_jsonl(out_store, train_path, [])
    write_jsonl(out_store, valid_path, [])

    train_buf: List[Dict[str, Any]] = []
    valid_buf: List[Dict[str, Any]] = []

    n_train_rows = 0
    n_valid_rows = 0

    def flush_train() -> None:
        nonlocal train_buf
        if train_buf:
            append_jsonl(out_store, train_path, train_buf)
            train_buf = []

    def flush_valid() -> None:
        nonlocal valid_buf
        if valid_buf:
            append_jsonl(out_store, valid_path, valid_buf)
            valid_buf = []

    on_error = None if fail_fast else _on_jsonl_error
    for row in read_jsonl(in_store, pairs_path, on_error=on_error):
        try:
            qid = _extract_query_id_or_none(row)
        except ValueError:
            raise
        if not qid:
            skipped_rows += 1
            continue

        if qid in valid_qids:
            valid_buf.append(row)  # as-is
            n_valid_rows += 1
            if len(valid_buf) >= batch_size:
                flush_valid()
        else:
            train_buf.append(row)  # as-is
            n_train_rows += 1
            if len(train_buf) >= batch_size:
                flush_train()

    flush_train()
    flush_valid()

    all_unique_qids = set(all_qids)
    train_qids = all_unique_qids - valid_qids

    return {
        "pairs_path": pairs_path,
        "train_path": train_path,
        "valid_path": valid_path,
        "num_input_rows": n_input_rows,
        "num_unique_qids": len(all_unique_qids),
        "num_train_qids": len(train_qids),
        "num_train_rows": n_train_rows,
        "num_valid_qids": len(valid_qids),
        "num_valid_rows": n_valid_rows,
        "seed": seed,
        "train_ratio": train_ratio,
        "batch_size": batch_size,
        "fail_fast": fail_fast,
        "skipped_rows": skipped_rows,
        "bad_rows": bad_rows,
        "jsonl_errors": jsonl_errors,
    }



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--fail_fast", action="store_true")

    args = p.parse_args()

    settings = load_settings(args.config)

    res = build_train_and_valid_query_pack_jsonl_from_pairs(
        settings,
        fail_fast=args.fail_fast,
    )

    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()