from __future__ import annotations

import argparse
import random
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from reranker_training.settings import load_settings
from reranker_training.stores.registry import build_store_registry
from reranker_training.io.jsonl import read_jsonl, write_jsonl


# -------------------------
# path helpers
# -------------------------

def _get_store_and_base(stores: Dict[str, Any], cfg: Dict[str, Any]):
    store = stores[cfg["store"]]
    base = str(cfg["base"]).rstrip("/")
    return store, base


def _join(base: str, name: str) -> str:
    name = str(name).lstrip("/")
    return f"{base}/{name}" if base else name


def _fail_or_skip(msg: str, fail_fast: bool) -> None:
    if fail_fast:
        raise ValueError(msg)
    print(f"[SKIP] {msg}")


# -------------------------
# chunks pool (FULL ChunkDoc)
# -------------------------

def _load_chunk_pool(
    chunks_store: Any,
    chunks_path: str,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """
    Load chunks.jsonl once and build:
    - all_chunk_ids: List[str]
    - chunk_map: Dict[chunk_id, ChunkDoc]
    """
    all_ids: List[str] = []
    chunk_map: Dict[str, Dict[str, Any]] = {}

    for row in read_jsonl(chunks_store, chunks_path):
        if not isinstance(row, dict):
            continue
        cid = row.get("chunk_id")
        if not cid or not isinstance(cid, str):
            continue
        # keep first occurrence (dedup by chunk_id)
        if cid in chunk_map:
            continue
        chunk_map[cid] = row
        all_ids.append(cid)

    return all_ids, chunk_map


# -------------------------
# core logic
# -------------------------

def build_epoch_pairs(
    *,
    query_packs: Iterable[Dict[str, Any]],
    all_chunk_ids: List[str],
    chunk_map: Dict[str, Dict[str, Any]],
    epoch: int,
    seed: int,
    hard_negative_per_positive: int,
    random_negative_per_positive: int,
    fail_fast: bool,
    random_neg_max_tries: int = 50,
) -> List[Dict[str, Any]]:
    """
    Expand QueryPack into pairwise rows:
    { "query": Query, "positive": ChunkDoc, "negative": ChunkDoc, "meta": {...} }

    Notes:
    - positives/hard negatives are already full ChunkDoc inside QueryPack
    - random negatives are sampled from chunks.jsonl pool, and stored as full ChunkDoc too
    """
    rnd = random.Random(seed + epoch)

    output_rows: List[Dict[str, Any]] = []

    for qp_idx, qp in enumerate(query_packs):
        if not isinstance(qp, dict):
            _fail_or_skip(f"query_pack[{qp_idx}] is not a dict", fail_fast)
            continue

        query = qp.get("query")
        positives: List[Dict[str, Any]] = qp.get("positives") or []
        negatives: List[Dict[str, Any]] = qp.get("negatives") or []

        if not isinstance(query, dict):
            _fail_or_skip(f"query_pack[{qp_idx}] missing/invalid query", fail_fast)
            continue

        if not positives or not negatives:
            _fail_or_skip(
                f"query_pack[{qp_idx}] has empty positives or negatives",
                fail_fast,
            )
            continue

        # Required by your schema: query_text must exist
        qtext = query.get("query_text")
        if not isinstance(qtext, str) or not qtext.strip():
            _fail_or_skip(f"query_pack[{qp_idx}] query.query_text is empty", fail_fast)
            continue

        source_chunk_ids: Set[str] = set(query.get("source_chunk_ids") or [])

        # Collect ALL positive/negative ids for exclusion (not just sampled ones)
        positive_ids: Set[str] = set()
        for p in positives:
            cid = p.get("chunk_id") if isinstance(p, dict) else None
            if isinstance(cid, str) and cid:
                positive_ids.add(cid)

        negative_ids_all: Set[str] = set()
        for n in negatives:
            cid = n.get("chunk_id") if isinstance(n, dict) else None
            if isinstance(cid, str) and cid:
                negative_ids_all.add(cid)

        p_cnt = len(positives)
        h_n = len(negatives)

        # hard negative count per positive (your rule)
        if h_n < p_cnt:
            hard_k = 1
            allow_repeat = True
        else:
            hard_k = min(hard_negative_per_positive, h_n // p_cnt)
            allow_repeat = False

        if hard_k <= 0:
            _fail_or_skip(
                f"query_pack[{qp_idx}] hard_k=0 (h_n={h_n}, p={p_cnt})",
                fail_fast,
            )
            continue

        # For each positive, generate (q, d+, d-)
        for pos in positives:
            if not isinstance(pos, dict) or not isinstance(pos.get("chunk_id"), str) or not pos.get("chunk_id"):
                _fail_or_skip(f"query_pack[{qp_idx}] has invalid positive", fail_fast)
                continue

            # --- sample hard negatives ---
            if allow_repeat:
                hard_negs = [rnd.choice(negatives) for _ in range(hard_k)]
            else:
                hard_negs = rnd.sample(negatives, hard_k)

            hard_neg_ids: Set[str] = set()
            hard_negs_full: List[Dict[str, Any]] = []
            for hn in hard_negs:
                if isinstance(hn, dict) and isinstance(hn.get("chunk_id"), str) and hn.get("chunk_id"):
                    hard_negs_full.append(hn)
                    hard_neg_ids.add(hn["chunk_id"])

            if not hard_negs_full:
                _fail_or_skip(f"query_pack[{qp_idx}] sampled hard negatives are invalid", fail_fast)
                continue

            # exclusion set for random negative:
            # - source chunks
            # - ALL positives for this query
            # - ALL negatives pool for this query
            # - current positive + sampled hard negatives (redundant but explicit)
            exclude_ids: Set[str] = set(source_chunk_ids)
            exclude_ids.update(positive_ids)
            exclude_ids.update(negative_ids_all)
            exclude_ids.add(pos["chunk_id"])
            exclude_ids.update(hard_neg_ids)

            # --- sample random negatives as FULL ChunkDoc ---
            random_negs_full: List[Dict[str, Any]] = []
            for _ in range(random_negative_per_positive):
                chosen_doc: Optional[Dict[str, Any]] = None
                for _try in range(random_neg_max_tries):
                    cid = rnd.choice(all_chunk_ids)
                    if cid in exclude_ids:
                        continue
                    doc = chunk_map.get(cid)
                    if not isinstance(doc, dict):
                        continue
                    # ensure it has chunk_id and chunk_text at least (ChunkDoc required fields)
                    if not isinstance(doc.get("chunk_id"), str) or not doc.get("chunk_id"):
                        continue
                    if not isinstance(doc.get("chunk_text"), str) or not doc.get("chunk_text"):
                        continue
                    chosen_doc = doc
                    break
                if chosen_doc is not None:
                    random_negs_full.append(chosen_doc)

            # --- build rows ---
            for neg in hard_negs_full:
                output_rows.append(
                    {
                        "query": query,
                        "positive": pos,
                        "negative": neg,
                        "meta": {
                            "epoch": epoch,
                            "type": "hard_negative",
                        },
                    }
                )

            for neg in random_negs_full:
                output_rows.append(
                    {
                        "query": query,
                        "positive": pos,
                        "negative": neg,  # ✅ full ChunkDoc now
                        "meta": {
                            "epoch": epoch,
                            "type": "random_negative",
                        },
                    }
                )

    return output_rows


# -------------------------
# main entry
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fail_fast", action="store_true")
    args = parser.parse_args()

    s = load_settings(args.config)
    stores = build_store_registry(s)

    fail_fast = bool(args.fail_fast)

    # ---- input: train_query_pack.jsonl (from outputs.files.train_path) ----
    out_files_cfg = s["outputs"]["files"]
    qp_store, qp_base = _get_store_and_base(stores, out_files_cfg)
    query_pack_path = _join(qp_base, out_files_cfg["train_path"])

    # ---- input: chunks.jsonl pool ----
    chunks_cfg = s["inputs"]["chunks"]
    chunks_store, chunks_base = _get_store_and_base(stores, chunks_cfg)
    chunks_path = _join(chunks_base, chunks_cfg["chunks_file"])

    # ---- training knobs ----
    train_cfg = s["training"]
    num_epochs = int(train_cfg["num_epochs"])
    seed = int(train_cfg["seed"])
    hard_negative_per_positive = int(train_cfg["hard_negative_per_positive"])
    random_negative_per_positive = int(train_cfg["random_negative_per_positive"])

    print("Loading query packs...")
    query_packs = list(read_jsonl(qp_store, query_pack_path))
    print(f"Total query packs: {len(query_packs)}")

    print("Loading chunks pool (full ChunkDoc)...")
    all_chunk_ids, chunk_map = _load_chunk_pool(chunks_store, chunks_path)
    if not all_chunk_ids:
        raise RuntimeError("chunks.jsonl is empty or no valid chunk_id found")
    print(f"Total chunks in pool: {len(all_chunk_ids)}")

    # ---- generate per epoch ----
    out_store, out_base = _get_store_and_base(stores, out_files_cfg)

    for epoch in range(1, num_epochs + 1):
        print(f"=== Epoch {epoch} ===")

        rows = build_epoch_pairs(
            query_packs=query_packs,
            all_chunk_ids=all_chunk_ids,
            chunk_map=chunk_map,
            epoch=epoch,
            seed=seed,
            hard_negative_per_positive=hard_negative_per_positive,
            random_negative_per_positive=random_negative_per_positive,
            fail_fast=fail_fast,
            random_neg_max_tries=50,
        )

        out_path = _join(out_base, f"processed/train_pair_epoch_{epoch}.jsonl")
        write_jsonl(out_store, out_path, rows)
        print(f"Written {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
