# 根据config 路径, 将chunks 送入query_generation,  然后根据 存入的    queries_in_domain: queries/in_domain.json 和 queries_out_domain: queries/out_domain.jsonl
# 使用embedder 生成两个 二维矩阵,  送入 near_dedup_by_ann_faiss 获得 •	kept_indices：保留的行号（升序）removed_mask[i]==True：第 i 行应删除 , 
# 根据这个对 in_domain.json out_domain.jsonl  去重, 然后重新写回 queries_in_domain 和 queries_out_domain:
 
# src/qr_pipeline/pipeline/run.py
from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import numpy as np

from qr_pipeline.settings import load_settings
from qr_pipeline.stores.registry import build_store_registry
from qr_pipeline.io.jsonl import read_jsonl, write_jsonl, append_jsonl

from qr_pipeline.pipeline.query_generation import run_query_generation
from qr_pipeline.processing.embedder import DualInstructEmbedder
from qr_pipeline.processing.near_dedup import near_dedup_by_ann_faiss


# -----------------------------
# helpers
# -----------------------------
def _now_ms() -> int:
    return int(time.time() * 1000)


def _posix_join(*parts: str) -> str:
    return "/".join([p.strip("/").replace("\\", "/") for p in parts if p is not None and str(p) != ""])


def _get(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _log_error(out_store, errors_path: str, payload: Dict[str, Any]) -> None:
    # 统一 append 到 errors.jsonl
    payload = dict(payload)
    payload.setdefault("ts_ms", _now_ms())
    append_jsonl(out_store, errors_path, [payload])


def _load_queries_jsonl(out_store, path: str, errors_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    read_errors = 0

    def on_error(err: Dict[str, Any]) -> None:
        nonlocal read_errors
        read_errors += 1
        _log_error(out_store, errors_path, err)

    for r in read_jsonl(out_store, path, on_error=on_error):
        if isinstance(r, dict):
            rows.append(r)
        else:
            read_errors += 1
            _log_error(
                out_store,
                errors_path,
                {
                    "stage": "semantic_dedup.read_queries",
                    "path": path,
                    "error": f"Non-dict row: {type(r)}",
                },
            )

    return rows


def _semantic_dedup_one_file(
    *,
    out_store,
    errors_path: str,
    file_path: str,
    embedder: DualInstructEmbedder,
    dedup_cfg: Dict[str, Any],
    domain_label: str,  # "in" / "out" just for logging
) -> Dict[str, Any]:
    """
    读取 file_path -> 抽 query_text_norm -> embed -> ANN cosine dedup -> 覆盖写回 file_path
    返回统计信息
    """
    rows = _load_queries_jsonl(out_store, file_path, errors_path)
    n_total = len(rows)

    # 空文件直接返回
    if n_total == 0:
        return {
            "domain": domain_label,
            "file_path": file_path,
            "num_total": 0,
            "num_kept": 0,
            "num_removed": 0,
            "skipped": True,
            "reason": "empty_file",
        }

    # 抽取文本
    texts: List[str] = []
    valid_row_indices: List[int] = []
    for i, r in enumerate(rows):
        t = r.get("query_text_norm")
        if isinstance(t, str) and t.strip():
            texts.append(t)
            valid_row_indices.append(i)
        else:
            # 缺字段：记录错误，但不参与语义去重
            _log_error(
                out_store,
                errors_path,
                {
                    "stage": "semantic_dedup.missing_query_text_norm",
                    "path": file_path,
                    "row_index": i,
                    "row_preview": {k: r.get(k) for k in ("query_id", "domain", "query_text_norm")},
                },
            )

    if len(texts) == 0:
        return {
            "domain": domain_label,
            "file_path": file_path,
            "num_total": n_total,
            "num_kept": 0,
            "num_removed": 0,
            "skipped": True,
            "reason": "no_valid_query_text_norm",
        }

    # 如果太短的 query 你想跳过语义去重（按你的 API：min_text_chars）
    min_text_chars = int(dedup_cfg.get("min_text_chars", 0) or 0)
    if min_text_chars > 0:
        keep_mask_local = [len(t) >= min_text_chars for t in texts]
    else:
        keep_mask_local = [True] * len(texts)

    # 只对满足 min_text_chars 的子集做语义去重
    sub_texts: List[str] = [t for t, ok in zip(texts, keep_mask_local) if ok]
    sub_map: List[int] = [idx for idx, ok in enumerate(keep_mask_local) if ok]  # sub_index -> texts_index

    if len(sub_texts) == 0:
        # 全部都被 min_text_chars 跳过：直接不改文件
        return {
            "domain": domain_label,
            "file_path": file_path,
            "num_total": n_total,
            "num_kept": n_total,
            "num_removed": 0,
            "skipped": True,
            "reason": f"all_queries_shorter_than_{min_text_chars}",
        }

    # embed
    try:
        emb = embedder.encode_queries(sub_texts)  # shape [M, D]
    except Exception as e:
        _log_error(
            out_store,
            errors_path,
            {
                "stage": "semantic_dedup.embed_queries_failed",
                "path": file_path,
                "error": f"{type(e).__name__}: {e}",
            },
        )
        # embed 失败就不做去重，直接返回
        return {
            "domain": domain_label,
            "file_path": file_path,
            "num_total": n_total,
            "num_kept": n_total,
            "num_removed": 0,
            "skipped": True,
            "reason": "embed_failed",
        }

    if not isinstance(emb, np.ndarray) or emb.ndim != 2 or emb.shape[0] != len(sub_texts):
        _log_error(
            out_store,
            errors_path,
            {
                "stage": "semantic_dedup.embed_shape_invalid",
                "path": file_path,
                "error": f"Expected emb shape [M,D] with M={len(sub_texts)}, got {getattr(emb, 'shape', None)}",
            },
        )
        return {
            "domain": domain_label,
            "file_path": file_path,
            "num_total": n_total,
            "num_kept": n_total,
            "num_removed": 0,
            "skipped": True,
            "reason": "embed_shape_invalid",
        }

    # ANN + cosine verify
    ann = near_dedup_by_ann_faiss(
        emb.astype(np.float32, copy=False),
        threshold=float(dedup_cfg.get("threshold", 0.95)),
        topk=int(dedup_cfg.get("topk", 50)),
        hnsw_m=int(dedup_cfg.get("hnsw_m", 32)),
        ef_construction=int(dedup_cfg.get("ef_construction", 200)),
        ef_search=int(dedup_cfg.get("ef_search", 128)),
        normalize=bool(dedup_cfg.get("normalize", True)),
    )

    # ann.kept_indices 是 sub_texts 的行号，需要映射回 rows 的行号
    kept_rows_mask = [True] * len(rows)  # 默认都保留
    # 先把 sub_texts 中被删的标记出来
    removed_sub_mask = ann.removed_mask.tolist()  # len == M

    # 对参与去重的 query，如果被判重复，则删掉对应 rows 行
    removed_count = 0
    for sub_i, is_removed in enumerate(removed_sub_mask):
        if not is_removed:
            continue
        texts_i = sub_map[sub_i]                 # in texts[]
        row_i = valid_row_indices[texts_i]       # in rows[]
        kept_rows_mask[row_i] = False
        removed_count += 1

    kept_rows: List[Dict[str, Any]] = [r for i, r in enumerate(rows) if kept_rows_mask[i]]

    # 覆盖写回
    write_jsonl(out_store, file_path, kept_rows)

    return {
        "domain": domain_label,
        "file_path": file_path,
        "num_total": n_total,
        "num_kept": len(kept_rows),
        "num_removed": removed_count,
        "ann": {
            "threshold": float(dedup_cfg.get("threshold", 0.95)),
            "topk": int(dedup_cfg.get("topk", 50)),
            "hnsw_m": int(dedup_cfg.get("hnsw_m", 32)),
            "ef_construction": int(dedup_cfg.get("ef_construction", 200)),
            "ef_search": int(dedup_cfg.get("ef_search", 128)),
            "normalize": bool(dedup_cfg.get("normalize", True)),
            "min_text_chars": min_text_chars,
        },
    }


def run_pipeline(config_path: str) -> Dict[str, Any]:
    s = load_settings(config_path)
    stores = build_store_registry(s)

    # -----------------------------
    # outputs location
    # -----------------------------
    out_store_name = s["outputs"]["store"]
    out_base = s["outputs"]["base"]
    out_files = s["outputs"]["files"]

    out_store = stores[out_store_name]

    errors_path = _posix_join(out_base, out_files["errors"])
    in_domain_path = _posix_join(out_base, out_files["queries_in_domain"])
    out_domain_path = _posix_join(out_base, out_files["queries_out_domain"])
    stats_path = _posix_join(out_base, out_files["stats"])

    # -----------------------------
    # step 1) query generation
    # -----------------------------
    qg_stats = run_query_generation(s)

    # -----------------------------
    # step 2) semantic dedup (ANN) for queries
    # -----------------------------
    sd_cfg = _get(s, ["processing", "dedup", "semantic_dedup"], default={}) or {}
    sd_enable = bool(sd_cfg.get("enable", False))

    dedup_stats: Dict[str, Any] = {"enabled": sd_enable, "in_domain": None, "out_domain": None}

    if sd_enable:
        emb_cfg = s["models"]["embedding"]
        embedder = DualInstructEmbedder(
            model_name=str(emb_cfg["model_name"]),
            passage_instruction=str(emb_cfg["instructions"]["passage"]),
            query_instruction=str(emb_cfg["instructions"]["query"]),
            batch_size=int(emb_cfg.get("batch_size", 64)),
            normalize_embeddings=bool(emb_cfg.get("normalize_embeddings", True)),
            device=emb_cfg.get("device", None),
        )

        dedup_stats["in_domain"] = _semantic_dedup_one_file(
            out_store=out_store,
            errors_path=errors_path,
            file_path=in_domain_path,
            embedder=embedder,
            dedup_cfg=sd_cfg,
            domain_label="in",
        )
        dedup_stats["out_domain"] = _semantic_dedup_one_file(
            out_store=out_store,
            errors_path=errors_path,
            file_path=out_domain_path,
            embedder=embedder,
            dedup_cfg=sd_cfg,
            domain_label="out",
        )

    # -----------------------------
    # write merged stats
    # -----------------------------
    final_stats: Dict[str, Any] = {
        "ts_ms": _now_ms(),
        "config_path": config_path,
        "query_generation": qg_stats,
        "semantic_dedup": dedup_stats,
        "outputs": {
            "queries_in_domain_path": in_domain_path,
            "queries_out_domain_path": out_domain_path,
            "errors_path": errors_path,
            "stats_path": stats_path,
        },
    }

    # stats 写 json 文本（你现有 write_text 也行，这里用标准库简单写）
    import json

    out_store.write_text(stats_path, json.dumps(final_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return final_stats


def main() -> None:
    import argparse
    import json
    from pprint import pformat

    parser = argparse.ArgumentParser(
        description="qr-pipeline stage: query_generation -> (optional) semantic dedup by ANN"
    )
    parser.add_argument("--config", required=True, help="Path to pipeline.yaml, e.g. configs/pipeline.yaml")
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Also print the full stats dict as pretty JSON (can be very long).",
    )
    args = parser.parse_args()

    stats = run_pipeline(args.config)

    def _j(x: object) -> str:
        return json.dumps(x, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("QR PIPELINE: RUN_QG")
    print("=" * 100)
    print(f"config_path: {stats.get('config_path')}")
    print(f"ts_ms:       {stats.get('ts_ms')}")
    print("-" * 100)

    # -----------------------------
    # outputs
    # -----------------------------
    outputs = stats.get("outputs", {}) or {}
    print("[OUTPUTS]")
    for k in ("queries_in_domain_path", "queries_out_domain_path", "errors_path", "stats_path"):
        if k in outputs:
            print(f"  {k}: {outputs[k]}")
    print("-" * 100)

    # -----------------------------
    # query generation stats
    # -----------------------------
    qg = stats.get("query_generation", {}) or {}
    print("[QUERY_GENERATION]")
    if "ts_ms" in qg:
        print(f"  ts_ms: {qg.get('ts_ms')}")
    # inputs
    qg_inputs = qg.get("inputs", {}) or {}
    if qg_inputs:
        print("  inputs:")
        for k, v in qg_inputs.items():
            print(f"    - {k}: {v}")
    # outputs
    qg_outputs = qg.get("outputs", {}) or {}
    if qg_outputs:
        print("  outputs:")
        for k, v in qg_outputs.items():
            print(f"    - {k}: {v}")
    # counters
    qg_counters = qg.get("counters", {}) or {}
    if qg_counters:
        print("  counters:")
        for k, v in qg_counters.items():
            print(f"    - {k}: {v}")
    # meta
    qg_meta = qg.get("meta", {}) or {}
    if qg_meta:
        print("  meta:")
        # meta 可能嵌套较深，直接 pretty 一下
        print(_j(qg_meta))
    print("-" * 100)

    # -----------------------------
    # semantic dedup stats
    # -----------------------------
    sd = stats.get("semantic_dedup", {}) or {}
    print("[SEMANTIC_DEDUP]")
    enabled = bool(sd.get("enabled", False))
    print(f"  enabled: {enabled}")

    def _print_one(label: str, d: dict | None) -> tuple[int, int]:
        if not d:
            print(f"  [{label}] <none>")
            return 0, 0

        print(f"  [{label}]")
        print(f"    file_path:   {d.get('file_path')}")
        print(f"    num_total:   {d.get('num_total')}")
        print(f"    num_kept:    {d.get('num_kept')}")
        print(f"    num_removed: {d.get('num_removed')}")
        if d.get("skipped"):
            print(f"    skipped:     True")
            print(f"    reason:      {d.get('reason')}")
        ann = d.get("ann")
        if isinstance(ann, dict):
            print("    ann_params:")
            for k, v in ann.items():
                print(f"      - {k}: {v}")

        return int(d.get("num_kept") or 0), int(d.get("num_removed") or 0)

    total_kept = 0  
    total_removed = 0
    if enabled:
        k1, r1 = _print_one("IN_DOMAIN", sd.get("in_domain"))
        k2, r2 = _print_one("OUT_DOMAIN", sd.get("out_domain"))
        total_kept = k1 + k2
        total_removed = r1 + r2

        print("  [SUMMARY]")
        print(f"    total_kept:    {total_kept}")
        print(f"    total_removed: {total_removed}")
    print("-" * 100)

    # -----------------------------
    # optionally print full stats json
    # -----------------------------
    if args.print_json:
        print("[FULL_STATS_JSON]")
        print(_j(stats))
        print("-" * 100)

    print("DONE")
    print("=" * 100 + "\n")



if __name__ == "__main__":
    main()
