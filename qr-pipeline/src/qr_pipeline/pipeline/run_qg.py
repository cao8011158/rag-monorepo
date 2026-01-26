# 根据config 路径, 将chunks 送入query_generation,  然后根据 存入的    queries_in_domain: queries/in_domain.json 和 queries_out_domain: queries/out_domain.jsonl
# 使用embedder 生成两个 二维矩阵,  送入 near_dedup_by_ann_faiss 获得 •	kept_indices：保留的行号（升序）removed_mask[i]==True：第 i 行应删除 , 
# 根据这个对 in_domain.json out_domain.jsonl  去重, 然后重新写回 queries_in_domain 和 queries_out_domain:
 
# src/qr_pipeline/pipeline/run_qg.py
from __future__ import annotations

import time
from typing import Any, Dict, List

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
    domain_label: str,  # "in" / "out"
) -> Dict[str, Any]:
    """
    读取 file_path -> 抽 query_text_norm -> embed -> ANN cosine dedup -> 覆盖写回 file_path
    返回统计信息
    """
    rows = _load_queries_jsonl(out_store, file_path, errors_path)
    n_total = len(rows)

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

    texts: List[str] = []
    valid_row_indices: List[int] = []
    for i, r in enumerate(rows):
        t = r.get("query_text_norm")
        if isinstance(t, str) and t.strip():
            texts.append(t)
            valid_row_indices.append(i)
        else:
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

    min_text_chars = int(dedup_cfg.get("min_text_chars", 0) or 0)
    if min_text_chars > 0:
        keep_mask_local = [len(t) >= min_text_chars for t in texts]
    else:
        keep_mask_local = [True] * len(texts)

    sub_texts: List[str] = [t for t, ok in zip(texts, keep_mask_local) if ok]
    sub_map: List[int] = [idx for idx, ok in enumerate(keep_mask_local) if ok]  # sub_index -> texts_index

    if len(sub_texts) == 0:
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

    ann = near_dedup_by_ann_faiss(
        emb.astype(np.float32, copy=False),
        threshold=float(dedup_cfg.get("threshold", 0.95)),
        topk=int(dedup_cfg.get("topk", 50)),
        hnsw_m=int(dedup_cfg.get("hnsw_m", 32)),
        ef_construction=int(dedup_cfg.get("ef_construction", 200)),
        ef_search=int(dedup_cfg.get("ef_search", 128)),
        normalize=bool(dedup_cfg.get("normalize", True)),
    )

    kept_rows_mask = [True] * len(rows)
    removed_sub_mask = ann.removed_mask.tolist()

    removed_count = 0
    for sub_i, is_removed in enumerate(removed_sub_mask):
        if not is_removed:
            continue
        texts_i = sub_map[sub_i]
        row_i = valid_row_indices[texts_i]
        kept_rows_mask[row_i] = False
        removed_count += 1

    kept_rows: List[Dict[str, Any]] = [r for i, r in enumerate(rows) if kept_rows_mask[i]]

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
    # -----------------------------
    # overall timing start
    # -----------------------------
    t_total0 = time.perf_counter()
    start_ts_ms = _now_ms()

    # -----------------------------
    # stage: load settings + stores
    # -----------------------------
    t0 = time.perf_counter()
    s = load_settings(config_path)
    stores = build_store_registry(s)
    t_load_settings_and_stores = time.perf_counter() - t0

    # -----------------------------
    # stage: outputs location
    # -----------------------------
    t0 = time.perf_counter()
    out_store_name = s["outputs"]["store"]
    out_base = s["outputs"]["base"]
    out_files = s["outputs"]["files"]

    out_store = stores[out_store_name]

    errors_path = _posix_join(out_base, out_files["errors"])
    in_domain_path = _posix_join(out_base, out_files["queries_in_domain"])
    out_domain_path = _posix_join(out_base, out_files["queries_out_domain"])
    stats_path = _posix_join(out_base, out_files["stats"])
    t_resolve_paths = time.perf_counter() - t0

    # -----------------------------
    # stage: query generation
    # -----------------------------
    t0 = time.perf_counter()
    qg_stats = run_query_generation(s)
    t_query_generation = time.perf_counter() - t0

    # -----------------------------
    # stage: semantic dedup
    # -----------------------------
    t0 = time.perf_counter()
    sd_cfg = _get(s, ["processing", "dedup", "semantic_dedup"], default={}) or {}
    sd_enable = bool(sd_cfg.get("enable", False))

    dedup_stats: Dict[str, Any] = {"enabled": sd_enable, "in_domain": None, "out_domain": None}

    # 细分计时
    t_embedder_init = 0.0
    t_dedup_in = 0.0
    t_dedup_out = 0.0

    if sd_enable:
        # embedder init timing
        t1 = time.perf_counter()
        emb_cfg = s["models"]["embedding"]
        embedder = DualInstructEmbedder(
            model_name=str(emb_cfg["model_name"]),
            passage_instruction=str(emb_cfg["instructions"]["passage"]),
            query_instruction=str(emb_cfg["instructions"]["query"]),
            batch_size=int(emb_cfg.get("batch_size", 64)),
            normalize_embeddings=bool(emb_cfg.get("normalize_embeddings", True)),
            device=emb_cfg.get("device", None),
        )
        t_embedder_init = time.perf_counter() - t1

        # in-domain dedup timing
        t1 = time.perf_counter()
        dedup_stats["in_domain"] = _semantic_dedup_one_file(
            out_store=out_store,
            errors_path=errors_path,
            file_path=in_domain_path,
            embedder=embedder,
            dedup_cfg=sd_cfg,
            domain_label="in",
        )
        t_dedup_in = time.perf_counter() - t1

        # out-domain dedup timing
        t1 = time.perf_counter()
        dedup_stats["out_domain"] = _semantic_dedup_one_file(
            out_store=out_store,
            errors_path=errors_path,
            file_path=out_domain_path,
            embedder=embedder,
            dedup_cfg=sd_cfg,
            domain_label="out",
        )
        t_dedup_out = time.perf_counter() - t1

    t_semantic_dedup_total = time.perf_counter() - t0

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

    # timing end + attach
    end_ts_ms = _now_ms()
    t_total = time.perf_counter() - t_total0

    final_stats["timing"] = {
        "start_ts_ms": start_ts_ms,
        "end_ts_ms": end_ts_ms,
        "total_sec": round(t_total, 3),
        "stages_sec": {
            "load_settings_and_stores": round(t_load_settings_and_stores, 3),
            "resolve_output_paths": round(t_resolve_paths, 3),
            "query_generation": round(t_query_generation, 3),
            "semantic_dedup_total": round(t_semantic_dedup_total, 3),
            "semantic_dedup_breakdown": {
                "embedder_init": round(t_embedder_init, 3),
                "dedup_in_domain": round(t_dedup_in, 3),
                "dedup_out_domain": round(t_dedup_out, 3),
            },
        },
    }

    # 写 stats.json
    t0 = time.perf_counter()
    import json

    out_store.write_text(stats_path, json.dumps(final_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    t_write_stats = time.perf_counter() - t0

    # 把写文件也记录进去（可选）
    final_stats["timing"]["stages_sec"]["write_stats_json"] = round(t_write_stats, 3)

    return final_stats


def main() -> None:
    import argparse
    import json

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
    # timing summary
    # -----------------------------
    timing = stats.get("timing", {}) or {}
    if timing:
        print("[TIMING]")
        print(f"  total_sec:   {timing.get('total_sec')}")
        stages = timing.get("stages_sec", {}) or {}
        if stages:
            print("  stages_sec:")
            # 固定顺序打印（更易读）
            for k in (
                "load_settings_and_stores",
                "resolve_output_paths",
                "query_generation",
                "semantic_dedup_total",
                "write_stats_json",
            ):
                if k in stages:
                    print(f"    - {k}: {stages.get(k)}")
            # breakdown
            b = stages.get("semantic_dedup_breakdown", {}) or {}
            if b:
                print("  semantic_dedup_breakdown:")
                for k in ("embedder_init", "dedup_in_domain", "dedup_out_domain"):
                    if k in b:
                        print(f"    - {k}: {b.get(k)}")
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
    qg_inputs = qg.get("inputs", {}) or {}
    if qg_inputs:
        print("  inputs:")
        for k, v in qg_inputs.items():
            print(f"    - {k}: {v}")
    qg_outputs = qg.get("outputs", {}) or {}
    if qg_outputs:
        print("  outputs:")
        for k, v in qg_outputs.items():
            print(f"    - {k}: {v}")
    qg_counters = qg.get("counters", {}) or {}
    if qg_counters:
        print("  counters:")
        for k, v in qg_counters.items():
            print(f"    - {k}: {v}")
    qg_meta = qg.get("meta", {}) or {}
    if qg_meta:
        print("  meta:")
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
            print("    skipped:     True")
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

