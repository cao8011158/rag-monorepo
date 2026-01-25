from __future__ import annotations

from typing import Any, Dict

from qr_pipeline.pipeline.query_stage import run_query_generation_stage


def run_pipeline(cfg: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"stages": []}

    # Stage 1: query generation (from chunks)
    if bool(cfg.get("query_generation", {}).get("enable", True)):
        res = run_query_generation_stage(cfg)
        summary["stages"].append(res)

    # Placeholders (candidate mining, labeling, export)
    if bool(cfg.get("candidate_mining", {}).get("enable", False)):
        summary["stages"].append({"name": "candidate_mining", "status": "skipped_scaffold"})
    if bool(cfg.get("labeling", {}).get("enable", False)):
        summary["stages"].append({"name": "labeling", "status": "skipped_scaffold"})
    if bool(cfg.get("export", {}).get("enable", True)):
        summary["stages"].append({"name": "export", "status": "skipped_scaffold"})

    summary["status"] = "ok"
    return summary
