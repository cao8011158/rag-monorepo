# src/cc_pipeline/settings.py
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml


@dataclass
class Cfg:
    # project/storage
    run_date: str
    local_root: Path

    # crawl basics
    user_agent: str
    timeout_sec: int
    max_retries: int
    per_host_rps: float

    # crawl exploration controls
    max_depth: int
    same_domain_only: bool
    allow_domains: list[str]
    max_links_per_page: int
    max_pages_total: int
    max_pages_per_seed: int
    drop_url_patterns: list[str]

    # clean
    min_text_chars: int
    # IMPORTANT: store as absolute path TEMPLATE under local_root
    # e.g. "/content/.../cleaned/{run_date}/documents.jsonl"
    output_jsonl: str

    # manifest
    manifest_latest: Path      # absolute path
    manifest_run: str          # absolute path template under local_root

    # seeds
    seeds_path: Path           # absolute path


def _as_rooted_path(root: Path, p: str | Path) -> Path:
    """Resolve a possibly-relative path under root."""
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp)


def _as_rooted_template(root: Path, tmpl: str) -> str:
    """
    Resolve a path template under root.
    Example: "cleaned/{run_date}/documents.jsonl" ->
             "/root/cleaned/{run_date}/documents.jsonl"
    """
    # Keep template braces intact; just join as a path
    # If tmpl is absolute already, keep it.
    tp = Path(tmpl)
    if tp.is_absolute():
        return str(tp)
    return str(root / tp)


def load_cfg(path: str) -> Cfg:
    obj: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    # project
    run_date = obj["project"]["run_date"]
    if run_date == "auto":
        run_date = dt.date.today().isoformat()

    # storage
    local_root = Path(obj["storage"]["local_root"])

    # sections
    crawl = obj.get("crawl", {}) or {}
    clean = obj.get("clean", {}) or {}
    man = obj.get("manifest", {}) or {}
    seeds = obj.get("seeds", {}) or {}

    # Normalize all file paths relative to local_root
    output_jsonl = _as_rooted_template(local_root, str(clean["output_jsonl"]))
    manifest_latest = _as_rooted_path(local_root, man["latest_path"])
    manifest_run = _as_rooted_template(local_root, str(man["run_path"]))
    seeds_path = _as_rooted_path(local_root, seeds["path"])

    return Cfg(
        # project/storage
        run_date=str(run_date),
        local_root=local_root,

        # crawl basics
        user_agent=str(crawl["user_agent"]),
        timeout_sec=int(crawl["timeout_sec"]),
        max_retries=int(crawl["max_retries"]),
        per_host_rps=float(crawl["per_host_rps"]),

        # crawl exploration controls
        max_depth=int(crawl.get("max_depth", 0)),
        same_domain_only=bool(crawl.get("same_domain_only", True)),
        allow_domains=list(crawl.get("allow_domains", []) or []),
        max_links_per_page=int(crawl.get("max_links_per_page", 50)),
        max_pages_total=int(crawl.get("max_pages_total", 1000)),
        max_pages_per_seed=int(crawl.get("max_pages_per_seed", 200)),
        drop_url_patterns=list(crawl.get("drop_url_patterns", []) or []),

        # clean
        min_text_chars=int(clean["min_text_chars"]),
        output_jsonl=output_jsonl,

        # manifest
        manifest_latest=manifest_latest,
        manifest_run=manifest_run,

        # seeds
        seeds_path=seeds_path,
    )
