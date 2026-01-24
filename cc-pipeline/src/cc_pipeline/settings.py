from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
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
    output_jsonl: str

    # manifest
    manifest_latest: Path
    manifest_run: str

    # seeds
    seeds_path: Path


def load_cfg(path: str) -> Cfg:
    obj = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    # project
    run_date = obj["project"]["run_date"]
    if run_date == "auto":
        run_date = dt.date.today().isoformat()

    # storage
    local_root = Path(obj["storage"]["local_root"])

    # sections
    crawl = obj["crawl"]
    clean = obj["clean"]
    man = obj["manifest"]
    seeds = obj["seeds"]

    return Cfg(
        # project/storage
        run_date=run_date,
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
        output_jsonl=str(clean["output_jsonl"]),

        # manifest
        manifest_latest=Path(man["latest_path"]),
        manifest_run=str(man["run_path"]),

        # seeds
        seeds_path=Path(seeds["path"]),
    )