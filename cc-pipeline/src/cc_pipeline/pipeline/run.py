# src/cc_pipeline/pipeline/run.py
from __future__ import annotations

import argparse
import datetime as dt
from collections import deque
from pathlib import Path
from urllib.parse import urlparse

import yaml

from cc_pipeline.settings import load_cfg
from cc_pipeline.common.hashing import sha256_hex, stable_url_hash
from cc_pipeline.common.io import LocalStore
from cc_pipeline.crawl.fetcher import Fetcher
from cc_pipeline.crawl.manifest import ManifestEntry, load_manifest, write_manifest
from cc_pipeline.clean.html_cleaner import html_to_text
from cc_pipeline.clean.pdf_cleaner import pdf_to_text
from cc_pipeline.clean.writer import append_jsonl

from cc_pipeline.crawl.url_utils import (
    canonicalize_url,
    is_http_url,
    should_drop,
)
from cc_pipeline.crawl.link_extractor import extract_links


def _load_seed_urls(seeds_path: Path) -> list[str]:
    obj = yaml.safe_load(seeds_path.read_text(encoding="utf-8")) or {}
    urls: list[str] = []
    for block in obj.get("seeds", []):
        urls.extend(block.get("urls", []))

    # de-duplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def _ensure_abs_under_root(root: Path, p: Path) -> Path:
    """If p is relative, treat it as under root."""
    return p if p.is_absolute() else (root / p)


def run_pipeline(config_path: str, mode: str) -> None:
    cfg = load_cfg(config_path)
    store = LocalStore(cfg.local_root)

    # Paths: MUST be under local_root (avoid cwd dependence)
    latest_path: Path = _ensure_abs_under_root(cfg.local_root, cfg.manifest_latest)
    run_manifest_path: Path = _ensure_abs_under_root(
        cfg.local_root, Path(cfg.manifest_run.format(run_date=cfg.run_date))
    )
    seeds_path = Path("/content/rag-monorepo/cc-pipeline/configs/seeds.yaml")
    if not seeds_path.exists():
        raise FileNotFoundError(f"[SEEDS NOT FOUND] {seeds_path}")

    # Output (cleaned) path — cfg.output_jsonl is a template string
    out_path: Path = _ensure_abs_under_root(
        cfg.local_root, Path(cfg.output_jsonl.format(run_date=cfg.run_date))
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Debug: print resolved paths (safe)
    print(f"[DEBUG] config={config_path}")
    print(f"[DEBUG] local_root={cfg.local_root}")
    print(f"[DEBUG] seeds_path={seeds_path}")
    print(f"[DEBUG] manifest_latest={latest_path}")
    print(f"[DEBUG] manifest_run={run_manifest_path}")
    print(f"[DEBUG] out_path={out_path}")

    # Load old (latest) snapshot if exists
    old = load_manifest(latest_path)
    fetcher = Fetcher(cfg.user_agent, cfg.timeout_sec, cfg.per_host_rps)
    seeds = _load_seed_urls(seeds_path)

    # =========================
    # 1) CRAWL
    # =========================
    if mode in ("crawl", "run"):
        total_fetched = 0

        # latest snapshot for THIS run
        new_by_url: dict[str, ManifestEntry] = {}

        # latest.jsonl represents "latest run snapshot" (your requirement)
        # For this run, we start from empty; no carry-over.
        # (If you want reuse for unchanged URLs in the same run, we still can read old for conditional reuse,
        # but we won't keep entries for URLs that we never crawl this run.)
        #
        # However, to reduce redundant downloads, we can reuse old entry IF:
        # - URL is fetched this run
        # - hash unchanged vs old
        #
        # So we keep old as a lookup table, but snapshot = only-crawled-this-run.

        for seed_url in seeds:
            seed_domain = urlparse(seed_url).netloc.lower()

            fetched_for_seed = 0
            q: deque[tuple[str, int]] = deque()
            discovered: set[str] = set()
            fetched: set[str] = set()

            allowset = {d.lower() for d in (cfg.allow_domains or [])}

            def allowed(u: str) -> bool:
                if not is_http_url(u):
                    return False
                if should_drop(u, cfg.drop_url_patterns):
                    return False

                if allowset:
                    return urlparse(u).netloc.lower() in allowset

                if cfg.same_domain_only:
                    return urlparse(u).netloc.lower() == seed_domain

                return True

            def push(u: str, depth: int) -> None:
                cu = canonicalize_url(u)
                if not cu:
                    return
                if not allowed(cu):
                    return
                if cu in discovered:
                    return
                discovered.add(cu)
                q.append((cu, depth))

            push(seed_url, 0)

            while q:
                if total_fetched >= cfg.max_pages_total:
                    break
                if fetched_for_seed >= cfg.max_pages_per_seed:
                    break

                url, depth = q.popleft()
                url_key = canonicalize_url(url)
                if not url_key:
                    continue
                if url_key in fetched:
                    continue

                try:
                    resp = fetcher.get(url)
                except Exception as e:
                    print(f"[FETCH FAIL] {url} -> {e}")
                    fetched.add(url_key)
                    continue

                content = resp.content
                content_hash = sha256_hex(content)

                ctype = (resp.headers.get("Content-Type") or "").lower()
                is_pdf = ("application/pdf" in ctype) or url.lower().endswith(".pdf")
                ext = "pdf" if is_pdf else "html"

                fetched.add(url_key)
                total_fetched += 1
                fetched_for_seed += 1

                # If unchanged vs old, reuse old rel_path (no rewrite needed)
                if url in old and old[url].content_hash == content_hash:
                    entry = old[url]
                    # But ensure content_type matches our supported set
                    new_by_url[url] = ManifestEntry(
                        url=url,
                        content_hash=entry.content_hash,
                        rel_path=entry.rel_path,
                        content_type=entry.content_type,
                        fetched_at=dt.datetime.utcnow().isoformat() + "Z",
                    )
                else:
                    url_hash = stable_url_hash(url)
                    rel = f"raw/{ext}/{cfg.run_date}/crawl/{url_hash}.{ext}"
                    store.write_bytes(rel, content)

                    new_by_url[url] = ManifestEntry(
                        url=url,
                        content_hash=content_hash,
                        rel_path=rel,
                        content_type="application/pdf" if is_pdf else "text/html",
                        fetched_at=dt.datetime.utcnow().isoformat() + "Z",
                    )

                # Expand only HTML and depth < max_depth
                if (not is_pdf) and depth < cfg.max_depth:
                    html = content.decode("utf-8", errors="ignore")
                    links = extract_links(html, base_url=url)

                    cnt = 0
                    for link in links:
                        if cnt >= cfg.max_links_per_page:
                            break
                        before = len(discovered)
                        push(link, depth + 1)
                        after = len(discovered)
                        if after > before:
                            cnt += 1

        # Write run manifest + latest manifest as "this run snapshot"
        new_entries = list(new_by_url.values())
        run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        latest_path.parent.mkdir(parents=True, exist_ok=True)

        write_manifest(run_manifest_path, new_entries)
        write_manifest(latest_path, new_entries)

        print(
            f"[CRAWL DONE] run_date={cfg.run_date} "
            f"urls={len(new_entries)} "
            f"manifest_run={run_manifest_path} "
            f"manifest_latest={latest_path}"
        )

    # =========================
    # 2) CLEAN
    # =========================
    if mode in ("clean", "run"):
        # Prefer this run's manifest; fallback to latest
        man_path = run_manifest_path if run_manifest_path.exists() else latest_path
        man = load_manifest(man_path)

        # Overwrite output for determinism
        if out_path.exists():
            out_path.unlink()

        kept = 0
        dropped_too_short = 0
        errors = 0
        total = 0

        for url, e in sorted(man.items(), key=lambda x: x[0]):
            total += 1
            try:
                raw = store.read_bytes(e.rel_path)

                if e.content_type == "text/html":
                    title, text = html_to_text(raw.decode("utf-8", errors="ignore"))
                else:
                    # per your constraint, only html/pdf exist
                    title, text = "", pdf_to_text(raw)

                if len(text) < cfg.min_text_chars:
                    dropped_too_short += 1
                    continue

                doc = {
                    "doc_id": sha256_hex((url + e.content_hash).encode("utf-8"))[:24],
                    "url": url,
                    "title": title,
                    "text": text,
                    "source": "seed",
                    "content_hash": e.content_hash,
                    "content_type": e.content_type,
                    "fetched_at": e.fetched_at,
                    "run_date": cfg.run_date,
                }
                append_jsonl(out_path, doc)
                kept += 1

            except Exception as ex:
                errors += 1
                print(f"[CLEAN FAIL] url={url} rel={getattr(e, 'rel_path', '')} -> {ex}")

        print(
            f"[CLEAN DONE] run_date={cfg.run_date} "
            f"manifest={man_path} "
            f"out={out_path} "
            f"total={total} kept={kept} dropped_too_short={dropped_too_short} errors={errors}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="rag-cc pipeline: crawl and/or clean")
    ap.add_argument("--config", required=True, help="Path to pipeline YAML")
    ap.add_argument(
        "--mode",
        choices=["crawl", "clean", "run"],
        default="run",
        help="crawl: fetch+write manifest; clean: read manifest+write cleaned jsonl; run: both",
    )
    args = ap.parse_args()

    run_pipeline(args.config, args.mode)


if __name__ == "__main__":
    main()
