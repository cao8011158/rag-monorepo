# run.py
from __future__ import annotations

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
    obj = yaml.safe_load(seeds_path.read_text(encoding="utf-8"))
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


def run_pipeline(config_path: str, mode: str) -> None:
    cfg = load_cfg(config_path)
    store = LocalStore(cfg.local_root)

    latest_path = cfg.manifest_latest
    run_manifest_path = Path(cfg.manifest_run.format(run_date=cfg.run_date))

    old = load_manifest(latest_path)
    fetcher = Fetcher(cfg.user_agent, cfg.timeout_sec, cfg.per_host_rps)
    seeds = _load_seed_urls(cfg.seeds_path)

    # =========================
    # 1) CRAWL
    # =========================
    if mode in ("crawl", "run"):
        # Global limits
        total_fetched = 0

        # We'll build new_entries as "latest snapshot" for this run
        new_by_url: dict[str, ManifestEntry] = {}

        # Preload old entries (so we can reuse unchanged ones)
        for url, entry in old.items():
            new_by_url[url] = entry

        for seed_url in seeds:
            seed_domain = urlparse(seed_url).netloc.lower()

            # per-seed limits
            fetched_for_seed = 0

            q: deque[tuple[str, int]] = deque()
            discovered: set[str] = set()  # queue-level dedup
            fetched: set[str] = set()     # request-level dedup (this run)

            def allowed(u: str) -> bool:
                if not is_http_url(u):
                    return False
                if should_drop(u, cfg.drop_url_patterns):
                    return False

                # allowlist has highest priority (if provided)
                if cfg.allow_domains:
                    return urlparse(u).netloc.lower() in {d.lower() for d in cfg.allow_domains}

                # otherwise, optionally enforce same-domain crawling
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

                # =========================
                # ✅ CHANGE: fetch failure should NOT crash the whole program
                # =========================
                try:
                    # Fetch (no conditional requests; always GET)
                    resp = fetcher.get(url)
                except Exception as e:
                    # 不写失败日志文件；只在控制台打印，并跳过这个 URL
                    print(f"[FETCH FAIL] {url} -> {e}")
                    fetched.add(url_key)  # 避免同一 run 内重复尝试同一个 URL
                    continue

                content = resp.content
                content_hash = sha256_hex(content)

                # Determine type (prefer header; fallback to URL suffix)
                ctype = (resp.headers.get("Content-Type") or "").lower()
                is_pdf = ("application/pdf" in ctype) or url.lower().endswith(".pdf")
                ext = "pdf" if is_pdf else "html"

                fetched.add(url_key)
                total_fetched += 1
                fetched_for_seed += 1

                # If unchanged and exists in old -> reuse old entry (no write)
                if url in old and old[url].content_hash == content_hash:
                    new_by_url[url] = old[url]
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

                # Expand only HTML, only if depth < max_depth
                if (not is_pdf) and depth < cfg.max_depth:
                    html = content.decode("utf-8", errors="ignore")
                    links = extract_links(html, base_url=url)

                    # Gate: max_links_per_page
                    cnt = 0
                    for link in links:
                        if cnt >= cfg.max_links_per_page:
                            break
                        before = len(discovered)
                        push(link, depth + 1)
                        after = len(discovered)
                        if after > before:
                            cnt += 1

        # write run manifest + latest manifest (full snapshot)
        new_entries = list(new_by_url.values())
        write_manifest(run_manifest_path, new_entries)
        write_manifest(latest_path, new_entries)

    # =========================
    # 2) CLEAN
    # =========================
    if mode in ("clean", "run"):
        man = load_manifest(run_manifest_path if run_manifest_path.exists() else latest_path)
        out_path = Path(cfg.output_jsonl.format(run_date=cfg.run_date))

        for url, e in man.items():
            raw = store.read_bytes(e.rel_path)
            if e.content_type == "text/html":
                title, text = html_to_text(raw.decode("utf-8", errors="ignore"))
            else:
                title, text = "", pdf_to_text(raw)

            if len(text) < cfg.min_text_chars:
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