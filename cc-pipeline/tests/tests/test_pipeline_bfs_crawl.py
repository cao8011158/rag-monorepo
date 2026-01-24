from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cc_pipeline.pipeline.run as runmod


@dataclass
class DummyResp:
    content: bytes
    headers: dict[str, str]


class DummyFetcher:
    def __init__(self, pages: dict[str, str]):
        self.pages = pages
        self.calls: list[str] = []

    def get(self, url: str):
        self.calls.append(url)
        html = self.pages.get(url)
        if html is None:
            raise RuntimeError(f"unexpected url fetch: {url}")
        return DummyResp(content=html.encode("utf-8"), headers={"Content-Type": "text/html"})


def test_bfs_crawl_same_domain_depth_and_dedup(tmp_path: Path, monkeypatch):
    # --- fake web graph ---
    # seed -> a (same domain), x (other domain), dup a
    # a -> b (same), c (same)
    pages = {
        "https://example.com/seed": """
            <a href="/a">a</a>
            <a href="https://other.com/x">x</a>
            <a href="/a">dup-a</a>
        """,
        "https://example.com/a": """
            <a href="/b">b</a>
            <a href="/c">c</a>
        """,
        "https://example.com/b": "<p>b</p>",
        "https://example.com/c": "<p>c</p>",
    }

    # --- write config + seeds ---
    seeds_path = tmp_path / "seeds.yaml"
    cfg_path = tmp_path / "pipeline.yaml"

    seeds_path.write_text(
        "seeds:\n  - name: t\n    urls:\n      - https://example.com/seed\n",
        encoding="utf-8",
    )

    cfg_path.write_text(
        f"""
project:
  run_date: "2026-01-01"

storage:
  mode: "local"
  local_root: "{tmp_path.as_posix()}"

crawl:
  user_agent: "ua"
  timeout_sec: 1
  max_retries: 1
  per_host_rps: 1000.0
  max_depth: 1
  same_domain_only: true
  allow_domains: []
  max_links_per_page: 10
  max_pages_total: 100
  max_pages_per_seed: 100
  drop_url_patterns: ["mailto:", "javascript:"]

clean:
  min_text_chars: 1
  output_jsonl: "data/cleaned/{{run_date}}/documents.jsonl"

manifest:
  latest_path: "{(tmp_path / 'data/manifests/latest.jsonl').as_posix()}"
  run_path: "{(tmp_path / 'data/manifests/{{run_date}}.jsonl').as_posix()}"

seeds:
  path: "{seeds_path.as_posix()}"
""".strip(),
        encoding="utf-8",
    )

    dummy = DummyFetcher(pages)

    # Patch Fetcher constructor used in run_pipeline to return our dummy
    monkeypatch.setattr(runmod, "Fetcher", lambda *args, **kwargs: dummy)

    # Run crawl only
    runmod.run_pipeline(str(cfg_path), mode="crawl")

    # With max_depth=1:
    # should fetch seed and a (depth1). b/c are depth2 -> not fetched.
    # other.com/x filtered by same_domain_only -> not fetched.
    assert dummy.calls.count("https://example.com/seed") == 1
    assert dummy.calls.count("https://example.com/a") == 1
    assert "https://other.com/x" not in dummy.calls
    assert "https://example.com/b" not in dummy.calls
    assert "https://example.com/c" not in dummy.calls


def test_bfs_crawl_max_links_per_page_gate(tmp_path: Path, monkeypatch):
    # seed has many links; gate should limit discovered expansion
    seed_html = "\n".join([f'<a href="/p{i}">p{i}</a>' for i in range(100)])
    pages = {"https://example.com/seed": seed_html}
    # provide content for only first few; if crawler tries beyond gate, it will crash
    for i in range(5):
        pages[f"https://example.com/p{i}"] = "<p>ok</p>"

    seeds_path = tmp_path / "seeds.yaml"
    cfg_path = tmp_path / "pipeline.yaml"

    seeds_path.write_text(
        "seeds:\n  - name: t\n    urls:\n      - https://example.com/seed\n",
        encoding="utf-8",
    )

    cfg_path.write_text(
        f"""
project:
  run_date: "2026-01-01"

storage:
  mode: "local"
  local_root: "{tmp_path.as_posix()}"

crawl:
  user_agent: "ua"
  timeout_sec: 1
  max_retries: 1
  per_host_rps: 1000.0
  max_depth: 1
  same_domain_only: true
  allow_domains: []
  max_links_per_page: 5
  max_pages_total: 100
  max_pages_per_seed: 100
  drop_url_patterns: []

clean:
  min_text_chars: 1
  output_jsonl: "data/cleaned/{{run_date}}/documents.jsonl"

manifest:
  latest_path: "{(tmp_path / 'data/manifests/latest.jsonl').as_posix()}"
  run_path: "{(tmp_path / 'data/manifests/{{run_date}}.jsonl').as_posix()}"

seeds:
  path: "{seeds_path.as_posix()}"
""".strip(),
        encoding="utf-8",
    )

    dummy = DummyFetcher(pages)
    monkeypatch.setattr(runmod, "Fetcher", lambda *args, **kwargs: dummy)

    runmod.run_pipeline(str(cfg_path), mode="crawl")

    # Should fetch seed + first 5 pages at most due to gate + depth=1
    assert dummy.calls[0] == "https://example.com/seed"
    fetched_children = [u for u in dummy.calls if u != "https://example.com/seed"]
    assert len(fetched_children) <= 5