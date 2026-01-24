from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cc_pipeline.pipeline.run as runmod


@dataclass
class DummyResp:
    content: bytes
    headers: dict[str, str]


class DummyFetcher:
    """
    Offline fake fetcher:
    - Records every URL that pipeline tries to fetch
    - Returns HTML pages from an in-memory dict
    """
    def __init__(self, pages: dict[str, str]):
        self.pages = pages
        self.calls: list[str] = []

    def get(self, url: str) -> DummyResp:
        self.calls.append(url)
        if url not in self.pages:
            raise RuntimeError(f"Unexpected fetch: {url}")
        html = self.pages[url]
        return DummyResp(content=html.encode("utf-8"), headers={"Content-Type": "text/html"})


def test_same_domain_only_blocks_cross_domain_fetch(tmp_path: Path, monkeypatch):
    """
    Seed domain: example.com
    Seed page contains:
      - /in  (same domain)
      - https://ecode360.com/PI6865  (cross domain)
    Expectation:
      - should fetch seed and /in
      - should NOT fetch ecode360.com
    """

    # ----- fake web graph -----
    pages = {
        "https://example.com/seed": """
            <a href="/in">in</a>
            <a href="https://ecode360.com/PI6865">ecode</a>
        """,
        "https://example.com/in": "<p>ok</p>",
        # Intentionally do NOT provide ecode360 page.
        # If crawler tries to fetch it, DummyFetcher will record it (and/or raise).
    }

    dummy = DummyFetcher(pages)

    # Patch Fetcher() in run module to return our dummy fetcher instance
    monkeypatch.setattr(runmod, "Fetcher", lambda *args, **kwargs: dummy)

    # ----- write seeds + config -----
    workdir = tmp_path / "proj"
    workdir.mkdir()

    seeds_path = workdir / "configs" / "seeds.yaml"
    seeds_path.parent.mkdir(parents=True, exist_ok=True)
    seeds_path.write_text(
        "seeds:\n  - name: t\n    urls:\n      - https://example.com/seed\n",
        encoding="utf-8",
    )

    data_root = workdir / "data"
    cfg_path = workdir / "configs" / "pipeline.yaml"

    cfg_path.write_text(
        f"""
project:
  run_date: "2026-01-16"

storage:
  mode: "local"
  local_root: "{data_root.as_posix()}"

crawl:
  user_agent: "ua"
  timeout_sec: 1
  max_retries: 1
  per_host_rps: 1000.0
  max_depth: 2
  same_domain_only: true
  allow_domains: []
  max_links_per_page: 50
  max_pages_total: 100
  max_pages_per_seed: 100
  drop_url_patterns: []

clean:
  min_text_chars: 1
  output_jsonl: "{(data_root / 'cleaned/{{run_date}}/documents.jsonl').as_posix()}"

manifest:
  latest_path: "{(data_root / 'manifests/latest.jsonl').as_posix()}"
  run_path: "{(data_root / 'manifests/{{run_date}}.jsonl').as_posix()}"

seeds:
  path: "{seeds_path.as_posix()}"
""".strip(),
        encoding="utf-8",
    )

    # ----- run crawl only -----
    runmod.run_pipeline(config_path=str(cfg_path), mode="crawl")

    # ----- assertions -----
    # Must fetch seed and same-domain child
    assert "https://example.com/seed" in dummy.calls
    assert "https://example.com/in" in dummy.calls

    # Must NOT fetch cross-domain
    bad = [u for u in dummy.calls if "ecode360.com" in u]
    assert not bad, f"Cross-domain URLs were fetched unexpectedly: {bad}"