from __future__ import annotations

from pathlib import Path

from cc_pipeline.settings import load_cfg


def test_load_cfg_new_crawl_fields_defaults(tmp_path: Path):
    cfg_path = tmp_path / "pipeline.yaml"
    seeds_path = tmp_path / "seeds.yaml"

    seeds_path.write_text(
        "seeds:\n  - name: t\n    urls:\n      - https://example.com\n",
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
  per_host_rps: 10.0

clean:
  min_text_chars: 1
  output_jsonl: "data/cleaned/{{run_date}}/documents.jsonl"

manifest:
  latest_path: "data/manifests/latest.jsonl"
  run_path: "data/manifests/{{run_date}}.jsonl"

seeds:
  path: "{seeds_path.as_posix()}"
""".strip(),
        encoding="utf-8",
    )

    cfg = load_cfg(str(cfg_path))

    # defaults
    assert cfg.max_depth == 0
    assert cfg.same_domain_only is True
    assert cfg.allow_domains == []
    assert cfg.max_links_per_page == 50
    assert cfg.max_pages_total == 1000
    assert cfg.max_pages_per_seed == 200
    assert cfg.drop_url_patterns == []


def test_load_cfg_new_crawl_fields_custom(tmp_path: Path):
    cfg_path = tmp_path / "pipeline.yaml"
    seeds_path = tmp_path / "seeds.yaml"

    seeds_path.write_text(
        "seeds:\n  - name: t\n    urls:\n      - https://example.com\n",
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
  per_host_rps: 10.0
  max_depth: 2
  same_domain_only: true
  allow_domains: ["example.com"]
  max_links_per_page: 7
  max_pages_total: 9
  max_pages_per_seed: 3
  drop_url_patterns: ["mailto:", ".png"]

clean:
  min_text_chars: 1
  output_jsonl: "data/cleaned/{{run_date}}/documents.jsonl"

manifest:
  latest_path: "data/manifests/latest.jsonl"
  run_path: "data/manifests/{{run_date}}.jsonl"

seeds:
  path: "{seeds_path.as_posix()}"
""".strip(),
        encoding="utf-8",
    )

    cfg = load_cfg(str(cfg_path))
    assert cfg.max_depth == 2
    assert cfg.same_domain_only is True
    assert cfg.allow_domains == ["example.com"]
    assert cfg.max_links_per_page == 7
    assert cfg.max_pages_total == 9
    assert cfg.max_pages_per_seed == 3
    assert cfg.drop_url_patterns == ["mailto:", ".png"]