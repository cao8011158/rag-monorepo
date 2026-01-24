from __future__ import annotations

from pathlib import Path
import yaml
import orjson

from cc_pipeline.pipeline.run import run_pipeline


class FakeResp:
    def __init__(self, content: bytes, content_type: str):
        self.content = content
        self.headers = {"Content-Type": content_type}


def _read_manifest_urls(path: Path) -> list[str]:
    urls: list[str] = []
    if not path.exists():
        return urls
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = orjson.loads(line)
        urls.append(obj["url"])
    return urls


def test_pipeline_run_end_to_end_without_network(tmp_path: Path, monkeypatch):
    # 1) temp project dir
    workdir = tmp_path / "proj"
    workdir.mkdir()

    # 2) seeds.yaml (HTML + PDF)
    seeds_path = workdir / "configs" / "seeds.yaml"
    seeds_path.parent.mkdir(parents=True, exist_ok=True)
    seeds_obj = {
        "seeds": [
            {"name": "Test", "urls": ["https://example.com/a", "https://example.com/b.pdf"]}
        ]
    }
    seeds_path.write_text(yaml.safe_dump(seeds_obj, sort_keys=False), encoding="utf-8")

    # 3) pipeline.yaml
    data_root = workdir / "data"
    cfg_path = workdir / "configs" / "pipeline.yaml"

    cfg_obj = {
        "project": {"run_date": "2026-01-16"},
        "storage": {"mode": "local", "local_root": str(data_root)},
        "crawl": {
            "user_agent": "cc-pipeline-bot/0.1 (+test)",
            "timeout_sec": 5,
            "max_retries": 1,
            "per_host_rps": 1000.0,  # no sleep
            # NEW BFS params (keep it stable)
            "max_depth": 0,  # only fetch seed URLs, do not expand
            "same_domain_only": True,
            "allow_domains": [],
            "max_links_per_page": 10,
            "max_pages_total": 10,
            "max_pages_per_seed": 10,
            "drop_url_patterns": ["mailto:", "javascript:"],
        },
        "clean": {
            "min_text_chars": 1,
            "output_jsonl": str(data_root / "cleaned/{run_date}/documents.jsonl"),
        },
        "manifest": {
            "latest_path": str(data_root / "manifests/latest.jsonl"),
            "run_path": str(data_root / "manifests/{run_date}.jsonl"),
        },
        "seeds": {"path": str(seeds_path)},
    }
    cfg_path.write_text(yaml.safe_dump(cfg_obj, sort_keys=False), encoding="utf-8")

    # 4) Monkeypatch Fetcher.get -> offline fake responses
    html_bytes = b"""
    <html><head><title>T</title></head>
    <body><main><h1>Hello</h1><p>World</p></main></body></html>
    """

    pdf_path = Path(__file__).parent / "fixtures" / "Hello.pdf"
    pdf_bytes = pdf_path.read_bytes()

    def fake_get(self, url: str):
        if url.endswith(".pdf"):
            return FakeResp(pdf_bytes, content_type="application/pdf")
        return FakeResp(html_bytes, content_type="text/html")

    from cc_pipeline.crawl.fetcher import Fetcher
    monkeypatch.setattr(Fetcher, "get", fake_get)

    # 5) Run pipeline (no network)
    run_pipeline(config_path=str(cfg_path), mode="run")

    # 6) assert artifacts exist
    latest_manifest = data_root / "manifests" / "latest.jsonl"
    run_manifest = data_root / "manifests" / "2026-01-16.jsonl"
    out_jsonl = data_root / "cleaned" / "2026-01-16" / "documents.jsonl"

    assert latest_manifest.exists()
    assert run_manifest.exists()
    assert out_jsonl.exists()

    # 7) JSONL has at least 1 line and required keys
    lines1 = out_jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines1) >= 1

    obj = orjson.loads(lines1[0])
    for k in ["doc_id", "url", "title", "text", "content_hash", "content_type", "fetched_at", "run_date"]:
        assert k in obj

    # 8) Run again: manifest should not "grow" in unique URLs
    urls_before = _read_manifest_urls(latest_manifest)
    run_pipeline(config_path=str(cfg_path), mode="run")
    urls_after = _read_manifest_urls(latest_manifest)

    assert set(urls_after) == set(urls_before)
    assert len(urls_after) == len(urls_before)

