from pathlib import Path

from cc_pipeline.crawl.manifest import ManifestEntry, load_manifest, write_manifest


def test_manifest_roundtrip(tmp_path: Path):
    p = tmp_path / "m.jsonl"
    entries = [
        ManifestEntry(
            url="https://example.com/a",
            content_hash="abc",
            rel_path="raw/html/2026-01-16/seed/xxx.html",
            content_type="text/html",
            fetched_at="2026-01-16T00:00:00Z",
        ),
        ManifestEntry(
            url="https://example.com/b.pdf",
            content_hash="def",
            rel_path="raw/pdf/2026-01-16/seed/yyy.pdf",
            content_type="application/pdf",
            fetched_at="2026-01-16T00:00:01Z",
        ),
    ]

    write_manifest(p, entries)
    loaded = load_manifest(p)

    assert set(loaded.keys()) == {"https://example.com/a", "https://example.com/b.pdf"}
    assert loaded["https://example.com/a"].content_type == "text/html"
    assert loaded["https://example.com/b.pdf"].rel_path.endswith(".pdf")