# tests/test_run_cleaning.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest


# ----------------------------
# Helpers / fakes
# ----------------------------
@dataclass
class ManifestEntry:
    rel_path: str
    content_type: str
    content_hash: str
    fetched_at: str


class AppendCapture:
    def __init__(self) -> None:
        self.calls = []  # list[tuple[Path, dict]]

    def __call__(self, out_path: Path, doc: Dict[str, Any]) -> None:
        self.calls.append((Path(out_path), doc))


class FakeStore:
    def __init__(self, local_root: str, mapping: Dict[str, bytes], fail_on: Optional[str] = None) -> None:
        self.local_root = local_root
        self.mapping = mapping
        self.fail_on = fail_on

    def read_bytes(self, rel_path: str) -> bytes:
        if self.fail_on and rel_path == self.fail_on:
            raise RuntimeError("boom")
        return self.mapping[rel_path]


def _make_cfg(tmp_path: Path, *, min_text_chars: int = 10) -> Any:
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # output_jsonl uses .format(run_date=cfg.run_date)
    return SimpleNamespace(
        local_root=str(tmp_path),
        manifest_latest=tmp_path / "manifests" / "latest.json",
        output_jsonl=str(out_dir / "cleaned_{run_date}.jsonl"),
        run_date="2026-02-01",
        min_text_chars=min_text_chars,
    )


# ----------------------------
# Tests
# ----------------------------
def test_run_cleaning_reads_latest_manifest_and_overwrites_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    - Always load manifest from cfg.manifest_latest
    - If output exists, unlink it (overwrite behavior)
    """
    import cc_pipeline.run_cleaning as m

    cfg = _make_cfg(tmp_path, min_text_chars=5)

    loaded_manifest_path = {"value": None}

    def fake_load_cfg(config_path: str) -> Any:
        return cfg

    def fake_load_manifest(path: Path) -> Dict[str, ManifestEntry]:
        loaded_manifest_path["value"] = Path(path)
        return {}

    # Create an existing output file that should be removed
    out_path = Path(cfg.output_jsonl.format(run_date=cfg.run_date))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("old", encoding="utf-8")
    assert out_path.exists()

    monkeypatch.setattr(m, "load_cfg", fake_load_cfg)
    monkeypatch.setattr(m, "load_manifest", fake_load_manifest)
    monkeypatch.setattr(m, "LocalStore", lambda local_root: FakeStore(local_root, {}))
    monkeypatch.setattr(m, "append_jsonl", AppendCapture())

    m.run_cleaning("configs/pipeline.yaml")

    # Manifest path must be cfg.manifest_latest
    assert loaded_manifest_path["value"] == Path(cfg.manifest_latest)

    # Output file should be unlinked (since we never write anything for empty manifest)
    assert not out_path.exists()


def test_run_cleaning_html_filters_short_and_writes_long(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    HTML:
      - too short -> dropped
      - long enough -> kept and written
    """
    import cc_pipeline.run_cleaning as m

    cfg = _make_cfg(tmp_path, min_text_chars=10)

    man = {
        "https://a.com/short": ManifestEntry(
            rel_path="raw/short.html",
            content_type="text/html",
            content_hash="h1",
            fetched_at="t1",
        ),
        "https://a.com/long": ManifestEntry(
            rel_path="raw/long.html",
            content_type="text/html",
            content_hash="h2",
            fetched_at="t2",
        ),
    }

    store = FakeStore(
        cfg.local_root,
        mapping={
            "raw/short.html": b"<html>short</html>",
            "raw/long.html": b"<html>this is definitely long enough</html>",
        },
    )

    def fake_load_cfg(_: str) -> Any:
        return cfg

    def fake_load_manifest(_: Path) -> Dict[str, ManifestEntry]:
        return man

    def fake_html_to_text(_: str) -> tuple[str, str]:
        # decide based on original bytes? easiest: use sentinel in raw bytes decode
        # Here, store bytes decode contains "short" or "definitely"
        # We'll just return based on substring.
        if "short" in _:
            return "T-short", "12345"  # len=5 < 10 => drop
        return "T-long", "x" * 25  # keep

    app = AppendCapture()

    monkeypatch.setattr(m, "load_cfg", fake_load_cfg)
    monkeypatch.setattr(m, "load_manifest", fake_load_manifest)
    monkeypatch.setattr(m, "LocalStore", lambda local_root: store)
    monkeypatch.setattr(m, "html_to_text", fake_html_to_text)
    monkeypatch.setattr(m, "append_jsonl", app)

    m.run_cleaning("configs/pipeline.yaml")

    assert len(app.calls) == 1
    out_path, doc = app.calls[0]

    assert out_path.name == f"cleaned_{cfg.run_date}.jsonl"
    assert doc["url"] == "https://a.com/long"
    assert doc["title"] == "T-long"
    assert isinstance(doc["text"], str) and len(doc["text"]) == 25
    assert doc["structure"] is None
    assert doc["content_type"] == "text/html"
    assert doc["content_hash"] == "h2"
    assert doc["fetched_at"] == "t2"
    assert doc["run_date"] == cfg.run_date
    assert doc["source"] == "seed"
    assert isinstance(doc["doc_id"], str) and len(doc["doc_id"]) == 24


def test_run_cleaning_pdf_current_behavior_gets_dropped_when_min_text_chars_gt_0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    IMPORTANT: With your current _clean_one() implementation for PDF:
        return "", "", docling_dict
    so text == "" (not None), and len(text) < min_text_chars => dropped.
    This test locks in the *current* behavior so regressions are explicit.
    """
    import cc_pipeline.run_cleaning as m

    cfg = _make_cfg(tmp_path, min_text_chars=1)

    man = {
        "https://a.com/doc.pdf": ManifestEntry(
            rel_path="raw/doc.pdf",
            content_type="application/pdf",
            content_hash="hp",
            fetched_at="tp",
        ),
    }

    store = FakeStore(cfg.local_root, mapping={"raw/doc.pdf": b"%PDF-1.4..."})

    def fake_load_cfg(_: str) -> Any:
        return cfg

    def fake_load_manifest(_: Path) -> Dict[str, ManifestEntry]:
        return man

    def fake_pdf_to_docling_dict(_: bytes, filename: str) -> Dict[str, Any]:
        return {"pages": [{"blocks": [{"text": "hello"}]}]}

    app = AppendCapture()

    monkeypatch.setattr(m, "load_cfg", fake_load_cfg)
    monkeypatch.setattr(m, "load_manifest", fake_load_manifest)
    monkeypatch.setattr(m, "LocalStore", lambda local_root: store)
    monkeypatch.setattr(m, "pdf_to_docling_dict", fake_pdf_to_docling_dict)
    monkeypatch.setattr(m, "append_jsonl", app)

    m.run_cleaning("configs/pipeline.yaml")

    # Dropped because text=="" and min_text_chars==1
    assert len(app.calls) == 0


@pytest.mark.xfail(reason="Likely intended behavior: PDF should set text=None so proxy/structure filter runs.")
def test_run_cleaning_pdf_intended_behavior_kept_by_structure_proxy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    If you change _clean_one() PDF branch to:
        return "", None, docling_dict
    then this test should pass: it will keep PDFs if approx_chars >= min_text_chars.
    """
    import cc_pipeline.run_cleaning as m

    cfg = _make_cfg(tmp_path, min_text_chars=5)

    man = {
        "https://a.com/doc.pdf": ManifestEntry(
            rel_path="raw/doc.pdf",
            content_type="application/pdf",
            content_hash="hp",
            fetched_at="tp",
        ),
    }

    store = FakeStore(cfg.local_root, mapping={"raw/doc.pdf": b"%PDF-1.4..."})

    def fake_load_cfg(_: str) -> Any:
        return cfg

    def fake_load_manifest(_: Path) -> Dict[str, ManifestEntry]:
        return man

    def fake_pdf_to_docling_dict(_: bytes, filename: str) -> Dict[str, Any]:
        return {"pages": [{"blocks": [{"text": "hello world"}]}]}  # approx_chars=11 >= 5

    # Patch _clean_one to simulate intended return (text=None)
    def fake_clean_one(raw: bytes, content_type: str, filename_for_pdf: str):
        assert content_type != "text/html"
        return "", None, fake_pdf_to_docling_dict(raw, filename_for_pdf)

    app = AppendCapture()

    monkeypatch.setattr(m, "load_cfg", fake_load_cfg)
    monkeypatch.setattr(m, "load_manifest", fake_load_manifest)
    monkeypatch.setattr(m, "LocalStore", lambda local_root: store)
    monkeypatch.setattr(m, "_clean_one", fake_clean_one)
    monkeypatch.setattr(m, "append_jsonl", app)

    m.run_cleaning("configs/pipeline.yaml")

    assert len(app.calls) == 1
    _, doc = app.calls[0]
    assert doc["content_type"] == "application/pdf"
    assert doc["text"] is None
    assert isinstance(doc["structure"], dict)


def test_run_cleaning_deterministic_order_by_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    The loop iterates sorted(man.items(), key=url), so write order should follow URL order.
    """
    import cc_pipeline.run_cleaning as m

    cfg = _make_cfg(tmp_path, min_text_chars=1)

    man = {
        "https://b.com/z": ManifestEntry("raw/z.html", "text/html", "hz", "tz"),
        "https://a.com/a": ManifestEntry("raw/a.html", "text/html", "ha", "ta"),
        "https://c.com/m": ManifestEntry("raw/m.html", "text/html", "hm", "tm"),
    }

    store = FakeStore(
        cfg.local_root,
        mapping={"raw/z.html": b"z", "raw/a.html": b"a", "raw/m.html": b"m"},
    )

    def fake_load_cfg(_: str) -> Any:
        return cfg

    def fake_load_manifest(_: Path) -> Dict[str, ManifestEntry]:
        return man

    def fake_html_to_text(_: str) -> tuple[str, str]:
        # Always long enough (min_text_chars=1)
        return "T", "ok"

    app = AppendCapture()

    monkeypatch.setattr(m, "load_cfg", fake_load_cfg)
    monkeypatch.setattr(m, "load_manifest", fake_load_manifest)
    monkeypatch.setattr(m, "LocalStore", lambda local_root: store)
    monkeypatch.setattr(m, "html_to_text", fake_html_to_text)
    monkeypatch.setattr(m, "append_jsonl", app)

    m.run_cleaning("configs/pipeline.yaml")

    urls_written = [doc["url"] for _, doc in app.calls]
    assert urls_written == sorted(man.keys())


def test_run_cleaning_counts_errors_and_continues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any) -> None:
    """
    If one entry fails, it should print [CLEAN FAIL] and continue processing others.
    """
    import cc_pipeline.run_cleaning as m

    cfg = _make_cfg(tmp_path, min_text_chars=1)

    man = {
        "https://a.com/good": ManifestEntry("raw/good.html", "text/html", "h1", "t1"),
        "https://a.com/bad": ManifestEntry("raw/bad.html", "text/html", "h2", "t2"),
    }

    store = FakeStore(
        cfg.local_root,
        mapping={"raw/good.html": b"good", "raw/bad.html": b"bad"},
        fail_on="raw/bad.html",
    )

    def fake_load_cfg(_: str) -> Any:
        return cfg

    def fake_load_manifest(_: Path) -> Dict[str, ManifestEntry]:
        return man

    def fake_html_to_text(_: str) -> tuple[str, str]:
        return "T", "ok"

    app = AppendCapture()

    monkeypatch.setattr(m, "load_cfg", fake_load_cfg)
    monkeypatch.setattr(m, "load_manifest", fake_load_manifest)
    monkeypatch.setattr(m, "LocalStore", lambda local_root: store)
    monkeypatch.setattr(m, "html_to_text", fake_html_to_text)
    monkeypatch.setattr(m, "append_jsonl", app)

    m.run_cleaning("configs/pipeline.yaml")

    # One good write, one failure
    assert len(app.calls) == 1
    assert app.calls[0][1]["url"] == "https://a.com/good"

    out = capsys.readouterr().out
    assert "[CLEAN FAIL]" in out
