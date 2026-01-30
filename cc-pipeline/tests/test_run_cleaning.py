# tests/test_run_cleaning.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import json

import pytest


# ---- helpers ----
@dataclass
class DummyEntry:
    url: str
    content_hash: str
    rel_path: str
    content_type: str
    fetched_at: str


class DummyStore:
    def __init__(self, root: str):
        self.root = root
        self._by_path: dict[str, bytes] = {}

    def read_bytes(self, rel_path: str) -> bytes:
        return self._by_path[rel_path]

    def write_bytes(self, rel_path: str, content: bytes) -> None:
        self._by_path[rel_path] = content


def _append_jsonl_real(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


@pytest.fixture
def mod():
    # 直接 import 被测模块（你实际路径）
    import cc_pipeline.run_cleaning as m
    return m


def test_always_reads_latest_manifest(monkeypatch, tmp_path, mod):
    # cfg.output_jsonl 包含 run_date -> 走正常 output_jsonl
    cfg = SimpleNamespace(
        local_root=str(tmp_path),
        manifest_latest=str(tmp_path / "manifests" / "latest.jsonl"),
        output_jsonl=str(tmp_path / "cleaned" / "{run_date}" / "cleaned.jsonl"),
        run_date="2026-01-22",
        min_text_chars=1,
    )

    called = {"manifest_path": None}

    def fake_load_cfg(p: str):
        return cfg

    def fake_load_manifest(p: str):
        called["manifest_path"] = p
        return {}

    monkeypatch.setattr(mod, "load_cfg", fake_load_cfg)
    monkeypatch.setattr(mod, "load_manifest", fake_load_manifest)
    monkeypatch.setattr(mod, "LocalStore", DummyStore)

    mod.run_cleaning("configs/pipeline.yaml")
    assert called["manifest_path"] == cfg.manifest_latest


def test_overwrite_same_date(monkeypatch, tmp_path, mod):
    cfg = SimpleNamespace(
        local_root=str(tmp_path),
        manifest_latest=str(tmp_path / "manifests" / "latest.jsonl"),
        output_jsonl=str(tmp_path / "data" / "cleaned" / "{run_date}" / "cleaned.jsonl"),
        run_date="2026-01-22",
        min_text_chars=1,
    )

    # 预先写入旧文件
    out_path = Path(cfg.output_jsonl.format(run_date=cfg.run_date))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("OLD\n", encoding="utf-8")

    # 构造 manifest + store 内容
    man = {
        "http://a.com": DummyEntry(
            url="http://a.com",
            content_hash="H1",
            rel_path="raw/html/2026-01-22/crawl/a.html",
            content_type="text/html",
            fetched_at="T",
        )
    }

    store = DummyStore(cfg.local_root)
    store._by_path[man["http://a.com"].rel_path] = b"<html>hi</html>"

    monkeypatch.setattr(mod, "load_cfg", lambda p: cfg)
    monkeypatch.setattr(mod, "load_manifest", lambda p: man)
    monkeypatch.setattr(mod, "LocalStore", lambda root: store)
    monkeypatch.setattr(mod, "html_to_text", lambda s: ("Title", "OK"))
    monkeypatch.setattr(mod, "pdf_to_text", lambda b: "PDF")
    monkeypatch.setattr(mod, "append_jsonl", _append_jsonl_real)

    # 固定 sha256_hex，让 doc_id 可预测
    monkeypatch.setattr(mod, "sha256_hex", lambda b: "x" * 64)

    mod.run_cleaning("configs/pipeline.yaml")

    # 旧内容应被覆盖（不再出现 OLD）
    text = out_path.read_text(encoding="utf-8")
    assert "OLD" not in text
    lines = [l for l in text.splitlines() if l.strip()]
    assert len(lines) == 1


def test_schema_is_exact_and_pdf_title_empty(monkeypatch, tmp_path, mod):
    cfg = SimpleNamespace(
        local_root=str(tmp_path),
        manifest_latest=str(tmp_path / "manifests" / "latest.jsonl"),
        output_jsonl=str(tmp_path / "cleaned" / "{run_date}" / "cleaned.jsonl"),
        run_date="2026-01-22",
        min_text_chars=1,
    )

    man = {
        "http://a.com/a.pdf": DummyEntry(
            url="http://a.com/a.pdf",
            content_hash="CH",
            rel_path="raw/pdf/2026-01-22/crawl/a.pdf",
            content_type="application/pdf",
            fetched_at="FA",
        )
    }

    store = DummyStore(cfg.local_root)
    store._by_path[man["http://a.com/a.pdf"].rel_path] = b"%PDF-1.4 ..."

    captured: list[dict] = []

    def fake_append(path: Path, obj: dict) -> None:
        captured.append(obj)

    monkeypatch.setattr(mod, "load_cfg", lambda p: cfg)
    monkeypatch.setattr(mod, "load_manifest", lambda p: man)
    monkeypatch.setattr(mod, "LocalStore", lambda root: store)
    monkeypatch.setattr(mod, "pdf_to_text", lambda b: "PDF TEXT")
    monkeypatch.setattr(mod, "append_jsonl", fake_append)
    monkeypatch.setattr(mod, "sha256_hex", lambda b: "a" * 64)

    mod.run_cleaning("configs/pipeline.yaml")

    assert len(captured) == 1
    doc = captured[0]

    # ✅ schema keys exactly as required
    assert set(doc.keys()) == {
        "doc_id",
        "url",
        "title",
        "text",
        "source",
        "content_hash",
        "content_type",
        "fetched_at",
        "run_date",
    }
    assert doc["title"] == ""          # PDF title 为空
    assert doc["source"] == "seed"
    assert doc["run_date"] == cfg.run_date
    assert doc["doc_id"] == ("a" * 24)  # sha256_hex[:24]


def test_min_text_chars_filter(monkeypatch, tmp_path, mod):
    cfg = SimpleNamespace(
        local_root=str(tmp_path),
        manifest_latest=str(tmp_path / "manifests" / "latest.jsonl"),
        output_jsonl=str(tmp_path / "cleaned" / "{run_date}" / "cleaned.jsonl"),
        run_date="2026-01-22",
        min_text_chars=10,  # 需要 >=10
    )

    man = {
        "http://a.com": DummyEntry(
            url="http://a.com",
            content_hash="H",
            rel_path="raw/html/2026-01-22/crawl/a.html",
            content_type="text/html",
            fetched_at="T",
        )
    }

    store = DummyStore(cfg.local_root)
    store._by_path[man["http://a.com"].rel_path] = b"<html>hi</html>"

    captured: list[dict] = []

    monkeypatch.setattr(mod, "load_cfg", lambda p: cfg)
    monkeypatch.setattr(mod, "load_manifest", lambda p: man)
    monkeypatch.setattr(mod, "LocalStore", lambda root: store)
    monkeypatch.setattr(mod, "html_to_text", lambda s: ("T", "short"))  # len=5
    monkeypatch.setattr(mod, "append_jsonl", lambda p, o: captured.append(o))

    mod.run_cleaning("configs/pipeline.yaml")
    assert captured == []  # 被过滤掉


def test_clean_fail_does_not_crash(monkeypatch, tmp_path, mod, capsys):
    cfg = SimpleNamespace(
        local_root=str(tmp_path),
        manifest_latest=str(tmp_path / "manifests" / "latest.jsonl"),
        output_jsonl=str(tmp_path / "cleaned" / "{run_date}" / "cleaned.jsonl"),
        run_date="2026-01-22",
        min_text_chars=1,
    )

    man = {
        "http://bad.com": DummyEntry(
            url="http://bad.com",
            content_hash="H",
            rel_path="raw/html/2026-01-22/crawl/bad.html",
            content_type="text/html",
            fetched_at="T",
        ),
        "http://good.com": DummyEntry(
            url="http://good.com",
            content_hash="H2",
            rel_path="raw/html/2026-01-22/crawl/good.html",
            content_type="text/html",
            fetched_at="T2",
        ),
    }

    store = DummyStore(cfg.local_root)
    store._by_path[man["http://bad.com"].rel_path] = b"<html>bad</html>"
    store._by_path[man["http://good.com"].rel_path] = b"<html>good</html>"

    captured: list[dict] = []

    def fake_html_to_text(s: str):
        if "bad" in s:
            raise RuntimeError("boom")
        return ("T", "OK TEXT")

    monkeypatch.setattr(mod, "load_cfg", lambda p: cfg)
    monkeypatch.setattr(mod, "load_manifest", lambda p: man)
    monkeypatch.setattr(mod, "LocalStore", lambda root: store)
    monkeypatch.setattr(mod, "html_to_text", fake_html_to_text)
    monkeypatch.setattr(mod, "append_jsonl", lambda p, o: captured.append(o))
    monkeypatch.setattr(mod, "sha256_hex", lambda b: "b" * 64)

    mod.run_cleaning("configs/pipeline.yaml")

    # bad 失败但不崩；good 仍然输出
    assert len(captured) == 1
    out = capsys.readouterr().out
    assert "[CLEAN FAIL]" in out


def test_fallback_output_path_when_output_jsonl_missing_run_date(monkeypatch, tmp_path, mod):
    # output_jsonl 不包含 run_date -> 应兜底到 local_root/cleaned/<run_date>/cleaned.jsonl
    cfg = SimpleNamespace(
        local_root=str(tmp_path),
        manifest_latest=str(tmp_path / "manifests" / "latest.jsonl"),
        output_jsonl=str(tmp_path / "cleaned" / "cleaned.jsonl"),  # 不含 run_date
        run_date="2026-01-22",
        min_text_chars=1,
    )

    man = {
        "http://a.com": DummyEntry(
            url="http://a.com",
            content_hash="H",
            rel_path="raw/html/2026-01-22/crawl/a.html",
            content_type="text/html",
            fetched_at="T",
        )
    }

    store = DummyStore(cfg.local_root)
    store._by_path[man["http://a.com"].rel_path] = b"<html>hi</html>"

    written_paths: list[Path] = []

    def fake_append(path: Path, obj: dict):
        written_paths.append(path)

    monkeypatch.setattr(mod, "load_cfg", lambda p: cfg)
    monkeypatch.setattr(mod, "load_manifest", lambda p: man)
    monkeypatch.setattr(mod, "LocalStore", lambda root: store)
    monkeypatch.setattr(mod, "html_to_text", lambda s: ("T", "OK TEXT"))
    monkeypatch.setattr(mod, "append_jsonl", fake_append)
    monkeypatch.setattr(mod, "sha256_hex", lambda b: "c" * 64)

    mod.run_cleaning("configs/pipeline.yaml")

    assert len(written_paths) == 1
    assert written_paths[0] == (Path(cfg.local_root) / "cleaned" / cfg.run_date / "cleaned.jsonl")
