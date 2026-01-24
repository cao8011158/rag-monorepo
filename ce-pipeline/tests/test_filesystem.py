from __future__ import annotations

from pathlib import Path

import pytest

from ce_pipeline.stores.filesystem import FilesystemStore


def test_exists_false_then_true(tmp_path: Path) -> None:
    store = FilesystemStore(root=tmp_path)

    assert store.exists("a.txt") is False

    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    assert store.exists("a.txt") is True


def test_write_and_read_text_roundtrip(tmp_path: Path) -> None:
    store = FilesystemStore(root=tmp_path)

    store.write_text("dir1/hello.txt", "你好", encoding="utf-8")
    assert (tmp_path / "dir1" / "hello.txt").exists()

    got = store.read_text("dir1/hello.txt", encoding="utf-8")
    assert got == "你好"


def test_write_and_read_bytes_roundtrip(tmp_path: Path) -> None:
    store = FilesystemStore(root=tmp_path)

    payload = b"\x00\x01\x02abc"
    store.write_bytes("bin/data.bin", payload)
    got = store.read_bytes("bin/data.bin")

    assert got == payload


def test_append_bytes_appends(tmp_path: Path) -> None:
    store = FilesystemStore(root=tmp_path)

    store.write_bytes("log/out.bin", b"hello")
    store.append_bytes("log/out.bin", b" world")
    assert store.read_bytes("log/out.bin") == b"hello world"


def test_list_returns_empty_when_prefix_not_exists(tmp_path: Path) -> None:
    store = FilesystemStore(root=tmp_path)

    assert list(store.list("missing")) == []


def test_list_file_prefix_returns_same_prefix(tmp_path: Path) -> None:
    store = FilesystemStore(root=tmp_path)

    store.write_text("one.txt", "1")
    assert list(store.list("one.txt")) == ["one.txt"]


def test_list_directory_prefix_lists_all_files_posix_paths(tmp_path: Path) -> None:
    store = FilesystemStore(root=tmp_path)

    store.write_text("a/x.txt", "x")
    store.write_text("a/b/y.txt", "y")
    store.write_text("a/b/c/z.txt", "z")

    got = sorted(store.list("a"))
    # 关键点：实现里 replace(os.sep, "/")，所以期望永远是 POSIX 分隔符
    assert got == ["a/b/c/z.txt", "a/b/y.txt", "a/x.txt"]


def test_list_root_prefix_lists_all_files(tmp_path: Path) -> None:
    store = FilesystemStore(root=tmp_path)

    store.write_text("r1.txt", "1")
    store.write_text("d/r2.txt", "2")

    got = sorted(store.list(""))
    assert got == ["d/r2.txt", "r1.txt"]
