# tests/test_io_jsonl.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List

import pytest
import orjson

from ce_pipeline.io.jsonl import read_jsonl, write_jsonl, append_jsonl


@dataclass
class MemoryStore:
    """
    A minimal in-memory Store implementation for unit tests.

    It mimics the subset of Store API used by jsonl.py:
    - exists(path)
    - read_bytes(path)
    - write_bytes(path, content)
    - list(prefix)  (not used in these tests, but included for completeness)
    """
    blobs: Dict[str, bytes]

    def exists(self, path: str) -> bool:
        return path in self.blobs

    def read_bytes(self, path: str) -> bytes:
        return self.blobs[path]

    def write_bytes(self, path: str, content: bytes) -> None:
        self.blobs[path] = content

    def list(self, prefix: str) -> Iterable[str]:
        return [k for k in self.blobs.keys() if k.startswith(prefix)]


def _b(*lines: bytes) -> bytes:
    """Helper: join lines with newlines (exact bytes)."""
    return b"\n".join(lines)


def test_read_jsonl_yields_dict_rows_and_skips_blank_lines() -> None:
    store = MemoryStore(
        blobs={
            "a.jsonl": _b(
                orjson.dumps({"a": 1}),
                b"",  # blank line
                orjson.dumps({"b": 2}),
            )
        }
    )

    rows = list(read_jsonl(store, "a.jsonl"))
    assert rows == [{"a": 1}, {"b": 2}]


def test_read_jsonl_fail_fast_on_invalid_json_when_on_error_none() -> None:
    store = MemoryStore(
        blobs={
            "bad.jsonl": _b(
                orjson.dumps({"ok": True}),
                b"{not-json",
                orjson.dumps({"never": "reached"}),
            )
        }
    )

    it = read_jsonl(store, "bad.jsonl", on_error=None)
    assert next(it) == {"ok": True}
    with pytest.raises(Exception):
        list(it)  # should raise on the bad line


def test_read_jsonl_fail_fast_on_non_object_row() -> None:
    store = MemoryStore(
        blobs={
            "nonobj.jsonl": _b(
                orjson.dumps({"ok": 1}),
                orjson.dumps([1, 2, 3]),  # not a dict/object
            )
        }
    )

    it = read_jsonl(store, "nonobj.jsonl")
    assert next(it) == {"ok": 1}
    with pytest.raises(ValueError, match="JSONL row must be an object"):
        next(it)


def test_read_jsonl_tolerant_mode_skips_bad_lines_and_calls_on_error_payload() -> None:
    store = MemoryStore(
        blobs={
            "mix.jsonl": _b(
                orjson.dumps({"ok": 1}),
                b"{bad-json",
                orjson.dumps({"ok": 2}),
                orjson.dumps([1, 2]),  # non-object
            )
        }
    )

    errors: List[dict] = []

    def on_error(payload: dict) -> None:
        errors.append(payload)

    rows = list(read_jsonl(store, "mix.jsonl", on_error=on_error))

    assert rows == [{"ok": 1}, {"ok": 2}]
    assert len(errors) == 2

    # Validate error payload shape
    for e in errors:
        assert e["stage"] == "read_jsonl"
        assert e["path"] == "mix.jsonl"
        assert isinstance(e["line_no"], int)
        assert isinstance(e["error"], str)
        assert isinstance(e["line_preview"], str)
        assert len(e["line_preview"]) <= 200


def test_write_jsonl_writes_newline_delimited_bytes() -> None:
    store = MemoryStore(blobs={})
    rows = [{"a": 1}, {"b": 2}]
    write_jsonl(store, "out.jsonl", rows)

    data = store.read_bytes("out.jsonl")
    # Must end with newline because code appends b"\n" per row
    assert data.endswith(b"\n")

    parsed = [orjson.loads(line) for line in data.splitlines() if line.strip()]
    assert parsed == rows


def test_append_jsonl_creates_file_if_missing() -> None:
    store = MemoryStore(blobs={})
    append_jsonl(store, "new.jsonl", [{"x": 1}])

    data = store.read_bytes("new.jsonl")
    assert data == orjson.dumps({"x": 1}) + b"\n"


def test_append_jsonl_preserves_existing_newline_if_already_present() -> None:
    store = MemoryStore(
        blobs={
            "a.jsonl": (orjson.dumps({"a": 1}) + b"\n")
        }
    )
    append_jsonl(store, "a.jsonl", [{"b": 2}])

    data = store.read_bytes("a.jsonl")
    parsed = [orjson.loads(line) for line in data.splitlines() if line.strip()]
    assert parsed == [{"a": 1}, {"b": 2}]


def test_append_jsonl_inserts_newline_if_existing_missing_trailing_newline() -> None:
    store = MemoryStore(
        blobs={
            "a.jsonl": (orjson.dumps({"a": 1}))  # no trailing \n
        }
    )
    append_jsonl(store, "a.jsonl", [{"b": 2}])

    data = store.read_bytes("a.jsonl")
    # Ensure it became valid JSONL with line break between objects
    parsed = [orjson.loads(line) for line in data.splitlines() if line.strip()]
    assert parsed == [{"a": 1}, {"b": 2}]
