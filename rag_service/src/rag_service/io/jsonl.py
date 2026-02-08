from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, Callable, Optional
import orjson

from ..stores.base import Store


def read_jsonl(
    store: Store,
    path: str,
    *,
    on_error: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Robust JSONL reader.

    - If on_error is None: fail-fast (raise on first bad line)
    - If on_error is provided: skip bad lines and call on_error(payload)
    """
    data = store.read_bytes(path)
    for lineno, line in enumerate(data.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = orjson.loads(line)
            if not isinstance(obj, dict):
                raise ValueError("JSONL row must be an object")
            yield obj
        except Exception as e:
            if on_error is None:
                raise
            on_error(
                {
                    "stage": "read_jsonl",
                    "path": path,
                    "line_no": lineno,
                    "error": f"{type(e).__name__}: {e}",
                    "line_preview": line[:200].decode("utf-8", errors="replace"),
                }
            )
            continue


def write_jsonl(store: Store, path: str, rows: Iterable[Dict[str, Any]]) -> None:
    out = b"".join(orjson.dumps(r) + b"\n" for r in rows)
    store.write_bytes(path, out)


def append_jsonl(store: Store, path: str, rows: Iterable[Dict[str, Any]]) -> None:
    existing = b""
    if store.exists(path):
        existing = store.read_bytes(path)
        if existing and not existing.endswith(b"\n"):
            existing += b"\n"
    out = existing + b"".join(orjson.dumps(r) + b"\n" for r in rows)
    store.write_bytes(path, out)