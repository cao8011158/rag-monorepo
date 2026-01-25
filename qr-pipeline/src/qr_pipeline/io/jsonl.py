from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

import orjson


def read_jsonl(text: str, on_error: Optional[Callable[[Exception, str], None]] = None) -> List[Dict[str, Any]]:
    """Parse JSONL from a string.

    - strict: on_error=None -> raise
    - best-effort: on_error=callable -> skip bad lines
    """
    rows: List[Dict[str, Any]] = []
    for i, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        try:
            obj = orjson.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Line {i} is not a JSON object")
            rows.append(obj)
        except Exception as e:
            if on_error is None:
                raise
            on_error(e, line)
    return rows


def write_jsonl(rows: Iterable[Dict[str, Any]]) -> str:
    """Serialize rows to JSONL string."""
    out: List[bytes] = []
    for r in rows:
        out.append(orjson.dumps(r))
    return b"\n".join(out).decode("utf-8") + ("\n" if out else "")
