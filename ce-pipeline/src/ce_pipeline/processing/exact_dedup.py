from __future__ import annotations
import json
from typing import Dict, Any, Iterable, Optional


def exact_dedup_jsonl_by_hash_meta(
    in_path: str,
    out_path: str,
    *,
    hash_field: str = "chunk_text_hash",
    encoding: str = "utf-8",
) -> int:
    """
    Stream exact dedup from JSONL to JSONL using metadata hash field.
    Keeps first occurrence; preserves order.

    Returns: number of kept records.
    """
    seen: set[str] = set()
    kept = 0

    with open(in_path, "r", encoding=encoding) as fin, open(out_path, "w", encoding=encoding) as fout:
        for lineno, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj: Dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON at line {lineno}: {e}") from e

            h = obj.get(hash_field)
            if not h:
                raise KeyError(f"Missing '{hash_field}' at line {lineno}")

            if h in seen:
                continue

            seen.add(h)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    return kept

