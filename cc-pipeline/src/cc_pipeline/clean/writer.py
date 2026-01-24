from __future__ import annotations

from pathlib import Path
import orjson

def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as f:
        f.write(orjson.dumps(obj) + b"\n")
