from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import orjson

@dataclass
class ManifestEntry:
    url: str
    content_hash: str
    rel_path: str
    content_type: str
    fetched_at: str

def load_manifest(path: Path) -> dict[str, ManifestEntry]:
    if not path.exists():
        return {}
    out: dict[str, ManifestEntry] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = orjson.loads(line)
        out[obj["url"]] = ManifestEntry(**obj)
    return out

def write_manifest(path: Path, entries: list[ManifestEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for e in entries:
            f.write(orjson.dumps(e.__dict__) + b"\n")
