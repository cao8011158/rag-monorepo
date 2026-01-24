from __future__ import annotations

from pathlib import Path

class LocalStore:
    def __init__(self, root: Path):
        self.root = root

    def write_bytes(self, rel: str, data: bytes) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p

    def read_bytes(self, rel: str) -> bytes:
        return (self.root / rel).read_bytes()

    def exists(self, rel: str) -> bool:
        return (self.root / rel).exists()
