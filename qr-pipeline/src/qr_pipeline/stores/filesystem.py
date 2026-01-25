from __future__ import annotations

from pathlib import Path
from typing import Iterable

from qr_pipeline.stores.base import Store


class FilesystemStore(Store):
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _p(self, path: str) -> Path:
        # treat 'path' as POSIX-like relative path under root
        p = Path(path)
        return (self.root / p).resolve()

    def exists(self, path: str) -> bool:
        return self._p(path).exists()

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        return self._p(path).read_text(encoding=encoding)

    def write_text(self, path: str, data: str, encoding: str = "utf-8") -> None:
        p = self._p(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(data, encoding=encoding)

    def read_bytes(self, path: str) -> bytes:
        return self._p(path).read_bytes()

    def write_bytes(self, path: str, data: bytes) -> None:
        p = self._p(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def list(self, prefix: str = "") -> Iterable[str]:
        base = self._p(prefix)
        if not base.exists():
            return []
        if base.is_file():
            return [prefix]
        out: list[str] = []
        for p in sorted(base.rglob("*")):
            if p.is_file():
                out.append(str(p.relative_to(self.root)).replace("\\", "/"))
        return out
