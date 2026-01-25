from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import os

from .base import Store


@dataclass
class FilesystemStore(Store):
    root: Path

    def _abspath(self, path: str) -> Path:
        # treat input as posix-like relative path
        rel = Path(path)
        return (self.root / rel).resolve()

    def exists(self, path: str) -> bool:
        return self._abspath(path).exists()

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        return self._abspath(path).read_text(encoding=encoding)

    def write_text(self, path: str, content: str, encoding: str = "utf-8") -> None:
        p = self._abspath(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding)

    def read_bytes(self, path: str) -> bytes:
        return self._abspath(path).read_bytes()

    def write_bytes(self, path: str, content: bytes) -> None:
        p = self._abspath(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)

    def append_bytes(self, path: str, content: bytes) -> None:
        p = self._abspath(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("ab") as f:
            f.write(content)

    def list(self, prefix: str) -> Iterable[str]:
        base = self._abspath(prefix)
        if not base.exists():
            return []
        if base.is_file():
            return [prefix]
        out: list[str] = []
        for p in base.rglob("*"):
            if p.is_file():
                out.append(str(p.relative_to(self.root)).replace(os.sep, "/"))
        return out
