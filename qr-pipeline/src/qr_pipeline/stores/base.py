from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable


class Store(ABC):
    """Minimal Store interface (filesystem / object storage compatible)."""

    @abstractmethod
    def exists(self, path: str) -> bool: ...

    @abstractmethod
    def read_text(self, path: str, encoding: str = "utf-8") -> str: ...

    @abstractmethod
    def write_text(self, path: str, data: str, encoding: str = "utf-8") -> None: ...

    @abstractmethod
    def read_bytes(self, path: str) -> bytes: ...

    @abstractmethod
    def write_bytes(self, path: str, data: bytes) -> None: ...

    @abstractmethod
    def list(self, prefix: str = "") -> Iterable[str]: ...
