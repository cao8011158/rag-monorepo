from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class StorePath:
    # Logical path inside a store (posix-like).
    path: str


class Store(ABC):
    @abstractmethod
    def exists(self, path: str) -> bool: ...

    @abstractmethod
    def read_text(self, path: str, encoding: str = "utf-8") -> str: ...

    @abstractmethod
    def write_text(self, path: str, content: str, encoding: str = "utf-8") -> None: ...

    @abstractmethod
    def read_bytes(self, path: str) -> bytes: ...

    @abstractmethod
    def write_bytes(self, path: str, content: bytes) -> None: ...

    @abstractmethod
    def list(self, prefix: str) -> Iterable[str]: ...
