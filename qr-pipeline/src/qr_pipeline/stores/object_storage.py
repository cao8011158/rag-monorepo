from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .base import Store


@dataclass
class S3Store(Store):
    # Placeholder S3 store. Implement with boto3 if you want real S3 IO.
    bucket: str
    prefix: str

    def _key(self, path: str) -> str:
        p = path.lstrip("/")
        if self.prefix:
            return f"{self.prefix.rstrip('/')}/{p}"
        return p

    def exists(self, path: str) -> bool:
        raise NotImplementedError("Implement with boto3: head_object")

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        return self.read_bytes(path).decode(encoding)

    def write_text(self, path: str, content: str, encoding: str = "utf-8") -> None:
        self.write_bytes(path, content.encode(encoding))

    def read_bytes(self, path: str) -> bytes:
        raise NotImplementedError("Implement with boto3: get_object")

    def write_bytes(self, path: str, content: bytes) -> None:
        raise NotImplementedError("Implement with boto3: put_object")

    def list(self, prefix: str) -> Iterable[str]:
        raise NotImplementedError("Implement with boto3: list_objects_v2 pagination")
