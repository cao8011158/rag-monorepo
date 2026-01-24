from __future__ import annotations

import hashlib

def sha256_hex(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def stable_url_hash(url: str) -> str:
    # short hash for filenames
    return sha256_hex(url.encode("utf-8"))[:24]
