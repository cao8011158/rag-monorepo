from __future__ import annotations

import io
from pypdf import PdfReader

def pdf_to_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    text = "\n".join(parts)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)
