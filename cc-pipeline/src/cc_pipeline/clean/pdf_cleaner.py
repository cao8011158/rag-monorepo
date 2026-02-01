from __future__ import annotations

import json
from io import BytesIO
from typing import Any, Dict

from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter


def pdf_to_docling_dict(pdf_bytes: bytes, filename: str = "document.pdf") -> Dict[str, Any]:
    """
    Convert PDF bytes -> DoclingDocument -> Python dict (Docling JSON structure).
    """
    source = DocumentStream(name=filename, stream=BytesIO(pdf_bytes))
    converter = DocumentConverter()
    result = converter.convert(source)  # returns ConversionResult
    doc = result.document
    return doc.export_to_dict()  # official API :contentReference[oaicite:3]{index=3}


def pdf_to_docling_json(pdf_bytes: bytes, filename: str = "document.pdf") -> str:
    """
    Convert PDF bytes -> Docling JSON string.
    """
    data = pdf_to_docling_dict(pdf_bytes, filename=filename)
    return json.dumps(data, ensure_ascii=False)
