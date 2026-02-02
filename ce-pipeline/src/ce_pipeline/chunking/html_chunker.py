from __future__ import annotations
from pathlib import Path
from typing import List, Union

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import HTMLSemanticPreservingSplitter


def html_chunker(html_real_path: Union[str, Path]) -> List[Document]:
    """
    Input: html real path
    Output: List[langchain_core.documents.Document]
    """
    html_real_path = Path(html_real_path)
    if not html_real_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_real_path}")

    # Read HTML
    try:
        html_string = html_real_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        html_string = html_real_path.read_text(encoding="utf-8", errors="ignore")

    # -------- extract title with bs4 --------
    soup = BeautifulSoup(html_string, "html.parser")
    html_title = None
    if soup.title and soup.title.string:
        html_title = soup.title.string.strip()

    # ---- hard-coded splitter config ----
    splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=[
            ("h1", "Header 1"),
            ("h2", "Header 2"),
        ],
        separators=["\n\n", "\n", ".", "!", "?", " "],
        max_chunk_size=1000,
        preserve_images=True,
        preserve_videos=True,
        elements_to_preserve=["table", "ul", "ol", "code"],
        denylist_tags=["script", "style", "head"],
    )

    documents: List[Document] = splitter.split_text(html_string)

    # ---- enrich metadata ----
    for doc in documents:
        if html_title:
            doc.metadata["title"] = html_title

    return documents
