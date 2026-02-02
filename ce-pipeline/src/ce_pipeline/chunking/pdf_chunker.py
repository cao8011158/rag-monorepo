from typing import List
from transformers import AutoTokenizer
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer


def pdf_chunker_contextualized_strings(pdf_path: str, max_tokens: int = 256) -> List[str]:
    """
    PDF -> DoclingDocument -> HybridChunker -> contextualized string list
    """

    # 1. tokenizer
    EMBED_MODEL_ID = "intfloat/e5-base-v2"
    hf_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID, use_fast=True)

    docling_tokenizer = HuggingFaceTokenizer(
        tokenizer=hf_tokenizer,
        max_tokens=max_tokens,
    )

    # 2. chunker
    chunker = HybridChunker(
        tokenizer=docling_tokenizer,
        merge_peers=True,
    )

    # 3. pdf -> DoclingDocument
    doc = DocumentConverter().convert(pdf_path).document

    # 4. chunk
    chunk_iter = chunker.chunk(dl_doc=doc)

    contextualized_texts: List[str] = []

    for i, chunk in enumerate(chunk_iter):
        enriched_text = chunker.contextualize(chunk=chunk)

        contextualized_texts.append(enriched_text)

        # optional debug print
        print(f"=== {i} ===")
        print(f"chunk.text:\n{chunk.text[:300]!r}…")
        print(f"chunker.contextualize(chunk):\n{enriched_text[:300]!r}…\n")

    return contextualized_texts
