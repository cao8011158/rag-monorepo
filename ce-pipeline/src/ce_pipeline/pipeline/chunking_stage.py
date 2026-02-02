# src/ce_pipeline/pipeline/chunking_stage.py
from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from ce_pipeline.settings import load_settings
from ce_pipeline.stores.registry import build_store_registry
from ce_pipeline.io.jsonl import read_jsonl, write_jsonl, append_jsonl

from ce_pipeline.chunking.html_chunker import html_chunker
from ce_pipeline.chunking.pdf_chunker import pdf_chunker_contextualized_strings

from ce_pipeline.embedding.embedder import DualInstructEmbedder
from ce_pipeline.processing.near_dedup import near_dedup_by_ann_faiss


# --------------------------
# local helpers
# --------------------------
def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _chunks_paths(cfg: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """
    Returns:
      out_store_name, base_dir, chunks_path, chunks_dedup_path
    """
    out_cfg = cfg["outputs"]["chunks"]
    out_store_name = out_cfg["store"]
    base = str(out_cfg["base"]).rstrip("/")

    filename = str(out_cfg.get("filename", "chunks.jsonl"))
    dedup_filename = str(out_cfg.get("dedup_filename", "chunks.dedup.jsonl"))

    chunks_path = f"{base}/{filename}"
    chunks_dedup_path = f"{base}/{dedup_filename}"
    return out_store_name, base, chunks_path, chunks_dedup_path


def _get_fs_root(cfg: Dict[str, Any], store_name: str) -> Path:
    """
    For filesystem store, real root path is cfg["stores"][store_name]["root"].
    """
    stores_cfg = cfg.get("stores", {})
    if store_name not in stores_cfg:
        raise KeyError(f"Store {store_name!r} not found in cfg['stores'].")
    root = stores_cfg[store_name].get("root")
    if not root:
        raise KeyError(f"Store {store_name!r} missing cfg['stores'][name]['root'].")
    return Path(root)


def _logical_to_real(root: Path, rel_path: str) -> str:
    return str((root / rel_path).resolve())


def _load_existing_chunk_ids(store: Any, chunks_path: str) -> Set[str]:
    if not store.exists(chunks_path):
        return set()
    s: Set[str] = set()
    for row in read_jsonl(store, chunks_path):
        cid = row.get("chunk_id")
        if isinstance(cid, str) and cid:
            s.add(cid)
    return s


def _format_html_chunk_text(meta: Dict[str, Any], page_content: str) -> str:
    """
    meta order is preserved as you confirmed.
    Returns:
      "Header 1:Main Title, Header 2:Section 1\n<page_content>"
    """
    parts: List[str] = []
    for k, v in meta.items():
        if v is None:
            continue
        parts.append(f"{k}:{v}")
    prefix = ", ".join(parts)
    if prefix:
        return (prefix + "\n" + (page_content or "")).strip()
    return (page_content or "").strip()


# --------------------------
# Chunk schema
# --------------------------
@dataclass
class ChunkDoc:
    chunk_id: str
    doc_id: str
    chunk_index: int
    chunk_text: str
    chunk_text_hash: str

    url: Optional[str] = None
    title: Optional[str] = None
    source: Optional[str] = None
    content_hash: Optional[str] = None
    content_type: Optional[str] = None
    fetched_at: Optional[str] = None
    run_date: Optional[str] = None


def _make_chunk_doc(
    *,
    chunk_text: str,
    doc_id: str,
    chunk_index: int,
    url: Optional[str],
    title: Optional[str],
    source: Optional[str],
    content_hash: Optional[str],
    content_type: Optional[str],
    fetched_at: Optional[str],
    run_date: str,
) -> ChunkDoc:
    h = sha256_hex(chunk_text)
    return ChunkDoc(
        chunk_id=h[:24],
        doc_id=doc_id,
        chunk_index=chunk_index,
        chunk_text=chunk_text,
        chunk_text_hash=h,
        url=url,
        title=title,
        source=source,
        content_hash=content_hash,
        content_type=content_type,
        fetched_at=fetched_at,
        run_date=run_date,
    )


def _chunk_one_manifest_doc(
    *,
    manifest_row: Dict[str, Any],
    fs_root: Path,
    pdf_max_tokens: int,
    run_date: str,
) -> List[ChunkDoc]:
    """
    Produce doc-internal deduped ChunkDocs for one manifest row, and assign chunk_index 0..n-1.
    doc_id = url (as you decided)
    """
    url = manifest_row.get("url")
    rel_path = manifest_row.get("rel_path")
    content_type = manifest_row.get("content_type")
    content_hash = manifest_row.get("content_hash")
    fetched_at = manifest_row.get("fetched_at")

    if not isinstance(url, str) or not url:
        return []
    if not isinstance(rel_path, str) or not rel_path:
        return []
    if not isinstance(content_type, str) or not content_type:
        return []

    doc_id = url
    source = rel_path
    real_path = _logical_to_real(fs_root, rel_path)

    out: List[ChunkDoc] = []
    seen: Set[str] = set()  # doc-internal exact dedup by chunk_id

    if content_type == "text/html":
        docs = html_chunker(real_path)  # List[langchain Document]
        for d in docs:
            meta = dict(getattr(d, "metadata", {}) or {})
            title = meta.pop("title", None)
            page_content = getattr(d, "page_content", "") or ""

            chunk_text = _format_html_chunk_text(meta, page_content)
            if not chunk_text:
                continue

            cd = _make_chunk_doc(
                chunk_text=chunk_text,
                doc_id=doc_id,
                chunk_index=0,  # filled later
                url=url,
                title=title if isinstance(title, str) else None,
                source=source,
                content_hash=content_hash if isinstance(content_hash, str) else None,
                content_type=content_type,
                fetched_at=fetched_at if isinstance(fetched_at, str) else None,
                run_date=run_date,
            )
            if cd.chunk_id in seen:
                continue
            seen.add(cd.chunk_id)
            out.append(cd)

    elif content_type == "application/pdf":
        texts = pdf_chunker_contextualized_strings(real_path, max_tokens=pdf_max_tokens)
        for t in texts:
            if not isinstance(t, str):
                continue
            chunk_text = t.strip()
            if not chunk_text:
                continue

            cd = _make_chunk_doc(
                chunk_text=chunk_text,
                doc_id=doc_id,
                chunk_index=0,  # filled later
                url=url,
                title=None,
                source=source,
                content_hash=content_hash if isinstance(content_hash, str) else None,
                content_type=content_type,
                fetched_at=fetched_at if isinstance(fetched_at, str) else None,
                run_date=run_date,
            )
            if cd.chunk_id in seen:
                continue
            seen.add(cd.chunk_id)
            out.append(cd)

    else:
        # ignore unknown types
        return []

    for i, cd in enumerate(out):
        cd.chunk_index = i

    return out


def _semantic_dedup(
    *,
    store: Any,
    chunks_path: str,
    chunks_dedup_path: str,
    cfg: Dict[str, Any],
) -> None:
    rows = list(read_jsonl(store, chunks_path))
    if not rows:
        write_jsonl(store, chunks_dedup_path, [])
        return

    texts: List[str] = []
    for r in rows:
        t = r.get("chunk_text")
        texts.append(t if isinstance(t, str) else "")

    emb_cfg = cfg["embedding"]
    inst = emb_cfg.get("instructions", {})
    embedder = DualInstructEmbedder(
        model_name=str(emb_cfg["model_name"]),
        passage_instruction=str(inst["passage"]),
        query_instruction=str(inst["query"]),
        batch_size=int(emb_cfg.get("batch_size", 64)),
        normalize_embeddings=bool(emb_cfg.get("normalize_embeddings", True)),
        device=emb_cfg.get("device"),
    )

    emb = embedder.encode_passages(texts)

    sd = cfg["processing"]["dedup"]["semantic_dedup"]
    res = near_dedup_by_ann_faiss(
        emb,
        threshold=float(sd.get("threshold", 0.95)),
        topk=int(sd.get("topk", 20)),
        hnsw_m=int(sd.get("hnsw_m", 32)),
        ef_construction=int(sd.get("ef_construction", 200)),
        ef_search=int(sd.get("ef_search", 64)),
        normalize=bool(sd.get("normalize", True)),
    )

    kept_rows = [rows[i] for i in res.kept_indices]
    write_jsonl(store, chunks_dedup_path, kept_rows)


def run_chunking_stage(config_path: str) -> None:
    cfg = load_settings(config_path)
    stores = build_store_registry(cfg)

    # input / manifest store
    in_store_name = cfg["input"]["input_store"]
    in_store = stores[in_store_name]

    # manifest path (logical)
    manifest_path = str(cfg["input"]["manifest_path"]).lstrip("/")

    # output store + paths
    out_store_name, _, chunks_path, chunks_dedup_path = _chunks_paths(cfg)
    out_store = stores[out_store_name]

    # fs root for real-path conversion
    # Your data root is in cfg["stores"]["fs_local"]["root"]
    fs_root = _get_fs_root(cfg, "fs_local")

    # knobs
    pdf_max_tokens = int(cfg.get("chunking", {}).get("pdf_max_tokens", 256))
    run_date = utc_now_iso_z()

    # global exact dedup based on existing chunks.jsonl
    existing_chunk_ids = _load_existing_chunk_ids(out_store, chunks_path)

    # stream append
    buffer: List[Dict[str, Any]] = []
    for m in read_jsonl(in_store, manifest_path):
        chunk_docs = _chunk_one_manifest_doc(
            manifest_row=m,
            fs_root=fs_root,
            pdf_max_tokens=pdf_max_tokens,
            run_date=run_date,
        )
        for cd in chunk_docs:
            if cd.chunk_id in existing_chunk_ids:
                continue
            existing_chunk_ids.add(cd.chunk_id)
            buffer.append(asdict(cd))

        if len(buffer) >= 2000:
            append_jsonl(out_store, chunks_path, buffer)
            buffer = []

    if buffer:
        append_jsonl(out_store, chunks_path, buffer)

    # semantic dedup
    sd = cfg["processing"]["dedup"]["semantic_dedup"]
    if bool(sd.get("enable", False)):
        _semantic_dedup(
            store=out_store,
            chunks_path=chunks_path,
            chunks_dedup_path=chunks_dedup_path,
            cfg=cfg,
        )
    else:
        # still write a dedup file for determinism
        write_jsonl(out_store, chunks_dedup_path, read_jsonl(out_store, chunks_path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/pipeline.yaml")
    args = ap.parse_args()
    run_chunking_stage(args.config)


if __name__ == "__main__":
    main()
