# src/qr_pipeline/llm/pairing.py
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple, TypedDict, cast

import numpy as np

from qr_pipeline.llm.doc_classification_gemini import run_gemini_classification_PN


# -----------------------------
# Types (match run_pairing.py)
# -----------------------------
class ChunkDoc(TypedDict, total=False):
    chunk_id: str
    doc_id: str
    chunk_index: int
    chunk_text: str
    chunk_text_hash: str
    # optional passthrough fields:
    url: str
    title: str
    source: str
    content_hash: str
    content_type: str
    fetched_at: str
    run_date: str


class Query(TypedDict, total=False):
    query_text: str
    source_chunk_ids: List[str]
    query_id: str
    domain: str


class QueryPack(TypedDict, total=False):
    query: Query
    positives: List[ChunkDoc]
    negatives: List[ChunkDoc]
    meta: Dict[str, Any]


# -----------------------------
# Small utils
# -----------------------------
def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _dedup_by_chunk_id_keep_order(docs: Sequence[ChunkDoc]) -> List[ChunkDoc]:
    out: List[ChunkDoc] = []
    seen: set[str] = set()
    for d in docs:
        cid = _safe_str(d.get("chunk_id")).strip()
        if not cid or cid in seen:
            continue
        seen.add(cid)
        out.append(d)
    return out


def _validate_positive_ids(positive_ids: List[Any], n: int) -> List[int]:
    """
    Gemini returns 0-based indices. We enforce:
      - int coercion
      - 0 <= id < n
      - dedup
      - stable order (by document order)
    """
    cleaned: List[int] = []
    for x in positive_ids:
        try:
            cleaned.append(int(x))
        except Exception:
            continue

    cleaned = [i for i in cleaned if 0 <= i < n]

    # dedup keep order
    seen: set[int] = set()
    dedup: List[int] = []
    for i in cleaned:
        if i in seen:
            continue
        seen.add(i)
        dedup.append(i)

    # stable by doc order
    s = set(dedup)
    return [i for i in range(n) if i in s]


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def _embed_passages(embedder: Any, texts: List[str]) -> np.ndarray:
    """
      DualInstructEmbedder provides:
      encode_passages(passages: List[str]) -> np.ndarray
    """
    fn = getattr(embedder, "encode_passages", None)
    if not callable(fn):
        raise AttributeError("embedder must provide encode_passages(passages: List[str]) -> np.ndarray")

    vecs = fn(texts)
    arr = np.asarray(vecs, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != len(texts):
        raise ValueError(f"embedder.encode_passages returned invalid shape: {arr.shape}")
    return arr


def _cosine_dedup_keep_order(
    *,
    source_doc: ChunkDoc,
    candidate_docs: Sequence[ChunkDoc],
    embedder: Any,
    cosine_threshold: float = 0.90,
) -> List[ChunkDoc]:
    """
    Greedy cosine dedup (stable order):

    - source_doc is ALWAYS kept, but it is NOT returned.
      It is only used as an always-present reference in the "kept set".
    - candidate_docs are scanned in order; a candidate is dropped if its cosine
      similarity with ANY kept vector (including source_doc) is > threshold.

    Returns:
      deduped candidates (NOT including source_doc)
    """
    src_text = _safe_str(source_doc.get("chunk_text")).strip()

    # filter empty candidate text to avoid embedder issues
    filtered_docs: List[ChunkDoc] = []
    filtered_texts: List[str] = []
    for d in candidate_docs:
        t = _safe_str(d.get("chunk_text")).strip()
        if not t:
            continue
        filtered_docs.append(d)
        filtered_texts.append(t)

    if not filtered_docs:
        return []

    # embed once: [source] + candidates
    all_texts = [src_text] + filtered_texts
    vecs = _embed_passages(embedder, all_texts)
    vecs = _normalize_rows(vecs)

    src_vec = vecs[0]      # (D,)
    cand_vecs = vecs[1:]   # (M, D)

    kept_docs: List[ChunkDoc] = []
    kept_vecs: List[np.ndarray] = [src_vec]  # include source in kept set

    for i, d in enumerate(filtered_docs):
        v = cand_vecs[i]
        K = np.stack(kept_vecs, axis=0).astype(np.float32)  # (k, D)
        sims = K @ v                                        # (k,)
        if float(np.max(sims)) > float(cosine_threshold):
            continue
        kept_docs.append(d)
        kept_vecs.append(v)

    return kept_docs


# -----------------------------
# Public API used by run_pairing
# -----------------------------
def build_pairs_for_query(
    *,
    query: Query,
    source_doc: ChunkDoc,
    candidate_docs: Sequence[ChunkDoc],
    cfg: Dict[str, Any],
    embedder: Any,
    cosine_threshold: float = 0.92,
) -> Tuple[QueryPack, Dict[str, Any]]:
    """
    Gemini PN pairing strategy (NO reranker scores, NO extra positives margin):

    Steps:
      1) Remove source_doc from retrieved candidate_docs (by chunk_id).
      2) Candidate internal dedup:
           - chunk_id dedup (always)
           - chunk_text_hash dedup (optional)
      3) Cosine dedup using [source_doc + candidates], where:
           - source_doc is always kept (as reference)
           - candidates with cosine > threshold to any kept are dropped
      4) Call Gemini classifier on deduped candidates ONLY (excluding source_doc):
           run_gemini_classification_PN(query, documents, cfg) -> {"positive_ids":[...]}
      5) positives = [source_doc] + candidates[positive_ids]
         negatives = remaining candidates
         NOTE: negatives may be empty; still emit QueryPack
    """
    qtext = _safe_str(query.get("query_text")).strip()
    src_cid = _safe_str(source_doc.get("chunk_id")).strip()

    # (1) remove source from candidates
    filtered: List[ChunkDoc] = []
    for d in candidate_docs:
        cid = _safe_str(d.get("chunk_id")).strip()
        if not cid:
            continue
        if src_cid and cid == src_cid:
            continue
        filtered.append(d)

    # (2) candidate internal dedup
    filtered = _dedup_by_chunk_id_keep_order(filtered)

    # (3) cosine dedup (source always kept as reference)
    deduped = _cosine_dedup_keep_order(
        source_doc=source_doc,
        candidate_docs=filtered,
        embedder=embedder,
        cosine_threshold=float(cosine_threshold),
    )

    # (4) Gemini classification on candidates ONLY (no source)
    documents = [_safe_str(d.get("chunk_text")).strip() for d in deduped]
    raw = run_gemini_classification_PN(query=qtext, documents=documents, cfg=cfg)

    pos_any = raw.get("positive_ids") if isinstance(raw, dict) else []
    pos_ids = _validate_positive_ids(cast(List[Any], pos_any), n=len(deduped))

    # (5) build positives/negatives
    extra_pos = [deduped[i] for i in pos_ids]
    pos_set = set(pos_ids)
    negatives = [d for i, d in enumerate(deduped) if i not in pos_set]

    positives: List[ChunkDoc] = [source_doc] + extra_pos  # source ALWAYS included

    pack: QueryPack = {
        "query": query,
        "positives": positives,
        "negatives": negatives,  # can be []
        "meta": {
            "method": "gemini_pn",
        },
    }

    stats: Dict[str, Any] = {
        "num_candidates_in": len(candidate_docs),
        "num_candidates_after_remove_source": len(filtered),
        "num_candidates_after_cos_dedup": len(deduped),
        "num_docs_sent_to_gemini": len(deduped),
        "num_extra_positives": len(extra_pos),
        "num_negatives": len(negatives),
    }

    return pack, stats
