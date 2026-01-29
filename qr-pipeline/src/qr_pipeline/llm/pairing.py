# src/qr_pipeline/llm/pairing.py
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Sequence, Tuple, TypedDict, Optional


# -----------------------------
# Schemas
# -----------------------------
class ChunkDoc(TypedDict, total=False):
    chunk_id: str
    doc_id: str
    chunk_index: int
    chunk_text: str
    chunk_text_hash: str

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


class QueryPack(TypedDict):
    query: Query
    positives: List[ChunkDoc]
    negatives: List[ChunkDoc]
    meta: Dict[str, Any]


# -----------------------------
# Text helpers
# -----------------------------
_WS_RE = re.compile(r"\s+", re.UNICODE)


def _norm_text(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").strip()).lower()


def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _get_chunk_id(d: Any) -> str:
    if not isinstance(d, dict):
        return ""
    return _safe_str(d.get("chunk_id")).strip()


def _get_chunk_text(d: Any) -> str:
    if not isinstance(d, dict):
        return ""
    return _safe_str(d.get("chunk_text")).strip()


# -----------------------------
# Vector helpers (optional)
# -----------------------------
def _to_2d_float_array(x: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        return x

    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        x = x.detach().cpu().numpy()

    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _dot(a: Any, b: Any) -> float:
    try:
        import numpy as np
    except Exception:
        return float(sum(float(x) * float(y) for x, y in zip(a, b)))

    a1 = np.asarray(a, dtype=np.float32).reshape(-1)
    b1 = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(a1.dot(b1))


# -----------------------------
# Main (reranker + margin)
# -----------------------------
def build_pairs_for_query(
    *,
    query: Query,
    source_doc: ChunkDoc,
    candidate_docs: Sequence[ChunkDoc],  # from retriever (top_k=30). order doesn't matter; reranker will rerank.
    reranker: Any,                       # reranker.score(query_text, docs_texts)->list[float]
    embedder: Optional[Any] = None,       # optional: embedder.encode_passages(list[str])->vecs (normalized recommended)
    # knobs
    max_extra_positives: int = 2,
    extra_pos_margin: float = 0.8,        # margin m in (s_i - s_10 >= m)
    neg_rank_start: int = 10,             # 1-indexed rank start
    neg_rank_end: int = 24,               # 1-indexed rank end (inclusive)
    num_hard_negatives: int = 15,         # final cap
    cosine_threshold: float = 0.92,       # only used if embedder is provided
    enable_text_hash_dedup: bool = True,
) -> Tuple[QueryPack, Dict[str, Any]]:
    if not hasattr(reranker, "score"):
        raise TypeError("reranker must have method .score(query_text, docs_texts)->list[float].")

    query_text = _safe_str(query.get("query_text")).strip()
    if not query_text:
        raise ValueError("query['query_text'] is required and must be non-empty.")

    src_cid = _get_chunk_id(source_doc)
    if not src_cid:
        raise ValueError("source_doc['chunk_id'] is required and must be non-empty.")

    src_text = _get_chunk_text(source_doc)
    if not src_text:
        raise ValueError("source_doc['chunk_text'] is required and must be non-empty.")

    # copy query (avoid side-effects)
    query_out: Query = dict(query)
    scids = query_out.get("source_chunk_ids")
    if not isinstance(scids, list) or not scids:
        query_out["source_chunk_ids"] = [src_cid]

    stats: Dict[str, Any] = {
        "num_candidates_in": len(candidate_docs),
        "max_extra_positives": int(max_extra_positives),
        "extra_pos_margin": float(extra_pos_margin),
        "neg_rank_start": int(neg_rank_start),
        "neg_rank_end": int(neg_rank_end),
        "num_hard_negatives": int(num_hard_negatives),
        "enable_text_hash_dedup": bool(enable_text_hash_dedup),
        "cosine_threshold": float(cosine_threshold),
        "used_embedder_cosine_filter": bool(embedder is not None),
    }

    # ---- build doc_by_id (first wins), ensure source overrides
    doc_by_id: Dict[str, ChunkDoc] = {}
    dup_cid = 0
    bad_no_cid = 0
    for d in candidate_docs:
        cid = _get_chunk_id(d)
        if not cid:
            bad_no_cid += 1
            continue
        if cid in doc_by_id:
            dup_cid += 1
            continue
        doc_by_id[cid] = d

    doc_by_id[src_cid] = source_doc
    if bad_no_cid:
        stats["bad_candidate_no_chunk_id"] = bad_no_cid
    if dup_cid:
        stats["dup_candidate_chunk_id"] = dup_cid

    # ---- rerank: score [source_doc] + candidates (source score first)
    cand_ids: List[str] = []
    cand_texts: List[str] = []
    for cid, d in doc_by_id.items():
        if cid == src_cid:
            continue
        txt = _get_chunk_text(d)
        if not txt:
            continue
        cand_ids.append(cid)
        cand_texts.append(txt)

    stats["num_candidates_scored"] = len(cand_ids)

    # If no candidates (only source exists), we still return with source_score only
    if not cand_ids:
        source_score_only = reranker.score(query_text, [src_text])
        if not isinstance(source_score_only, list) or len(source_score_only) != 1:
            raise RuntimeError("reranker.score returned invalid source-only score.")
        src_score = float(source_score_only[0])
        stats["source_score"] = src_score
        stats["rerank_scores"] = [src_score]  # source at first
        pack: QueryPack = {
            "query": query_out,
            "positives": [source_doc],
            "negatives": [],
            "meta": {"stats": stats},
        }
        stats["num_samples"] = 1
        return pack, stats

    # score source + candidates together
    all_texts = [src_text] + cand_texts
    all_scores = reranker.score(query_text, all_texts)
    if not isinstance(all_scores, list) or len(all_scores) != len(all_texts):
        raise RuntimeError(
            f"reranker.score returned invalid scores: len(scores)={len(all_scores)} len(texts)={len(all_texts)}"
        )

    source_score = float(all_scores[0])
    cand_scores = [float(s) for s in all_scores[1:]]

    stats["source_score"] = source_score

    # sort candidates by score desc (training logic uses this)
    ranked = sorted(zip(cand_ids, cand_scores), key=lambda x: float(x[1]), reverse=True)
    ranked_ids = [cid for cid, _ in ranked]
    ranked_scores = [float(s) for _, s in ranked]  # candidates only, sorted

    # ✅ 你要的：source 分数放第一个，然后接 candidates(sorted) 的分数
    stats["rerank_scores"] = [source_score] + ranked_scores

    # ---- determine s10 (rank is 1-indexed)  (based on candidates, not including source)
    if len(ranked_scores) >= int(neg_rank_start):
        s10 = ranked_scores[int(neg_rank_start) - 1]
        stats["s10_score"] = float(s10)
        allow_extra_pos = True
    else:
        stats["s10_score"] = None
        allow_extra_pos = False
        stats["note"] = "not enough reranked candidates to define s10; extra positives disabled"

    # ---- positives:
    kept_pos_ids: List[str] = [src_cid]

    extra_ids: List[str] = []
    if allow_extra_pos and int(max_extra_positives) > 0:
        m = float(extra_pos_margin)
        k = int(max_extra_positives)

        for i in range(min(len(ranked_ids), max(2, k))):  # look at top few
            cid = ranked_ids[i]
            si = ranked_scores[i]
            if (si - s10) >= m:
                extra_ids.append(cid)
            if len(extra_ids) >= k:
                break

    stats["extra_pos_by_margin_raw"] = list(extra_ids)

    # optional: hash dedup against source/previous positives
    if enable_text_hash_dedup:
        seen_h: set[str] = set()

        def _get_hash(cid: str) -> str:
            doc = doc_by_id.get(cid, {})
            h = _safe_str(doc.get("chunk_text_hash")).strip()
            if h:
                return h
            return _sha1(_norm_text(_get_chunk_text(doc)))

        seen_h.add(_get_hash(src_cid))

        extra_dedup: List[str] = []
        for cid in extra_ids:
            h = _get_hash(cid)
            if h in seen_h:
                continue
            seen_h.add(h)
            extra_dedup.append(cid)
        extra_ids = extra_dedup

    # optional: cosine dedup against source/positives (if embedder provided)
    if embedder is not None and extra_ids:
        pos_texts = [_get_chunk_text(doc_by_id[src_cid])] + [_get_chunk_text(doc_by_id[cid]) for cid in extra_ids]
        vecs = _to_2d_float_array(embedder.encode_passages(pos_texts))

        keep_extra: List[str] = []
        kept_vec_idx = [0]  # source idx=0
        for i, cid in enumerate(extra_ids, start=1):
            keep = True
            for j in kept_vec_idx:
                sim = _dot(vecs[i], vecs[j])
                if sim > float(cosine_threshold):
                    keep = False
                    break
            if keep:
                keep_extra.append(cid)
                kept_vec_idx.append(i)
        extra_ids = keep_extra

    kept_pos_ids.extend(extra_ids)
    stats["num_pos_kept_final"] = len(kept_pos_ids)
    stats["num_extra_pos_final"] = max(0, len(kept_pos_ids) - 1)

    kept_pos_set = set(kept_pos_ids)

    # ---- negatives: take ranks [neg_rank_start, neg_rank_end], 1-indexed, inclusive (candidates only)
    ns = int(neg_rank_start)
    ne = int(neg_rank_end)
    if ns < 1:
        ns = 1
    if ne < ns:
        ne = ns

    start0 = ns - 1
    end0 = min(ne, len(ranked_ids))
    window_ids = ranked_ids[start0:end0]

    neg_pool = [cid for cid in window_ids if cid not in kept_pos_set]
    stats["num_neg_pool_window"] = len(neg_pool)

    filtered_neg_ids: List[str] = []
    if embedder is not None and neg_pool:
        pos_texts2 = [_get_chunk_text(doc_by_id[pid]) for pid in kept_pos_ids]
        neg_texts2 = [_get_chunk_text(doc_by_id[nid]) for nid in neg_pool]

        pos_vecs = _to_2d_float_array(embedder.encode_passages(pos_texts2))
        neg_vecs = _to_2d_float_array(embedder.encode_passages(neg_texts2))

        for i, nid in enumerate(neg_pool):
            max_sim = -1.0
            for p in range(len(kept_pos_ids)):
                sim = _dot(pos_vecs[p], neg_vecs[i])
                if sim > max_sim:
                    max_sim = sim
            if max_sim > float(cosine_threshold):
                continue
            filtered_neg_ids.append(nid)
    else:
        filtered_neg_ids = neg_pool

    stats["num_neg_after_cos_filter"] = len(filtered_neg_ids)

    if enable_text_hash_dedup and filtered_neg_ids:
        seen_h2: set[str] = set()
        deduped: List[str] = []
        for nid in filtered_neg_ids:
            doc = doc_by_id.get(nid, {})
            h = _safe_str(doc.get("chunk_text_hash")).strip()
            if not h:
                h = _sha1(_norm_text(_get_chunk_text(doc)))
            if h in seen_h2:
                continue
            seen_h2.add(h)
            deduped.append(nid)
        filtered_neg_ids = deduped

    stats["num_neg_after_hash_dedup"] = len(filtered_neg_ids)

    k_neg = max(0, int(num_hard_negatives))
    final_neg_ids = filtered_neg_ids[:k_neg]
    stats["num_neg_final"] = len(final_neg_ids)
    stats["neg_shortage"] = max(0, k_neg - len(final_neg_ids))

    pos_docs_final: List[ChunkDoc] = [doc_by_id[pid] for pid in kept_pos_ids if pid in doc_by_id]
    neg_docs_final: List[ChunkDoc] = [doc_by_id[nid] for nid in final_neg_ids if nid in doc_by_id]

    pack: QueryPack = {
        "query": query_out,
        "positives": pos_docs_final,
        "negatives": neg_docs_final,
        "meta": {"stats": stats},
    }

    stats["num_samples"] = 1
    return pack, stats
