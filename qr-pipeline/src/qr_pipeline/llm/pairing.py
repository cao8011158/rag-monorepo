# src/qr_pipeline/llm/pairing.py
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Sequence, Tuple, TypedDict, Optional


# -----------------------------
# Schemas
# -----------------------------
class ChunkDoc(TypedDict, total=False):
    # required-ish (but total=False => validate at runtime)
    chunk_id: str
    doc_id: str
    chunk_index: int
    chunk_text: str
    chunk_text_hash: str

    # optional (copied from document)
    url: str
    title: str
    source: str
    content_hash: str
    content_type: str
    fetched_at: str
    run_date: str


class Query(TypedDict, total=False):
    query_text: str
    source_chunk_ids: List[str]  # NEW: which source chunk(s) produced this query

    # optional
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
# Candidate labeling (C1, C2, ...)
# -----------------------------
def _label_candidates(
    candidate_docs: Sequence[ChunkDoc],
    *,
    stats: Optional[Dict[str, Any]] = None,
) -> Tuple[List[ChunkDoc], Dict[str, str], Dict[str, str]]:
    """
    Return:
      - docs (list, same order, skipping invalid chunk_id)
      - label_to_chunkid: {"C1": "<chunk_id>", ...}
      - chunkid_to_label: {"<chunk_id>": "C1", ...}
    Labels are assigned by order (RRF order).
    """
    out_docs: List[ChunkDoc] = []
    label_to_chunkid: Dict[str, str] = {}
    chunkid_to_label: Dict[str, str] = {}

    bad_no_cid = 0
    for d in list(candidate_docs):
        cid = _get_chunk_id(d)
        if not cid:
            bad_no_cid += 1
            continue
        out_docs.append(d)

    if stats is not None and bad_no_cid:
        stats["bad_candidate_no_chunk_id"] = stats.get("bad_candidate_no_chunk_id", 0) + bad_no_cid

    for i, d in enumerate(out_docs, start=1):
        lab = f"C{i}"
        cid = _get_chunk_id(d)
        label_to_chunkid[lab] = cid
        chunkid_to_label[cid] = lab

    return out_docs, label_to_chunkid, chunkid_to_label


# -----------------------------
# Prompting (labels-only)
# -----------------------------
def build_candidate_classify_prompt_labels_only(
    query_text: str,
    labeled_docs: Sequence[Tuple[str, ChunkDoc]],
    *,
    max_extra_positives: int = 2,
) -> str:
    per_doc_cap = 900

    def _clip(t: str) -> str:
        t = (t or "").strip()
        return t if len(t) <= per_doc_cap else (t[:per_doc_cap] + " ...")

    lines: List[str] = []
    lines.append("You are a strict judge that matches a query to candidate documents.")
    lines.append("Precision is more important than recall.")
    lines.append("")
    lines.append("Task:")
    lines.append("- Evaluate EACH candidate document independently.")
    lines.append("- A document is POSITIVE only if it directly and explicitly answers the query.")
    lines.append("- If you are unsure, DO NOT mark it as positive.")
    lines.append("")
    lines.append("Output rules (IMPORTANT):")
    lines.append("- Output EXACTLY ONE LINE.")
    lines.append("- The line MUST start with: POSITIVES:")
    lines.append("- After the colon, output either:")
    lines.append("  (A) the word NONE")
    lines.append("  OR")
    lines.append("  (B) one or more labels separated by a single space.")
    lines.append("- Each label must match the pattern: C<number> (e.g., C1, C2, C10).")
    lines.append("- If none are positive: POSITIVES: NONE")
    lines.append("Examples:")
    lines.append("POSITIVES: C1")
    lines.append("POSITIVES: C1 C3")
    lines.append("POSITIVES: NONE")
    lines.append("")
    lines.append(f"Query: {query_text}")
    lines.append("")
    lines.append("Candidate documents:")
    for lab, d in labeled_docs:
        lines.append(f"- {lab}: {_clip(_get_chunk_text(d))}")
    return "\n".join(lines)


# -----------------------------
# Parser (strict-first + fallback)
# -----------------------------
# ✅ IMPORTANT FIX:
# If you used r"\bC(\d+)\b", findall() returns ["1","2"] which won't match valid_labels {"C1","C2"}.
# Use capturing group that includes the "C".
_LABEL_RE = re.compile(r"\b(C\d+)\b", re.IGNORECASE)


def parse_positives_from_labels_line(
    raw: str,
    *,
    valid_labels: set[str],
    max_n: int,
    stats: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """
    Best-practice parser:
      1) STRICT: find the first line that starts with "POSITIVES:" (case-insensitive).
      2) If not found, FALLBACK: regex-scan the entire output for labels like C12.
      3) Filter by valid_labels, de-dup preserving order, cap to max_n.

    Records debug signals in stats:
      - llm_parse_mode: "strict" or "fallback"
      - llm_parse_fail: 1 if nothing could be parsed
      - llm_raw_prefix: first 200 chars (for debugging)
    """
    text = (raw or "").strip()
    if stats is not None:
        stats["llm_raw_prefix"] = (text[:200] + ("..." if len(text) > 200 else "")) if text else ""

    if not text:
        if stats is not None:
            stats["llm_parse_fail"] = stats.get("llm_parse_fail", 0) + 1
        return []

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # ---- strict: find POSITIVES: line
    pos_line = None
    for ln in lines:
        if ln.upper().startswith("POSITIVES:"):
            pos_line = ln
            break

    cand: List[str]
    if pos_line is not None:
        payload = pos_line.split(":", 1)[1].strip()
        if payload.upper() == "NONE" or payload == "":
            if stats is not None:
                stats["llm_parse_mode"] = "strict"
            return []
        cand = payload.split()
        if stats is not None:
            stats["llm_parse_mode"] = "strict"
    else:
        # ---- fallback: scan whole text for labels
        cand = _LABEL_RE.findall(text)
        if stats is not None:
            stats["llm_parse_mode"] = "fallback"
            stats["llm_fallback_used"] = stats.get("llm_fallback_used", 0) + 1

    # ---- normalize, filter, dedup, cap
    out: List[str] = []
    seen = set()
    for x in cand:
        lab = _safe_str(x).strip().upper()
        if lab in valid_labels and lab not in seen:
            out.append(lab)
            seen.add(lab)
        if len(out) >= max(0, int(max_n)):
            break

    if not out and stats is not None:
        stats["llm_parse_fail"] = stats.get("llm_parse_fail", 0) + 1

    return out


# -----------------------------
# Vector helpers (robust)
# -----------------------------
def _to_2d_float_array(x: Any) -> Any:
    """
    Convert x (numpy array / torch tensor / list) to a 2D float array (N, D).
    Returns a numpy.ndarray if numpy is available; otherwise returns input as-is.
    """
    try:
        import numpy as np  # local import to avoid hard dependency surprises
    except Exception:
        return x

    # torch tensor?
    if hasattr(x, "detach") and hasattr(x, "cpu") and hasattr(x, "numpy"):
        x = x.detach().cpu().numpy()

    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _dot(a: Any, b: Any) -> float:
    """
    Dot product between 1D vectors, returns python float.
    """
    try:
        import numpy as np
    except Exception:
        # very last-resort fallback
        return float(sum(float(x) * float(y) for x, y in zip(a, b)))

    a1 = np.asarray(a, dtype=np.float32).reshape(-1)
    b1 = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(a1.dot(b1))


# -----------------------------
# Main
# -----------------------------
def build_pairs_for_query(
    *,
    query: Query,
    source_doc: ChunkDoc,                 # known positive (a chunk)
    candidate_docs: Sequence[ChunkDoc],   # RRF-ordered candidates (may include source)
    llm: Any,                             # llm.generate(prompt)->str
    embedder: Any,                        # embedder.encode_passages(list[str])->array/tensor (normalized recommended)
    # knobs
    max_extra_positives: int = 2,
    cosine_threshold: float = 0.92,
    num_hard_negatives: int = 15,
    enable_text_hash_dedup: bool = True,
) -> Tuple[QueryPack, Dict[str, Any]]:
    if not hasattr(llm, "generate"):
        raise TypeError("llm must have method .generate(prompt)->str.")
    if not hasattr(embedder, "encode_passages"):
        raise TypeError("embedder must have method .encode_passages(list[str])->np.ndarray/torch.Tensor.")

    query_text = _safe_str(query.get("query_text")).strip()
    if not query_text:
        raise ValueError("query['query_text'] is required and must be non-empty.")

    src_cid = _get_chunk_id(source_doc)
    if not src_cid:
        raise ValueError("source_doc['chunk_id'] is required and must be non-empty.")

    # copy query to avoid side effects
    query_out: Query = dict(query)

    # ensure source_chunk_ids exists (best-effort)
    scids = query_out.get("source_chunk_ids")
    if not isinstance(scids, list) or not scids:
        query_out["source_chunk_ids"] = [src_cid]

    stats: Dict[str, Any] = {
        "num_candidates_in": len(candidate_docs),
        "max_extra_positives": max_extra_positives,
        "cosine_threshold": cosine_threshold,
        "num_hard_negatives": num_hard_negatives,
        "enable_text_hash_dedup": enable_text_hash_dedup,
    }

    # ---- doc lookup by chunk_id (first occurrence in RRF order wins)
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

    if bad_no_cid:
        stats["bad_candidate_no_chunk_id"] = stats.get("bad_candidate_no_chunk_id", 0) + bad_no_cid
    if dup_cid:
        stats["dup_candidate_chunk_id"] = dup_cid

    # source_doc should override (known positive truth)
    doc_by_id[src_cid] = source_doc

    # ---- LLM sees only extra candidates (exclude source chunk)
    cand_for_llm: List[ChunkDoc] = [d for d in candidate_docs if _get_chunk_id(d) and _get_chunk_id(d) != src_cid]
    stats["num_candidates_for_llm_raw"] = len(cand_for_llm)

    # ---- label them as C1, C2, ... (RRF order), skipping invalid chunk_id
    cand_for_llm_list, label_to_real, real_to_label = _label_candidates(cand_for_llm, stats=stats)
    valid_labels = set(label_to_real.keys())
    labeled_docs: List[Tuple[str, ChunkDoc]] = [(real_to_label[_get_chunk_id(d)], d) for d in cand_for_llm_list]
    stats["num_candidates_for_llm"] = len(cand_for_llm_list)

    # ---- prompt + parse (labels-only)
    prompt = build_candidate_classify_prompt_labels_only(
        query_text=query_text,
        labeled_docs=labeled_docs,
        max_extra_positives=max_extra_positives,
    )
    raw = llm.generate(prompt)

    llm_pos_labels_uniq = parse_positives_from_labels_line(
        raw,
        valid_labels=valid_labels,
        max_n=max(0, int(max_extra_positives)),
        stats=stats,
    )
    stats["llm_pos_valid_total"] = len(llm_pos_labels_uniq)

    # map labels -> real chunk_ids (RRF order already preserved by parser)
    llm_pos_real_ids: List[str] = [label_to_real[lab] for lab in llm_pos_labels_uniq]

    # ---- 2) cosine dedup over (source + ALL llm positives), keep source first
    pos_ids_pre_dedup: List[str] = [src_cid] + llm_pos_real_ids

    kept_pos_ids: List[str]
    if len(pos_ids_pre_dedup) <= 1:
        kept_pos_ids = pos_ids_pre_dedup
    else:
        pos_texts = [_get_chunk_text(doc_by_id.get(cid, {})) for cid in pos_ids_pre_dedup]
        # embedder should return normalized vectors; we still handle shapes robustly
        pos_vecs = _to_2d_float_array(embedder.encode_passages(pos_texts))

        kept_pos_ids = [pos_ids_pre_dedup[0]]  # source always first
        kept_idx = [0]

        for i in range(1, len(pos_ids_pre_dedup)):
            keep = True
            for j in kept_idx:
                sim = _dot(pos_vecs[i], pos_vecs[j])
                if sim > float(cosine_threshold):
                    keep = False
                    break
            if keep:
                kept_pos_ids.append(pos_ids_pre_dedup[i])
                kept_idx.append(i)

    stats["num_pos_after_cos_dedup"] = len(kept_pos_ids)

    # ---- 3) truncate extras AFTER dedup
    k_extra = max(0, int(max_extra_positives))
    extras_after_dedup = kept_pos_ids[1:][:k_extra]
    kept_pos_ids = [kept_pos_ids[0]] + extras_after_dedup

    stats["num_pos_kept_final"] = len(kept_pos_ids)
    stats["num_extra_pos_final"] = max(0, len(kept_pos_ids) - 1)

    kept_pos_set = set(kept_pos_ids)

    # ---- 4) negatives pool (preserve RRF order): complement of positives among cand_for_llm_list
    neg_pool_real_ids: List[str] = []
    for d in cand_for_llm_list:
        cid = _get_chunk_id(d)
        if cid and cid not in kept_pos_set:
            neg_pool_real_ids.append(cid)
    stats["num_neg_pool"] = len(neg_pool_real_ids)

    # ---- 5) negative cosine filter vs each positive (>threshold => drop)
    filtered_neg_ids: List[str] = []
    if neg_pool_real_ids:
        pos_texts2 = [_get_chunk_text(doc_by_id.get(pid, {})) for pid in kept_pos_ids]
        neg_texts = [_get_chunk_text(doc_by_id.get(nid, {})) for nid in neg_pool_real_ids]

        pos_vecs2 = _to_2d_float_array(embedder.encode_passages(pos_texts2))  # (P,D)
        neg_vecs = _to_2d_float_array(embedder.encode_passages(neg_texts))    # (N,D)

        for i, nid in enumerate(neg_pool_real_ids):
            # compute max similarity to any positive
            max_sim = -1.0
            for p in range(len(kept_pos_ids)):
                sim = _dot(pos_vecs2[p], neg_vecs[i])
                if sim > max_sim:
                    max_sim = sim
            if max_sim > float(cosine_threshold):
                continue
            filtered_neg_ids.append(nid)

    stats["num_neg_after_cos_filter"] = len(filtered_neg_ids)

    # ---- 6) negative hash dedup (preserve order)
    if enable_text_hash_dedup:
        seen_h: set[str] = set()
        deduped_neg_ids: List[str] = []
        fallback_used = 0

        for nid in filtered_neg_ids:
            doc = doc_by_id.get(nid, {})
            h = _safe_str(doc.get("chunk_text_hash")).strip()
            if not h:
                fallback_used += 1
                h = _sha1(_norm_text(_get_chunk_text(doc)))
            if h in seen_h:
                continue
            seen_h.add(h)
            deduped_neg_ids.append(nid)

        if fallback_used:
            stats["neg_hash_fallback_used"] = fallback_used
    else:
        deduped_neg_ids = filtered_neg_ids

    stats["num_neg_after_hash_dedup"] = len(deduped_neg_ids)

    # ---- 7) cap negatives AFTER all filters, keep RRF order
    k_neg = max(0, int(num_hard_negatives))
    final_neg_ids = deduped_neg_ids[:k_neg]
    stats["num_neg_final"] = len(final_neg_ids)
    stats["neg_shortage"] = max(0, k_neg - len(deduped_neg_ids))

    # ---- 8) build packed sample
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
