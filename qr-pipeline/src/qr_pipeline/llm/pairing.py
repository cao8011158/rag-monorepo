# src/qr_pipeline/llm/pairing.py
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Sequence, Tuple, TypedDict


# -----------------------------
# Schemas (UPDATED)
# -----------------------------
class ChunkDoc(TypedDict, total=False):
    # required
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
    # required
    query_text: str
    # NEW: store which source chunk(s) produced this query
    source_chunk_ids: List[str]

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


def _safe_json_loads(s: str) -> Dict[str, Any]:
    raw = (s or "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    l = raw.find("{")
    r = raw.rfind("}")
    if 0 <= l < r:
        cand = raw[l : r + 1]
        try:
            obj = json.loads(cand)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


# -----------------------------
# Candidate labeling (C1, C2, ...)
# -----------------------------
def _label_candidates(candidate_docs: Sequence[ChunkDoc]) -> Tuple[List[ChunkDoc], Dict[str, str], Dict[str, str]]:
    """
    Return:
      - docs (list, same order)
      - label_to_chunkid: {"C1": "<chunk_id>", ...}
      - chunkid_to_label: {"<chunk_id>": "C1", ...}
    Labels are assigned by order (RRF order).
    """
    docs = list(candidate_docs)
    label_to_chunkid: Dict[str, str] = {}
    chunkid_to_label: Dict[str, str] = {}
    for i, d in enumerate(docs, start=1):
        lab = f"C{i}"
        cid = d["chunk_id"]
        label_to_chunkid[lab] = cid
        chunkid_to_label[cid] = lab
    return docs, label_to_chunkid, chunkid_to_label


# -----------------------------
# Prompting (positives only) - NO url/title
# -----------------------------
def build_candidate_classify_prompt(
    query_text: str,
    labeled_docs: Sequence[Tuple[str, ChunkDoc]],  # [("C1", doc), ...]
    *,
    max_extra_positives: int = 2,
    include_one_shot: bool = True,
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
    lines.append(f"- You may output at most {max_extra_positives} POSITIVE documents.")
    lines.append("")
    lines.append("Hard rules:")
    lines.append("- You MUST only use the provided labels (C1, C2, ...).")
    lines.append("- Output MUST be a single valid JSON object only.")
    lines.append("- No extra text. No markdown. No explanations.")
    lines.append("")
    lines.append("Output JSON schema:")
    lines.append("{")
    lines.append('  "positives": [')
    lines.append('    { "doc_id": "C2", "evidence": "verbatim quote from the document" }')
    lines.append("  ]")
    lines.append("}")
    lines.append("")
    if include_one_shot:
        lines.append("One-shot example (FORMAT ONLY; do not reuse the content):")
        lines.append("Query: What is the capital of France?")
        lines.append("Candidate documents:")
        lines.append("- doc_id: C1")
        lines.append("  text: Paris is the capital and most populous city of France.")
        lines.append("- doc_id: C2")
        lines.append("  text: Berlin is the capital of Germany.")
        lines.append("")
        lines.append("Output JSON:")
        lines.append("{")
        lines.append('  "positives": [')
        lines.append('    {')
        lines.append('      "doc_id": "C1",')
        lines.append('      "evidence": "Paris is the capital and most populous city of France."')
        lines.append("    }")
        lines.append("  ]")
        lines.append("}")
        lines.append("")

    lines.append(f"Query: {query_text}")
    lines.append("")
    lines.append("Candidate documents:")
    for lab, d in labeled_docs:
        lines.append(f"- doc_id: {lab}")
        lines.append(f"  text: {_clip(d.get('chunk_text', ''))}")
    return "\n".join(lines)


# -----------------------------
# Main (UPDATED I/O + Output Pack)
# -----------------------------
def build_pairs_for_query(
    *,
    query: Query,
    source_doc: ChunkDoc,                 # known positive (a chunk)
    candidate_docs: Sequence[ChunkDoc],   # RRF-ordered candidates (may include source)
    llm: Any,                             # llm.generate(prompt)->str
    embedder: Any,                        # embedder.encode_passages(list[str])->np.ndarray (normalized)
    # knobs
    max_extra_positives: int = 2,
    cosine_threshold: float = 0.92,
    num_hard_negatives: int = 15,
    enable_text_hash_dedup: bool = True,
    include_one_shot: bool = True,
) -> Tuple[QueryPack, Dict[str, Any]]:
    if not hasattr(llm, "generate"):
        raise TypeError("llm must have method .generate(prompt)->str.")
    if not hasattr(embedder, "encode_passages"):
        raise TypeError("embedder must have method .encode_passages(list[str])->np.ndarray.")

    query_text = str(query.get("query_text") or "").strip()
    if not query_text:
        raise ValueError("query['query_text'] is required and must be non-empty.")

    # Ensure source_chunk_ids exists (best-effort)
    if "source_chunk_ids" not in query or not isinstance(query.get("source_chunk_ids"), list):
        query["source_chunk_ids"] = [source_doc["chunk_id"]]

    stats: Dict[str, Any] = {
        "num_candidates_in": len(candidate_docs),
        "max_extra_positives": max_extra_positives,
        "cosine_threshold": cosine_threshold,
        "num_hard_negatives": num_hard_negatives,
        "enable_text_hash_dedup": enable_text_hash_dedup,
        "include_one_shot": include_one_shot,
    }

    # ---- doc lookup by chunk_id
    doc_by_id: Dict[str, ChunkDoc] = {}
    for d in candidate_docs:
        cid = d["chunk_id"]
        if cid not in doc_by_id:
            doc_by_id[cid] = d
    doc_by_id[source_doc["chunk_id"]] = source_doc

    # ---- LLM sees only extra candidates (exclude source chunk)
    cand_for_llm: List[ChunkDoc] = [d for d in candidate_docs if d["chunk_id"] != source_doc["chunk_id"]]
    stats["num_candidates_for_llm"] = len(cand_for_llm)

    # ---- label them as C1, C2, ... (RRF order)
    cand_for_llm_list, label_to_real, real_to_label = _label_candidates(cand_for_llm)
    valid_labels = set(label_to_real.keys())
    labeled_docs: List[Tuple[str, ChunkDoc]] = [(real_to_label[d["chunk_id"]], d) for d in cand_for_llm_list]

    # ---- prompt + parse
    prompt = build_candidate_classify_prompt(
        query_text=query_text,
        labeled_docs=labeled_docs,
        max_extra_positives=max_extra_positives,
        include_one_shot=include_one_shot,
    )
    raw = llm.generate(prompt)
    out = _safe_json_loads(raw)
    pos_raw = out.get("positives") or []

    # ---- 1) collect ALL valid LLM positives (labels), do NOT truncate yet
    llm_pos_labels: List[str] = []
    invalid_label = 0
    if isinstance(pos_raw, list):
        for item in pos_raw:
            if not isinstance(item, dict):
                continue
            lab = str(item.get("doc_id") or "").strip()
            if lab not in valid_labels:
                invalid_label += 1
                continue
            llm_pos_labels.append(lab)

    # de-dup labels preserving first occurrence, then sort by numeric suffix to guarantee RRF order.
    seen_pos: set[str] = set()
    llm_pos_labels_uniq: List[str] = []
    for lab in llm_pos_labels:
        if lab in seen_pos:
            continue
        seen_pos.add(lab)
        llm_pos_labels_uniq.append(lab)

    def _label_rank(lab: str) -> int:
        m = re.match(r"^C(\d+)$", lab)
        return int(m.group(1)) if m else 10**9

    llm_pos_labels_uniq.sort(key=_label_rank)

    stats["invalid_label"] = invalid_label
    stats["llm_pos_valid_total"] = len(llm_pos_labels_uniq)

    # map labels -> real chunk_ids
    llm_pos_real_ids: List[str] = [label_to_real[lab] for lab in llm_pos_labels_uniq]

    # ---- 2) cosine dedup over (source + ALL llm positives), keep source first
    pos_ids_pre_dedup: List[str] = [source_doc["chunk_id"]] + llm_pos_real_ids

    if len(pos_ids_pre_dedup) <= 1:
        kept_pos_ids = pos_ids_pre_dedup
    else:
        pos_texts = [doc_by_id[cid]["chunk_text"] for cid in pos_ids_pre_dedup]
        pos_vecs = embedder.encode_passages(pos_texts)  # normalized (P, D)

        kept_pos_ids = [pos_ids_pre_dedup[0]]  # source always first
        kept_idx = [0]

        for i in range(1, len(pos_ids_pre_dedup)):
            keep = True
            for j in kept_idx:
                sim = float(pos_vecs[i] @ pos_vecs[j])
                if sim > cosine_threshold:
                    keep = False
                    break
            if keep:
                kept_pos_ids.append(pos_ids_pre_dedup[i])
                kept_idx.append(i)

    stats["num_pos_after_cos_dedup"] = len(kept_pos_ids)

    # ---- 3) truncate extras AFTER dedup
    extras_after_dedup = kept_pos_ids[1:]
    extras_after_dedup = extras_after_dedup[: max(0, int(max_extra_positives))]
    kept_pos_ids = [kept_pos_ids[0]] + extras_after_dedup

    stats["num_pos_kept_final"] = len(kept_pos_ids)
    stats["num_extra_pos_final"] = max(0, len(kept_pos_ids) - 1)

    kept_pos_set = set(kept_pos_ids)

    # ---- 4) negatives pool (preserve RRF order): complement of positives
    neg_pool_real_ids: List[str] = [d["chunk_id"] for d in cand_for_llm_list if d["chunk_id"] not in kept_pos_set]
    stats["num_neg_pool"] = len(neg_pool_real_ids)

    # ---- 5) negative cosine filter vs each positive (>threshold => drop)
    neg_texts = [doc_by_id[nid]["chunk_text"] for nid in neg_pool_real_ids]
    if neg_texts:
        pos_texts2 = [doc_by_id[pid]["chunk_text"] for pid in kept_pos_ids]
        pos_vecs2 = embedder.encode_passages(pos_texts2)  # (P, D)
        neg_vecs = embedder.encode_passages(neg_texts)    # (N, D)

        filtered_neg_ids: List[str] = []
        for i, nid in enumerate(neg_pool_real_ids):
            sims = pos_vecs2 @ neg_vecs[i]  # (P,)
            max_sim = float(sims.max()) if hasattr(sims, "max") else float(max(sims))
            if max_sim > cosine_threshold:
                continue
            filtered_neg_ids.append(nid)
    else:
        filtered_neg_ids = []

    stats["num_neg_after_cos_filter"] = len(filtered_neg_ids)

    # ---- 6) negative hash dedup (preserve order)
    if enable_text_hash_dedup:
        seen_h: set[str] = set()
        deduped_neg_ids: List[str] = []
        for nid in filtered_neg_ids:
            h = _sha1(_norm_text(doc_by_id[nid]["chunk_text"]))
            if h in seen_h:
                continue
            seen_h.add(h)
            deduped_neg_ids.append(nid)
    else:
        deduped_neg_ids = filtered_neg_ids

    stats["num_neg_after_hash_dedup"] = len(deduped_neg_ids)

    # ---- 7) cap negatives AFTER all filters, keep RRF order
    k = max(0, int(num_hard_negatives))
    final_neg_ids = deduped_neg_ids[:k]
    neg_docs_final: List[ChunkDoc] = [doc_by_id[nid] for nid in final_neg_ids]
    stats["num_neg_final"] = len(final_neg_ids)

    # ---- 8) build ONE packed sample per query (positives + negatives)
    pos_docs_final: List[ChunkDoc] = [doc_by_id[pid] for pid in kept_pos_ids]

    pack: QueryPack = {
        "query": query,
        "positives": pos_docs_final,
        "negatives": neg_docs_final,
        "meta": {"stats": stats},
    }

    stats["num_samples"] = 1
    return pack, stats
