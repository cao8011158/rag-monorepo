# src/qr_pipeline/llm/pairing.py
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Sequence, Tuple, TypedDict


# -----------------------------
# Schemas
# -----------------------------
class Document(TypedDict):
    doc_id: str
    text: str


class PairSample(TypedDict):
    query_text: str
    positive: Document
    negatives: List[Document]
    source_chunk: str


# -----------------------------
# Text helpers
# -----------------------------
_WS_RE = re.compile(r"\s+", re.UNICODE)
_SENT_SPLIT = re.compile(r"(?<=[。！？.!?])\s+|\n+", re.UNICODE)


def _norm_text(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").strip()).lower()


def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def _best_effort_contains(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    if needle in haystack:
        return True
    return _norm_text(needle) in _norm_text(haystack)


def _extract_source_chunk(source_doc_text: str, max_chars: int = 260) -> str:
    t = (source_doc_text or "").strip()
    if not t:
        return ""
    parts = [p.strip() for p in _SENT_SPLIT.split(t) if p.strip()]
    s = parts[0] if parts else t
    return s if len(s) <= max_chars else (s[:max_chars] + " ...")


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
def _label_candidates(candidate_docs: Sequence[Document]) -> Tuple[List[Document], Dict[str, str], Dict[str, str]]:
    """
    Return:
      - candidate_docs (as list, same order)
      - label_to_docid: {"C1": "<real_doc_id>", ...}
      - docid_to_label: {"<real_doc_id>": "C1", ...}
    Labels are assigned by order (RRF order).
    """
    docs = list(candidate_docs)
    label_to_docid: Dict[str, str] = {}
    docid_to_label: Dict[str, str] = {}
    for i, d in enumerate(docs, start=1):
        lab = f"C{i}"
        rid = d["doc_id"]
        label_to_docid[lab] = rid
        docid_to_label[rid] = lab
    return docs, label_to_docid, docid_to_label


# -----------------------------
# Prompting (uses C1/C2 labels)
# -----------------------------
def build_candidate_classify_prompt(
    query_text: str,
    labeled_docs: Sequence[Tuple[str, Document]],  # [("C1", doc), ...]
    *,
    max_extra_positives: int = 2,
    require_evidence: bool = True,
    include_one_shot: bool = True,
) -> str:
    per_doc_cap = 900

    def _clip(t: str) -> str:
        t = (t or "").strip()
        return t if len(t) <= per_doc_cap else (t[:per_doc_cap] + " ...")

    lines: List[str] = []
    lines.append("You are a strict judge that matches a query to candidate documents.")
    lines.append("")
    lines.append("Task:")
    lines.append("- You MUST evaluate EACH candidate document independently (one-by-one).")
    lines.append("- A document is POSITIVE ONLY if it directly answers the query (not just related).")
    lines.append("- All documents that are not POSITIVE must be NEGATIVE.")
    if require_evidence:
        lines.append("- For every POSITIVE, provide EVIDENCE as a verbatim quote copied from that document.")
    lines.append("")
    lines.append("Hard rules:")
    lines.append("- You MUST only use the provided labels (C1, C2, ...) (no other ids).")
    lines.append("- Output MUST be a single valid JSON object only. No extra text. No markdown.")
    lines.append("")
    lines.append("Output JSON schema:")
    lines.append("{")
    if require_evidence:
        lines.append('  "positives": [{"doc_id": "C2", "evidence": "verbatim quote"}],')
    else:
        lines.append('  "positives": [{"doc_id": "C2"}],')
    lines.append('  "negatives": ["C1", "C3"]')
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
        lines.append("Output JSON:")
        if require_evidence:
            lines.append('{ "positives": [{"doc_id": "C1", "evidence": "Paris is the capital and most populous city of France."}], "negatives": ["C2"] }')
        else:
            lines.append('{ "positives": [{"doc_id": "C1"}], "negatives": ["C2"] }')
        lines.append("")

    lines.append(f"Query: {query_text}")
    lines.append("")
    lines.append("Candidate documents:")
    for lab, d in labeled_docs:
        lines.append(f"- doc_id: {lab}")
        lines.append(f"  text: {_clip(d['text'])}")
    return "\n".join(lines)


# -----------------------------
# Main
# -----------------------------
def build_pairs_for_query(
    *,
    query_text: str,
    source_doc: Document,                 # known positive
    candidate_docs: Sequence[Document],   # RRF-ordered candidates (may include source)
    llm: Any,                             # llm.generate(prompt)->str
    embedder: Any,                        # embedder.encode_passages(list[str])->np.ndarray (normalized)
    # knobs
    max_extra_positives: int = 2,
    require_evidence: bool = True,
    cosine_threshold: float = 0.92,
    num_hard_negatives: int = 6,
    enable_text_hash_dedup: bool = True,
    include_one_shot: bool = True,
) -> Tuple[List[PairSample], Dict[str, Any]]:
    if not hasattr(llm, "generate"):
        raise TypeError("llm must have method .generate(prompt)->str.")
    if not hasattr(embedder, "encode_passages"):
        raise TypeError("embedder must have method .encode_passages(list[str])->np.ndarray.")

    stats: Dict[str, Any] = {
        "num_candidates_in": len(candidate_docs),
        "max_extra_positives": max_extra_positives,
        "require_evidence": require_evidence,
        "cosine_threshold": cosine_threshold,
        "num_hard_negatives": num_hard_negatives,
        "enable_text_hash_dedup": enable_text_hash_dedup,
        "include_one_shot": include_one_shot,
    }

    # ---- doc lookup
    doc_by_id: Dict[str, Document] = {}
    for d in candidate_docs:
        did = d["doc_id"]
        if did not in doc_by_id:
            doc_by_id[did] = d
    doc_by_id[source_doc["doc_id"]] = source_doc

    # ---- source_chunk (traceability)
    source_chunk = _extract_source_chunk(source_doc.get("text", ""))

    # ---- LLM sees only extra candidates (exclude source)
    cand_for_llm: List[Document] = [d for d in candidate_docs if d["doc_id"] != source_doc["doc_id"]]
    stats["num_candidates_for_llm"] = len(cand_for_llm)

    # ---- label them as C1, C2, ... (RRF order)
    cand_for_llm_list, label_to_real, real_to_label = _label_candidates(cand_for_llm)
    valid_labels = set(label_to_real.keys())  # {"C1","C2",...}

    labeled_docs: List[Tuple[str, Document]] = [(real_to_label[d["doc_id"]], d) for d in cand_for_llm_list]

    # ---- prompt + parse
    prompt = build_candidate_classify_prompt(
        query_text=query_text,
        labeled_docs=labeled_docs,
        max_extra_positives=max_extra_positives,
        require_evidence=require_evidence,
        include_one_shot=include_one_shot,
    )
    raw = llm.generate(prompt)
    out = _safe_json_loads(raw)

    pos_raw = out.get("positives") or []
    neg_raw = out.get("negatives") or []

    # ---- 1) collect ALL valid LLM positives (labels), do NOT truncate yet
    llm_pos_labels: List[str] = []
    invalid_label = 0
    invalid_evidence = 0

    if isinstance(pos_raw, list):
        for item in pos_raw:
            if not isinstance(item, dict):
                continue
            lab = str(item.get("doc_id") or "").strip()
            if lab not in valid_labels:
                invalid_label += 1
                continue

            real_id = label_to_real[lab]

            if require_evidence:
                ev = item.get("evidence")
                ev_s = str(ev).strip() if ev is not None else ""
                if not ev_s or not _best_effort_contains(doc_by_id[real_id]["text"], ev_s):
                    invalid_evidence += 1
                    continue

            llm_pos_labels.append(lab)

    # de-dup labels preserving first occurrence, already in RRF label order typically,
    # but we still sort by numeric suffix to guarantee RRF order.
    seen_pos: set[str] = set()
    llm_pos_labels_uniq: List[str] = []
    for lab in llm_pos_labels:
        if lab in seen_pos:
            continue
        seen_pos.add(lab)
        llm_pos_labels_uniq.append(lab)

    def _label_rank(lab: str) -> int:
        # "C12" -> 12 ; fallback big number
        m = re.match(r"^C(\d+)$", lab)
        return int(m.group(1)) if m else 10**9

    llm_pos_labels_uniq.sort(key=_label_rank)

    stats["invalid_label"] = invalid_label
    stats["invalid_evidence"] = invalid_evidence
    stats["llm_pos_valid_total"] = len(llm_pos_labels_uniq)

    # map labels -> real doc_ids
    llm_pos_real_ids: List[str] = [label_to_real[lab] for lab in llm_pos_labels_uniq]

    # ---- 2) cosine dedup over (source + ALL llm positives), keep source first
    pos_ids_pre_dedup: List[str] = [source_doc["doc_id"]] + llm_pos_real_ids

    if len(pos_ids_pre_dedup) <= 1:
        kept_pos_ids = pos_ids_pre_dedup
    else:
        pos_texts = [doc_by_id[did]["text"] for did in pos_ids_pre_dedup]
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

    # ---- 3) truncate extras AFTER dedup (this is your correction)
    extras_after_dedup = kept_pos_ids[1:]
    extras_after_dedup = extras_after_dedup[: max(0, int(max_extra_positives))]
    kept_pos_ids = [kept_pos_ids[0]] + extras_after_dedup

    stats["num_pos_kept_final"] = len(kept_pos_ids)
    stats["num_extra_pos_final"] = max(0, len(kept_pos_ids) - 1)

    kept_pos_set = set(kept_pos_ids)

    # ---- 4) negatives pool (preserve RRF order)
    # default: everything else in cand_for_llm excluding kept positives
    neg_pool_real_ids: List[str] = [d["doc_id"] for d in cand_for_llm_list if d["doc_id"] not in kept_pos_set]

    # if LLM provides explicit negatives labels, use strict pool (still RRF order)
    if isinstance(neg_raw, list) and len(neg_raw) > 0:
        llm_neg_labels = [str(x).strip() for x in neg_raw if str(x).strip()]
        llm_neg_labels = [lab for lab in llm_neg_labels if lab in valid_labels]
        # de-dup + sort by RRF label rank
        seen_n: set[str] = set()
        llm_neg_labels_uniq: List[str] = []
        for lab in llm_neg_labels:
            if lab in seen_n:
                continue
            seen_n.add(lab)
            llm_neg_labels_uniq.append(lab)
        llm_neg_labels_uniq.sort(key=_label_rank)

        if llm_neg_labels_uniq:
            neg_pool_real_ids = [label_to_real[lab] for lab in llm_neg_labels_uniq if label_to_real[lab] not in kept_pos_set]

    stats["num_neg_pool"] = len(neg_pool_real_ids)

    # ---- 5) negative cosine filter vs each positive (>threshold => drop)
    neg_texts = [doc_by_id[nid]["text"] for nid in neg_pool_real_ids]
    if neg_texts:
        pos_texts2 = [doc_by_id[did]["text"] for did in kept_pos_ids]
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
            h = _sha1(_norm_text(doc_by_id[nid]["text"]))
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
    neg_docs_final: List[Document] = [doc_by_id[nid] for nid in final_neg_ids]

    stats["num_neg_final"] = len(final_neg_ids)

    # ---- 8) build samples: one per positive (negatives shared)
    samples: List[PairSample] = []
    for did in kept_pos_ids:
        samples.append(
            PairSample(
                query_text=query_text,
                positive=doc_by_id[did],
                negatives=neg_docs_final,
                source_chunk=source_chunk,
            )
        )

    stats["num_samples"] = len(samples)
    return samples, stats
