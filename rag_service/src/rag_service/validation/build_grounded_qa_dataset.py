from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from rag_service.settings import load_settings
from rag_service.validation.val_data_generation_gemini import run_grounded_qa_generation


# -------------------------
# Fixed paths (as you specified)
# -------------------------
VALID_QP_PATH = Path("/content/drive/MyDrive/rag-kb-data/reranker_out/processed/valid_query_pack.jsonl")
CHUNKS_PATH   = Path("/content/drive/MyDrive/rag-kb-data/ce_out/chunks/chunks.jsonl")
OUT_DIR       = Path("/content/drive/MyDrive/rag-kb-data/server/validation")
OUT_JSONL     = OUT_DIR / "qa_pairs.jsonl"


# -------------------------
# JSONL helpers
# -------------------------
def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl_line(fp, obj: Dict[str, Any]) -> None:
    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -------------------------
# Build chunk_id -> chunk_text index
# -------------------------
def build_chunk_index(chunks_path: Path) -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for row in iter_jsonl(chunks_path):
        idx[row["chunk_id"]] = row["chunk_text"]
    return idx


# -------------------------
# Main: generate QA pairs for every QueryPack line
# -------------------------
def generate_qa_pairs_from_valid_query_pack(
    *,
    cfg_path: str = "configs/rag.yaml",
    valid_qp_path: Path = VALID_QP_PATH,
    chunks_path: Path = CHUNKS_PATH,
    out_jsonl_path: Path = OUT_JSONL,
    source_pick: str = "first",  # "first" | "concat"
    concat_max_chunks: int = 3,
    max_rows: Optional[int] = None,  # None = all rows
) -> Tuple[int, int, int]:
    """
    Reads valid_query_pack.jsonl (each line is a QueryPack),
    extracts QueryPack["query"]["query_text"] and ["source_chunk_ids"],
    loads chunk_text from chunks.jsonl by chunk_id,
    calls run_grounded_qa_generation(query_text, chunk_text, cfg=settings),
    writes qa_pairs.jsonl under /server/validation.

    Returns: (written, skipped_missing_chunk, skipped_empty_answer)
    """
    settings = load_settings(cfg_path)
    chunk_index = build_chunk_index(chunks_path)

    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_missing_chunk = 0
    skipped_empty_answer = 0

    with out_jsonl_path.open("w", encoding="utf-8") as out_f:
        for i, qp in enumerate(iter_jsonl(valid_qp_path), 1):
            if max_rows is not None and i > max_rows:
                break

            q: Dict[str, Any] = qp["query"]
            query_text: str = q["query_text"]
            source_chunk_ids: List[str] = q.get("source_chunk_ids") or []
            query_id = q.get("query_id")
            domain = q.get("domain")

            # collect available chunk_texts for provided source_chunk_ids
            found_ids: List[str] = []
            found_texts: List[str] = []
            for cid in source_chunk_ids:
                if cid in chunk_index:
                    found_ids.append(cid)
                    found_texts.append(chunk_index[cid])

            if not found_texts:
                skipped_missing_chunk += 1
                continue

            if source_pick == "concat":
                used_ids = found_ids[:concat_max_chunks]
                chunk_text = "\n\n---\n\n".join(chunk_index[cid] for cid in used_ids)
            else:
                used_ids = [found_ids[0]]
                chunk_text = found_texts[0]

            resp = run_grounded_qa_generation(
                query_text=query_text,
                chunk_text=chunk_text,
                cfg=settings,
            )

            question = (resp.get("question") or query_text).strip()
            answer = (resp.get("answer") or "").strip()

            if not answer:
                skipped_empty_answer += 1
                continue

            out_row: Dict[str, Any] = {
                "question": question,
                "answer": answer,
                "query_id": query_id,
                "domain": domain,
                "source_chunk_ids": used_ids,
                "chunk_id": used_ids[0] if used_ids else None,
                "chunk_text": chunk_text,
                "meta": {"from": "valid_query_pack", "line_no": i},
            }
            write_jsonl_line(out_f, out_row)
            written += 1

            if written % 50 == 0:
                print(
                    f"[progress] written={written}, "
                    f"skipped_missing_chunk={skipped_missing_chunk}, "
                    f"skipped_empty_answer={skipped_empty_answer}"
                )

    print(f"[done] wrote {written} lines -> {out_jsonl_path}")
    print(f"[stats] skipped_missing_chunk={skipped_missing_chunk}, skipped_empty_answer={skipped_empty_answer}")
    return written, skipped_missing_chunk, skipped_empty_answer


# -------------------------
# Usage
# -------------------------
# 全量生成：
# generate_qa_pairs_from_valid_query_pack()

# 先试跑 20 条：
# generate_qa_pairs_from_valid_query_pack(max_rows=20)

# -------------------------
# CLI entry
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build grounded QA dataset from QueryPack"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N rows (for testing)",
    )

    parser.add_argument(
        "--concat",
        action="store_true",
        help="Use concatenated source chunks instead of first chunk only",
    )

    args = parser.parse_args()

    written, skipped_missing, skipped_empty = generate_qa_pairs_from_valid_query_pack(
        max_rows=args.limit,
        source_pick="concat" if args.concat else "first",
    )

    print("\n========== SUMMARY ==========")
    print("Written:", written)
    print("Skipped (no chunk):", skipped_missing)
    print("Skipped (empty answer):", skipped_empty)
