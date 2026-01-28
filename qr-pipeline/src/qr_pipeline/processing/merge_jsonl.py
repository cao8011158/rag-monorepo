import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from qr_pipeline.processing.embedder import DualInstructEmbedder
from qr_pipeline.processing.near_dedup import near_dedup_by_ann_faiss


# ----------------------------
# 1) 输入 / 输出路径（你已确认）
# ----------------------------
FILE_A = "/content/drive/MyDrive/rag-kb-data/rq_out/queries/in_domain.jsonl"
FILE_B = "/content/drive/MyDrive/rag-kb-testdata/rq_out/queries/in_domain.jsonl"

OUT_FINAL = "/content/drive/MyDrive/rag-kb-data/rq_out/queries/in_domain_merged_exact.jsonl"

# HF cache（你截图给的）
HF_CACHE_DIR = "/content/drive/MyDrive/rag-kb-data/data/.hf_cache"

# ----------------------------
# 2) embedding 参数（按你截图）
# ----------------------------
MODEL_NAME = "intfloat/e5-base-v2"
DEVICE = "cuda"          # 没 GPU 就改 "cpu"
BATCH_SIZE = 64
NORMALIZE = True
PASSAGE_INSTR = "passage: "
QUERY_INSTR = "query: "  # 注意你截图是 "query: "（带空格）

# ----------------------------
# 3) semantic dedup 参数（你没指定就用默认 0.95）
# ----------------------------
SEM_THRESHOLD = 0.95
SEM_TOPK = 20
HNSW_M = 32
EF_CONSTRUCTION = 200
EF_SEARCH = 64
SEM_NORMALIZE = True


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_qid(row: Dict[str, Any]) -> Optional[str]:
    qid = row.get("query_id")
    return qid if isinstance(qid, str) and qid else None


def get_qtext_norm(row: Dict[str, Any]) -> str:
    t = row.get("query_text_norm")
    return t if isinstance(t, str) else ""


# ----------------------------
# A) 读两个文件
# ----------------------------
rows_a = read_jsonl(FILE_A)
rows_b = read_jsonl(FILE_B)

# ----------------------------
# B) 跨文件 exact dedup（只去 B 中与 A 重复的 query_id）
#    文件内不去重：A 内重复保留；B 内重复也保留（只要不在 A 里出现）
# ----------------------------
seen = set()
for r in rows_a:
    qid = get_qid(r)
    if qid is not None:
        seen.add(qid)

rows_b_kept: List[Dict[str, Any]] = []
removed_cross = 0
for r in rows_b:
    qid = get_qid(r)
    if qid is not None and qid in seen:
        removed_cross += 1
        continue
    rows_b_kept.append(r)

merged = rows_a + rows_b_kept
print(f"[Exact cross-file dedup] A={len(rows_a)} B={len(rows_b)} kept_from_B={len(rows_b_kept)} removed_cross={removed_cross}")
print(f"[Merged] total={len(merged)}")

# ----------------------------
# C) semantic dedup（对 merged 后整体做）
#    使用 query_text_norm 生成 embedding
# ----------------------------

# 让 SentenceTransformers / Transformers 使用 Drive 里的 cache
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = HF_CACHE_DIR

embedder = DualInstructEmbedder(
    model_name=MODEL_NAME,
    passage_instruction=PASSAGE_INSTR,
    query_instruction=QUERY_INSTR,
    batch_size=BATCH_SIZE,
    normalize_embeddings=NORMALIZE,
    device=DEVICE,
)

texts = [get_qtext_norm(r) for r in merged]

# 对空文本行：不参与 semantic dedup，但最终保留（避免误删）
idx_map = [i for i, t in enumerate(texts) if t.strip()]
texts_nonempty = [texts[i] for i in idx_map]

if not texts_nonempty:
    # 全是空文本：直接输出 merged（只做了 exact cross-file dedup）
    write_jsonl(OUT_FINAL, merged)
    print(f"[Semantic dedup] skipped (no non-empty query_text_norm). Wrote: {OUT_FINAL}")
else:
    emb = embedder.encode_queries(texts_nonempty)  # [N, D]
    res = near_dedup_by_ann_faiss(
        emb,
        threshold=SEM_THRESHOLD,
        topk=SEM_TOPK,
        hnsw_m=HNSW_M,
        ef_construction=EF_CONSTRUCTION,
        ef_search=EF_SEARCH,
        normalize=SEM_NORMALIZE,
    )

    kept_global = set(idx_map[i] for i in res.kept_indices)
    # 空文本行全部保留
    for i, t in enumerate(texts):
        if not t.strip():
            kept_global.add(i)

    final_rows = [r for i, r in enumerate(merged) if i in kept_global]
    write_jsonl(OUT_FINAL, final_rows)

    print(f"[Semantic dedup] merged={len(merged)} -> kept={len(final_rows)} removed={len(merged)-len(final_rows)}")
    print(f"Wrote FINAL: {OUT_FINAL}")
