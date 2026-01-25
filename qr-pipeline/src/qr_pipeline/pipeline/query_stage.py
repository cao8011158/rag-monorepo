from __future__ import annotations

from typing import Any, Dict, List

from qr_pipeline.stores.registry import build_store_registry
from qr_pipeline.io.jsonl import read_jsonl, write_jsonl


def _posix_join(*parts: str) -> str:
    return "/".join([p.strip("/").replace("\\", "/") for p in parts if p is not None and str(p) != ""])


def run_query_generation_stage(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Scaffold stage: read chunks.jsonl and output queries.jsonl.

    This stage currently implements a **simple baseline**:
    - N queries per chunk: uses the first sentence-ish snippet to form a question
    Replace `generate_queries_for_chunk()` with an actual LLM later.
    """
    stores = build_store_registry(cfg)

    in_cfg = cfg["input"]
    in_store = stores[in_cfg["input_store"]]
    chunks_path = in_cfg["input_path"]

    out_cfg = cfg["outputs"]["queries"]
    out_store = stores[out_cfg["store"]]
    out_base = out_cfg["base"]
    out_file = _posix_join(out_base, "queries.jsonl")

    # Read input
    raw = in_store.read_text(chunks_path, encoding="utf-8")

    # best-effort reader for scaffold
    bad: List[str] = []

    def _on_err(e: Exception, line: str) -> None:
        bad.append(str(e))

    rows = read_jsonl(raw, on_error=_on_err)

    # Build queries
    q_rows: List[Dict[str, Any]] = []
    q_per = int(cfg.get("query_generation", {}).get("queries_per_chunk", 1))
    max_chars = int(cfg.get("query_generation", {}).get("max_chunk_chars", 1800))

    for r in rows:
        chunk_id = r.get("chunk_id")
        txt = str(r.get("chunk_text", ""))[:max_chars]
        for j in range(max(q_per, 1)):
            q = generate_queries_for_chunk(txt, variant=j)
            q_rows.append(
                {
                    "query_id": f"{chunk_id}::q{j}",
                    "source_chunk_id": chunk_id,
                    "query": q,
                }
            )

    out_store.write_text(out_file, write_jsonl(q_rows), encoding="utf-8")

    return {
        "name": "query_generation",
        "status": "ok",
        "input_chunks": len(rows),
        "output_queries": len(q_rows),
        "bad_lines": len(bad),
        "output_path": out_file,
        "note": "Scaffold baseline generator (replace with LLM).",
    }


def generate_queries_for_chunk(chunk_text: str, variant: int = 0) -> str:
    """Baseline query generator (placeholder). Replace with Qwen/Mistral inference."""
    t = " ".join(chunk_text.strip().split())
    if not t:
        return "What is this passage about?"
    cut = min(len(t), 180)
    snippet = t[:cut]
    if "." in snippet:
        snippet = snippet.split(".", 1)[0]
    snippet = snippet.replace("?", "")
    if variant % 2 == 0:
        return f"What does the passage say about: {snippet}?"
    return f"Can you summarize the key point about: {snippet}?"
