# src/rag_service/validation/eval_direct_vs_rag.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rag_service.settings import load_settings
from rag_service.wiring import build_app_chain
from rag_service.lcel.direct_answer_node import create_direct_answer_runnable
from rag_service.validation.judge_answer_with_gemini import run_answer_judge


def _read_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"QA jsonl not found: {p}")

    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _safe_get_answer(resp: Any) -> str:
    # Your chains return dict like {"answer": "...", "mode": "...", ...}
    if isinstance(resp, dict):
        return str(resp.get("answer") or "").strip()
    return str(resp or "").strip()


def _invoke_with_retry(runnable, inp: Any, *, max_retries: int = 6, backoff_sec: float = 1.5) -> Any:
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return runnable.invoke(inp)
        except BaseException as e:
            last_exc = e
            if attempt >= max_retries:
                break
            time.sleep(backoff_sec * (attempt + 1))
    raise RuntimeError(f"Invoke failed after {max_retries+1} tries: {last_exc}") from last_exc


def _accumulate(metrics: Dict[str, Any], prefix: str, jr: Dict[str, Any]) -> None:
    # jr: JudgeResult TypedDict total=False
    is_correct = bool(jr.get("is_correct", False))
    score = float(jr.get("score", 0.0))

    metrics[f"{prefix}_n"] += 1
    metrics[f"{prefix}_correct"] += 1 if is_correct else 0
    metrics[f"{prefix}_score_sum"] += score


def _print_summary(metrics: Dict[str, Any]) -> None:
    def _fmt(prefix: str) -> Tuple[float, float]:
        n = int(metrics[f"{prefix}_n"])
        if n <= 0:
            return 0.0, 0.0
        acc = metrics[f"{prefix}_correct"] / n
        avg = metrics[f"{prefix}_score_sum"] / n
        return acc, avg

    direct_acc, direct_avg = _fmt("direct")
    rag_acc, rag_avg = _fmt("rag")

    print("\n================= SUMMARY =================")
    print(f"Total evaluated: {int(metrics['total'])}")
    print(f"Direct  | acc: {direct_acc:.3f} | avg_score: {direct_avg:.3f}")
    print(f"RAG     | acc: {rag_acc:.3f} | avg_score: {rag_avg:.3f}")
    print("==========================================\n")


def evaluate(
    *,
    config_path: str = "configs/rag.yaml",
    qa_path: str = "/content/drive/MyDrive/rag-kb-data/server/validation/qa_pairs.jsonl",
    out_path: str = "/content/drive/MyDrive/rag-kb-data/server/validation/results.jsonl",
    limit: int = 90,
) -> None:
    settings = load_settings(config_path)

    # Build the two systems:
    direct_chain = create_direct_answer_runnable(settings)
    rag_chain = build_app_chain(settings)

    qa_rows = _read_jsonl(qa_path, limit=limit)
    if not qa_rows:
        raise RuntimeError(f"No QA rows loaded from {qa_path}")

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    metrics: Dict[str, Any] = {
        "total": 0,
        "direct_n": 0,
        "direct_correct": 0,
        "direct_score_sum": 0.0,
        "rag_n": 0,
        "rag_correct": 0,
        "rag_score_sum": 0.0,
    }

    with out_p.open("w", encoding="utf-8") as wf:
        for i, qa in enumerate(qa_rows, start=1):
            q = str(qa.get("question") or "").strip()
            gold = str(qa.get("answer") or "").strip()

            if not q:
                # Skip empty question rows but record something useful
                rec = {
                    "idx": i,
                    "query_id": qa.get("query_id"),
                    "skipped": True,
                    "reason": "empty_question",
                }
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            metrics["total"] += 1

            # 1) Direct Gemini
            direct_resp = _invoke_with_retry(direct_chain, q, max_retries=2, backoff_sec=1.5)
            direct_text = _safe_get_answer(direct_resp)

            direct_jr = run_answer_judge(
                qa_pair={"question": q, "answer": gold},
                llm_output=direct_text,
                cfg=settings,
                check_grounding=False,
            )
            _accumulate(metrics, "direct", direct_jr)

            # 2) RAG chain
            rag_resp = _invoke_with_retry(rag_chain, q, max_retries=2, backoff_sec=1.5)
            rag_text = _safe_get_answer(rag_resp)

            rag_jr = run_answer_judge(
                qa_pair={"question": q, "answer": gold},
                llm_output=rag_text,
                cfg=settings,
                check_grounding=False,
            )
            _accumulate(metrics, "rag", rag_jr)

            # Save per-row record
            rec = {
                "idx": i,
                "query_id": qa.get("query_id"),
                "source_chunk_ids": qa.get("source_chunk_ids"),
                "question": q,
                "gold_answer": gold,
                "direct": {
                    "answer": direct_text,
                    "judge": direct_jr,
                },
                "rag": {
                    "answer": rag_text,
                    "mode": (rag_resp or {}).get("mode") if isinstance(rag_resp, dict) else None,
                    "router": (rag_resp or {}).get("router") if isinstance(rag_resp, dict) else None,
                    "judge": rag_jr,
                },
            }
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Light progress
            if i % 10 == 0 or i == len(qa_rows):
                print(f"[{i}/{len(qa_rows)}] done. direct_acc={metrics['direct_correct']/max(1,metrics['direct_n']):.3f} "
                      f"rag_acc={metrics['rag_correct']/max(1,metrics['rag_n']):.3f}")

    _print_summary(metrics)
    print(f"Saved detailed results to: {out_p}")


if __name__ == "__main__":
    evaluate()
