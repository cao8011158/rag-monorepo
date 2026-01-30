# src/cc_pipeline/pipeline/run_cleaning.py
from __future__ import annotations

import argparse
from pathlib import Path

from cc_pipeline.settings import load_cfg
from cc_pipeline.common.hashing import sha256_hex
from cc_pipeline.common.io import LocalStore
from cc_pipeline.crawl.manifest import load_manifest
from cc_pipeline.clean.html_cleaner import html_to_text
from cc_pipeline.clean.pdf_cleaner import pdf_to_text
from cc_pipeline.clean.writer import append_jsonl


def run_cleaning(config_path: str) -> None:
    cfg = load_cfg(config_path)
    store = LocalStore(cfg.local_root)

    # 永远读 latest manifest
    latest_path = cfg.manifest_latest
    man = load_manifest(latest_path)

    # 输出：按日期落到日期文件夹；同日期覆盖
    # 这里优先尊重你已有 cfg.output_jsonl（通常已经带 run_date）
    out_path = Path(cfg.output_jsonl.format(run_date=cfg.run_date))

    # 如果 output_jsonl 没有包含 run_date（或你想强制日期文件夹），就兜底拼一下：
    # data/cleaned/<run_date>/cleaned.jsonl
    out_str = str(out_path)
    if cfg.run_date not in out_str:
        out_path = Path(cfg.local_root) / "cleaned" / cfg.run_date / "cleaned.jsonl"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 覆盖：先删旧文件（否则 append_jsonl 会一直追加）
    if out_path.exists():
        out_path.unlink()

    kept = 0
    dropped_too_short = 0
    errors = 0
    total = 0

    # 让输出更稳定：按 URL 排序处理（可复现）
    for url, e in sorted(man.items(), key=lambda x: x[0]):
        total += 1
        try:
            raw = store.read_bytes(e.rel_path)

            if e.content_type == "text/html":
                title, text = html_to_text(raw.decode("utf-8", errors="ignore"))
            else:
                title, text = "", pdf_to_text(raw)

            if len(text) < cfg.min_text_chars:
                dropped_too_short += 1
                continue

            # ✅ doc schema：保持和你 run.py CLEAN 部分完全一致
            doc = {
                "doc_id": sha256_hex((url + e.content_hash).encode("utf-8"))[:24],
                "url": url,
                "title": title,
                "text": text,
                "source": "seed",
                "content_hash": e.content_hash,
                "content_type": e.content_type,
                "fetched_at": e.fetched_at,
                "run_date": cfg.run_date,
            }
            append_jsonl(out_path, doc)
            kept += 1

        except Exception as ex:
            errors += 1
            # 按你的 crawl 风格：不写失败日志文件，只在控制台提示
            print(f"[CLEAN FAIL] url={url} rel={getattr(e, 'rel_path', '')} -> {ex}")

    print(
        f"[CLEAN DONE] run_date={cfg.run_date} "
        f"manifest={latest_path} "
        f"out={out_path} "
        f"total={total} kept={kept} dropped_too_short={dropped_too_short} errors={errors}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean crawled raw data into JSONL (always from latest manifest).")
    ap.add_argument("--config", required=True, help="Path to pipeline YAML, e.g. configs/pipeline.yaml")
    args = ap.parse_args()

    run_cleaning(args.config)


if __name__ == "__main__":
    main()
