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

    # ✅ Always read latest manifest (ABSOLUTE path already normalized in settings)
    latest_path: Path = cfg.manifest_latest
    man = load_manifest(str(latest_path))

    # ✅ Output path template is absolute under local_root already
    out_path = Path(cfg.output_jsonl.format(run_date=cfg.run_date))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Overwrite: remove old output file
    if out_path.exists():
        out_path.unlink()

    kept = 0
    dropped_too_short = 0
    errors = 0
    total = 0

    # Helpful debug (safe, no side-effects)
    print(f"[DEBUG] local_root={cfg.local_root}")
    print(f"[DEBUG] manifest_latest={latest_path} count={len(man)}")
    print(f"[DEBUG] out_path={out_path}")

    # Deterministic order by URL
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
            print(f"[CLEAN FAIL] url={url} rel={getattr(e, 'rel_path', '')} -> {ex}")

    print(
        f"[CLEAN DONE] run_date={cfg.run_date} "
        f"manifest={latest_path} "
        f"out={out_path} "
        f"total={total} kept={kept} dropped_too_short={dropped_too_short} errors={errors}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Clean crawled raw data into JSONL (always from latest manifest)."
    )
    ap.add_argument("--config", required=True, help="Path to pipeline YAML, e.g. configs/pipeline.yaml")
    args = ap.parse_args()

    run_cleaning(args.config)


if __name__ == "__main__":
    main()
