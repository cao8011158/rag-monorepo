from __future__ import annotations

import argparse
import sys
from pathlib import Path
import json

from ce_pipeline.settings import load_settings
from ce_pipeline.pipeline.run import run_pipeline


def main() -> None:
    ap = argparse.ArgumentParser(prog="ce")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run chunking+embedding+indexing pipeline")
    p_run.add_argument("--config", required=True, help="Path to pipeline YAML config")

    args = ap.parse_args()

    if args.cmd == "run":
        try:
            cfg = load_settings(Path(args.config))
            res = run_pipeline(cfg)

            # pretty print summary
            print(json.dumps(res, ensure_ascii=False, indent=2))
            sys.exit(0)

        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
            sys.exit(2)
