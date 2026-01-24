from __future__ import annotations

import argparse
from cc_pipeline.pipeline.run import run_pipeline

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rag-cc", description="Module A: Crawl + Clean")
    p.add_argument("--config", default="configs/pipeline.yaml", help="Path to config YAML")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("crawl", help="Fetch raw HTML/PDF into data/raw and update manifest")
    sub.add_parser("clean", help="Clean raw into JSONL documents using manifest")
    sub.add_parser("run", help="crawl -> clean")
    return p

def main():
    args = build_parser().parse_args()
    run_pipeline(config_path=args.config, mode=args.cmd)

if __name__ == "__main__":
    main()
