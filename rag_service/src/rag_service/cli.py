from __future__ import annotations

import argparse
from pathlib import Path

from rag_service.config import load_config
from rag_service.server import run_server


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["serve"], help="Run service")
    ap.add_argument("--config", required=True, help="Path to configs/rag.yaml")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))

    if args.cmd == "serve":
        run_server(cfg)
