#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .

echo "✅ Installed."
echo "Try:"
echo "  rag-cc --help"
echo "  rag-cc run --config configs/pipeline.yaml"
