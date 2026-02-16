#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data}"
BUCKET="${S3_BUCKET:-rag-kb-storage-ruoxin-002}"
PREFIX="${S3_PREFIX:-rag-kb-data}"
CONFIG_DST="${CONFIG_DST:-/app/configs}"

mkdir -p "${DATA_ROOT}/ce_out" "${DATA_ROOT}/run1" "${DATA_ROOT}/.hf_cache" "${CONFIG_DST}"

echo "[bootstrap] Sync configs..."
aws s3 sync "s3://${BUCKET}/${PREFIX}/configs/" "${CONFIG_DST}/" --only-show-errors

echo "[bootstrap] Sync ce_out..."
aws s3 sync "s3://${BUCKET}/${PREFIX}/ce_out/" "${DATA_ROOT}/ce_out/" --only-show-errors

echo "[bootstrap] Sync run1 (LoRA adapter)..."
aws s3 sync "s3://${BUCKET}/${PREFIX}/run1/" "${DATA_ROOT}/run1/" --only-show-errors

exec "$@"
