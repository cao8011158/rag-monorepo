# reranker_training

A minimal, reproducible scaffold to fine-tune a **cross-encoder reranker** with **LoRA/QLoRA**.

## What you get
- Pairwise training (q, pos, neg) for a reranker (cross-encoder)
- **Negative resampling per epoch** (data augmentation)
- YAML config driven training
- Simple evaluation placeholders (MRR/nDCG can be added later)

## Quickstart

### 1) Create env
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### 2) Install
You can choose either:
- install as a package (recommended):
  ```bash
  pip install -U pip
  pip install -e ".[dev]"
  ```
- or just run with `PYTHONPATH=src` (not recommended long-term):
  ```bash
  pip install -U pip
  pip install -r requirements_fallback.txt
  ```

### 3) Put data
Input JSONL format (one line per query sample):
```json
{
  "query_text": "...",
  "positive": {"doc_id": "D1", "text": "..."},
  "negatives": [{"doc_id": "N1", "text": "..."}, ...]
}
```

Put train/valid at:
- `data/processed/train.jsonl`
- `data/processed/valid.jsonl`

### 4) Run training
```bash
python -m reranker_training.train --config configs/train_qlora.yaml
```

Outputs go to `outputs/`.

## Notes
- For English chunks <= 2000 characters, `max_length=512` is a good default.
- If you want MRR/nDCG, implement in `src/reranker_training/metrics.py`.
