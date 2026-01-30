# Project Overview

This project implements a three-stage data pipeline:

## Pipeline Stages

### 1. Chunking Stage

Cleaned documents → text chunks → exact deduplication

### 2. Embedding Stage

Chunks → dense embeddings → semantic deduplication → FAISS index

### 3. Indexing Stage

Chunks → BM25 lexical index

Each stage is explicit, deterministic, and independently debuggable,
following a strict data-contract design.

---

# Directory Structure (Simplified)

```text
ce_pipeline/
├── cli.py                 # CLI entrypoint
├── settings.py            # load_settings()
│
├── pipeline/
│   ├── run.py             # Orchestrates all stages
│   ├── chunking_stage.py  # documents → chunks
│   ├── embedding_stage.py # chunks → vectors
│   └── indexing_stage.py  # chunks → BM25
│
├── chunking/
│   ├── chunker.py         # Core chunking logic
│   └── sliding_window.py
│
├── embedding/
│   └── dual_instruct_embedder.py
│
├── indexing/
│   ├── vector.py          # FAISS helpers
│   └── bm25.py
│
├── processing/
│   ├── exact_dedup.py     # Hash-based dedup
│   └── near_dedup.py      # ANN semantic dedup
│
├── stores/
│   ├── base.py            # Store interface
│   ├── filesystem.py     # Local filesystem store
│   └── registry.py
│
└── io/
    └── jsonl.py           # read_jsonl / append_jsonl
```

---

# Installation

## 1. Environment

- Python 3.10+
- Recommended: virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.\venv\Scripts\Activate.ps1 #powershell
# .venv\Scripts\activate    # Windows
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

Typical dependencies include:

- numpy\
- orjson\
- faiss-cpu\
- sentence-transformers\
- rank-bm25

---

# Configuration (YAML)

The pipeline is fully configured via a single YAML file.

## 1. Input Configuration

```yaml
input:
  input_store: fs_local
  input_path: cleaned/latest/documents.jsonl
```

## 2. Output Configuration

```yaml
outputs:
  chunks:
    store: fs_local
    base: ce_out/chunks

  vector_index:
    store: fs_local
    base: ce_out/indexes/vector

  bm25_index:
    store: fs_local
    base: ce_out/indexes/bm25
```

## 3. Store Configuration

```yaml
stores:
  fs_local:
    kind: filesystem
    root: .
```

## 4. Chunking Configuration

```yaml
chunking:
  window_chars: 1200
  overlap_chars: 200
  min_chunk_chars: 200
```

## 5. Embedding Configuration

```yaml
embedding:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  batch_size: 64
  normalize_embeddings: true
  instructions:
    passage: "passage: "
    query: "query: "
```

## 6. Deduplication Configuration

```yaml
processing:
  dedup:
    exact_dedup:
      hash_field: chunk_text_hash

    semantic_dedup:
      enable: true
      threshold: 0.95
      topk: 20
      hnsw_m: 32
      ef_construction: 200
      ef_search: 128
      normalize: true
```

---

# Data Contracts

## Input Document Schema (documents.jsonl)

```json
{
  "doc_id": "string",
  "text": "string",
  "url": "string",
  "title": "string",
  "source": "string",
  "content_hash": "string",
  "content_type": "string",
  "fetched_at": "string",
  "run_date": "string"
}
```

## Chunk Schema (chunks.jsonl)

```json
{
  "chunk_id": "string",
  "doc_id": "string",
  "chunk_index": 0,
  "chunk_text": "string",
  "chunk_text_hash": "string",

  "url": "string",
  "title": "string",
  "source": "string",
  "content_hash": "string",
  "content_type": "string",
  "fetched_at": "string",
  "run_date": "string"
}
```

---

# Running the Pipeline

```bash
ce run --config configs/pipeline.yaml
```

---

# Failure Semantics

Stage Strategy

---

Chunking Best-effort
Embedding Fail-fast
Indexing Best-effort
