1. Project Overview

This project implements a three-stage data pipeline:

Chunking Stage
Cleaned documents → text chunks → exact deduplication

Embedding Stage
Chunks → dense embeddings → semantic deduplication → FAISS index

Indexing Stage
Chunks → BM25 lexical index

Each stage is explicit, deterministic, and independently debuggable, following a strict data-contract design.

2. Directory Structure (Simplified)
   ce_pipeline/
   ├─ cli.py # CLI entrypoint
   ├─ settings.py # load_settings()
   │
   ├─ pipeline/
   │ ├─ run.py # Orchestrates all stages
   │ ├─ chunking_stage.py # documents → chunks
   │ ├─ embedding_stage.py # chunks → vectors
   │ └─ indexing_stage.py # chunks → BM25
   │
   ├─ chunking/
   │ ├─ chunker.py # Core chunking logic
   │ └─ sliding_window.py
   │
   ├─ embedding/
   │ └─ dual_instruct_embedder.py
   │
   ├─ indexing/
   │ ├─ vector.py # FAISS helpers
   │ └─ bm25.py
   │
   ├─ processing/
   │ ├─ exact_dedup.py # Hash-based dedup
   │ └─ near_dedup.py # ANN semantic dedup
   │
   ├─ stores/
   │ ├─ base.py # Store interface
   │ ├─ filesystem.py # Local filesystem store
   │ └─ registry.py
   │
   └─ io/
   └─ jsonl.py # read_jsonl / append_jsonl

3. Installation
   3.1 Environment

Python 3.10+

Recommended: virtual environment

python -m venv .venv
source .venv/bin/activate # Linux / macOS

# .venv\Scripts\activate # Windows

3.2 Install dependencies
pip install -r requirements.txt

Typical dependencies include:

numpy

orjson

faiss-cpu

sentence-transformers

rank-bm25

4. Configuration (YAML)

The pipeline is fully configured via a single YAML file.

4.1 Input Configuration
input:
input_store: fs_local
input_path: cleaned/latest/documents.jsonl

input_store
Name of the store defined in stores:

input_path
Logical path (POSIX style) to documents.jsonl

4.2 Output Configuration
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

All output artifacts are written under their respective base directories.

4.3 Store Configuration
stores:
fs_local:
kind: filesystem
root: .

Logical paths are resolved relative to root

Required for exact deduplication (filesystem-backed)

4.4 Chunking Configuration
chunking:
window_chars: 1200
overlap_chars: 200
min_chunk_chars: 200

Applied at the document → chunk stage.

4.5 Embedding Configuration
embedding:
model_name: sentence-transformers/all-MiniLM-L6-v2
batch_size: 64
normalize_embeddings: true
instructions:
passage: "passage: "
query: "query: "

Used by DualInstructEmbedder.

4.6 Deduplication Configuration
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

Exact dedup: hash-based, streaming

Semantic dedup: ANN + cosine similarity

5. Data Contracts
   5.1 Input Document Schema (documents.jsonl)
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

5.2 Chunk Schema (chunks.jsonl)
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

Contract rule:
Downstream stages rely strictly on:

chunk_id

chunk_text

6. Pipeline Stages & Outputs
   6.1 Chunking Stage

Reads

input_store + input_path

Writes

outputs.chunks.base/
├─ chunks.raw.jsonl
├─ chunks.jsonl
├─ errors.documents.jsonl
└─ errors.chunking.jsonl

Streaming write

Deterministic (overwrites per run)

Exact deduplication applied

6.2 Embedding Stage

Reads

outputs.chunks.base/chunks.jsonl

Writes

outputs.vector_index.base/
├─ faiss.index
├─ id_map.jsonl
└─ meta.json

Dense embeddings

Optional semantic deduplication

FAISS index serialization

6.3 Indexing Stage (BM25)

Reads

outputs.chunks.base/chunks.jsonl

Writes

outputs.bm25_index.base/
├─ bm25.pkl
└─ errors.indexing.read_chunks.jsonl (best-effort mode)

7. Running the Pipeline
   7.1 CLI Usage
   ce run --config configs/pipeline.yaml

This runs:

Chunking

Embedding

BM25 indexing

7.2 Failure Semantics (Defaults)
Stage Strategy
Chunking Best-effort (skip bad docs, log errors)
Embedding Fail-fast (strict schema)
Indexing Best-effort

Defaults are chosen for robust real-world data processing.
