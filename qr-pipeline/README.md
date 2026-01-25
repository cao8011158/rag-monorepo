# qr-pipeline

**Query + Reranker-data pipeline**

This module generates synthetic queries from chunks and builds a pairwise ranking dataset:

- Input: `chunks.jsonl` (from ce-pipeline)
- Output:
  - `queries.jsonl`
  - `pairwise.jsonl` (records with query + pos/neg chunk ids/text)

> Note: This scaffold intentionally keeps LLM and retrieval implementations minimal.
> You can plug in Qwen / Mistral / HF Inference / vLLM later.

## Quickstart

```bash
pip install -e .
qr run --config configs/pipeline.yaml
```

## Config

See `configs/pipeline.yaml` for an example.
