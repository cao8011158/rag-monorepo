# Placeholder for prompt templates (LLM-based query generation and labeling).

QUERY_GEN_PROMPT = """You are generating training queries for a retrieval system.

Given a passage, generate {k} questions whose answers can be found ONLY in the passage.
Return as JSON list of strings.

Passage:
{passage}
"""

LABEL_PROMPT = """You are labeling retrieval candidates for reranker training.

Given:
- Query
- Candidate passages (each with chunk_id + text)

Pick:
- 1-2 positives (can directly answer the query)
- {n} hard negatives (look relevant but do NOT answer)

Return JSON with:
{{
  "positives": ["chunk_id", ...],
  "hard_negatives": ["chunk_id", ...]
}}
Query:
{query}

Candidates:
{candidates}
"""
