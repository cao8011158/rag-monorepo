cc-pipeline

Module A — Crawl + Clean (Knowledge Ingestion Layer for RAG)

cc-pipeline implements Module A of a modular Retrieval-Augmented Generation (RAG) knowledge pipeline.

Module A is responsible for building a stable, versioned, normalized knowledge snapshot from a controlled set of seed URLs.
It provides the ingestion boundary for downstream chunking, embedding, indexing, and retrieval modules.

This module focuses exclusively on:

Controlled crawling (HTML / PDF)

Manifest-based incremental updates

Deterministic normalization into JSONL documents

It does not perform chunking, embedding, vector indexing, or retrieval.

System Philosophy

Module A is designed as a knowledge engineering system, not a general-purpose web crawler.

Its goals are:

Deterministic data collection

Stable document identities

Incremental updates

Noise-free content extraction

Reproducible knowledge snapshots

The output of Module A is treated as a versioned knowledge artifact.

Downstream modules must never access raw files or manifests directly.

Architecture Overview
Seed URLs
↓
[Crawl] → raw HTML/PDF files (data/raw/)
↓
Manifest (incremental resource index)
↓
[Clean] → normalized documents.jsonl

Module A implements the data ingestion layer of the RAG system.

Pipeline Stages

1. Crawl — Knowledge Collection

Fetches seed URLs and explores controlled link neighborhoods using a BFS strategy.

Features

Custom User-Agent

Per-host rate limiting (RPS)

Timeout + retry with exponential backoff

SHA-256 content hashing

Manifest-based incremental updates

Controlled BFS exploration (depth-limited)

Same-domain enforcement

Hard crawl budgets (anti-explosion guards)

Exploration Model

Crawling is performed using Breadth-First Search (BFS) starting from seed URLs.

Depth semantics:

Depth Meaning
0 Seed URL
1 Links found on seed pages
2 Links found on depth-1 pages

Exploration is controlled by the following guards:

max_depth — maximum BFS depth

same_domain_only — restricts traversal to seed domain

allow_domains — optional allowlist override

max_links_per_page — expansion cap per page

max_pages_per_seed — per-seed crawl budget

max_pages_total — global crawl budget

drop_url_patterns — blacklist filters

This guarantees:

No infinite loops

No cross-domain pollution

No crawl explosion

Fully deterministic scope

Raw Storage Layout
data/raw/html/{run_date}/crawl/{url_hash}.html
data/raw/pdf/{run_date}/crawl/{url_hash}.pdf

Raw content is stored as immutable blobs.

Manifest Tracking (Incremental Index)

Each resource is tracked by a manifest entry:

data/manifests/{run_date}.jsonl # run snapshot
data/manifests/latest.jsonl # rolling latest snapshot

Manifest entry schema:

{
"url": "...",
"content_hash": "...",
"rel_path": "...",
"content_type": "text/html | application/pdf",
"fetched_at": "2026-01-17T12:34:56Z"
}

Incremental Semantics

If a URL already exists in the latest manifest

And its content hash is unchanged
→ the previous raw file is reused
→ no duplicate storage is produced

This enables true incremental crawling.

The manifest is effectively a global knowledge index.

2. Clean — Knowledge Normalization

Reads raw files referenced by the manifest and converts them into normalized JSONL documents.

HTML Processing

HTML documents are normalized using BeautifulSoup (lxml) with multi-layer boilerplate removal.

1. Tag-level Boilerplate Removal

The following structural tags are removed entirely:

script
style
noscript
header
footer
nav
aside

2. UI / Layout Boilerplate Filtering

UI containers and non-content blocks are removed based on class/id keyword matching:

Cookie banners & consent dialogs

Subscription / newsletter modals

Login / signup panels

Popups and overlays

Social share widgets

Breadcrumb navigation

Related / recommended content blocks

Comment sections

Ads and sponsored blocks

Typical matched keywords:

cookie, consent, gdpr, privacy
subscribe, newsletter, signup, register, login
modal, dialog, popup, overlay
share, social, follow
breadcrumb
related, recommend, suggested
comment, comments, disqus
ads, adslot, sponsored, promo, banner

3. Text Extraction

After boilerplate removal:

<title> is extracted

Visible text is extracted line-by-line

Empty lines are removed

Whitespace is normalized

PDF Processing

PDF documents are parsed using pypdf:

Page-by-page extraction

Whitespace normalization

Page concatenation into a single document

Filtering

Documents are excluded if:

len(text) < min_text_chars

This prevents indexing:

Navigation stubs

Category pages

Low-information placeholders

Output
data/cleaned/{run_date}/documents.jsonl

Each line represents one normalized document.

Output Contract

Module A → Module B Interface

Downstream modules must consume only:

data/cleaned/{run_date}/documents.jsonl

They must never read raw files or manifests directly.

Format

Encoding: UTF-8

Format: JSON Lines (one JSON object per line)

Granularity: one document per line (not chunked)

Schema
{
"doc_id": "string",
"url": "string",
"title": "string",
"text": "string",
"source": "string",
"content_hash": "string",
"content_type": "string",
"fetched_at": "string",
"run_date": "string"
}

Field Semantics
Field Description
doc_id Stable document version ID: sha256(url + content_hash)[:24]
url Original source URL
title HTML <title> (empty for PDFs)
text Cleaned main content
source Data source tag (currently "seed")
content_hash SHA-256 hash of raw content
content_type MIME type (text/html / application/pdf)
fetched_at Crawl timestamp (UTC ISO-8601)
run_date Pipeline run date (YYYY-MM-DD)
Example Record
{
"doc_id": "2f1c0a9d4b7e1c3a8d0f2b1a",
"url": "https://www.cmu.edu/about/",
"title": "About CMU - Carnegie Mellon University",
"text": "Carnegie Mellon University is a private research university...",
"source": "seed",
"content_hash": "9e1f8c7a2d...",
"content_type": "text/html",
"fetched_at": "2026-01-17T12:34:56Z",
"run_date": "2026-01-17"
}

Processing Guarantees

Module A guarantees:

Each output line corresponds to one successfully crawled URL

Same-domain traversal only (unless allowlist overrides)

No infinite loops or crawl explosion

HTML boilerplate and UI noise removed

Cookie banners, modals, ads, comments filtered

Documents shorter than min_text_chars excluded

content_hash supports incremental update detection

doc_id changes when content changes

Manifest provides global deduplication

Output represents a stable knowledge snapshot

Project Structure
cc-pipeline/
├── configs/
│ ├── pipeline.yaml
│ └── seeds.yaml
├── data/
│ ├── raw/
│ ├── cleaned/
│ └── manifests/
├── scripts/
│ └── bootstrap.sh
├── src/cc_pipeline/
│ ├── crawl/
│ ├── clean/
│ ├── pipeline/
│ └── common/
└── tests/

Installation
macOS / Linux / WSL
bash scripts/bootstrap.sh

Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
pip install -e .

Command Line Interface

This project provides a pipeline-style CLI tool:

rag-cc

Global Options
--config <path> (default: configs/pipeline.yaml)

Available Commands

1. Full Pipeline — Crawl → Clean
   rag-cc run

# or

rag-cc --config configs/pipeline.yaml run

Equivalent to:

rag-cc crawl
rag-cc clean

2. Crawl Only — Raw Collection
   rag-cc --config configs/pipeline.yaml crawl

Outputs:

data/raw/html/{run_date}/...
data/raw/pdf/{run_date}/...
data/manifests/{run_date}.jsonl
data/manifests/latest.jsonl

3. Clean Only — Normalization
   rag-cc --config configs/pipeline.yaml clean

Outputs:

data/cleaned/{run_date}/documents.jsonl

Configuration

Main configuration:

configs/pipeline.yaml

Seed URLs:

configs/seeds.yaml
