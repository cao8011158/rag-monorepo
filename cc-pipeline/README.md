# cc-pipeline

**Module A --- Crawl + Clean (Knowledge Ingestion Layer for RAG)**

`cc-pipeline` implements Module A of a modular Retrieval-Augmented
Generation (RAG) knowledge pipeline.

Module A is responsible for building a **stable, versioned, and
normalized knowledge snapshot** from a controlled set of seed URLs.\
It defines the ingestion boundary for downstream chunking, embedding,
indexing, and retrieval modules.

This module focuses exclusively on:

-   Controlled crawling (HTML / PDF)
-   Manifest-based incremental updates
-   Deterministic normalization into JSONL documents

It does **NOT** perform:

-   Chunking
-   Embedding
-   Vector indexing
-   Retrieval

------------------------------------------------------------------------

## 1. Overview

Module A provides the knowledge ingestion layer of the RAG system.

``` text
Seed URLs
   ↓
[Crawl] → raw HTML / PDF
   ↓
Manifest (incremental resource index)
   ↓
[Clean] → documents.jsonl
```

The output of Module A is treated as a **versioned knowledge
artifact**.\
Downstream modules must never access raw files or manifests directly.

They must only consume:

``` text
data/cleaned/{run_date}/documents.jsonl
```

------------------------------------------------------------------------

## 2. System Philosophy

Module A is designed as a **knowledge engineering system**, not a
general-purpose web crawler.

Its goals are:

-   Deterministic data collection
-   Stable document identities
-   Incremental updates
-   Noise-free content extraction
-   Reproducible knowledge snapshots

The system guarantees:

-   No infinite crawling loops
-   No cross-domain pollution
-   No crawl explosion
-   Fully deterministic scope

------------------------------------------------------------------------

## 3. Architecture

High-level architecture:

``` text
Seed URLs
   ↓
Crawl Stage
   ↓
Raw Files (HTML / PDF)
   ↓
Manifest (content hash index)
   ↓
Clean Stage
   ↓
documents.jsonl
```

Module A defines the ingestion contract between the web and downstream
RAG modules.

------------------------------------------------------------------------

## 4. Pipeline Stages

### 4.1 Crawl --- Knowledge Collection

Crawling is performed using **Breadth-First Search (BFS)** starting from
seed URLs.

#### Exploration Model

Depth semantics:

  Depth   Meaning
  ------- ------------------------------
  0       Seed URL
  1       Links found on seed pages
  2       Links found on depth-1 pages

Traversal is controlled by the following guards:

-   `max_depth` --- BFS depth limit
-   `same_domain_only` --- restrict traversal to seed domain
-   `allow_domains` --- optional allowlist override
-   `max_links_per_page` --- expansion cap per page
-   `max_pages_per_seed` --- per-seed crawl budget
-   `max_pages_total` --- global crawl budget
-   `drop_url_patterns` --- blacklist filters

This guarantees:

-   No infinite loops
-   No crawl explosion
-   Deterministic scope

#### Crawl Features

-   Custom User-Agent
-   Per-host rate limiting (RPS)
-   Timeout + retry with exponential backoff
-   SHA-256 content hashing
-   Manifest-based incremental updates

------------------------------------------------------------------------

### 4.2 Raw Storage Layout

Raw content is stored as immutable blobs:

``` text
data/raw/html/{run_date}/crawl/{url_hash}.html
data/raw/pdf/{run_date}/crawl/{url_hash}.pdf
```

------------------------------------------------------------------------

## 5. Manifest Tracking (Incremental Index)

Each fetched resource is tracked by a manifest entry:

``` text
data/manifests/{run_date}.jsonl
data/manifests/latest.jsonl
```

Manifest schema:

``` json
{
  "url": "...",
  "content_hash": "...",
  "rel_path": "...",
  "content_type": "text/html | application/pdf",
  "fetched_at": "2026-01-17T12:34:56Z"
}
```

### Incremental Semantics

If a URL already exists in the latest manifest:

-   And its `content_hash` is unchanged\
    → the previous raw file is reused\
    → no duplicate storage is produced

This enables **true incremental crawling**.

The manifest acts as a **global knowledge index**.

------------------------------------------------------------------------

## 6. Clean --- Knowledge Normalization

The clean stage reads raw files referenced by the manifest and converts
them into normalized JSONL documents.

### 6.1 HTML Processing

HTML documents are normalized using BeautifulSoup (lxml) with
multi-layer boilerplate removal.

#### Structural Tag Removal

The following tags are removed entirely:

``` text
script, style, noscript, header, footer, nav, aside
```

#### UI / Layout Boilerplate Filtering

Removed blocks include:

-   Cookie banners & consent dialogs
-   Subscription / newsletter modals
-   Login / signup panels
-   Popups and overlays
-   Social share widgets
-   Breadcrumb navigation
-   Related / recommended content
-   Comment sections
-   Ads and sponsored blocks

Typical matched keywords:

``` text
cookie, consent, gdpr, privacy
subscribe, newsletter, signup, login
modal, dialog, popup, overlay
share, social, follow
breadcrumb, related, recommend
comment, comments, disqus
ads, adslot, sponsored, promo, banner
```

#### Text Extraction

After boilerplate removal:

-   `<title>` is extracted
-   Visible text is extracted line-by-line
-   Empty lines removed
-   Whitespace normalized

------------------------------------------------------------------------

### 6.2 PDF Processing

PDF documents are parsed using `pypdf`:

-   Page-by-page extraction
-   Whitespace normalization
-   Pages concatenated into a single document

------------------------------------------------------------------------

### 6.3 Filtering Rules

Documents are excluded if:

``` text
len(text) < min_text_chars
```

This prevents indexing:

-   Navigation stubs
-   Category pages
-   Low-information placeholders

------------------------------------------------------------------------

## 7. Output Contract (Module A → Module B)

Downstream modules must consume only:

``` text
data/cleaned/{run_date}/documents.jsonl
```

They must never read raw files or manifests directly.

### Format

-   Encoding: UTF-8
-   Format: JSON Lines
-   Granularity: one document per line

### Schema

``` json
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
```

### Field Semantics

  Field          Description
  -------------- ----------------------------------------------
  doc_id         Stable ID: sha256(url + content_hash)\[:24\]
  url            Original source URL
  title          HTML `<title>` (empty for PDFs)
  text           Cleaned main content
  source         Data source tag ("seed")
  content_hash   SHA-256 hash of raw content
  content_type   MIME type
  fetched_at     Crawl timestamp (UTC ISO-8601)
  run_date       Pipeline run date (YYYY-MM-DD)

### Example

``` json
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
```

------------------------------------------------------------------------

## 8. Processing Guarantees

Module A guarantees:

-   Each output line corresponds to one successfully crawled URL
-   Same-domain traversal only (unless allowlist overrides)
-   No infinite loops or crawl explosion
-   HTML boilerplate and UI noise removed
-   Cookie banners, modals, ads, comments filtered
-   Documents shorter than `min_text_chars` excluded
-   `content_hash` enables incremental updates
-   `doc_id` changes when content changes
-   Output represents a stable knowledge snapshot

------------------------------------------------------------------------

## 9. Installation

### macOS / Linux / WSL

``` bash
bash scripts/bootstrap.sh
```

### Windows (PowerShell)

``` powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
pip install -e .
```

------------------------------------------------------------------------

## 10. Command Line Interface

This project provides a pipeline-style CLI:

``` text
rag-cc
```

### Global Options

``` text
--config configs/pipeline.yaml
```

### Full Pipeline

``` bash
rag-cc run
```

Equivalent to:

``` bash
rag-cc crawl
rag-cc clean
```

### Crawl Only

``` bash
rag-cc crawl
```

Outputs:

``` text
data/raw/html/{run_date}/...
data/raw/pdf/{run_date}/...
data/manifests/{run_date}.jsonl
data/manifests/latest.jsonl
```

### Clean Only

``` bash
rag-cc clean
```

Outputs:

``` text
data/cleaned/{run_date}/documents.jsonl
```

------------------------------------------------------------------------

## 11. Configuration

Main configuration:

``` text
configs/pipeline.yaml
```

Seed URLs:

``` text
configs/seeds.yaml
```

------------------------------------------------------------------------

## 12. Project Structure

``` text
cc-pipeline/
├── configs/
│   ├── pipeline.yaml
│   └── seeds.yaml
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── manifests/
├── scripts/
│   └── bootstrap.sh
├── src/cc_pipeline/
│   ├── crawl/
│   ├── clean/
│   ├── pipeline/
│   └── common/
└── tests/
```

------------------------------------------------------------------------

## 13. Design Philosophy (Optional)

-   Determinism over convenience
-   Knowledge snapshot over live crawling
-   Manifest-driven incremental updates
-   Clean content over raw HTML
-   Reproducible artifacts

Module A treats knowledge ingestion as an **engineering discipline**,
not scraping.
# cc-pipeline

**Module A --- Crawl + Clean (Knowledge Ingestion Layer for RAG)**

`cc-pipeline` implements Module A of a modular Retrieval-Augmented
Generation (RAG) knowledge pipeline.

Module A is responsible for building a **stable, versioned, and
normalized knowledge snapshot** from a controlled set of seed URLs.\
It defines the ingestion boundary for downstream chunking, embedding,
indexing, and retrieval modules.

This module focuses exclusively on:

-   Controlled crawling (HTML / PDF)
-   Manifest-based incremental updates
-   Deterministic normalization into JSONL documents

It does **NOT** perform:

-   Chunking
-   Embedding
-   Vector indexing
-   Retrieval

------------------------------------------------------------------------

## 1. Overview

Module A provides the knowledge ingestion layer of the RAG system.

``` text
Seed URLs
   ↓
[Crawl] → raw HTML / PDF
   ↓
Manifest (incremental resource index)
   ↓
[Clean] → documents.jsonl
```

The output of Module A is treated as a **versioned knowledge
artifact**.\
Downstream modules must never access raw files or manifests directly.

They must only consume:

``` text
data/cleaned/{run_date}/documents.jsonl
```

------------------------------------------------------------------------

## 2. System Philosophy

Module A is designed as a **knowledge engineering system**, not a
general-purpose web crawler.

Its goals are:

-   Deterministic data collection
-   Stable document identities
-   Incremental updates
-   Noise-free content extraction
-   Reproducible knowledge snapshots

The system guarantees:

-   No infinite crawling loops
-   No cross-domain pollution
-   No crawl explosion
-   Fully deterministic scope

------------------------------------------------------------------------

## 3. Architecture

High-level architecture:

``` text
Seed URLs
   ↓
Crawl Stage
   ↓
Raw Files (HTML / PDF)
   ↓
Manifest (content hash index)
   ↓
Clean Stage
   ↓
documents.jsonl
```

Module A defines the ingestion contract between the web and downstream
RAG modules.

------------------------------------------------------------------------

## 4. Pipeline Stages

### 4.1 Crawl --- Knowledge Collection

Crawling is performed using **Breadth-First Search (BFS)** starting from
seed URLs.

#### Exploration Model

Depth semantics:

  Depth   Meaning
  ------- ------------------------------
  0       Seed URL
  1       Links found on seed pages
  2       Links found on depth-1 pages

Traversal is controlled by the following guards:

-   `max_depth` --- BFS depth limit
-   `same_domain_only` --- restrict traversal to seed domain
-   `allow_domains` --- optional allowlist override
-   `max_links_per_page` --- expansion cap per page
-   `max_pages_per_seed` --- per-seed crawl budget
-   `max_pages_total` --- global crawl budget
-   `drop_url_patterns` --- blacklist filters

This guarantees:

-   No infinite loops
-   No crawl explosion
-   Deterministic scope

#### Crawl Features

-   Custom User-Agent
-   Per-host rate limiting (RPS)
-   Timeout + retry with exponential backoff
-   SHA-256 content hashing
-   Manifest-based incremental updates

------------------------------------------------------------------------

### 4.2 Raw Storage Layout

Raw content is stored as immutable blobs:

``` text
data/raw/html/{run_date}/crawl/{url_hash}.html
data/raw/pdf/{run_date}/crawl/{url_hash}.pdf
```

------------------------------------------------------------------------

## 5. Manifest Tracking (Incremental Index)

Each fetched resource is tracked by a manifest entry:

``` text
data/manifests/{run_date}.jsonl
data/manifests/latest.jsonl
```

Manifest schema:

``` json
{
  "url": "...",
  "content_hash": "...",
  "rel_path": "...",
  "content_type": "text/html | application/pdf",
  "fetched_at": "2026-01-17T12:34:56Z"
}
```

### Incremental Semantics

If a URL already exists in the latest manifest:

-   And its `content_hash` is unchanged\
    → the previous raw file is reused\
    → no duplicate storage is produced

This enables **true incremental crawling**.

The manifest acts as a **global knowledge index**.

------------------------------------------------------------------------

## 6. Clean --- Knowledge Normalization

The clean stage reads raw files referenced by the manifest and converts
them into normalized JSONL documents.

### 6.1 HTML Processing

HTML documents are normalized using BeautifulSoup (lxml) with
multi-layer boilerplate removal.

#### Structural Tag Removal

The following tags are removed entirely:

``` text
script, style, noscript, header, footer, nav, aside
```

#### UI / Layout Boilerplate Filtering

Removed blocks include:

-   Cookie banners & consent dialogs
-   Subscription / newsletter modals
-   Login / signup panels
-   Popups and overlays
-   Social share widgets
-   Breadcrumb navigation
-   Related / recommended content
-   Comment sections
-   Ads and sponsored blocks

Typical matched keywords:

``` text
cookie, consent, gdpr, privacy
subscribe, newsletter, signup, login
modal, dialog, popup, overlay
share, social, follow
breadcrumb, related, recommend
comment, comments, disqus
ads, adslot, sponsored, promo, banner
```

#### Text Extraction

After boilerplate removal:

-   `<title>` is extracted
-   Visible text is extracted line-by-line
-   Empty lines removed
-   Whitespace normalized

------------------------------------------------------------------------

### 6.2 PDF Processing

PDF documents are parsed using `pypdf`:

-   Page-by-page extraction
-   Whitespace normalization
-   Pages concatenated into a single document

------------------------------------------------------------------------

### 6.3 Filtering Rules

Documents are excluded if:

``` text
len(text) < min_text_chars
```

This prevents indexing:

-   Navigation stubs
-   Category pages
-   Low-information placeholders

------------------------------------------------------------------------

## 7. Output Contract (Module A → Module B)

Downstream modules must consume only:

``` text
data/cleaned/{run_date}/documents.jsonl
```

They must never read raw files or manifests directly.

### Format

-   Encoding: UTF-8
-   Format: JSON Lines
-   Granularity: one document per line

### Schema

``` json
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
```

### Field Semantics

  Field          Description
  -------------- ----------------------------------------------
  doc_id         Stable ID: sha256(url + content_hash)\[:24\]
  url            Original source URL
  title          HTML `<title>` (empty for PDFs)
  text           Cleaned main content
  source         Data source tag ("seed")
  content_hash   SHA-256 hash of raw content
  content_type   MIME type
  fetched_at     Crawl timestamp (UTC ISO-8601)
  run_date       Pipeline run date (YYYY-MM-DD)

### Example

``` json
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
```

------------------------------------------------------------------------

## 8. Processing Guarantees

Module A guarantees:

-   Each output line corresponds to one successfully crawled URL
-   Same-domain traversal only (unless allowlist overrides)
-   No infinite loops or crawl explosion
-   HTML boilerplate and UI noise removed
-   Cookie banners, modals, ads, comments filtered
-   Documents shorter than `min_text_chars` excluded
-   `content_hash` enables incremental updates
-   `doc_id` changes when content changes
-   Output represents a stable knowledge snapshot

------------------------------------------------------------------------

## 9. Installation

### macOS / Linux / WSL

``` bash
bash scripts/bootstrap.sh
```

### Windows (PowerShell)

``` powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
pip install -e .
```

------------------------------------------------------------------------

## 10. Command Line Interface

This project provides a pipeline-style CLI:

``` text
rag-cc
```

### Global Options

``` text
--config configs/pipeline.yaml
```

### Full Pipeline

``` bash
rag-cc run
```

Equivalent to:

``` bash
rag-cc crawl
rag-cc clean
```

### Crawl Only

``` bash
rag-cc crawl
```

Outputs:

``` text
data/raw/html/{run_date}/...
data/raw/pdf/{run_date}/...
data/manifests/{run_date}.jsonl
data/manifests/latest.jsonl
```

### Clean Only

``` bash
rag-cc clean
```

Outputs:

``` text
data/cleaned/{run_date}/documents.jsonl
```

------------------------------------------------------------------------

## 11. Configuration

Main configuration:

``` text
configs/pipeline.yaml
```

Seed URLs:

``` text
configs/seeds.yaml
```

------------------------------------------------------------------------

## 12. Project Structure

``` text
cc-pipeline/
├── configs/
│   ├── pipeline.yaml
│   └── seeds.yaml
├── data/
│   ├── raw/
│   ├── cleaned/
│   └── manifests/
├── scripts/
│   └── bootstrap.sh
├── src/cc_pipeline/
│   ├── crawl/
│   ├── clean/
│   ├── pipeline/
│   └── common/
└── tests/
```

------------------------------------------------------------------------

## 13. Design Philosophy (Optional)

-   Determinism over convenience
-   Knowledge snapshot over live crawling
-   Manifest-driven incremental updates
-   Clean content over raw HTML
-   Reproducible artifacts

Module A treats knowledge ingestion as an **engineering discipline**,
not scraping.
