# 项目概述（Project Overview）

本项目实现了一个三阶段的数据处理流水线：

## 流水线阶段（Pipeline Stages）

### 1. 分块阶段（Chunking Stage）

清洗后的文档 → 文本分块 → 精确去重（Exact Deduplication）

### 2. 向量化阶段（Embedding Stage）

文本块 → 稠密向量 → 语义去重 → FAISS 向量索引

### 3. 索引阶段（Indexing Stage）

文本块 → BM25 词法索引

每个阶段都是： - 显式定义的（Explicit） - 确定性的（Deterministic） -
可独立调试的（Independently debuggable）

并严格遵循数据契约（Data Contract）设计。

---

# 目录结构（简化版）

```text
ce_pipeline/
├── cli.py                 # CLI 入口
├── settings.py            # load_settings()
│
├── pipeline/
│   ├── run.py             # 编排所有阶段
│   ├── chunking_stage.py  # documents → chunks
│   ├── embedding_stage.py # chunks → vectors
│   └── indexing_stage.py  # chunks → BM25
│
├── chunking/
│   ├── chunker.py         # 核心分块逻辑
│   └── sliding_window.py
│
├── embedding/
│   └── dual_instruct_embedder.py
│
├── indexing/
│   ├── vector.py          # FAISS 工具函数
│   └── bm25.py
│
├── processing/
│   ├── exact_dedup.py     # 基于哈希的精确去重
│   └── near_dedup.py      # 基于 ANN 的语义去重
│
├── stores/
│   ├── base.py            # Store 接口
│   ├── filesystem.py     # 本地文件系统 Store
│   └── registry.py
│
└── io/
    └── jsonl.py           # read_jsonl / append_jsonl
```

---

# 安装（Installation）

## 1. 环境要求

- Python 3.10+
- 推荐使用虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
```

## 2. 安装依赖

```bash
pip install -r requirements.txt
```

常见依赖包括：

- numpy\
- orjson\
- faiss-cpu\
- sentence-transformers\
- rank-bm25

---

# 配置说明（YAML）

整个流水线通过一个 YAML 文件进行配置。

## 1. 输入配置

```yaml
input:
  input_store: fs_local
  input_path: cleaned/latest/documents.jsonl
```

说明：

- `input_store`\
  在 `stores` 中定义的存储名称

- `input_path`\
  指向 `documents.jsonl` 的逻辑路径（POSIX 风格）

---

## 2. 输出配置

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

所有输出产物都会写入各自的 `base` 目录下。

---

## 3. Store 配置

```yaml
stores:
  fs_local:
    kind: filesystem
    root: .
```

- 逻辑路径相对于 `root` 解析
- 精确去重必须使用文件系统存储

---

## 4. 分块配置（Chunking）

```yaml
chunking:
  window_chars: 1200
  overlap_chars: 200
  min_chunk_chars: 200
```

该配置应用于：文档 → 文本块 阶段。

---

## 5. 向量化配置（Embedding）

```yaml
embedding:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  batch_size: 64
  normalize_embeddings: true
  instructions:
    passage: "passage: "
    query: "query: "
```

由 `DualInstructEmbedder` 使用。

---

## 6. 去重配置（Deduplication）

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

- 精确去重：基于哈希，流式处理\
- 语义去重：基于 ANN + 余弦相似度

---

# 数据契约（Data Contracts）

## 输入文档结构（documents.jsonl）

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

---

## 文本块结构（chunks.jsonl）

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

契约规则：

下游阶段只依赖以下字段：

- `chunk_id`
- `chunk_text`

---

# 运行流水线（Running the Pipeline）

```bash
ce run --config configs/pipeline.yaml
```

该命令将执行：

- 分块阶段（Chunking）
- 向量化阶段（Embedding）
- BM25 索引阶段（Indexing）

---

# 失败语义（Failure Semantics）

阶段 策略

---

Chunking Best-effort（跳过坏数据并记录错误）
Embedding Fail-fast（严格校验数据格式）
Indexing Best-effort

默认策略适用于真实世界中的鲁棒数据处理。
