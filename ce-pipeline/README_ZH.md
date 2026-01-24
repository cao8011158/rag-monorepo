CE Pipeline —— Chunking · Embedding · Indexing 工程文档

一个 模块化、可复现、面向 RAG（Retrieval-Augmented Generation）系统 的文档处理流水线。
该项目将清洗后的文档转换为：

去重后的文本切分（Chunks）

稠密向量索引（FAISS）

稀疏词法索引（BM25）

整体设计遵循 数据契约（Data Contract）+ 分阶段流水线（Pipeline Stages） 的工程原则。

1. 项目概述

本项目实现了一个 三阶段数据处理流水线：

Chunking 阶段（文本切分）
清洗后的文档 → 文本块（chunks）→ 精确去重

Embedding 阶段（向量化）
文本块 → 向量嵌入 → 语义去重 → FAISS 向量索引

Indexing 阶段（索引构建）
文本块 → BM25 词法索引

每个阶段：

输入 / 输出 明确

行为 可复现

可以 独立调试

下游只依赖稳定的数据契约

2. 项目目录结构（简化版）
   ce_pipeline/
   ├─ cli.py # CLI 程序入口
   ├─ settings.py # load_settings()：加载 YAML 配置
   │
   ├─ pipeline/
   │ ├─ run.py # Pipeline 总调度器
   │ ├─ chunking_stage.py # 文档 → chunks
   │ ├─ embedding_stage.py # chunks → 向量索引
   │ └─ indexing_stage.py # chunks → BM25
   │
   ├─ chunking/
   │ ├─ chunker.py # 核心 chunk 逻辑
   │ └─ sliding_window.py
   │
   ├─ embedding/
   │ └─ dual_instruct_embedder.py
   │
   ├─ indexing/
   │ ├─ vector.py # FAISS 工具
   │ └─ bm25.py
   │
   ├─ processing/
   │ ├─ exact_dedup.py # 基于 hash 的精确去重
   │ └─ near_dedup.py # 基于 ANN 的语义去重
   │
   ├─ stores/
   │ ├─ base.py # Store 抽象接口
   │ ├─ filesystem.py # 本地文件系统 Store
   │ └─ registry.py
   │
   └─ io/
   └─ jsonl.py # read_jsonl / append_jsonl

3. 安装与环境准备
   3.1 Python 环境

Python 3.10+

推荐使用虚拟环境

python -m venv .venv
source .venv/bin/activate # Linux / macOS

# .venv\Scripts\activate # Windows

3.2 安装依赖
pip install -r requirements.txt

常用依赖包括：

numpy

orjson

faiss-cpu

sentence-transformers

rank-bm25

4. 配置文件（YAML）

整个 pipeline 只依赖一个 YAML 配置文件。

4.1 输入配置（Input）
input:
input_store: fs_local
input_path: cleaned/latest/documents.jsonl

含义：

input_store
Store 名称（在 stores: 中定义）

input_path
输入文件的逻辑路径（POSIX 风格）

4.2 输出配置（Outputs）
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

所有产物都会写入各自的 base 目录下。

4.3 Store 配置
stores:
fs_local:
kind: filesystem
root: .

所有逻辑路径都相对于 root

chunking 阶段的精确去重必须使用 filesystem store

4.4 Chunking 配置
chunking:
window_chars: 1200
overlap_chars: 200
min_chunk_chars: 200

用于 文档 → chunk 阶段。

4.5 Embedding 配置
embedding:
model_name: sentence-transformers/all-MiniLM-L6-v2
batch_size: 64
normalize_embeddings: true
instructions:
passage: "passage: "
query: "query: "

由 DualInstructEmbedder 使用。

4.6 去重配置（Dedup）
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

精确去重：基于内容 hash

语义去重：基于 ANN + 余弦相似度

5. 数据契约（Data Contracts）
   5.1 输入文档格式（documents.jsonl）
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

5.2 Chunk 输出格式（chunks.jsonl）
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

下游阶段强依赖字段：

chunk_id

chunk_text

6. Pipeline 各阶段与输出产物
   6.1 Chunking 阶段

读取

input_store + input_path

输出

outputs.chunks.base/
├─ chunks.raw.jsonl
├─ chunks.jsonl
├─ errors.documents.jsonl
└─ errors.chunking.jsonl

特点：

流式写入

每次运行覆盖旧产物

启用精确去重

6.2 Embedding 阶段

读取

outputs.chunks.base/chunks.jsonl

输出

outputs.vector_index.base/
├─ faiss.index
├─ id_map.jsonl
└─ meta.json

稠密向量嵌入

可选语义去重

FAISS 索引序列化

6.3 Indexing 阶段（BM25）

读取

outputs.chunks.base/chunks.jsonl

输出

outputs.bm25_index.base/
├─ bm25.pkl
└─ errors.indexing.read_chunks.jsonl （best-effort 模式）

7. 运行方式
   7.1 CLI 使用
   ce run --config configs/pipeline.yaml

该命令依次执行：

Chunking

Embedding

BM25 Indexing

8. 失败策略（Failure Semantics）
   阶段 默认策略
   Chunking Best-effort（跳过坏文档，记录错误）
   Embedding Fail-fast（严格 schema）
   Indexing Best-effort

默认策略适合 真实世界、噪声数据 的稳健处理。

9. 可复现性（Reproducibility）

每次运行都会覆盖输出目录

chunk_id 使用稳定 hash

去重过程确定性

所有产物自包含、可追溯

10. 可扩展性

该 pipeline 设计支持后续扩展：

Reranker（交叉编码器 / LLM）

Citation / Evidence Tracking

Graph-RAG / Structured-RAG

多索引融合（BM25 + 向量）

新增阶段 不需要破坏现有数据契约。
