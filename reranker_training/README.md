# 📘 Reranker Training 项目说明

本项目用于训练一个 **Cross-Encoder Reranker（重排序模型）**，服务于 RAG（Retrieval-Augmented Generation）系统，通过 **pairwise 排序数据 + QLoRA 微调** 来提升检索结果排序质量。

项目支持：

- ✅ 一个 query 对应多个 positive 文档（multi-positive）
- ✅ 每个 epoch 动态随机采样 negative（数据增广）
- ✅ Hard Negative + Random Negative 混合采样
- ✅ LoRA / QLoRA 微调
- ✅ 按 query 切分 Train / Validation（避免数据泄漏）
- ✅ YAML 配置驱动训练流程

---

## 📂 项目结构

```
reranker_training/
├── configs/
│   └── train_qlora.yaml
├── data/
│   └── processed/
│       ├── train.jsonl
│       └── valid.jsonl
├── src/
│   └── reranker_training/
│       ├── settings.py
│       ├── data.py
│       ├── modeling.py
│       └── train.py
├── outputs/
│   └── run1/
└── README.md
```

---

## 🧠 训练目标

对每个 query 学习：

score(query, positive) > score(query, negative)

使用 pairwise margin loss，支持 multi-positive 训练。

---

## 📊 数据格式（JSONL）

```json
{
  "query_text": "string",
  "positives": [{ "doc_id": "p1", "text": "正样本文本" }],
  "negatives": [
    { "doc_id": "n1", "text": "负样本文本" },
    { "doc_id": "n2", "text": "负样本文本" }
  ],
  "source_chunk": "chunk_id",
  "meta": {
    "domain": "cmu",
    "prompt_style": "qg_v1"
  }
}
```

##运行测试
pytest -m smoke -q
pytest -m slow -q

## 🚀 运行训练

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
python -m reranker_training.train_reranker --config configs/train_qlora.yaml
reranker-t --config configs/train_qlora.yaml
```

---

## 🧪 训练与验证策略

- Training：每个 epoch 动态采样 negatives（数据增广）
- Validation：使用固定完整验证集（不随机）

---

## 📌 设计原则

- 按 query 切分数据集
- 支持 multi-positive
- QLoRA 微调
- 配置驱动
- 可复现

---

本项目可直接集成到 RAG pipeline 中作为 reranker 训练模块。
