# rag_service

最终 RAG 系统（服务化）骨架：

- `rag_common/`：共享 schema + artifact loader（BM25 portable 等）
- `rag_service/`：FastAPI 服务（/v1/query, /healthz）
- `stores/`：统一 Store 抽象（filesystem / s3 可扩展）
- `retrieval/`：BM25 + FAISS（预留）+ hybrid 合并
- `generation/`：LLM 调用（占位，后续接 Gemini/OpenAI/HF）

## Quickstart

```bash


rag serve --config configs/rag.yaml
# open http://127.0.0.1:8000/docs
```

## Endpoints

- `GET /healthz`
- `POST /v1/query`

请求示例：

```json
{ "question": "What is ...?", "top_k": 8 }
```
