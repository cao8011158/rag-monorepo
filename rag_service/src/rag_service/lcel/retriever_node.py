# rag_service/nodelcel/retriever_node.py
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from rag_service.common.retriever import HybridRetriever

def rows_to_docs(rows):
    return [
        Document(
            page_content=r["chunk_text"],
            metadata={"chunk_id": r["key"], "rrf_score": r["rrf_score"]},
        )
        for r in rows
    ]

def create_retriever_runnable(settings):
    hr = HybridRetriever.from_settings(settings)   # ✅ 构建阶段注入 config
    return RunnableLambda(lambda q: rows_to_docs(hr.retrieve(q)))  # ✅ 闭包捕获 hr