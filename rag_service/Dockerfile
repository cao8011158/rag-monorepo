

FROM python:3.11-slim

# 基础环境：更稳 + 不输出缓存文件
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 系统依赖（faiss / torch / 一些库可能需要）
# - libgomp1: faiss/一些向量库常用
# - ca-certificates: https
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     git \
#     curl \
#     ca-certificates \
#     libgomp1 \
#   && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    apt-get update -o Acquire::Retries=5; \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      libgomp1; \
    rm -rf /var/lib/apt/lists/*

# 复制依赖描述文件 安装依赖
COPY requirements.lock /app/
RUN pip install --upgrade pip && pip install -r requirements.lock


# 复制依赖描述文件,复制源码与配置
COPY pyproject.toml README.md /app/
COPY src/ /app/src/
COPY configs/ /app/configs/


# 安装构建工具 + 安装包
RUN pip install . --no-deps

# 安装 awscli（用于启动时 sync S3）
RUN pip install awscli

# 安装  拉取 rag knowledge base + reranker lora adapter
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh


# 运行端口（ECS/ALB 常用 8000）
EXPOSE 8000

# 启动 FastAPI
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "rag_service.server:app", "--host", "0.0.0.0", "--port", "8000"]