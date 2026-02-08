#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_rag_final_scaffold.py

生成“最终 RAG 项目”框架（FastAPI 服务 + rag_common shared util + stores + retriever + generator）。
默认不覆盖已存在文件；用 --force 强制覆盖。

用法：
  python generate_rag_final_scaffold.py --name rag_service
  python generate_rag_final_scaffold.py --name rag_service --root .
  python generate_rag_final_scaffold.py --name rag_service --force
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


# -----------------------------
# Small FS helpers
# -----------------------------
def _norm(s: str) -> str:
    return (s or "").strip()


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, content: str, *, force: bool) -> None:
    if path.exists() and not force:
        return
    _mkdir(path.parent)
    path.write_text(content, encoding="utf-8")


def _rel(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)


@dataclass(frozen=True)
class FileSpec:
    relpath: str
    content: str


# -----------------------------
# Templates
# -----------------------------
def t_pyproject(project_name: str) -> str:
    return f"""\
[build-system]
requires = ["setuptools>=69.0.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{project_name}"
version = "0.1.0"
description = "Final RAG service scaffold (FastAPI + retrieval + shared artifacts)."
readme = "README.md"
requires-python = ">=3.10"
license = {{ text = "MIT" }}

dependencies = [
  "pyyaml>=6.0.1",
  "orjson>=3.10.0",
  "tqdm>=4.66.0",
  "numpy>=1.26.0",
  "rank-bm25>=0.2.2",
  "whoosh>=2.7.4",
  "fastapi>=0.110.0",
  "uvicorn[standard]>=0.27.0",
  # optional:
  # "faiss-cpu>=1.8.0",
  # "httpx>=0.27.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0.0",
  "ruff>=0.4.0",
  "mypy>=1.8.0",
  "types-PyYAML>=6.0.12.20241230",
]

[project.scripts]
rag = "{project_name}.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
"""


def t_readme(project_name: str) -> str:
    return f"""\
# {project_name}

最终 RAG 系统（服务化）骨架：

- `rag_common/`：共享 schema + artifact loader（BM25 portable 等）
- `{project_name}/`：FastAPI 服务（/v1/query, /healthz）
- `stores/`：统一 Store 抽象（filesystem / s3 可扩展）
- `retrieval/`：BM25 + FAISS（预留）+ hybrid 合并
- `generation/`：LLM 调用（占位，后续接 Gemini/OpenAI/HF）

## Quickstart

```bash
pip install -e .[dev]
rag serve --config configs/rag.yaml
# open http://127.0.0.1:8000/docs
```

## Endpoints

- `GET /healthz`
- `POST /v1/query`

请求示例：

```json
{{"question":"What is ...?", "top_k": 8}}
```
"""


def t_configs_rag_yaml(project_name: str) -> str:
    return """\
# configs/rag.yaml

service:
  host: "0.0.0.0"
  port: 8000

stores:
  fs_local:
    kind: filesystem
    root: "./data"   # 你可以换成 /mnt/efs 或挂载目录

artifacts:
  # BM25 portable artifact
  bm25:
    store: fs_local
    path: "ce_out/bm25/bm25_portable.pkl"

  # FAISS artifacts（预留）
  faiss:
    store: fs_local
    index_path: "ce_out/faiss/index.faiss"
    idmap_path: "ce_out/faiss/id_map.jsonl"

data:
  # chunk 文本库（用于把 doc_id/chunk_id 映射到 chunk_text 等）
  chunks:
    store: fs_local
    path: "ce_out/chunks/chunks.jsonl"

retrieval:
  bm25:
    enabled: true
    top_k: 50
  faiss:
    enabled: false
    top_k: 50

rerank:
  enabled: false
  # model_name: "BAAI/bge-reranker-v2-m3"

generation:
  enabled: true
  provider: "dummy"   # "gemini" | "openai" | "hf"（你后面接）
  max_context_chars: 20000
"""


def t_src_init() -> str:
    return "__all__ = []\n"


def t_cli(project_name: str) -> str:
    return f"""\
from __future__ import annotations

import argparse
from pathlib import Path

from {project_name}.config import load_config
from {project_name}.server import run_server


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["serve"], help="Run service")
    ap.add_argument("--config", required=True, help="Path to configs/rag.yaml")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))

    if args.cmd == "serve":
        run_server(cfg)
"""


def t_config_py(project_name: str) -> str:
    return """\
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class AppConfig:
    raw: Dict[str, Any]


def load_config(path: Path) -> AppConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML mapping")
    return AppConfig(raw=data)
"""


def t_server_py(project_name: str) -> str:
    return f"""\
from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from {project_name}.config import AppConfig
from {project_name}.wiring import build_app_state
from {project_name}.routes import attach_routes


def create_app(cfg: AppConfig) -> FastAPI:
    app = FastAPI(title="{project_name}", version="0.1.0")
    state = build_app_state(cfg)
    app.state.rag = state
    attach_routes(app)
    return app


def run_server(cfg: AppConfig) -> None:
    host = str(cfg.raw.get("service", {{}}).get("host", "0.0.0.0"))
    port = int(cfg.raw.get("service", {{}}).get("port", 8000))

    app = create_app(cfg)
    uvicorn.run(app, host=host, port=port)
"""


def t_routes_py(project_name: str) -> str:
    return f"""\
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from {project_name}.rag import answer_question


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(8, ge=1, le=50)


class SourceItem(BaseModel):
    doc_id: str
    chunk_id: str
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]


def attach_routes(app: FastAPI) -> None:
    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {{"status": "ok"}}

    @app.post("/v1/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        state = app.state.rag
        ans, sources = answer_question(
            state=state,
            question=req.question,
            top_k=req.top_k,
        )
        return QueryResponse(
            answer=ans,
            sources=[SourceItem(**s) for s in sources],
        )
"""


def t_wiring_py(project_name: str) -> str:
    return f"""\
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from {project_name}.config import AppConfig
from {project_name}.stores.registry import build_store_registry
from {project_name}.retrieval.bm25_retriever import BM25Retriever
from {project_name}.data.chunks import ChunkStore
from rag_common.bm25_portable import load_bm25_portable_whoosh


@dataclass
class AppState:
    stores: Dict[str, Any]
    chunks: ChunkStore
    bm25: BM25Retriever | None


def build_app_state(cfg: AppConfig) -> AppState:
    stores = build_store_registry(cfg.raw)

    # chunks
    chunks_cfg = cfg.raw.get("data", {{}}).get("chunks", {{}})
    chunk_store = ChunkStore(
        store=stores[chunks_cfg["store"]],
        path=str(chunks_cfg["path"]),
    )

    # bm25
    bm25_ret = None
    art = cfg.raw.get("artifacts", {{}}).get("bm25", None)
    if art and bool(cfg.raw.get("retrieval", {{}}).get("bm25", {{}}).get("enabled", True)):
        bm25_obj, doc_ids = load_bm25_portable_whoosh(
            store=stores[art["store"]],
            path=str(art["path"]),
        )
        bm25_ret = BM25Retriever(bm25=bm25_obj, doc_ids=doc_ids)

    return AppState(stores=stores, chunks=chunk_store, bm25=bm25_ret)
"""


def t_rag_py(project_name: str) -> str:
    return f"""\
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from {project_name}.wiring import AppState


def answer_question(*, state: AppState, question: str, top_k: int) -> Tuple[str, List[Dict[str, Any]]]:
    # 1) retrieval
    hits: List[Dict[str, Any]] = []
    if state.bm25 is not None:
        hits = state.bm25.search(question, top_k=top_k)

    # 2) fetch chunk texts
    sources: List[Dict[str, Any]] = []
    for h in hits:
        chunk_id = h["chunk_id"]
        row = state.chunks.get(chunk_id)
        if row is None:
            continue
        sources.append(
            {{
                "doc_id": str(row.get("doc_id", "")),
                "chunk_id": str(chunk_id),
                "score": float(h["score"]),
                "text": str(row.get("chunk_text", "")),
            }}
        )

    # 3) generation (placeholder)
    # 你后面把这块替换成真实 LLM：Gemini/OpenAI/HF
    if not sources:
        return "I could not find relevant context.", []

    answer = "DUMMY_ANSWER: " + sources[0]["text"][:500]
    return answer, sources
"""


def t_chunks_py(project_name: str) -> str:
    return f"""\
from __future__ import annotations

from typing import Any, Dict, Optional

from rag_common.jsonl import read_jsonl_map_by_key
from {project_name}.stores.base import Store


class ChunkStore:
    def __init__(self, *, store: Store, path: str) -> None:
        self.store = store
        self.path = str(path)
        self._map: Optional[Dict[str, Dict[str, Any]]] = None

    def _ensure_loaded(self) -> None:
        if self._map is not None:
            return
        data = self.store.read_text(self.path)
        self._map = read_jsonl_map_by_key(data, key="chunk_id")

    def get(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        self._ensure_loaded()
        assert self._map is not None
        return self._map.get(str(chunk_id))
"""


def t_bm25_retriever_py(project_name: str) -> str:
    return """\
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi
from rag_common.bm25_portable import whoosh_tokenize_query


@dataclass
class BM25Retriever:
    bm25: BM25Okapi
    doc_ids: List[str]  # aligned with corpus order

    def search(self, query: str, *, top_k: int = 10) -> List[Dict[str, Any]]:
        q_tokens = whoosh_tokenize_query(query)
        scores = self.bm25.get_scores(q_tokens)

        pairs = list(enumerate(scores))
        pairs.sort(key=lambda x: float(x[1]), reverse=True)

        out: List[Dict[str, Any]] = []
        for idx, s in pairs[: int(top_k)]:
            out.append({"chunk_id": self.doc_ids[idx], "score": float(s)})
        return out
"""


def t_store_base_py(project_name: str) -> str:
    return """\
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable


class Store(ABC):
    @abstractmethod
    def exists(self, path: str) -> bool: ...

    @abstractmethod
    def read_text(self, path: str, encoding: str = "utf-8") -> str: ...

    @abstractmethod
    def write_text(self, path: str, content: str, encoding: str = "utf-8") -> None: ...

    @abstractmethod
    def read_bytes(self, path: str) -> bytes: ...

    @abstractmethod
    def write_bytes(self, path: str, content: bytes) -> None: ...

    @abstractmethod
    def list(self, prefix: str) -> Iterable[str]: ...
"""


def t_store_filesystem_py(project_name: str) -> str:
    return f"""\
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from {project_name}.stores.base import Store


class FilesystemStore(Store):
    def __init__(self, *, root: str) -> None:
        self.root = Path(root)

    def _p(self, path: str) -> Path:
        path = str(path).lstrip("/")
        return self.root / path

    def exists(self, path: str) -> bool:
        return self._p(path).exists()

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        return self._p(path).read_text(encoding=encoding)

    def write_text(self, path: str, content: str, encoding: str = "utf-8") -> None:
        p = self._p(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding=encoding)

    def read_bytes(self, path: str) -> bytes:
        return self._p(path).read_bytes()

    def write_bytes(self, path: str, content: bytes) -> None:
        p = self._p(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)

    def list(self, prefix: str) -> Iterable[str]:
        base = self._p(prefix)
        if not base.exists():
            return []
        if base.is_file():
            return [prefix]
        out = []
        for p in base.rglob("*"):
            if p.is_file():
                out.append(str(p.relative_to(self.root)).replace("\\\\", "/"))
        return out
"""


def t_store_registry_py(project_name: str) -> str:
    return f"""\
from __future__ import annotations

from typing import Any, Dict

from {project_name}.stores.filesystem import FilesystemStore


def build_store_registry(cfg: Dict[str, Any]) -> Dict[str, Any]:
    stores_cfg = cfg.get("stores", {{}})
    if not isinstance(stores_cfg, dict):
        raise ValueError("cfg['stores'] must be a mapping")

    out: Dict[str, Any] = {{}}
    for name, sc in stores_cfg.items():
        if not isinstance(sc, dict):
            raise ValueError(f"store {{name}} must be a mapping")
        kind = sc.get("kind")
        if kind == "filesystem":
            out[name] = FilesystemStore(root=str(sc["root"]))
        else:
            raise ValueError(f"Unsupported store kind: {{kind}} (store={{name}})")
    return out
"""


def t_common_jsonl_py() -> str:
    return """\
from __future__ import annotations

import json
from typing import Any, Dict


def read_jsonl_map_by_key(text: str, *, key: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            continue
        k = obj.get(key)
        if k is None:
            continue
        out[str(k)] = obj
    return out
"""


def t_common_bm25_portable_py() -> str:
    return """\
from __future__ import annotations

import pickle
from typing import Any, Dict, List, Tuple

from rank_bm25 import BM25Okapi
from whoosh.analysis import StemmingAnalyzer

_ANALYZER = StemmingAnalyzer()
_TOKENIZER_ID = "whoosh_stemming_v1"


def whoosh_tokenize(text: str) -> List[str]:
    return [t.text for t in _ANALYZER(text or "")]


def whoosh_tokenize_query(text: str) -> List[str]:
    return whoosh_tokenize(text)


def save_bm25_portable_whoosh(
    *,
    store: Any,
    path: str,
    corpus_tokens: List[List[str]],
    doc_ids: List[str],
) -> None:
    payload: Dict[str, Any] = {
        "version": 1,
        "tokenizer": _TOKENIZER_ID,
        "doc_ids": doc_ids,
        "corpus_tokens": corpus_tokens,
    }
    store.write_bytes(path, pickle.dumps(payload))


def load_bm25_portable_whoosh(
    *,
    store: Any,
    path: str,
) -> Tuple[BM25Okapi, List[str]]:
    payload = pickle.loads(store.read_bytes(path))
    tok = payload.get("tokenizer")
    if tok is not None and tok != _TOKENIZER_ID:
        raise ValueError(f"BM25 tokenizer mismatch: expected={_TOKENIZER_ID} got={tok}")

    bm25 = BM25Okapi(payload["corpus_tokens"])
    return bm25, payload["doc_ids"]
"""


def t_gitignore() -> str:
    return """\
__pycache__/
*.pyc
*.pyo
*.pyd
.venv/
.env
dist/
build/
*.egg-info/
data/
"""


def t_dockerfile(project_name: str) -> str:
    return f"""\
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install -U pip \\
 && pip install -e .

EXPOSE 8000
CMD ["rag", "serve", "--config", "configs/rag.yaml"]
"""


def build_files(project_name: str) -> List[FileSpec]:
    p = project_name
    files: List[FileSpec] = []

    files.append(FileSpec("pyproject.toml", t_pyproject(p)))
    files.append(FileSpec("README.md", t_readme(p)))
    files.append(FileSpec(".gitignore", t_gitignore()))
    files.append(FileSpec("Dockerfile", t_dockerfile(p)))
    files.append(FileSpec("configs/rag.yaml", t_configs_rag_yaml(p)))

    files.append(FileSpec(f"src/{p}/__init__.py", t_src_init()))
    files.append(FileSpec(f"src/{p}/cli.py", t_cli(p)))
    files.append(FileSpec(f"src/{p}/config.py", t_config_py(p)))
    files.append(FileSpec(f"src/{p}/server.py", t_server_py(p)))
    files.append(FileSpec(f"src/{p}/routes.py", t_routes_py(p)))
    files.append(FileSpec(f"src/{p}/wiring.py", t_wiring_py(p)))
    files.append(FileSpec(f"src/{p}/rag.py", t_rag_py(p)))

    files.append(FileSpec(f"src/{p}/data/__init__.py", "__all__ = []\n"))
    files.append(FileSpec(f"src/{p}/data/chunks.py", t_chunks_py(p)))

    files.append(FileSpec(f"src/{p}/retrieval/__init__.py", "__all__ = []\n"))
    files.append(FileSpec(f"src/{p}/retrieval/bm25_retriever.py", t_bm25_retriever_py(p)))

    files.append(FileSpec(f"src/{p}/stores/__init__.py", "__all__ = []\n"))
    files.append(FileSpec(f"src/{p}/stores/base.py", t_store_base_py(p)))
    files.append(FileSpec(f"src/{p}/stores/filesystem.py", t_store_filesystem_py(p)))
    files.append(FileSpec(f"src/{p}/stores/registry.py", t_store_registry_py(p)))

    files.append(FileSpec("src/rag_common/__init__.py", "__all__ = []\n"))
    files.append(FileSpec("src/rag_common/jsonl.py", t_common_jsonl_py()))
    files.append(FileSpec("src/rag_common/bm25_portable.py", t_common_bm25_portable_py()))

    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="Python package name for final RAG service (e.g., rag_service)")
    ap.add_argument("--root", default=".", help="Where to create the project (default: current dir)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = ap.parse_args()

    project_name = _norm(args.name).replace("-", "_")
    if not project_name or any(c.isspace() for c in project_name):
        raise SystemExit("Invalid --name")

    root = Path(args.root).resolve()
    _mkdir(root)

    files = build_files(project_name)

    created: List[str] = []
    skipped: List[str] = []

    for fs in files:
        dst = root / fs.relpath
        if dst.exists() and not args.force:
            skipped.append(_rel(dst, root))
            continue
        _write_text(dst, fs.content, force=True)
        created.append(_rel(dst, root))

    print(f"[OK] scaffold generated: {project_name}")
    if created:
        print("Created/Updated:")
        for x in created:
            print(f"  - {x}")
    if skipped:
        print("Skipped (exists, use --force to overwrite):")
        for x in skipped:
            print(f"  - {x}")


if __name__ == "__main__":
    main()
