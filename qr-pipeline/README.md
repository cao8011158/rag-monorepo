qr-pipeline

qr-pipeline 用于生成 reranker 训练数据：

输入：来自 ce-pipeline 的 chunks.jsonl + faiss.index + id_map.jsonl + bm25.pkl

流程：chunk -> LLM 生成 query -> retrieval top-k -> 选 1 个 positive + 6 个 hard negatives

输出：约 12k 条 pairwise 训练样本（JSONL）

目录结构（关键部分）
qr-pipeline/
configs/
pipeline.yaml
src/qr_pipeline/
cli.py
settings.py
query_generation.py
processing/
embedder.py
near_dedup.py
llm/
stores/
tests/

安装

1. 创建虚拟环境（推荐）

Windows (PowerShell)：

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip

macOS/Linux：

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

2. 安装本项目（可编辑模式）

在项目根目录（有 pyproject.toml 的地方）执行：

pip install -e .

如果你在 pyproject.toml 里定义了可选依赖（例如 .[dev]），就用：

pip install -e ".[dev]"

配置文件

默认配置在：

configs/pipeline.yaml

你已经把 ce-pipeline 的 artifacts 写进了：

inputs.ce_artifacts.chunks

inputs.ce_artifacts.vector_index

inputs.ce_artifacts.bm25_index

输出在：

outputs.base: rq_out（挂在 stores.fs_local.root: data 下）

运行方式（CLI）

你项目里有 src/qr_pipeline/cli.py，通常有两种启动方式：

方式 A：如果你在 pyproject 里配置了 console_scripts（推荐）

可能的命令名一般是 qr / rq / qr-pipeline 之一。你可以先试：

qr --help

如果报 “command not found”，就用方式 B。

方式 B：直接用 Python 模块方式运行（一定能用）

在项目根目录执行：

python -m qr_pipeline.cli --help

一键跑完整流程（推荐）

假设 CLI 提供 run 子命令（你的 ce-pipeline 就是这种风格），则：

python -m qr_pipeline.cli run --config configs/pipeline.yaml

如果你确实配了 console script（比如 qr），那么就是：

qr-qg --config configs/pipeline.yaml
pair-pair --config configs/pipeline.yaml
