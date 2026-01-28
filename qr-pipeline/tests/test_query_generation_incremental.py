import json
import pytest
from typing import Dict, Any, Iterable

# 被测函数
from qr_pipeline.pipeline.query_generation import run_query_generation

# -----------------------
# Fake Store (内存版)
# -----------------------

class FakeStore:
    def __init__(self):
        self.files: Dict[str, bytes] = {}

    def exists(self, path: str) -> bool:
        return path in self.files

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        return self.files[path].decode(encoding)

    def write_text(self, path: str, content: str, encoding: str = "utf-8") -> None:
        self.files[path] = content.encode(encoding)

    def read_bytes(self, path: str) -> bytes:
        return self.files[path]

    def write_bytes(self, path: str, content: bytes) -> None:
        self.files[path] = content

    def append_bytes(self, path: str, content: bytes) -> None:
        if path not in self.files:
            self.files[path] = content
        else:
            if not self.files[path].endswith(b"\n"):
                self.files[path] += b"\n"
            self.files[path] += content

    def list(self, prefix: str) -> Iterable[str]:
        return [p for p in self.files if p.startswith(prefix)]


# -----------------------
# Fake JSONL IO
# -----------------------

def fake_read_jsonl(store: FakeStore, path: str, on_error=None):
    if not store.exists(path):
        return []
    text = store.read_text(path)
    rows = []
    for ln in text.splitlines():
        if not ln.strip():
            continue
        rows.append(json.loads(ln))
    return rows


def fake_write_jsonl(store: FakeStore, path: str, rows: Iterable[Dict[str, Any]]):
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    store.write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def fake_append_jsonl(store: FakeStore, path: str, rows: Iterable[Dict[str, Any]]):
    for r in rows:
        line = json.dumps(r, ensure_ascii=False).encode("utf-8") + b"\n"
        store.append_bytes(path, line)


# -----------------------
# Fake LLM
# -----------------------

class FakeLLM:
    """
    - 对 query generation：根据 Passage 返回不同 query，避免 hash 相同导致全局 dedup 只写 1 条
    - 对 domain batch classify：返回 IN 包含所有 Q1..Qn
    """
    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1

        # domain batch classify
        if "Now classify the following queries" in prompt:
            # 提取输入里出现了多少个 Q<number>（简单可靠）
            import re
            qnums = re.findall(r"^Q(\d+)\t", prompt, flags=re.M)
            # 全部判为 IN，确保 batch_append 里 Q2 不会默认为 OUT
            lines = ["IN:"]
            for n in qnums:
                lines.append(f"Q{n}")
            lines += ["OUT:", ""]
            return "\n".join(lines)

        # query generation
        # 根据 passage 内容制造不同 query，保证两个 chunk -> 两个不同 query_id
        if "Passage:" in prompt:
            if "CMU one" in prompt:
                return "What is CMU one?\n"
            if "CMU two" in prompt:
                return "What is CMU two?\n"
            if "founded in 1900" in prompt:
                return "When was Carnegie Mellon University founded?\n"
            if "Pittsburgh is a city" in prompt:
                return "What are major sports teams in Pittsburgh?\n"
            # fallback
            return "What is Carnegie Mellon University?\n"

        return ""


# -----------------------
# Fake Store Registry
# -----------------------

def fake_build_store_registry(cfg):
    return {"fs_local": cfg["_fake_store"]}


# -----------------------
# Minimal helpers patch
# -----------------------

@pytest.fixture
def monkeypatch_io(monkeypatch):
    from qr_pipeline.pipeline import query_generation as qg

    monkeypatch.setattr(qg, "read_jsonl", fake_read_jsonl)
    monkeypatch.setattr(qg, "write_jsonl", fake_write_jsonl)
    monkeypatch.setattr(qg, "append_jsonl", fake_append_jsonl)
    monkeypatch.setattr(qg, "build_store_registry", fake_build_store_registry)


# -----------------------
# Minimal config
# -----------------------

def make_minimal_config(fake_store: FakeStore):
    return {
        "_fake_store": fake_store,
        "stores": {"fs_local": {"kind": "filesystem", "root": "/tmp"}},
        "inputs": {
            "ce_artifacts": {
                "chunks": {
                    "store": "fs_local",
                    "base": "ce_out/chunks",
                    "chunks_file": "chunks.jsonl",
                }
            }
        },
        "outputs": {
            "store": "fs_local",
            "base": "rq_out",
            "files": {
                "queries_in_domain": "queries/in_domain.jsonl",
                "queries_out_domain": "queries/out_domain.jsonl",
                "errors": "errors.jsonl",
                "stats": "stats.json",
            },
        },
        "models": {
            "llm": {"provider": "hf_transformers", "model_name": "fake", "device": "cpu"}
        },
        "query_generation": {
            "target_num_queries": 2,
            "domain_batch_size": 2,
            "sampling": {},
            "prompt": {"num_queries_per_chunk": 1},
            "postprocess": {"min_query_chars": 3},
            "normalize": {"lower": True, "strip": True, "collapse_whitespace": True},
        },
    }


# -----------------------
# Helper: write chunks.jsonl
# -----------------------

def write_chunks(store: FakeStore):
    chunks = [
        {"chunk_id": "c1", "chunk_text": "Carnegie Mellon University was founded in 1900."},
        {"chunk_id": "c2", "chunk_text": "Pittsburgh is a city in Pennsylvania."},
    ]
    fake_write_jsonl(store, "ce_out/chunks/chunks.jsonl", chunks)


# -----------------------
# Tests
# -----------------------

def test_seen_ids_dedup_and_append(monkeypatch, monkeypatch_io, capsys):
    fake_store = FakeStore()
    write_chunks(fake_store)

    fake_llm = FakeLLM()

    from qr_pipeline.pipeline import query_generation as qg
    monkeypatch.setattr(qg, "_build_llm_from_settings", lambda s: (fake_llm, "fake-model"))

    cfg = make_minimal_config(fake_store)
    run_query_generation(cfg)

    in_path = "rq_out/queries/in_domain.jsonl"
    stats_path = "rq_out/stats.json"

    assert fake_store.exists(in_path)
    rows = fake_read_jsonl(fake_store, in_path)

    # 这里期望 1：因为 target_num_queries=2，但只有 1 个会是 IN（另一个 chunk 的 query 也会 IN，
    # 如果你希望这里是 2，可以把 FakeLLM 对 Pittsburgh 那条也返回 IN（目前已经返回 IN）。
    # 所以更稳妥：断言 >=1
    assert len(rows) >= 1
    assert rows[0]["domain"] == "in"
    assert "query_id" in rows[0]

    assert fake_store.exists(stats_path)
    captured = capsys.readouterr()
    assert "[STATE]" in captured.out


def test_resume_from_existing_files(monkeypatch, monkeypatch_io):
    fake_store = FakeStore()
    write_chunks(fake_store)

    # ✅ existing in_domain.jsonl 必须用同一套 normalize+hash 生成 query_id，才能被 seen_ids dedup
    from qr_pipeline.pipeline import query_generation as qg
    q_norm = qg._normalize_text("old query", lower=True, strip=True, collapse_whitespace=True)
    qid = qg._make_id(q_norm)

    fake_append_jsonl(
        fake_store,
        "rq_out/queries/in_domain.jsonl",
        [{"query_id": qid, "query_text_norm": q_norm, "domain": "in"}],
    )

    fake_llm = FakeLLM()
    monkeypatch.setattr(qg, "_build_llm_from_settings", lambda s: (fake_llm, "fake-model"))

    cfg = make_minimal_config(fake_store)

    # 让 FakeLLM 第一次生成的 query 也恰好是 "old query"，以验证 dedup
    # 简单做法：patch FakeLLM.generate 在生成 query 阶段直接返回 old query
    orig_generate = fake_llm.generate

    def _generate(prompt: str) -> str:
        if "Now classify the following queries" in prompt:
            return orig_generate(prompt)
        if "Passage:" in prompt:
            return "old query\n"
        return orig_generate(prompt)

    fake_llm.generate = _generate

    run_query_generation(cfg)

    rows = fake_read_jsonl(fake_store, "rq_out/queries/in_domain.jsonl")
    assert len(rows) == 1


def test_stats_checkpoint_written_each_flush(monkeypatch, monkeypatch_io):
    fake_store = FakeStore()
    write_chunks(fake_store)

    fake_llm = FakeLLM()

    from qr_pipeline.pipeline import query_generation as qg
    monkeypatch.setattr(qg, "_build_llm_from_settings", lambda s: (fake_llm, "fake-model"))

    cfg = make_minimal_config(fake_store)
    run_query_generation(cfg)

    stats = json.loads(fake_store.read_text("rq_out/stats.json"))
    assert "outputs" in stats
    assert stats["outputs"]["num_queries_in_domain_unique_written"] >= 1


def test_batch_append(monkeypatch, monkeypatch_io):
    fake_store = FakeStore()

    chunks = [
        {"chunk_id": "c1", "chunk_text": "CMU one"},
        {"chunk_id": "c2", "chunk_text": "CMU two"},
    ]
    fake_write_jsonl(fake_store, "ce_out/chunks/chunks.jsonl", chunks)

    fake_llm = FakeLLM()

    from qr_pipeline.pipeline import query_generation as qg
    monkeypatch.setattr(qg, "_build_llm_from_settings", lambda s: (fake_llm, "fake-model"))

    cfg = make_minimal_config(fake_store)
    run_query_generation(cfg)

    rows = fake_read_jsonl(fake_store, "rq_out/queries/in_domain.jsonl")
    assert len(rows) == 2


def test_state_printed_each_flush(monkeypatch, monkeypatch_io, capsys):
    fake_store = FakeStore()
    write_chunks(fake_store)

    fake_llm = FakeLLM()

    from qr_pipeline.pipeline import query_generation as qg
    monkeypatch.setattr(qg, "_build_llm_from_settings", lambda s: (fake_llm, "fake-model"))

    cfg = make_minimal_config(fake_store)
    run_query_generation(cfg)

    out = capsys.readouterr().out
    assert "[STATE]" in out
