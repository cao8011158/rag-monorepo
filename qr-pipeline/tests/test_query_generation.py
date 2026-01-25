from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Iterator, List

import pytest


class MemoryStore:
    def __init__(self) -> None:
        self._fs: Dict[str, bytes] = {}

    def exists(self, path: str) -> bool:
        return path in self._fs

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        if path not in self._fs:
            raise FileNotFoundError(path)
        return self._fs[path].decode(encoding)

    def write_text(self, path: str, content: str, encoding: str = "utf-8") -> None:
        self._fs[path] = content.encode(encoding)

    def read_bytes(self, path: str) -> bytes:
        if path not in self._fs:
            raise FileNotFoundError(path)
        return self._fs[path]

    def write_bytes(self, path: str, content: bytes) -> None:
        self._fs[path] = content

    def append_bytes(self, path: str, content: bytes) -> None:
        if path not in self._fs:
            self._fs[path] = b""
        self._fs[path] += content

    def list(self, prefix: str) -> Iterable[str]:
        for k in sorted(self._fs.keys()):
            if k.startswith(prefix):
                yield k


def _read_jsonl(store: MemoryStore, path: str, *, on_error=None) -> Iterator[Dict[str, Any]]:
    text = store.read_text(path)
    for i, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError("JSON line is not an object")
            yield obj
        except Exception as e:
            if on_error is None:
                raise
            on_error({
                "stage": "read_jsonl",
                "path": path,
                "line_no": i,
                "error": f"{type(e).__name__}: {e}",
                "line_preview": line[:200],
            })


def _write_jsonl(store: MemoryStore, path: str, rows: Iterable[Dict[str, Any]]) -> None:
    out_lines: List[str] = []
    for r in rows:
        out_lines.append(json.dumps(r, ensure_ascii=False))
    # 覆盖写入
    store.write_text(path, "\n".join(out_lines) + ("\n" if out_lines else ""))


def _append_jsonl(store: MemoryStore, path: str, rows: Iterable[Dict[str, Any]]) -> None:
    buf = b""
    for r in rows:
        buf += (json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8")
    store.append_bytes(path, buf)


class FakeLLM:
    """
    伪 LLM：
    - query 生成：每个 chunk 输出 3 行（含重复 + 域外）
    - domain 分类：包含 cmu/pittsburgh => IN，否则 OUT
    """
    def __init__(self) -> None:
        self.calls: List[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)

        if "strict text classifier" in prompt:
            q = prompt.split("Query:", 1)[-1].strip().lower()
            if ("cmu" in q) or ("pittsburgh" in q):
                return "IN"
            return "OUT"

        return "\n".join([
            "1) What is CMU famous for?",
            "2) What is CMU famous for?   ",  # 重复
            "3) best sushi recipe",            # 域外
        ])


def _make_settings() -> Dict[str, Any]:
    return {
        "stores": {
            "fs_local": {"kind": "filesystem", "root": "data"}
        },
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
                "errors": "errors.jsonl",
                "stats": "run_stats.json",
                # ✅ 关键：使用你 config 风格的路径
                "queries_in_domain": "queries/in_domain.jsonl",
                "queries_out_domain": "queries/out_domain.jsonl",
            },
        },
        "models": {
            "llm": {
                "provider": "hf_transformers",
                "model_name": "FAKE",
                "device": "cpu",
                "max_new_tokens": 64,
                "temperature": 0.7,
                "top_p": 0.9,
            }
        },
        "query_generation": {
            "target_num_queries": 2,  # 只按 IN-domain unique 早停
            "sampling": {
                "seed": 42,
                "strategy": "uniform_random",
                "max_chunks_considered": 6500,
            },
            "prompt": {
                "language": "en",
                "style": "information-seeking",
                "num_queries_per_chunk": 3,
                "max_chunk_chars": 1800,
                "diversify": True,
                "diversity_hints": ["fact", "definition", "relationship"],
                "avoid_near_duplicates": True,
            },
            "postprocess": {
                "min_query_chars": 8,
                "max_query_chars": 160,
            },
            "normalize": {
                "lower": True,
                "strip": True,
                "collapse_whitespace": True,
            },
        },
    }


def _seed_chunks(store: MemoryStore, path: str) -> None:
    rows = [
        {"chunk_id": "c1", "chunk_text": "Carnegie Mellon University is in Pittsburgh."},
        {"chunk_id": "c2", "chunk_text": "Pittsburgh has many bridges."},
        {"chunk_id": "c3", "chunk_text": "Some unrelated cooking content."},
    ]
    store.write_text(path, "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


def _load_jsonl_from_store(store: MemoryStore, path: str) -> List[Dict[str, Any]]:
    if not store.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    for line in store.read_text(path).splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def test_query_generation_uses_config_paths_and_splits_in_out_and_dedups(monkeypatch: pytest.MonkeyPatch) -> None:
    import qr_pipeline.pipeline.query_generation as qg

    s = _make_settings()
    store = MemoryStore()
    _seed_chunks(store, "ce_out/chunks/chunks.jsonl")

    # patch stores / io
    monkeypatch.setattr(qg, "build_store_registry", lambda _s: {"fs_local": store})
    monkeypatch.setattr(qg, "read_jsonl", _read_jsonl)
    monkeypatch.setattr(qg, "write_jsonl", _write_jsonl)
    monkeypatch.setattr(qg, "append_jsonl", _append_jsonl)

    fake_llm = FakeLLM()
    monkeypatch.setattr(qg, "_build_llm_from_settings", lambda _s: (fake_llm, "FAKE-MODEL"))

    stats = qg.run_query_generation(s)

    # ✅ 路径必须来自 config（rq_out + queries/xxx.jsonl）
    in_path = "rq_out/queries/in_domain.jsonl"
    out_path = "rq_out/queries/out_domain.jsonl"

    in_rows = _load_jsonl_from_store(store, in_path)
    out_rows = _load_jsonl_from_store(store, out_path)

    # FakeLLM 每个 chunk 都会产出同一个 IN query（CMU famous for）
    # => in-domain 去重后只有 1 条，但会聚合所有 chunk_id
    assert stats["outputs"]["num_queries_in_domain_unique_written"] == 1
    assert len(in_rows) == 1
    assert in_rows[0]["domain"] == "in"
    assert in_rows[0]["llm_model"] == "FAKE-MODEL"
    assert in_rows[0]["prompt_style"] == "information-seeking"
    assert in_rows[0]["query_text_norm"] == "what is cmu famous for?"
    assert set(in_rows[0]["source_chunk_ids"]).issuperset({"c1", "c2", "c3"})

    # out-domain：sushi recipe 去重后 1 条，聚合所有 chunk_id
    assert len(out_rows) == 1
    assert out_rows[0]["domain"] == "out"
    assert out_rows[0]["query_text_norm"] == "best sushi recipe"
    assert set(out_rows[0]["source_chunk_ids"]).issuperset({"c1", "c2", "c3"})


def test_domain_parse_failure_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    # ✅ FIX: match actual module location
    import qr_pipeline.pipeline.query_generation as qg

    s = _make_settings()
    store = MemoryStore()
    _seed_chunks(store, "ce_out/chunks/chunks.jsonl")

    monkeypatch.setattr(qg, "build_store_registry", lambda _s: {"fs_local": store})
    monkeypatch.setattr(qg, "read_jsonl", _read_jsonl)
    monkeypatch.setattr(qg, "write_jsonl", _write_jsonl)
    monkeypatch.setattr(qg, "append_jsonl", _append_jsonl)

    class BadDomainLLM(FakeLLM):
        def generate(self, prompt: str) -> str:
            if "strict text classifier" in prompt:
                return "MAYBE"
            return super().generate(prompt)

    bad_llm = BadDomainLLM()
    monkeypatch.setattr(qg, "_build_llm_from_settings", lambda _s: (bad_llm, "FAKE-MODEL"))

    qg.run_query_generation(s)

    in_rows = _load_jsonl_from_store(store, "rq_out/queries/in_domain.jsonl")
    out_rows = _load_jsonl_from_store(store, "rq_out/queries/out_domain.jsonl")
    assert in_rows == []
    assert out_rows == []

    errs = _load_jsonl_from_store(store, "rq_out/errors.jsonl")
    assert any(e.get("stage") == "parse_domain_label" for e in errs)


def test_read_jsonl_error_best_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    # ✅ FIX: match actual module location
    import qr_pipeline.pipeline.query_generation as qg

    s = _make_settings()
    store = MemoryStore()

    # 一行坏 JSON
    store.write_text(
        "ce_out/chunks/chunks.jsonl",
        "\n".join([
            json.dumps({"chunk_id": "c1", "chunk_text": "CMU in Pittsburgh"}, ensure_ascii=False),
            "{bad json",
            json.dumps({"chunk_id": "c2", "chunk_text": "Pittsburgh bridges"}, ensure_ascii=False),
        ]) + "\n",
    )

    monkeypatch.setattr(qg, "build_store_registry", lambda _s: {"fs_local": store})
    monkeypatch.setattr(qg, "read_jsonl", _read_jsonl)
    monkeypatch.setattr(qg, "write_jsonl", _write_jsonl)
    monkeypatch.setattr(qg, "append_jsonl", _append_jsonl)

    fake_llm = FakeLLM()
    monkeypatch.setattr(qg, "_build_llm_from_settings", lambda _s: (fake_llm, "FAKE-MODEL"))

    stats = qg.run_query_generation(s)

    assert stats["counters"]["read_errors"] == 1
    errs = _load_jsonl_from_store(store, "rq_out/errors.jsonl")
    assert any(e.get("stage") == "read_jsonl" for e in errs)
