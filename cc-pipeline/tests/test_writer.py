import orjson
from pathlib import Path

from cc_pipeline.clean.writer import append_jsonl


def test_append_jsonl_writes_valid_json_lines(tmp_path: Path):
    p = tmp_path / "out.jsonl"
    append_jsonl(p, {"a": 1})
    append_jsonl(p, {"b": "x"})

    lines = p.read_bytes().splitlines()
    assert len(lines) == 2
    assert orjson.loads(lines[0]) == {"a": 1}
    assert orjson.loads(lines[1]) == {"b": "x"}