from __future__ import annotations

from pathlib import Path

from qr_pipeline.settings import load_settings


def test_load_settings_smoke(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        """input:
  input_store: fs_local
  input_path: ce_out/chunks/chunks.jsonl
outputs:
  queries:
    store: fs_local
    base: qr_out/queries
stores:
  fs_local:
    kind: filesystem
    root: data
""",
        encoding="utf-8",
    )
    cfg = load_settings(p)
    assert cfg["input"]["input_store"] == "fs_local"
