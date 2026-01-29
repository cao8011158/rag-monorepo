# src/rq_pipeline/settings.py
from __future___toggle import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Set

import hashlib
import json

import yaml

SettingsDict = Dict[str, Any]


# =========================
# Public API
# =========================
def load_settings(path: str | Path) -> SettingsDict:
    """
    Load rq-pipeline YAML -> normalized nested dict settings.

    Guarantees:
    - defaults are applied (so required nested maps exist)
    - validation is executed (ValueError with clear messages)
    - runtime metadata is attached into settings["_meta"]
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError("pipeline config root must be a mapping (YAML dict)")

    s = apply_defaults(raw)
    validate_settings(s)

    s.setdefault("_meta", {})
    s["_meta"]["config_path"] = str(path)
    s["_meta"]["config_hash"] = hash_settings(s, exclude_keys={"_meta"})
    return s


def apply_defaults(raw: SettingsDict) -> SettingsDict:
    """
    Apply defaults to raw YAML, ensuring required nested objects exist.

    This version matches the NEW reranker-training config schema:

    - model_name: str
    - stores: {<name>: {kind, root, ...}}
    - inputs:
        pairs: {store, base, pairs}
        chunks: {store, base, chunks_file}
    - outputs:
        files: {store, base, train_path, valid_path, train_pair_path}
    - max_length, pair_format, data_split
    - Training: {Optimizer, output_dir, seed, num_epochs, ...}
    - lora: {enabled, qlora_4bit, r, alpha, dropout, target_modules}
    - bf16, fp16, num_workers
    """
    s: SettingsDict = _deep_copy_dict(raw)

    # ---- model ----
    s.setdefault("model_name", "")

    # ---- stores ----
    s.setdefault("stores", {})
    _must_be_mapping(s["stores"], "stores")

    # ---- inputs ----
    s.setdefault("inputs", {})
    _must_be_mapping(s["inputs"], "inputs")

    s["inputs"].setdefault("pairs", {})
    _must_be_mapping(s["inputs"]["pairs"], "inputs.pairs")
    ip = s["inputs"]["pairs"]
    ip.setdefault("store", "")
    ip.setdefault("base", "")
    ip.setdefault("pairs", "query_pack.jsonl")

    s["inputs"].setdefault("chunks", {})
    _must_be_mapping(s["inputs"]["chunks"], "inputs.chunks")
    ic = s["inputs"]["chunks"]
    ic.setdefault("store", "")
    ic.setdefault("base", "")
    ic.setdefault("chunks_file", "chunks.jsonl")

    # ---- outputs ----
    s.setdefault("outputs", {})
    _must_be_mapping(s["outputs"], "outputs")

    s["outputs"].setdefault("files", {})
    _must_be_mapping(s["outputs"]["files"], "outputs.files")
    of = s["outputs"]["files"]
    of.setdefault("store", "")
    of.setdefault("base", "")

    # file paths under outputs.files.base
    of.setdefault("train_path", "processed/train_query_pack.jsonl")
    of.setdefault("valid_path", "processed/valid_query_pack.jsonl")
    of.setdefault("train_pair_path", "processed/train_pair_epoch_n.jsonl")

    # ---- tokenization / format / split ----
    s.setdefault("max_length", 512)
    s.setdefault("pair_format", "query_doc")  # only "query_doc" for now
    s.setdefault("data_split", 0.85)

    # ---- training ----
    s.setdefault("Training", {})
    _must_be_mapping(s["Training"], "Training")
    tr = s["Training"]
    tr.setdefault("Optimizer", "AdamW")
    tr.setdefault("output_dir", "")
    tr.setdefault("seed", 42)
    tr.setdefault("num_epochs", 1)

    # sampling knobs
    tr.setdefault("hard_negative_per_positive", 4)
    tr.setdefault("random_negative_per_positive", 1)
    tr.setdefault("random_neg_ratio", 0.0)

    # optimizer knobs
    tr.setdefault("lr", 2.0e-5)
    tr.setdefault("weight_decay", 0.0)
    tr.setdefault("warmup_ratio", 0.0)

    # batch / schedule
    tr.setdefault("per_device_train_batch_size", 8)
    tr.setdefault("per_device_eval_batch_size", 16)
    tr.setdefault("grad_accum_steps", 1)

    tr.setdefault("log_every_steps", 50)
    tr.setdefault("eval_every_steps", 200)
    tr.setdefault("save_every_steps", 200)
    tr.setdefault("max_steps", None)  # optional int cap

    # ---- lora ----
    s.setdefault("lora", {})
    _must_be_mapping(s["lora"], "lora")
    lora = s["lora"]
    lora.setdefault("enabled", False)
    lora.setdefault("qlora_4bit", False)
    lora.setdefault("r", 16)
    lora.setdefault("alpha", 32)
    lora.setdefault("dropout", 0.05)
    lora.setdefault("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])

    # ---- precision ----
    s.setdefault("bf16", False)
    s.setdefault("fp16", False)

    # ---- misc ----
    s.setdefault("num_workers", 0)

    return s


def validate_settings(s: SettingsDict) -> None:
    """
    Validate normalized settings dict (after apply_defaults).
    Raises ValueError with explicit messages.
    """
    # ---- model ----
    _require_nonempty_str(s.get("model_name"), "model_name")

    # ---- stores ----
    if not s.get("stores"):
        raise ValueError("stores is required")
    _must_be_mapping(s["stores"], "stores")

    # validate each store minimal schema
    for name, cfg in s["stores"].items():
        if not isinstance(cfg, dict):
            raise ValueError(f"stores.{name} must be a mapping (YAML dict)")
        _require_nonempty_str(cfg.get("kind"), f"stores.{name}.kind")
        # filesystem store requires root
        if str(cfg.get("kind")) == "filesystem":
            _require_nonempty_str(cfg.get("root"), f"stores.{name}.root")

    # ---- inputs ----
    inp = s["inputs"]
    if not isinstance(inp, dict):
        raise ValueError("inputs must be a mapping (YAML dict)")

    ip = inp["pairs"]
    _require_nonempty_str(ip.get("store"), "inputs.pairs.store")
    _require_nonempty_str(ip.get("base"), "inputs.pairs.base")
    _require_nonempty_str(ip.get("pairs"), "inputs.pairs.pairs")

    ic = inp["chunks"]
    _require_nonempty_str(ic.get("store"), "inputs.chunks.store")
    _require_nonempty_str(ic.get("base"), "inputs.chunks.base")
    _require_nonempty_str(ic.get("chunks_file"), "inputs.chunks.chunks_file")

    # ---- outputs ----
    out = s["outputs"]
    if not isinstance(out, dict):
        raise ValueError("outputs must be a mapping (YAML dict)")

    of = out.get("files")
    if not isinstance(of, dict):
        raise ValueError("outputs.files must be a mapping (YAML dict)")

    _require_nonempty_str(of.get("store"), "outputs.files.store")
    _require_nonempty_str(of.get("base"), "outputs.files.base")
    _require_nonempty_str(of.get("train_path"), "outputs.files.train_path")
    _require_nonempty_str(of.get("valid_path"), "outputs.files.valid_path")
    _require_nonempty_str(of.get("train_pair_path"), "outputs.files.train_pair_path")

    # ---- tokenization / format / split ----
    _as_int(s.get("max_length"), "max_length", min_value=8)
    _validate_enum(str(s.get("pair_format")), {"query_doc"}, "pair_format")
    split = _as_float(s.get("data_split"), "data_split", min_value=0.0, max_value=1.0)
    if not (0.0 < split < 1.0):
        raise ValueError(f"data_split must be in (0,1), got {split}")

    # ---- training ----
    tr = s["Training"]
    _must_be_mapping(tr, "Training")
    _require_nonempty_str(tr.get("Optimizer"), "Training.Optimizer")
    _require_nonempty_str(tr.get("output_dir"), "Training.output_dir")
    _as_int(tr.get("seed"), "Training.seed")
    _as_int(tr.get("num_epochs"), "Training.num_epochs", min_value=1)

    _as_int(tr.get("hard_negative_per_positive"), "Training.hard_negative_per_positive", min_value=0)
    _as_int(tr.get("random_negative_per_positive"), "Training.random_negative_per_positive", min_value=0)
    _as_float(tr.get("random_neg_ratio"), "Training.random_neg_ratio", min_value=0.0, max_value=1.0)

    _as_float(tr.get("lr"), "Training.lr", min_value=0.0)
    _as_float(tr.get("weight_decay"), "Training.weight_decay", min_value=0.0)
    _as_float(tr.get("warmup_ratio"), "Training.warmup_ratio", min_value=0.0, max_value=1.0)

    _as_int(tr.get("per_device_train_batch_size"), "Training.per_device_train_batch_size", min_value=1)
    _as_int(tr.get("per_device_eval_batch_size"), "Training.per_device_eval_batch_size", min_value=1)
    _as_int(tr.get("grad_accum_steps"), "Training.grad_accum_steps", min_value=1)

    _as_int(tr.get("log_every_steps"), "Training.log_every_steps", min_value=1)
    _as_int(tr.get("eval_every_steps"), "Training.eval_every_steps", min_value=1)
    _as_int(tr.get("save_every_steps"), "Training.save_every_steps", min_value=1)

    if tr.get("max_steps") is not None:
        _as_int(tr.get("max_steps"), "Training.max_steps", min_value=1)

    # ---- lora ----
    lora = s["lora"]
    _must_be_mapping(lora, "lora")
    if not isinstance(lora.get("enabled"), bool):
        raise ValueError("lora.enabled must be boolean")
    if not isinstance(lora.get("qlora_4bit"), bool):
        raise ValueError("lora.qlora_4bit must be boolean")
    _as_int(lora.get("r"), "lora.r", min_value=1)
    _as_int(lora.get("alpha"), "lora.alpha", min_value=1)
    _as_float(lora.get("dropout"), "lora.dropout", min_value=0.0, max_value=1.0)
    tm = lora.get("target_modules")
    if not isinstance(tm, list) or not all(isinstance(x, str) and x.strip() for x in tm):
        raise ValueError("lora.target_modules must be a list[str] (non-empty strings)")

    # ---- precision ----
    if not isinstance(s.get("bf16"), bool):
        raise ValueError("bf16 must be boolean")
    if not isinstance(s.get("fp16"), bool):
        raise ValueError("fp16 must be boolean")
    if bool(s.get("bf16")) and bool(s.get("fp16")):
        raise ValueError("bf16 and fp16 cannot both be true")

    # ---- misc ----
    _as_int(s.get("num_workers"), "num_workers", min_value=0)

    # ---- referenced stores exist ----
    referenced: Set[str] = set()
    referenced.add(str(ip.get("store", "") or ""))
    referenced.add(str(ic.get("store", "") or ""))
    referenced.add(str(of.get("store", "") or ""))

    referenced.discard("")
    missing = [name for name in sorted(referenced) if name not in s["stores"]]
    if missing:
        raise ValueError(f"stores missing definitions for: {missing}")


def hash_settings(s: SettingsDict, *, exclude_keys: Optional[Set[str]] = None) -> str:
    """
    Stable hash for settings dict (used as config fingerprint).
    """
    exclude_keys = exclude_keys or set()
    filtered = {k: v for k, v in s.items() if k not in exclude_keys}
    blob = json.dumps(filtered, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# =========================
# Internal helpers
# =========================
def _deep_copy_dict(d: SettingsDict) -> SettingsDict:
    # yaml-safe types -> json roundtrip keeps it simple
    return json.loads(json.dumps(d, ensure_ascii=False))


def _must_be_mapping(v: Any, path: str) -> None:
    if not isinstance(v, dict):
        raise ValueError(f"{path} must be a mapping (YAML dict)")


def _require_nonempty_str(v: Any, path: str) -> str:
    vv = str(v or "")
    if not vv.strip():
        raise ValueError(f"{path} is required")
    return vv


def _validate_enum(v: str, allowed: Set[str], path: str) -> None:
    if v not in allowed:
        raise ValueError(f"{path} must be one of {sorted(allowed)}, got {v!r}")


def _as_int(v: Any, path: str, *, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    try:
        x = int(v)
    except Exception as e:
        raise ValueError(f"{path} must be int-like, got {v!r}") from e
    if min_value is not None and x < min_value:
        raise ValueError(f"{path} must be >= {min_value}, got {x}")
    if max_value is not None and x > max_value:
        raise ValueError(f"{path} must be <= {max_value}, got {x}")
    return x


def _as_float(v: Any, path: str, *, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    try:
        x = float(v)
    except Exception as e:
        raise ValueError(f"{path} must be float-like, got {v!r}") from e
    if min_value is not None and x < min_value:
        raise ValueError(f"{path} must be >= {min_value}, got {x}")
    if max_value is not None and x > max_value:
        raise ValueError(f"{path} must be <= {max_value}, got {x}")
    return x
