from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# -----------------------------
# Helpers
# -----------------------------
def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _dedent(s: str) -> str:
    return textwrap.dedent(s).lstrip("\n")


# -----------------------------
# Project Scaffold
# -----------------------------
@dataclass
class ProjectSpec:
    name: str = "reranker_training"


def main() -> None:
    spec = ProjectSpec()
    root = Path.cwd() / spec.name

    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"[ERROR] Target folder already exists and is not empty: {root}")

    # Folders
    safe_mkdir(root / "src" / spec.name)
    safe_mkdir(root / "scripts")
    safe_mkdir(root / "configs")
    safe_mkdir(root / "data" / "raw")
    safe_mkdir(root / "data" / "processed")
    safe_mkdir(root / "outputs")
    safe_mkdir(root / "tests")

    # .gitignore
    write_text(
        root / ".gitignore",
        _dedent(
            """
            # Python
            __pycache__/
            *.pyc
            *.pyo
            *.pyd
            .Python
            .venv/
            env/
            build/
            dist/
            *.egg-info/
            .pytest_cache/
            .mypy_cache/
            .ruff_cache/

            # Data & outputs
            data/
            outputs/
            wandb/
            mlruns/

            # OS
            .DS_Store
            Thumbs.db
            """
        ),
    )

    # pyproject.toml (optional but recommended)
    write_text(
        root / "pyproject.toml",
        _dedent(
            f"""
            [build-system]
            requires = ["setuptools>=69.0.0", "wheel"]
            build-backend = "setuptools.build_meta"

            [project]
            name = "{spec.name}"
            version = "0.1.0"
            description = "Cross-encoder reranker LoRA/QLoRA training scaffold."
            readme = "README.md"
            requires-python = ">=3.10"
            license = {{ text = "MIT" }}
            dependencies = [
              "pyyaml>=6.0.1",
              "orjson>=3.10.0",
              "tqdm>=4.66.0",
              "numpy>=1.26.0",
              "scikit-learn>=1.4.0",
              "torch>=2.2.0",
              "transformers>=4.41.0",
              "datasets>=2.19.0",
              "accelerate>=0.33.0",
              "peft>=0.12.0",
              "bitsandbytes>=0.43.0",
            ]

            [project.optional-dependencies]
            dev = [
              "pytest>=8.0.0",
              "ruff>=0.6.0",
            ]

            [tool.setuptools]
            package-dir = {{"" = "src"}}

            [tool.ruff]
            line-length = 110
            """
        ),
    )

    # README
    write_text(
        root / "README.md",
        _dedent(
            f"""
            # {spec.name}

            A minimal, reproducible scaffold to fine-tune a **cross-encoder reranker** with **LoRA/QLoRA**.

            ## What you get
            - Pairwise training (q, pos, neg) for a reranker (cross-encoder)
            - **Negative resampling per epoch** (data augmentation)
            - YAML config driven training
            - Simple evaluation placeholders (MRR/nDCG can be added later)

            ## Quickstart

            ### 1) Create env
            ```bash
            python -m venv .venv
            # Windows:
            .venv\\Scripts\\activate
            # Linux/Mac:
            source .venv/bin/activate
            ```

            ### 2) Install
            You can choose either:
            - install as a package (recommended):
              ```bash
              pip install -U pip
              pip install -e ".[dev]"
              ```
            - or just run with `PYTHONPATH=src` (not recommended long-term):
              ```bash
              pip install -U pip
              pip install -r requirements_fallback.txt
              ```

            ### 3) Put data
            Input JSONL format (one line per query sample):
            ```json
            {{
              "query_text": "...",
              "positive": {{"doc_id": "D1", "text": "..."}},
              "negatives": [{{"doc_id": "N1", "text": "..."}}, ...]
            }}
            ```

            Put train/valid at:
            - `data/processed/train.jsonl`
            - `data/processed/valid.jsonl`

            ### 4) Run training
            ```bash
            python -m {spec.name}.train --config configs/train_qlora.yaml
            ```

            Outputs go to `outputs/`.

            ## Notes
            - For English chunks <= 2000 characters, `max_length=512` is a good default.
            - If you want MRR/nDCG, implement in `src/{spec.name}/metrics.py`.
            """
        ),
    )

    # requirements fallback (if user doesn't want editable install)
    write_text(
        root / "requirements_fallback.txt",
        _dedent(
            """
            pyyaml>=6.0.1
            orjson>=3.10.0
            tqdm>=4.66.0
            numpy>=1.26.0
            scikit-learn>=1.4.0
            torch>=2.2.0
            transformers>=4.41.0
            datasets>=2.19.0
            accelerate>=0.33.0
            peft>=0.12.0
            bitsandbytes>=0.43.0
            """
        ),
    )

    # configs
    write_text(
        root / "configs" / "train_qlora.yaml",
        _dedent(
            f"""
            # Base reranker model (cross-encoder).
            # Example (you choose one):
            # - BAAI/bge-reranker-v2-m3
            # - mixedbread-ai/mxbai-rerank-large-v1 (example name, use what you actually download)
            model_name: "BAAI/bge-reranker-v2-m3"

            # Data
            train_path: "data/processed/train.jsonl"
            valid_path: "data/processed/valid.jsonl"

            # Tokenization
            max_length: 512
            # combine strategy:
            # "query_doc" means we encode as (query, doc) pair for cross-encoder.
            pair_format: "query_doc"

            # Training
            output_dir: "outputs/run1"
            seed: 42
            num_epochs: 2
            lr: 2.0e-5
            weight_decay: 0.01
            warmup_ratio: 0.05
            per_device_train_batch_size: 8
            per_device_eval_batch_size: 16
            grad_accum_steps: 2
            log_every_steps: 20
            eval_every_steps: 200
            save_every_steps: 200
            max_steps: null  # set integer to cap training

            # Pairwise sampling
            # Re-sample negatives each epoch:
            negs_per_query_per_epoch: 2
            # if you already have hard negatives, keep it true; if not, mix random:
            mix_random_negs: true
            random_neg_ratio: 0.25

            # QLoRA / LoRA
            lora:
              enabled: true
              qlora_4bit: true
              r: 16
              alpha: 32
              dropout: 0.05
              # typical target modules for transformer blocks:
              target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]

            # Precision
            bf16: true
            fp16: false

            # Misc
            num_workers: 2
            """
        ),
    )

    # src package files
    write_text(
        root / "src" / spec.name / "__init__.py",
        _dedent(
            """
            __all__ = ["config", "data", "modeling", "train"]
            """
        ),
    )

    write_text(
        root / "src" / spec.name / "config.py",
        _dedent(
            """
            from __future__ import annotations

            from dataclasses import dataclass
            from typing import List, Optional

            import yaml


            @dataclass
            class LoraConfig:
                enabled: bool = True
                qlora_4bit: bool = True
                r: int = 16
                alpha: int = 32
                dropout: float = 0.05
                target_modules: List[str] = None  # will be set in from_dict

                @staticmethod
                def from_dict(d: dict) -> "LoraConfig":
                    cfg = LoraConfig()
                    cfg.enabled = bool(d.get("enabled", True))
                    cfg.qlora_4bit = bool(d.get("qlora_4bit", True))
                    cfg.r = int(d.get("r", 16))
                    cfg.alpha = int(d.get("alpha", 32))
                    cfg.dropout = float(d.get("dropout", 0.05))
                    cfg.target_modules = list(d.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]))
                    return cfg


            @dataclass
            class TrainConfig:
                model_name: str
                train_path: str
                valid_path: str
                max_length: int = 512
                pair_format: str = "query_doc"

                output_dir: str = "outputs/run1"
                seed: int = 42
                num_epochs: int = 2
                lr: float = 2e-5
                weight_decay: float = 0.01
                warmup_ratio: float = 0.05
                per_device_train_batch_size: int = 8
                per_device_eval_batch_size: int = 16
                grad_accum_steps: int = 2
                log_every_steps: int = 20
                eval_every_steps: int = 200
                save_every_steps: int = 200
                max_steps: Optional[int] = None

                negs_per_query_per_epoch: int = 2
                mix_random_negs: bool = True
                random_neg_ratio: float = 0.25

                lora: LoraConfig = None

                bf16: bool = True
                fp16: bool = False
                num_workers: int = 2

                @staticmethod
                def load(path: str) -> "TrainConfig":
                    with open(path, "r", encoding="utf-8") as f:
                        d = yaml.safe_load(f)

                    cfg = TrainConfig(
                        model_name=str(d["model_name"]),
                        train_path=str(d["train_path"]),
                        valid_path=str(d["valid_path"]),
                    )
                    cfg.max_length = int(d.get("max_length", cfg.max_length))
                    cfg.pair_format = str(d.get("pair_format", cfg.pair_format))
                    cfg.output_dir = str(d.get("output_dir", cfg.output_dir))
                    cfg.seed = int(d.get("seed", cfg.seed))
                    cfg.num_epochs = int(d.get("num_epochs", cfg.num_epochs))
                    cfg.lr = float(d.get("lr", cfg.lr))
                    cfg.weight_decay = float(d.get("weight_decay", cfg.weight_decay))
                    cfg.warmup_ratio = float(d.get("warmup_ratio", cfg.warmup_ratio))
                    cfg.per_device_train_batch_size = int(d.get("per_device_train_batch_size", cfg.per_device_train_batch_size))
                    cfg.per_device_eval_batch_size = int(d.get("per_device_eval_batch_size", cfg.per_device_eval_batch_size))
                    cfg.grad_accum_steps = int(d.get("grad_accum_steps", cfg.grad_accum_steps))
                    cfg.log_every_steps = int(d.get("log_every_steps", cfg.log_every_steps))
                    cfg.eval_every_steps = int(d.get("eval_every_steps", cfg.eval_every_steps))
                    cfg.save_every_steps = int(d.get("save_every_steps", cfg.save_every_steps))
                    cfg.max_steps = d.get("max_steps", None)
                    if cfg.max_steps is not None:
                        cfg.max_steps = int(cfg.max_steps)

                    cfg.negs_per_query_per_epoch = int(d.get("negs_per_query_per_epoch", cfg.negs_per_query_per_epoch))
                    cfg.mix_random_negs = bool(d.get("mix_random_negs", cfg.mix_random_negs))
                    cfg.random_neg_ratio = float(d.get("random_neg_ratio", cfg.random_neg_ratio))

                    cfg.lora = LoraConfig.from_dict(d.get("lora", {}))

                    cfg.bf16 = bool(d.get("bf16", cfg.bf16))
                    cfg.fp16 = bool(d.get("fp16", cfg.fp16))
                    cfg.num_workers = int(d.get("num_workers", cfg.num_workers))
                    return cfg
            """
        ),
    )

    write_text(
        root / "src" / spec.name / "io_utils.py",
        _dedent(
            """
            from __future__ import annotations

            import orjson
            from typing import Any, Dict, Iterator


            def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
                with open(path, "rb") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        yield orjson.loads(line)
            """
        ),
    )

    write_text(
        root / "src" / spec.name / "data.py",
        _dedent(
            """
            from __future__ import annotations

            import random
            from dataclasses import dataclass
            from typing import Any, Dict, List, Optional, Tuple

            import torch
            from torch.utils.data import Dataset
            from transformers import PreTrainedTokenizerBase

            from .io_utils import read_jsonl


            @dataclass
            class PairSample:
                query_text: str
                pos_text: str
                neg_texts: List[str]  # pool


            def load_pair_samples(path: str) -> List[PairSample]:
                out: List[PairSample] = []
                for obj in read_jsonl(path):
                    q = str(obj["query_text"])
                    pos = str(obj["positive"]["text"])
                    negs = [str(x["text"]) for x in (obj.get("negatives") or [])]
                    if not negs:
                        # skip if no negatives (can't train pairwise)
                        continue
                    out.append(PairSample(query_text=q, pos_text=pos, neg_texts=negs))
                return out


            class PairwiseResampleDataset(Dataset):
                \"\"\"
                Each epoch, call .set_epoch(epoch) to change negative sampling seed.
                Returns one training item per query (or more if you want to upsample).
                Item: tokenized (q,pos) and (q,neg) pair.
                \"\"\"

                def __init__(
                    self,
                    samples: List[PairSample],
                    tokenizer: PreTrainedTokenizerBase,
                    max_length: int = 512,
                    negs_per_query_per_epoch: int = 1,
                    mix_random_negs: bool = True,
                    random_neg_ratio: float = 0.25,
                    seed: int = 42,
                ) -> None:
                    self.samples = samples
                    self.tok = tokenizer
                    self.max_length = max_length
                    self.negs_per_query_per_epoch = max(1, int(negs_per_query_per_epoch))
                    self.mix_random_negs = bool(mix_random_negs)
                    self.random_neg_ratio = float(random_neg_ratio)
                    self.base_seed = int(seed)
                    self.epoch = 0

                    # Pre-build index to allow stable length
                    self._index: List[Tuple[int, int]] = []
                    for i in range(len(self.samples)):
                        for j in range(self.negs_per_query_per_epoch):
                            self._index.append((i, j))

                def set_epoch(self, epoch: int) -> None:
                    self.epoch = int(epoch)

                def __len__(self) -> int:
                    return len(self._index)

                def _sample_neg(self, neg_pool: List[str], rng: random.Random) -> str:
                    if len(neg_pool) == 1:
                        return neg_pool[0]
                    # If you already have hard negatives in the pool, uniform sampling is OK.
                    # Optionally mix random negatives by sampling from the tail more often.
                    if self.mix_random_negs and rng.random() < self.random_neg_ratio:
                        # sample from the entire pool (random-ish)
                        return rng.choice(neg_pool)
                    # prefer harder: sample more from the front (if your pool is ranked hard->easy)
                    # Without rank info, just uniform:
                    return rng.choice(neg_pool)

                def __getitem__(self, idx: int) -> Dict[str, Any]:
                    qi, rep = self._index[idx]
                    s = self.samples[qi]

                    rng = random.Random(self.base_seed + self.epoch * 1000003 + qi * 97 + rep)
                    neg = self._sample_neg(s.neg_texts, rng)

                    # Encode (query, doc) for cross-encoder
                    pos_enc = self.tok(
                        s.query_text,
                        s.pos_text,
                        truncation=True,
                        max_length=self.max_length,
                        padding=False,
                        return_tensors=None,
                    )
                    neg_enc = self.tok(
                        s.query_text,
                        neg,
                        truncation=True,
                        max_length=self.max_length,
                        padding=False,
                        return_tensors=None,
                    )
                    return {
                        "pos": pos_enc,
                        "neg": neg_enc,
                    }


            def collate_pairwise(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
                \"\"\"
                Collate pos/neg pairs into padded tensors.
                Output:
                  input_ids_pos, attention_mask_pos, input_ids_neg, attention_mask_neg
                \"\"\"
                def _pad(key: str) -> Tuple[torch.Tensor, torch.Tensor]:
                    input_ids = [b[key]["input_ids"] for b in batch]
                    attn = [b[key]["attention_mask"] for b in batch]
                    maxlen = max(len(x) for x in input_ids)

                    ids_t = torch.full((len(batch), maxlen), pad_token_id, dtype=torch.long)
                    attn_t = torch.zeros((len(batch), maxlen), dtype=torch.long)
                    for i, (ids, am) in enumerate(zip(input_ids, attn)):
                        ids_t[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                        attn_t[i, : len(am)] = torch.tensor(am, dtype=torch.long)
                    return ids_t, attn_t

                ids_pos, attn_pos = _pad("pos")
                ids_neg, attn_neg = _pad("neg")
                return {
                    "input_ids_pos": ids_pos,
                    "attention_mask_pos": attn_pos,
                    "input_ids_neg": ids_neg,
                    "attention_mask_neg": attn_neg,
                }
            """
        ),
    )

    write_text(
        root / "src" / spec.name / "modeling.py",
        _dedent(
            """
            from __future__ import annotations

            from typing import Optional

            import torch
            import torch.nn as nn
            from transformers import AutoModelForSequenceClassification


            class CrossEncoderReranker(nn.Module):
                \"\"\"
                Wraps a (query, doc) cross-encoder returning a single relevance logit.
                Many reranker checkpoints already are sequence classification with 1 logit.
                \"\"\"

                def __init__(self, model_name: str) -> None:
                    super().__init__()
                    self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

                def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
                    out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = out.logits
                    if logits.dim() == 2 and logits.size(-1) == 1:
                        return logits.squeeze(-1)
                    if logits.dim() == 2 and logits.size(-1) > 1:
                        # If model outputs multiple labels, take the last logit as "relevance" fallback.
                        return logits[:, -1]
                    return logits
            """
        ),
    )

    write_text(
        root / "src" / spec.name / "train.py",
        _dedent(
            f"""
            from __future__ import annotations

            import argparse
            import os
            import random
            from pathlib import Path
            from typing import Any, Dict, Optional

            import numpy as np
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader
            from transformers import (
                AutoTokenizer,
                get_cosine_schedule_with_warmup,
            )
            from accelerate import Accelerator

            from .config import TrainConfig
            from .data import load_pair_samples, PairwiseResampleDataset, collate_pairwise
            from .modeling import CrossEncoderReranker


            def set_seed(seed: int) -> None:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)


            def maybe_apply_lora(model: nn.Module, cfg: TrainConfig) -> nn.Module:
                if not cfg.lora or not cfg.lora.enabled:
                    return model

                from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training

                if cfg.lora.qlora_4bit:
                    # QLoRA: model weights loaded in 4bit is typically done via bitsandbytes config.
                    # But sequence classification models vary; simplest robust path:
                    # - load normally (fp16/bf16)
                    # - prepare for k-bit training (works if quantized)
                    # If you want strict 4-bit loading, you can extend this scaffold later.
                    model = prepare_model_for_kbit_training(model)

                peft_cfg = PeftLoraConfig(
                    r=cfg.lora.r,
                    lora_alpha=cfg.lora.alpha,
                    lora_dropout=cfg.lora.dropout,
                    bias="none",
                    task_type="SEQ_CLS",
                    target_modules=cfg.lora.target_modules,
                )
                model = get_peft_model(model, peft_cfg)
                return model


            def pairwise_margin_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
                # want pos >= neg + margin
                return torch.clamp(margin - (pos_scores - neg_scores), min=0.0).mean()


            def save_model(acc: Accelerator, model: nn.Module, out_dir: str) -> None:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                unwrapped = acc.unwrap_model(model)
                # If using PEFT, save adapter; otherwise save full model
                try:
                    unwrapped.save_pretrained(out_dir)
                except Exception:
                    # fallback: state_dict
                    torch.save(unwrapped.state_dict(), os.path.join(out_dir, "pytorch_model.bin"))


            def main() -> None:
                ap = argparse.ArgumentParser()
                ap.add_argument("--config", required=True, help="Path to YAML config.")
                args = ap.parse_args()

                cfg = TrainConfig.load(args.config)
                set_seed(cfg.seed)

                accelerator = Accelerator(
                    gradient_accumulation_steps=cfg.grad_accum_steps,
                    mixed_precision="bf16" if cfg.bf16 else ("fp16" if cfg.fp16 else "no"),
                )

                tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
                if tokenizer.pad_token_id is None:
                    # some models don't define pad; fallback to eos
                    tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
                pad_id = int(tokenizer.pad_token_id)

                train_samples = load_pair_samples(cfg.train_path)
                valid_samples = load_pair_samples(cfg.valid_path)

                train_ds = PairwiseResampleDataset(
                    train_samples,
                    tokenizer=tokenizer,
                    max_length=cfg.max_length,
                    negs_per_query_per_epoch=cfg.negs_per_query_per_epoch,
                    mix_random_negs=cfg.mix_random_negs,
                    random_neg_ratio=cfg.random_neg_ratio,
                    seed=cfg.seed,
                )
                valid_ds = PairwiseResampleDataset(
                    valid_samples,
                    tokenizer=tokenizer,
                    max_length=cfg.max_length,
                    negs_per_query_per_epoch=1,
                    mix_random_negs=False,
                    random_neg_ratio=0.0,
                    seed=cfg.seed,
                )

                train_loader = DataLoader(
                    train_ds,
                    batch_size=cfg.per_device_train_batch_size,
                    shuffle=True,
                    num_workers=cfg.num_workers,
                    collate_fn=lambda b: collate_pairwise(b, pad_id),
                )
                valid_loader = DataLoader(
                    valid_ds,
                    batch_size=cfg.per_device_eval_batch_size,
                    shuffle=False,
                    num_workers=cfg.num_workers,
                    collate_fn=lambda b: collate_pairwise(b, pad_id),
                )

                model = CrossEncoderReranker(cfg.model_name)
                model = maybe_apply_lora(model, cfg)

                optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

                # total steps
                steps_per_epoch = max(1, len(train_loader) // max(1, cfg.grad_accum_steps))
                total_steps = steps_per_epoch * cfg.num_epochs
                if cfg.max_steps is not None:
                    total_steps = min(total_steps, int(cfg.max_steps))

                warmup_steps = int(total_steps * cfg.warmup_ratio)

                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps,
                )

                model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
                    model, optimizer, train_loader, valid_loader, scheduler
                )

                global_step = 0
                best_valid_loss: Optional[float] = None

                out_dir = cfg.output_dir
                Path(out_dir).mkdir(parents=True, exist_ok=True)

                for epoch in range(cfg.num_epochs):
                    train_ds.set_epoch(epoch)
                    model.train()

                    for batch in train_loader:
                        with accelerator.accumulate(model):
                            pos_scores = model(batch["input_ids_pos"], batch["attention_mask_pos"])
                            neg_scores = model(batch["input_ids_neg"], batch["attention_mask_neg"])

                            loss = pairwise_margin_loss(pos_scores, neg_scores, margin=1.0)
                            accelerator.backward(loss)

                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()

                        if accelerator.is_main_process and global_step % cfg.log_every_steps == 0:
                            accelerator.print(f"[train] step={global_step} loss={loss.item():.4f}")

                        if cfg.eval_every_steps and global_step > 0 and global_step % cfg.eval_every_steps == 0:
                            vloss = run_eval(accelerator, model, valid_loader)
                            accelerator.print(f"[valid] step={global_step} loss={vloss:.4f}")
                            if best_valid_loss is None or vloss < best_valid_loss:
                                best_valid_loss = vloss
                                accelerator.print(f"[valid] new best loss={best_valid_loss:.4f} -> saving")
                                save_model(accelerator, model, os.path.join(out_dir, "best"))

                        if cfg.save_every_steps and global_step > 0 and global_step % cfg.save_every_steps == 0:
                            save_model(accelerator, model, os.path.join(out_dir, f"step_{global_step}"))

                        global_step += 1
                        if cfg.max_steps is not None and global_step >= int(cfg.max_steps):
                            break

                    if cfg.max_steps is not None and global_step >= int(cfg.max_steps):
                        break

                if accelerator.is_main_process:
                    save_model(accelerator, model, os.path.join(out_dir, "last"))
                    accelerator.print("[done] saved last model")

                accelerator.end_training()


            @torch.no_grad()
            def run_eval(accelerator: Accelerator, model: nn.Module, loader: DataLoader) -> float:
                model.eval()
                losses = []
                for batch in loader:
                    pos_scores = model(batch["input_ids_pos"], batch["attention_mask_pos"])
                    neg_scores = model(batch["input_ids_neg"], batch["attention_mask_neg"])
                    loss = torch.clamp(1.0 - (pos_scores - neg_scores), min=0.0).mean()
                    losses.append(accelerator.gather(loss.detach()).float().cpu().numpy())
                if not losses:
                    return 0.0
                return float(np.mean(np.concatenate(losses)))


            if __name__ == "__main__":
                main()
            """
        ),
    )

    # scripts convenience entry
    write_text(
        root / "scripts" / "train.sh",
        _dedent(
            f"""
            #!/usr/bin/env bash
            set -euo pipefail
            python -m {spec.name}.train --config configs/train_qlora.yaml
            """
        ),
    )

    # tests placeholder
    write_text(
        root / "tests" / "test_smoke.py",
        _dedent(
            f"""
            def test_import() -> None:
                import {spec.name}  # noqa: F401
            """
        ),
    )

    # Example data templates
    write_text(
        root / "data" / "processed" / "README_DATA.md",
        _dedent(
            """
            Put your processed jsonl files here.

            Required:
            - train.jsonl
            - valid.jsonl

            Each line:
            {
              "query_text": "...",
              "positive": {"doc_id": "D1", "text": "..."},
              "negatives": [{"doc_id":"N1","text":"..."}, ...]
            }
            """
        ),
    )

    print(f"[OK] Project created at: {root}")
    print("")
    print("Next steps:")
    print(f"  1) cd {spec.name}")
    print("  2) python -m venv .venv && activate it")
    print('  3) pip install -e ".[dev]"   (recommended)')
    print("  4) Put data into data/processed/train.jsonl and valid.jsonl")
    print(f"  5) python -m {spec.name}.train --config configs/train_qlora.yaml")


if __name__ == "__main__":
    main()
