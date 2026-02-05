import types
import pytest

import reranker_training.train_reranker as tr


# -------------------------
# Fakes
# -------------------------
class DummyTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "<eos>"

    def pad(self, *args, **kwargs):
        raise RuntimeError("not used in this test")


class DummyBaseModel:
    """Stand-in for HF AutoModelForSequenceClassification."""
    def __init__(self):
        # we will pretend there are some params
        self._params = [types.SimpleNamespace(requires_grad=True, numel=lambda: 10)]

    def parameters(self):
        return list(self._params)


class DummyTrainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.train_called = False
        self.save_called = False
        self.saved_dir = None

    def train(self):
        self.train_called = True

    def save_model(self, output_dir):
        self.save_called = True
        self.saved_dir = output_dir


class DummyDataset:
    def __init__(self, items, tokenizer, max_length):
        self.items = list(items)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)


def make_settings_dict():
    # 对齐你贴的 config（只保留 train_reranker.py 用到的字段）
    return {
        "model_name": "BAAI/bge-reranker-v2-m3",
        "stores": {
            "fs_local": {"kind": "filesystem", "root": "/content/drive/MyDrive/rag-kb-data"}
        },
        "outputs": {
            "files": {
                "store": "fs_local",
                "base": "reranker_out",
                "train_path": "processed/train_query_pack.jsonl",
                "valid_path": "processed/valid_query_pack.jsonl",
                "train_pair_path": "processed/train_pair_epoch_{epoch}.jsonl",
            }
        },
        "max_length": 512,
        "training": {
            "output_dir": "/content/drive/MyDrive/rag-kb-data/run1",
            "seed": 42,
            "num_epochs": 3,
            "lr": 2.0e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.05,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 20,
            "grad_accum_steps": 2,
            "log_every_steps": 20,
            "eval_every_steps": 200,
            "save_every_steps": 200,
            "max_steps": None,
            "num_workers": 2,
        },
        "eval": {
            "ndcg_k": 10,
            "mrr_k": 10,
            "infer_batch_size": 32,
            "max_negatives_per_query": 50,
        },
        "lora": {
            "enabled": True,
            "qlora_4bit": False,
            "r": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": ["query", "key", "value", "dense"],
        },
        "bf16": True,
        "fp16": False,
    }


def test_posix_join():
    assert tr._posix_join("a/b/", "/c") == "a/b/c"
    assert tr._posix_join("a", "c") == "a/c"
    assert tr._posix_join("", "x") == "x"


def test_main_happy_path(monkeypatch, capsys):
    # ---- argparse ----
    monkeypatch.setattr(
        tr.argparse.ArgumentParser,
        "parse_args",
        lambda self: types.SimpleNamespace(config="configs/train.yaml"),
    )

    # ---- settings + stores ----
    s = make_settings_dict()
    monkeypatch.setattr(tr, "load_settings", lambda p: s)
    monkeypatch.setattr(tr, "build_store_registry", lambda cfg: {"fs_local": object()})

    # ---- data loaders ----
    # epoch0 pairs
    ep0 = [{"dummy": "pair0"}]
    monkeypatch.setattr(tr, "load_pairs_for_epoch", lambda **kwargs: list(ep0))

    # valid packs (dict 也行，只要你的 load_valid_query_packs 返回的类型能被 trainer 使用)
    valid_packs = [{"query_text": "q", "doc_texts": ["d1", "d2"], "labels": [1, 0]}]
    monkeypatch.setattr(tr, "load_valid_query_packs", lambda **kwargs: list(valid_packs))

    # ---- tokenizer / model ----
    monkeypatch.setattr(tr.AutoTokenizer, "from_pretrained", lambda *a, **k: DummyTokenizer())
    monkeypatch.setattr(tr.AutoModelForSequenceClassification, "from_pretrained", lambda *a, **k: DummyBaseModel())

    # ---- LoRA / wrapper ----
    # 不测 peft 真注入：直接返回 base_model（但要确保有 requires_grad 参数，否则你的 trainable 断言会炸）
    monkeypatch.setattr(tr, "_apply_lora_once", lambda base_model, **k: base_model)
    monkeypatch.setattr(tr, "CrossEncoderReranker", lambda m: m)

    # ---- dataset/collator/trainer ----
    monkeypatch.setattr(tr, "CrossEncoderPairwiseDataset", DummyDataset)
    monkeypatch.setattr(tr, "PairwiseCollator", lambda tok: object())

    created = {}
    def _mk_trainer(**kwargs):
        t = DummyTrainer(**kwargs)
        created["trainer"] = t
        return t

    monkeypatch.setattr(tr, "PairwiseTrainerWithRankingEval", _mk_trainer)

    # ---- run ----
    tr.main()

    out = capsys.readouterr().out
    assert "[DATA] train epoch=0 pairs=1" in out
    assert "[DATA] valid query_packs=1" in out
    assert "[OK] Training done." in out

    # trainer train/save called
    assert created["trainer"].train_called is True
    assert created["trainer"].save_called is True
    assert created["trainer"].saved_dir == s["training"]["output_dir"]

    # tokenizer pad_token filled
    tok = created["trainer"].kwargs["tokenizer"]
    assert tok.pad_token == "<eos>"


def test_lora_target_modules_validation(monkeypatch):
    monkeypatch.setattr(
        tr.argparse.ArgumentParser,
        "parse_args",
        lambda self: types.SimpleNamespace(config="configs/train.yaml"),
    )

    s = make_settings_dict()
    s["lora"]["target_modules"] = "not-a-list"  # invalid

    monkeypatch.setattr(tr, "load_settings", lambda p: s)
    monkeypatch.setattr(tr, "build_store_registry", lambda cfg: {"fs_local": object()})

    monkeypatch.setattr(tr, "load_pairs_for_epoch", lambda **kwargs: [{"dummy": 1}])
    monkeypatch.setattr(tr, "load_valid_query_packs", lambda **kwargs: [])

    monkeypatch.setattr(tr.AutoTokenizer, "from_pretrained", lambda *a, **k: DummyTokenizer())
    monkeypatch.setattr(tr.AutoModelForSequenceClassification, "from_pretrained", lambda *a, **k: DummyBaseModel())

    with pytest.raises(ValueError, match="lora.target_modules"):
        tr.main()


def test_training_arguments_mapping(monkeypatch):
    monkeypatch.setattr(
        tr.argparse.ArgumentParser,
        "parse_args",
        lambda self: types.SimpleNamespace(config="configs/train.yaml"),
    )

    s = make_settings_dict()
    monkeypatch.setattr(tr, "load_settings", lambda p: s)
    monkeypatch.setattr(tr, "build_store_registry", lambda cfg: {"fs_local": object()})

    monkeypatch.setattr(tr, "load_pairs_for_epoch", lambda **kwargs: [{"dummy": 1}])
    monkeypatch.setattr(tr, "load_valid_query_packs", lambda **kwargs: [])

    monkeypatch.setattr(tr.AutoTokenizer, "from_pretrained", lambda *a, **k: DummyTokenizer())
    monkeypatch.setattr(tr.AutoModelForSequenceClassification, "from_pretrained", lambda *a, **k: DummyBaseModel())
    monkeypatch.setattr(tr, "_apply_lora_once", lambda base_model, **k: base_model)
    monkeypatch.setattr(tr, "CrossEncoderReranker", lambda m: m)
    monkeypatch.setattr(tr, "CrossEncoderPairwiseDataset", DummyDataset)
    monkeypatch.setattr(tr, "PairwiseCollator", lambda tok: object())

    created = {}
    monkeypatch.setattr(tr, "PairwiseTrainerWithRankingEval", lambda **kwargs: created.setdefault("trainer", DummyTrainer(**kwargs)))

    tr.main()

    args_tr = created["trainer"].kwargs["args"]
    tr_cfg = s["training"]

    assert args_tr.per_device_train_batch_size == tr_cfg["per_device_train_batch_size"]
    assert args_tr.per_device_eval_batch_size == tr_cfg["per_device_eval_batch_size"]
    assert args_tr.gradient_accumulation_steps == tr_cfg["grad_accum_steps"]
    assert args_tr.logging_steps == tr_cfg["log_every_steps"]
    assert args_tr.eval_steps == tr_cfg["eval_every_steps"]
    assert args_tr.save_steps == tr_cfg["save_every_steps"]
    assert args_tr.dataloader_num_workers == tr_cfg["num_workers"]

    assert args_tr.bf16 is True
    assert args_tr.fp16 is False

    assert args_tr.evaluation_strategy == "steps"
    assert args_tr.save_strategy == "steps"
