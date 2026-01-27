# tests/test_hf_transformers_llm.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pytest
import torch

from qr_pipeline.llm.hf_transformers_llm import HFTransformersLLM


# -----------------------------
# Fakes
# -----------------------------

class FakeTokenizer:
    def __init__(self) -> None:
        self.eos_token_id = 2
        self.last_prompt: Optional[str] = None

    def __call__(self, prompt: str, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        self.last_prompt = prompt
        # 固定 prompt token 长度=5
        return {"input_ids": torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long)}

    def decode(self, ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        assert skip_special_tokens is True
        if ids.numel() == 0:
            return ""
        # 将 token id 转为字符串，便于断言是否只 decode 了新生成部分
        return " ".join(str(int(x)) for x in ids.tolist())


class FakeModel:
    """
    关键点：为了让测试在 CPU-only 环境也稳定：
    - .to("cuda") 只做记录，不真的把参数/张量搬到 CUDA
    - parameters() 返回的参数永远在 CPU，这样 generate() 时 dev=cpu，不会触发 inputs.to(cuda)
    """
    def __init__(self) -> None:
        self.eval_called = False
        self.last_generate_kwargs: Dict[str, Any] = {}
        self.to_calls = []  # 记录 .to(...) 被调用的 device

        # 这个参数永远留在 CPU，确保 next(model.parameters()).device == cpu
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        yield self._param

    def to(self, device: str):
        self.to_calls.append(device)
        return self

    def eval(self):
        self.eval_called = True
        return self

    def generate(self, **kwargs):
        self.last_generate_kwargs = dict(kwargs)
        input_ids: torch.Tensor = kwargs["input_ids"]
        assert input_ids.ndim == 2 and input_ids.shape[0] == 1

        # 输出：prompt(5 tokens) + 新生成 3 tokens (99,98,97)
        appended = torch.tensor([[99, 98, 97]], dtype=torch.long, device=input_ids.device)
        out = torch.cat([input_ids, appended], dim=1)
        return out


@dataclass
class FakeHF:
    tok: FakeTokenizer
    model: FakeModel
    tok_calls: list
    model_calls: list


def _get_module_under_test():
    # HFTransformersLLM 定义所在模块
    modname = HFTransformersLLM.__module__
    return __import__(modname, fromlist=["*"])


def _install_fakes(monkeypatch: pytest.MonkeyPatch) -> FakeHF:
    """
    monkeypatch 被测模块命名空间里的 AutoTokenizer/AutoModelForCausalLM
    """
    mod = _get_module_under_test()

    tok = FakeTokenizer()
    model = FakeModel()
    tok_calls = []
    model_calls = []

    def fake_tok_from_pretrained(model_name: str, **kwargs):
        tok_calls.append((model_name, dict(kwargs)))
        return tok

    def fake_model_from_pretrained(model_name: str, **kwargs):
        model_calls.append((model_name, dict(kwargs)))
        return model

    monkeypatch.setattr(mod.AutoTokenizer, "from_pretrained", fake_tok_from_pretrained)
    monkeypatch.setattr(mod.AutoModelForCausalLM, "from_pretrained", fake_model_from_pretrained)

    return FakeHF(tok=tok, model=model, tok_calls=tok_calls, model_calls=model_calls)


# -----------------------------
# Tests
# -----------------------------

def test_load_cpu_passes_cache_dir_and_moves_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fakes(monkeypatch)

    # 强制无 GPU
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    llm = HFTransformersLLM(
        model_name="some-model",
        device="cpu",
        cache_dir="/tmp/hf_cache",
    )
    llm.load()

    assert llm._tok is fake.tok
    assert llm._model is fake.model
    assert fake.model.eval_called is True

    # tokenizer / model 都应收到 cache_dir
    assert fake.tok_calls == [("some-model", {"cache_dir": "/tmp/hf_cache"})]
    assert fake.model_calls == [("some-model", {"cache_dir": "/tmp/hf_cache"})]

    # load() 最终应调用 .to("cpu")
    assert fake.model.to_calls[-1] == "cpu"


def test_load_cuda_sets_fp16_and_calls_to_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fakes(monkeypatch)

    # 伪装“有 GPU”
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    llm = HFTransformersLLM(model_name="some-model", device="cuda")
    llm.load()

    # from_pretrained 应带 torch_dtype=float16
    assert len(fake.model_calls) == 1
    _mn, kwargs = fake.model_calls[0]
    assert kwargs.get("torch_dtype") == torch.float16

    # 应调用 .to("cuda")
    assert "cuda" in fake.model.to_calls


def test_load_cuda_falls_back_to_cpu_when_not_available(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fakes(monkeypatch)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    llm = HFTransformersLLM(model_name="some-model", device="cuda")
    llm.load()

    # 不应设置 torch_dtype=float16（你的实现：cuda 且可用才设）
    assert len(fake.model_calls) == 1
    _mn, kwargs = fake.model_calls[0]
    assert "torch_dtype" not in kwargs

    # 应回落到 .to("cpu")
    assert fake.model.to_calls[-1] == "cpu"


def test_generate_lazy_load_and_decodes_only_new_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fakes(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    llm = HFTransformersLLM(model_name="some-model", device="cpu")

    # 尚未 load
    assert llm._tok is None
    assert llm._model is None

    out = llm.generate("Hello prompt")

    # 会触发 lazy load
    assert len(fake.tok_calls) == 1
    assert len(fake.model_calls) == 1

    # FakeModel.generate: prompt_ids(5个) + [99,98,97]
    # 被测代码切掉 prompt 部分，只 decode 新生成 => "99 98 97"
    assert out == "99 98 97"


def test_generate_sampling_flags_when_temperature_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fakes(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    llm = HFTransformersLLM(
        model_name="some-model",
        device="cpu",
        max_new_tokens=55,
        temperature=0.7,
        top_p=0.9,
    )

    _ = llm.generate("x")

    kwargs = fake.model.last_generate_kwargs
    assert kwargs["max_new_tokens"] == 55
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == pytest.approx(0.7)
    assert kwargs["top_p"] == pytest.approx(0.9)
    assert kwargs["pad_token_id"] == fake.tok.eos_token_id
    assert kwargs["eos_token_id"] == fake.tok.eos_token_id


def test_generate_greedy_when_temperature_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fakes(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    llm = HFTransformersLLM(
        model_name="some-model",
        device="cpu",
        temperature=0.0,  # => do_sample False
        top_p=0.9,
    )

    _ = llm.generate("x")

    kwargs = fake.model.last_generate_kwargs
    assert kwargs["do_sample"] is False

    # 你的实现：do_sample=False 时 temperature/top_p 传 None
    assert kwargs["temperature"] is None
    assert kwargs["top_p"] is None


def test_generate_uses_configured_max_new_tokens_int_cast(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fakes(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    llm = HFTransformersLLM(
        model_name="some-model",
        device="cpu",
        max_new_tokens=128,
    )

    _ = llm.generate("x")
    assert fake.model.last_generate_kwargs["max_new_tokens"] == 128
