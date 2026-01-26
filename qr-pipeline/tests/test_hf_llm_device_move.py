# tests/test_hf_llm_device_move.py
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict


# --- ensure src/ is importable (src-layout project) ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class FakeTensor:
    def __init__(self) -> None:
        self.moved_to: str | None = None

    def to(self, device: Any) -> "FakeTensor":
        self.moved_to = str(device)
        return self


class FakeTokenizer:
    eos_token_id = 0

    def __call__(self, prompt: str, return_tensors: str = "pt") -> Dict[str, FakeTensor]:
        # minimal set a CausalLM expects
        return {
            "input_ids": FakeTensor(),
            "attention_mask": FakeTensor(),
        }

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        # generate() 会把 prompt 前缀剥掉，所以这里必须以 prompt 开头
        return "PROMPT" + " GENERATED"


class FakeModel:
    def __init__(self, device: str) -> None:
        self.device = device

    def eval(self) -> None:
        return None

    def generate(self, **kwargs: Any) -> Any:
        # 只检查 tokenizer 产生的 tensor 是否被搬到 model.device
        for k in ("input_ids", "attention_mask", "token_type_ids", "position_ids"):
            if k in kwargs:
                v = kwargs[k]
                assert hasattr(v, "moved_to"), f"{k} is not a FakeTensor"
                assert v.moved_to == str(self.device), f"{k} not moved to model.device"
        # 返回一个能被 tokenizer.decode 接受的“假输出”
        return [[1, 2, 3]]


def test_generate_always_moves_inputs_to_model_device(monkeypatch) -> None:
    # 1) 注入一个假的 torch，满足 `import torch` 和 `torch.no_grad()`
    @contextmanager
    def _no_grad():
        yield

    class FakeTorch:
        no_grad = staticmethod(_no_grad)

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())

    # 2) 导入你的类（注意：你的文件在 src/qr_pipeline/pipeline/query_generation.py）
    from qr_pipeline.pipeline.query_generation import HFTransformersLLM

    # 3) 不走 __post_init__（避免加载 transformers），手动构造对象
    llm = HFTransformersLLM.__new__(HFTransformersLLM)
    llm._tok = FakeTokenizer()
    llm._model = FakeModel(device="cuda:0")  # 假设模型实际在 GPU
    llm.device = "cuda"  # 关键：以前 bug 在 cuda 时不会搬
    llm.max_new_tokens = 8
    llm.temperature = 0.7
    llm.top_p = 0.9

    # 4) 调用 generate：如果 inputs 没搬到 model.device，这里会断言失败
    out = llm.generate("PROMPT")
    assert out == "GENERATED"
