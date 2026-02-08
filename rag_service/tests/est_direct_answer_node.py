# tests/test_direct_answer_node.py
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# 改成你真实的模块路径
from rag_service.lcel.direct_answer_node import create_direct_answer_runnable


@pytest.fixture
def settings():
    # 你给的 settings（最小可用）
    return {
        "models": {
            "gemini_api": {
                "model_name": "gemini-3-flash-preview",
            }
        }
    }


def _install_genai_client_mock(monkeypatch, module_path: str, resp_text: str = "hello"):
    """
    Monkeypatch module's `genai.Client` so no real network call happens.
    Returns (client_mock, generate_content_mock) for assertions.
    """
    # importlib 用于按字符串导入模块对象
    import importlib

    mod = importlib.import_module(module_path)

    client_mock = MagicMock(name="GenAIClientMock")
    generate_content_mock = MagicMock(name="generate_content")
    client_mock.models.generate_content = generate_content_mock

    # Gemini SDK 的返回对象一般有 `.text`
    generate_content_mock.return_value = SimpleNamespace(text=resp_text)

    # 关键：把 mod.genai.Client 替换成一个返回 client_mock 的 callable
    monkeypatch.setattr(mod.genai, "Client", MagicMock(return_value=client_mock))

    return client_mock, generate_content_mock


def test_empty_query_returns_empty_answer_and_no_llm_call(monkeypatch, settings):
    # 模块路径要与 import 对应
    module_path = "rag_service.lcel.direct_answer_node"
    client_mock, generate_content_mock = _install_genai_client_mock(
        monkeypatch, module_path, resp_text="SHOULD_NOT_BE_USED"
    )

    runnable = create_direct_answer_runnable(settings)

    # 1) 空字符串
    out = runnable.invoke("")
    assert out == {"answer": "", "mode": "direct"}
    generate_content_mock.assert_not_called()

    # 2) None
    out = runnable.invoke(None)
    assert out == {"answer": "", "mode": "direct"}
    generate_content_mock.assert_not_called()

    # 3) 只有空白
    out = runnable.invoke("   \n\t  ")
    assert out == {"answer": "", "mode": "direct"}
    generate_content_mock.assert_not_called()

    # client 仍然会在 create_direct_answer_runnable 时被构造（你的代码就是这样写的）
    # 所以这里只检查没有调用 generate_content
    assert client_mock is not None


def test_non_empty_query_calls_llm_and_returns_stripped_text(monkeypatch, settings):
    module_path = "rag_service.lcel.direct_answer_node"
    client_mock, generate_content_mock = _install_genai_client_mock(
        monkeypatch, module_path, resp_text="  hello world  \n"
    )

    runnable = create_direct_answer_runnable(settings)

    out = runnable.invoke("  What is CMU?  ")
    assert out == {"answer": "hello world", "mode": "direct"}

    generate_content_mock.assert_called_once()

    # 检查调用参数：model、contents、config.system_instruction
    _, kwargs = generate_content_mock.call_args
    assert kwargs["model"] == "gemini-3-flash-preview"
    assert kwargs["contents"] == "What is CMU?"  # 说明 query 先 strip 了

    cfg = kwargs["config"]
    assert isinstance(cfg, dict)
    assert "system_instruction" in cfg
    assert "Answer directly using general knowledge" in cfg["system_instruction"]


def test_llm_returns_none_text_should_fallback_to_empty(monkeypatch, settings):
    module_path = "rag_service.lcel.direct_answer_node"
    client_mock, generate_content_mock = _install_genai_client_mock(
        monkeypatch, module_path, resp_text=None
    )

    runnable = create_direct_answer_runnable(settings)

    out = runnable.invoke("hi")
    assert out == {"answer": "", "mode": "direct"}
    generate_content_mock.assert_called_once()


def test_missing_model_name_raises_keyerror(settings):
    bad = {"models": {"gemini_api": {}}}  # 缺 model_name
    with pytest.raises(KeyError) as e:
        create_direct_answer_runnable(bad)
    assert "models.gemini_api.model_name" in str(e.value)
