# tests/test_parse_in_out_queries.py
from __future__ import annotations

import pytest

from qr_pipeline.pipeline.query_generation import _parse_in_out_queries


@pytest.mark.parametrize(
    "raw, expected_in, expected_out",
    [
        # 1) 基本：严格格式
        (
            "IN:\nA?\nB?\nOUT:\nC?\n",
            ["A?", "B?"],
            ["C?"],
        ),
        # 2) 大小写 + 额外空白
        (
            "  in:  \n  A?  \n\n OuT:\n  C?  \n",
            ["A?"],
            ["C?"],
        ),
        # 3) 带 bullet / numbering（应该被去掉）
        (
            "IN:\n- What is CMU?\n1) When was CMU founded?\nOUT:\n• How to bake bread?\n2: Buy a car?\n",
            ["What is CMU?", "When was CMU founded?"],
            ["How to bake bread?", "Buy a car?"],
        ),
        # 4) header 之前有废话（应该忽略）
        (
            "Sure, here you go:\nSome explanation...\nIN:\nA?\nOUT:\nC?\n",
            ["A?"],
            ["C?"],
        ),
        # 5) 只有 IN 段（OUT 缺失）
        (
            "IN:\nA?\nB?\n",
            ["A?", "B?"],
            [],
        ),
        # 6) 只有 OUT 段（IN 缺失）
        (
            "OUT:\nC?\nD?\n",
            [],
            ["C?", "D?"],
        ),
        # 7) 空输入
        (
            "   \n\n",
            [],
            [],
        ),
        # 8) 只有 header 没内容
        (
            "IN:\n\nOUT:\n\n",
            [],
            [],
        ),
        # 9) header 后出现空行和杂质行（空行应跳过；杂质行会按普通 query 处理）
        (
            "IN:\n\n  \n- A?\nOUT:\n\n• C?\n",
            ["A?"],
            ["C?"],
        ),
    ],
)
def test_parse_in_out_queries(raw: str, expected_in: list[str], expected_out: list[str]) -> None:
    in_qs, out_qs = _parse_in_out_queries(raw)
    assert in_qs == expected_in
    assert out_qs == expected_out


def test_parse_in_out_queries_ignores_text_without_headers() -> None:
    raw = "No headers here.\nJust text.\n- A?\n"
    in_qs, out_qs = _parse_in_out_queries(raw)
    assert in_qs == []
    assert out_qs == []
