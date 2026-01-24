from __future__ import annotations

from ce_pipeline.processing.repair import (
    _looks_truncated_end,
    _looks_truncated_start,
    repair_boundary_truncation,
)


# -------------------------
# Heuristic unit tests
# -------------------------

def test_looks_truncated_end_false_when_has_terminal_punct() -> None:
    assert _looks_truncated_end("Hello world.") is False
    assert _looks_truncated_end("Hello world!") is False
    assert _looks_truncated_end("Hello world?") is False
    assert _looks_truncated_end("你好世界。") is False
    assert _looks_truncated_end("真的吗？   ") is False
    assert _looks_truncated_end("这是省略号……") is False


def test_looks_truncated_end_true_when_ends_with_dash() -> None:
    assert _looks_truncated_end("This is truncated -") is True
    assert _looks_truncated_end("This is truncated —  ") is True
    assert _looks_truncated_end("This is truncated –") is True


def test_looks_truncated_end_true_when_ends_with_delimiters() -> None:
    assert _looks_truncated_end("This is truncated,") is True
    assert _looks_truncated_end("This is truncated:") is True
    assert _looks_truncated_end("This is truncated;") is True


def test_looks_truncated_end_true_when_long_and_no_terminal_punct() -> None:
    t = "This is a fairly long chunk that does not end with a proper punctuation mark and keeps going"
    assert len(t) > 60
    assert _looks_truncated_end(t) is True


def test_looks_truncated_end_false_when_empty_or_whitespace() -> None:
    assert _looks_truncated_end("") is False
    assert _looks_truncated_end("   \n\t") is False


def test_looks_truncated_start_true_when_starts_with_lowercase() -> None:
    assert _looks_truncated_start("continuation of a sentence") is True
    assert _looks_truncated_start("a") is True


def test_looks_truncated_start_true_when_starts_with_continuation_punct() -> None:
    assert _looks_truncated_start(", and then it continues") is True
    assert _looks_truncated_start("; and then it continues") is True
    assert _looks_truncated_start(": and then it continues") is True
    assert _looks_truncated_start(") and then it continues") is True
    assert _looks_truncated_start("] and then it continues") is True
    assert _looks_truncated_start("} and then it continues") is True


def test_looks_truncated_start_false_when_starts_normally() -> None:
    assert _looks_truncated_start("This starts a sentence.") is False
    assert _looks_truncated_start("A proper start") is False
    assert _looks_truncated_start("你好世界。") is False


# -------------------------
# Repair function tests
# -------------------------

def test_repair_no_change_when_cur_is_complete_sentence() -> None:
    prev = "Prev chunk content that should not be used."
    cur = "This is a complete sentence."
    nxt = "Next chunk content that should not be used."
    out = repair_boundary_truncation(prev, cur, nxt)
    assert out == "This is a complete sentence."


def test_repair_returns_stripped_when_cur_empty() -> None:
    assert repair_boundary_truncation("prev", "   ", "next") == ""
    assert repair_boundary_truncation(None, "\n\t", None) == ""


def test_repair_prepends_prev_tail_when_cur_looks_truncated_start() -> None:
    prev = "This is the previous chunk ending with a partial sentence"
    cur = "continuation that starts with lowercase and should be repaired."
    out = repair_boundary_truncation(prev, cur, None)

    # 应该包含 prev 的一部分 + cur
    assert "previous chunk" in out
    assert out.endswith("repaired.")


def test_repair_appends_next_head_when_cur_looks_truncated_end() -> None:
    cur = "This is a chunk that seems truncated at the end,"
    nxt = "and here is the continuation in the next chunk. Finally ends."
    out = repair_boundary_truncation(None, cur, nxt)

    assert out.startswith("This is a chunk")
    assert "and here is the continuation" in out


def test_repair_both_sides_when_start_and_end_truncated() -> None:
    prev = "We introduce a method that can scale to large corpora and"
    cur = "improves retrieval quality significantly,"
    nxt = "especially when combined with reranking. This is the end."
    out = repair_boundary_truncation(prev, cur, nxt)

    assert "scale to large corpora" in out
    assert "combined with reranking" in out


def test_repair_limits_prev_tail_to_300_chars() -> None:
    prev = "A" * 500 + " TAIL"
    cur = "continuation"
    out = repair_boundary_truncation(prev, cur, None)

    assert out.endswith("continuation")
    # prev tail should be bounded
    assert out.count("A") <= 305


def test_repair_limits_next_head_to_300_chars() -> None:
    cur = "This is truncated at end:"
    nxt = "B" * 500 + " HEAD"
    out = repair_boundary_truncation(None, cur, nxt)

    assert out.startswith("This is truncated at end:")
    # next head should be bounded
    assert out.count("B") <= 305