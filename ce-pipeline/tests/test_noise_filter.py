from __future__ import annotations

from ce_pipeline.processing.noise_filter import is_noise_chunk


def test_filter_drops_empty_or_too_short():
    assert is_noise_chunk("") is True
    assert is_noise_chunk("short text", min_chunk_chars=200) is True


def test_filter_drops_separator_dense_nav_when_short():
    t = "Home | News | Sports | Politics | Economy"
    assert is_noise_chunk(t, min_chunk_chars=10) is True


def test_filter_keeps_long_chunk_with_weak_noise():
    # Long chunk with a weak footer phrase should not be dropped.
    long_text = ("This is a long chunk with meaningful content. " * 80) + "\n\nAll rights reserved."
    assert is_noise_chunk(long_text, min_chunk_chars=200) is False


def test_filter_drops_short_chunk_with_weak_noise():
    assert is_noise_chunk("Privacy Policy", min_chunk_chars=10) is True


def test_filter_drops_strong_noise_unless_long():
    assert is_noise_chunk("Accept cookies. Cookie settings.", min_chunk_chars=10) is True

    long_with_strong = ("Meaningful content " * 120) + "\n\nAccept cookies"
    # Still long enough: we keep (because strong noise might be contamination at edge trimmed stage,
    # and this function is conservative for long chunks).
    assert is_noise_chunk(long_with_strong, min_chunk_chars=200) is False
