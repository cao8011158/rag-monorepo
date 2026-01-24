from __future__ import annotations

from ce_pipeline.processing.noise_trim import trim_noise_edges


def test_trim_removes_footer_cookie_tail():
    raw = (
        "This is the useful end paragraph of the article. It contains meaningful information.\n\n"
        "Related articles | Privacy Policy | Terms of Service\n"
        "Accept cookies\n"
        "Cookie settings\n"
    )
    trimmed = trim_noise_edges(raw)
    assert "useful end paragraph" in trimmed.lower()
    assert "accept cookies" not in trimmed.lower()
    assert "cookie settings" not in trimmed.lower()
    assert "related articles" not in trimmed.lower()


def test_trim_removes_cookie_banner_head():
    raw = (
        "Accept cookies\n"
        "Cookie settings\n\n"
        "Here starts the real content of the page. It is useful and long enough.\n"
    )
    trimmed = trim_noise_edges(raw)
    assert "accept cookies" not in trimmed.lower()
    assert "cookie settings" not in trimmed.lower()
    assert "real content" in trimmed.lower()


def test_trim_keeps_mid_chunk_mentions_conservatively():
    # Mentioning "privacy policy" inside content should not be removed unless it appears as an edge part.
    raw = (
        "This article discusses why a privacy policy matters for users and websites. "
        "It provides historical context and analysis. " * 10
    )
    trimmed = trim_noise_edges(raw)
    assert trimmed == raw.strip()
