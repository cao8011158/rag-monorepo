# tests/test_html_to_text.py
from __future__ import annotations

import json
import pytest


from cc_pipeline.clean.html_cleaner import html_to_text


@pytest.fixture
def sample_html() -> str:
    # 含 title + nav/header/footer/script/style + 正文
    return """
    <html>
      <head>
        <title>  My Page Title  </title>
        <style>.x{color:red}</style>
        <script>console.log("hi")</script>
      </head>
      <body>
        <header>Header Nav</header>
        <nav>Home | About | Contact</nav>
        <aside>Sidebar stuff</aside>

        <main>
          <h1>Heading</h1>
          <p>First paragraph.</p>
          <p>Second paragraph.</p>
        </main>

        <footer>© 2026 All rights reserved</footer>
      </body>
    </html>
    """


def test_trafilatura_success_path(monkeypatch, sample_html: str) -> None:
    """
    trafilatura returns JSON with long-enough text => return its title/text
    and normalize to non-empty stripped lines joined by '\n'.
    """
    # Build a long-enough text with messy whitespace/newlines
    raw_text = " Line1  \n\n  Line2 \n   \nLine3 "
    # Make it >= 200 chars
    raw_text = raw_text + (" x" * 120)

    payload = {
        "title": " Trafilatura Title ",
        "text": raw_text,
        "date": "2026-01-01",
        "author": "Someone",
    }
    extracted_json = json.dumps(payload)

    import cc_pipeline.clean.html_cleaner as m

    def fake_extract(*args, **kwargs):
        return extracted_json

    monkeypatch.setattr(m.trafilatura, "extract", fake_extract)

    title, text = html_to_text(sample_html)
    assert title == "Trafilatura Title"
    # lines normalized
    assert text.startswith("Line1\nLine2\nLine3")
    assert "\n\n" not in text


def test_trafilatura_returns_none_falls_back(monkeypatch, sample_html: str) -> None:
    """
    trafilatura returns None => fallback to bs4 extraction.
    """
    import cc_pipeline.clean.html_cleaner as m

    monkeypatch.setattr(m.trafilatura, "extract", lambda *a, **k: None)

    title, text = html_to_text(sample_html)
    assert title == "My Page Title"
    # fallback removes nav/header/footer/script/style/aside
    assert "Home | About | Contact" not in text
    assert "Header Nav" not in text
    assert "All rights reserved" not in text
    assert "console.log" not in text
    assert ".x{color:red}" not in text
    assert "Sidebar stuff" not in text

    #正文仍在
    assert "First paragraph." in text
    assert "Second paragraph." in text


def test_trafilatura_bad_json_falls_back(monkeypatch, sample_html: str) -> None:
    """
    trafilatura returns invalid JSON => fallback to bs4 extraction.
    """
    import cc_pipeline.clean.html_cleaner as m

    monkeypatch.setattr(m.trafilatura, "extract", lambda *a, **k: "{not valid json")

    title, text = html_to_text(sample_html)
    assert title == "My Page Title"
    assert "First paragraph." in text


def test_trafilatura_short_text_falls_back(monkeypatch, sample_html: str) -> None:
    """
    trafilatura returns JSON but text is too short (<200) => fallback.
    """
    import cc_pipeline.clean.html_cleaner as m

    payload = {"title": "Trafilatura Title", "text": "Too short."}
    monkeypatch.setattr(m.trafilatura, "extract", lambda *a, **k: json.dumps(payload))

    title, text = html_to_text(sample_html)
    # Expect fallback title/text (not trafilatura's short output)
    assert title == "My Page Title"
    assert "Too short." not in text  # fallback doesn't contain that phrase
    assert "First paragraph." in text


def test_trafilatura_missing_fields_falls_back(monkeypatch, sample_html: str) -> None:
    """
    trafilatura returns JSON without text => fallback.
    """
    import cc_pipeline.clean.html_cleaner as m

    payload = {"title": "Trafilatura Title"}  # no "text"
    monkeypatch.setattr(m.trafilatura, "extract", lambda *a, **k: json.dumps(payload))

    title, text = html_to_text(sample_html)
    assert title == "My Page Title"
    assert "First paragraph." in text
