from __future__ import annotations

from cc_pipeline.crawl.link_extractor import extract_links


def test_extract_links_returns_absolute_urls():
    html = """
    <html><body>
      <a href="/a">A</a>
      <a href="b">B</a>
      <a href="https://other.com/x">X</a>
      <a>No href</a>
    </body></html>
    """
    base = "https://example.com/dir/page.html"
    links = extract_links(html, base_url=base)

    assert "https://example.com/a" in links
    assert "https://example.com/dir/b" in links
    assert "https://other.com/x" in links