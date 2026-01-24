from __future__ import annotations

from cc_pipeline.crawl.url_utils import (
    canonicalize_url,
    is_http_url,
    should_drop,
    absolutize,
)


def test_canonicalize_drops_fragment_and_tracking_params():
    u = "HTTPS://En.Wikipedia.Org/wiki/Pittsburgh/?utm_source=google&gclid=123#History"
    cu = canonicalize_url(u)
    assert "#history" not in cu.lower()
    assert "utm_source" not in cu.lower()
    assert "gclid" not in cu.lower()
    assert cu.startswith("https://en.wikipedia.org/wiki/Pittsburgh")


def test_canonicalize_sorts_query_params_and_trims_trailing_slash():
    u = "https://example.com/a/?b=2&a=1&utm_medium=x"
    cu = canonicalize_url(u)
    assert cu == "https://example.com/a?a=1&b=2"


def test_is_http_url():
    assert is_http_url("https://example.com/x")
    assert is_http_url("http://example.com/x")
    assert not is_http_url("mailto:test@example.com")
    assert not is_http_url("javascript:void(0)")
    assert not is_http_url("/relative/path")


def test_should_drop():
    patterns = ["mailto:", "javascript:", ".png", ".jpg"]
    assert should_drop("mailto:test@example.com", patterns)
    assert should_drop("javascript:void(0)", patterns)
    assert should_drop("https://example.com/a/b/cat.PNG", patterns)
    assert not should_drop("https://example.com/page.html", patterns)


def test_absolutize_relative_links():
    base = "https://example.com/dir/page.html"
    assert absolutize(base, "/x") == "https://example.com/x"
    assert absolutize(base, "sub") == "https://example.com/dir/sub"