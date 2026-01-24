from __future__ import annotations

from bs4 import BeautifulSoup

from cc_pipeline.crawl.url_utils import absolutize

def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    out: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        if not href:
            continue
        out.append(absolutize(base_url, href))
    return out