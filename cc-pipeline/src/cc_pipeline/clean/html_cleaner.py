from __future__ import annotations

import json
import trafilatura
from bs4 import BeautifulSoup


def _fallback_bs4(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.get_text(strip=True) if soup.title else "").strip()

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    text = soup.get_text("\n", strip=True)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return title, "\n".join(lines)


def html_to_text(html: str) -> tuple[str, str]:
    """
    Same interface as before:
        input:  html (str)
        output: (title: str, text: str)
    """
    html = html or ""

    extracted = trafilatura.extract(
        html,
        output_format="json",
        with_metadata=True,
        include_comments=False,
        include_tables=False,
        include_links=False,
        favor_recall=False,
    )

    if extracted:
        try:
            data = json.loads(extracted)
        except Exception:
            data = None

        if isinstance(data, dict):
            title = (data.get("title") or "").strip()
            text = (data.get("text") or "").strip()

            if text:
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                text = "\n".join(lines)

            if len(text) >= 200:   # 这个阈值可调
                return title, text

    # fallback to old behavior
    return _fallback_bs4(html)
