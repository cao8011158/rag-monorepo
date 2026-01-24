from __future__ import annotations

from urllib.parse import urlparse, urljoin, urlunparse, parse_qsl, urlencode

DROP_QUERY_KEYS_PREFIX = ("utm_",)
DROP_QUERY_KEYS_EXACT = {"gclid", "fbclid"}

def canonicalize_url(url: str) -> str:
    """Make URL stable for dedup: remove fragment, drop tracking params, normalize host/scheme."""
    u = url.strip()
    if not u:
        return ""

    p = urlparse(u)
    scheme = (p.scheme or "http").lower()
    netloc = (p.netloc or "").lower()
    path = p.path or "/"
    # normalize trailing slash (keep root '/')
    if path != "/" and path.endswith("/"):
        path = path[:-1]

    # drop fragment
    fragment = ""

    # drop tracking params, sort remaining
    q = []
    for k, v in parse_qsl(p.query, keep_blank_values=True):
        kl = k.lower()
        if kl in DROP_QUERY_KEYS_EXACT:
            continue
        if any(kl.startswith(pref) for pref in DROP_QUERY_KEYS_PREFIX):
            continue
        q.append((k, v))
    q.sort(key=lambda kv: (kv[0], kv[1]))
    query = urlencode(q, doseq=True)

    return urlunparse((scheme, netloc, path, p.params, query, fragment))

def same_domain(url: str, seed_url: str) -> bool:
    return urlparse(url).netloc.lower() == urlparse(seed_url).netloc.lower()

def is_http_url(url: str) -> bool:
    p = urlparse(url)
    return p.scheme in ("http", "https") and bool(p.netloc)

def should_drop(url: str, patterns: list[str]) -> bool:
    u = url.lower()
    for pat in patterns:
        if u.startswith(pat.lower()) or pat.lower() in u:
            return True
    return False

def absolutize(base_url: str, href: str) -> str:
    return urljoin(base_url, href)