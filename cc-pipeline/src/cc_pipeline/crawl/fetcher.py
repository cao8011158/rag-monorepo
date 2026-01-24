from __future__ import annotations

import time
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

class Fetcher:
    def __init__(self, user_agent: str, timeout_sec: int, per_host_rps: float):
        self.ua = user_agent
        self.timeout = timeout_sec
        self.delay = 1.0 / max(per_host_rps, 0.001)
        self._last: dict[str, float] = {}

    def _rate_limit(self, host: str) -> None:
        now = time.time()
        last = self._last.get(host, 0.0)
        sleep = (last + self.delay) - now
        if sleep > 0:
            time.sleep(sleep)
        self._last[host] = time.time()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def get(self, url: str) -> requests.Response:
        host = requests.utils.urlparse(url).netloc
        self._rate_limit(host)
        r = requests.get(url, timeout=self.timeout, headers={"User-Agent": self.ua}, allow_redirects=True)
        r.raise_for_status()
        return r
