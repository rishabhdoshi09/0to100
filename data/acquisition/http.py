"""Bounded HTTP for ingest only. Not imported by EvidenceSnapshot."""
from __future__ import annotations

import time
from typing import Any

import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://www.nseindia.com/",
    "Accept": "*/*",
}


def nse_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    try:
        s.get(
            "https://www.nseindia.com/companies-listing/corporate-filings-financial-results",
            timeout=20,
        )
    except Exception:
        pass
    return s


def get_bytes(
    url: str,
    *,
    session: requests.Session | None = None,
    retries: int = 3,
    timeout: int = 60,
    sleep: float = 0.4,
) -> tuple[bytes | None, dict[str, Any]]:
    sess = session or requests.Session()
    if "nseindia.com" in url and "User-Agent" not in sess.headers:
        sess.headers.update(HEADERS)
    meta: dict[str, Any] = {"url": url, "attempts": 0}
    last_err = None
    for i in range(retries + 1):
        meta["attempts"] = i + 1
        try:
            resp = sess.get(url, timeout=timeout)
            meta["http_status"] = resp.status_code
            if resp.status_code == 200 and resp.content:
                return resp.content, meta
            if resp.status_code == 404:
                return None, meta
            last_err = f"http_{resp.status_code}"
        except Exception as exc:
            last_err = str(exc)
        time.sleep(sleep * (i + 1))
    meta["error"] = last_err
    return None, meta
