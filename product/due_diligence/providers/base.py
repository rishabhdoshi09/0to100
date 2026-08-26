"""Source adapters. Business logic must not bind to HTML selectors."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "logs" / "research_evidence"
DEFAULT_TIMEOUT_S = 20
DEFAULT_RETRIES = 2
DEFAULT_RATE_GAP_S = 1.0
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
    "QuantTermPersonalResearch/1.0"
)


class ProviderError(RuntimeError):
    """Fetch or parse failed. Callers fall back; they never invent a number."""


@dataclass(frozen=True)
class FetchResult:
    ok: bool
    provider: str
    url: str
    retrieved_at: str
    status_code: int | None = None
    content_type: str = ""
    body: bytes = b""
    error: str = ""
    content_hash: str = ""
    archived_path: str = ""


@dataclass
class ProviderPolicy:
    timeout_s: float = DEFAULT_TIMEOUT_S
    retries: int = DEFAULT_RETRIES
    rate_gap_s: float = DEFAULT_RATE_GAP_S
    user_agent: str = USER_AGENT
    allowed_hosts: tuple[str, ...] = ()


class SourceAdapter(Protocol):
    """One website / filing source. Returns normalized dicts, never HTML blobs."""

    name: str
    source_type: str

    def fetch(self, symbol: str, *, force: bool = False) -> FetchResult: ...

    def parse(self, result: FetchResult) -> dict[str, Any]: ...


_LAST_HIT: dict[str, float] = {}


def respect_rate_limit(provider: str, gap_s: float) -> None:
    last = _LAST_HIT.get(provider, 0.0)
    wait = gap_s - (time.monotonic() - last)
    if wait > 0:
        time.sleep(wait)
    _LAST_HIT[provider] = time.monotonic()


def content_hash(body: bytes) -> str:
    return hashlib.sha256(body or b"").hexdigest()


def archive_bytes(symbol: str, provider: str, body: bytes, *, suffix: str = ".bin") -> Path | None:
    if not body:
        return None
    digest = content_hash(body)
    folder = ARCHIVE / symbol.upper() / "raw" / provider
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{digest[:16]}{suffix}"
    if not path.exists():
        path.write_bytes(body)
    index = folder / "index.jsonl"
    with index.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"hash": digest, "path": str(path), "bytes": len(body)}) + "\n")
    return path


def host_allowed(url: str, allowed: tuple[str, ...]) -> bool:
    if not allowed:
        return True
    host = (urlparse(url).hostname or "").lower()
    return any(host == item or host.endswith("." + item) for item in allowed)


def empty_normalized() -> dict[str, Any]:
    return {
        "quarterly_results": [],
        "profit_loss": [],
        "cash_flow": [],
        "balance_sheet": [],
        "shareholding": [],
        "key_ratios": [],
        "peer_comparison": [],
        "announcements": [],
        "about": "",
        "url": "",
        "conflicts": [],
    }


def merge_normalized(*packs: Mapping[str, Any]) -> dict[str, Any]:
    """Concatenate table rows from several adapters. Conflicts are detected later."""
    out = empty_normalized()
    for pack in packs:
        if not isinstance(pack, Mapping):
            continue
        for key in (
            "quarterly_results", "profit_loss", "cash_flow", "balance_sheet",
            "shareholding", "key_ratios", "peer_comparison", "announcements",
        ):
            out[key].extend(list(pack.get(key) or []))
        if pack.get("about") and not out["about"]:
            out["about"] = str(pack.get("about") or "")
        if pack.get("url") and not out["url"]:
            out["url"] = str(pack.get("url") or "")
        out["conflicts"].extend(list(pack.get("conflicts") or []))
    return out
