"""
US index membership — browse/search US stocks index-wise.

The US analogue of NSE's Nifty groupings: instead of one flat ~5,000-name
listing, let the user scope to the index they actually think in —
**S&P 500**, **NASDAQ-100**, **Dow 30**.

Membership is fetched from Wikipedia's maintained constituents tables and
cross-checked against our own authoritative listed universe
(`data.us_universe`) so only genuine common stocks survive; cached daily.
If the live fetch fails we fall back to a curated list and SAY so
(invariant: stale must look stale — the picker labels its source).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from logger import get_logger

log = get_logger(__name__)

_CACHE_FILE = Path(__file__).resolve().parent.parent / "logs" / "us_indices.json"
_TTL_S = 86400                        # refresh membership once a day
_TICKER = re.compile(r"^[A-Z]{1,5}$")  # plain common-stock tickers

# Display order = how traders rank them.
INDICES = ("S&P 500", "NASDAQ-100", "Dow 30")

# Wikipedia constituents pages + the anchor id of the table to scope to.
_WIKI = {
    "S&P 500":    ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                   "constituents"),
    "NASDAQ-100": ("https://en.wikipedia.org/wiki/Nasdaq-100", "constituents"),
    "Dow 30":     ("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
                   "constituents"),
}

# ── Curated fallbacks (used only when the live fetch fails) ───────────────────
# Dow 30 is small + stable → complete + authoritative even offline.
_DOW30 = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT",
]
# NASDAQ-100 / S&P 500 change often — the curated set is a liquid SUBSET,
# labeled as such; the live fetch supplies the complete, current membership.
_NDX_SUBSET = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO", "TSLA",
    "COST", "NFLX", "AMD", "PEP", "ADBE", "LIN", "CSCO", "TMUS", "INTU",
    "QCOM", "TXN", "AMGN", "ISRG", "AMAT", "BKNG", "HON", "VRTX", "ADP",
    "MU", "ADI", "PANW", "GILD", "LRCX", "REGN", "SBUX", "MDLZ", "KLAC",
    "SNPS", "CDNS", "MELI", "CRWD", "MAR", "CTAS", "ORLY", "CEG", "ABNB",
    "PYPL", "MRVL", "FTNT", "DASH", "ADSK", "WDAY", "NXPI", "ROP", "PCAR",
    "MNST", "AEP", "PAYX", "KDP", "TTD", "ROST", "KHC", "EA", "VRSK",
    "EXC", "CTSH", "FAST", "XEL", "DDOG", "LULU", "CSGP", "ON", "IDXX",
    "ZS", "ANSS", "DXCM", "CDW", "BIIB", "SMCI", "ARM", "PLTR", "APP",
]
_SPX_SUBSET = sorted(set(_NDX_SUBSET) | set(_DOW30) | {
    "BRK", "JPM", "V", "MA", "UNH", "LLY", "XOM", "CVX", "PG", "HD", "ABBV",
    "KO", "PEP", "MRK", "WMT", "BAC", "WFC", "GS", "MS", "C", "SCHW", "AXP",
    "PFE", "TMO", "ABT", "DHR", "BMY", "GILD", "CVS", "CI", "ELV", "MDT",
    "T", "VZ", "CMCSA", "DIS", "NKE", "MCD", "SBUX", "LOW", "TGT", "BKNG",
    "CAT", "DE", "GE", "BA", "LMT", "RTX", "UPS", "UNP", "HON", "MMM",
    "COP", "SLB", "EOG", "PSX", "MPC", "OXY", "NEE", "DUK", "SO", "F", "GM",
})


def us_indices() -> list[str]:
    return list(INDICES)


# ── Pure, testable parser ─────────────────────────────────────────────────────

def _extract_symbols(html: str, anchor_id: str) -> set[str]:
    """Scope to the constituents table, then pull ticker-looking tokens from
    cell/link inner-text. Cross-checking against the real universe (done by
    the caller) removes any stray non-constituent tokens."""
    i = html.find(f'id="{anchor_id}"')
    seg = html[i:] if i != -1 else html
    j = seg.find("</table>")
    if j != -1:
        seg = seg[:j]
    toks = set(re.findall(r">([A-Z]{1,5})<", seg))     # >MMM< in a cell/link
    return {t for t in toks if _TICKER.match(t)}


def _fetch_index(index: str) -> dict[str, str]:
    """Live membership {symbol: name} from Wikipedia, filtered to real listed
    common stocks. Empty dict on any failure (caller falls back to curated)."""
    url, anchor = _WIKI[index]
    try:
        import requests
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        if r.status_code != 200 or not r.text:
            return {}
        cand = _extract_symbols(r.text, anchor)
    except Exception as exc:
        log.debug("us_index_fetch_failed", index=index, error=str(exc)[:80])
        return {}
    try:
        from data.us_universe import get_us_universe_with_names
        names = get_us_universe_with_names()
    except Exception:
        names = {}
    # keep only genuine listed common stocks; carry authoritative names
    return {s: names.get(s, s) for s in sorted(cand) if s in names} if names \
        else {s: s for s in sorted(cand)}


def _curated(index: str) -> dict[str, str]:
    syms = {"Dow 30": _DOW30, "NASDAQ-100": _NDX_SUBSET,
            "S&P 500": _SPX_SUBSET}.get(index, [])
    try:
        from data.us_universe import get_us_universe_with_names
        names = get_us_universe_with_names()
    except Exception:
        names = {}
    return {s: names.get(s, s) for s in sorted(syms)}


# ── Cache ─────────────────────────────────────────────────────────────────────

def _load_cache() -> dict | None:
    try:
        if _CACHE_FILE.exists():
            data = json.loads(_CACHE_FILE.read_text())
            if time.time() - data.get("ts", 0) < _TTL_S and data.get("indices"):
                return data
    except Exception:
        pass
    return None


def _save_cache(payload: dict) -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(payload))
    except Exception as exc:
        log.debug("us_index_cache_save_failed", error=str(exc))


# ── Public API ────────────────────────────────────────────────────────────────

def get_index_members(index: str) -> tuple[dict[str, str], str]:
    """Return ({symbol: name}, source) for an index.

    source is 'live' when Wikipedia membership was used, 'curated' (a labeled
    liquid subset) when the fetch was unavailable. Dow 30's curated list is
    complete, so it reads 'curated (complete)'.
    """
    if index not in _WIKI:
        return {}, "unknown"

    cache = _load_cache()
    if cache and index in cache.get("indices", {}):
        entry = cache["indices"][index]
        return dict(entry.get("members", {})), entry.get("source", "cached")

    members = _fetch_index(index)
    if len(members) >= (25 if index == "Dow 30" else 80):
        source = "live"
    else:
        members = _curated(index)
        source = "curated (complete)" if index == "Dow 30" else "curated (subset)"

    # merge into the day's cache without dropping other indices
    payload = _load_cache() or {"ts": time.time(), "indices": {}}
    payload.setdefault("indices", {})[index] = {"members": members, "source": source}
    payload["ts"] = payload.get("ts", time.time())
    _save_cache(payload)
    return members, source
