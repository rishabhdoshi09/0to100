"""
Google Finance live quotes for NSE.

Google Finance has no official API — this module reads the public quote
page (google.com/finance/quote/SYMBOL:NSE) and extracts price, previous
close, and day change. Used for LIVE QUOTES only; Google publishes no
historical OHLCV, so daily history still comes from Kite/yfinance.

Endpoints:
  Stocks : https://www.google.com/finance/quote/RELIANCE:NSE
  Indices: https://www.google.com/finance/quote/NIFTY_50:INDEXNSE
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from logger import get_logger

log = get_logger(__name__)

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}

# Index symbol → Google Finance path
INDEX_MAP = {
    "^NSEI":     "NIFTY_50:INDEXNSE",
    "NIFTY":     "NIFTY_50:INDEXNSE",
    "^NSEBANK":  "NIFTY_BANK:INDEXNSE",
    "BANKNIFTY": "NIFTY_BANK:INDEXNSE",
    "^BSESN":    "SENSEX:INDEXBOM",
    "SENSEX":    "SENSEX:INDEXBOM",
}

# Primary: stable data attributes Google embeds on the quote container
#   <div ... data-last-price="1388.7" data-last-normal-market-timestamp="..." ...>
_ATTR_PRICE_RE = re.compile(r'data-last-price="([\d.]+)"')
# Fallbacks: rendered text (CSS classes — Google changes these sometimes)
_PRICE_RE = re.compile(r'class="YMlKec fxKbKc"[^>]*>\s*(?:₹|Rs\.?|INR|\$)?\s*([\d,]+\.?\d*)')
# Previous close appears in the stats table inside class="P6K39c"
_PREV_RE = re.compile(r'[Pp]revious close.*?class="P6K39c"[^>]*>\s*(?:₹|Rs\.?|INR|\$)?\s*([\d,]+\.?\d*)',
                      re.DOTALL)


def _to_float(s: str) -> float:
    return float(s.replace(",", ""))


def get_quote(symbol: str, timeout: int = 8) -> Optional[dict]:
    """
    Live quote for one NSE symbol (or index). Returns
    {price, prev_close, chg_pct} or None if the page can't be parsed.
    """
    import requests
    path = INDEX_MAP.get(symbol.upper(), f"{symbol.upper()}:NSE")
    url = f"https://www.google.com/finance/quote/{path}"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        if resp.status_code != 200:
            log.debug("google_finance_http", symbol=symbol, status=resp.status_code)
            return None
        html = resp.text
        # Primary: data attribute (stable). Fallback: rendered price div.
        m_attr = _ATTR_PRICE_RE.search(html)
        if m_attr:
            price = float(m_attr.group(1))
        else:
            m_price = _PRICE_RE.search(html)
            if not m_price:
                log.debug("google_finance_parse_miss", symbol=symbol)
                return None
            price = _to_float(m_price.group(1))
        m_prev = _PREV_RE.search(html)
        prev = _to_float(m_prev.group(1)) if m_prev else 0.0
        chg = (price - prev) / prev * 100 if prev else 0.0
        return {"price": price, "prev_close": prev, "chg_pct": round(chg, 2)}
    except Exception as exc:
        log.debug("google_finance_quote_failed", symbol=symbol, error=str(exc))
        return None


def get_quotes(symbols: list[str], max_workers: int = 8) -> dict[str, dict]:
    """Parallel live quotes for many symbols. Missing symbols are omitted."""
    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(get_quote, s): s for s in symbols}
        for fut in as_completed(futs):
            try:
                q = fut.result()
                if q:
                    out[futs[fut]] = q
            except Exception:
                pass
    if out:
        log.debug("google_finance_quotes", requested=len(symbols), got=len(out))
    return out
