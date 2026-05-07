"""
Screener.in deep fundamentals scraper.

Fetches the full consolidated company page from screener.in and parses
every data section into clean Python dicts:
  - key_ratios
  - profit_loss
  - balance_sheet
  - quarterly_results
  - shareholding
  - cash_flow
  - peer_comparison   (if present)
  - about             (company description text)

Polite scraping constraints:
  • time.sleep(1) before every HTTP request
  • Warm the session with the screener.in homepage first (gets cookies + CSRF)
  • Optional login via SCREENER_EMAIL / SCREENER_PASSWORD in .env
  • Aggressive SQLite cache (1-day TTL via fundamentals/cache.py)
  • No redistribution of scraped data

403 / bot-protection handling:
  The scraper first GETs the homepage to pick up any session cookies, then
  fetches the company page.  If screener.in has added stricter bot detection
  (Cloudflare JS challenge) the request will still 403; in that case install
  `cloudscraper` (pip install cloudscraper) and it will be used automatically.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup, Tag

from config import settings
from logger import get_logger

log = get_logger(__name__)

_BASE_URL = "https://www.screener.in/company/{symbol}/consolidated/"
_FALLBACK_URL = "https://www.screener.in/company/{symbol}/"
_HOME_URL = "https://www.screener.in/"
_LOGIN_URL = "https://www.screener.in/login/"

# Minimal headers that mimic a real browser.  Deliberately simple —
# matching what the existing app.py scraper uses successfully.
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
}

_TIMEOUT = 25
_REQUEST_DELAY = 1.0  # polite delay before every HTTP request

# Section IDs as found in screener.in HTML source
_SECTION_IDS: Dict[str, str] = {
    "quarterly_results": "quarters",
    "profit_loss":       "profit-loss",
    "balance_sheet":     "balance-sheet",
    "cash_flow":         "cash-flow",
    "shareholding":      "shareholding",
    "peer_comparison":   "peers",
}


def _build_session() -> requests.Session:
    """
    Try cloudscraper first (handles Cloudflare JS challenges automatically).
    Fall back to a plain requests.Session if cloudscraper is not installed.
    """
    try:
        import cloudscraper  # type: ignore[import]
        sess = cloudscraper.create_scraper(
            browser={"browser": "firefox", "platform": "linux", "mobile": False}
        )
        log.debug("using_cloudscraper")
        return sess
    except ImportError:
        pass

    sess = requests.Session()
    sess.headers.update(_HEADERS)
    return sess


class ScreenerDeepFetcher:
    """
    Parse a screener.in company page into structured Python dicts.

    Usage
    -----
    fetcher = ScreenerDeepFetcher()
    data    = fetcher.fetch_all("BEL")
    """

    def __init__(self) -> None:
        self._session = _build_session()
        self._warmed = False

    # ── Public API ─────────────────────────────────────────────────────────

    def fetch_all(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch and parse all fundamental sections for *symbol*.

        Returns a dict with keys: symbol, url, about, key_ratios,
        profit_loss, balance_sheet, quarterly_results, shareholding,
        cash_flow, peer_comparison, metadata.
        """
        symbol = symbol.upper().strip()
        url, soup = self._fetch_page(symbol)

        result: Dict[str, Any] = {
            "symbol":  symbol,
            "url":     url,
            "about":   self._parse_about(soup),
            "key_ratios":        self._parse_key_ratios(soup),
            "profit_loss":       self._parse_table_section(soup, "profit_loss"),
            "balance_sheet":     self._parse_table_section(soup, "balance_sheet"),
            "quarterly_results": self._parse_table_section(soup, "quarterly_results"),
            "shareholding":      self._parse_shareholding(soup),
            "cash_flow":         self._parse_table_section(soup, "cash_flow"),
            "peer_comparison":   self._parse_table_section(soup, "peer_comparison"),
        }

        total_rows = sum(len(v) for v in result.values() if isinstance(v, list))
        result["metadata"] = {
            "consolidated": "consolidated" in url,
            "total_rows_scraped": total_rows,
        }
        log.info("screener_fetch_complete", symbol=symbol, total_rows=total_rows)
        return result

    # ── HTTP ───────────────────────────────────────────────────────────────

    def _warm_session(self) -> None:
        """GET the screener.in homepage to pick up session cookies."""
        if self._warmed:
            return
        try:
            time.sleep(_REQUEST_DELAY)
            resp = self._session.get(_HOME_URL, timeout=_TIMEOUT)
            log.debug("session_warmed", status=resp.status_code)
            # Attempt login if credentials are configured in .env
            email = getattr(settings, "screener_email", "")
            password = getattr(settings, "screener_password", "")
            if email and password:
                self._login(email, password, resp.text)
        except Exception as exc:
            log.warning("session_warm_failed", error=str(exc))
        finally:
            self._warmed = True

    def _login(self, email: str, password: str, homepage_html: str) -> None:
        """POST login credentials to screener.in (enables full 10-year data)."""
        try:
            soup = BeautifulSoup(homepage_html, "lxml")
            csrf_input = soup.find("input", {"name": "csrfmiddlewaretoken"})
            csrf = csrf_input["value"] if csrf_input else ""
            time.sleep(_REQUEST_DELAY)
            resp = self._session.post(
                _LOGIN_URL,
                data={
                    "username": email,
                    "password": password,
                    "csrfmiddlewaretoken": csrf,
                    "next": "/",
                },
                headers={"Referer": _HOME_URL},
                timeout=_TIMEOUT,
            )
            if resp.url and "login" not in resp.url:
                log.info("screener_login_success")
            else:
                log.warning("screener_login_failed", hint="Check SCREENER_EMAIL / SCREENER_PASSWORD in .env")
        except Exception as exc:
            log.warning("screener_login_error", error=str(exc))

    def _fetch_page(self, symbol: str) -> Tuple[str, BeautifulSoup]:
        """Warm session, then GET the company page. Falls back to standalone URL on 404."""
        self._warm_session()

        url = _BASE_URL.format(symbol=symbol)
        log.info("screener_fetching", symbol=symbol, url=url)
        time.sleep(_REQUEST_DELAY)

        resp = self._session.get(url, timeout=_TIMEOUT)

        if resp.status_code == 404:
            url = _FALLBACK_URL.format(symbol=symbol)
            log.info("screener_fallback_url", symbol=symbol, url=url)
            time.sleep(_REQUEST_DELAY)
            resp = self._session.get(url, timeout=_TIMEOUT)

        if resp.status_code == 404:
            raise ValueError(
                f"Symbol '{symbol}' not found on screener.in (HTTP 404). "
                "Verify the NSE symbol spelling."
            )

        if resp.status_code == 403:
            raise RuntimeError(
                f"Screener.in blocked the request for '{symbol}' (HTTP 403). "
                "Install `cloudscraper` (pip install cloudscraper) to bypass "
                "Cloudflare protection, or add SCREENER_EMAIL / SCREENER_PASSWORD "
                "to .env to use an authenticated session."
            )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Screener.in returned HTTP {resp.status_code} for '{symbol}'."
            )

        return url, BeautifulSoup(resp.text, "lxml")

    # ── Section Parsers ───────────────────────────────────────────────────

    def _parse_about(self, soup: BeautifulSoup) -> str:
        for sel_tag, sel_attr in [
            ("div",     {"class": "about"}),
            ("section", {"id":    "about"}),
            ("div",     {"id":    "about"}),
        ]:
            tag = soup.find(sel_tag, sel_attr)
            if tag:
                return tag.get_text(" ", strip=True)[:2000]
        sub = soup.find("div", {"class": "sub"})
        if sub:
            return sub.get_text(" ", strip=True)[:2000]
        return ""

    def _parse_key_ratios(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        ratios: List[Dict[str, str]] = []

        top = soup.find("ul", id="top-ratios")
        if top:
            for li in top.find_all("li"):
                name_span = li.find("span", {"class": "name"}) or li.find("span")
                val_span  = li.find("span", {"class": "number"})
                if name_span and val_span:
                    ratios.append({
                        "name":  name_span.get_text(strip=True),
                        "value": val_span.get_text(strip=True),
                    })

        if not ratios:
            for li in soup.select("ul.company-ratios li"):
                spans = li.find_all("span")
                if len(spans) >= 2:
                    ratios.append({"name": spans[0].get_text(strip=True), "value": spans[-1].get_text(strip=True)})

        if not ratios:
            for li in soup.select("li.flex-column"):
                spans = li.find_all("span")
                if len(spans) >= 2:
                    ratios.append({"name": spans[0].get_text(strip=True), "value": spans[-1].get_text(strip=True)})

        log.debug("key_ratios_parsed", count=len(ratios))
        return ratios

    def _parse_table_section(
        self, soup: BeautifulSoup, section_key: str
    ) -> List[Dict[str, Any]]:
        section_id = _SECTION_IDS.get(section_key)
        if not section_id:
            return []
        section = soup.find(["section", "div"], id=section_id)
        if section is None:
            log.debug("section_not_found", section=section_key)
            return []
        table = section.find("table", class_="data-table") or section.find("table")
        if table is None:
            return []
        headers = self._extract_headers(table)
        rows    = self._extract_rows(table, headers)
        log.debug("table_parsed", section=section_key, rows=len(rows))
        return rows

    def _parse_shareholding(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        section = soup.find(["section", "div"], id="shareholding")
        if section is None:
            return []
        rows: List[Dict[str, Any]] = []
        for table in section.find_all("table"):
            rows.extend(self._extract_rows(table, self._extract_headers(table)))
        log.debug("shareholding_parsed", rows=len(rows))
        return rows

    # ── HTML helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_headers(table: Tag) -> List[str]:
        thead = table.find("thead")
        if thead:
            ths = thead.find_all("th") or thead.find_all("td")
            if ths:
                return [th.get_text(strip=True) for th in ths]
        tbody = table.find("tbody") or table
        first = tbody.find("tr")
        if first:
            return [c.get_text(strip=True) for c in first.find_all(["th", "td"])]
        return []

    @staticmethod
    def _extract_rows(table: Tag, headers: List[str]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        tbody = table.find("tbody") or table
        for tr in tbody.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            if not cells:
                continue
            row: Dict[str, Any] = {}
            for i, cell in enumerate(cells):
                raw = cell.get_text(strip=True).replace("\xa0", " ").replace(",", "").strip()
                key = (headers[i] if i < len(headers) else f"col_{i}") or (
                    "row_label" if i == 0 else f"col_{i}"
                )
                try:
                    if raw in ("", "-", "--", "N/A", "—"):
                        row[key] = None
                    elif "%" in raw:
                        row[key] = float(raw.replace("%", "").strip())
                    else:
                        row[key] = float(raw)
                except ValueError:
                    row[key] = raw
            if any(v is not None and v != "" for v in row.values()):
                rows.append(row)
        return rows
