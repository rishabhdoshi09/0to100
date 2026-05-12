from __future__ import annotations
import structlog
from bs4 import BeautifulSoup
from app.scrapers.base import BaseScraper, BrowserPool, ScrapeResult
from app.core.config import settings

log = structlog.get_logger(__name__)


class ScreenerScraper(BaseScraper):
    def __init__(self, browser_pool: BrowserPool) -> None:
        super().__init__(browser_pool)

    async def scrape_company(self, symbol: str) -> ScrapeResult:
        ctx = await self._pool.new_context()
        try:
            page = await ctx.new_page()
            self._setup_interceptor(page)
            url = f"{settings.screener_base_url}/company/{symbol}/consolidated/"
            await self._navigate_with_retry(page, url)
            await self._scroll_to_bottom(page)
            await self._human_delay()
            html = await page.content()
            soup = BeautifulSoup(html, "lxml")
            data = {
                "key_ratios": self._extract_key_ratios(soup),
                "financials_income": self._extract_table(soup, "income"),
                "financials_balance": self._extract_table(soup, "balance-sheet"),
                "shareholding": self._extract_shareholding(soup),
                "peers": self._extract_peers(soup),
            }
            return ScrapeResult(symbol=symbol, success=True, data=data)
        except Exception as exc:
            log.error("screener_scrape_failed", symbol=symbol, error=str(exc))
            return ScrapeResult(symbol=symbol, success=False, error=str(exc))
        finally:
            await ctx.close()

    def _extract_key_ratios(self, soup: BeautifulSoup) -> dict:
        ratios = {}
        for li in soup.select("#top-ratios li"):
            name_el = li.select_one(".name")
            value_el = li.select_one(".value")
            if name_el and value_el:
                key = name_el.get_text(strip=True).lower().replace(" ", "_").replace(".", "")
                ratios[key] = self._parse_number(value_el.get_text(strip=True))
        return ratios

    def _extract_table(self, soup: BeautifulSoup, section_id: str) -> list[dict]:
        section = soup.find("section", {"id": section_id})
        if not section:
            return []
        table = section.find("table")
        if not table:
            return []
        headers = [th.get_text(strip=True) for th in table.select("thead th")]
        rows = []
        for tr in table.select("tbody tr"):
            cells = [td.get_text(strip=True) for td in tr.select("td")]
            if cells and len(cells) == len(headers):
                rows.append(dict(zip(headers, cells)))
        return rows

    def _extract_shareholding(self, soup: BeautifulSoup) -> dict:
        result = {}
        section = soup.find("section", {"id": "shareholding"})
        if not section:
            return result
        for row in section.select("tr"):
            cells = row.select("td")
            if len(cells) >= 2:
                key = cells[0].get_text(strip=True).lower().replace(" ", "_")
                result[key] = self._parse_number(cells[-1].get_text(strip=True))
        return result

    def _extract_peers(self, soup: BeautifulSoup) -> list[dict]:
        peers = []
        section = soup.find("section", {"id": "peers"})
        if not section:
            return peers
        headers = [th.get_text(strip=True) for th in section.select("thead th")]
        for tr in section.select("tbody tr"):
            cells = [td.get_text(strip=True) for td in tr.select("td")]
            if cells:
                peers.append(dict(zip(headers, cells)))
        return peers

    @staticmethod
    def _parse_number(text: str) -> float | str:
        text = text.strip().replace(",", "")
        multiplier = 1
        if text.endswith("Cr"):
            text, multiplier = text[:-2].strip(), 10_000_000
        elif text.endswith("L"):
            text, multiplier = text[:-1].strip(), 100_000
        try:
            return float(text) * multiplier
        except ValueError:
            return text
