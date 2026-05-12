from __future__ import annotations
import structlog
from app.scrapers.base import BaseScraper, BrowserPool, ScrapeResult
from app.core.config import settings

log = structlog.get_logger(__name__)

_BSE_API = "https://api.bseindia.com/BseIndiaAPI/api"
_ENDPOINTS = {
    "company_info": f"{_BSE_API}/ComHeadernew/w?scrip_cd={{scrip_code}}",
    "filings": f"{_BSE_API}/AnnSubCategoryGetData/w?strCat=-1&strPrevDate={{from_date}}&strScrip={{scrip_code}}&strType=C&strVal=0&Category=EQT",
}


class BSEScraper(BaseScraper):
    def __init__(self, browser_pool: BrowserPool) -> None:
        super().__init__(browser_pool)

    async def scrape_company(self, symbol: str, scrip_code: str) -> ScrapeResult:
        ctx = await self._pool.new_context()
        try:
            page = await ctx.new_page()
            self._setup_interceptor(page)
            await self._navigate_with_retry(page, settings.bse_base_url)
            await self._human_delay()
            data = {}
            for key, url_tpl in _ENDPOINTS.items():
                try:
                    url = url_tpl.format(scrip_code=scrip_code, from_date="20240101")
                    resp = await page.evaluate(
                        f"""async () => {{
                            const r = await fetch('{url}', {{ headers: {{ 'Accept': 'application/json' }} }});
                            return r.ok ? await r.json() : null;
                        }}"""
                    )
                    if resp:
                        data[key] = resp
                except Exception as exc:
                    log.warning("bse_endpoint_failed", symbol=symbol, key=key, error=str(exc))
            return ScrapeResult(symbol=symbol, success=bool(data), data=data)
        except Exception as exc:
            return ScrapeResult(symbol=symbol, success=False, error=str(exc))
        finally:
            await ctx.close()
