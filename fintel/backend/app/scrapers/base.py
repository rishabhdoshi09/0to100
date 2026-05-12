from __future__ import annotations
import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any
from playwright.async_api import Browser, BrowserContext, Page, async_playwright

_STEALTH_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
Object.defineProperty(navigator, 'languages', { get: () => ['en-IN', 'en-US', 'en'] });
window.chrome = { runtime: {} };
"""


@dataclass
class InterceptedRequest:
    url: str
    method: str
    response_body: Any
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScrapeResult:
    symbol: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    intercepted: list[InterceptedRequest] = field(default_factory=list)


class BrowserPool:
    def __init__(self, pool_size: int = 2) -> None:
        self._pool_size = pool_size
        self._playwright = None
        self._browser: Browser | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
            ],
        )

    async def stop(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def new_context(self) -> BrowserContext:
        if self._browser is None:
            await self.start()
        ctx = await self._browser.new_context(
            viewport={"width": 1366, "height": 768},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="en-IN",
            timezone_id="Asia/Kolkata",
        )
        await ctx.add_init_script(_STEALTH_SCRIPT)
        return ctx


class BaseScraper:
    def __init__(self, browser_pool: BrowserPool) -> None:
        self._pool = browser_pool
        self._intercepted: list[InterceptedRequest] = []

    def _find_json_endpoint(self, pattern: str) -> InterceptedRequest | None:
        for req in self._intercepted:
            if pattern in req.url:
                return req
        return None

    def _setup_interceptor(self, page: Page) -> None:
        async def handle_response(response):
            ct = response.headers.get("content-type", "")
            if "json" in ct:
                try:
                    body = await response.json()
                    self._intercepted.append(
                        InterceptedRequest(
                            url=response.url,
                            method=response.request.method,
                            response_body=body,
                        )
                    )
                except Exception:
                    pass

        page.on("response", handle_response)

    async def _navigate_with_retry(
        self, page: Page, url: str, wait_for: str = "networkidle", max_retries: int = 3
    ) -> None:
        for attempt in range(max_retries):
            try:
                await page.goto(url, wait_until=wait_for, timeout=30000)
                return
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def _human_delay(self, min_ms: int = 800, max_ms: int = 2500) -> None:
        delay = random.randint(min_ms, max_ms) / 1000
        await asyncio.sleep(delay)

    async def _scroll_to_bottom(self, page: Page, steps: int = 5) -> None:
        for i in range(steps):
            await page.evaluate(f"window.scrollTo(0, document.body.scrollHeight * {(i+1)/steps})")
            await asyncio.sleep(0.3)
