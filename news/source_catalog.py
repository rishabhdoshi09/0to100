"""Curated high-coverage source catalog for Indian retail-market news.

Official exchange/regulator sources receive the highest trust tier. Business
media and Google News discovery feeds broaden coverage but never outrank an
official filing merely because they publish more often.
"""
from __future__ import annotations

from urllib.parse import quote_plus, urlparse

from news.curator_models import SourceSpec


_OFFICIAL_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        key="nse_announcements",
        name="NSE Corporate Announcements",
        url="https://www.nseindia.com/api/corporate-announcements?index=equities",
        category="company",
        tier=1,
        official=True,
        kind="nse_announcements",
        max_items=500,
    ),
    SourceSpec(
        key="sebi",
        name="SEBI",
        url="https://www.sebi.gov.in/sebirss.xml",
        category="regulation",
        tier=1,
        official=True,
    ),
    SourceSpec(
        key="rbi_press",
        name="RBI Press Releases",
        url="https://rbi.org.in/pressreleases_rss.xml",
        category="economy",
        tier=1,
        official=True,
    ),
    SourceSpec(
        key="rbi_notifications",
        name="RBI Notifications",
        url="https://rbi.org.in/notifications_rss.xml",
        category="regulation",
        tier=1,
        official=True,
    ),
    SourceSpec(
        key="rbi_speeches",
        name="RBI Speeches",
        url="https://rbi.org.in/speeches_rss.xml",
        category="economy",
        tier=1,
        official=True,
    ),
    SourceSpec(
        key="pib_releases",
        name="Press Information Bureau",
        url="https://www.pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3",
        category="economy",
        tier=1,
        official=True,
        max_items=150,
    ),
)


_DISCOVERY_QUERIES: tuple[tuple[str, str, str], ...] = (
    ("google_india_markets", "India stock market NSE BSE shares", "market"),
    ("google_fno", "India NSE futures options F&O stocks", "derivatives"),
    ("google_economy", "India economy RBI inflation GDP rupee crude oil", "economy"),
    ("google_results", "India listed company results order win merger acquisition", "company"),
    ("google_global", "US Fed crude oil dollar China tariffs India stocks", "global"),
)


def _name_from_url(url: str) -> str:
    host = (urlparse(url).hostname or url).lower().removeprefix("www.")
    names = {
        "economictimes.indiatimes.com": "Economic Times",
        "livemint.com": "Mint",
        "business-standard.com": "Business Standard",
        "cnbctv18.com": "CNBC-TV18",
        "moneycontrol.com": "Moneycontrol",
        "thehindubusinessline.com": "BusinessLine",
    }
    return names.get(host, host)


def _key_from_url(url: str) -> str:
    host = (urlparse(url).hostname or "feed").lower().removeprefix("www.")
    return "configured_" + "".join(ch if ch.isalnum() else "_" for ch in host).strip("_")


def default_sources(extra_feed_urls: list[str] | tuple[str, ...] | None = None) -> list[SourceSpec]:
    """Return a deduplicated source catalog.

    `extra_feed_urls` usually comes from the existing `settings.rss_feed_list`,
    preserving the user's configured publishers without making the catalog a
    second configuration authority.
    """
    sources = list(_OFFICIAL_SOURCES)
    for url in extra_feed_urls or ():
        clean = str(url or "").strip()
        if not clean:
            continue
        sources.append(
            SourceSpec(
                key=_key_from_url(clean),
                name=_name_from_url(clean),
                url=clean,
                category="market",
                tier=2,
                official=False,
                max_items=100,
            )
        )

    for key, query, category in _DISCOVERY_QUERIES:
        sources.append(
            SourceSpec(
                key=key,
                name=f"Google News · {category.title()}",
                url=(
                    "https://news.google.com/rss/search?q="
                    + quote_plus(query)
                    + "&hl=en-IN&gl=IN&ceid=IN:en"
                ),
                category=category,
                tier=3,
                official=False,
                max_items=100,
            )
        )

    deduped: dict[str, SourceSpec] = {}
    for source in sources:
        if source.enabled:
            deduped.setdefault(source.url, source)
    return list(deduped.values())
