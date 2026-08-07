"""Automated high-coverage news curation for Indian retail investors.

Pipeline: fetch -> validate -> deduplicate -> map NSE/F&O entities -> classify
-> impact-score -> explain -> persist. News is context, never an order signal.
"""
from __future__ import annotations

import hashlib
import html
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from typing import Iterable, Mapping
from urllib.parse import urljoin

import requests

from news.curator_models import (
    CuratedArticle,
    FetchedNews,
    RefreshReport,
    SourceHealth,
    SourceSpec,
)
from news.curator_store import NewsCuratorStore
from news.source_catalog import default_sources

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/xml, application/json, text/html;q=0.9, */*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
}

_STOP_COMPANY_WORDS = {
    "LIMITED", "LTD", "INDIA", "INDIAN", "THE", "COMPANY", "CORPORATION",
    "CORP", "INDUSTRIES", "INDUSTRY", "ENTERPRISES", "HOLDINGS", "PLC",
}
_AMBIGUOUS_SYMBOLS = {"IT", "ON", "GO", "ARE", "CAN", "SET", "GET", "ALL", "ONE", "AND"}

_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "economy": (
        "GDP", "INFLATION", "CPI", "WPI", "IIP", "FISCAL", "BUDGET", "TAX",
        "INTEREST RATE", "REPO RATE", "LIQUIDITY", "RBI", "RUPEE", "FOREX",
        "BOND YIELD", "ECONOMY", "ECONOMIC", "MONETARY POLICY",
    ),
    "regulation": (
        "SEBI", "CIRCULAR", "REGULATION", "REGULATORY", "ORDER", "RULING",
        "COMPLIANCE", "BAN", "SURVEILLANCE", "MARGIN RULE", "DISCLOSURE",
    ),
    "derivatives": (
        "FUTURES", "OPTIONS", "F&O", "OPEN INTEREST", "EXPIRY", "DERIVATIVE",
        "SHORT COVERING", "LONG BUILDUP", "SHORT BUILDUP", "LONG UNWINDING",
    ),
    "global": (
        "FED", "FEDERAL RESERVE", "CHINA", "TARIFF", "DOLLAR INDEX", "US MARKET",
        "WALL STREET", "CRUDE OIL", "BRENT", "OPEC", "GEOPOLITICAL", "WAR",
        "EUROPE", "JAPAN", "GLOBAL MARKET",
    ),
    "company": (
        "QUARTERLY RESULT", "FINANCIAL RESULT", "EARNINGS", "ORDER WIN", "CONTRACT",
        "MERGER", "ACQUISITION", "DEMERGER", "BUYBACK", "DIVIDEND", "BONUS",
        "STOCK SPLIT", "FUND RAISE", "QIP", "BOARD MEETING", "PROMOTER",
    ),
}

_EVENT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("listing_ipo", ("IPO", "LISTING", "DEBUT", "LISTED TODAY", "OPENS AT", "GREY MARKET", "SUBSCRIPTION")),
    ("results", ("RESULT", "EARNINGS", "REVENUE", "PROFIT", "EBITDA", "MARGIN", "NET PROFIT", "TOP LINE")),
    (
        "order_or_contract",
        (
            "ORDER WIN", "WINS ORDER", "BAGS ORDER", "WON ORDER", "ORDER BOOK", "ORDERBOOK",
            "CONTRACT AWARD", "LETTER OF AWARD", "CRORE ORDER", "DEFENCE ORDER",
        ),
    ),
    ("merger_or_acquisition", ("MERGER", "ACQUISITION", "TAKEOVER", "DEMERGER", "AMALGAMATION")),
    ("capital_action", ("DIVIDEND", "BONUS", "STOCK SPLIT", "BUYBACK", "RIGHTS ISSUE")),
    ("fund_raising", ("FUND RAISE", "QIP", "PREFERENTIAL ISSUE", "DEBT ISSUE", "FPO")),
    ("rating", ("UPGRADE", "DOWNGRADE", "RATING", "TARGET PRICE")),
    ("promoter_or_insider", ("PROMOTER", "INSIDER TRADING", "PLEDGE", "ENCUMBRANCE")),
    ("macro", ("GDP", "INFLATION", "CPI", "REPO RATE", "BUDGET", "FISCAL", "RUPEE", "CRUDE")),
    ("regulatory", ("SEBI", "RBI", "CIRCULAR", "REGULATION", "PENALTY", "CONSULTATION PAPER")),
    ("derivatives", ("FUTURES", "OPTIONS", "OPEN INTEREST", "F&O", "EXPIRY", "US FUTURES")),
)

_POSITIVE_WORDS = (
    "BEATS ESTIMATE", "RECORD PROFIT", "PROFIT RISE", "REVENUE RISE", "UPGRADE",
    "ORDER WIN", "WINS ORDER", "APPROVAL", "DIVIDEND", "BUYBACK", "BONUS",
    "EXPANSION", "LOWER COST", "RATE CUT", "CUTS REPO RATE", "INFLATION EASES",
    "STRONG GROWTH", "DEBT REDUCTION",
)
_NEGATIVE_WORDS = (
    "MISSES ESTIMATE", "PROFIT FALL", "LOSS WIDENS", "DOWNGRADE", "PENALTY",
    "BAN", "DEFAULT", "FRAUD", "INVESTIGATION", "RESIGNATION", "PLEDGE",
    "DEBT RISE", "MARGIN FALL", "RATE HIKE", "WEAK GROWTH", "ORDER CANCEL",
)

_SECTOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Banking": ("BANK", "NBFC", "LENDER", "CREDIT", "LOAN", "DEPOSIT"),
    "IT": ("IT SERVICES", "SOFTWARE", "TECHNOLOGY", "AI ", "CLOUD", "DIGITAL"),
    "Auto": ("AUTO", "VEHICLE", "CAR", "TWO-WHEELER", "EV ", "TRACTOR"),
    "Pharma": ("PHARMA", "DRUG", "USFDA", "HEALTHCARE", "HOSPITAL"),
    "Metals": ("STEEL", "METAL", "ALUMINIUM", "COPPER", "MINING"),
    "Energy": ("CRUDE", "OIL", "GAS", "POWER", "ENERGY", "ELECTRICITY"),
    "Realty": ("REAL ESTATE", "REALTY", "HOUSING", "PROPERTY"),
    "FMCG": ("FMCG", "CONSUMER", "FOOD", "BEVERAGE", "RURAL DEMAND"),
    "Defence": ("DEFENCE", "MISSILE", "ARMY", "NAVY", "AEROSPACE"),
    "Railways": ("RAILWAY", "RAIL ", "METRO", "WAGON"),
    "Telecom": ("TELECOM", "5G", "SPECTRUM", "MOBILE SUBSCRIBER"),
}


def _clean_text(value: str) -> str:
    text = html.unescape(re.sub(r"<[^>]+>", " ", str(value or "")))
    return re.sub(r"\s+", " ", text).strip()


def _normalise_headline(value: str) -> str:
    text = _clean_text(value).upper()
    text = re.sub(r"\s*[-|:]\s*(REUTERS|BLOOMBERG|MINT|ECONOMIC TIMES|CNBC-TV18|MONEYCONTROL)\s*$", "", text)
    text = re.sub(r"[^A-Z0-9%₹$]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _cluster_key(headline: str) -> str:
    normalised = _normalise_headline(headline)
    tokens = [t for t in normalised.split() if len(t) > 2]
    stable = " ".join(tokens[:18]) or normalised
    return hashlib.sha1(stable.encode("utf-8")).hexdigest()[:20]


def _article_id(item: FetchedNews) -> str:
    base = f"{_cluster_key(item.headline)}|{item.source_key}|{item.url}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:24]


def _parse_datetime(value: object, fallback: datetime | None = None) -> datetime:
    fallback = fallback or datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value or "").strip()
    if text:
        try:
            parsed = parsedate_to_datetime(text)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except Exception:
            pass
        for fmt in (
            "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%d-%b-%Y %H:%M:%S",
            "%d-%b-%Y", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y",
        ):
            try:
                parsed = datetime.strptime(text[:25], fmt)
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
    return fallback


class EntityResolver:
    """Maps article text to the full NSE cash universe and current F&O underlyings."""

    def __init__(
        self,
        symbol_names: Mapping[str, str] | None = None,
        fno_symbols: Iterable[str] = (),
    ) -> None:
        symbol_names = symbol_names or {}
        self.symbol_names = {str(k).upper(): str(v or k) for k, v in symbol_names.items()}
        self.fno_symbols = {str(s).upper() for s in fno_symbols}
        self._name_aliases: list[tuple[str, str]] = []
        for symbol, name in self.symbol_names.items():
            words = [
                word for word in re.findall(r"[A-Z0-9]+", name.upper())
                if word not in _STOP_COMPANY_WORDS and len(word) >= 3
            ]
            if words:
                alias = " ".join(words[:3])
                if len(alias) >= 5:
                    self._name_aliases.append((alias, symbol))
        self._name_aliases.sort(key=lambda pair: len(pair[0]), reverse=True)

    @classmethod
    def from_quantterm(cls) -> "EntityResolver":
        names: dict[str, str] = {}
        fno: set[str] = set()
        try:
            from data.nse_universe import get_nse_universe_with_names
            names = dict(get_nse_universe_with_names())
        except Exception:
            try:
                from config import settings
                names = {symbol: symbol for symbol in settings.symbol_list}
            except Exception:
                names = {}
        try:
            from data.fno_universe import current_fno_universe
            fno = set(current_fno_universe().symbols)
        except Exception:
            fno = set()
        return cls(names, fno)

    def resolve(self, text: str, hinted_symbols: Iterable[str] = ()) -> tuple[tuple[str, ...], tuple[str, ...]]:
        upper = _clean_text(text).upper()
        found = {str(s).upper() for s in hinted_symbols if str(s).strip()}
        for symbol in self.symbol_names:
            if symbol in _AMBIGUOUS_SYMBOLS:
                continue
            pattern = rf"(?<![A-Z0-9]){re.escape(symbol)}(?![A-Z0-9])"
            if re.search(pattern, upper):
                found.add(symbol)
        for alias, symbol in self._name_aliases:
            if alias in upper:
                found.add(symbol)
        valid = tuple(sorted(s for s in found if not self.symbol_names or s in self.symbol_names))
        return valid, tuple(sorted(set(valid).intersection(self.fno_symbols)))


@lru_cache(maxsize=1)
def get_entity_resolver() -> EntityResolver:
    return EntityResolver.from_quantterm()


def _category(text: str, hint: str, symbols: tuple[str, ...]) -> str:
    upper = text.upper()
    scores = {name: sum(1 for word in words if word in upper) for name, words in _CATEGORY_KEYWORDS.items()}
    best = max(scores, key=scores.get) if any(scores.values()) else ""
    if best:
        return best
    if symbols:
        return "company"
    return hint if hint in {"economy", "regulation", "derivatives", "global", "company", "market"} else "market"


def _event_type(text: str, category: str) -> str:
    upper = text.upper()
    for name, words in _EVENT_KEYWORDS:
        if any(word in upper for word in words):
            return name
    return category


def _direction(text: str) -> str:
    upper = text.upper()
    pos = sum(1 for word in _POSITIVE_WORDS if word in upper)
    neg = sum(1 for word in _NEGATIVE_WORDS if word in upper)
    if pos and neg:
        return "mixed"
    if pos > neg:
        return "likely_positive"
    if neg > pos:
        return "likely_negative"
    return "unclear"


def _sectors(text: str) -> tuple[str, ...]:
    upper = text.upper()
    return tuple(sorted(name for name, words in _SECTOR_KEYWORDS.items() if any(word in upper for word in words)))


def _impact_score(
    item: FetchedNews,
    *,
    symbols: tuple[str, ...],
    fno_symbols: tuple[str, ...],
    category: str,
    event_type: str,
    corroboration_count: int,
    now: datetime,
) -> int:
    age_hours = max(0.0, (now - item.published_at.astimezone(timezone.utc)).total_seconds() / 3600)
    recency = max(0, 25 - int(age_hours / 2))
    source = {1: 24, 2: 16, 3: 9}.get(int(item.source_tier), 5)
    score = recency + source
    if item.official:
        score += 6
    if symbols:
        score += min(20, 10 + len(symbols) * 3)
    if fno_symbols:
        score += min(12, 6 + len(fno_symbols) * 2)
    if category in {"economy", "regulation", "global", "derivatives"}:
        score += 8
    if event_type in {
        "listing_ipo", "results", "order_or_contract", "merger_or_acquisition", "regulatory", "macro",
    }:
        score += 12
    elif event_type in {"capital_action", "fund_raising", "rating", "promoter_or_insider", "derivatives"}:
        score += 8
    score += min(10, max(0, corroboration_count - 1) * 3)
    return max(0, min(100, int(score)))


def _why_it_matters(
    *,
    category: str,
    event_type: str,
    symbols: tuple[str, ...],
    fno_symbols: tuple[str, ...],
    sectors: tuple[str, ...],
    direction: str,
) -> str:
    parts: list[str] = []
    if symbols:
        parts.append(f"Directly linked to {', '.join(symbols[:5])}")
    if fno_symbols:
        parts.append("these underlyings also have current F&O contracts, so volatility and open interest may react")
    if event_type == "listing_ipo":
        parts.append("IPO/listing days often show valuation froth versus lasting business quality")
    elif event_type == "results":
        parts.append("earnings can reset price, volume and analyst expectations")
    elif event_type == "order_or_contract":
        parts.append("a material order book can set multi-year revenue visibility")
    elif event_type == "merger_or_acquisition":
        parts.append("deal terms can reprice both the buyer and target")
    elif event_type == "regulatory":
        parts.append("regulatory changes can alter costs, eligibility or market behaviour")
    elif event_type == "macro":
        parts.append("macro changes can affect rates, currency, demand and sector valuations")
    elif category == "global":
        parts.append("global risk, crude and dollar moves can spill into Indian equities")
    if sectors:
        parts.append("relevant sectors: " + ", ".join(sectors[:4]))
    if direction == "likely_positive":
        parts.append("wording suggests a potentially positive effect, but price confirmation is still required")
    elif direction == "likely_negative":
        parts.append("wording suggests a potentially negative effect, but price confirmation is still required")
    elif direction == "mixed":
        parts.append("the article contains both positive and negative cues")
    if not parts:
        parts.append("broad market context that may change risk appetite")
    return "; ".join(parts) + "."


def curate_articles(
    items: Iterable[FetchedNews],
    resolver: EntityResolver | None = None,
    *,
    now: datetime | None = None,
) -> list[CuratedArticle]:
    resolver = resolver or get_entity_resolver()
    now = now or datetime.now(timezone.utc)
    clusters: dict[str, list[FetchedNews]] = defaultdict(list)
    for item in items:
        if item.headline.strip():
            clusters[_cluster_key(item.headline)].append(item)

    output: list[CuratedArticle] = []
    for cluster, cluster_items in clusters.items():
        cluster_items.sort(key=lambda item: (item.source_tier, -item.published_at.timestamp()))
        lead = cluster_items[0]
        combined_text = " ".join(f"{item.headline} {item.summary}" for item in cluster_items[:5])
        hinted = {symbol for item in cluster_items for symbol in item.hinted_symbols}
        symbols, fno = resolver.resolve(combined_text, hinted)
        category = _category(combined_text, lead.category_hint, symbols)
        event = _event_type(combined_text, category)
        direction = _direction(combined_text)
        sectors = _sectors(combined_text)
        corroboration = len({item.source_key for item in cluster_items})
        score = _impact_score(
            lead,
            symbols=symbols,
            fno_symbols=fno,
            category=category,
            event_type=event,
            corroboration_count=corroboration,
            now=now,
        )
        source_names = list(dict.fromkeys(item.source_name for item in cluster_items))
        tags = tuple(sorted(set((category, event, direction, *sectors))))
        output.append(
            CuratedArticle(
                article_id=_article_id(lead),
                cluster_key=cluster,
                headline=_clean_text(lead.headline),
                summary=_clean_text(lead.summary)[:1200],
                source=" · ".join(source_names[:4]),
                source_key=lead.source_key,
                source_tier=lead.source_tier,
                official=lead.official,
                url=lead.url,
                published_at=lead.published_at.astimezone(timezone.utc).isoformat(),
                fetched_at=now.isoformat(),
                category=category,
                event_type=event,
                impact_score=score,
                direction=direction,
                why_it_matters=_why_it_matters(
                    category=category,
                    event_type=event,
                    symbols=symbols,
                    fno_symbols=fno,
                    sectors=sectors,
                    direction=direction,
                ),
                mentioned_symbols=symbols,
                fno_symbols=fno,
                sectors=sectors,
                tags=tags,
                corroboration_count=corroboration,
            )
        )
    return sorted(output, key=lambda article: (article.impact_score, article.published_at), reverse=True)


class NewsCurator:
    def __init__(
        self,
        *,
        sources: Iterable[SourceSpec] | None = None,
        store: NewsCuratorStore | None = None,
        resolver: EntityResolver | None = None,
        timeout: int = 12,
        workers: int = 8,
    ) -> None:
        if sources is None:
            extra: list[str] = []
            try:
                from config import settings
                extra = list(settings.rss_feed_list)
            except Exception:
                pass
            sources = default_sources(extra)
        self.sources = tuple(source for source in sources if source.enabled)
        self.store = store or NewsCuratorStore()
        self.resolver = resolver or get_entity_resolver()
        self.timeout = max(3, int(timeout))
        self.workers = max(1, min(16, int(workers)))

    def refresh(self, *, max_age_hours: int = 168, keep_days: int = 30) -> RefreshReport:
        started = datetime.now(timezone.utc)
        fetched: list[FetchedNews] = []
        health: list[SourceHealth] = []
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=min(self.workers, max(1, len(self.sources)))) as pool:
            future_map = {
                pool.submit(self._fetch_source, source, max_age_hours): source
                for source in self.sources
            }
            for future in as_completed(future_map):
                source = future_map[future]
                try:
                    articles, status = future.result()
                    fetched.extend(articles)
                    health.append(status)
                    if status.status != "OK":
                        errors.append(f"{source.name}: {status.error or status.status}")
                except Exception as exc:
                    now = datetime.now(timezone.utc).isoformat()
                    health.append(SourceHealth(source.key, source.name, "ERROR", now, error=str(exc)))
                    errors.append(f"{source.name}: {exc}")
        curated = curate_articles(fetched, self.resolver, now=datetime.now(timezone.utc))
        written = self.store.upsert_articles(curated)
        self.store.upsert_source_health(health)
        self.store.prune(keep_days=keep_days)
        completed = datetime.now(timezone.utc)
        return RefreshReport(
            started_at=started.isoformat(),
            completed_at=completed.isoformat(),
            sources_attempted=len(self.sources),
            sources_ok=sum(1 for item in health if item.status == "OK"),
            fetched_articles=len(fetched),
            curated_articles=len(curated),
            inserted_or_updated=written,
            high_impact_articles=sum(1 for item in curated if item.impact_score >= 70),
            errors=tuple(errors),
        )

    def _fetch_source(self, source: SourceSpec, max_age_hours: int) -> tuple[list[FetchedNews], SourceHealth]:
        started = time.monotonic()
        try:
            if source.kind == "nse_announcements":
                items = self._fetch_nse_announcements(source, max_age_hours)
            else:
                items = self._fetch_rss(source, max_age_hours)
            status = "OK" if items else "EMPTY"
            error = "" if items else "No recent usable entries"
        except Exception as exc:
            items = []
            status = "ERROR"
            error = str(exc)
        latency = int((time.monotonic() - started) * 1000)
        return items, SourceHealth(
            source_key=source.key,
            source_name=source.name,
            status=status,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            article_count=len(items),
            latency_ms=latency,
            error=error,
        )

    def _fetch_rss(self, source: SourceSpec, max_age_hours: int) -> list[FetchedNews]:
        import feedparser
        response = requests.get(source.url, headers=_BROWSER_HEADERS, timeout=self.timeout)
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, max_age_hours))
        output: list[FetchedNews] = []
        for entry in list(feed.entries)[: source.max_items]:
            published = _parse_datetime(entry.get("published") or entry.get("updated") or entry.get("pubDate"))
            if published < cutoff:
                continue
            headline = _clean_text(entry.get("title", ""))
            if not headline:
                continue
            output.append(
                FetchedNews(
                    headline=headline,
                    summary=_clean_text(entry.get("summary", entry.get("description", "")))[:1500],
                    source_key=source.key,
                    source_name=str(feed.feed.get("title") or source.name),
                    url=str(entry.get("link") or source.url),
                    published_at=published,
                    category_hint=source.category,
                    source_tier=source.tier,
                    official=source.official,
                )
            )
        return output

    def _fetch_nse_announcements(self, source: SourceSpec, max_age_hours: int) -> list[FetchedNews]:
        session = requests.Session()
        session.headers.update(_BROWSER_HEADERS)
        session.get("https://www.nseindia.com/", timeout=self.timeout)
        response = session.get(source.url, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, max_age_hours))
        output: list[FetchedNews] = []
        for row in list(rows)[: source.max_items]:
            symbol = str(row.get("symbol") or row.get("sm_symbol") or "").strip().upper()
            headline = _clean_text(
                row.get("attchmntText") or row.get("desc") or row.get("subject") or row.get("sm_name") or ""
            )
            if symbol and symbol not in headline.upper():
                headline = f"{symbol}: {headline}"
            if not headline:
                continue
            published = _parse_datetime(
                row.get("an_dt") or row.get("sort_date") or row.get("date") or row.get("broadcastDateTime")
            )
            if published < cutoff:
                continue
            attachment = str(row.get("attchmntFile") or row.get("attachment") or row.get("fileName") or "")
            if attachment and not attachment.startswith("http"):
                attachment = urljoin("https://nsearchives.nseindia.com/corporate/", attachment)
            output.append(
                FetchedNews(
                    headline=headline,
                    summary=_clean_text(row.get("details") or row.get("remarks") or row.get("desc") or "")[:1500],
                    source_key=source.key,
                    source_name=source.name,
                    url=attachment or "https://www.nseindia.com/companies-listing/corporate-filings-announcements?tabIndex=equity",
                    published_at=published,
                    category_hint="company",
                    source_tier=source.tier,
                    official=True,
                    hinted_symbols=(symbol,) if symbol else (),
                )
            )
        return output

    def latest(self, **filters) -> list[CuratedArticle]:
        return self.store.recent(**filters)

    def source_health(self) -> list[SourceHealth]:
        return self.store.source_health()

    def stats(self, hours: int = 24) -> dict[str, int]:
        return self.store.stats(hours=hours)

    def raw_articles(self, *, hours: int = 24, limit: int = 500):
        """Compatibility bridge into the existing LLM news-context pipeline."""
        from news.fetcher import RawArticle
        return [
            RawArticle(
                headline=item.headline,
                summary=item.summary,
                source=item.source,
                url=item.url,
                published_at=_parse_datetime(item.published_at),
            )
            for item in self.latest(hours=hours, limit=limit)
        ]
