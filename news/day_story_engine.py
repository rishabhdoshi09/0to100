"""Day-story engine — ranks today's market news for a lively Wrap of the Day.

Honesty rules (hard):
  • Only uses curated articles already fetched from real sources.
  • Never invents prices, CA, listings, earnings, or order sizes.
  • Teach points are generic event framing; company facts stay in the headline/summary.
  • Missing news stays missing — wrap then falls back to tape/global cues.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "logs" / "news_curator.sqlite3"

# Retail wrap prefers company/day tape over paperwork.
_WRAP_EVENT_BOOST: dict[str, int] = {
    "listing_ipo": 42,
    "results": 38,
    "order_or_contract": 38,
    "merger_or_acquisition": 32,
    "fund_raising": 28,
    "capital_action": 22,
    "macro": 18,
    "derivatives": 14,
    "rating": 16,
    "promoter_or_insider": 14,
    "regulatory": 6,
    "other": 8,
}

_PAPERWORK_MARKERS = (
    "CIRCULAR",
    "MASTER CIRCULAR",
    "CONSULTATION PAPER",
    "PRESS RELEASE",
    "NOTIFICATION",
    "SPEECH",
    "AGENDA",
    "MINUTES OF",
    "CORRIGENDUM",
)

_TEACH: dict[str, dict[str, str]] = {
    "listing_ipo": {
        "likely_negative": (
            "showing that even strong listings can see short-term profit booking "
            "when valuations run ahead"
        ),
        "likely_positive": (
            "keeping the listing spotlight on business quality versus opening-day froth"
        ),
        "default": (
            "bringing the IPO/listing debate back to valuation versus lasting cash flows"
        ),
    },
    "results": {
        "likely_positive": (
            "proving that strong execution in a focused franchise can still deliver "
            "exceptional earnings growth"
        ),
        "likely_negative": (
            "reminding investors that a soft print can quickly reset expectations"
        ),
        "default": (
            "keeping the focus on whether the earnings print resets the growth story"
        ),
    },
    "order_or_contract": {
        "likely_positive": (
            "showing that large order books can still set years of growth visibility"
        ),
        "default": (
            "putting the revenue pipeline and execution risk back under the lens"
        ),
    },
    "merger_or_acquisition": {
        "default": "because deal terms can reprice both the buyer and the target",
    },
    "fund_raising": {
        "default": "as funding terms can change dilution, leverage and growth runway",
    },
    "macro": {
        "default": "because macro prints still steer rates, the rupee and risk appetite",
    },
    "derivatives": {
        "default": "keeping F&O positioning and expiry risk in the same frame as the cash market",
    },
    "global": {
        "default": (
            "as investors stay glued to global earnings and risk cues despite mixed signals"
        ),
    },
}


@dataclass(frozen=True)
class DayStory:
    headline: str
    wrap_line: str
    source: str
    source_key: str
    url: str
    event_type: str
    category: str
    impact_score: int
    wrap_score: int
    direction: str
    published_at: str
    mentioned_symbols: tuple[str, ...] = field(default_factory=tuple)
    why_it_matters: str = ""
    summary: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean(text: str) -> str:
    value = re.sub(r"<[^>]+>", " ", str(text or ""))
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(
        r"\s*[-|:]\s*(Reuters|Bloomberg|Mint|Economic Times|CNBC-TV18|Moneycontrol|Business Standard)\s*$",
        "",
        value,
        flags=re.I,
    )
    return value.strip(" -|:")


def _teach_for(event_type: str, direction: str, category: str) -> str:
    bucket = _TEACH.get(event_type) or (
        _TEACH["global"] if category == "global" else {"default": "keeping today's tape honest about what the evidence actually shows"}
    )
    return str(bucket.get(direction) or bucket.get("default") or "").strip()


def wrap_line_from_article(article: Mapping[str, Any] | Any) -> str:
    """Turn one curated article into a newsletter-style wrap bullet.

    Facts come from the headline (and light summary cues). The teach clause is
    generic event framing — never a fabricated price or order size.
    """
    if hasattr(article, "headline"):
        headline = _clean(getattr(article, "headline", ""))
        event_type = str(getattr(article, "event_type", "") or "other")
        direction = str(getattr(article, "direction", "") or "unclear")
        category = str(getattr(article, "category", "") or "")
        summary = _clean(getattr(article, "summary", ""))
    else:
        headline = _clean(str(article.get("headline") or ""))
        event_type = str(article.get("event_type") or "other")
        direction = str(article.get("direction") or "unclear")
        category = str(article.get("category") or "")
        summary = _clean(str(article.get("summary") or ""))

    if not headline:
        return ""

    # Prefer a short factual clause from summary only when it clearly extends the headline.
    fact = headline
    if summary and len(summary) > 40:
        # Pull a first-sentence hint if it shares a token with the headline and adds verbs.
        first = re.split(r"(?<=[.!?])\s+", summary)[0].strip()
        head_tokens = {t for t in re.findall(r"[A-Za-z]{4,}", headline.lower())}
        first_tokens = {t for t in re.findall(r"[A-Za-z]{4,}", first.lower())}
        overlap = head_tokens & first_tokens
        if (
            first
            and first.lower() not in headline.lower()
            and len(overlap) >= 2
            and len(first) <= 160
            and not any(m in first.upper() for m in ("CLICK HERE", "READ MORE", "SUBSCRIBE"))
        ):
            # Only use summary when it looks like the same story — still sourced text.
            fact = first

    teach = _teach_for(event_type, direction, category)
    fact = fact.rstrip(".")
    if teach:
        teach_bit = teach[0].lower() + teach[1:] if teach else teach
        line = f"{fact}, {teach_bit}"
    else:
        line = fact
    return line.rstrip(".") + "."


def wrap_relevance_score(article: Mapping[str, Any] | Any) -> int:
    if hasattr(article, "headline"):
        headline = str(getattr(article, "headline", "") or "")
        event_type = str(getattr(article, "event_type", "") or "other")
        category = str(getattr(article, "category", "") or "")
        impact = int(getattr(article, "impact_score", 0) or 0)
        tier = int(getattr(article, "source_tier", 3) or 3)
        official = bool(getattr(article, "official", False))
        symbols = tuple(getattr(article, "mentioned_symbols", ()) or ())
        corroboration = int(getattr(article, "corroboration_count", 1) or 1)
        source_key = str(getattr(article, "source_key", "") or "")
    else:
        headline = str(article.get("headline") or "")
        event_type = str(article.get("event_type") or "other")
        category = str(article.get("category") or "")
        impact = int(article.get("impact_score") or 0)
        tier = int(article.get("source_tier") or 3)
        official = bool(article.get("official"))
        symbols = tuple(article.get("mentioned_symbols") or ())
        corroboration = int(article.get("corroboration_count") or 1)
        source_key = str(article.get("source_key") or "")

    upper = headline.upper()
    score = impact + _WRAP_EVENT_BOOST.get(event_type, 8)
    if symbols:
        score += min(18, 8 + 3 * len(symbols))
    if tier == 2:
        score += 14  # business media carries the retail day narrative
    elif tier == 3 and category in {"company", "market", "global"}:
        score += 8
    if category in {"company", "market", "derivatives"}:
        score += 8
    if category == "global":
        score += 6
    if corroboration >= 2:
        score += 6

    # Demote pure regulator paperwork that crowds out day stories.
    if official and not symbols and event_type in {"regulatory", "other", "macro"}:
        score -= 22
    if any(marker in upper for marker in _PAPERWORK_MARKERS) and not symbols:
        score -= 18
    if source_key.startswith("rbi_") or source_key in {"sebi", "pib_releases"}:
        if not symbols and event_type not in {"macro"}:
            score -= 12
        if event_type == "macro" and any(w in upper for w in ("REPO", "CPI", "INFLATION", "GDP")):
            score += 10  # keep the real macro movers

    # Lexical boosts for day-tape words even before event tagging catches up.
    for word, bump in (
        ("IPO", 12),
        ("LISTING", 12),
        ("DEBUT", 10),
        ("ORDER BOOK", 12),
        ("EARNINGS", 10),
        ("PROFIT", 6),
        ("REVENUE", 6),
        ("FUTURES", 6),
        ("DEFENCE", 6),
        ("PHARMA", 4),
    ):
        if word in upper:
            score += bump
    return int(score)


def _to_story(article: Any) -> DayStory | None:
    headline = _clean(getattr(article, "headline", ""))
    if not headline:
        return None
    line = wrap_line_from_article(article)
    if not line:
        return None
    return DayStory(
        headline=headline,
        wrap_line=line,
        source=str(getattr(article, "source", "") or ""),
        source_key=str(getattr(article, "source_key", "") or ""),
        url=str(getattr(article, "url", "") or ""),
        event_type=str(getattr(article, "event_type", "") or "other"),
        category=str(getattr(article, "category", "") or ""),
        impact_score=int(getattr(article, "impact_score", 0) or 0),
        wrap_score=wrap_relevance_score(article),
        direction=str(getattr(article, "direction", "") or "unclear"),
        published_at=str(getattr(article, "published_at", "") or ""),
        mentioned_symbols=tuple(getattr(article, "mentioned_symbols", ()) or ()),
        why_it_matters=str(getattr(article, "why_it_matters", "") or ""),
        summary=_clean(getattr(article, "summary", ""))[:280],
    )


def select_day_stories(
    articles: Sequence[Any],
    *,
    limit: int = 5,
) -> list[DayStory]:
    """Rank curated articles into a diversified day-story set."""
    scored: list[DayStory] = []
    for article in articles:
        story = _to_story(article)
        if story:
            scored.append(story)
    scored.sort(key=lambda s: (s.wrap_score, s.impact_score), reverse=True)

    picked: list[DayStory] = []
    seen_clusters: set[str] = set()
    event_counts: dict[str, int] = {}
    for story in scored:
        # Light de-dupe on first 8 meaningful tokens.
        tokens = " ".join(re.findall(r"[A-Za-z0-9]{3,}", story.headline.lower())[:8])
        if tokens in seen_clusters:
            continue
        # Keep variety: max 2 of same event type in a 5-bullet wrap.
        if event_counts.get(story.event_type, 0) >= 2:
            continue
        # Prefer at least some company/market stories over all-macro.
        if (
            len(picked) >= 2
            and story.event_type in {"regulatory"}
            and not story.mentioned_symbols
            and any(p.event_type not in {"regulatory"} for p in picked)
        ):
            continue
        picked.append(story)
        seen_clusters.add(tokens)
        event_counts[story.event_type] = event_counts.get(story.event_type, 0) + 1
        if len(picked) >= max(1, int(limit)):
            break
    return picked


def build_day_stories(
    *,
    hours: int = 20,
    limit: int = 5,
    path: Path | None = None,
    refresh_if_stale: bool = False,
    stale_minutes: int = 45,
) -> dict[str, Any]:
    """Load curator rows and emit wrap-ready day stories."""
    db_path = Path(path or DEFAULT_DB)
    stories: list[DayStory] = []
    refreshed = False
    refresh_error = ""
    article_count = 0

    if refresh_if_stale:
        try:
            age_min = None
            if db_path.exists():
                age_min = (datetime.now(timezone.utc).timestamp() - db_path.stat().st_mtime) / 60.0
            if age_min is None or age_min >= float(stale_minutes):
                from news.curator import NewsCurator
                from news.curator_store import NewsCuratorStore

                NewsCurator(store=NewsCuratorStore(db_path)).refresh()
                refreshed = True
        except Exception as exc:
            refresh_error = str(exc)

    try:
        from news.curator_store import NewsCuratorStore

        store = NewsCuratorStore(db_path)
        try:
            rows = store.recent(hours=hours, limit=250, min_impact=20)
            article_count = len(rows)
            stories = select_day_stories(rows, limit=limit)
        finally:
            store.close()
    except Exception as exc:
        return {
            "available": False,
            "stories": [],
            "count": 0,
            "article_count": 0,
            "refreshed": refreshed,
            "message": f"Day-story engine unavailable: {exc}",
            "refresh_error": refresh_error,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "honesty": "No invented day stories — curator store must supply real articles.",
        }

    return {
        "available": bool(stories),
        "stories": [s.as_dict() for s in stories],
        "count": len(stories),
        "article_count": article_count,
        "refreshed": refreshed,
        "refresh_error": refresh_error,
        "message": (
            f"{len(stories)} day story(ies) from {article_count} curated article(s)"
            if stories
            else "No wrap-ready day stories in the last session — refresh market news."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "places_orders": False,
        "honesty": (
            "Day stories are ranked from fetched Moneycontrol/ET/Mint/BS/CNBC/Google News "
            "plus official filings. Missing stays missing."
        ),
    }
