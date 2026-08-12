"""Data contracts for the retail news-curation pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class SourceSpec:
    key: str
    name: str
    url: str
    category: str
    tier: int = 3
    official: bool = False
    kind: str = "rss"  # rss | nse_announcements
    max_items: int = 80
    enabled: bool = True


@dataclass(frozen=True)
class FetchedNews:
    headline: str
    summary: str
    source_key: str
    source_name: str
    url: str
    published_at: datetime
    category_hint: str
    source_tier: int
    official: bool
    hinted_symbols: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CuratedArticle:
    article_id: str
    cluster_key: str
    headline: str
    summary: str
    source: str
    source_key: str
    source_tier: int
    official: bool
    url: str
    published_at: str
    fetched_at: str
    category: str
    event_type: str
    impact_score: int
    direction: str
    why_it_matters: str
    mentioned_symbols: tuple[str, ...] = field(default_factory=tuple)
    fno_symbols: tuple[str, ...] = field(default_factory=tuple)
    sectors: tuple[str, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)
    corroboration_count: int = 1

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceHealth:
    source_key: str
    source_name: str
    status: str
    fetched_at: str
    article_count: int = 0
    latency_ms: int = 0
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RefreshReport:
    started_at: str
    completed_at: str
    sources_attempted: int
    sources_ok: int
    fetched_articles: int
    curated_articles: int
    inserted_or_updated: int
    high_impact_articles: int
    errors: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
