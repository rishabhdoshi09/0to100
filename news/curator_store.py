"""Durable SQLite store for curated market news and source health."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from news.curator_models import CuratedArticle, SourceHealth


class NewsCuratorStore:
    def __init__(self, path: str | Path = "logs/news_curator.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS curated_articles (
                    article_id TEXT PRIMARY KEY,
                    cluster_key TEXT NOT NULL,
                    headline TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    source TEXT NOT NULL,
                    source_key TEXT NOT NULL,
                    source_tier INTEGER NOT NULL,
                    official INTEGER NOT NULL,
                    url TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    category TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    impact_score INTEGER NOT NULL,
                    direction TEXT NOT NULL,
                    why_it_matters TEXT NOT NULL,
                    mentioned_symbols TEXT NOT NULL,
                    fno_symbols TEXT NOT NULL,
                    sectors TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    corroboration_count INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_news_published ON curated_articles(published_at DESC)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_news_impact ON curated_articles(impact_score DESC)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_news_cluster ON curated_articles(cluster_key)"
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS source_health (
                    source_key TEXT PRIMARY KEY,
                    source_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    article_count INTEGER NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    error TEXT NOT NULL
                )
                """
            )

    @staticmethod
    def _dump(values: Iterable[str]) -> str:
        return json.dumps(list(values), ensure_ascii=False)

    @staticmethod
    def _load(value: str) -> tuple[str, ...]:
        try:
            return tuple(str(x) for x in json.loads(value or "[]"))
        except Exception:
            return ()

    def upsert_articles(self, articles: Iterable[CuratedArticle]) -> int:
        sql = """
            INSERT INTO curated_articles (
                article_id, cluster_key, headline, summary, source, source_key,
                source_tier, official, url, published_at, fetched_at, category,
                event_type, impact_score, direction, why_it_matters,
                mentioned_symbols, fno_symbols, sectors, tags, corroboration_count
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(article_id) DO UPDATE SET
                cluster_key=excluded.cluster_key,
                headline=excluded.headline,
                summary=excluded.summary,
                source=excluded.source,
                source_key=excluded.source_key,
                source_tier=excluded.source_tier,
                official=excluded.official,
                url=excluded.url,
                published_at=excluded.published_at,
                fetched_at=excluded.fetched_at,
                category=excluded.category,
                event_type=excluded.event_type,
                impact_score=MAX(curated_articles.impact_score, excluded.impact_score),
                direction=excluded.direction,
                why_it_matters=excluded.why_it_matters,
                mentioned_symbols=excluded.mentioned_symbols,
                fno_symbols=excluded.fno_symbols,
                sectors=excluded.sectors,
                tags=excluded.tags,
                corroboration_count=MAX(curated_articles.corroboration_count, excluded.corroboration_count)
        """
        rows = []
        for article in articles:
            rows.append((
                article.article_id,
                article.cluster_key,
                article.headline,
                article.summary,
                article.source,
                article.source_key,
                int(article.source_tier),
                int(bool(article.official)),
                article.url,
                article.published_at,
                article.fetched_at,
                article.category,
                article.event_type,
                int(article.impact_score),
                article.direction,
                article.why_it_matters,
                self._dump(article.mentioned_symbols),
                self._dump(article.fno_symbols),
                self._dump(article.sectors),
                self._dump(article.tags),
                int(article.corroboration_count),
            ))
        if not rows:
            return 0
        with self._lock, self._conn:
            self._conn.executemany(sql, rows)
        return len(rows)

    def upsert_source_health(self, health: Iterable[SourceHealth]) -> None:
        rows = [(
            h.source_key, h.source_name, h.status, h.fetched_at,
            int(h.article_count), int(h.latency_ms), h.error,
        ) for h in health]
        if not rows:
            return
        with self._lock, self._conn:
            self._conn.executemany(
                """
                INSERT INTO source_health (
                    source_key, source_name, status, fetched_at,
                    article_count, latency_ms, error
                ) VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(source_key) DO UPDATE SET
                    source_name=excluded.source_name,
                    status=excluded.status,
                    fetched_at=excluded.fetched_at,
                    article_count=excluded.article_count,
                    latency_ms=excluded.latency_ms,
                    error=excluded.error
                """,
                rows,
            )

    def recent(
        self,
        *,
        hours: int = 72,
        limit: int = 1000,
        min_impact: int = 0,
        category: str | None = None,
        fno_only: bool = False,
        symbol: str | None = None,
        search: str | None = None,
    ) -> list[CuratedArticle]:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=max(1, hours))).isoformat()
        where = ["published_at >= ?", "impact_score >= ?"]
        params: list[object] = [cutoff, int(min_impact)]
        if category:
            where.append("category = ?")
            params.append(category)
        if fno_only:
            where.append("fno_symbols <> '[]'")
        if symbol:
            where.append("mentioned_symbols LIKE ?")
            params.append(f'%"{symbol.upper()}"%')
        if search:
            where.append("(headline LIKE ? OR summary LIKE ? OR source LIKE ?)")
            term = f"%{search}%"
            params.extend([term, term, term])
        params.append(max(1, int(limit)))
        query = (
            "SELECT * FROM curated_articles WHERE " + " AND ".join(where)
            + " ORDER BY impact_score DESC, published_at DESC LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_article(row) for row in rows]

    def source_health(self) -> list[SourceHealth]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM source_health ORDER BY status ASC, article_count DESC, source_name ASC"
            ).fetchall()
        return [SourceHealth(**dict(row)) for row in rows]

    def stats(self, hours: int = 24) -> dict[str, int]:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=max(1, hours))).isoformat()
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN impact_score >= 70 THEN 1 ELSE 0 END) AS important,
                    SUM(CASE WHEN fno_symbols <> '[]' THEN 1 ELSE 0 END) AS fno_linked,
                    SUM(CASE WHEN category IN ('economy','regulation','global') THEN 1 ELSE 0 END) AS macro,
                    COUNT(DISTINCT source_key) AS sources
                FROM curated_articles WHERE published_at >= ?
                """,
                (cutoff,),
            ).fetchone()
        return {key: int((row[key] if row else 0) or 0) for key in
                ("total", "important", "fno_linked", "macro", "sources")}

    def prune(self, keep_days: int = 30) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=max(1, keep_days))).isoformat()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "DELETE FROM curated_articles WHERE published_at < ?", (cutoff,)
            )
        return int(cur.rowcount or 0)

    def _row_to_article(self, row: sqlite3.Row) -> CuratedArticle:
        payload = dict(row)
        payload["official"] = bool(payload["official"])
        for key in ("mentioned_symbols", "fno_symbols", "sectors", "tags"):
            payload[key] = self._load(payload[key])
        return CuratedArticle(**payload)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
