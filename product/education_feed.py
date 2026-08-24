"""Market Education feed — crunch curated news into learnable cards.

Honesty rules:
  • Never invents articles, headlines, or blog posts.
  • Cards are projections over fetched curator rows + static concept teach-ins.
  • Macro themes come from keyword corroboration (macro_pulse), not forecasts.
  • Every news-backed card keeps the original source URL and publish time.
  • Education is context for learning — never a buy/sell signal.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from product.desk_note import GOLD_LOAN_CONCEPT, MIX_SHIFT_CONCEPT

# Evergreen teach-ins — fixed copy, not generated "blogs".
_CONCEPT_CARDS: list[dict[str, Any]] = [
    {
        "id": "concept-macro-vs-micro",
        "lens": "CONCEPT",
        "title": "Macro vs micro in the share market",
        "teach_point": (
            "Macro is the weather (rates, inflation, flows, geopolitics). "
            "Micro is the company (results, orders, balance sheet, sector peers). "
            "A strong stock can still fall in a risk-off tape."
        ),
        "why_it_matters": (
            "Retail mistakes often come from reading only one lens. "
            "QuantTerm keeps both visible — and never converts either into an order."
        ),
        "tags": ["macro", "micro", "basics"],
        "level": "beginner",
    },
    {
        "id": "concept-fii-dii",
        "lens": "CONCEPT",
        "title": "What FII / DII flows actually mean",
        "teach_point": (
            "FIIs are foreign institutional investors; DIIs are domestic institutions. "
            "One-day flow prints are noisy. Multi-day corroboration matters more than a single headline."
        ),
        "why_it_matters": (
            "Flow headlines move sentiment, but they are context — not a signal to chase "
            "or panic-sell."
        ),
        "tags": ["flows", "macro", "fii"],
        "level": "beginner",
    },
    {
        "id": "concept-rbi-rates",
        "lens": "CONCEPT",
        "title": "Repo rate and equity sectors",
        "teach_point": (
            "Rate hikes usually pressure rate-sensitive names (banks, NBFCs, realty, auto). "
            "Cuts can ease that pressure. The path of rates matters more than one decision day."
        ),
        "why_it_matters": (
            "Policy days are educational events: read the source (RBI), then watch how "
            "sectors react — do not invent a trade from the press release alone."
        ),
        "tags": ["rates", "rbi", "macro"],
        "level": "intermediate",
    },
    {
        "id": "concept-earnings",
        "lens": "CONCEPT",
        "title": "How to read a results headline",
        "teach_point": (
            "Ask: revenue vs profit growth, one-time items, guidance, and peer context. "
            "A beat with weak cash conversion is not automatically high quality."
        ),
        "why_it_matters": (
            "Company news is micro education. Pair it with your scan/long-term coverage — "
            "missing fundamentals stay missing."
        ),
        "tags": ["earnings", "micro", "fundamentals"],
        "level": "intermediate",
    },
    {
        "id": "concept-fo-basics",
        "lens": "CONCEPT",
        "title": "F&O context without direction myths",
        "teach_point": (
            "OI, IV, PCR and max pain describe positioning and uncertainty. "
            "They do not tell you which way price must go next."
        ),
        "why_it_matters": (
            "QuantTerm’s F&O desk is evidence, not a directional tip. "
            "Education here stops at reading the board honestly."
        ),
        "tags": ["fno", "derivatives", "oi"],
        "level": "intermediate",
    },
    {
        "id": "concept-sebi-policy",
        "lens": "CONCEPT",
        "title": "Why SEBI / policy headlines matter",
        "teach_point": (
            "Regulation changes rules of the game — margins, disclosure, product design. "
            "Retail edge starts with knowing what changed, from the official source."
        ),
        "why_it_matters": (
            "Policy cards in this feed always prefer official-tier sources when available."
        ),
        "tags": ["sebi", "policy", "regulation"],
        "level": "beginner",
    },
    {
        "id": GOLD_LOAN_CONCEPT["id"],
        "lens": "CONCEPT",
        "title": GOLD_LOAN_CONCEPT["title"],
        "teach_point": GOLD_LOAN_CONCEPT["teach_point"],
        "why_it_matters": GOLD_LOAN_CONCEPT["why_it_matters"],
        "tags": ["gold", "nbfc", "collateral"],
        "level": "intermediate",
    },
    {
        "id": MIX_SHIFT_CONCEPT["id"],
        "lens": "CONCEPT",
        "title": MIX_SHIFT_CONCEPT["title"],
        "teach_point": MIX_SHIFT_CONCEPT["teach_point"],
        "why_it_matters": MIX_SHIFT_CONCEPT["why_it_matters"],
        "tags": ["mix", "margins", "speciality"],
        "level": "intermediate",
    },
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canon_lens(article: Mapping[str, Any]) -> str:
    text = " ".join(
        [
            str(article.get("category") or ""),
            str(article.get("event_type") or ""),
            " ".join(str(t) for t in (article.get("tags") or [])),
            str(article.get("headline") or ""),
        ]
    ).lower()
    if any(k in text for k in ("derivative", "future", "option", "f&o", "expiry", "oi ", "pcr")):
        return "DERIVATIVES"
    if any(k in text for k in ("sebi", "regulation", "policy", "tax", "government", "court", "circular")):
        return "POLICY"
    if any(
        k in text
        for k in (
            "result", "order", "contract", "promoter", "dividend", "merger",
            "acquisition", "company", "corporate", "earnings", "q1", "q2", "q3", "q4",
        )
    ):
        return "MICRO"
    if any(
        k in text
        for k in (
            "economy", "macro", "inflation", "gdp", "rate", "rbi", "currency",
            "bond", "fii", "fpi", "crude", "fed", "geopolit", "global",
        )
    ):
        return "MACRO"
    if article.get("mentioned_symbols"):
        return "MICRO"
    return "MACRO"


def _teach_from_article(article: Mapping[str, Any], lens: str) -> str:
    why = str(article.get("why_it_matters") or "").strip()
    if why:
        return why
    summary = str(article.get("summary") or "").strip()
    if summary:
        return summary[:280]
    headline = str(article.get("headline") or "Market update").strip()
    if lens == "MICRO":
        return f"Company/context headline to study with source evidence: {headline}"
    if lens == "POLICY":
        return f"Policy/regulatory development to read from the original filing: {headline}"
    if lens == "DERIVATIVES":
        return f"Derivatives-market context (positioning/uncertainty) — not a direction call: {headline}"
    return f"Macro tape context to understand with the original source: {headline}"


def _news_card(article: Mapping[str, Any]) -> dict[str, Any]:
    lens = _canon_lens(article)
    symbols = [str(s).upper() for s in (article.get("mentioned_symbols") or []) if str(s).strip()]
    fno = [str(s).upper() for s in (article.get("fno_symbols") or []) if str(s).strip()]
    return {
        "id": f"news-{article.get('article_id') or article.get('cluster_key') or hash(str(article.get('headline')))}",
        "lens": lens,
        "kind": "NEWS_LESSON",
        "title": str(article.get("headline") or "Untitled market item"),
        "teach_point": _teach_from_article(article, lens),
        "why_it_matters": str(article.get("why_it_matters") or article.get("summary") or ""),
        "summary": str(article.get("summary") or ""),
        "level": "current_events",
        "impact_score": int(article.get("impact_score") or 0),
        "direction": str(article.get("direction") or "unclear"),
        "category": str(article.get("category") or ""),
        "event_type": str(article.get("event_type") or ""),
        "source": str(article.get("source") or ""),
        "source_tier": int(article.get("source_tier") or 0),
        "official": bool(article.get("official")),
        "url": str(article.get("url") or ""),
        "published_at": str(article.get("published_at") or ""),
        "fetched_at": str(article.get("fetched_at") or ""),
        "symbols": symbols[:12],
        "fno_symbols": fno[:8],
        "sectors": [str(s) for s in (article.get("sectors") or [])][:8],
        "tags": [str(t) for t in (article.get("tags") or [])][:12],
        "corroboration_count": int(article.get("corroboration_count") or 1),
        "places_orders": False,
        "is_signal": False,
    }


def _macro_theme_card(theme: Mapping[str, Any]) -> dict[str, Any]:
    label = str(theme.get("label") or theme.get("name") or "Macro theme")
    direction = str(theme.get("direction") or "mixed")
    count = int(theme.get("count") or 0)
    sample = str(theme.get("sample") or "").strip()
    hit = ", ".join(str(s) for s in (theme.get("sectors_hit") or [])[:4]) or "broad market"
    help_ = ", ".join(str(s) for s in (theme.get("sectors_help") or [])[:3])
    teach = (
        f"{label} is corroborated across {count} recent headlines "
        f"(direction cue: {direction}). First-order watch: {hit}."
    )
    if help_:
        teach += f" Often discussed as relative support for: {help_}."
    teach += " This is a weather report for learning — not a trade call."
    return {
        "id": f"macro-{theme.get('name') or label}".lower().replace(" ", "-"),
        "lens": "MACRO",
        "kind": "MACRO_THEME",
        "title": f"Macro theme: {label}",
        "teach_point": teach,
        "why_it_matters": (
            sample
            and f"Sample headline in the cluster: {sample}"
            or "Corroborated keyword theme from the news stream."
        ),
        "summary": sample,
        "level": "current_events",
        "impact_score": min(100, count * 12),
        "direction": direction,
        "category": "macro",
        "event_type": str(theme.get("name") or "THEME"),
        "source": "macro_pulse (corroborated keywords)",
        "source_tier": 0,
        "official": False,
        "url": "",
        "published_at": "",
        "fetched_at": _now_iso(),
        "symbols": [],
        "fno_symbols": [],
        "sectors": list(theme.get("sectors_hit") or [])[:8],
        "tags": ["macro", str(theme.get("name") or "").lower()],
        "corroboration_count": count,
        "places_orders": False,
        "is_signal": False,
    }


def _concept_cards() -> list[dict[str, Any]]:
    out = []
    for row in _CONCEPT_CARDS:
        out.append({
            **row,
            "kind": "CONCEPT",
            "impact_score": 0,
            "direction": "unclear",
            "category": "education",
            "event_type": "CONCEPT",
            "source": "QuantTerm concept library",
            "source_tier": 0,
            "official": False,
            "url": "",
            "published_at": "",
            "fetched_at": "",
            "symbols": [],
            "fno_symbols": [],
            "sectors": [],
            "summary": row["teach_point"],
            "corroboration_count": 0,
            "places_orders": False,
            "is_signal": False,
        })
    return out


def build_education_feed(
    *,
    articles: Sequence[Mapping[str, Any]] | None = None,
    macro_themes: Sequence[Mapping[str, Any]] | None = None,
    include_concepts: bool = True,
    min_impact: int = 40,
    limit: int = 40,
) -> dict[str, Any]:
    """Project learnable cards from curated news + macro themes + concepts."""
    articles = list(articles or [])
    news_cards = []
    for article in articles:
        if not isinstance(article, Mapping):
            continue
        if not str(article.get("headline") or "").strip():
            continue
        try:
            impact = int(article.get("impact_score") or 0)
        except (TypeError, ValueError):
            impact = 0
        # Keep a lower bar for education than "Important Now", but skip empty noise.
        if impact < int(min_impact) and not article.get("official"):
            continue
        news_cards.append(_news_card(article))

    if macro_themes is None:
        try:
            from core.macro_pulse import detect_macro_themes

            macro_themes = detect_macro_themes([dict(a) for a in articles])
        except Exception:
            macro_themes = []

    theme_cards = [_macro_theme_card(t) for t in (macro_themes or []) if isinstance(t, Mapping)]
    concepts = _concept_cards() if include_concepts else []

    # Rank: official / high-impact / corroboration first.
    news_cards.sort(
        key=lambda c: (
            int(bool(c.get("official"))),
            int(c.get("impact_score") or 0),
            int(c.get("corroboration_count") or 0),
            str(c.get("published_at") or ""),
        ),
        reverse=True,
    )

    combined = news_cards[: max(0, int(limit))] + theme_cards + concepts
    by_lens: dict[str, int] = {}
    for card in combined:
        lens = str(card.get("lens") or "OTHER")
        by_lens[lens] = by_lens.get(lens, 0) + 1

    return {
        "schema_version": 1,
        "generated_at": _now_iso(),
        "available": bool(news_cards or theme_cards or concepts),
        "honesty": (
            "Educational cards crunch dated news and fixed concept teach-ins. "
            "QuantTerm never invents articles or blogs. Learning context is not a trade signal."
        ),
        "places_orders": False,
        "summary": {
            "news_lessons": len(news_cards),
            "macro_themes": len(theme_cards),
            "concepts": len(concepts),
            "by_lens": by_lens,
            "articles_considered": len(articles),
        },
        "lenses": ["MACRO", "MICRO", "POLICY", "DERIVATIVES", "CONCEPT"],
        "cards": combined,
        "empty_hint": (
            None
            if news_cards
            else "No high-enough-impact curated news yet — refresh News & Events, then reopen Education."
        ),
    }


def education_from_news_payload(news: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Convenience: build feed from terminal `_news_payload()` shape."""
    news = dict(news or {})
    articles = list(news.get("articles") or [])
    return build_education_feed(articles=articles)
