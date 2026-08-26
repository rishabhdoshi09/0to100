"""Material news/events from the curator store. No invented headlines."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from product.due_diligence.materiality import classify_taxonomy, materiality

_MATERIAL_EVENTS = frozenset({
    "order_or_contract", "merger_or_acquisition", "regulatory",
    "promoter_or_insider", "rating", "fund_raising",
})
_RED_TOKENS = (
    "investigation", "raid", "fraud", "auditor resign", "warning letter",
    "usfda", "import alert", "pledge increase", "downgrade", "penalty",
    "show cause", "sebi ban", "rbi ban", "litigation", "default",
)
_BROKER_NOISE = ("picks ", "target price", "upside", "overweight", "initiate")


def _headline(article: Mapping[str, Any]) -> str:
    return str(article.get("headline") or "").strip()


def is_material(article: Mapping[str, Any], symbol: str) -> bool:
    text = _headline(article).lower()
    mentioned = [str(s).upper() for s in (article.get("mentioned_symbols") or [])]
    want = str(symbol or "").upper()
    if mentioned and want and want not in mentioned:
        return False
    if any(tok in text for tok in _BROKER_NOISE) and not article.get("official"):
        if len(mentioned) > 2:
            return False
    if any(tok in text for tok in _RED_TOKENS):
        return True
    if bool(article.get("official")):
        return True
    event = str(article.get("event_type") or "")
    impact = int(article.get("impact_score") or 0)
    if event in _MATERIAL_EVENTS and impact >= 50:
        return True
    if event == "results" and impact >= 70 and bool(article.get("official")):
        return True
    return False


def event_impact(article: Mapping[str, Any]) -> str:
    direction = str(article.get("direction") or "").lower()
    text = _headline(article).lower()
    if any(tok in text for tok in _RED_TOKENS) or direction in {"negative", "bearish"}:
        return "negative"
    if direction in {"positive", "bullish"}:
        return "positive"
    return "neutral"


def classify_event(article: Mapping[str, Any]) -> str:
    event = str(article.get("event_type") or "company").strip() or "company"
    text = _headline(article).lower()
    if any(tok in text for tok in ("usfda", "warning letter", "import alert")):
        return "regulatory_action"
    if "pledge" in text:
        return "pledge"
    if "auditor" in text:
        return "governance"
    return event


def material_events(
    articles: Sequence[Mapping[str, Any]],
    symbol: str,
    *,
    limit: int = 12,
    context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    ctx = dict(context or {})
    out: list[dict[str, Any]] = []
    for article in articles:
        if not isinstance(article, Mapping):
            continue
        if not is_material(article, symbol):
            continue
        impact = event_impact(article)
        event_type = classify_event(article)
        meta = materiality(
            {**dict(article), "event_type": event_type, "headline": _headline(article)},
            revenue_cr=ctx.get("revenue_cr"),
            market_cap_cr=ctx.get("market_cap_cr"),
            pat_cr=ctx.get("pat_cr"),
            debt_cr=ctx.get("debt_cr"),
            promoter_pct=ctx.get("promoter_pct"),
        )
        out.append({
            "headline": _headline(article) or "Data unavailable",
            "published_at": str(article.get("published_at") or article.get("fetched_at") or ""),
            "source": str(article.get("source") or article.get("source_name") or ""),
            "url": str(article.get("url") or ""),
            "official": bool(article.get("official")),
            "verified": bool(article.get("official")),
            "event_type": event_type,
            "category": meta["category"] or classify_taxonomy(_headline(article), event_type),
            "impact": impact,
            "impact_score": int(article.get("impact_score") or 0),
            "summary": str(article.get("summary") or article.get("why_it_matters") or ""),
            "material": True,
            "materiality": meta["materiality"],
            "materiality_basis": meta["basis"],
            "amount_cr": meta.get("amount_cr"),
            "original_source": meta.get("original_source"),
            "thesis_change": impact == "negative" or (
                event_type in {"regulatory_action", "pledge", "governance"}
            ),
        })
        if len(out) >= limit:
            break
    return out


def news_verdict(events: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    if not events:
        return "Neutral", "No material company-tagged development on file."
    negatives = [e for e in events if e.get("impact") == "negative"]
    positives = [e for e in events if e.get("impact") == "positive"]
    if negatives and not positives:
        return "Negative", negatives[0]["headline"]
    if positives and not negatives:
        return "Neutral to Positive", positives[0]["headline"]
    if negatives and positives:
        return "Mixed", "Both supportive and adverse material items are on file."
    return "Neutral", "Material items are present but direction is not signed."
