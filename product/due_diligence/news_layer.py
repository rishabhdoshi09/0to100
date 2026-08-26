"""Material news/events from the curator store. No invented headlines."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from product.due_diligence.materiality import classify_taxonomy, materiality

_MATERIAL_EVENTS = frozenset({
    "order_or_contract", "merger_or_acquisition", "regulatory",
    "promoter_or_insider", "rating", "fund_raising",
})
# Headlines that are material only for named frameworks. A USFDA letter is
# not a bank event; an RBI PCA item is not a pharma event.
_LOCKED_TOKENS: tuple[tuple[str, frozenset[str]], ...] = (
    ("usfda", frozenset({"pharma"})),
    ("import alert", frozenset({"pharma"})),
    ("form 483", frozenset({"pharma"})),
    ("anda", frozenset({"pharma"})),
    ("prompt corrective", frozenset({"bank", "nbfc", "nbfc_gold", "nbfc_housing"})),
)
_GLOBAL_RED = (
    "investigation", "raid", "fraud", "auditor resign",
    "pledge increase", "downgrade", "penalty",
    "show cause", "sebi ban", "rbi ban", "litigation", "default",
)
_BROKER_NOISE = ("picks ", "target price", "upside", "overweight", "initiate")


def _headline(article: Mapping[str, Any]) -> str:
    return str(article.get("headline") or "").strip()


def _framework_tokens(framework_id: str) -> tuple[str, ...]:
    if not framework_id:
        return ()
    try:
        from product.due_diligence.frameworks import get_framework
        return tuple(get_framework(framework_id).get("material_tokens") or ())
    except Exception:
        return ()


def is_material(article: Mapping[str, Any], symbol: str, *, framework_id: str = "") -> bool:
    text = _headline(article).lower()
    mentioned = [str(s).upper() for s in (article.get("mentioned_symbols") or [])]
    want = str(symbol or "").upper()
    if mentioned and want and want not in mentioned:
        return False
    if framework_id:
        for token, allowed in _LOCKED_TOKENS:
            if token in text and framework_id not in allowed:
                return False
    if any(tok in text for tok in _BROKER_NOISE) and not article.get("official"):
        if len(mentioned) > 2:
            return False
    if any(tok in text for tok in _GLOBAL_RED):
        return True
    sector_tokens = _framework_tokens(framework_id)
    if sector_tokens and any(tok in text for tok in sector_tokens):
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


def event_impact(article: Mapping[str, Any], *, framework_id: str = "") -> str:
    direction = str(article.get("direction") or "").lower()
    text = _headline(article).lower()
    tokens = _GLOBAL_RED + _framework_tokens(framework_id)
    if any(tok in text for tok in tokens) or direction in {"negative", "bearish"}:
        return "negative"
    if direction in {"positive", "bullish"}:
        return "positive"
    return "neutral"


def classify_event(article: Mapping[str, Any]) -> str:
    event = str(article.get("event_type") or "company").strip() or "company"
    text = _headline(article).lower()
    if any(tok in text for tok in ("usfda", "warning letter", "import alert", "form 483")):
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
    framework_id: str = "",
) -> list[dict[str, Any]]:
    ctx = dict(context or {})
    out: list[dict[str, Any]] = []
    for article in articles:
        if not isinstance(article, Mapping):
            continue
        if not is_material(article, symbol, framework_id=framework_id):
            continue
        impact = event_impact(article, framework_id=framework_id)
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
