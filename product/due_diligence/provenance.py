"""Typed provenance for every research field. Never drop the source."""
from __future__ import annotations

from typing import Any, Mapping

# Highest trust first. A newspaper article must not override an official filing.
SOURCE_TRUST: dict[str, int] = {
    "exchange_filing": 100,
    "regulator": 100,
    "company_ir": 92,
    "investor_presentation": 90,
    "annual_report": 90,
    "rating_agency": 85,
    "established_aggregator": 70,
    "financial_media": 45,
    "unknown": 20,
}

SOURCE_TYPE_LABEL: dict[str, str] = {
    "exchange_filing": "Exchange / company filing",
    "regulator": "Regulator",
    "company_ir": "Company investor relations",
    "investor_presentation": "Investor presentation",
    "annual_report": "Annual report",
    "rating_agency": "Credit-rating agency",
    "established_aggregator": "Established financial-data provider",
    "financial_media": "Financial media",
    "unknown": "Unclassified source",
}


def classify_source_type(source: str, source_url: str = "") -> str:
    blob = f"{source} {source_url}".lower()
    if any(tok in blob for tok in ("nseindia", "bseindia", "exchange filing", "corporate announcement")):
        return "exchange_filing"
    if any(tok in blob for tok in ("sebi", "rbi.org", "rbi ", "mca.gov")):
        return "regulator"
    if "investor presentation" in blob or "/ir" in blob:
        return "investor_presentation"
    if "annual report" in blob:
        return "annual_report"
    if any(tok in blob for tok in ("crisil", "icra", "care rating", "india ratings")):
        return "rating_agency"
    if any(tok in blob for tok in ("screener.in", "screener cache", "moneycontrol", "trendlyne")):
        return "established_aggregator"
    if any(tok in blob for tok in ("economic times", "business standard", "reuters", "mint", "cnbc")):
        return "financial_media"
    if "filing" in blob or "nse" in blob:
        return "exchange_filing"
    return "unknown"


def provenance(
    *,
    value: Any,
    period: str = "",
    source: str = "",
    source_url: str = "",
    retrieved_at: str = "",
    published_at: str = "",
    source_type: str = "",
    confidence: str = "low",
    raw_reference: str = "",
) -> dict[str, Any]:
    kind = source_type or classify_source_type(source, source_url)
    return {
        "value": value,
        "period": period or "Data unavailable",
        "source": source or "Source unavailable",
        "source_url": source_url,
        "retrieved_at": retrieved_at or "Data unavailable",
        "published_at": published_at or period or "Data unavailable",
        "source_type": kind,
        "source_type_label": SOURCE_TYPE_LABEL.get(kind, SOURCE_TYPE_LABEL["unknown"]),
        "trust": SOURCE_TRUST.get(kind, SOURCE_TRUST["unknown"]),
        "confidence": confidence,
        "raw_reference": raw_reference,
    }


def unavailable_provenance(*, source: str = "", reason: str = "Data unavailable") -> dict[str, Any]:
    return provenance(
        value=None,
        source=source or "Source unavailable",
        confidence="none",
        raw_reference=reason,
    )


def material_disagreement(a: Any, b: Any, *, kind: str = "level", pct: float = 1.5, pts: float = 0.3) -> bool:
    """True when two numeric prints disagree enough to surface as a conflict."""
    try:
        left = float(a)
        right = float(b)
    except (TypeError, ValueError):
        return str(a).strip() != str(b).strip() and a not in (None, "") and b not in (None, "")
    if left == right:
        return False
    if kind == "rate":
        return abs(left - right) >= pts
    base = max(abs(left), abs(right), 1e-9)
    return abs(left - right) / base * 100.0 >= pct


def resolve_fact(
    *,
    official: Mapping[str, Any] | None = None,
    secondary: Mapping[str, Any] | None = None,
    last_good: Mapping[str, Any] | None = None,
    on_file: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Official filing → reputable secondary → persisted last-good → on-file tables.

    A missing value stays None. Zero is kept only when a source actually printed 0.
    Last-good is labelled stale and is never treated as a live print.
    """

    def _pack(raw: Mapping[str, Any] | None, *, tier: str, stale: bool = False) -> dict[str, Any] | None:
        if not isinstance(raw, Mapping):
            return None
        if "value" not in raw and not raw:
            return None
        value = raw.get("value") if "value" in raw else None
        if value is None or value == "":
            return None
        row = provenance(
            value=value,
            period=str(raw.get("period") or ""),
            source=str(raw.get("source") or tier),
            source_url=str(raw.get("source_url") or ""),
            retrieved_at=str(raw.get("retrieved_at") or ""),
            published_at=str(raw.get("published_at") or raw.get("period") or ""),
            source_type=str(raw.get("source_type") or ""),
            confidence=str(raw.get("confidence") or ("high" if tier.startswith("official") else "medium")),
            raw_reference=str(raw.get("raw_reference") or tier),
        )
        row["tier"] = tier
        row["stale"] = bool(stale or raw.get("stale"))
        if row["stale"]:
            row["confidence"] = "low"
            row["source_label"] = "last_good_snapshot"
        return row

    for raw, tier, stale in (
        (official, "official_exchange_or_filing", False),
        (secondary, "reputable_public_secondary", False),
        (last_good, "persisted_last_good", True),
        (on_file, "tables_on_file", False),
    ):
        packed = _pack(raw, tier=tier, stale=stale)
        if packed is not None:
            return packed
    return unavailable_provenance(reason="No official, secondary, last-good, or on-file print")


def conflict_record(
    field: str,
    preferred: Mapping[str, Any],
    other: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "field": field,
        "status": "Source discrepancy detected",
        "preferred": dict(preferred),
        "other": dict(other),
        "note": (
            "Authoritative source is retained for scoring. "
            "The conflicting print is kept for investigation — it is not silently replaced."
        ),
    }
