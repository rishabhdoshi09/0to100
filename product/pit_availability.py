"""Point-in-time availability contract for historical replay.

The important date is not the financial period label (FY2025). It is when
that information was publicly available. If availability cannot be proven,
the item is PIT_UNVERIFIED and must not silently enter a historical decision.
"""
from __future__ import annotations

from datetime import datetime, date
from typing import Any, Mapping, Sequence

PIT_STRONG = "PIT_STRONG"
PIT_PARTIAL = "PIT_PARTIAL"
PIT_MARKET_ONLY = "PIT_MARKET_ONLY"
PIT_UNAVAILABLE = "PIT_UNAVAILABLE"
PIT_UNVERIFIED = "PIT_UNVERIFIED"

# Adversarial / contract outcomes
AVAILABLE_AT_T = "AVAILABLE_AT_T"
UNAVAILABLE_AT_T = "UNAVAILABLE_AT_T"
UNPROVEN_AT_T = "UNPROVEN_AT_T"


def _parse_day(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def available_to_engine_at_t(
    *,
    as_of: Any,
    period_end: Any = None,
    publication_date: Any = None,
    filing_date: Any = None,
    acquired_at: Any = None,
    source_date: Any = None,
) -> dict[str, Any]:
    """Best truthful availability model supported by actual metadata.

    Priority for *public* availability:
      1. publication_date / filing_date
      2. source_date only if it is explicitly a publication stamp
      3. acquired_at (engine saw it — still not proof of public date)
      4. period_end alone is NEVER sufficient
    """
    t = _parse_day(as_of)
    published = _parse_day(publication_date) or _parse_day(filing_date)
    acquired = _parse_day(acquired_at)
    period = _parse_day(period_end)
    source = _parse_day(source_date)

    if t is None:
        return {
            "as_of": None,
            "period_end": period.isoformat() if period else None,
            "publication_date": published.isoformat() if published else None,
            "acquired_at": acquired.isoformat() if acquired else None,
            "available_to_engine_at_T": False,
            "verdict": UNPROVEN_AT_T,
            "pit_status": PIT_UNVERIFIED,
            "reason": "decision date missing — cannot prove availability",
        }

    if published is not None:
        ok = published <= t
        return {
            "as_of": t.isoformat(),
            "period_end": period.isoformat() if period else None,
            "publication_date": published.isoformat(),
            "acquired_at": acquired.isoformat() if acquired else None,
            "available_to_engine_at_T": ok,
            "verdict": AVAILABLE_AT_T if ok else UNAVAILABLE_AT_T,
            "pit_status": PIT_STRONG if ok else PIT_UNAVAILABLE,
            "reason": (
                "publication/filing on or before T" if ok
                else f"published {published.isoformat()} after decision {t.isoformat()}"
            ),
        }

    if acquired is not None:
        ok = acquired <= t
        return {
            "as_of": t.isoformat(),
            "period_end": period.isoformat() if period else None,
            "publication_date": None,
            "acquired_at": acquired.isoformat(),
            "available_to_engine_at_T": ok,
            "verdict": AVAILABLE_AT_T if ok else UNAVAILABLE_AT_T,
            "pit_status": PIT_PARTIAL if ok else PIT_UNAVAILABLE,
            "reason": (
                "acquired_at <= T but public publication_date unproven"
                if ok else "acquired after decision T"
            ),
        }

    if source is not None and period is not None and source == period:
        # Period label reused as a date is not a publication proof.
        return {
            "as_of": t.isoformat(),
            "period_end": period.isoformat(),
            "publication_date": None,
            "acquired_at": None,
            "available_to_engine_at_T": False,
            "verdict": UNPROVEN_AT_T,
            "pit_status": PIT_UNVERIFIED,
            "reason": "source_date equals period_end — not a publication proof",
        }

    if period is not None and published is None:
        return {
            "as_of": t.isoformat(),
            "period_end": period.isoformat(),
            "publication_date": None,
            "acquired_at": None,
            "available_to_engine_at_T": False,
            "verdict": UNPROVEN_AT_T,
            "pit_status": PIT_UNVERIFIED,
            "reason": "only period_end is known — FY label is not availability",
        }

    return {
        "as_of": t.isoformat(),
        "period_end": period.isoformat() if period else None,
        "publication_date": None,
        "acquired_at": None,
        "available_to_engine_at_T": False,
        "verdict": UNPROVEN_AT_T,
        "pit_status": PIT_UNVERIFIED,
        "reason": "no publication_date, filing_date, or acquired_at",
    }


def reject_if_future(*, as_of: Any, evidence_date: Any, kind: str = "evidence") -> dict[str, Any]:
    """Adversarial helper: future stamps are refused, never consumed."""
    t = _parse_day(as_of)
    ev = _parse_day(evidence_date)
    if t is None or ev is None:
        return {"accepted": False, "reason": f"{kind} date unproven", "pit_status": PIT_UNVERIFIED}
    if ev > t:
        return {
            "accepted": False,
            "reason": f"future {kind} {ev.isoformat()} after {t.isoformat()}",
            "pit_status": PIT_UNAVAILABLE,
        }
    return {"accepted": True, "reason": f"{kind} on or before T", "pit_status": PIT_STRONG}


def filter_items_as_of(items: Sequence[Mapping[str, Any]], as_of: Any) -> dict[str, Any]:
    """Keep only items proven available at T. Unproven items are excluded."""
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    unverified = 0
    for raw in items or []:
        if not isinstance(raw, Mapping):
            continue
        item = dict(raw)
        check = available_to_engine_at_t(
            as_of=as_of,
            period_end=item.get("period_end") or item.get("period"),
            publication_date=item.get("publication_date") or item.get("published_at"),
            filing_date=item.get("filing_date"),
            acquired_at=item.get("acquired_at"),
            source_date=item.get("source_date") or item.get("source_date_raw"),
        )
        item["pit"] = check
        if check["verdict"] == AVAILABLE_AT_T:
            kept.append(item)
        else:
            if check["verdict"] == UNPROVEN_AT_T:
                unverified += 1
            excluded.append(item)
    return {
        "as_of": str(as_of)[:10],
        "kept": kept,
        "excluded": excluded,
        "n_kept": len(kept),
        "n_excluded": len(excluded),
        "n_unverified": unverified,
    }


def grade_replay(
    *,
    as_of: Any,
    market_bars_ok: bool,
    company_items: Sequence[Mapping[str, Any]] | None = None,
    used_today_fundamentals: bool = False,
    used_today_research: bool = False,
    used_future_bar: bool = False,
) -> dict[str, Any]:
    """Classify a historical decision's evidence quality. Never fake completeness."""
    if used_future_bar or used_today_fundamentals or used_today_research:
        return {
            "as_of": str(as_of)[:10],
            "grade": PIT_UNAVAILABLE,
            "reason": "lookahead or current-world evidence entered the path",
            "used_today_fundamentals": used_today_fundamentals,
            "used_today_research": used_today_research,
            "used_future_bar": used_future_bar,
            "comparable_to_forward": False,
        }
    if not market_bars_ok:
        return {
            "as_of": str(as_of)[:10],
            "grade": PIT_UNAVAILABLE,
            "reason": "PIT market bars unavailable",
            "comparable_to_forward": False,
        }
    filtered = filter_items_as_of(company_items or [], as_of)
    if filtered["n_kept"] <= 0 and filtered["n_unverified"] <= 0 and not (company_items or []):
        return {
            "as_of": str(as_of)[:10],
            "grade": PIT_MARKET_ONLY,
            "reason": "OHLCV/regime only — company evidence not supplied at T",
            "company": filtered,
            "comparable_to_forward": False,
        }
    if filtered["n_kept"] <= 0:
        return {
            "as_of": str(as_of)[:10],
            "grade": PIT_UNVERIFIED if filtered["n_unverified"] else PIT_MARKET_ONLY,
            "reason": "company evidence present but availability unproven or post-T",
            "company": filtered,
            "comparable_to_forward": False,
        }
    if filtered["n_unverified"] or filtered["n_excluded"]:
        return {
            "as_of": str(as_of)[:10],
            "grade": PIT_PARTIAL,
            "reason": "some company evidence proven at T; remainder excluded",
            "company": filtered,
            "comparable_to_forward": False,
        }
    return {
        "as_of": str(as_of)[:10],
        "grade": PIT_STRONG,
        "reason": "market bars and company evidence proven available at T",
        "company": filtered,
        "comparable_to_forward": True,
    }
