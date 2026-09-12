"""The one place a recommendation card becomes a canonical Decision.

The pipeline still produces cards: dictionaries with whatever keys the stage
that built them happened to use. This module is the single crossing point where
that becomes a :class:`~product.decision.Decision`, so every downstream
consumer — API, UI, paper intent, ledger, learning — reads one shape instead of
each re-deriving its own reading of the card.

One adapter, not one per consumer. The moment a second subsystem starts
interpreting card keys itself, the two interpretations can disagree about what
the desk decided, and nothing will notice.

The mapping is deliberately conservative: a card field the adapter cannot
interpret becomes MISSING evidence rather than a neutral default, because a
neutral default is a quiet claim that the system checked and found nothing
wrong.
"""
from __future__ import annotations

from typing import Any, Mapping

from product.decision import (
    AVOID,
    BUY,
    CONFLICTING,
    Decision,
    EvidenceItem,
    MISSING,
    NO_TRADE,
    SUPPORTING,
    WAIT,
    WATCH,
)

# Card tiers, in the vocabulary the reco pipeline emits.
_TIER_TO_STATE = {
    "high_conviction": BUY,
    "good_setup": BUY,
    "watch": WATCH,
    "watchlist": WATCH,
    "wait": WAIT,
    "avoid": AVOID,
    "no_trade": NO_TRADE,
}

# reco_methods emits lowercase pass/fail/unknown; evidence_families emits
# SUPPORTIVE/NEUTRAL/OPPOSED/UNKNOWN. Both land in one direction vocabulary.
_STATUS_TO_DIRECTION = {
    "pass": SUPPORTING,
    "supportive": SUPPORTING,
    "ok": SUPPORTING,
    "confirmed": SUPPORTING,
    "fail": CONFLICTING,
    "opposed": CONFLICTING,
    "blocked": CONFLICTING,
    "unknown": MISSING,
    "unavailable": MISSING,
    "unmeasured": MISSING,
    "unproven": MISSING,
    "": MISSING,
}

#: NEUTRAL is not evidence in either direction. It is recorded as missing
#: because "we looked and it said nothing" is a gap, not a supporting fact.
_NEUTRAL = {"neutral"}


def _f(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if out != out else out


def _direction(status: Any) -> str:
    text = str(status or "").strip().lower()
    if text in _NEUTRAL:
        return MISSING
    return _STATUS_TO_DIRECTION.get(text, MISSING)


def _state(card: Mapping[str, Any]) -> str:
    tier = str(card.get("reco_tier") or "").strip().lower()
    if tier in _TIER_TO_STATE:
        state = _TIER_TO_STATE[tier]
        if state == BUY and not card.get("allows_recommend", True):
            # The card scored well but a hard gate refused it. The gate wins.
            return WATCH
        return state
    badge = str(card.get("action_badge") or "").strip().upper()
    if badge in {BUY, WATCH, WAIT, AVOID, NO_TRADE}:
        return badge
    return WAIT


def _evidence(card: Mapping[str, Any], as_of: str) -> dict[str, list[EvidenceItem]]:
    buckets: dict[str, list[EvidenceItem]] = {SUPPORTING: [], CONFLICTING: [], MISSING: []}

    for method in card.get("methods") or []:
        if not isinstance(method, Mapping):
            continue
        ident = str(method.get("id") or "")
        if not ident:
            continue
        direction = _direction(method.get("status"))
        buckets[direction].append(EvidenceItem(
            id=f"method:{ident}",
            label=str(method.get("label") or ident),
            direction=direction,
            detail=str(method.get("detail") or ""),
            value=method.get("points"),
            source="reco_methods",
            as_of=as_of,
        ))

    for family in card.get("families") or []:
        if not isinstance(family, Mapping):
            continue
        ident = str(family.get("id") or "")
        if not ident:
            continue
        direction = _direction(family.get("status"))
        buckets[direction].append(EvidenceItem(
            id=f"family:{ident}",
            label=str(family.get("label") or ident),
            direction=direction,
            detail=str(family.get("status") or ""),
            source="evidence_families",
            as_of=as_of,
        ))

    for conflict in card.get("conflicts") or []:
        text = str(conflict or "").strip()
        if text:
            buckets[CONFLICTING].append(EvidenceItem(
                id=f"conflict:{text[:40]}", label=text, direction=CONFLICTING,
                source="reco_ensemble", as_of=as_of,
            ))

    dd = str(card.get("dd_status") or card.get("dd_verdict") or "").strip()
    if not dd or dd.lower() in {"unknown", "unavailable", "unmeasured"}:
        buckets[MISSING].append(EvidenceItem(
            id="fundamentals", label="Fundamentals not available",
            direction=MISSING, source="due_diligence", as_of=as_of,
        ))
    else:
        direction = _direction(dd)
        buckets[direction].append(EvidenceItem(
            id="fundamentals", label=f"Due diligence: {dd}", direction=direction,
            source="due_diligence", as_of=as_of,
        ))

    return buckets


def decision_from_card(
    card: Mapping[str, Any],
    *,
    source_scan_id: str = "",
    market_state: str = "",
    sector_state: str = "",
    evidence_snapshot_id: str = "",
    evidence_class: str = "",
    decision_engine_version: str = "",
    feature_schema_version: str = "",
    strategy_version: str = "",
    generated_at: str = "",
) -> Decision:
    """Build the canonical Decision this card represents."""
    as_of = str(card.get("scan_scanned_at") or "")
    buckets = _evidence(card, as_of)
    entry = _f(card.get("entry"))
    stop = _f(card.get("stop"))
    target = _f(card.get("target"))

    invalidation: list[str] = []
    if stop is not None:
        invalidation.append(f"Closes below the stop at {stop:g}")
    for item in card.get("invalidation") or card.get("what_would_change_our_mind") or []:
        text = str(item or "").strip()
        if text:
            invalidation.append(text)

    return Decision(
        symbol=str(card.get("symbol") or ""),
        state=_state(card),
        setup=str(card.get("primary_thesis") or card.get("setup_label") or ""),
        score=_f(card.get("score") or card.get("conviction")),
        calibrated_confidence=_f(card.get("calibrated_confidence")),
        expected_value=_f(card.get("expected_value") or card.get("ev_pct")),
        market_state=market_state or str(card.get("market_state") or ""),
        sector_state=sector_state or str(card.get("sector_state") or ""),
        technical_evidence={
            "atr_pct": _f(card.get("atr_pct")),
            "pct_from_pivot": _f(card.get("pct_from_pivot")),
            "rsi": _f(card.get("rsi")),
            "entry_state": card.get("entry_state"),
            "timing": card.get("timing"),
        },
        fundamental_evidence={"dd_status": card.get("dd_status") or card.get("dd_verdict") or ""},
        news_evidence={"buzz": card.get("news_buzz")} if card.get("news_buzz") is not None else {},
        historical_evidence={"stock_quality": card.get("stock_quality")},
        supporting_evidence=tuple(buckets[SUPPORTING]),
        conflicting_evidence=tuple(buckets[CONFLICTING]),
        missing_evidence=tuple(buckets[MISSING]),
        entry=entry,
        stop=stop,
        target=target,
        chase_risk="CHASE" if card.get("chase_risk") else "",
        liquidity_state=str(card.get("liquidity_state") or ""),
        portfolio_effect=dict(card.get("portfolio_effect") or {}),
        position_size=dict(card.get("position_size") or {}),
        invalidation_conditions=tuple(invalidation),
        source_scan_id=source_scan_id or as_of,
        evidence_snapshot_id=evidence_snapshot_id,
        decision_engine_version=decision_engine_version,
        feature_schema_version=feature_schema_version,
        strategy_version=strategy_version,
        evidence_class=evidence_class,
        provenance={"built_from": "recommendation_card", "scan_scanned_at": as_of},
        generated_at=generated_at or as_of or "",
    )


def decisions_from_cards(cards, **kwargs) -> list[Decision]:
    out: list[Decision] = []
    for card in cards or []:
        if not isinstance(card, Mapping) or not str(card.get("symbol") or "").strip():
            continue
        out.append(decision_from_card(card, **kwargs))
    return out
