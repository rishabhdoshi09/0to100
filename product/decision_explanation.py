"""Why this decision — rendered from the decision, not narrated about it.

The user-facing answer to "why is this a BUY?" must be derivable from the
:class:`~product.decision.Decision` alone. That rules out the tempting shortcut:
asking a language model to look at the numbers and write a paragraph. A
paragraph is not checkable, does not fail when the evidence is missing, and
reads exactly as confident when the system knows nothing as when it knows a
great deal.

So the explanation is a structure. Each section either has a value or says
plainly that it does not. A model may narrate this structure for readability,
but the structure is the authority and the UI renders it directly.

The sections are fixed, and a section with nothing in it is still rendered —
"we could not obtain the fundamentals" is one of the more useful things this
screen can say, and dropping empty sections would hide it.
"""
from __future__ import annotations

from typing import Any, Mapping

from product.decision import CONFLICTING, Decision, EvidenceItem, MISSING, SUPPORTING

#: Marks the explanation as computed, so a consumer can never mistake a
#: generated narration for the authority.
AUTHORITY = "DETERMINISTIC_FROM_DECISION"

UNAVAILABLE = "Not available"


def _items(items: tuple[EvidenceItem, ...]) -> list[dict[str, Any]]:
    return [
        {
            "id": item.id,
            "label": item.label or item.id,
            "detail": item.detail,
            "value": item.value,
            "source": item.source,
            "as_of": item.as_of,
            "evidence_class": item.evidence_class,
        }
        for item in items
    ]


def _section(title: str, *, value: Any = None, text: str = "",
             items: list[dict[str, Any]] | None = None,
             fields: Mapping[str, Any] | None = None) -> dict[str, Any]:
    present = bool(
        items
        or (fields and any(v not in (None, "", [], {}) for v in fields.values()))
        or text
        or value not in (None, "")
    )
    return {
        "title": title,
        "available": present,
        "value": value,
        "text": text or ("" if present else UNAVAILABLE),
        "items": items or [],
        "fields": dict(fields or {}),
    }


def _headline(decision: Decision) -> str:
    """One sentence, assembled from facts, never from adjectives."""
    supporting = len(decision.supporting_evidence)
    conflicting = len(decision.conflicting_evidence)
    missing = len(decision.missing_evidence)
    setup = decision.setup or "no named setup"
    parts = [f"{decision.symbol}: {decision.state} on {setup}"]
    if decision.score is not None:
        parts.append(f"score {decision.score:g}")
    parts.append(
        f"{supporting} supporting, {conflicting} conflicting, {missing} missing"
    )
    return " · ".join(parts)


def _risk_fields(decision: Decision) -> dict[str, Any]:
    risk = decision.risk_per_share
    return {
        "risk_per_share": risk,
        "chase_risk": decision.chase_risk,
        "liquidity_state": decision.liquidity_state,
        "expected_R_from_levels": decision.computed_expected_R,
        "expected_R_as_published": decision.expected_R,
    }


def _freshness(decision: Decision) -> dict[str, Any]:
    """The oldest thing this decision is standing on.

    A decision is exactly as fresh as its stalest input, so the screen shows
    the stalest one rather than the newest.
    """
    stamps = [
        (item.as_of, item.label or item.id)
        for group in (decision.supporting_evidence, decision.conflicting_evidence)
        for item in group
        if item.as_of
    ]
    oldest = min(stamps, default=None, key=lambda pair: pair[0])
    return {
        "decision_generated_at": decision.generated_at,
        "oldest_input_as_of": oldest[0] if oldest else "",
        "oldest_input_label": oldest[1] if oldest else "",
        "evidence_snapshot_id": decision.evidence_snapshot_id,
        "source_scan_id": decision.source_scan_id,
    }


def _sources(decision: Decision) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for group in (decision.supporting_evidence, decision.conflicting_evidence,
                  decision.missing_evidence):
        for item in group:
            name = item.source or "unattributed"
            row = seen.setdefault(name, {"source": name, "as_of": item.as_of,
                                         "evidence_class": item.evidence_class,
                                         "used_for": []})
            row["used_for"].append(item.label or item.id)
            if item.as_of and (not row["as_of"] or item.as_of < row["as_of"]):
                row["as_of"] = item.as_of
    return [seen[name] for name in sorted(seen)]


def explain(decision: Decision) -> dict[str, Any]:
    """The full WHY THIS DECISION payload, in the order the screen shows it."""
    return {
        "authority": AUTHORITY,
        "note": (
            "Rendered from the decision record. A generated summary may sit "
            "beside this, never in place of it."
        ),
        "decision_id": decision.decision_id,
        "symbol": decision.symbol,
        "state": decision.state,
        "headline": _headline(decision),
        "sections": [
            _section("Supporting evidence", items=_items(decision.supporting_evidence)),
            _section("Conflicting evidence", items=_items(decision.conflicting_evidence)),
            _section("Missing evidence", items=_items(decision.missing_evidence)),
            _section("Market context", value=decision.market_state),
            _section("Sector context", value=decision.sector_state),
            _section("Setup", value=decision.setup),
            _section("Risk", fields=_risk_fields(decision)),
            _section("Entry", value=decision.entry),
            _section("Stop", value=decision.stop),
            _section("Target", value=decision.target),
            _section("Invalidation", items=[
                {"id": f"invalidation_{i}", "label": condition, "detail": "",
                 "value": None, "source": "", "as_of": "", "evidence_class": ""}
                for i, condition in enumerate(decision.invalidation_conditions)
            ]),
            _section("Expected value", fields={
                "expected_value": decision.expected_value,
                "calibrated_confidence": decision.calibrated_confidence,
                "evidence_class": decision.evidence_class,
            }),
            _section("Portfolio effect", fields=decision.portfolio_effect),
            _section("Data freshness", fields=_freshness(decision)),
            _section("Data sources", items=[
                {"id": row["source"], "label": row["source"],
                 "detail": ", ".join(row["used_for"]), "value": None,
                 "source": row["source"], "as_of": row["as_of"],
                 "evidence_class": row["evidence_class"]}
                for row in _sources(decision)
            ]),
        ],
        "versions": {
            "decision_engine_version": decision.decision_engine_version,
            "feature_schema_version": decision.feature_schema_version,
            "strategy_version": decision.strategy_version,
            "schema_version": decision.schema_version,
        },
    }


def missing_sections(explanation: Mapping[str, Any]) -> list[str]:
    """Section titles the desk could not fill. Useful as a data-gap report."""
    return [
        str(section["title"])
        for section in explanation.get("sections", [])
        if not section.get("available")
    ]
