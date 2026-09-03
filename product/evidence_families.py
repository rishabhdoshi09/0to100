"""Evidence-family map and deterministic no-double-count aggregation.

Committee methods remain. They are not independent votes. Tape / SEPA /
Trend / Conviction largely transform the same price (and sometimes volume)
information. This module collapses them into inspectable families before
any BUY / READY claim.

No machine-learning weights. No silent re-ranking of methods.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

FAMILY_SCHEMA_VERSION = "family_schema_v1"

# Canonical families used for confirmation. Context-only families can appear
# on the record but do not increment effective_confirmation_count.
PRICE_STRUCTURE = "PRICE_STRUCTURE"
VOLUME_DEMAND = "VOLUME_DEMAND"
RELATIVE_STRENGTH = "RELATIVE_STRENGTH"
BUSINESS_QUALITY = "BUSINESS_QUALITY"
FINANCIAL_QUALITY = "FINANCIAL_QUALITY"
SECTOR_CONTEXT = "SECTOR_CONTEXT"
CATALYST = "CATALYST"
VALUATION = "VALUATION"
ENTRY_RISK = "ENTRY_RISK"
PORTFOLIO_RISK = "PORTFOLIO_RISK"
REGIME = "REGIME"
HISTORICAL_EXPECTANCY = "HISTORICAL_EXPECTANCY"
EVIDENCE_COMPLETENESS = "EVIDENCE_COMPLETENESS"

CONFIRMATION_FAMILIES = (
    PRICE_STRUCTURE,
    VOLUME_DEMAND,
    RELATIVE_STRENGTH,
    BUSINESS_QUALITY,
    FINANCIAL_QUALITY,
    SECTOR_CONTEXT,
    CATALYST,
    VALUATION,
    HISTORICAL_EXPECTANCY,
)

# Price-derived families are useful but not three independent theses.
PRICE_DERIVED = frozenset({PRICE_STRUCTURE, VOLUME_DEMAND, RELATIVE_STRENGTH})
NON_PRICE_FAMILIES = frozenset({
    BUSINESS_QUALITY, FINANCIAL_QUALITY, SECTOR_CONTEXT, CATALYST,
    VALUATION, HISTORICAL_EXPECTANCY,
})

MIN_INDEPENDENT_FOR_BUY = 2
# High-conviction still needs a non-price family, matching reco_ensemble.
HC_NEEDS_NON_PRICE = True

SUPPORTIVE = "SUPPORTIVE"
NEUTRAL = "NEUTRAL"
OPPOSED = "OPPOSED"
UNKNOWN = "UNKNOWN"

# Explicit audit: raw inputs, derived features, horizon, consumption of others.
METHOD_AUDIT: dict[str, dict[str, Any]] = {
    "tape": {
        "label": "Tape",
        "raw_inputs": ("price", "high", "low", "volume", "rsi", "volume_ratio", "status", "breakout_grade"),
        "derived_features": ("chase_risk", "sniper_breakout", "volume_floor"),
        "time_horizon": "tactical (sessions)",
        "consumes_other_methods": (),
        "primary_family": PRICE_STRUCTURE,
        "secondary_families": (VOLUME_DEMAND,),
        "independent": False,
        "notes": "Price/volume behavior. Shares OHLCV with SEPA, Trend, Conviction.",
    },
    "sepa": {
        "label": "SEPA",
        "raw_inputs": ("close", "52w_high_proximity", "moving_averages", "relative_strength"),
        "derived_features": ("sepa_score",),
        "time_horizon": "swing / position",
        "consumes_other_methods": (),
        "primary_family": PRICE_STRUCTURE,
        "secondary_families": (RELATIVE_STRENGTH,),
        "independent": False,
        "notes": "Trend + 52w proximity + MAs + RS. Same evidence family as Trend.",
    },
    "trend": {
        "label": "Trend",
        "raw_inputs": ("close", "sma50", "sma200"),
        "derived_features": ("n_structure_passed", "above_sma50", "above_sma200"),
        "time_horizon": "swing",
        "consumes_other_methods": (),
        "primary_family": PRICE_STRUCTURE,
        "secondary_families": (),
        "independent": False,
        "notes": "Moving averages and structure checks. Overlaps SEPA completely on trend.",
    },
    "conviction": {
        "label": "Conviction",
        "raw_inputs": ("conviction_class", "conviction_score"),
        "derived_features": ("setup_quality_composite",),
        "time_horizon": "tactical",
        "consumes_other_methods": ("tape", "trend", "sepa", "rs"),
        "primary_family": PRICE_STRUCTURE,
        "secondary_families": (),
        "independent": False,
        "notes": "Meta-score of the same setup. Not a new evidence family.",
    },
    "rs": {
        "label": "RS",
        "raw_inputs": ("close", "benchmark_close"),
        "derived_features": ("rs_percentile", "rs_vs_nifty"),
        "time_horizon": "swing (cross-sectional)",
        "consumes_other_methods": (),
        "primary_family": RELATIVE_STRENGTH,
        "secondary_families": (),
        "independent": True,
        "notes": "Relative performance vs market. SEPA also consumes RS — do not double-count SEPA as RS.",
    },
    "volume": {
        "label": "Volume",
        "raw_inputs": ("volume", "avg_volume"),
        "derived_features": ("volume_ratio",),
        "time_horizon": "tactical",
        "consumes_other_methods": (),
        "primary_family": VOLUME_DEMAND,
        "secondary_families": (),
        "independent": True,
        "notes": "Participation. Tape already votes volume; a separate Volume chip is the same family.",
    },
    "funds": {
        "label": "Funds",
        "raw_inputs": ("classification", "fundamental_coverage", "fundamental_score"),
        "derived_features": ("quality_class",),
        "time_horizon": "position / fundamental period",
        "consumes_other_methods": (),
        "primary_family": FINANCIAL_QUALITY,
        "secondary_families": (BUSINESS_QUALITY,),
        "independent": True,
        "notes": "Reported financials / coverage. Not a price transform.",
    },
    "quality": {
        "label": "Quality",
        "raw_inputs": ("classification", "fundamentals"),
        "derived_features": ("business_quality_label",),
        "time_horizon": "position",
        "consumes_other_methods": (),
        "primary_family": BUSINESS_QUALITY,
        "secondary_families": (),
        "independent": True,
        "notes": "Business-quality expert. Overlaps Funds when both read the same statements.",
    },
    "sector": {
        "label": "Sector",
        "raw_inputs": ("sector", "peer_returns", "regime_leading_sectors"),
        "derived_features": ("sector_leadership_score", "sector_leader", "sector_laggard"),
        "time_horizon": "swing / regime",
        "consumes_other_methods": (),
        "primary_family": SECTOR_CONTEXT,
        "secondary_families": (),
        "independent": True,
        "notes": "Cross-sectional sector context. Not the same as the name's own trend.",
    },
    "ev": {
        "label": "Live EV",
        "raw_inputs": ("comparable_outcomes",),
        "derived_features": ("ev_lb_pct", "ev_n"),
        "time_horizon": "historical expectancy",
        "consumes_other_methods": (),
        "primary_family": HISTORICAL_EXPECTANCY,
        "secondary_families": (),
        "independent": True,
        "notes": "Needs n>=30. Historical, not a second tape read.",
    },
    "case": {
        "label": "Case memory",
        "raw_inputs": ("similar_setup_outcomes",),
        "derived_features": ("case_expectancy_r", "case_n_similar"),
        "time_horizon": "historical expectancy",
        "consumes_other_methods": (),
        "primary_family": HISTORICAL_EXPECTANCY,
        "secondary_families": (),
        "independent": True,
        "notes": "Same family as Live EV. n<30 stays unknown.",
    },
}

ENSEMBLE_FAMILY_MAP = {
    "price_leadership": PRICE_STRUCTURE,
    "structure": PRICE_STRUCTURE,
    "participation": VOLUME_DEMAND,
    "business_quality": BUSINESS_QUALITY,
    "fundamental_change": FINANCIAL_QUALITY,
    "market_context": SECTOR_CONTEXT,
    "catalyst": CATALYST,
}

# Legacy committee labels still seen on persisted rows.
LEGACY_FAMILY_MAP = {
    "TREND": PRICE_STRUCTURE,
    "TECHNICAL": PRICE_STRUCTURE,
    "VOLUME": VOLUME_DEMAND,
    "REL_STRENGTH": RELATIVE_STRENGTH,
    "BUSINESS": BUSINESS_QUALITY,
    "FINANCIAL": FINANCIAL_QUALITY,
    "SECTOR": SECTOR_CONTEXT,
    "CATALYST": CATALYST,
    "ENTRY": PRICE_STRUCTURE,
    "HISTORICAL": HISTORICAL_EXPECTANCY,
}


def method_dependency_map() -> dict[str, Any]:
    """Inspectable audit used by tests and operator artifacts."""
    overlaps = {
        "PRICE_STRUCTURE_CLUSTER": ["tape", "sepa", "trend", "conviction"],
        "VOLUME_CLUSTER": ["tape", "volume"],
        "RS_CLUSTER": ["rs", "sepa"],
        "FUNDAMENTAL_CLUSTER": ["funds", "quality"],
        "EXPECTANCY_CLUSTER": ["ev", "case"],
    }
    return {
        "methods": {k: dict(v) for k, v in METHOD_AUDIT.items()},
        "overlaps": overlaps,
        "rule": (
            "Count independent confirmation families, not method chips. "
            "SEPA BUY + Trend BUY + Conviction BUY is one PRICE_STRUCTURE vote."
        ),
        "min_independent_for_buy": MIN_INDEPENDENT_FOR_BUY,
        "hc_needs_non_price": HC_NEEDS_NON_PRICE,
    }


def _norm_status(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    if s in {"pass", "buy", "confirm", "supportive", "supports", "strong"}:
        return SUPPORTIVE
    if s in {"fail", "avoid", "reject", "conflict", "contradict", "opposed", "weak"}:
        return OPPOSED
    if s in {"wait", "near", "extended", "neutral", "caution", "mixed"}:
        return NEUTRAL
    return UNKNOWN


def _method_id(item: Mapping[str, Any]) -> str:
    raw = str(item.get("id") or item.get("label") or "").strip().lower()
    aliases = {
        "tape": "tape", "sepa": "sepa", "trend": "trend", "conviction": "conviction",
        "rs": "rs", "relative strength": "rs", "volume": "volume",
        "funds": "funds", "quality": "quality", "sector": "sector",
        "live ev": "ev", "ev": "ev", "case memory": "case", "case": "case",
    }
    return aliases.get(raw, raw)


def _family_status_rank(status: str) -> int:
    return {OPPOSED: 3, SUPPORTIVE: 2, NEUTRAL: 1, UNKNOWN: 0}.get(status, 0)


def _merge_status(current: str, incoming: str) -> str:
    """Deterministic: opposed wins, then supportive, then neutral, then unknown."""
    if _family_status_rank(incoming) > _family_status_rank(current):
        return incoming
    return current


def aggregate_evidence_families(
    *,
    methods: Sequence[Mapping[str, Any]] | None = None,
    ensemble_families: Sequence[Mapping[str, Any]] | None = None,
    extra_family_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Collapse method / ensemble votes into independent families.

    Conviction never opens a new family. SEPA does not add a second
    RELATIVE_STRENGTH vote if the dedicated RS method did not vote.
    """
    family_votes: dict[str, str] = {fam: UNKNOWN for fam in CONFIRMATION_FAMILIES}
    method_votes: list[dict[str, Any]] = []
    notes: list[str] = []
    contributors: dict[str, list[str]] = {fam: [] for fam in CONFIRMATION_FAMILIES}

    for item in methods or []:
        if not isinstance(item, Mapping):
            continue
        mid = _method_id(item)
        audit = METHOD_AUDIT.get(mid)
        status = _norm_status(item.get("status"))
        label = str(item.get("label") or (audit or {}).get("label") or mid)
        method_votes.append({
            "id": mid,
            "label": label,
            "status": status,
            "raw_status": str(item.get("status") or ""),
            "family": (audit or {}).get("primary_family"),
            "independent": bool((audit or {}).get("independent")),
            "consumes": list((audit or {}).get("consumes_other_methods") or ()),
        })
        if not audit:
            notes.append(f"{label}: unmapped method — ignored for confirmation")
            continue
        primary = str(audit["primary_family"])
        family_votes[primary] = _merge_status(family_votes.get(primary, UNKNOWN), status)
        contributors.setdefault(primary, []).append(label)
        if mid == "sepa" and RELATIVE_STRENGTH not in {
            _method_id(m) for m in (methods or []) if isinstance(m, Mapping)
        }:
            notes.append("SEPA uses relative strength internally; RS is not counted twice")
        if mid == "conviction":
            notes.append("Conviction is a setup meta-score — same PRICE_STRUCTURE family")
        if mid == "tape":
            detail = str(item.get("detail") or "").lower()
            if "volume" in detail or item.get("volume_status") not in (None, ""):
                notes.append(
                    "Tape mentions volume — same price/volume chip; "
                    "VOLUME_DEMAND confirms only via a dedicated volume/participation vote"
                )

    for fam in ensemble_families or []:
        if not isinstance(fam, Mapping):
            continue
        fid = str(fam.get("id") or "")
        mapped = ENSEMBLE_FAMILY_MAP.get(fid) or LEGACY_FAMILY_MAP.get(fid.upper())
        if not mapped:
            continue
        status = _norm_status(fam.get("status"))
        family_votes[mapped] = _merge_status(family_votes.get(mapped, UNKNOWN), status)
        contributors.setdefault(mapped, []).append(f"ensemble:{fid}")

    for key, status in (extra_family_status or {}).items():
        mapped = ENSEMBLE_FAMILY_MAP.get(str(key)) or LEGACY_FAMILY_MAP.get(str(key).upper()) or str(key).upper()
        if mapped not in family_votes and mapped not in CONFIRMATION_FAMILIES:
            family_votes[mapped] = _norm_status(status)
            continue
        family_votes[mapped] = _merge_status(family_votes.get(mapped, UNKNOWN), _norm_status(status))

    supportive = [f for f in CONFIRMATION_FAMILIES if family_votes.get(f) == SUPPORTIVE]
    opposed = [f for f in CONFIRMATION_FAMILIES if family_votes.get(f) == OPPOSED]
    neutral = [f for f in CONFIRMATION_FAMILIES if family_votes.get(f) == NEUTRAL]
    unknown = [f for f in CONFIRMATION_FAMILIES if family_votes.get(f) == UNKNOWN]
    non_price = [f for f in supportive if f in NON_PRICE_FAMILIES]
    # Tape-derived volume must not inflate confirmation. Dedicated volume/participation still counts.
    tape_only_volume = (
        VOLUME_DEMAND in supportive
        and all(str(c).startswith("Tape") or str(c).startswith("tape") for c in (contributors.get(VOLUME_DEMAND) or []))
    )
    countable = [f for f in supportive if not (f == VOLUME_DEMAND and tape_only_volume)]
    effective = len(countable)

    method_buy = [m["label"] for m in method_votes if m["status"] == SUPPORTIVE]
    if len(method_buy) > effective:
        notes.append(
            f"{len(method_buy)} method BUY chips collapse to {effective} independent "
            f"families: {', '.join(supportive) or 'none'}"
        )

    return {
        "method_votes": method_votes,
        "evidence_family_votes": {k: family_votes[k] for k in CONFIRMATION_FAMILIES},
        "family_contributors": {k: v for k, v in contributors.items() if v},
        "supportive_families": supportive,
        "opposed_families": opposed,
        "neutral_families": neutral,
        "unknown_families": unknown,
        "non_price_supportive": non_price,
        "countable_families": countable,
        "effective_confirmation_count": effective,
        "price_derived_only": bool(supportive) and not non_price,
        "dependency_notes": notes,
        "min_independent_for_buy": MIN_INDEPENDENT_FOR_BUY,
        "hc_needs_non_price": HC_NEEDS_NON_PRICE,
    }


def family_gate(*, aggregation: Mapping[str, Any], tier: str = "") -> dict[str, Any]:
    """Deterministic BUY gate on independent families. No weights."""
    n = int(aggregation.get("effective_confirmation_count") or 0)
    non_price = list(aggregation.get("non_price_supportive") or [])
    opposed = list(aggregation.get("opposed_families") or [])
    ok = n >= MIN_INDEPENDENT_FOR_BUY
    reason = ""
    if n < MIN_INDEPENDENT_FOR_BUY:
        reason = "INSUFFICIENT_INDEPENDENT_EVIDENCE"
        ok = False
    # A BUY needs a non-price family. Price+volume+RS from the same tape is not enough.
    if HC_NEEDS_NON_PRICE and not non_price:
        reason = "INSUFFICIENT_INDEPENDENT_EVIDENCE"
        ok = False
    if BUSINESS_QUALITY in opposed or FINANCIAL_QUALITY in opposed:
        # Family opposition is a veto signal; committee still owns hard-veto codes.
        reason = reason or "FAMILY_QUALITY_OPPOSED"
    return {
        "ok": ok,
        "reason_code": reason,
        "effective_confirmation_count": n,
        "non_price_supportive": non_price,
        "detail": (
            f"{n} independent families supportive"
            + (f"; non-price {', '.join(non_price)}" if non_price else "; no non-price family")
        ),
    }
