"""Common-man presentation layer for QuantTerm.

TECHNICAL TRUTH (internal enums, schemas, metrics) stays unchanged.
This module only translates values into user-facing copy:

    label · one-line explanation · practical implication · optional technical detail

UI and reports should call these helpers. Never rename canonical fields in
storage or research schemas merely to sound friendlier.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class PlainCard:
    """Two-layer presentation unit.

    Layer 1: ``label``, ``explanation``, ``implication``, ``state``
    Layer 2: ``technical`` (always kept; disclose behind Why? / details)
    """

    label: str
    explanation: str
    implication: str = ""
    state: str = ""  # GOOD | CAUTION | RISKY | NOT_ENOUGH_DATA | STRONG | …
    technical: str = ""
    topic: str = ""
    internal_key: str = ""
    internal_value: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Internal field → UI label (schemas keep the left-hand names)
# ---------------------------------------------------------------------------

FIELD_LABELS: dict[str, str] = {
    "regime": "Market condition",
    "regime_state": "Market condition",
    "market_regime": "Market condition",
    "network_concentration": "Portfolio overlap",
    "network_concentration_score": "Portfolio overlap risk",
    "betweenness_centrality": "Portfolio dependency",
    "centrality": "Portfolio dependency",
    "incremental_community_exposure": "How different is this stock from what we already own?",
    "community_exposure": "Overlap with what you already own",
    "calibration": "How reliable is this?",
    "calibration_score": "How reliable is this?",
    "expectancy": "How much can we expect?",
    "expectancy_r": "Average result per trade (in risk units)",
    "expected_value": "How much can we expect?",
    "ev": "How much can we expect?",
    "evidence_level": "Research confidence",
    "evidence_tier": "Research quality",
    "trust_class": "Data quality",
    "pit_state": "Historical data quality",
    "pit_compliance": "Historical data quality",
    "data_state": "Data quality",
    "decision_attribution": "Why was this rejected?",
    "gauntlet_verdict": "Past test result",
    "historical_gauntlet_verdict": "Past test result",
    "sharpe": "Risk-adjusted performance",
    "cluster_stability": "Market structure stability",
    "stability_ari": "Market structure stability",
    "fdr": "Multiple-test safeguard",
    "dsr": "Deflated skill check",
    "psr": "Probabilistic skill check",
    "snapshot_id": "Saved market snapshot id",
    "hypothesis_id": "Research test id",
    "research_grade": "Research-quality data?",
}


TERMINOLOGY: dict[str, str] = {
    **FIELD_LABELS,
    "snapshot": "Saved market data",
    "forward_eligible": "Data ready for trading",
    "evidence_card": "Why the system believes this",
    "allocation": "Paper-money amount",
    "risk_governor": "Safety checks",
    "intent": "Proposed paper trade",
    "reconciliation": "Record check",
    "in_sample": "Historical test",
    "no_eligible_intent": "No safe trade found",
    "feature_store": "Saved research inputs",
    "evidence_graph": "Linked research memory",
    "pit": "Point-in-time historical rules",
    "gauntlet": "Historical stress test",
    "counterfactual": "What-if comparison",
    "hrp": "Risk-balanced portfolio mix",
    "bayesian_posterior": "Updated belief after evidence",
}


# Job-oriented nav labels (engineering concepts stay out of the sidebar)
NAV_JOBS: dict[str, str] = {
    "home": "Home",
    "find_stocks": "Find Stocks",
    "portfolio": "Portfolio",
    "market": "Market",
    "research": "Research",
    "reports": "Reports",
    "alerts": "Alerts",
    "assistant": "Assistant",
    "settings": "Settings",
    "help": "Help",
}


def label_for(internal_key: str, fallback: str | None = None) -> str:
    key = str(internal_key or "").strip()
    if key in FIELD_LABELS:
        return FIELD_LABELS[key]
    if key in TERMINOLOGY:
        return TERMINOLOGY[key]
    return fallback if fallback is not None else key.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Trust / data quality
# ---------------------------------------------------------------------------

_TRUST: dict[str, PlainCard] = {
    "RESEARCH_GRADE": PlainCard(
        topic="data_quality",
        internal_key="trust_class",
        internal_value="RESEARCH_GRADE",
        label="Research quality",
        state="PROVEN",
        explanation="This dataset passed the checks required for scientific historical testing.",
        implication="Safe to use when judging whether a strategy deserves real money.",
        technical="trust_class=RESEARCH_GRADE",
    ),
    "OPERATIONAL_ONLY": PlainCard(
        topic="data_quality",
        internal_key="trust_class",
        internal_value="OPERATIONAL_ONLY",
        label="Data quality",
        state="CAUTION",
        explanation="Good enough for today's trading view, not fully reconstructible history.",
        implication="Use for live/paper decisions; do not treat results as scientific proof.",
        technical="trust_class=OPERATIONAL_ONLY",
    ),
    "DISPLAY_ONLY": PlainCard(
        topic="data_quality",
        internal_key="trust_class",
        internal_value="DISPLAY_ONLY",
        label="Data quality",
        state="UNPROVEN",
        explanation="Good enough for charts and exploration, not for proving a strategy.",
        implication="QuantTerm will not promote a strategy on this data alone.",
        technical="trust_class=DISPLAY_ONLY",
    ),
    "PROHIBITED": PlainCard(
        topic="data_quality",
        internal_key="trust_class",
        internal_value="PROHIBITED",
        label="Data quality",
        state="FAILED",
        explanation="This source must not be used for any trading or research decision.",
        implication="Ignore any chart or number that came only from this source.",
        technical="trust_class=PROHIBITED",
    ),
    "NOT_PIT_SAFE": PlainCard(
        topic="historical_data_quality",
        internal_key="pit_state",
        internal_value="NOT_PIT_SAFE",
        label="Historical data quality",
        state="RISKY",
        explanation=(
            "This historical data may contain information that was not actually known at the time. "
            "Do not use it for serious backtesting."
        ),
        implication="Treat any backtest on this series as exploratory only.",
        technical="pit_state=NOT_PIT_SAFE",
    ),
    "INCOMPLETE": PlainCard(
        topic="historical_data_quality",
        internal_key="pit_state",
        internal_value="INCOMPLETE",
        label="Historical data quality",
        state="NOT_ENOUGH_DATA",
        explanation="Important historical ledgers are missing, so the picture is incomplete.",
        implication="Fill corporate-action and universe history before trusting long tests.",
        technical="pit_state=INCOMPLETE",
    ),
    "BLOCKED": PlainCard(
        topic="historical_data_quality",
        internal_key="data_state",
        internal_value="BLOCKED",
        label="Data quality",
        state="RISKY",
        explanation="Checks failed or required inputs are missing, so research is blocked.",
        implication="Do not treat downstream results as trustworthy until the block clears.",
        technical="data_state=BLOCKED",
    ),
}


def explain_trust_class(value: str | None) -> PlainCard:
    key = str(value or "").strip().upper()
    if key in _TRUST:
        return _TRUST[key]
    return PlainCard(
        topic="data_quality",
        internal_key="trust_class",
        internal_value=str(value or ""),
        label="Data quality",
        state="NOT_ENOUGH_DATA",
        explanation="Data quality has not been classified yet.",
        implication="Ask for a quality check before trusting research conclusions.",
        technical=f"trust_class={value}",
    )


def explain_pit_state(value: str | None) -> PlainCard:
    key = str(value or "").strip().upper()
    if key in _TRUST:
        card = _TRUST[key]
        if card.internal_key in {"pit_state", "data_state"} or key == "NOT_PIT_SAFE":
            return card
    if key in {"READY", "PIT_SAFE", "OK"}:
        return PlainCard(
            topic="historical_data_quality",
            internal_key="pit_state",
            internal_value=key,
            label="Historical data quality",
            state="GOOD",
            explanation="History is being read with only information available as of each past date.",
            implication="Suitable for serious historical testing when other research checks also pass.",
            technical=f"pit_state={key}",
        )
    return explain_trust_class(value)


# ---------------------------------------------------------------------------
# Research / gauntlet verdicts
# ---------------------------------------------------------------------------

_VERDICTS: dict[str, PlainCard] = {
    "PASS": PlainCard(
        label="Past test result",
        state="PROVEN",
        explanation="We tested this idea properly. Historical evidence supports keeping it under review.",
        implication="Still subject to risk and live checks before any real-money use.",
        technical="verdict=PASS",
    ),
    "PASS_ALPHA": PlainCard(
        label="Past test result",
        state="PROVEN",
        explanation="Historical testing found a measurable advantage after the stated costs and safeguards.",
        implication="Eligible for further live/paper scrutiny — not an automatic go-live.",
        technical="verdict=PASS_ALPHA",
    ),
    "PASS_RISK": PlainCard(
        label="Past test result",
        state="PROMISING",
        explanation="The idea helped with risk control in historical tests more than it proved a profit edge.",
        implication="Useful as a risk lens; do not treat it as a standalone profit signal.",
        technical="verdict=PASS_RISK",
    ),
    "FAIL": PlainCard(
        label="Past test result",
        state="FAILED",
        explanation=(
            "We tested this idea properly. So far, it does not show a reliable advantage after trading costs. "
            "QuantTerm will not use it for real trades."
        ),
        implication="Do not promote this idea; keep or archive as a negative result.",
        technical="verdict=FAIL",
    ),
    "REJECT": PlainCard(
        label="Past test result",
        state="FAILED",
        explanation="The idea failed its pre-registered success checks.",
        implication="QuantTerm will not use it for live decisions.",
        technical="verdict=REJECT",
    ),
    "REJECTED": PlainCard(
        label="Past test result",
        state="FAILED",
        explanation="The idea failed its pre-registered success checks.",
        implication="QuantTerm will not use it for live decisions.",
        technical="verdict=REJECTED",
    ),
    "INCONCLUSIVE": PlainCard(
        label="Past test result",
        state="UNPROVEN",
        explanation=(
            "We do not have enough trustworthy data yet to say whether this strategy works. "
            "The result remains unproven."
        ),
        implication="Do not promote and do not expand models on this result alone.",
        technical="verdict=INCONCLUSIVE",
    ),
    "UNDERPOWERED": PlainCard(
        label="Past test result",
        state="NOT_ENOUGH_DATA",
        explanation="There were not enough clean trades or sessions to judge this idea fairly.",
        implication="Gather more trustworthy history before deciding.",
        technical="verdict=UNDERPOWERED",
    ),
    "EXPLORATORY": PlainCard(
        label="Past test result",
        state="UNPROVEN",
        explanation="This was an exploratory look only — useful for learning, not for proving edge.",
        implication="Keep as a notebook result; never treat as promotion evidence.",
        technical="verdict=EXPLORATORY",
    ),
}


def explain_research_verdict(verdict: str | None, *, reason: str = "") -> PlainCard:
    key = str(verdict or "").strip().upper()
    base = _VERDICTS.get(key)
    if base is None:
        base = PlainCard(
            label="Past test result",
            state="NOT_ENOUGH_DATA",
            explanation="No clear research conclusion is available yet.",
            implication="Wait for a completed test on trustworthy data.",
            technical=f"verdict={verdict}",
        )
    implication = base.implication
    if reason:
        implication = f"{implication} ({reason})" if implication else reason
    return PlainCard(
        topic="research_verdict",
        internal_key="verdict",
        internal_value=key,
        label=base.label,
        state=base.state,
        explanation=base.explanation,
        implication=implication,
        technical=base.technical + (f"; {reason}" if reason else ""),
    )


# ---------------------------------------------------------------------------
# Decisions (BUY / WATCH / REJECT / HOLD)
# ---------------------------------------------------------------------------

_DECISIONS: dict[str, str] = {
    "BUY": "STRONG",
    "STRONG_BUY": "STRONG",
    "WATCH": "CAUTION",
    "HOLD": "MODERATE",
    "AVOID": "RISKY",
    "REJECT": "RISKY",
    "SHORT": "CAUTION",
    "NO_TRADE": "CAUTION",
}


def explain_decision(
    decision: str | None,
    *,
    why: str = "",
    positives: list[str] | tuple[str, ...] = (),
    negatives: list[str] | tuple[str, ...] = (),
    technical: str = "",
) -> PlainCard:
    key = str(decision or "").strip().upper() or "WATCH"
    state = _DECISIONS.get(key, "CAUTION")
    bits = []
    if positives:
        bits.append("Helping: " + "; ".join(positives[:4]))
    if negatives:
        bits.append("Holding back: " + "; ".join(negatives[:4]))
    explanation = why or {
        "BUY": "The setup clears QuantTerm's current checks.",
        "WATCH": "Interesting, but something important is still missing or risky.",
        "HOLD": "Keep the existing position; no fresh action is required.",
        "REJECT": "QuantTerm would not take this trade right now.",
        "AVOID": "Risk or quality checks say stay away.",
        "NO_TRADE": "No safe new trade was found.",
    }.get(key, "QuantTerm recorded a decision for this name.")
    implication = " · ".join(bits) if bits else {
        "BUY": "Size only through the normal risk ticket.",
        "WATCH": "Wait for a cleaner setup or less portfolio overlap.",
        "REJECT": "Do not force the trade.",
        "HOLD": "Review again if thesis or risk changes.",
        "AVOID": "Look elsewhere.",
        "NO_TRADE": "Protect capital; scan again later.",
    }.get(key, "Review the reasons before acting.")
    return PlainCard(
        topic="decision",
        internal_key="decision",
        internal_value=key,
        label=f"Decision: {key}",
        state=state,
        explanation=explanation,
        implication=implication,
        technical=technical or f"decision={key}",
    )


# ---------------------------------------------------------------------------
# Metric framing — every number needs meaning
# ---------------------------------------------------------------------------

def traffic_from_score(
    score: float | None,
    *,
    good_at: float = 0.7,
    caution_at: float = 0.4,
    higher_is_better: bool = True,
) -> str:
    if score is None:
        return "NOT_ENOUGH_DATA"
    try:
        x = float(score)
    except (TypeError, ValueError):
        return "NOT_ENOUGH_DATA"
    if not higher_is_better:
        x = -x
        good_at, caution_at = -good_at, -caution_at
        good_at, caution_at = max(good_at, caution_at), min(good_at, caution_at)
    if x >= good_at:
        return "GOOD"
    if x >= caution_at:
        return "CAUTION"
    return "RISKY"


def explain_metric(
    internal_key: str,
    value: Any,
    *,
    state: str | None = None,
    explanation: str = "",
    implication: str = "",
    higher_is_better: bool = True,
    good_at: float = 0.7,
    caution_at: float = 0.4,
) -> PlainCard:
    label = label_for(internal_key)
    numeric: float | None
    try:
        numeric = float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        numeric = None
    st = state or traffic_from_score(
        numeric, good_at=good_at, caution_at=caution_at, higher_is_better=higher_is_better
    )
    defaults = {
        "network_concentration_score": (
            "Several holdings may be behaving like one large position."
            if st in {"RISKY", "CAUTION"} else
            "Holdings look reasonably diversified on this lens."
        ),
        "betweenness_centrality": (
            "This stock sits in the middle of several correlated holdings, so adding it may "
            "increase hidden concentration."
            if st in {"RISKY", "CAUTION"} else
            "This name is not a major bridge between your other holdings."
        ),
        "cluster_stability": (
            "Current stock groupings have been relatively consistent."
            if st == "GOOD" else
            "Stock groupings are shifting — treat structure features cautiously."
        ),
        "sharpe": (
            "Risk-adjusted results look acceptable on this sample."
            if st == "GOOD" else
            "Risk-adjusted results are weak or unstable on this sample."
        ),
    }
    expl = explanation or defaults.get(internal_key, f"{label} reading: {st.replace('_', ' ').title()}.")
    impl = implication or {
        "GOOD": "No extra caution from this metric alone.",
        "CAUTION": "Pay attention before sizing up.",
        "RISKY": "Reduce size or skip until this improves.",
        "NOT_ENOUGH_DATA": "Do not decide from this metric yet.",
    }.get(st, "")
    tech = f"{internal_key}={value}"
    return PlainCard(
        topic="metric",
        internal_key=internal_key,
        internal_value=str(value),
        label=label,
        state=st,
        explanation=expl,
        implication=impl,
        technical=tech,
    )


def explain_unresolved_ca() -> PlainCard:
    return PlainCard(
        topic="data_quality",
        internal_key="UNRESOLVED_CA",
        internal_value="UNRESOLVED",
        label="Historical price adjustment",
        state="NOT_READY",
        explanation=(
            "We found a large historical price change that may be related to a corporate "
            "action, but we do not yet have enough official evidence to adjust it safely."
        ),
        implication="Do not treat strategy tests on affected names as fully trustworthy yet.",
        technical="UNRESOLVED_CA",
    )


def explain_unresolved_lineage() -> PlainCard:
    return PlainCard(
        topic="data_quality",
        internal_key="UNRESOLVED_LINEAGE",
        internal_value="UNRESOLVED",
        label="Stock history link",
        state="NOT_READY",
        explanation=(
            "The company's ticker/history changed, and QuantTerm cannot yet prove that the "
            "old and new records represent the same security."
        ),
        implication="Unresolved lineage can block research-grade certification.",
        technical="UNRESOLVED_LINEAGE",
    )
    """One plain paragraph for research report headers."""
    v = explain_research_verdict(verdict)
    parts = [v.explanation]
    if trust_class:
        parts.append(explain_trust_class(trust_class).explanation)
    if v.implication:
        parts.append(v.implication)
    return " ".join(p for p in parts if p)


def beginner_ok(card: PlainCard) -> bool:
    """Crude 10-second check: short label/explanation, no raw jargon-only dump."""
    if not card.label or not card.explanation:
        return False
    if len(card.label) > 60 or len(card.explanation) > 240:
        return False
    banned = ("betweenness", "posterior", "FDR-adjusted", "embargoed CV")
    low = card.label.lower()
    return not any(b in low for b in banned)


def render_layers(card: PlainCard) -> dict[str, Any]:
    """Shape used by UI: layer1 always, layer2 optional expander payload."""
    return {
        "layer1": {
            "label": card.label,
            "state": card.state,
            "explanation": card.explanation,
            "implication": card.implication,
            "what_it_means": card.explanation,
            "what_to_do": card.implication,
        },
        "layer2": {
            "technical_details": card.technical,
            "internal_key": card.internal_key,
            "internal_value": card.internal_value,
            "topic": card.topic,
        },
        "beginner_ok": beginner_ok(card),
    }


def bulk_explain(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Translate a small dict of known keys into layered cards."""
    out: dict[str, dict[str, Any]] = {}
    if "trust_class" in payload:
        out["trust_class"] = render_layers(explain_trust_class(payload.get("trust_class")))
    if "pit_state" in payload:
        out["pit_state"] = render_layers(explain_pit_state(payload.get("pit_state")))
    if "verdict" in payload:
        out["verdict"] = render_layers(explain_research_verdict(payload.get("verdict")))
    if "decision" in payload:
        out["decision"] = render_layers(
            explain_decision(
                payload.get("decision"),
                why=str(payload.get("why") or ""),
                positives=tuple(payload.get("positives") or ()),
                negatives=tuple(payload.get("negatives") or ()),
            )
        )
    for key in (
        "network_concentration_score",
        "betweenness_centrality",
        "cluster_stability",
        "sharpe",
        "expectancy_r",
    ):
        if key in payload:
            hib = key not in {"network_concentration_score", "betweenness_centrality"}
            out[key] = render_layers(
                explain_metric(key, payload.get(key), higher_is_better=hib)
            )
    return out
