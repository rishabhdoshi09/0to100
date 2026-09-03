"""Decision committee over existing evidence producers.

Does not invent a second recommendation engine. Methods / families / research /
paper gates already exist. This layer names the judgment, the vetoes, and the
disagreement — and refuses to call a name READY just because ENTER_NOW fired.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from product import decision_taxonomy as T
from product.due_diligence.evidence import DEFAULT_MIN_DECISION_COVERAGE
from product.paper_autopilot import (
    BLOCK,
    ENTER_NOW,
    PORTFOLIO_BLOCK,
    WAIT,
    WATCH,
    evaluate_candidate,
)
from product.reco_ensemble import TIER_GOOD, TIER_HIGH


FAMILY_MAP = {
    "price_leadership": "TREND",
    "structure": "TECHNICAL",
    "participation": "VOLUME",
    "business_quality": "BUSINESS",
    "fundamental_change": "FINANCIAL",
    "market_context": "SECTOR",
    "catalyst": "CATALYST",
}

METHOD_TO_FAMILY = {
    "tape": "TECHNICAL",
    "sepa": "TREND",
    "trend": "TREND",
    "rs": "REL_STRENGTH",
    "volume": "VOLUME",
    "funds": "FINANCIAL",
    "quality": "BUSINESS",
    "sector": "SECTOR",
    "conviction": "ENTRY",
    "ev": "HISTORICAL",
    "case": "HISTORICAL",
}

HC_NEEDS_RESEARCH = True


@dataclass
class CommitteeRecord:
    symbol: str
    decision: str
    candidate_state: str
    entry_state: str
    execution_state: str
    reason_code: str
    reason: str
    tier: str = ""
    families: dict[str, str] = field(default_factory=dict)
    methods_buy: list[str] = field(default_factory=list)
    methods_wait: list[str] = field(default_factory=list)
    methods_avoid: list[str] = field(default_factory=list)
    methods_unknown: list[str] = field(default_factory=list)
    vetoes: list[dict[str, str]] = field(default_factory=list)
    missing_critical: list[str] = field(default_factory=list)
    evidence_coverage_pct: float | None = None
    information_value: str = "NONE"
    research_required: bool = False
    framework_id: str = ""
    wait_trigger: dict[str, Any] = field(default_factory=dict)
    positives: list[str] = field(default_factory=list)
    paper_decision: str = ""
    paper_reason: str = ""
    entry: float | None = None
    stop: float | None = None
    target: float | None = None
    references: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "decision": self.decision,
            "candidate_state": self.candidate_state,
            "entry_state": self.entry_state,
            "execution_state": self.execution_state,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "tier": self.tier,
            "families": self.families,
            "methods_buy": self.methods_buy,
            "methods_wait": self.methods_wait,
            "methods_avoid": self.methods_avoid,
            "methods_unknown": self.methods_unknown,
            "disagreement": bool(self.methods_buy and (self.methods_wait or self.methods_avoid)),
            "vetoes": self.vetoes,
            "missing_critical": self.missing_critical,
            "evidence_coverage_pct": self.evidence_coverage_pct,
            "information_value": self.information_value,
            "research_required": self.research_required,
            "framework_id": self.framework_id,
            "wait_trigger": self.wait_trigger,
            "positives": self.positives,
            "paper_decision": self.paper_decision,
            "paper_reason": self.paper_reason,
            "entry": self.entry,
            "stop": self.stop,
            "target": self.target,
            "references": self.references,
        }


def _f(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _method_vote(status: str) -> str:
    s = str(status or "").lower()
    if s in {"pass", "buy", "confirm"}:
        return T.BUY
    if s in {"fail", "avoid", "reject", "conflict"}:
        return T.AVOID
    if s in {"wait", "near", "extended"}:
        return T.WAIT_DECISION
    return "UNKNOWN"


def _research_snapshot(symbol: str, card: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from product.due_diligence.research_engine import StockResearchEngine

        report = StockResearchEngine().investigate(symbol)
    except Exception as exc:
        return {"error": str(exc)[:200], "available": False}
    if not isinstance(report, dict) or not report:
        return {"available": False}
    coverage = dict(report.get("decision_coverage") or {})
    quality = dict(report.get("fundamental_quality") or {})
    framework = dict(report.get("framework") or {})
    missing = []
    for row in coverage.get("missing") or []:
        if isinstance(row, Mapping):
            if str(row.get("importance") or "") == "critical":
                missing.append(str(row.get("id") or row.get("label") or ""))
    flags = [str(x.get("id") or x.get("label") or x)[:80] for x in (report.get("red_flags") or []) if x]
    return {
        "available": True,
        "framework_id": str(framework.get("id") or report.get("framework_id") or ""),
        "framework_label": str(framework.get("label") or report.get("framework_label") or ""),
        "coverage_pct": coverage.get("coverage_pct") if coverage else report.get("decision_coverage_pct"),
        "critical_n": coverage.get("critical_n"),
        "critical_available": coverage.get("critical_available"),
        "missing_critical": missing,
        "quality_label": quality.get("label") or report.get("quality_label") or "",
        "red_flags": flags,
        "vs_technical": str(report.get("vs_technical_setup") or ""),
        "acquired_at": ((report.get("as_of") or {}).get("autonomy_acquired_at") if isinstance(report.get("as_of"), Mapping) else ""),
    }


def _information_value(
    card: Mapping[str, Any],
    paper_decision: str,
    paper_reason: str,
    research: Mapping[str, Any],
) -> str:
    """Would missing evidence realistically change BUY / WAIT / AVOID? Deterministic."""
    tier = str(card.get("reco_tier") or "")
    if paper_decision in {BLOCK, WATCH} and tier not in {TIER_HIGH, TIER_GOOD}:
        return "NONE"
    if paper_decision == WAIT and paper_reason == T.ENTRY_TOO_EXTENDED:
        # Price is the gate; an annual report will not un-extend it.
        return "LOW"
    missing = list(research.get("missing_critical") or [])
    if not research.get("available"):
        if tier == TIER_HIGH:
            return "HIGH"
        if tier == TIER_GOOD and paper_decision == ENTER_NOW:
            return "MEDIUM"
        return "LOW"
    if missing and paper_decision == ENTER_NOW:
        return "HIGH"
    if missing and paper_decision == WAIT:
        return "MEDIUM"
    return "NONE"


def _wait_trigger(entry_state: str, card: Mapping[str, Any], reason: str) -> dict[str, Any]:
    entry = _f(card.get("entry") or card.get("cmp"))
    zone_hi = _f(card.get("buy_zone_high"))
    zone_lo = _f(card.get("buy_zone_low"))
    if entry_state == T.EXTENDED or reason == T.ENTRY_TOO_EXTENDED:
        level = zone_hi or (entry * 0.97 if entry else None)
        return {
            "kind": "PRICE_LTE",
            "reason": T.ENTRY_TOO_EXTENDED,
            "price": level,
            "reconsider_when": f"price <= {level}" if level else "price contracts into buy zone",
        }
    if entry_state in {T.NEAR_SETUP, T.WAIT_FOR_ENTRY} or reason == T.ENTRY_NOT_TRIGGERED:
        level = zone_lo or entry
        return {
            "kind": "PRICE_GTE",
            "reason": T.ENTRY_NOT_TRIGGERED,
            "price": level,
            "reconsider_when": f"breakout/hold above {level}" if level else "entry trigger prints",
        }
    if reason == T.INSUFFICIENT_EVIDENCE:
        return {
            "kind": "EVIDENCE_ACQUIRED",
            "reason": T.INSUFFICIENT_EVIDENCE,
            "reconsider_when": "required evidence acquired",
        }
    if reason in {T.PORTFOLIO_CONCENTRATION, T.CORRELATION_LIMIT, T.MAX_PORTFOLIO_RISK, T.SECTOR_CAP}:
        return {
            "kind": "RISK_BUDGET",
            "reason": reason,
            "reconsider_when": "portfolio risk capacity frees",
        }
    return {}


def evaluate_committee(
    card: Mapping[str, Any],
    *,
    book=None,
    broker_ok: bool = False,
    entry_window: bool = False,
    workspace: Mapping[str, Any] | None = None,
    load_research: bool = True,
) -> CommitteeRecord:
    symbol = str(card.get("symbol") or "").upper()
    tier = str(card.get("reco_tier") or "")
    entry_state = T.entry_from_card(dict(card))
    families: dict[str, str] = {}
    for fam in card.get("families") or []:
        if isinstance(fam, Mapping) and fam.get("id"):
            families[FAMILY_MAP.get(str(fam.get("id")), str(fam.get("id")).upper())] = str(fam.get("status") or "unknown").upper()
    methods_buy, methods_wait, methods_avoid, methods_unknown = [], [], [], []
    for meth in card.get("methods") or []:
        if not isinstance(meth, Mapping):
            continue
        label = str(meth.get("label") or meth.get("id") or "")
        vote = _method_vote(str(meth.get("status") or ""))
        if vote == T.BUY:
            methods_buy.append(label)
        elif vote == T.WAIT_DECISION:
            methods_wait.append(label)
        elif vote == T.AVOID:
            methods_avoid.append(label)
        else:
            methods_unknown.append(label)
        mapped = METHOD_TO_FAMILY.get(str(meth.get("id") or ""))
        if mapped and mapped not in families:
            families[mapped] = str(meth.get("status") or "unknown").upper()

    paper = evaluate_candidate(card, book=book, entries_allowed=True, workspace=workspace)
    paper_decision = paper.decision
    paper_reason = str(paper.reason_code or "")

    research = _research_snapshot(symbol, card) if load_research else {"available": False}
    info_value = _information_value(card, paper_decision, paper_reason, research)
    coverage = research.get("coverage_pct")
    if coverage is None:
        coverage = card.get("research_decision_coverage") or card.get("evidence_coverage")
    try:
        coverage_f = float(coverage) if coverage is not None else None
    except (TypeError, ValueError):
        coverage_f = None

    vetoes: list[dict[str, str]] = []
    missing_critical = list(research.get("missing_critical") or [])
    quality_label = str(research.get("quality_label") or card.get("stock_quality") or "")
    if paper_reason in T.HARD_VETO_CODES:
        vetoes.append({"code": paper_reason, "detail": str(paper.detail or paper_reason)})
    vs_tech = str(research.get("vs_technical") or "").upper()
    if vs_tech.find("CONTRADICT") >= 0:
        vetoes.append({"code": T.FINANCIAL_QUALITY_FAIL, "detail": "research contradicts technical setup"})
    if quality_label.lower() in {"weak", "avoid", "poor"}:
        vetoes.append({"code": T.BUSINESS_QUALITY_FAIL, "detail": quality_label})
    caution_wait = vs_tech == "CAUTION" and quality_label.lower() in {"mixed", "weak", "avoid", "poor", ""}
    if families.get("BUSINESS") == "FAIL":
        vetoes.append({"code": T.BUSINESS_QUALITY_FAIL, "detail": "business quality family failed"})
    if families.get("SECTOR") == "FAIL" and tier == TIER_HIGH:
        vetoes.append({"code": T.WEAK_SECTOR, "detail": "high-conviction inside a failed sector family"})

    research_required = bool(
        tier == TIER_HIGH
        or (info_value == "HIGH" and paper_decision == ENTER_NOW)
    )
    research_ok = True
    if research_required:
        facts_ok = bool(research.get("available") and research.get("acquired_at"))
        cov_ok = coverage_f is not None and (
            coverage_f >= (DEFAULT_MIN_DECISION_COVERAGE * 100.0 if coverage_f > 1 else DEFAULT_MIN_DECISION_COVERAGE)
        )
        if not facts_ok and not cov_ok:
            research_ok = False
            # Evidence is a BUY prerequisite, not a replacement for an already-known price veto.
            if paper_decision == ENTER_NOW and not any(v["code"] == T.INSUFFICIENT_EVIDENCE for v in vetoes):
                vetoes.append({"code": T.INSUFFICIENT_EVIDENCE, "detail": "high-value name missing required research"})

    hard = [v for v in vetoes if v["code"] in T.HARD_VETO_CODES or v["code"] == T.INSUFFICIENT_EVIDENCE]

    if paper_decision == ENTER_NOW and not hard and research_ok and not caution_wait:
        decision = T.BUY
        candidate_state = T.READY
        reason_code = "COMMITTEE_BUY"
        reason = "Independent families and gates justify taking risk."
    elif paper_decision == ENTER_NOW and not hard and caution_wait:
        decision = T.WAIT_DECISION
        candidate_state = T.WAIT
        reason_code = T.INSUFFICIENT_EVIDENCE
        reason = "Research does not yet confirm the technical setup."
    elif paper_decision == WAIT or (paper_decision == ENTER_NOW and hard and any(v["code"] in {T.ENTRY_TOO_EXTENDED, T.INSUFFICIENT_EVIDENCE} for v in hard)):
        decision = T.WAIT_DECISION
        if paper_reason == T.ENTRY_TOO_EXTENDED or entry_state == T.EXTENDED:
            candidate_state = T.WAIT
            reason_code = T.ENTRY_TOO_EXTENDED
            reason = str(paper.detail or "Valid candidate, not the right price.")
        elif any(v["code"] == T.INSUFFICIENT_EVIDENCE for v in hard) or (
            not research_ok and paper_decision == ENTER_NOW
        ):
            candidate_state = T.WAIT_EVIDENCE
            reason_code = T.INSUFFICIENT_EVIDENCE
            reason = "Setup is interesting; required evidence is missing."
        else:
            candidate_state = T.WAIT
            reason_code = paper_reason or T.ENTRY_NOT_TRIGGERED
            reason = str(paper.detail or "Valid candidate, not the right price/trigger.")
    elif paper_decision in {WATCH, BLOCK, PORTFOLIO_BLOCK} and paper_reason in {
        T.PORTFOLIO_CONCENTRATION, T.CORRELATION_LIMIT, T.MAX_PORTFOLIO_RISK, T.SECTOR_CAP,
    }:
        decision = T.WAIT_DECISION
        candidate_state = T.WAIT
        reason_code = paper_reason
        reason = "Investment thesis can wait on portfolio capacity."
    else:
        decision = T.AVOID
        candidate_state = T.REJECTED
        reason_code = paper_reason or T.LOW_QUALITY_SETUP
        reason = str(paper.detail or "Not enough independent evidence to take risk.")

    if decision == T.BUY and entry_state != T.ENTER_NOW:
        decision = T.WAIT_DECISION
        candidate_state = T.WAIT
        reason_code = T.ENTRY_NOT_TRIGGERED
        reason = "Committee will not mark READY without a valid entry."

    execution = T.NOT_APPLICABLE
    if decision == T.BUY:
        if not broker_ok:
            execution = T.BLOCKED_BROKER_AUTH
        elif not entry_window:
            execution = T.BLOCKED_WINDOW
        else:
            execution = T.ELIGIBLE
    elif candidate_state in {T.WAIT, T.WAIT_EVIDENCE, T.WATCH}:
        execution = T.NOT_APPLICABLE

    positives = []
    for name, status in families.items():
        if status == "PASS":
            positives.append(name)
    positives.extend(f"method {m}" for m in methods_buy[:4])

    trigger = _wait_trigger(entry_state, card, reason_code) if decision == T.WAIT_DECISION else {}

    return CommitteeRecord(
        symbol=symbol,
        decision=decision,
        candidate_state=candidate_state,
        entry_state=entry_state,
        execution_state=execution,
        reason_code=reason_code,
        reason=reason,
        tier=tier,
        families=families,
        methods_buy=methods_buy,
        methods_wait=methods_wait,
        methods_avoid=methods_avoid,
        methods_unknown=methods_unknown,
        vetoes=hard,
        missing_critical=missing_critical,
        evidence_coverage_pct=coverage_f,
        information_value=info_value,
        research_required=research_required,
        framework_id=str(research.get("framework_id") or ""),
        wait_trigger=trigger,
        positives=positives[:8],
        paper_decision=paper_decision,
        paper_reason=paper_reason,
        entry=_f(card.get("entry")),
        stop=_f(card.get("stop")),
        target=_f(card.get("target")),
        references={
            "reco_tier": tier,
            "primary_thesis": card.get("primary_thesis"),
            "research": {k: research.get(k) for k in (
                "framework_id", "framework_label", "coverage_pct", "quality_label",
                "acquired_at", "vs_technical",
            ) if research.get(k) not in (None, "")},
        },
    )


def evaluate_many(cards: list[Mapping[str, Any]], **kwargs: Any) -> list[CommitteeRecord]:
    out = []
    for card in cards:
        if not card.get("symbol"):
            continue
        try:
            out.append(evaluate_committee(card, **kwargs))
        except Exception as exc:
            out.append(CommitteeRecord(
                symbol=str(card.get("symbol") or "").upper(),
                decision=T.AVOID,
                candidate_state=T.REJECTED,
                entry_state=T.NO_TRIGGER,
                execution_state=T.NOT_APPLICABLE,
                reason_code=T.DATA_INTEGRITY,
                reason=str(exc)[:200],
            ))
    return out
