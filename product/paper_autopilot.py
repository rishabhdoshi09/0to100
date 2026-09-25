"""Recommendation-driven paper autopilot.

This is the missing money-path handoff:

  saved recommendations → selection authority → evidence policies →
  portfolio / entry gates → TradeIntent → PaperBook / PaperExecutionPipeline

It does not scan the market and does not invent a BUY. Watch/Avoid never auto-enter.
Every skip has a machine-readable reason_code. Live money stays locked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from product.reco_ensemble import TIER_GOOD, TIER_HIGH, TIER_WATCH
from product.strategy_catalog import ENSEMBLE_ID, ensemble_identity

# ── machine-readable rejection / wait codes ──────────────────────────────────
PAPER_TRADING_DISABLED = "PAPER_TRADING_DISABLED"
MARKET_NOT_READY = "MARKET_NOT_READY"
OUTSIDE_ENTRY_WINDOW = "OUTSIDE_ENTRY_WINDOW"
STALE_RECOMMENDATION = "STALE_RECOMMENDATION"
LOW_QUALITY_SETUP = "LOW_QUALITY_SETUP"
INVALID_SYMBOL = "INVALID_SYMBOL"
DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
ANALYSIS_ERROR = "ANALYSIS_ERROR"
WATCH_ONLY = "WATCH_ONLY"
DD_GATE_FAILED = "DD_GATE_FAILED"
EMPIRICAL_GATE_FAILED = "EMPIRICAL_GATE_FAILED"
EVIDENCE_POLICY_BLOCK = "EVIDENCE_POLICY_BLOCK"
HISTORICAL_EVIDENCE_PENDING = "HISTORICAL_EVIDENCE_PENDING"
ENTRY_TOO_EXTENDED = "ENTRY_TOO_EXTENDED"
NO_VALID_ENTRY = "NO_VALID_ENTRY"
INVALID_STOP = "INVALID_STOP"
DUPLICATE_POSITION = "DUPLICATE_POSITION"
MAX_POSITIONS = "MAX_POSITIONS"
MAX_PORTFOLIO_RISK = "MAX_PORTFOLIO_RISK"
SECTOR_CAP = "SECTOR_CAP"
CORRELATION_CAP = "CORRELATION_CAP"
PER_NAME_CAP = "PER_NAME_CAP"
INSUFFICIENT_CAPITAL = "INSUFFICIENT_CAPITAL"
LIQUIDITY_FAILED = "LIQUIDITY_FAILED"
REGIME_STANDDOWN = "REGIME_STANDDOWN"
PORTFOLIO_GATE_ERROR = "PORTFOLIO_GATE_ERROR"
UNRECONCILED = "UNRECONCILED"
NO_TRADE = "NO_TRADE"
WAIT_FOR_ENTRY = "WAIT_FOR_ENTRY"
NOT_SURFACED = "NOT_SURFACED"
BROKER_LOGIN_REQUIRED = "BROKER_LOGIN_REQUIRED"
DECISION_FINGERPRINT_FAILED = "DECISION_FINGERPRINT_FAILED"

ENTER_NOW = "ENTER_NOW"
WAIT = "WAIT"
WATCH = "WATCH"
BLOCK = "BLOCK"
PORTFOLIO_BLOCK = "PORTFOLIO_BLOCK"

ELIGIBLE_TIERS = {TIER_HIGH, TIER_GOOD}
STALE_MAX_AGE = timedelta(hours=36)
HOLD_DAYS = 20
DEFAULT_RISK_PCT = 1.0  # percentage points, matching PaperBook


def _f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return out


def _parse_ts(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass
class AutopilotDecision:
    symbol: str
    decision: str
    reason_code: str
    detail: str = ""
    card: dict[str, Any] = field(default_factory=dict)
    selection_score: float | None = None
    policy_effect: str = "NEUTRAL"
    intent: Any = None
    context: dict[str, Any] = field(default_factory=dict)
    breakdown: dict[str, Any] = field(default_factory=dict)
    why: dict[str, Any] = field(default_factory=dict)
    group: str = ""
    portfolio: dict[str, Any] = field(default_factory=dict)
    freeze_id: str = ""
    evidence_fingerprint: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "decision": self.decision,
            "reason_code": self.reason_code,
            "detail": self.detail,
            "tier": self.card.get("reco_tier"),
            "entry_state": self.card.get("entry_state"),
            "setup_label": self.card.get("setup_label"),
            "sector": self.card.get("sector"),
            "primary_thesis": self.card.get("primary_thesis"),
            "selection_score": self.selection_score,
            "policy_effect": self.policy_effect,
            "entry": self.card.get("entry"),
            "stop": self.card.get("stop"),
            "target": self.card.get("target"),
            "group": self.group,
            "why": self.why,
            "breakdown": self.breakdown,
            "regime": (self.context or {}).get("regime"),
            "dd_status": (self.context or {}).get("dd_status"),
            "entry_quality": (self.context or {}).get("entry_quality"),
            "missing_evidence": (self.context or {}).get("missing_evidence") or [],
            "portfolio_authority": self.portfolio or None,
            "freeze_id": self.freeze_id,
            "evidence_fingerprint": self.evidence_fingerprint,
        }


def _identity() -> dict[str, Any]:
    try:
        return ensemble_identity()
    except Exception:
        return {"strategy_id": ENSEMBLE_ID, "strategy_version": 1, "rules_hash": "unverified"}


def reco_is_stale(workspace: Mapping[str, Any] | None, *, now: datetime | None = None) -> bool:
    payload = dict(workspace or {})
    if payload.get("point_in_time"):
        # Historical reconstructions are dated to T, not to wall-clock freshness.
        return False
    stamp = (
        _parse_ts(payload.get("generated_at"))
        or _parse_ts(payload.get("scan_scanned_at"))
        or _parse_ts(payload.get("scanned_at"))
    )
    if stamp is None:
        return True
    clock = now or datetime.now(timezone.utc)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=timezone.utc)
    return (clock - stamp) > STALE_MAX_AGE


def _dd_status(card: Mapping[str, Any]) -> str:
    explicit = str(card.get("dd_verdict") or card.get("dd_status") or "").strip().upper()
    if explicit:
        return explicit
    for method in card.get("methods") or []:
        if str(method.get("id") or "") == "funds":
            return str(method.get("status") or "unknown").upper()
    return "UNKNOWN"


def _empirical_failed_methods(card: Mapping[str, Any]) -> tuple[str, ...]:
    failed: list[str] = []
    for method in card.get("methods") or []:
        mid = str(method.get("id") or "")
        if (
            mid in {"ev", "case"}
            and str(method.get("status") or "").lower() == "fail"
            and mid not in failed
        ):
            failed.append(mid)
    return tuple(sorted(failed))


def _empirical_fail(card: Mapping[str, Any]) -> bool:
    """Hard-block only a real empirical consensus failure.

    Forward paper trading is itself an evidence-acquisition lane. A single
    negative historical stream (EV *or* case memory) should reduce selection
    priority, not permanently prevent the system from collecting fresh forward
    evidence. Explicit policy blocks remain authoritative, and agreement from
    both empirical streams is still a hard veto.
    """
    if bool(card.get("empirical_block")):
        return True
    return set(_empirical_failed_methods(card)) == {"ev", "case"}


def selection_score(card: Mapping[str, Any], policy: Mapping[str, Any] | None = None) -> float:
    """Rank among already-eligible names. The learner can only reorder them."""
    from product.decision_context import score_breakdown
    base = float(score_breakdown(card, policy).get("selection_rank") or 0.0)

    # One negative empirical stream is an exploration warning, not a paper-entry
    # veto. Penalise it so clean-evidence setups rank first while still allowing
    # forward-paper evidence to accumulate when no stronger setup exists.
    empirical_penalty = 15.0 if len(_empirical_failed_methods(card)) == 1 else 0.0
    try:
        from product.challenger_learning import paper_selection_adjustment
        learned = paper_selection_adjustment(card)
        return base - empirical_penalty + float(learned.get("adjustment") or 0.0)
    except Exception:
        return base - empirical_penalty


def carried_sector_risk(book) -> tuple[dict[str, float], dict[str, float]]:
    """Risk already carried by the persisted paper book, keyed by sector.

    Both the family and current correlation-cluster fallback use sector until a
    stronger persisted cluster identity is available. Keeping this calculation
    in one seam prevents discovery, present paper and historical paper from
    disagreeing after restart.
    """
    family: dict[str, float] = {}
    cluster: dict[str, float] = {}
    if book is None:
        return family, cluster
    cap = float(getattr(book, "capital", 0.0) or 0.0)
    for pos in (getattr(book, "open", {}) or {}).values():
        sector = str(getattr(pos, "sector", "") or "")
        if not sector:
            continue
        approved = _f(getattr(pos, "approved_risk_pct", None))
        if approved is None and cap > 0:
            approved = float(getattr(pos, "risk_amount", 0.0) or 0.0) / cap * 100.0
        risk_pct = float(approved or 0.0)
        family[sector] = family.get(sector, 0.0) + risk_pct
        cluster[sector] = cluster.get(sector, 0.0) + risk_pct
    return family, cluster


def _group_for(decision: str) -> str:
    if decision == ENTER_NOW:
        return "TAKEN"
    if decision == WAIT:
        return "RECOMMENDED_BUT_NOT_FILLED"
    if decision == WATCH:
        return "REJECTED"
    if decision in {BLOCK, PORTFOLIO_BLOCK, NO_TRADE}:
        return "REJECTED"
    return "REJECTED"


def _decorate(decision: AutopilotDecision, *, policy: Mapping[str, Any] | None, context: Mapping[str, Any] | None) -> AutopilotDecision:
    from product.decision_context import explain, score_breakdown
    decision.context = dict(context or {})
    decision.policy_effect = str((policy or {}).get("final_effect") or decision.policy_effect or "NEUTRAL")
    decision.breakdown = score_breakdown(decision.card, policy, context)
    base_rank = float(decision.breakdown.get("selection_rank") or 0.0)
    try:
        from product.challenger_learning import paper_selection_adjustment
        learned = paper_selection_adjustment(decision.card)
    except Exception:
        learned = {
            "available": False,
            "affects_selection": False,
            "adjustment": 0.0,
            "live_locked": True,
        }
    decision.breakdown["learning_challenger"] = learned
    decision.selection_score = base_rank + float(learned.get("adjustment") or 0.0)
    decision.why = explain(
        decision=decision.decision,
        reason_code=decision.reason_code,
        card=decision.card,
        context=context,
        policy=policy,
        breakdown=decision.breakdown,
    )
    decision.group = _group_for(decision.decision)
    return decision


def evaluate_candidate(
    card: Mapping[str, Any],
    *,
    book,
    entries_allowed: bool = True,
    entry_block_reason: str = "",
    paper_enabled: bool = True,
    workspace: Mapping[str, Any] | None = None,
    now: datetime | None = None,
    regime: str = "RISK_ON",
    policy: Mapping[str, Any] | None = None,
    family_risk: dict | None = None,
    cluster_risk: dict | None = None,
) -> AutopilotDecision:
    """Gate one recommendation card. First hard-block wins. No silent skip."""
    symbol = str(card.get("symbol") or "").strip().upper()
    row = dict(card)
    row["symbol"] = symbol
    try:
        from data.nse_universe import _is_valid_symbol
        plausible = bool(symbol) and _is_valid_symbol(symbol)
    except Exception:
        plausible = bool(symbol) and len(symbol) <= 15 and " " not in symbol
    if not symbol or not plausible:
        return AutopilotDecision(
            symbol, BLOCK, INVALID_SYMBOL,
            "not a valid NSE equity ticker" if symbol else "symbol missing",
            row,
        )
    status = str(row.get("status") or row.get("data_status") or "").upper()
    if status in {DATA_UNAVAILABLE, INVALID_SYMBOL, ANALYSIS_ERROR}:
        return AutopilotDecision(symbol, BLOCK, status, str(row.get("reason") or status), row)
    if not paper_enabled:
        return AutopilotDecision(symbol, BLOCK, PAPER_TRADING_DISABLED, "paper auto is off", row)
    if not entries_allowed:
        code = str(entry_block_reason or OUTSIDE_ENTRY_WINDOW).upper()
        if code in {"ENTRY_WINDOW_CLOSED", "OUTSIDE_ENTRY_WINDOW"}:
            code = OUTSIDE_ENTRY_WINDOW
        if code in {"CAPABILITY_BLOCKED", "MARKET_NOT_READY"}:
            code = MARKET_NOT_READY if "DATA" in str(entry_block_reason).upper() else code
        return AutopilotDecision(symbol, BLOCK, code or OUTSIDE_ENTRY_WINDOW, entry_block_reason, row)
    if regime == "RISK_OFF":
        return AutopilotDecision(symbol, BLOCK, REGIME_STANDDOWN, "regime is RISK_OFF", row)
    if workspace is not None and reco_is_stale(workspace, now=now):
        return AutopilotDecision(symbol, BLOCK, STALE_RECOMMENDATION, "recommendation file is stale", row)

    tier = str(row.get("reco_tier") or TIER_WATCH)
    if tier not in ELIGIBLE_TIERS:
        return AutopilotDecision(symbol, WATCH, LOW_QUALITY_SETUP, f"tier={tier} is not auto-enter", row)

    entry_state = str(row.get("entry_state") or "")
    if bool(row.get("chase_risk")) or entry_state == "extended":
        return AutopilotDecision(symbol, WAIT, ENTRY_TOO_EXTENDED, "chase/extension block", row)
    if entry_state in {"near_setup", "watch"}:
        return AutopilotDecision(symbol, WAIT, WAIT_FOR_ENTRY, f"entry_state={entry_state}", row)
    if entry_state == "broken":
        return AutopilotDecision(symbol, BLOCK, NO_VALID_ENTRY, "structure broken", row)

    dd = _dd_status(row)
    if dd in {"FAIL", "FAILED", "BLOCK", "AVOID"}:
        return AutopilotDecision(symbol, BLOCK, DD_GATE_FAILED, f"dd={dd}", row)

    failed_empirical = _empirical_failed_methods(row)
    if _empirical_fail(row):
        detail = (
            "explicit empirical policy block"
            if bool(row.get("empirical_block"))
            else "empirical consensus failed: " + "+".join(failed_empirical)
        )
        return AutopilotDecision(symbol, BLOCK, EMPIRICAL_GATE_FAILED, detail, row)
    if failed_empirical:
        row["paper_evidence_mode"] = "EXPLORATORY_EMPIRICAL_CONFLICT"
        row["empirical_conflict_methods"] = list(failed_empirical)

    policy = dict(policy or {})
    if str(policy.get("final_effect") or "") == "BLOCK":
        historical = dict(policy.get("historical_forward_confidence") or {})
        if historical.get("required") and not historical.get("paper_eligible"):
            stage = str(historical.get("confidence_stage") or "HISTORICAL_EVIDENCE")
            ready = int(historical.get("paper_ready_setups") or 0)
            return AutopilotDecision(
                symbol, BLOCK, HISTORICAL_EVIDENCE_PENDING,
                f"history-first paper gate pending · {stage} · paper_ready_setups={ready}",
                row, policy_effect="BLOCK",
            )
        return AutopilotDecision(
            symbol, BLOCK, EVIDENCE_POLICY_BLOCK,
            "active learning policy blocks this setup", row, policy_effect="BLOCK",
        )

    entry = _f(row.get("entry") or row.get("entry_price") or row.get("cmp"))
    stop = _f(row.get("stop") or row.get("stop_price"))
    target = _f(row.get("target") or row.get("target_price"))
    if entry is None or entry <= 0:
        return AutopilotDecision(symbol, BLOCK, NO_VALID_ENTRY, "missing entry", row)
    if stop is None or stop <= 0:
        return AutopilotDecision(symbol, BLOCK, INVALID_STOP, "missing stop", row)
    if stop >= entry:
        return AutopilotDecision(symbol, BLOCK, INVALID_STOP, f"stop {stop} >= entry {entry}", row)
    if target is None or target <= entry:
        return AutopilotDecision(symbol, WAIT, NO_VALID_ENTRY, "missing/invalid target", row)

    vol = _f(row.get("volume_ratio"))
    if vol is not None and vol < 0.7:
        return AutopilotDecision(symbol, BLOCK, LIQUIDITY_FAILED, f"volume_ratio={vol}", row)

    if book is not None:
        if any(getattr(p, "symbol", "") == symbol for p in getattr(book, "open", {}).values()):
            return AutopilotDecision(symbol, BLOCK, DUPLICATE_POSITION, "already held", row)
        if len(getattr(book, "open", {})) >= int(getattr(book, "max_positions", 5)):
            return AutopilotDecision(symbol, PORTFOLIO_BLOCK, MAX_POSITIONS, "max positions", row)

        from research.intelligence.schemas import TradeIntent
        ident = _identity()
        probe = TradeIntent(
            strategy_id=ident["strategy_id"],
            strategy_version=int(ident.get("strategy_version") or 1),
            rules_hash=str(ident.get("rules_hash") or ""),
            data_snapshot_id=str((workspace or {}).get("scan_scanned_at") or "reco"),
            source="selection_authority",
            event_ts=(now or datetime.now(timezone.utc)).isoformat(),
            symbol=symbol,
            intended_entry=entry,
            intended_risk_pct=DEFAULT_RISK_PCT,
            stop_price=stop,
            target_price=target,
            holding_horizon_days=HOLD_DAYS,
        )
        try:
            from research.intelligence.runtime import portfolio_gate as PG
            cfg = SimpleNamespace(
                max_family_risk_pct=2.5,
                max_cluster_risk_pct=3.0,
            )
            gate = PG.check(
                probe,
                family=str(row.get("sector") or ""),
                book=book,
                family_risk=dict(family_risk or {}),
                cluster_risk=dict(cluster_risk or {}),
                cluster_of=str(row.get("sector") or ""),
                cfg=cfg,
                regime=regime,
                data_ok=True,
                reconciled=True,
            )
            if not gate.ok:
                mapped = {
                    "DUPLICATE_SYMBOL": DUPLICATE_POSITION,
                    "FAMILY_CAP": SECTOR_CAP,
                    "CLUSTER_CAP": CORRELATION_CAP,
                    "MAX_POSITIONS": MAX_POSITIONS,
                    "REGIME_STANDDOWN": REGIME_STANDDOWN,
                    "NO_DATA": MARKET_NOT_READY,
                    "UNRECONCILED": UNRECONCILED,
                }.get(gate.reason_code, gate.reason_code or PORTFOLIO_BLOCK)
                kind = PORTFOLIO_BLOCK if mapped in {
                    SECTOR_CAP, CORRELATION_CAP, MAX_POSITIONS, MAX_PORTFOLIO_RISK,
                    UNRECONCILED, PORTFOLIO_GATE_ERROR,
                } else BLOCK
                return AutopilotDecision(symbol, kind, mapped, gate.detail, row)
        except Exception as exc:
            return AutopilotDecision(
                symbol, PORTFOLIO_BLOCK, PORTFOLIO_GATE_ERROR,
                str(exc)[:200], row,
            )

        from research.intelligence.runtime.position_sizing import size_long_cash
        sizing = size_long_cash(
            capital=float(getattr(book, "capital", 0.0) or 0.0),
            entry=entry,
            stop=stop,
            requested_risk_pct=DEFAULT_RISK_PCT,
            max_risk_fraction=float(getattr(book, "risk_per_trade_pct", 0.01) or 0.01),
            max_position_fraction=float(getattr(book, "max_position_pct", 0.10) or 0.10),
            slippage_bps=float(getattr(book, "slippage_bps", 0.0) or 0.0),
        )
        if not sizing.ok:
            code = {
                "INVALID_ENTRY_STOP": INVALID_STOP,
                "INVALID_EFFECTIVE_RISK": INVALID_STOP,
                "NON_POSITIVE_CAPITAL": INSUFFICIENT_CAPITAL,
                "RISK_BUDGET_TOO_SMALL": INSUFFICIENT_CAPITAL,
                "POSITION_CAP_TOO_SMALL": PER_NAME_CAP,
                "QUANTITY_EXCEEDS_APPROVED_LIMIT": PER_NAME_CAP,
            }.get(sizing.reason_code, INSUFFICIENT_CAPITAL)
            return AutopilotDecision(symbol, PORTFOLIO_BLOCK, code, sizing.reason_code, row)
        open_risk = float(book.open_risk()) if hasattr(book, "open_risk") else 0.0
        cap = float(getattr(book, "capital", 0.0) or 0.0)
        max_total = cap * float(getattr(book, "max_total_risk_pct", 0.05) or 0.05)
        if open_risk + sizing.risk_amount > max_total + 1e-6:
            return AutopilotDecision(
                symbol, PORTFOLIO_BLOCK, MAX_PORTFOLIO_RISK, "total open risk cap", row,
            )
        cash = cap + float(getattr(book, "realized_pnl", 0.0) or 0.0)
        notional = sizing.effective_entry * sizing.quantity
        if notional > cash + 1e-6:
            return AutopilotDecision(
                symbol, PORTFOLIO_BLOCK, INSUFFICIENT_CAPITAL,
                f"need {notional:.0f} have {cash:.0f}", row,
            )
        row["approved_quantity"] = int(sizing.quantity)
        row["approved_risk_pct"] = float(sizing.actual_risk_pct)

    row["entry"] = entry
    row["stop"] = stop
    row["target"] = target
    score = selection_score(row, policy)
    return AutopilotDecision(
        symbol, ENTER_NOW, "ELIGIBLE", "passed all gates", row,
        selection_score=score,
        policy_effect=str(policy.get("final_effect") or "NEUTRAL"),
    )


def evaluate_selection_candidate(
    card: Mapping[str, Any],
    *,
    book=None,
    workspace: Mapping[str, Any] | None = None,
    now: datetime | None = None,
    entries_allowed: bool = True,
    entry_block_reason: str = "",
    paper_enabled: bool = True,
    regime: str = "RISK_ON",
    policy_path=None,
    policies: Sequence[Mapping[str, Any]] | None = None,
    enforce_history: bool | None = None,
    family_risk: dict | None = None,
    cluster_risk: dict | None = None,
) -> AutopilotDecision:
    """Canonical selection-thesis evaluation without execution.

    Current best-trade discovery, historical PIT replay and present paper
    trading all call this seam. enforce_history=False is reserved for
    discovery/replay because those lanes produce the prerequisite historical
    evidence; PAPER_FORWARD uses the production default (history gate on).
    """
    from product.decision_context import snapshot
    from product.evidence_policy_engine import evaluate_policies

    ctx = snapshot(card, book=book, regime=regime)
    merged = dict(card)
    for key, value in ctx.items():
        if key == "methods":
            continue
        merged.setdefault(key, value)
    policy = evaluate_policies(
        merged,
        policies=policies,
        path=policy_path,
        regime=regime,
        book=book,
        enforce_history=enforce_history,
    )
    decision = evaluate_candidate(
        merged,
        book=book,
        entries_allowed=entries_allowed,
        entry_block_reason=entry_block_reason,
        paper_enabled=paper_enabled,
        workspace=workspace,
        now=now,
        regime=regime,
        policy=policy,
        family_risk=family_risk,
        cluster_risk=cluster_risk,
    )
    return _decorate(decision, policy=policy, context=ctx)


def _canonical_decision(decision: AutopilotDecision, *, as_of: str, snapshot_id: str):
    """The canonical Decision behind this autopilot decision.

    Built through the one adapter rather than re-reading card keys here, so the
    decision the paper book records is the same object the desk published and
    the UI explained. Never raises: a paper cycle must not stop because a
    decision could not be canonicalised, it simply records no linkage and its
    outcome cannot become conditional evidence.
    """
    try:
        from product.decision_adapter import decision_from_card
        from product.decision_ranking import decision_context_key
        from product.evidence_class import PAPER_FORWARD

        canonical = decision_from_card(
            decision.card,
            source_scan_id=str(snapshot_id or ""),
            market_state=str(decision.card.get("market_state") or ""),
            sector_state=str(decision.card.get("sector_state") or ""),
            evidence_class=PAPER_FORWARD,
            generated_at=str(snapshot_id or as_of or ""),
        )
        try:
            from dataclasses import replace
            from product.trading_thesis import manifest as thesis_manifest
            canonical = replace(
                canonical,
                provenance={
                    **dict(canonical.provenance or {}),
                    "thesis_hash": str(thesis_manifest().get("thesis_hash") or ""),
                },
            )
        except Exception:
            pass
        try:
            from product.evidence_intelligence import enrich
            canonical = enrich(canonical)
        except Exception:
            pass
        return canonical.decision_id, decision_context_key(canonical)
    except Exception:
        return "", ""


def _decision_fingerprint_evidence(
    decision: AutopilotDecision,
    *,
    as_of: str,
    snapshot_id: str,
    regime: str,
    thesis_hash: str = "",
) -> dict[str, Any]:
    canonical_id, context_key = _canonical_decision(
        decision,
        as_of=as_of,
        snapshot_id=snapshot_id,
    )
    try:
        from product.pit_versions import current_versions
        versions = current_versions().as_dict()
    except Exception:
        versions = {}
    try:
        from product.evidence_class import PAPER_FORWARD
        evidence_class = PAPER_FORWARD
    except Exception:
        evidence_class = "PAPER_FORWARD"
    ctx = dict(decision.context or {})
    card = dict(decision.card or {})
    return {
        **ctx,
        "decision_id": canonical_id,
        "context_key": context_key,
        "symbol": decision.symbol,
        "decision": decision.decision,
        "reason_code": decision.reason_code,
        "entry": _f(card.get("entry") or card.get("entry_price") or card.get("cmp")),
        "stop": _f(card.get("stop") or card.get("stop_price")),
        "target": _f(card.get("target") or card.get("target_price")),
        "setup_label": card.get("setup_label") or card.get("primary_thesis"),
        "sector": card.get("sector"),
        "regime": regime,
        "selection_score": decision.selection_score,
        "policy_effect": decision.policy_effect,
        "portfolio": decision.portfolio or ctx.get("portfolio"),
        "rules_hash": _identity().get("rules_hash"),
        "thesis_hash": str(thesis_hash or card.get("thesis_hash") or ""),
        "calibration_snapshot_id": str(
            card.get("calibration_snapshot_id")
            or ctx.get("calibration_snapshot_id")
            or ""
        ),
        "data_snapshot_id": str(snapshot_id or ""),
        "source_scan_id": str(snapshot_id or ""),
        "evidence_class": evidence_class,
        "versions": versions,
    }



def _freeze_taken_decision(
    decision: AutopilotDecision,
    *,
    as_of: str,
    snapshot_id: str,
    regime: str,
    thesis_hash: str = "",
) -> dict[str, Any]:
    from product.decision_freeze import freeze

    evidence = _decision_fingerprint_evidence(
        decision,
        as_of=as_of,
        snapshot_id=snapshot_id,
        regime=regime,
        thesis_hash=thesis_hash,
    )
    frozen = freeze(evidence)
    decision.freeze_id = str(frozen.get("freeze_id") or "")
    decision.evidence_fingerprint = str(frozen.get("fingerprint") or "")
    return frozen



def _intent_for(decision: AutopilotDecision, *, as_of: str, snapshot_id: str):
    from research.intelligence.schemas import TradeIntent
    ident = _identity()
    card = decision.card
    decision_id, context_key = _canonical_decision(
        decision, as_of=as_of, snapshot_id=snapshot_id
    )
    return TradeIntent(
        strategy_id=ident["strategy_id"],
        strategy_version=int(ident.get("strategy_version") or 1),
        rules_hash=str(ident.get("rules_hash") or ""),
        data_snapshot_id=snapshot_id or "reco",
        source="selection_authority",
        event_ts=as_of,
        cycle_id=f"reco:{as_of}",
        symbol=decision.symbol,
        intended_entry=float(_f(card.get("entry")) or _f(card.get("cmp")) or 0.0),
        intended_risk_pct=DEFAULT_RISK_PCT,
        stop_price=float(_f(card.get("stop")) or 0.0),
        target_price=float(_f(card.get("target")) or 0.0),
        holding_horizon_days=HOLD_DAYS,
        target_portfolio_id=f"reco-portfolio:{snapshot_id}",
        target_position_id=f"reco-position:{decision.symbol}:{snapshot_id}",
        current_quantity=0,
        desired_quantity=int(_f(card.get("approved_quantity")) or 0),
        required_quantity=int(_f(card.get("approved_quantity")) or 0),
        entry_rule="recommendation_entry",
        stop_rule="recommendation_stop",
        exit_rule="recommendation_target",
        reasons=(decision.reason_code, str(card.get("primary_thesis") or "")),
        decision_id=decision_id,
        context_key=context_key,
    )


def execute_paper_decision(decision: AutopilotDecision, *, book, as_of: str, snapshot_id: str, runtime_state=None):
    """Paper venue only. Live adapter is a separate class and stays locked."""
    intent = _intent_for(decision, as_of=as_of, snapshot_id=snapshot_id)
    decision.intent = intent
    store = getattr(getattr(book, "_pipeline", None), "events", None)
    if store is not None and hasattr(store, "append"):
        try:
            store.append(intent)
        except Exception:
            pass
    if hasattr(book, "open_intent"):
        return book.open_intent(intent, date=as_of)
    return book.open_position(
        intent.strategy_id,
        intent.symbol,
        float(intent.intended_entry),
        float(intent.stop_price),
        float(intent.target_price),
        as_of,
        int(intent.holding_horizon_days),
        risk_pct_of_capital=float(intent.intended_risk_pct),
        decision_id=str(getattr(intent, "decision_id", "") or ""),
        paper_intent_id=str(getattr(intent, "record_id", "") or ""),
        context_key=str(getattr(intent, "context_key", "") or ""),
    )


def _execute(decision: AutopilotDecision, *, book, as_of: str, snapshot_id: str, runtime_state=None):
    from product.execution_adapter import default_adapter
    return default_adapter(live=False).submit(
        decision, book=book, as_of=as_of, snapshot_id=snapshot_id,
    )


def run_reco_paper_cycle(
    *,
    book,
    workspace: Mapping[str, Any] | None = None,
    cards: Sequence[Mapping[str, Any]] | None = None,
    as_of: str = "",
    now: datetime | None = None,
    entries_allowed: bool = True,
    entry_block_reason: str = "",
    session_phase: str = "",
    paper_enabled: bool = True,
    regime: str = "RISK_ON",
    persist_journal: bool = True,
    max_new: int = 3,
    policy_path=None,
    policies: Sequence[Mapping[str, Any]] | None = None,
    enforce_history: bool | None = None,
    scan_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Consume saved recommendations and open paper positions for ENTER_NOW names.

    Does not mock or bypass risk. Returns a cycle dict the supervisor can merge.
    """
    from product.autopilot_journal import flatten_cards, record_cycle

    clock = now or datetime.now(timezone.utc)
    day = as_of or clock.date().isoformat()
    ident = _identity()
    try:
        from product.trading_thesis import manifest as thesis_manifest
        thesis = thesis_manifest()
    except Exception:
        thesis = {}
    payload = dict(workspace or {})
    if cards is None:
        if not payload:
            # load_recommendations() already fails closed to None for the normal
            # "no file yet" / corrupt-JSON / schema-mismatch cases -- it never
            # raises for those. Do NOT also swallow a genuine exception here
            # (e.g. a broken import, a real code bug): callers of
            # run_reco_paper_cycle (research/autonomy/jobs.py,
            # research/autonomy/paper_cycle_truth.py) are specifically built to
            # catch that and classify it as a system/execution failure rather
            # than a quiet no-opportunity day. Catching it here first used to
            # turn a provider/code failure into an indistinguishable
            # NO_ELIGIBLE_TRADE cycle with an empty card_list -- exactly the
            # "system failure disguised as a valid decision" this pipeline is
            # supposed to prevent.
            from product.recommendations_store import load_recommendations
            payload = load_recommendations() or {}
        card_list = flatten_cards(payload)
    else:
        card_list = [dict(c) for c in cards if isinstance(c, Mapping)]

    decisions: list[AutopilotDecision] = []
    taken: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    waits: list[dict[str, Any]] = []
    not_surfaced: list[dict[str, Any]] = []
    opened: list[Any] = []
    family_risk, cluster_risk = carried_sector_risk(book)
    cycle_reasons: list[str] = []

    if not paper_enabled:
        cycle_reasons.append(PAPER_TRADING_DISABLED)
    if not entries_allowed:
        cycle_reasons.append(str(entry_block_reason or OUTSIDE_ENTRY_WINDOW))
    if payload and reco_is_stale(payload, now=clock):
        cycle_reasons.append(STALE_RECOMMENDATION)
    if not card_list and not cycle_reasons:
        cycle_reasons.append(NOT_SURFACED)

    def _freeze(decision: AutopilotDecision, *, group: str = "") -> None:
        try:
            from product.decision_taxonomy import is_non_judgment
            from product.counterfactual_learning import freeze_decision
            if is_non_judgment(decision.decision, decision.reason_code):
                return
            evidence = _decision_fingerprint_evidence(
                decision,
                as_of=day,
                snapshot_id=str(payload.get("scan_scanned_at") or day),
                regime=regime,
                thesis_hash=str(thesis.get("thesis_hash") or ""),
            )
            evidence.update({
                "group": group or decision.group,
                "detail": decision.detail,
                "why": decision.why,
            })
            frozen = freeze_decision(
                symbol=decision.symbol,
                reason_code=decision.reason_code,
                decision=decision.decision,
                entry=_f(decision.card.get("entry")),
                stop=_f(decision.card.get("stop")),
                target=_f(decision.card.get("target")),
                as_of=day,
                evidence=evidence,
            )
            decision.freeze_id = str(
                frozen.get("canonical_freeze_id")
                or frozen.get("counterfactual_id")
                or ""
            )
            decision.evidence_fingerprint = str(frozen.get("decision_fingerprint") or "")
        except Exception:
            pass

    ranked: list[tuple[float, AutopilotDecision]] = []
    for card in card_list:
        decision = evaluate_selection_candidate(
            card,
            book=book,
            workspace=payload or None,
            now=clock,
            entries_allowed=entries_allowed,
            entry_block_reason=entry_block_reason,
            paper_enabled=paper_enabled,
            regime=regime,
            policy_path=policy_path,
            policies=policies,
            enforce_history=enforce_history,
            family_risk=family_risk,
            cluster_risk=cluster_risk,
        )
        decisions.append(decision)
        from product.decision_taxonomy import is_non_judgment
        if is_non_judgment(decision.decision, decision.reason_code):
            continue
        if decision.decision == ENTER_NOW:
            ranked.append((float(decision.selection_score or 0.0), decision))
        elif decision.decision == WAIT:
            _freeze(decision, group="RECOMMENDED_BUT_NOT_FILLED")
            waits.append(decision.as_dict())
        else:
            _freeze(decision, group="REJECTED")
            rejections.append(decision.as_dict())

    ranked.sort(key=lambda item: (-item[0], item[1].symbol))
    try:
        from product.portfolio_selection_authority import apply_portfolio_authority
        ranked, diverted = apply_portfolio_authority(
            ranked, book=book, max_new=max_new, regime=regime,
        )
        for decision in diverted:
            if str(getattr(decision, "decision", "")) == WAIT:
                _freeze(decision, group="RECOMMENDED_BUT_NOT_FILLED")
                row = decision.as_dict() if hasattr(decision, "as_dict") else dict(decision)
                waits.append(row)
            else:
                _freeze(decision, group="REJECTED")
                row = decision.as_dict() if hasattr(decision, "as_dict") else dict(decision)
                rejections.append(row)
    except Exception:
        diverted = []
    snapshot_id = str(payload.get("scan_scanned_at") or day)
    entered = 0
    for _score, decision in ranked:
        if entered >= int(max_new):
            decision.decision = NO_TRADE
            decision.reason_code = NO_TRADE
            decision.detail = "not top-of-the-top this cycle"
            _freeze(decision, group="REJECTED")
            leftover = dict(decision.as_dict())
            leftover["group"] = "REJECTED"
            rejections.append(leftover)
            continue

        # Every BUY must have a durable immutable fingerprint before the
        # PaperBook can be mutated. Identity/provenance failure is therefore a
        # real safety block, not a warning attached after the fill.
        try:
            _freeze_taken_decision(
                decision,
                as_of=day,
                snapshot_id=snapshot_id,
                regime=regime,
                thesis_hash=str(thesis.get("thesis_hash") or ""),
            )
        except Exception as exc:
            if DECISION_FINGERPRINT_FAILED not in cycle_reasons:
                cycle_reasons.append(DECISION_FINGERPRINT_FAILED)
            fail = decision.as_dict()
            fail["decision"] = BLOCK
            fail["reason_code"] = DECISION_FINGERPRINT_FAILED
            fail["detail"] = f"{type(exc).__name__}: {exc}"[:200]
            fail["group"] = "REJECTED"
            rejections.append(fail)
            continue

        try:
            pos = _execute(decision, book=book, as_of=day, snapshot_id=snapshot_id)
        except Exception as exc:
            fail = decision.as_dict()
            fail["reason_code"] = "EXECUTION_ERROR"
            fail["detail"] = str(exc)[:200]
            rejections.append(fail)
            continue
        if pos is None:
            reason = ""
            refusals = list(getattr(book, "refusals", []) or [])
            if refusals:
                last = refusals[-1]
                reason = last[1] if isinstance(last, (list, tuple)) and len(last) > 1 else str(last)
            mapped = DUPLICATE_POSITION if "already" in reason.lower() else (
                MAX_PORTFOLIO_RISK if "risk" in reason.lower() else (
                    INSUFFICIENT_CAPITAL if "capital" in reason.lower() or "qty" in reason.lower()
                    else "BOOK_REFUSED"
                )
            )
            fail = decision.as_dict()
            fail["reason_code"] = mapped
            fail["detail"] = reason or "book refused"
            rejections.append(fail)
            continue
        entered += 1
        sector = str(decision.card.get("sector") or "")
        try:
            pos.sector = sector
        except Exception:
            pass
        family_risk[sector] = family_risk.get(sector, 0.0) + DEFAULT_RISK_PCT
        cluster_risk[sector] = cluster_risk.get(sector, 0.0) + DEFAULT_RISK_PCT
        opened.append((ENSEMBLE_ID, decision.symbol))
        taken_row = {
            **decision.as_dict(),
            "thesis_hash": str(thesis.get("thesis_hash") or ""),
            "qty": getattr(pos, "qty", None),
            "entry_fill": getattr(pos, "entry_price", None),
            "status": "TAKEN",
            "group": "TAKEN",
        }
        try:
            from product.execution_reality import shadow_for_paper_fill
            shadow = shadow_for_paper_fill(
                qty=taken_row.get("qty"),
                entry=taken_row.get("entry_fill") or taken_row.get("entry"),
                target=taken_row.get("target"),
                stop=taken_row.get("stop"),
            )
            if shadow:
                # Nested analytics only — qty / entry_fill stay the book fill.
                taken_row["execution_reality_shadow"] = shadow
        except Exception:
            pass
        taken.append(taken_row)
        try:
            from product.paper_learning_loop import note_later_entry
            note_later_entry(decision.symbol, path=policy_path)
        except Exception:
            pass

    reco_symbols = {str(c.get("symbol") or "").upper() for c in card_list}
    scan_rows = list(scan_records or payload.get("scan_records") or [])
    close_misses = []
    for row in scan_rows:
        if not isinstance(row, Mapping):
            continue
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol or symbol in reco_symbols:
            continue
        try:
            score = float(row.get("score") or 0)
        except (TypeError, ValueError):
            score = 0.0
        close_misses.append((score, row, symbol))
    close_misses.sort(key=lambda item: -item[0])
    for _score, row, symbol in close_misses[:20]:
        miss = {
            "symbol": symbol,
            "decision": NOT_SURFACED,
            "reason_code": NOT_SURFACED,
            "group": "NOT_SURFACED",
            "detail": "checked by the scan but not a final recommendation",
            "setup_label": row.get("setup_label") or row.get("classification") or "",
            "sector": row.get("sector") or "",
            "score": row.get("score"),
        }
        not_surfaced.append(miss)
        try:
            from product.counterfactual_learning import freeze_decision
            freeze_decision(
                symbol=symbol,
                reason_code=NOT_SURFACED,
                decision=NOT_SURFACED,
                entry=_f(row.get("price") or row.get("close") or row.get("entry")),
                stop=_f(row.get("stop")),
                target=_f(row.get("target")),
                as_of=day,
                evidence={
                    "group": "NOT_SURFACED",
                    "rules_hash": ident.get("rules_hash"),
                    "regime": regime,
                    "score": row.get("score"),
                    "verdict": row.get("verdict"),
                },
            )
        except Exception:
            pass

    final = ENTER_NOW if taken else (WAIT if waits and not rejections else NO_TRADE)
    if not card_list and not taken:
        final = NO_TRADE
    reason_counts: dict[str, int] = {}
    for item in [*rejections, *waits]:
        code = str(item.get("reason_code") or "UNKNOWN")
        reason_counts[code] = reason_counts.get(code, 0) + 1

    summary = (
        f"taken={len(taken)} rejected={len(rejections)} wait={len(waits)} "
        f"seen={len(card_list)} not_surfaced={len(not_surfaced)}"
    )
    from product.live_safety import live_safety_projection

    safety = live_safety_projection()
    cycle = {
        "as_of": day,
        "session_phase": session_phase,
        "paper_enabled": bool(paper_enabled),
        "entries_allowed": bool(entries_allowed),
        "entry_block_reason": entry_block_reason,
        "candidates_seen": len(card_list),
        "eligible_count": sum(1 for d in decisions if d.decision == ENTER_NOW) + len(taken),
        "taken": taken,
        "rejections": rejections,
        "waits": waits,
        "not_surfaced": not_surfaced,
        "positions_opened": opened,
        "final_decision": final if taken else NO_TRADE,
        "cycle_reasons": cycle_reasons,
        "reason_counts": dict(sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))),
        "summary": summary,
        "eligibility": "TRADED" if taken else (
            "DATA_UNAVAILABLE" if str(entry_block_reason or "") in {
                "NO_DATA_SNAPSHOT", "DATA_UNAVAILABLE", "MARKET_NOT_READY", "NO_TRUSTED_MARKET_DATA",
            } else
            "BLOCKED_SAFETY" if not entries_allowed or not paper_enabled else "NO_ELIGIBLE_TRADE"
        ),
        "source": "recommendation_selection_authority",
        "adapter": "paper",
        "rules_hash": ident.get("rules_hash"),
        "thesis_hash": str(thesis.get("thesis_hash") or ""),
        "thesis": thesis,
        "execution_reality": {
            "shadow_mode": True,
            "affects_paper_orders": False,
            "engine_version": "1",
            "schema_version": 1,
            "note": "Analytics only. Paper fills remain intended-price until promotion.",
        },
        "regime_intelligence_shadow": None,
        "portfolio_authority": "after_selection_authority",
        "cycle_id": f"{day}:{ident.get('rules_hash') or ''}:{clock.isoformat()}",
        **safety,
    }
    try:
        from product.regime_intelligence import shadow_classify
        cycle["regime_intelligence_shadow"] = shadow_classify(
            None, production_regime=str(regime or "RISK_ON"),
        )
    except Exception:
        cycle["regime_intelligence_shadow"] = {"state": "UNKNOWN", "affects_production": False}
    if persist_journal:
        try:
            record_cycle(cycle)
        except Exception:
            pass
        try:
            from product.paper_learning_loop import record_taken_evidence
            record_taken_evidence(taken, as_of=day)
        except Exception:
            pass
        try:
            from product.forward_soak import record_cycle_evidence
            record_cycle_evidence(cycle)
        except Exception:
            pass
    return cycle


def execution_health(
    *,
    autonomy: Mapping[str, Any] | None = None,
    paper: Mapping[str, Any] | None = None,
    workspace: Mapping[str, Any] | None = None,
    journal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Split health lanes so a green autonomy badge cannot imply paper execution."""
    from product.autopilot_journal import why_no_trade

    autonomy = dict(autonomy or {})
    paper = dict(paper or {})
    workspace = dict(workspace or {})
    why = dict(journal or why_no_trade())
    scheduler_running = bool(autonomy.get("running")) and bool(autonomy.get("process_running", True))
    heartbeat = str(autonomy.get("heartbeat_ist") or "")
    reco_ok = bool(workspace.get("categories") is not None) or bool(workspace.get("available"))
    if "schema_version" in workspace:
        reco_ok = True
    latest = why if why.get("available") else {}
    paper_exec = "UNKNOWN"
    detail = "No autopilot cycle recorded"
    if not paper.get("enabled", True) and "enabled" in paper:
        paper_exec = "BROKEN"
        detail = "Paper trading disabled"
    elif latest:
        if latest.get("taken"):
            paper_exec = "HEALTHY"
            detail = why.get("headline") or "Paper positions opened"
        elif PAPER_TRADING_DISABLED in (latest.get("reasons") or []):
            paper_exec = "BROKEN"
            detail = "Paper auto disabled — eligible recos cannot execute"
        elif not latest.get("entries_allowed", True):
            paper_exec = "WAITING"
            detail = why.get("headline") or "Entries not allowed this session"
        else:
            paper_exec = "WAITING"
            detail = why.get("headline") or "No eligible trade"
    return {
        "why_no_trade": why,
        "lanes": {
            "scanner": "HEALTHY" if workspace.get("scan_scanned_at") or workspace.get("from_saved_market_scan") else "UNKNOWN",
            "recommendations": "HEALTHY" if reco_ok else "MISSING",
            "selection_authority": "HEALTHY" if latest else "WAITING",
            "autonomy_scheduler": "HEALTHY" if scheduler_running else "BROKEN" if heartbeat else "WAITING",
            "paper_execution": paper_exec,
            "exit_supervisor": (
                "HEALTHY" if scheduler_running else "WAITING"
            ),
        },
        "paper_execution_detail": detail,
        "scheduler_running": scheduler_running,
        "heartbeat_ist": heartbeat,
    }
