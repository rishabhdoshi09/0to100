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
WATCH_ONLY = "WATCH_ONLY"
DD_GATE_FAILED = "DD_GATE_FAILED"
EMPIRICAL_GATE_FAILED = "EMPIRICAL_GATE_FAILED"
EVIDENCE_POLICY_BLOCK = "EVIDENCE_POLICY_BLOCK"
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
        }


def _identity() -> dict[str, Any]:
    try:
        return ensemble_identity()
    except Exception:
        return {"strategy_id": ENSEMBLE_ID, "strategy_version": 1, "rules_hash": "unverified"}


def reco_is_stale(workspace: Mapping[str, Any] | None, *, now: datetime | None = None) -> bool:
    payload = dict(workspace or {})
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


def _empirical_fail(card: Mapping[str, Any]) -> bool:
    for method in card.get("methods") or []:
        mid = str(method.get("id") or "")
        if mid in {"ev", "case"} and str(method.get("status") or "") == "fail":
            return True
    return bool(card.get("empirical_block"))


def selection_score(card: Mapping[str, Any], policy: Mapping[str, Any] | None = None) -> float:
    """Rank among already-eligible names. Not a BUY oracle."""
    from product.decision_context import score_breakdown
    return float(score_breakdown(card, policy).get("selection_rank") or 0.0)


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
    decision.selection_score = float(decision.breakdown.get("selection_rank") or 0.0)
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

    if _empirical_fail(row):
        return AutopilotDecision(symbol, BLOCK, EMPIRICAL_GATE_FAILED, "empirical method failed", row)

    policy = dict(policy or {})
    if str(policy.get("final_effect") or "") == "BLOCK":
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


def _intent_for(decision: AutopilotDecision, *, as_of: str, snapshot_id: str):
    from research.intelligence.schemas import TradeIntent
    ident = _identity()
    card = decision.card
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
    scan_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Consume saved recommendations and open paper positions for ENTER_NOW names.

    Does not mock or bypass risk. Returns a cycle dict the supervisor can merge.
    """
    from product.autopilot_journal import flatten_cards, record_cycle
    from product.decision_context import snapshot
    from product.evidence_policy_engine import evaluate_policies

    clock = now or datetime.now(timezone.utc)
    day = as_of or clock.date().isoformat()
    ident = _identity()
    payload = dict(workspace or {})
    if cards is None:
        if not payload:
            try:
                from product.recommendations_store import load_recommendations
                payload = load_recommendations() or {}
            except Exception:
                payload = {}
        card_list = flatten_cards(payload)
    else:
        card_list = [dict(c) for c in cards if isinstance(c, Mapping)]

    decisions: list[AutopilotDecision] = []
    taken: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    waits: list[dict[str, Any]] = []
    not_surfaced: list[dict[str, Any]] = []
    opened: list[Any] = []
    family_risk: dict[str, float] = {}
    cluster_risk: dict[str, float] = {}
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
            from product.counterfactual_learning import freeze_decision
            evidence = {
                **dict(decision.context or {}),
                "rules_hash": ident.get("rules_hash"),
                "group": group or decision.group,
                "detail": decision.detail,
                "why": decision.why,
                "setup_label": decision.card.get("setup_label"),
                "sector": decision.card.get("sector"),
                "regime": regime,
            }
            freeze_decision(
                symbol=decision.symbol,
                reason_code=decision.reason_code,
                decision=decision.decision,
                entry=_f(decision.card.get("entry")),
                stop=_f(decision.card.get("stop")),
                target=_f(decision.card.get("target")),
                as_of=day,
                evidence=evidence,
            )
        except Exception:
            pass

    ranked: list[tuple[float, AutopilotDecision]] = []
    for card in card_list:
        ctx = snapshot(card, book=book, regime=regime)
        merged = dict(card)
        for key, value in ctx.items():
            if key == "methods":
                continue
            merged.setdefault(key, value)
        policy = evaluate_policies(merged, path=policy_path, regime=regime, book=book)
        decision = evaluate_candidate(
            merged,
            book=book,
            entries_allowed=entries_allowed,
            entry_block_reason=entry_block_reason,
            paper_enabled=paper_enabled,
            workspace=payload or None,
            now=clock,
            regime=regime,
            policy=policy,
            family_risk=family_risk,
            cluster_risk=cluster_risk,
        )
        _decorate(decision, policy=policy, context=ctx)
        decisions.append(decision)
        if decision.decision == ENTER_NOW:
            ranked.append((float(decision.selection_score or 0.0), decision))
        elif decision.decision == WAIT:
            waits.append(decision.as_dict())
            _freeze(decision, group="RECOMMENDED_BUT_NOT_FILLED")
        else:
            rejections.append(decision.as_dict())
            _freeze(decision, group="REJECTED")

    ranked.sort(key=lambda item: (-item[0], item[1].symbol))
    try:
        from product.portfolio_selection_authority import apply_portfolio_authority
        ranked, diverted = apply_portfolio_authority(
            ranked, book=book, max_new=max_new, regime=regime,
        )
        for decision in diverted:
            row = decision.as_dict() if hasattr(decision, "as_dict") else dict(decision)
            if str(getattr(decision, "decision", row.get("decision"))) == WAIT:
                waits.append(row)
                _freeze(decision, group="RECOMMENDED_BUT_NOT_FILLED")
            else:
                rejections.append(row)
                _freeze(decision, group="REJECTED")
    except Exception:
        diverted = []
    snapshot_id = str(payload.get("scan_scanned_at") or day)
    entered = 0
    for _score, decision in ranked:
        if entered >= int(max_new):
            leftover = dict(decision.as_dict())
            leftover["reason_code"] = NO_TRADE
            leftover["detail"] = "not top-of-the-top this cycle"
            leftover["decision"] = NO_TRADE
            leftover["group"] = "REJECTED"
            rejections.append(leftover)
            decision.decision = NO_TRADE
            decision.reason_code = NO_TRADE
            _freeze(decision, group="REJECTED")
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
        family_risk[sector] = family_risk.get(sector, 0.0) + DEFAULT_RISK_PCT
        cluster_risk[sector] = cluster_risk.get(sector, 0.0) + DEFAULT_RISK_PCT
        opened.append((ENSEMBLE_ID, decision.symbol))
        taken_row = {
            **decision.as_dict(),
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
    summary = (
        f"taken={len(taken)} rejected={len(rejections)} wait={len(waits)} "
        f"seen={len(card_list)} not_surfaced={len(not_surfaced)}"
    )
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
        "summary": summary,
        "eligibility": "TRADED" if taken else (
            "BLOCKED_SAFETY" if not entries_allowed or not paper_enabled else "NO_ELIGIBLE_TRADE"
        ),
        "source": "recommendation_selection_authority",
        "live_locked": True,
        "adapter": "paper",
        "rules_hash": ident.get("rules_hash"),
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
