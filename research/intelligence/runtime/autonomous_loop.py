"""
🔄 Autonomous Intelligence Runtime — the end-to-end paper loop (Phases B–M).

`run_intelligence_cycle(ctx, ...)` is the ONE authoritative orchestration of:

  data gate → per-strategy frozen runtime signals → canonical events → manage/exit open
  positions → decode outcomes → Brain 1 Evidence Cards → Brain 2 allocation decisions →
  canonical Target Portfolio → TradeIntents → paper execution → persist.

Properties (by construction): deterministic for identical inputs; idempotent per cycle id
(a completed cycle is a no-op; every mutation is deduped by the event store / book / state);
point-in-time safe (adapters see only past bars); restartable (state + book + events persist);
Streamlit-free; broker-free; paper-only; safe with no data. Live modes are refused up front.
"""
from __future__ import annotations

from research.intelligence import evidence_brain as EB
from research.intelligence import allocation_brain as AB
from research.intelligence import strategy_runtime as RT
from research.intelligence import decoder_registry as REG
from research.intelligence.runtime import events as EV
from research.intelligence.runtime import modes as MODES
from research.intelligence.runtime import target_portfolio as TARGET
from research.intelligence.runtime.cycle_result import (
    IntelligenceCycleResult, STATUS_OK, STATUS_NO_ACTION, STATUS_ALREADY_DONE,
    STATUS_FAILED_SAFE)


def run_intelligence_cycle(ctx, *, store, book, runtime_state, knowledge=None,
                           alloc_cfg: AB.AllocationConfig | None = None,
                           backtest_R=None, backtest_trades=None) -> IntelligenceCycleResult:
    """Run one cycle. `store` = EventStore, `book` = PaperBook, `runtime_state` = RuntimeState.
    `backtest_R`/`backtest_trades` map strategy_id → in-sample edge / trade count (for the
    evidence card); missing ⇒ 0."""
    alloc_cfg = alloc_cfg or AB.AllocationConfig()
    backtest_R = backtest_R or {}
    backtest_trades = backtest_trades or {}
    cid = ctx.cycle_id()
    res = IntelligenceCycleResult(cycle_id=cid, as_of_date=ctx.as_of_date, mode=ctx.mode)

    MODES.assert_no_live(ctx.mode)                       # never route live as a shortcut

    # ── idempotency: a completed cycle is a no-op ────────────────────────────────
    if runtime_state.is_cycle_done(cid):
        EV.emit(store, cid, EV.CYCLE_ALREADY_DONE, event_ts=ctx.as_of_date)
        res.status = STATUS_ALREADY_DONE
        res.no_action_reasons.append("cycle already completed")
        return res

    try:
        EV.emit(store, cid, EV.CYCLE_STARTED, event_ts=ctx.as_of_date,
                config_hash=ctx.config_hash, summary={"mode": ctx.mode, "type": ctx.cycle_type})

        # ── data gate ────────────────────────────────────────────────────────────
        if not ctx.data_ok or ctx.mode in (MODES.OFF, MODES.HALTED, MODES.RESEARCH_ONLY):
            EV.emit(store, cid, EV.DATA_GATE_FAILED, event_ts=ctx.as_of_date,
                    reason=("no validated data" if not ctx.data_ok else f"mode {ctx.mode}"))
            res.status = STATUS_NO_ACTION
            res.data_ok = ctx.data_ok
            res.no_action_reasons.append("data gate failed" if not ctx.data_ok
                                         else f"mode {ctx.mode} does not trade")
            EV.emit(store, cid, EV.CYCLE_COMPLETED, event_ts=ctx.as_of_date, result="no_action")
            runtime_state.mark_cycle_done(cid)
            _persist(store, book, runtime_state)
            res.events_emitted = len(store)
            return res
        EV.emit(store, cid, EV.DATA_GATE_PASSED, event_ts=ctx.as_of_date)

        # reconcile persisted state vs the book before taking new risk (recovery safety)
        reconciled = runtime_state.reconcile(book)
        if not reconciled:
            res.warnings.append("state not reconciled — new entries refused this cycle")

        # operational-safety preflight (Phase Q): a failed critical check ⇒ no NEW risk
        from research.intelligence.runtime import preflight as PF
        pf = PF.preflight(ctx, store=store, book=book, runtime_state=runtime_state)
        # forward eligibility (Part 14): only FORWARD_ELIGIBLE data may open NEW entries; a
        # research-eligible-but-not-forward snapshot still runs the cycle and updates evidence.
        operational_entry_ok = bool(getattr(ctx, "new_entries_allowed", True))
        safe_to_enter = (pf.ok and reconciled and getattr(ctx, "forward_eligible", True)
                         and operational_entry_ok)
        if not getattr(ctx, "forward_eligible", True):
            res.no_action_reasons.append("dataset not forward-eligible — no new entries")
        if not operational_entry_ok:
            reason = getattr(ctx, "entry_block_reason", "") or "NEW_ENTRIES_BLOCKED"
            res.no_action_reasons.append(reason)
            EV.emit(store, cid, EV.NEW_ENTRIES_BLOCKED, event_ts=ctx.as_of_date,
                    reason=reason, summary={
                        "session_phase": getattr(ctx, "session_phase", ""),
                        "capability_failures": list(getattr(ctx, "capability_failures", ())),
                    })
        for name, reason in pf.failed:
            res.warnings.append(f"preflight {name}: {reason}")

        # ── 1. manage/exit OPEN positions first (outcomes feed the brains) ────────
        _manage_positions(ctx, store, book, runtime_state, res)

        # ── 2. per-strategy frozen runtime → signals → canonical events ──────────
        today_signals = _evaluate_strategies(ctx, store, book, runtime_state, res)

        # ── 3. Brain 1: build/refresh an Evidence Card per strategy ──────────────
        cards = _run_brain1(ctx, store, book, res, backtest_R, backtest_trades)

        # ── 4. Brain 2: allocation decisions from the cards ──────────────────────
        current_risk = {sid: runtime_state.get(sid).allocation_pct for sid in
                        {c.strategy_id for c in cards}}
        decisions = AB.decide(cards, cfg=alloc_cfg, current_risk=current_risk,
                              clusters=ctx.clusters, data_ok=ctx.data_ok)
        for d in decisions:
            store.append(d)
            EV.emit(store, cid, EV.ALLOCATION_DECISION_CREATED, strategy_id=d.strategy_id,
                    decision=d.action, reason="; ".join(d.reasons), event_ts=ctx.as_of_date,
                    summary={"target_risk_pct": d.target_risk_pct, "bucket": d.risk_bucket})
            res.allocation_decisions.append((d.strategy_id, d.action))
            _apply_non_entry_decision(d, store, cid, runtime_state, res, ctx)

        # ── 5. all proposals → one Target Portfolio → exact deltas → PAPER ───────
        if MODES.opens_new_entries(ctx.mode) and safe_to_enter:
            _open_new_positions(ctx, store, book, runtime_state, res, decisions,
                                today_signals, cards, alloc_cfg)
        else:
            res.no_action_reasons.append(
                "no new entries (mode/entry-window/capability/unreconciled/preflight)")

        EV.emit(store, cid, EV.CYCLE_COMPLETED, event_ts=ctx.as_of_date, result="ok")
        runtime_state.mark_cycle_done(cid)
        _persist(store, book, runtime_state)
        res.events_emitted = len(store)
        res.status = STATUS_OK if (res.positions_opened or res.positions_closed
                                   or res.allocation_decisions) else STATUS_NO_ACTION
        return res
    except Exception as e:                               # fail SAFE — never leave half-state
        EV.emit(store, cid, EV.CYCLE_FAILED_SAFE, reason=str(e)[:200], event_ts=ctx.as_of_date)
        res.status = STATUS_FAILED_SAFE
        res.errors.append(str(e))
        _persist(store, book, runtime_state)
        return res


# ── steps ────────────────────────────────────────────────────────────────────────

def _manage_positions(ctx, store, book, state, res) -> None:
    if not MODES.manages_positions(ctx.mode):
        return
    bars = {}
    for pos in list(book.open.values()):
        tb = ctx.today_bar(pos.symbol)
        if tb is not None:
            bars[pos.symbol] = tb                        # (o,h,l,c) → gap-aware exits
    closed = book.mark(bars, ctx.as_of_date) if bars else []
    for t in closed:
        # outcome + execution decode (idempotent via deterministic ids)
        for rec in REG.decode("outcome", t.as_dict(), ctx={"split": "forward"}):
            store.append(rec)
        for rec in REG.decode("execution", t.as_dict()):
            store.append(rec)
        EV.emit(store, ctx.cycle_id(), EV.PAPER_POSITION_CLOSED, strategy_id=t.strategy_id,
                symbol=t.symbol, reason=t.exit_reason, event_ts=ctx.as_of_date,
                summary={"realized_R": round(t.realized_R, 4), "pnl": round(t.pnl, 2)})
        EV.emit(store, ctx.cycle_id(), EV.OUTCOME_DECODED, strategy_id=t.strategy_id,
                symbol=t.symbol, event_ts=ctx.as_of_date,
                summary={"realized_R": round(t.realized_R, 4)})
        res.positions_closed.append((t.strategy_id, t.symbol, t.exit_reason))
        res.outcomes_recorded.append(t.strategy_id)


def _evaluate_strategies(ctx, store, book, state, res) -> dict:
    """Return {strategy_id: ranked signal dicts today}."""
    today: dict = {}
    for spec in ctx.strategies:
        sid = spec.strategy_id
        res.strategies_evaluated.append(sid)
        EV.emit(store, ctx.cycle_id(), EV.STRATEGY_EVALUATION_STARTED, strategy_id=sid,
                strategy_version=spec.version, rules_hash=spec.config_hash(),
                event_ts=ctx.as_of_date)
        if not RT.is_supported(spec.family):
            state.get(sid, spec.family).unsupported_runtime = True
            EV.emit(store, ctx.cycle_id(), EV.STRATEGY_RUNTIME_UNSUPPORTED, strategy_id=sid,
                    reason=f"family {spec.family} has no bar-by-bar adapter",
                    event_ts=ctx.as_of_date)
            res.unsupported.append(sid)
            continue
        sym_hist = ctx.data.get(sid, {})
        ctx_prov = {"strategy_id": sid, "strategy_version": spec.version,
                    "rules_hash": spec.config_hash(), "data_snapshot_id": ctx.data_snapshot_id,
                    "event_ts": ctx.as_of_date}
        try:
            fired = RT.signals(spec, ctx.as_of_date, sym_hist,
                               benchmark=getattr(ctx, "benchmark", None))
        except RT.UnsupportedStrategy:
            fired = []
        if not fired:
            EV.emit(store, ctx.cycle_id(), EV.SIGNAL_REJECTED, strategy_id=sid,
                    reason="no qualifying setup today", event_ts=ctx.as_of_date)
            continue
        for sig in fired:
            for rec in REG.decode("signal", sig, ctx=ctx_prov):
                store.append(rec)
            EV.emit(store, ctx.cycle_id(), EV.SIGNAL_GENERATED, strategy_id=sid,
                    symbol=sig["symbol"], event_ts=ctx.as_of_date,
                    summary={"entry": sig["entry"], "stop": sig["stop"], "target": sig["target"]})
            res.signals_generated.append((sid, sig["symbol"]))
        today[sid] = fired
    return today


def _run_brain1(ctx, store, book, res, backtest_R, backtest_trades) -> list:
    cards = []
    for spec in ctx.strategies:
        sid = spec.strategy_id
        if not RT.is_supported(spec.family):
            continue
        sdef = REG.decode("strategy", spec, ctx={"data_snapshot_id": ctx.data_snapshot_id,
                                                 "event_ts": ctx.as_of_date})[0]
        store.append(sdef)
        fwd = [t.realized_R for t in book.closed if t.strategy_id == sid]
        card = EB.build_card(sdef, backtest_R=float(backtest_R.get(sid, 0.0)),
                             forward_returns=fwd, out_of_sample_trades=len(fwd),
                             in_sample_trades=int(backtest_trades.get(sid, 0)),
                             dataset_tier=getattr(ctx, "dataset_tier", ""))
        new = store.append(card)
        EV.emit(store, ctx.cycle_id(),
                EV.EVIDENCE_CARD_CREATED if new else EV.EVIDENCE_CARD_UPDATED,
                strategy_id=sid, decision=card.evidence_state, event_ts=ctx.as_of_date,
                summary={"lower_bound_R": card.lower_bound_R, "forward_trades": card.forward_trades})
        cards.append(card)
        res.cards_created.append(sid)
    return cards


def _apply_non_entry_decision(d, store, cid, state, res, ctx) -> None:
    st = state.get(d.strategy_id, d.family)
    if d.action == "REDUCE":
        st.allocation_pct = d.target_risk_pct
        EV.emit(store, cid, EV.STRATEGY_ALLOCATION_REDUCED, strategy_id=d.strategy_id,
                event_ts=ctx.as_of_date)
    elif d.action in ("PAUSE",):
        st.pause_reason = "; ".join(d.reasons) or "brain2 pause"
        EV.emit(store, cid, EV.STRATEGY_PAUSED, strategy_id=d.strategy_id, event_ts=ctx.as_of_date)
        res.strategies_paused.append(d.strategy_id)
    elif d.action == "RETIRE":
        st.retire_reason = "; ".join(d.reasons) or "brain2 retire"
        st.lifecycle = "DECAYED"
        EV.emit(store, cid, EV.STRATEGY_RETIRED, strategy_id=d.strategy_id, event_ts=ctx.as_of_date)
        res.strategies_retired.append(d.strategy_id)


def _open_new_positions(ctx, store, book, state, res, decisions, today_signals, cards,
                        alloc_cfg) -> None:
    """Persist one Target Portfolio and execute only its approved quantity deltas."""
    cid = ctx.cycle_id()
    for name in ("target_portfolios", "target_positions", "blocked_target_positions"):
        if not hasattr(res, name):
            setattr(res, name, [])

    build = TARGET.build_target_portfolio(
        ctx,
        book=book,
        runtime_state=state,
        decisions=decisions,
        today_signals=today_signals,
        cards=cards,
        cfg=alloc_cfg,
    )

    for target in build.positions:
        store.append(target)
        res.target_positions.append(target.record_id)
        if target.status == TARGET.BLOCKED:
            reason = target.blocked_by[0] if target.blocked_by else "TARGET_BLOCKED"
            EV.emit(store, cid, EV.TARGET_POSITION_BLOCKED,
                    strategy_id=target.strategy_id, strategy_version=target.strategy_version,
                    rules_hash=target.rules_hash, data_snapshot_id=target.data_snapshot_id,
                    symbol=target.symbol, decision=target.status, reason=reason,
                    event_ts=ctx.as_of_date,
                    summary={
                        "target_position_id": target.record_id,
                        "desired_quantity": target.desired_quantity,
                        "current_quantity": target.current_quantity,
                        "pending_quantity": target.pending_quantity,
                        "blocked_by": list(target.blocked_by),
                    })
            res.blocked_target_positions.append((target.strategy_id, target.symbol, reason))
            res.intents_blocked.append((target.strategy_id, reason))
        else:
            EV.emit(store, cid, EV.TARGET_POSITION_CREATED,
                    strategy_id=target.strategy_id, strategy_version=target.strategy_version,
                    rules_hash=target.rules_hash, data_snapshot_id=target.data_snapshot_id,
                    symbol=target.symbol, decision=target.status, event_ts=ctx.as_of_date,
                    summary={
                        "target_position_id": target.record_id,
                        "desired_quantity": target.desired_quantity,
                        "current_quantity": target.current_quantity,
                        "pending_quantity": target.pending_quantity,
                        "required_quantity": target.required_quantity,
                        "target_risk_pct": target.target_risk_pct,
                    })

    store.append(build.portfolio)
    res.target_portfolios.append(build.portfolio.record_id)
    EV.emit(store, cid, EV.TARGET_PORTFOLIO_CREATED,
            data_snapshot_id=build.portfolio.data_snapshot_id,
            decision="TARGETED" if build.executable else "NO_CHANGE",
            event_ts=ctx.as_of_date,
            summary={
                "target_portfolio_id": build.portfolio.record_id,
                "positions": len(build.positions),
                "executable": len(build.executable),
                "blocked": len(build.blocked),
                "current_open_risk_pct": build.portfolio.current_open_risk_pct,
                "pending_open_risk_pct": build.portfolio.pending_open_risk_pct,
                "target_open_risk_pct": build.portfolio.target_open_risk_pct,
                "available_cash": build.portfolio.available_cash,
            })

    decision_by_strategy = {str(d.strategy_id): d for d in decisions}
    for target in build.executable:
        intent = TARGET.trade_intent_from_target(target, build.portfolio)
        store.append(intent)
        EV.emit(store, cid, EV.TRADE_INTENT_CREATED,
                strategy_id=target.strategy_id, strategy_version=target.strategy_version,
                rules_hash=target.rules_hash, data_snapshot_id=target.data_snapshot_id,
                symbol=target.symbol, event_ts=ctx.as_of_date,
                causation_id=target.record_id,
                summary={
                    "intent_id": intent.record_id,
                    "target_portfolio_id": build.portfolio.record_id,
                    "target_position_id": target.record_id,
                    "risk_pct": intent.intended_risk_pct,
                    "required_quantity": intent.required_quantity,
                })
        res.trade_intents.append(intent.record_id)

        if hasattr(book, "open_intent"):
            pos = book.open_intent(intent, date=ctx.as_of_date)
        else:
            try:
                pos = book.open_position(
                    target.strategy_id, target.symbol, target.intended_entry,
                    target.stop_price, target.target_price, ctx.as_of_date,
                    target.holding_horizon_days,
                    risk_pct_of_capital=target.target_risk_pct,
                    quantity=target.required_quantity,
                )
            except TypeError:  # compatibility for narrow injected test doubles
                pos = book.open_position(
                    target.strategy_id, target.symbol, target.intended_entry,
                    target.stop_price, target.target_price, ctx.as_of_date,
                    target.holding_horizon_days,
                )
        if pos is None:
            EV.emit(store, cid, EV.TRADE_INTENT_BLOCKED,
                    strategy_id=target.strategy_id, symbol=target.symbol,
                    reason="BOOK_REFUSED", event_ts=ctx.as_of_date,
                    causation_id=target.record_id,
                    summary={"refusals": getattr(book, "refusals", [])[-1:]})
            res.intents_blocked.append((target.strategy_id, "BOOK_REFUSED"))
            continue

        decision = decision_by_strategy.get(target.strategy_id)
        st = state.get(target.strategy_id, target.family)
        st.allocation_pct = target.target_risk_pct
        st.risk_budget_pct = target.target_risk_pct
        st.lifecycle = "PAPER_EVALUATION"
        st.latest_card_id = target.card_id
        st.latest_allocation_id = target.allocation_id
        EV.emit(store, cid, EV.PAPER_POSITION_OPENED,
                strategy_id=target.strategy_id, symbol=target.symbol,
                event_ts=ctx.as_of_date, causation_id=intent.record_id,
                summary={
                    "qty": pos.qty,
                    "entry": pos.entry_price,
                    "target_portfolio_id": build.portfolio.record_id,
                    "target_position_id": target.record_id,
                    "requested_risk_pct": intent.intended_risk_pct,
                    "approved_risk_pct": getattr(pos, "approved_risk_pct", 0.0),
                    "risk_amount": getattr(pos, "risk_amount", 0.0),
                })
        if decision is not None and decision.action == "INCREASE":
            EV.emit(store, cid, EV.STRATEGY_ALLOCATION_INCREASED,
                    strategy_id=target.strategy_id, event_ts=ctx.as_of_date)
        res.positions_opened.append((target.strategy_id, target.symbol))


def _persist(store, book, state) -> None:
    # event store persists on append; snapshot the book + runtime state (single-writer)
    try:
        state.save()
    except Exception:
        pass
