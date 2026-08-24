"""Canonical Target Portfolio construction for the broker-free intelligence runtime.

Strategies propose risk. This module converts ranked proposals into one immutable,
symbol-level desired portfolio after accounting for current positions, pending exposure,
cash, portfolio risk, family caps, cluster caps and position-count limits.

It never calls a broker and never mutates the paper book or runtime state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from product.paper_learning import select_paper_signal
from research.intelligence import schemas as SC
from research.intelligence.runtime.position_sizing import size_long_cash

TARGETED = "TARGETED"
HOLD = "HOLD"
BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class TargetPortfolioBuild:
    portfolio: SC.TargetPortfolio
    positions: tuple[SC.TargetPosition, ...]
    executable: tuple[SC.TargetPosition, ...]
    blocked: tuple[SC.TargetPosition, ...]


def build_target_portfolio(
    ctx,
    *,
    book,
    runtime_state,
    decisions,
    today_signals: Mapping[str, list],
    cards,
    cfg,
    pending_quantities: Mapping[str, int] | None = None,
    pending_risk_amounts: Mapping[str, float] | None = None,
    pending_capital_amounts: Mapping[str, float] | None = None,
) -> TargetPortfolioBuild:
    """Build one portfolio target from all deployable strategy proposals.

    Pending exposure is explicit. PAPER supplies empty maps; a future OMS must supply
    its open-order quantities and worst-case risk before this constructor can be
    certified for live use.
    """
    capital = float(getattr(book, "capital", 0.0) or 0.0)
    realized_pnl = float(getattr(book, "realized_pnl", 0.0) or 0.0)
    max_total_risk_fraction = float(getattr(book, "max_total_risk_pct", 0.0) or 0.0)
    max_position_fraction = float(getattr(book, "max_position_pct", 0.0) or 0.0)
    max_risk_fraction = float(getattr(book, "risk_per_trade_pct", 0.0) or 0.0)
    max_positions = int(getattr(book, "max_positions", 0) or 0)
    slippage_bps = float(getattr(book, "slippage_bps", 0.0) or 0.0)

    pending_quantities = _int_map(
        pending_quantities if pending_quantities is not None
        else getattr(ctx, "pending_quantities", {})
    )
    pending_risk_amounts = _float_map(
        pending_risk_amounts if pending_risk_amounts is not None
        else getattr(ctx, "pending_risk_amounts", {})
    )
    pending_capital_amounts = _float_map(
        pending_capital_amounts if pending_capital_amounts is not None
        else getattr(ctx, "pending_capital_amounts", {})
    )

    current_quantity: dict[str, int] = {}
    current_risk: dict[str, float] = {}
    current_owners: dict[str, set[str]] = {}
    invested_capital = 0.0
    current_symbols: set[str] = set()
    for position in getattr(book, "open", {}).values():
        symbol = str(position.symbol).upper()
        strategy_id = str(position.strategy_id)
        qty = max(0, int(position.qty))
        current_quantity[symbol] = current_quantity.get(symbol, 0) + qty
        current_risk[symbol] = current_risk.get(symbol, 0.0) + float(position.risk_amount)
        current_owners.setdefault(symbol, set()).add(strategy_id)
        invested_capital += qty * float(position.entry_price)
        if qty:
            current_symbols.add(symbol)

    pending_symbols = {symbol for symbol, qty in pending_quantities.items() if qty > 0}
    available_cash = max(
        0.0,
        capital + realized_pnl - invested_capital - sum(pending_capital_amounts.values()),
    )
    current_open_risk = sum(current_risk.values())
    pending_open_risk = sum(pending_risk_amounts.values())
    max_total_risk_amount = capital * max_total_risk_fraction

    spec_by_id = {
        str(spec.strategy_id): spec
        for spec in getattr(ctx, "strategies", ())
        if getattr(spec, "strategy_id", None)
    }
    card_by_id = {str(card.strategy_id): card for card in cards}

    family_risk, cluster_risk = _existing_group_risk(
        ctx=ctx,
        book=book,
        runtime_state=runtime_state,
        spec_by_id=spec_by_id,
        capital=capital,
    )
    family_risk = _merge_add(
        family_risk,
        _float_map(getattr(ctx, "pending_family_risk_pct", {})),
    )
    cluster_risk = _merge_add(
        cluster_risk,
        _float_map(getattr(ctx, "pending_cluster_risk_pct", {})),
    )

    candidates = [
        decision for decision in decisions
        if str(getattr(decision, "action", "")) in {"DEPLOY", "INCREASE"}
        and float(getattr(decision, "target_risk_pct", 0.0) or 0.0) > 0
    ]
    candidates.sort(
        key=lambda decision: float(getattr(decision, "score", 0.0) or 0.0),
        reverse=True,
    )

    positions: list[SC.TargetPosition] = []
    executable: list[SC.TargetPosition] = []
    blocked_positions: list[SC.TargetPosition] = []
    seen_proposal_symbols: set[str] = set()
    target_new_symbols: set[str] = set()
    planned_incremental_risk = 0.0

    for decision in candidates:
        strategy_id = str(decision.strategy_id)
        signals = list(today_signals.get(strategy_id) or [])
        if not signals:
            continue
        signal = dict(signals[0])
        skipped_lessons: tuple[str, ...] = ()
        blockers_pre: list[str] = []
        try:
            as_of = str(getattr(ctx, "as_of_date", "") or "")
            memory = getattr(ctx, "paper_memory", None)
            picked, skipped_lessons = select_paper_signal(signals, memory, as_of=as_of)
            if picked is None:
                blockers_pre = ["PAPER_LESSON_COOLDOWN"]
            else:
                signal = dict(picked)
        except Exception:
            skipped_lessons = ()
            blockers_pre = []
        symbol = str(signal.get("symbol", "")).upper()
        if not symbol:
            continue

        spec = spec_by_id.get(strategy_id)
        card = card_by_id.get(strategy_id)
        family = str(getattr(decision, "family", "") or getattr(spec, "family", ""))
        cluster = str(getattr(ctx, "clusters", {}).get(strategy_id, "") or "")
        reasons = list(getattr(decision, "reasons", ()) or ())
        reasons.extend(skipped_lessons)
        blockers: list[str] = []
        blockers.extend(blockers_pre)

        entry = float(signal.get("entry", 0.0) or 0.0)
        stop = float(signal.get("stop", 0.0) or 0.0)
        target = float(signal.get("target", 0.0) or 0.0)
        target_risk_pct = float(getattr(decision, "target_risk_pct", 0.0) or 0.0)
        current_qty = current_quantity.get(symbol, 0)
        pending_qty = pending_quantities.get(symbol, 0)

        # A different strategy may not interpret an existing symbol holding as its own
        # fulfilled target. Preserve an explicit duplicate-economic-exposure refusal.
        owners = current_owners.get(symbol, set())
        if current_qty > 0 and strategy_id not in owners:
            blockers.append("DUPLICATE_SYMBOL")

        requires_live = bool(getattr(ctx, "live_confirmation_required", False)) or (
            spec is not None and "live_ticks" in tuple(getattr(spec, "required_data", ()))
        )
        if requires_live and symbol not in set(getattr(ctx, "fresh_live_symbols", ())):
            blockers.append("LIVE_PRICE_STALE")
        if symbol in seen_proposal_symbols:
            blockers.append("DUPLICATE_SYMBOL_PROPOSAL")
        seen_proposal_symbols.add(symbol)
        if not bool(getattr(ctx, "data_ok", True)):
            blockers.append("NO_VALIDATED_DATA")
        if not bool(getattr(runtime_state, "reconciled", False)):
            blockers.append("UNRECONCILED_STATE")
        if str(getattr(ctx, "market_regime", "")) == "RISK_OFF":
            blockers.append("REGIME_STANDDOWN")

        sizing = size_long_cash(
            capital=capital,
            entry=entry,
            stop=stop,
            requested_risk_pct=target_risk_pct,
            max_risk_fraction=max_risk_fraction,
            max_position_fraction=max_position_fraction,
            slippage_bps=slippage_bps,
        )
        if not sizing.ok:
            blockers.append(sizing.reason_code)

        desired_qty = sizing.quantity if sizing.ok else 0
        required_qty = max(0, desired_qty - current_qty - pending_qty)
        incremental_risk = required_qty * sizing.risk_per_share if sizing.ok else 0.0
        capital_required = required_qty * sizing.effective_entry if sizing.ok else 0.0
        target_risk_amount = desired_qty * sizing.risk_per_share if sizing.ok else 0.0

        if required_qty <= 0 and not blockers:
            status = HOLD
            reasons.append("target already satisfied by current and pending quantity")
        else:
            if current_qty > 0 and strategy_id in owners and required_qty > 0:
                blockers.append("POSITION_INCREASE_NOT_SUPPORTED_BY_PAPER_ADAPTER")
            if current_qty == 0 and pending_qty == 0:
                projected_count = len(current_symbols | pending_symbols | target_new_symbols) + 1
                if max_positions > 0 and projected_count > max_positions:
                    blockers.append("MAX_POSITIONS")
            projected_total_risk = (
                current_open_risk
                + pending_open_risk
                + planned_incremental_risk
                + incremental_risk
            )
            if projected_total_risk > max_total_risk_amount + 1e-6:
                blockers.append("TOTAL_OPEN_RISK_CAP")
            incremental_risk_pct = _pct(incremental_risk, capital)
            if family_risk.get(family, 0.0) + incremental_risk_pct > float(cfg.max_family_risk_pct) + 1e-9:
                blockers.append("FAMILY_CAP")
            if cluster and cluster_risk.get(cluster, 0.0) + incremental_risk_pct > float(cfg.max_cluster_risk_pct) + 1e-9:
                blockers.append("CLUSTER_CAP")
            if capital_required > available_cash + 1e-6:
                blockers.append("INSUFFICIENT_CASH")
            status = BLOCKED if blockers else TARGETED

        position = SC.TargetPosition(
            strategy_id=strategy_id,
            strategy_version=int(getattr(decision, "strategy_version", 0) or 0),
            rules_hash=str(getattr(decision, "rules_hash", "") or ""),
            data_snapshot_id=str(getattr(ctx, "data_snapshot_id", "") or ""),
            source="target_portfolio",
            event_ts=str(getattr(ctx, "as_of_date", "") or ""),
            cycle_id=str(ctx.cycle_id()),
            symbol=symbol,
            direction="LONG",
            family=family,
            correlation_cluster=cluster,
            current_quantity=current_qty,
            pending_quantity=pending_qty,
            desired_quantity=desired_qty,
            required_quantity=required_qty,
            intended_entry=entry,
            stop_price=stop,
            target_price=target,
            target_risk_pct=target_risk_pct,
            target_risk_amount=target_risk_amount,
            incremental_risk_amount=incremental_risk,
            capital_required=capital_required,
            holding_horizon_days=int(signal.get("max_hold", 0) or 0),
            card_id=str(getattr(card, "record_id", "") or ""),
            allocation_id=str(getattr(decision, "record_id", "") or ""),
            status=status,
            reasons=tuple(reasons),
            blocked_by=tuple(dict.fromkeys(blockers)),
        )
        positions.append(position)

        if status == TARGETED:
            executable.append(position)
            available_cash -= capital_required
            planned_incremental_risk += incremental_risk
            incremental_risk_pct = _pct(incremental_risk, capital)
            family_risk[family] = family_risk.get(family, 0.0) + incremental_risk_pct
            if cluster:
                cluster_risk[cluster] = cluster_risk.get(cluster, 0.0) + incremental_risk_pct
            if current_qty == 0 and pending_qty == 0:
                target_new_symbols.add(symbol)
        elif status == BLOCKED:
            blocked_positions.append(position)

    target_symbols = current_symbols | pending_symbols | {position.symbol for position in executable}
    target_open_risk = current_open_risk + pending_open_risk + planned_incremental_risk
    portfolio = SC.TargetPortfolio(
        data_snapshot_id=str(getattr(ctx, "data_snapshot_id", "") or ""),
        source="target_portfolio",
        event_ts=str(getattr(ctx, "as_of_date", "") or ""),
        cycle_id=str(ctx.cycle_id()),
        mode=str(getattr(ctx, "mode", "") or ""),
        capital=capital,
        available_cash=max(0.0, available_cash),
        current_open_risk_pct=_pct(current_open_risk, capital),
        pending_open_risk_pct=_pct(pending_open_risk, capital),
        target_open_risk_pct=_pct(target_open_risk, capital),
        max_total_risk_pct=max_total_risk_fraction * 100.0,
        current_position_count=len(current_symbols),
        target_position_count=len(target_symbols),
        position_ids=tuple(position.record_id for position in positions),
        executable_position_ids=tuple(position.record_id for position in executable),
        blocked_position_ids=tuple(position.record_id for position in blocked_positions),
        reasons=("no executable portfolio changes",) if not executable else (),
    )
    return TargetPortfolioBuild(
        portfolio=portfolio,
        positions=tuple(positions),
        executable=tuple(executable),
        blocked=tuple(blocked_positions),
    )


def trade_intent_from_target(
    position: SC.TargetPosition,
    portfolio: SC.TargetPortfolio,
) -> SC.TradeIntent:
    """Create the only execution instruction permitted from a target delta."""
    if position.status != TARGETED or position.required_quantity <= 0:
        raise ValueError("target position is not executable")
    return SC.TradeIntent(
        strategy_id=position.strategy_id,
        strategy_version=position.strategy_version,
        rules_hash=position.rules_hash,
        data_snapshot_id=position.data_snapshot_id,
        source="target_portfolio",
        event_ts=position.event_ts,
        cycle_id=position.cycle_id,
        symbol=position.symbol,
        direction=position.direction,
        entry_rule="target_portfolio_delta",
        intended_entry=position.intended_entry,
        intended_risk_pct=position.target_risk_pct,
        max_capital=position.capital_required,
        stop_rule="absolute_price",
        stop_price=position.stop_price,
        exit_rule="absolute_target",
        target_price=position.target_price,
        holding_horizon_days=position.holding_horizon_days,
        card_id=position.card_id,
        allocation_id=position.allocation_id,
        target_portfolio_id=portfolio.record_id,
        target_position_id=position.record_id,
        current_quantity=position.current_quantity,
        pending_quantity=position.pending_quantity,
        desired_quantity=position.desired_quantity,
        required_quantity=position.required_quantity,
        reasons=position.reasons,
        invalidation=position.blocked_by,
    )


def _existing_group_risk(*, ctx, book, runtime_state, spec_by_id, capital: float):
    family_risk: dict[str, float] = {}
    cluster_risk: dict[str, float] = {}
    for position in getattr(book, "open", {}).values():
        strategy_id = str(position.strategy_id)
        spec = spec_by_id.get(strategy_id)
        state = getattr(runtime_state, "strategies", {}).get(strategy_id)
        family = str(getattr(spec, "family", "") or getattr(state, "family", "") or "")
        cluster = str(getattr(ctx, "clusters", {}).get(strategy_id, "") or "")
        risk_pct = _pct(float(position.risk_amount), capital)
        family_risk[family] = family_risk.get(family, 0.0) + risk_pct
        if cluster:
            cluster_risk[cluster] = cluster_risk.get(cluster, 0.0) + risk_pct
    return family_risk, cluster_risk


def _int_map(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, int] = {}
    for key, item in value.items():
        try:
            parsed = max(0, int(item))
        except (TypeError, ValueError):
            continue
        out[str(key).upper()] = parsed
    return out


def _float_map(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, item in value.items():
        try:
            parsed = max(0.0, float(item))
        except (TypeError, ValueError):
            continue
        out[str(key)] = parsed
    return out


def _merge_add(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    out = dict(left)
    for key, value in right.items():
        out[key] = out.get(key, 0.0) + float(value)
    return out


def _pct(amount: float, capital: float) -> float:
    return amount / capital * 100.0 if capital > 0 else 0.0
