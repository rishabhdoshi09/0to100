"""Pure read-only projections from canonical QuantTerm state to retail views.

No function in this module reads files, reaches the network, persists state,
calculates signals, or places orders. Callers provide already-observed backend
facts through :class:`ProductInputs`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Mapping, Sequence


class ProductStatus(str, Enum):
    READY = "Ready"
    ATTENTION = "Needs attention"
    MISSING = "Missing"
    STALE = "Stale"
    UNKNOWN = "Unknown"
    NOT_REQUIRED = "Not required for this mode"


@dataclass(frozen=True)
class ReadinessCard:
    key: str
    title: str
    status: ProductStatus
    summary: str
    detail: str = ""
    action: str = ""


@dataclass(frozen=True)
class ProductInputs:
    observed_at: datetime
    market_open: bool | None = None
    market_condition: str | None = None
    market_condition_reason: str = ""

    snapshot_id: str | None = None
    snapshot_verified: bool | None = None
    snapshot_last_trading_date: date | None = None
    snapshot_instrument_count: int | None = None
    snapshot_has_benchmark: bool | None = None
    snapshot_has_universe_history: bool | None = None
    snapshot_has_corporate_actions: bool | None = None

    live_data_available: bool | None = None
    live_data_timestamp: datetime | None = None
    broker_connected: bool | None = None
    instrument_master_source: str | None = None
    instrument_master_count: int | None = None

    paper_mode: str | None = None
    paper_auto_enabled: bool | None = None
    runtime_reconciled: bool | None = None
    cycle_running: bool | None = None
    last_completed_cycle: str | None = None
    last_cycle_error: str = ""

    paper_capital: float | None = None
    paper_equity: float | None = None
    paper_open_risk: float | None = None
    paper_open_positions: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    paper_closed_trades: int | None = None
    paper_refusals: Sequence[Sequence[Any]] = field(default_factory=tuple)

    qualified_opportunities: int | None = None
    opportunity_source: str | None = None
    opportunity_timestamp: datetime | None = None
    attention_items: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class HomeProjection:
    overall: ReadinessCard
    research: ReadinessCard
    live: ReadinessCard
    broker: ReadinessCard
    paper: ReadinessCard
    market: ReadinessCard
    opportunities: ReadinessCard
    primary_action: str
    attention_items: tuple[str, ...]


@dataclass(frozen=True)
class PaperTradingProjection:
    status: ProductStatus
    mode_label: str
    entries_allowed: bool
    entries_reason: str
    capital: float | None
    equity: float | None
    open_risk: float | None
    open_positions: tuple[Mapping[str, Any], ...]
    closed_trades: int | None
    last_completed_cycle: str
    cycle_running: bool | None
    last_error: str
    can_engage: bool
    can_disengage: bool


def _days_old(as_of: date, value: date | None) -> int | None:
    if value is None:
        return None
    return max(0, (as_of - value).days)


def _research_card(inputs: ProductInputs) -> ReadinessCard:
    if not inputs.snapshot_id:
        return ReadinessCard(
            "research", "Historical research data", ProductStatus.MISSING,
            "Historical test data is not ready.",
            "Activate a verified immutable snapshot before running evidence tests.",
            "Open Data & Broker",
        )
    if inputs.snapshot_verified is False:
        return ReadinessCard(
            "research", "Historical research data", ProductStatus.ATTENTION,
            "The active snapshot failed verification.",
            "QuantTerm will not use an unverifiable snapshot for research.",
            "Repair or activate another snapshot",
        )
    age = _days_old(inputs.observed_at.date(), inputs.snapshot_last_trading_date)
    missing = []
    if inputs.snapshot_has_benchmark is False:
        missing.append("benchmark")
    if inputs.snapshot_has_universe_history is False:
        missing.append("universe history")
    if inputs.snapshot_has_corporate_actions is False:
        missing.append("corporate actions")
    if missing:
        return ReadinessCard(
            "research", "Historical research data", ProductStatus.ATTENTION,
            "The snapshot is active but incomplete for some research.",
            "Missing: " + ", ".join(missing) + ".",
            "Review data coverage",
        )
    if age is not None and age > 7:
        return ReadinessCard(
            "research", "Historical research data", ProductStatus.STALE,
            f"The active research snapshot is {age} days behind.",
            "Existing tests remain reproducible, but recent research needs a newer snapshot.",
            "Refresh historical data",
        )
    instruments = inputs.snapshot_instrument_count
    detail = f"Snapshot {inputs.snapshot_id}"
    if instruments is not None:
        detail += f" · {instruments:,} instruments"
    return ReadinessCard(
        "research", "Historical research data", ProductStatus.READY,
        "Historical test data is verified and active.", detail,
    )


def _live_card(inputs: ProductInputs) -> ReadinessCard:
    if inputs.market_open is False and inputs.live_data_available is not True:
        return ReadinessCard(
            "live", "Live market data", ProductStatus.NOT_REQUIRED,
            "Live quotes are not required while the market is closed.",
            "Historical research and paper-book review remain available.",
        )
    if inputs.live_data_available is False:
        return ReadinessCard(
            "live", "Live market data", ProductStatus.MISSING,
            "Live market data is unavailable.",
            "This pauses live screening; it does not invalidate historical research.",
            "Check Data & Broker",
        )
    if inputs.live_data_available is None:
        return ReadinessCard(
            "live", "Live market data", ProductStatus.UNKNOWN,
            "Live data availability has not been verified.",
            "Unknown is kept separate from a healthy or failed state.",
            "Check Data & Broker",
        )
    if inputs.live_data_timestamp is not None:
        age_s = max(0.0, (inputs.observed_at - inputs.live_data_timestamp).total_seconds())
        if inputs.market_open is True and age_s > 15 * 60:
            return ReadinessCard(
                "live", "Live market data", ProductStatus.STALE,
                "Live data is connected but stale.",
                f"Last observation was {int(age_s // 60)} minutes ago.",
                "Refresh the live connection",
            )
    return ReadinessCard(
        "live", "Live market data", ProductStatus.READY,
        "Live market data is available.",
        f"Source timestamp: {inputs.live_data_timestamp.isoformat()}" if inputs.live_data_timestamp else "",
    )


def _broker_card(inputs: ProductInputs) -> ReadinessCard:
    if inputs.broker_connected is True:
        return ReadinessCard(
            "broker", "Zerodha connection", ProductStatus.READY,
            "Zerodha data access is connected.",
            "This status does not grant live-order permission.",
        )
    if inputs.broker_connected is False:
        return ReadinessCard(
            "broker", "Zerodha connection", ProductStatus.MISSING,
            "Zerodha is not connected.",
            "Paper research can still run from an active snapshot.",
            "Open Data & Broker",
        )
    return ReadinessCard(
        "broker", "Zerodha connection", ProductStatus.UNKNOWN,
        "Zerodha connection has not been verified.",
        "Unknown is not treated as connected.",
        "Open Data & Broker",
    )


def build_paper_trading_projection(inputs: ProductInputs) -> PaperTradingProjection:
    mode = (inputs.paper_mode or "UNKNOWN").upper()
    enabled = inputs.paper_auto_enabled is True and mode == "PAPER_AUTO"
    reconciled = inputs.runtime_reconciled is True
    has_error = bool(inputs.last_cycle_error)
    entries_allowed = enabled and reconciled and not has_error
    if mode != "PAPER_AUTO":
        reason = "Automatic paper entries are off because the runtime is not in PAPER_AUTO mode."
    elif inputs.paper_auto_enabled is not True:
        reason = "Automatic paper entries are switched off."
    elif inputs.runtime_reconciled is False:
        reason = "New paper trades are paused because saved state and the paper book do not reconcile."
    elif inputs.runtime_reconciled is None:
        reason = "New paper-trade eligibility is unknown because reconciliation was not verified."
    elif has_error:
        reason = "New paper trades are paused because the last intelligence cycle reported an error."
    else:
        reason = "New paper entries may proceed when the canonical strategy and data gates also pass."

    if entries_allowed:
        status = ProductStatus.READY
    elif inputs.paper_auto_enabled is None or inputs.runtime_reconciled is None:
        status = ProductStatus.UNKNOWN
    else:
        status = ProductStatus.ATTENTION

    label = {
        "PAPER_AUTO": "Automatic paper trading",
        "PAPER": "Paper observation",
        "OFF": "Off",
    }.get(mode, "Unknown")
    return PaperTradingProjection(
        status=status,
        mode_label=label,
        entries_allowed=entries_allowed,
        entries_reason=reason,
        capital=inputs.paper_capital,
        equity=inputs.paper_equity,
        open_risk=inputs.paper_open_risk,
        open_positions=tuple(inputs.paper_open_positions),
        closed_trades=inputs.paper_closed_trades,
        last_completed_cycle=inputs.last_completed_cycle or "Not yet completed",
        cycle_running=inputs.cycle_running,
        last_error=inputs.last_cycle_error,
        can_engage=mode == "PAPER_AUTO" and inputs.paper_auto_enabled is not True,
        can_disengage=mode == "PAPER_AUTO" and inputs.paper_auto_enabled is True,
    )


def _paper_card(inputs: ProductInputs) -> ReadinessCard:
    p = build_paper_trading_projection(inputs)
    return ReadinessCard(
        "paper", "Automatic paper trading", p.status,
        p.entries_reason,
        f"Open positions: {len(p.open_positions)} · Last cycle: {p.last_completed_cycle}",
        "Open Paper Trading",
    )


def _market_card(inputs: ProductInputs) -> ReadinessCard:
    if not inputs.market_condition:
        return ReadinessCard(
            "market", "Market condition", ProductStatus.UNKNOWN,
            "The market condition is not available yet.",
            "QuantTerm will not invent a bullish or bearish label.",
        )
    return ReadinessCard(
        "market", "Market condition", ProductStatus.READY,
        inputs.market_condition,
        inputs.market_condition_reason,
    )


def _opportunity_card(inputs: ProductInputs) -> ReadinessCard:
    if inputs.qualified_opportunities is None:
        return ReadinessCard(
            "opportunities", "Qualified opportunities", ProductStatus.UNKNOWN,
            "The opportunity scan has not produced a verified count.",
            "No unknown value is converted to zero.",
            "Open Momentum",
        )
    if inputs.qualified_opportunities == 0:
        return ReadinessCard(
            "opportunities", "Qualified opportunities", ProductStatus.READY,
            "No qualified opportunities right now.",
            "No trade can be the correct result when the rules reject every candidate.",
            "Review the evaluation funnel",
        )
    return ReadinessCard(
        "opportunities", "Qualified opportunities", ProductStatus.READY,
        f"{inputs.qualified_opportunities} qualified opportunit"
        + ("y" if inputs.qualified_opportunities == 1 else "ies") + " found.",
        f"Source: {inputs.opportunity_source or 'canonical scan'}",
        "Open Momentum",
    )


def build_home_projection(inputs: ProductInputs) -> HomeProjection:
    research = _research_card(inputs)
    live = _live_card(inputs)
    broker = _broker_card(inputs)
    paper = _paper_card(inputs)
    market = _market_card(inputs)
    opportunities = _opportunity_card(inputs)

    blockers = [card for card in (research, paper) if card.status in {
        ProductStatus.MISSING, ProductStatus.STALE, ProductStatus.ATTENTION, ProductStatus.UNKNOWN
    }]
    if blockers:
        overall = ReadinessCard(
            "overall", "QuantTerm readiness", ProductStatus.ATTENTION,
            "QuantTerm needs attention before new automatic paper entries.",
            "; ".join(card.summary for card in blockers),
        )
    else:
        overall = ReadinessCard(
            "overall", "QuantTerm readiness", ProductStatus.READY,
            "QuantTerm is ready for evidence-gated paper operation.",
            "Ready does not mean a trade must be placed.",
        )

    if research.status in {ProductStatus.MISSING, ProductStatus.STALE, ProductStatus.ATTENTION}:
        primary = "Open Data & Broker and fix historical research data."
    elif paper.status in {ProductStatus.ATTENTION, ProductStatus.UNKNOWN}:
        primary = "Open Paper Trading and review why new entries are paused."
    elif opportunities.status == ProductStatus.UNKNOWN:
        primary = "Run or open Momentum to inspect the complete evaluation funnel."
    elif inputs.qualified_opportunities == 0:
        primary = "Take no trade; review the funnel and wait for a qualified setup."
    else:
        primary = "Review the qualified setups in Momentum before paper execution."

    attention = list(inputs.attention_items)
    for card in (research, live, broker, paper):
        if card.status in {ProductStatus.MISSING, ProductStatus.STALE, ProductStatus.ATTENTION}:
            attention.append(card.summary)
    return HomeProjection(
        overall=overall,
        research=research,
        live=live,
        broker=broker,
        paper=paper,
        market=market,
        opportunities=opportunities,
        primary_action=primary,
        attention_items=tuple(dict.fromkeys(attention)),
    )
