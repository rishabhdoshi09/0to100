"""Read-only gatherer for the retail product projection.

This module observes canonical backend objects and converts them to ProductInputs.
It never writes state, starts daemons, runs a research cycle, or places an order.
All backend access is guarded so disconnected data cannot crash the retail UI.
"""
from __future__ import annotations

from datetime import time
from typing import Any

from core.market_clock import now_ist
from product.projection import ProductInputs


def _market_is_open(now) -> bool:
    return now.weekday() < 5 and time(9, 15) <= now.time().replace(tzinfo=None) <= time(15, 30)


def _safe_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def gather_product_inputs() -> ProductInputs:
    """Observe the current canonical runtime without mutating it."""
    now = now_ist()
    market_open = _market_is_open(now)

    snapshot_id = None
    snapshot_verified = None
    snapshot_last_date = None
    snapshot_instruments = None
    snapshot_benchmark = None
    snapshot_universe = None
    snapshot_actions = None

    paper_mode = None
    paper_enabled = None
    runtime_reconciled = None
    cycle_running = None
    last_completed_cycle = None
    last_cycle_error = ""
    paper_capital = None
    paper_equity = None
    paper_open_risk = None
    paper_open_positions = ()
    paper_closed_trades = None
    paper_refusals = ()

    attention: list[str] = []

    try:
        from research.auto_research import get_brain

        brain = get_brain()
        paper_mode = getattr(brain, "mode", None)
        paper_enabled = _safe_bool(getattr(brain, "paper_auto_enabled", None))
        cycle_running = _safe_bool(getattr(getattr(brain, "state", None), "running", None))
        last_cycle_error = str(getattr(getattr(brain, "state", None), "last_error", "") or "")

        runtime_state = getattr(brain, "runtime_state", None)
        runtime_reconciled = _safe_bool(getattr(runtime_state, "reconciled", None))
        last_completed_cycle = getattr(runtime_state, "last_completed_cycle", None)

        book = getattr(brain, "intel_book", None)
        if book is not None:
            paper_capital = float(getattr(book, "capital", 0.0))
            paper_equity = float(book.equity())
            paper_open_risk = float(book.open_risk())
            paper_open_positions = tuple(
                position.as_dict() if hasattr(position, "as_dict") else dict(vars(position))
                for position in getattr(book, "open", {}).values()
            )
            paper_closed_trades = len(getattr(book, "closed", ()))
            paper_refusals = tuple(getattr(book, "refusals", ()))

        store = getattr(brain, "snapshot_store", None)
        if store is not None:
            snapshot_id = store.get_active_snapshot()
            if snapshot_id:
                snapshot_verified, failures = store.verify_snapshot(snapshot_id)
                if failures:
                    attention.extend(str(item) for item in failures)
                if snapshot_verified:
                    snap = store.open_snapshot(snapshot_id)
                    health = snap.health()
                    raw_date = health.get("last_trading_date")
                    if raw_date:
                        try:
                            from datetime import date

                            snapshot_last_date = date.fromisoformat(str(raw_date)[:10])
                        except Exception:
                            snapshot_last_date = None
                    snapshot_instruments = health.get("instrument_count")
                    snapshot_benchmark = _safe_bool(health.get("has_benchmark"))
                    snapshot_universe = _safe_bool(health.get("has_universe_history"))
                    coverage = health.get("corporate_action_coverage")
                    if isinstance(coverage, bool):
                        snapshot_actions = coverage
                    elif isinstance(coverage, (int, float)):
                        snapshot_actions = coverage >= 0.99
    except Exception as exc:
        attention.append(f"Runtime state could not be read: {exc}")

    broker_connected = None
    instrument_source = None
    instrument_count = None
    try:
        from research.intelligence.data.kite_activation import KiteDataClient

        client = KiteDataClient.from_config()
        broker_connected = True
        try:
            from data.fno_universe import current_fno_universe

            report = current_fno_universe(client)
            instrument_source = report.source
            instrument_count = report.total_instrument_rows
        except Exception as exc:
            attention.append(f"Instrument master could not be read: {exc}")
    except Exception:
        broker_connected = False
        try:
            from data.fno_universe import current_fno_universe

            report = current_fno_universe(None)
            instrument_source = report.source
            instrument_count = report.total_instrument_rows
        except Exception:
            pass

    market_condition = None
    market_reason = ""
    try:
        from research.auto_research.providers import current_regime

        regime = current_regime()
        if regime:
            market_condition = str(regime).replace("_", " ").title()
            market_reason = "Canonical regime provider"
    except Exception:
        pass

    return ProductInputs(
        observed_at=now,
        market_open=market_open,
        market_condition=market_condition,
        market_condition_reason=market_reason,
        snapshot_id=snapshot_id,
        snapshot_verified=snapshot_verified,
        snapshot_last_trading_date=snapshot_last_date,
        snapshot_instrument_count=snapshot_instruments,
        snapshot_has_benchmark=snapshot_benchmark,
        snapshot_has_universe_history=snapshot_universe,
        snapshot_has_corporate_actions=snapshot_actions,
        live_data_available=None,
        live_data_timestamp=None,
        broker_connected=broker_connected,
        instrument_master_source=instrument_source,
        instrument_master_count=instrument_count,
        paper_mode=paper_mode,
        paper_auto_enabled=paper_enabled,
        runtime_reconciled=runtime_reconciled,
        cycle_running=cycle_running,
        last_completed_cycle=last_completed_cycle,
        last_cycle_error=last_cycle_error,
        paper_capital=paper_capital,
        paper_equity=paper_equity,
        paper_open_risk=paper_open_risk,
        paper_open_positions=paper_open_positions,
        paper_closed_trades=paper_closed_trades,
        paper_refusals=paper_refusals,
        qualified_opportunities=None,
        opportunity_source=None,
        attention_items=tuple(attention),
    )
