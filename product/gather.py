"""Guarded read-only collection of existing QuantTerm backend state."""
from __future__ import annotations

import json
from pathlib import Path

from product.projection import ProductInputs


def gather_product_inputs() -> ProductInputs:
    market_open = False
    market_label = "Market closed"
    try:
        from core.market_session import in_market_open, status_line
        market_open = bool(in_market_open())
        market_label = str(status_line())
    except Exception:
        pass

    kite_connected = False
    paper_capital = 100_000.0
    try:
        from config import settings
        kite_connected = bool(settings.kite_access_token)
        paper_capital = float(settings.trading_capital)
    except Exception:
        pass

    active_id = None
    latest_date = ""
    instrument_count = 0
    data_ready = False
    enabled = False
    running = False
    equity = paper_capital
    open_positions = closed_trades = 0
    last_cycle = last_error = ""

    try:
        from research.auto_research.scheduler import get_brain
        brain = get_brain()
        enabled = bool(brain.is_paper_auto_enabled())
        running = bool(brain.state.running)
        last_error = str(brain.state.last_error or "")
        last_intel = brain.state.last_intel_cycle or {}
        last_cycle = str(last_intel.get("status") or last_intel.get("decision") or "")
        book = brain.intel_book
        paper_capital = float(book.capital)
        equity = float(book.equity())
        open_positions = len(book.open)
        closed_trades = len(book.closed)
        store = brain.snapshot_store
        if store is not None:
            active_id = store.get_active_snapshot()
            if active_id:
                manifest = Path(store.root) / active_id / "manifest.json"
                if manifest.exists():
                    payload = json.loads(manifest.read_text(encoding="utf-8"))
                    latest_date = str(payload.get("last_trading_date") or "")
                    instrument_count = int(payload.get("instrument_count") or 0)
                    ok, _ = store.verify_snapshot(active_id)
                    data_ready = bool(ok and payload.get("has_benchmark"))
    except Exception as exc:
        last_error = last_error or str(exc)

    return ProductInputs(
        market_open=market_open,
        market_label=market_label,
        kite_connected=kite_connected,
        active_snapshot_id=active_id,
        latest_market_date=latest_date,
        instrument_count=instrument_count,
        data_ready=data_ready,
        paper_auto_enabled=enabled,
        worker_running=running,
        paper_capital=paper_capital,
        paper_equity=equity,
        open_positions=open_positions,
        closed_trades=closed_trades,
        last_cycle_status=last_cycle,
        last_error=last_error,
    )
