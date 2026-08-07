"""Guarded read-only collection of existing QuantTerm backend state."""
from __future__ import annotations

import json
from pathlib import Path

from product.projection import ProductInputs
from product.paper_status import read_paper_status
from product.autonomy_status import read_autonomy_status


def gather_product_inputs() -> ProductInputs:
    market_open = False
    market_label = "Market closed"
    try:
        from core.market_session import in_market_open, status_line
        market_open = bool(in_market_open()); market_label = str(status_line())
    except Exception:
        pass

    kite_connected = False
    try:
        from data.kite_client import _fresh_env
        kite_connected = bool(_fresh_env("KITE_ACCESS_TOKEN"))
    except Exception:
        pass

    active_id = None; latest_date = ""; instrument_count = 0; data_ready = False
    root = Path(__file__).resolve().parents[1]
    try:
        from research.intelligence.data.snapshot_store import SnapshotStore
        store = SnapshotStore(root / "logs" / "snapshots")
        active_id = store.get_active_snapshot()
        if active_id:
            manifest_path = Path(store.root) / active_id / "manifest.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            latest_date = str(payload.get("last_trading_date") or "")
            instrument_count = int(payload.get("instrument_count") or 0)
            verified, _ = store.verify_snapshot(active_id)
            fresh = False
            if latest_date:
                try:
                    from research.intelligence.data import nse_calendar as CAL
                    fresh = bool(CAL.snapshot_freshness(latest_date).get("fresh"))
                except Exception:
                    fresh = False
            data_ready = bool(verified and payload.get("has_benchmark") and fresh)
    except Exception:
        pass

    paper = read_paper_status(repo_root=root)
    autonomy = read_autonomy_status()
    return ProductInputs(
        market_open=market_open, market_label=market_label, kite_connected=kite_connected,
        active_snapshot_id=active_id, latest_market_date=latest_date,
        instrument_count=instrument_count, data_ready=data_ready,
        paper_auto_enabled=paper.enabled, worker_running=paper.supervisor_running,
        paper_capital=paper.capital, paper_equity=paper.equity,
        open_positions=len(paper.open_positions), closed_trades=len(paper.closed_trades),
        last_cycle_status=str(paper.last_cycle.get("status") or autonomy.get("state") or ""),
        last_error="" if autonomy.get("state") in ("OBSERVING", "PAPER_ACTIVE", "DATA_READY")
                   else str(autonomy.get("explanation") or ""),
    )
