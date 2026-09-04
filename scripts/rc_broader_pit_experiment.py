#!/usr/bin/env python3
"""Broader PIT-safe historical experiment. Observe-only. Do not tune from this run.

Keeps the 24x60 smoke experiment unchanged. Uses warehouse-covered liquid names
across a longer official-session window.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from product.historical_replay import official_sessions, run_walk_forward_sample
from product.pit_backfill import WALK_FORWARD_UNIVERSE
from product.pit_coverage import data_debt
from product.pit_warehouse import _connect, counts
from product.data_integrity import audit_decisions, audit_warehouse
from data.nse_universe import NIFTY50, NIFTY100

ARTIFACT_ROOT = Path(os.environ.get("QT_ARTIFACTS") or "/opt/cursor/artifacts")
DIR = Path(os.environ.get("QT_REPLAY_DIR") or ROOT / "logs" / "product" / "historical_replay_broader")


def warehouse_symbols() -> list[str]:
    con = _connect()
    try:
        rows = con.execute("SELECT DISTINCT symbol FROM evidence ORDER BY symbol").fetchall()
    finally:
        con.close()
    return [str(r[0]).upper() for r in rows if r[0]]


def main() -> None:
    sessions_on_disk = official_sessions()
    covered = set(warehouse_symbols())
    liquid = [s for s in NIFTY100 if s in covered]
    if len(liquid) < 30:
        liquid = [s for s in (list(NIFTY50) + list(WALK_FORWARD_UNIVERSE) + warehouse_symbols()) if s in covered]
    # Unique, stable order.
    universe = list(dict.fromkeys(liquid))[:50]
    window = min(180, max(0, len(sessions_on_disk) - 1))
    payload = run_walk_forward_sample(
        sessions=window,
        universe_limit=len(universe),
        symbols=universe,
        directory=DIR,
    )
    rows = list(payload.get("decisions") or payload.get("rows") or [])
    summary = {
        "kind": "broader_pit_observe_only",
        "not_promotion_evidence": True,
        "parameters_not_tuned": True,
        "period_start": payload.get("period_start"),
        "period_end": payload.get("period_end"),
        "trading_sessions": payload.get("trading_sessions"),
        "sessions_on_disk": len(sessions_on_disk),
        "universe": universe,
        "universe_n": len(universe),
        "universe_source": "NIFTY100 intersect PIT warehouse coverage, cap 50",
        "universe_observations": payload.get("universe_observations"),
        "stocks_evaluated": payload.get("stocks_evaluated"),
        "committee_evaluations": payload.get("decisions_tested"),
        "PIT_STRONG": payload.get("PIT_STRONG"),
        "PIT_PARTIAL": payload.get("PIT_PARTIAL"),
        "PIT_MARKET_ONLY": payload.get("PIT_MARKET_ONLY"),
        "PIT_UNAVAILABLE": payload.get("PIT_UNAVAILABLE"),
        "PIT_UNVERIFIED": payload.get("PIT_UNVERIFIED"),
        "BUY": payload.get("BUY"),
        "WAIT": payload.get("WAIT"),
        "AVOID": payload.get("AVOID"),
        "REJECT": payload.get("REJECT"),
        "R_outcomes": {
            "n_with_forward_return": sum(1 for r in rows if r.get("forward_return_pct") is not None),
            "n_missing_forward_return": sum(1 for r in rows if r.get("forward_return_pct") is None),
        },
        "sample_sizes": {
            "sessions": payload.get("trading_sessions"),
            "symbol_sessions": payload.get("universe_observations"),
            "committee": payload.get("decisions_tested"),
        },
        "warehouse": counts(),
        "data_debt": (payload.get("data_debt") or data_debt(rows)).get("summary") if isinstance(payload.get("data_debt") or data_debt(rows), dict) else payload.get("data_debt"),
        "integrity": {
            "warehouse": audit_warehouse(),
            "decisions": audit_decisions(rows),
        },
        "coverage_limitations": [
            "Warehouse coverage, not the full NSE listed universe.",
            "Static sector labels remain SECTOR_MEMBERSHIP_APPROXIMATE.",
            "Unverified publication dates stay excluded from get_evidence.",
            "This run is BACKTEST provenance and must not mix with REAL_FORWARD_MARKET.",
        ],
        "live_locked": True,
        "provenance": payload.get("provenance"),
        "simple": payload.get("simple"),
        "smoke_24x60_unchanged": True,
    }
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_ROOT / "broader_pit_experiment.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
