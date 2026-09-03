#!/usr/bin/env python3
"""Exact 60-session / 24-name walk-forward after Phase A backfill."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from product.historical_replay import run_walk_forward_sample
from product.pit_backfill import WALK_FORWARD_UNIVERSE
from product.pit_coverage import data_debt
from product.pit_warehouse import counts
from product.data_integrity import audit_decisions, audit_warehouse

OUT = Path("/opt/cursor/artifacts/phaseA_walkforward.json")
DIR = Path("/workspace/logs/product/historical_replay_phaseA")


def main() -> None:
    payload = run_walk_forward_sample(
        sessions=60,
        universe_limit=24,
        symbols=WALK_FORWARD_UNIVERSE,
        directory=DIR,
    )
    rows = list(payload.get("decisions") or payload.get("rows") or [])
    summary = {
        "period_start": payload.get("period_start"),
        "period_end": payload.get("period_end"),
        "trading_sessions": payload.get("trading_sessions"),
        "universe": list(WALK_FORWARD_UNIVERSE),
        "universe_n": 24,
        "universe_observations": payload.get("universe_observations"),
        "stocks_evaluated": payload.get("stocks_evaluated"),
        "decisions_tested": payload.get("decisions_tested"),
        "PIT_STRONG": payload.get("PIT_STRONG"),
        "PIT_PARTIAL": payload.get("PIT_PARTIAL"),
        "PIT_MARKET_ONLY": payload.get("PIT_MARKET_ONLY"),
        "PIT_UNAVAILABLE": payload.get("PIT_UNAVAILABLE"),
        "PIT_UNVERIFIED": payload.get("PIT_UNVERIFIED"),
        "BUY": payload.get("BUY"),
        "WAIT": payload.get("WAIT"),
        "AVOID": payload.get("AVOID"),
        "REJECT": payload.get("REJECT"),
        "warehouse": counts(),
        "data_debt": payload.get("data_debt") or data_debt(rows),
        "integrity": {
            "warehouse": audit_warehouse(),
            "decisions": audit_decisions(rows),
        },
        "scorecards_n": (payload.get("scorecards") or {}).get("n_rows"),
        "experiments_enqueued": len(payload.get("experiments_enqueued") or []),
        "simple": payload.get("simple"),
        "baseline_phase6": {
            "PIT_STRONG": 0,
            "PIT_PARTIAL": 179,
            "PIT_MARKET_ONLY": 0,
            "BUY": 0,
            "WAIT": 119,
            "AVOID": 60,
        },
        "note": (
            "BUY is not required. Independence and PIT rules are unchanged. "
            "PIT_STRONG is production-comparable judgment, not two PDFs."
        ),
        "live_locked": True,
        "provenance": payload.get("provenance"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    Path("/opt/cursor/artifacts/phaseA_walkforward_full.json").write_text(
        json.dumps({k: payload.get(k) for k in payload if k not in {"decisions", "rows"}}, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
