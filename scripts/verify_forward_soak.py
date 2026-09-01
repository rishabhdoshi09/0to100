#!/usr/bin/env python3
"""Operator verifier for the real forward paper-trading soak.

Reads persisted artifacts only. Does not inject mock market rows.

Usage (from repo root, after a market-day autonomy run):

    python scripts/verify_forward_soak.py

A valid no-trade day is not a failure when every eligible candidate was
rejected or waited with a machine-readable reason.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


LANE_ORDER = (
    "SCAN",
    "RECOMMENDATIONS",
    "SELECTION",
    "AUTOPILOT",
    "PAPER EXECUTION",
    "EXIT SUPERVISION",
    "FORWARD SETTLEMENT",
    "LEARNING INGESTION",
    "EXECUTION REALITY",
    "LIVE MONEY",
)


def main() -> int:
    from product.forward_soak import persist_soak_verification, verify_persisted_soak

    result = verify_persisted_soak()
    try:
        persist_soak_verification(force=True)
    except Exception:
        pass
    lanes = dict(result.get("lanes") or {})
    for name in LANE_ORDER:
        print(f"{name}: {lanes.get(name, 'FAIL')}")
    print(f"FORWARD_SOAK_STATUS: {result.get('soak_status')}")
    print(f"EVIDENCE: {result.get('scoreboard_evidence')}")
    print(f"REAL_FORWARD_N: {result.get('real_forward_n')}")
    coverage = result.get("execution_adjusted_coverage_pct")
    print(f"EXECUTION_ADJUSTED_COVERAGE_PCT: {coverage if coverage is not None else 'n/a'}")
    if result.get("valid_no_trade"):
        print("NOTE: valid no-trade day — eligible names were rejected or waited with reasons")
    if not result.get("live_locked"):
        print("LIVE MONEY UNLOCKED — fail-closed contract broken")
        return 1
    hard = {
        "SCAN",
        "RECOMMENDATIONS",
        "SELECTION",
        "AUTOPILOT",
        "PAPER EXECUTION",
        "LIVE MONEY",
    }
    failed = [
        name for name in hard
        if str(lanes.get(name) or "") == "FAIL"
    ]
    if failed:
        print("FAILED LANES: " + ", ".join(failed))
        return 1
    if str(lanes.get("LIVE MONEY") or "") != "LOCKED":
        print("LIVE MONEY is not LOCKED")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
