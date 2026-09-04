#!/usr/bin/env python3
"""Phase H residual checks after the walk-forward artifact exists."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from product.data_integrity import audit_warehouse
from product.pit_backfill import WALK_FORWARD_UNIVERSE
from product.pit_coverage import overall_replay_grade
from product.pit_financials import get_fact
from product.pit_warehouse import counts
from product.learning_ledger import learned_today

OUT = Path("/opt/cursor/artifacts/phaseH_validation.json")


def main() -> None:
    grades = {}
    for name in WALK_FORWARD_UNIVERSE:
        g = overall_replay_grade(name, as_of="2026-09-01", market_bars_ok=True)
        grades[name] = {
            "grade": g.get("grade"),
            "production_comparable": g.get("production_comparable"),
            "revenue": get_fact(name, "revenue", as_of="2026-09-01").get("value"),
            "available_from": get_fact(name, "revenue", as_of="2026-09-01").get("available_from"),
        }
    future = get_fact("INFY", "revenue", as_of="2026-06-10")
    q1 = get_fact("INFY", "revenue", as_of="2026-08-01")
    payload = {
        "warehouse": counts(),
        "integrity": audit_warehouse(as_of="2026-09-01"),
        "as_of_2026_09_01": grades,
        "n_strong": sum(1 for v in grades.values() if v["grade"] == "PIT_STRONG"),
        "n_with_revenue": sum(1 for v in grades.values() if v["revenue"] is not None),
        "infy_q1_invisible_on_2026_06_10": (
            future.get("available_from") or ""
        ) < "2026-07-23",
        "infy_q1_visible_on_2026_08_01": (q1.get("available_from") or "") >= "2026-07-23",
        "learned_today": learned_today(),
        "live_locked": True,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in (
        "warehouse", "n_strong", "n_with_revenue",
        "infy_q1_invisible_on_2026_06_10", "infy_q1_visible_on_2026_08_01",
    )}, indent=2))


if __name__ == "__main__":
    main()
