#!/usr/bin/env python3
"""Phase I soak: exercise paper/shadow/exit/heat without unlocking live money."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from product.exit_engine import evaluate_exit
from product.paper_execution_model import model_fill
from product.portfolio_heat import measure, persist
from product.portfolio_stress import run_scenarios
from product.shadow_execution import SHADOW_NOT_EXECUTED, freeze_shadow, is_paper_fill

OUT = Path("/opt/cursor/artifacts/phaseI_soak.json")


def main() -> None:
    buy = {
        "symbol": "INFY",
        "decision": "BUY",
        "entry": 1600,
        "stop": 1520,
        "target": 1760,
        "atr": 40,
        "qty": 12,
        "sector": "IT",
    }
    shadow = freeze_shadow(buy, path=Path("/tmp/phaseI_shadow.jsonl"))
    paper = model_fill({**buy, "status": "PAPER_ENTERED"}, last=1602, open_px=1598)
    blocked = model_fill({**buy, "status": SHADOW_NOT_EXECUTED, "not_a_trade": True})
    heat = measure(
        [buy, {"symbol": "TCS", "entry": 3200, "stop": 3040, "qty": 6, "sector": "IT"}],
        capital=1_000_000,
    )
    persist(heat, path=Path("/tmp/phaseI_heat.json"))
    stress = run_scenarios(
        [buy, {"symbol": "TCS", "entry": 3200, "stop": 3040, "qty": 6, "sector": "IT", "last": 3200}],
        capital=1_000_000,
    )
    exits = evaluate_exit(buy, last_price=1510)
    payload = {
        "shadow_status": shadow["status"],
        "shadow_is_paper_fill": is_paper_fill(shadow),
        "paper_perfect_fill": paper.get("perfect_fill"),
        "shadow_does_not_inflate_paper": blocked.get("inflates_paper_stats") is False,
        "heat": heat,
        "stress": {k: stress[k] for k in stress if k != "base_heat"},
        "exit_on_gap": exits,
        "live_locked": True,
        "note": "Soak exercises real policy objects. It does not send broker orders.",
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "shadow": shadow["status"],
        "paper_fill_not_perfect": paper.get("perfect_fill") is False,
        "heat_pct": heat.get("gross_open_risk_pct"),
        "exit": exits.get("action"),
        "live_locked": True,
    }, indent=2))


if __name__ == "__main__":
    main()
