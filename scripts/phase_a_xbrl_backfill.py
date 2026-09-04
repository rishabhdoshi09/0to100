#!/usr/bin/env python3
"""Bounded official XBRL backfill for the 24-name walk-forward universe."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from product.due_diligence.acquire import _nse_session
from product.pit_backfill import WALK_FORWARD_UNIVERSE, backfill_structured_financials
from product.pit_parser_qa import validate_xbrl_text
from product.pit_warehouse import counts

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = Path("/opt/cursor/artifacts/phaseA_xbrl_backfill.json")


def main() -> None:
    sess = _nse_session()
    reports = []
    for i, name in enumerate(WALK_FORWARD_UNIVERSE, 1):
        print(f"=== {i}/24 {name}", flush=True)
        try:
            row = backfill_structured_financials(name, session=sess, max_xbrl=8, sleep_s=0.35)
        except Exception as exc:
            row = {"symbol": name, "error": str(exc)[:300], "acquired": 0, "parsed": 0, "failed": 1}
        print(row, flush=True)
        reports.append(row)
    sample = {}
    infy_xml = ROOT / "logs" / "research_evidence" / "INFY" / "autonomy"
    for path in sorted(infy_xml.glob("nse_xbrl_*.xml"))[:1]:
        sample = validate_xbrl_text(
            path.read_text(encoding="utf-8", errors="ignore"),
            {"revenue": 39957, "pat": 7249, "basic_eps": 17.87},
            symbol="INFY",
            source=str(path),
            persist=True,
        )
    out = {
        "n": len(reports),
        "acquired": sum(int(r.get("acquired") or 0) for r in reports),
        "parsed": sum(int(r.get("parsed") or 0) for r in reports),
        "failed": sum(int(r.get("failed") or 0) for r in reports),
        "skipped": sum(int(r.get("skipped") or 0) for r in reports),
        "warehouse": counts(),
        "infy_parser_sample": sample,
        "details": reports,
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("DONE", {k: out[k] for k in ("n", "acquired", "parsed", "failed", "skipped", "warehouse")})


if __name__ == "__main__":
    main()
