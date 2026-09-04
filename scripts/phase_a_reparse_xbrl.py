#!/usr/bin/env python3
"""Re-parse already-downloaded official XBRL with the current parser.

Does not call NSE. Writes new revision rows; never mutates old extracts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from product.pit_backfill import WALK_FORWARD_UNIVERSE, _persist_parsed_xbrl
from product.pit_warehouse import get_evidence_raw

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "logs" / "research_evidence"
OUT = Path("/opt/cursor/artifacts/phaseA_xbrl_reparse.json")


def main() -> None:
    reports = []
    for name in WALK_FORWARD_UNIVERSE:
        folder = EVIDENCE / name / "autonomy"
        by_id = {}
        for row in get_evidence_raw(name):
            sid = str(row.get("source_identity") or "")
            if sid.startswith("nse_xbrl:"):
                by_id.setdefault(sid, row)
        parsed = 0
        missing = 0
        for xml in sorted(folder.glob("nse_xbrl_*.xml")):
            stem = xml.stem  # nse_xbrl_integrated_175753
            parts = stem.split("_")
            kind = parts[2] if len(parts) >= 4 else "xbrl"
            seq = parts[-1]
            sid = f"nse_xbrl:{kind}:{seq}"
            prior = by_id.get(sid) or {}
            xml_text = xml.read_text(encoding="utf-8", errors="ignore")
            stored = _persist_parsed_xbrl(
                name,
                xml_text=xml_text,
                publication=str(prior.get("publication_date") or prior.get("available_from") or ""),
                period_end=str(prior.get("period_end") or ""),
                source_url=str(prior.get("source_url") or xml.as_posix()),
                source_identity=sid,
                extra={
                    "broadcast": prior.get("exchange_timestamp"),
                    "consolidated": (prior.get("extracted") or {}).get("consolidated")
                    or (prior.get("extracted") or {}).get("nature"),
                    "revised": True,
                },
            )
            if (stored.get("extracted") or {}).get("numbers_parsed") or stored.get("deduped"):
                parsed += 1
            else:
                missing += 1
        reports.append({"symbol": name, "parsed": parsed, "missing": missing})
        print(reports[-1], flush=True)
    OUT.write_text(json.dumps({"details": reports}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
