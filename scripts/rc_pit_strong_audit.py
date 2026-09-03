#!/usr/bin/env python3
"""Courtroom audit of PIT_STRONG walk-forward rows. Not a product feature."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from product.pit_query import (
    get_financial_snapshot,
    get_research_snapshot,
    production_comparable,
    replay_grade_for_symbol,
)
from product.pit_warehouse import get_evidence

DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "product" / "historical_replay_phaseA"
latest = json.loads((DIR / "latest.json").read_text())
rows = [r for r in (latest.get("decisions") or latest.get("rows") or []) if r.get("pit_grade") == "PIT_STRONG"]
# Diverse sample: first, last, mid, and unique symbols.
sample = []
seen = set()
for row in rows:
    key = row.get("symbol")
    if key in seen:
        continue
    seen.add(key)
    sample.append(row)
    if len(sample) >= 10:
        break
if len(sample) < 10:
    sample = rows[:10]

out = []
for row in sample:
    symbol = str(row.get("symbol") or "").upper()
    as_of = str(row.get("as_of") or "")[:10]
    fin = get_financial_snapshot(symbol, as_of=as_of)
    research = get_research_snapshot(symbol, as_of=as_of)
    grade = replay_grade_for_symbol(symbol, as_of=as_of, market_bars_ok=True)
    items = get_evidence(symbol, as_of=as_of)
    pubs = [
        {
            "type": i.get("evidence_type"),
            "available_from": i.get("available_from"),
            "publication_date": i.get("publication_date"),
            "period_end": i.get("period_end"),
            "source": i.get("source"),
            "pit_status": i.get("pit_status"),
            "numbers_parsed": bool((i.get("extracted") or {}).get("numbers_parsed")),
        }
        for i in items[:12]
    ]
    families = row.get("evidence_family_votes") or row.get("pit_research") or {}
    rec = {
        "symbol": symbol,
        "date": as_of,
        "decision": row.get("decision"),
        "price_evidence": {
            "market_bars_ok": True,
            "max_bar_date": (row.get("pit") or {}).get("max_bar_date"),
            "future_bar": (row.get("pit") or {}).get("future_evidence_used"),
        },
        "financial_evidence": {
            "numbers_parsed": fin.get("numbers_parsed"),
            "n_parsed_results": fin.get("n_parsed_results"),
            "latest_publication": fin.get("latest_publication"),
            "latest_period_end": fin.get("latest_period_end"),
            "derived": {k: fin.get("derived", {}).get(k) for k in ("pat_margin_pct", "revenue_yoy_pct", "pbt_margin_pct")},
            "stale_for_production": fin.get("stale_for_production"),
        },
        "business_evidence": {
            "answered": research.get("answered"),
            "unknown": research.get("unknown"),
            "quality_label": research.get("quality_label"),
        },
        "sector_evidence": row.get("pit_sector") or (row.get("pit") or {}).get("coverage"),
        "other_evidence": research.get("coverage"),
        "independent_families": families,
        "unknown_families": research.get("unknown"),
        "publication_dates": pubs,
        "pit_eligibility": {
            "grade": grade.get("grade"),
            "reason": grade.get("reason"),
            "production_comparable": production_comparable(fin=fin, research=research),
            "comparable_to_forward": grade.get("comparable_to_forward"),
        },
        "reason_PIT_STRONG": grade.get("reason"),
        "one_dated_xbrl_would_not_suffice": int(fin.get("n_parsed_results") or 0) >= 2,
    }
    out.append(rec)

artifact = Path("/opt/cursor/artifacts/pit_strong_audit.json")
artifact.write_text(json.dumps({"n_strong": len(rows), "n_audited": len(out), "rows": out}, indent=2, default=str), encoding="utf-8")
print(json.dumps({"n_strong": len(rows), "n_audited": len(out), "symbols": [r["symbol"] for r in out]}, indent=2))
for rec in out:
    print(
        f"{rec['symbol']} {rec['date']} parsed={rec['financial_evidence']['n_parsed_results']} "
        f"pub={rec['financial_evidence']['latest_publication']} "
        f"answered={rec['business_evidence']['answered']} "
        f"comparable={rec['pit_eligibility']['production_comparable']} "
        f"grade={rec['pit_eligibility']['grade']}"
    )
