"""CA Phase II — classify unresolved discontinuity classes. No inferred factors."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from data.ca_research import CA_COMPLETE, CA_RESEARCH_ACCEPTABLE, disposition, research_status
from research.sepa.ca_audit import EVENT_CLASSES, classify_subject

ROOT = Path(__file__).resolve().parents[2] / "logs"


def audit() -> dict[str, Any]:
    st = research_status()
    div_path = ROOT / "ca_dividends_provenance.json"
    dividends = 0
    if div_path.exists():
        try:
            raw = json.loads(div_path.read_text(encoding="utf-8"))
            rows = raw.get("events") or raw.get("rows") or raw if isinstance(raw, list) else []
            dividends = len(rows) if isinstance(rows, list) else 0
        except Exception:
            dividends = 0
    todo = ROOT / "ca_events.todo.csv"
    todo_n = 0
    subjects: Counter[str] = Counter()
    if todo.exists():
        import csv
        with todo.open() as f:
            for row in csv.DictReader(f):
                todo_n += 1
                subjects[classify_subject(str(row.get("subject") or row.get("purpose") or ""))] += 1
    classes = {c: disposition(c) for c in (
        "rights", "demerger", "merger", "special_distribution",
        "symbol_restructuring", "unknown_discontinuity",
    )}
    return {
        **{k: st.get(k) for k in (
            "events", "symbols", "ca_research_acceptable", "ca_complete", "label", "status",
        )},
        "dividends_provenance_rows": dividends,
        "todo_rows": todo_n,
        "todo_subject_classes": dict(subjects),
        "unresolved_disposition": classes,
        "never_infers_from_gaps": True,
        "ca_complete": False,
        "note": (
            f"{CA_RESEARCH_ACCEPTABLE} unchanged. {CA_COMPLETE} still false. "
            "Rights/demergers/mergers remain quarantine/segment. "
            "Dividends stored as provenance only."
        ),
    }
