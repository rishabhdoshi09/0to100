"""Operational source quality — availability, not popularity.

Primary sources stay authoritative. A flaky aggregator can be deprioritized
for routing. Epistemic quality is never lowered because a weak source is easier.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "logs" / "product" / "source_quality.jsonl"


def record_source_event(
    *,
    source: str,
    available: bool,
    parser_ok: bool,
    freshness_s: float | None = None,
    conflict: bool = False,
    latency_s: float | None = None,
    primary: bool = False,
    path: Path | None = None,
) -> dict[str, Any]:
    row = {
        "at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "available": bool(available),
        "parser_ok": bool(parser_ok),
        "freshness_s": freshness_s,
        "conflict": bool(conflict),
        "latency_s": latency_s,
        "primary": bool(primary),
    }
    target = path or LEDGER
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def routing_hint(rows: list[Mapping[str, Any]] | None = None, *, path: Path | None = None) -> dict[str, Any]:
    if rows is None:
        target = path or LEDGER
        rows = []
        if target.exists():
            for line in target.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
    by: dict[str, dict[str, Any]] = {}
    for row in rows:
        src = str(row.get("source") or "unknown")
        bucket = by.setdefault(src, {"n": 0, "fail": 0, "conflict": 0, "primary": False})
        bucket["n"] += 1
        if not row.get("available") or not row.get("parser_ok"):
            bucket["fail"] += 1
        if row.get("conflict"):
            bucket["conflict"] += 1
        bucket["primary"] = bucket["primary"] or bool(row.get("primary"))
    hints = []
    for src, b in by.items():
        fail_rate = b["fail"] / b["n"] if b["n"] else 0
        if not b["primary"] and b["n"] >= 8 and fail_rate >= 0.5:
            hints.append({"source": src, "action": "DEPRIORITIZE_ROUTING", "fail_rate": round(fail_rate, 2), "n": b["n"]})
        else:
            hints.append({"source": src, "action": "KEEP", "fail_rate": round(fail_rate, 2), "n": b["n"]})
    return {
        "sources": by,
        "hints": hints,
        "note": "Routing only. Primary sources remain authoritative.",
        "affects_epistemic_rank": False,
    }
