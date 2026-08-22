"""Anomaly queue — never silently best-guess a bad row."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PATH = Path(__file__).resolve().parents[2] / "logs" / "acquisition" / "anomaly_queue.jsonl"

RESOLVE = "resolve"
QUARANTINE = "quarantine"
DESCRIPTIVE_ONLY = "descriptive_only"


def record(
    *,
    source: str,
    symbol: str | None,
    period: str | None,
    anomaly_type: str,
    severity: str,
    raw_evidence: Any,
    parser: str,
    suggested: str = QUARANTINE,
    extra: dict | None = None,
) -> None:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "symbol": symbol,
        "period": period,
        "anomaly_type": anomaly_type,
        "severity": severity,
        "raw_evidence": raw_evidence,
        "parser": parser,
        "suggested_disposition": suggested,
        **(extra or {}),
    }
    with PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def load(limit: int = 5000) -> list[dict[str, Any]]:
    if not PATH.exists():
        return []
    out = []
    for line in PATH.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out
