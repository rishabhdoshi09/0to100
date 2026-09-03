"""Learning ledger — conclusions, not silent policy mutation.

States: OBSERVED → HYPOTHESIS → TESTING → SUPPORTED / REJECTED
        → PROMOTION_PENDING → PROMOTED

Level 1 OBSERVE is the default. Level 2 RECOMMEND writes a recommendation
without changing production. Level 3 PROMOTE is never automatic here.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "logs" / "product" / "learning_ledger.db"

OBSERVED = "OBSERVED"
HYPOTHESIS = "HYPOTHESIS"
TESTING = "TESTING"
SUPPORTED = "SUPPORTED"
REJECTED = "REJECTED"
PROMOTION_PENDING = "PROMOTION_PENDING"
PROMOTED = "PROMOTED"

LEVEL_OBSERVE = 1
LEVEL_RECOMMEND = 2
LEVEL_PROMOTE = 3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(path: Path | None = None) -> sqlite3.Connection:
    target = path or DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(target))
    con.row_factory = sqlite3.Row
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS learnings (
            learning_id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            triggering_json TEXT,
            dataset TEXT,
            provenance TEXT,
            experiment TEXT,
            sample_n INTEGER,
            result TEXT,
            confidence TEXT,
            recommended_change TEXT,
            promotion_state TEXT,
            production_impact TEXT,
            level INTEGER,
            day TEXT,
            created_at TEXT,
            payload_json TEXT
        )
        """
    )
    return con


def record(
    *,
    learning_id: str,
    question: str,
    triggering: Mapping[str, Any] | None = None,
    dataset: str = "",
    provenance: str = "",
    experiment: str = "",
    sample_n: int = 0,
    result: str = "",
    confidence: str = "",
    recommended_change: str = "",
    promotion_state: str = OBSERVED,
    production_impact: str = "none",
    level: int = LEVEL_OBSERVE,
    day: str = "",
    path: Path | None = None,
) -> dict[str, Any]:
    if level >= LEVEL_PROMOTE:
        raise ValueError("learning_ledger refuses automatic promotion")
    row = {
        "learning_id": learning_id,
        "question": question,
        "triggering_json": json.dumps(dict(triggering or {}), default=str),
        "dataset": dataset,
        "provenance": provenance,
        "experiment": experiment,
        "sample_n": int(sample_n),
        "result": result,
        "confidence": confidence,
        "recommended_change": recommended_change if level >= LEVEL_RECOMMEND else "",
        "promotion_state": promotion_state,
        "production_impact": production_impact,
        "level": int(level),
        "day": day or _now()[:10],
        "created_at": _now(),
        "payload_json": "{}",
    }
    con = _connect(path)
    con.execute(
        """INSERT OR REPLACE INTO learnings (
            learning_id, question, triggering_json, dataset, provenance, experiment,
            sample_n, result, confidence, recommended_change, promotion_state,
            production_impact, level, day, created_at, payload_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        tuple(row[k] for k in (
            "learning_id", "question", "triggering_json", "dataset", "provenance",
            "experiment", "sample_n", "result", "confidence", "recommended_change",
            "promotion_state", "production_impact", "level", "day", "created_at",
            "payload_json",
        )),
    )
    con.commit()
    con.close()
    return row


def learned_today(day: str = "", *, path: Path | None = None) -> dict[str, Any]:
    """Honest answer. Nothing statistically meaningful is an acceptable result."""
    target_day = (day or _now()[:10])[:10]
    con = _connect(path)
    rows = [dict(r) for r in con.execute(
        "SELECT * FROM learnings WHERE day=? ORDER BY created_at",
        (target_day,),
    )]
    con.close()
    meaningful = [
        r for r in rows
        if int(r.get("sample_n") or 0) >= 20 and str(r.get("promotion_state") or "") in {SUPPORTED, REJECTED}
    ]
    if not rows:
        summary = "Nothing statistically meaningful."
    elif not meaningful:
        bits = []
        for r in rows:
            bits.append(
                f"{r.get('question')} n={r.get('sample_n') or 0} → {r.get('result') or r.get('promotion_state')}. "
                "No policy change."
            )
        summary = " ".join(bits) if bits else "Nothing statistically meaningful."
    else:
        summary = " ".join(
            f"{r.get('question')} n={r.get('sample_n')} {r.get('promotion_state')}."
            for r in meaningful
        )
    return {
        "day": target_day,
        "entries": rows,
        "statistically_meaningful": meaningful,
        "policy_changed": False,
        "production_impact": "none",
        "summary": summary,
        "learning_level": LEVEL_OBSERVE,
    }
