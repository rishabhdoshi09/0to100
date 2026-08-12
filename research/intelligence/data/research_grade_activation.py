"""Activation helpers: build operational snapshot + run earned RESEARCH_GRADE gate.

Never stamps RESEARCH_GRADE unless ``evaluate_research_grade`` returns earned=True.
Production trading paths are not modified.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from data.nse_ca_ingest import ADJUSTMENT_POLICY_VERSION
from research.intelligence.data.from_bhav import snapshot_from_bhav_store
from research.intelligence.data.research_grade_gate import (
    evaluate_research_grade,
    stamp_manifest_if_earned,
)
from research.intelligence.data.snapshot_store import SnapshotStore


def build_research_snapshot(
    *,
    root: str | Path = "logs/snapshots",
    max_symbols: int | None = None,
) -> dict:
    store = SnapshotStore(Path(root))
    gate = evaluate_research_grade(run_gauntlet_validate=True, sample=100)
    extra = stamp_manifest_if_earned(
        {
            "source": "nse_bhav+index",
            "adjustment_policy_version": ADJUSTMENT_POLICY_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        gate,
    )
    # Refuse to mint a RESEARCH_GRADE-labelled snapshot when the gate failed.
    # Still allow an OPERATIONAL_ONLY provenance snapshot for debugging.
    from research.intelligence.data.from_bhav import rows_from_bhav_store

    rows, row_report = rows_from_bhav_store(max_symbols=max_symbols)
    if not rows:
        return {"result": "no rows", "gate": gate, "row_report": row_report}
    # Attach index rows when available
    index_rows = []
    try:
        from data import index_store as IX
        IX.build_index_store(days=0)
        with IX._lock:
            for name, df in IX._store.items():
                for ts, bar in df.iterrows():
                    try:
                        d = getattr(ts, "date", lambda: ts)()
                        iso = d.isoformat() if hasattr(d, "isoformat") else str(d)[:10]
                        index_rows.append((
                            name,
                            iso,
                            float(bar.get("open", bar.get("close"))),
                            float(bar.get("high", bar.get("close"))),
                            float(bar.get("low", bar.get("close"))),
                            float(bar.get("close")),
                        ))
                    except Exception:
                        continue
    except Exception:
        index_rows = []

    sid = store.commit_snapshot(
        rows,
        index_rows=index_rows,
        extra_manifest=extra,
    )
    final = stamp_manifest_if_earned({**extra, "snapshot_id": sid}, gate)
    return {
        "result": "committed",
        "snapshot_id": sid,
        "manifest": final,
        "row_report": row_report,
        "gate": {
            "earned": gate["earned"],
            "trust_class": final.get("trust_class"),
            "failed": gate.get("failed"),
            "user_facing": gate.get("user_facing"),
            "checks": gate.get("checks"),
        },
    }

def write_gate_report(path: str | Path, gate: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(gate, indent=2, default=str), encoding="utf-8")
    return p
