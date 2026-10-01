"""Immutable point-in-time market/decision snapshots for the tournament.

Reuses product.decision_context.snapshot() for the evidence pack and
product.decision_freeze.freeze() for the collision-safe immutable ID --
this module invents no new capture mechanism, only a dedicated identity
space (its own freeze DB) so tournament snapshot IDs never collide with
the canonical production decision-freeze ledger, plus a small full-payload
store because decision_freeze only persists the fields its own identity
material needs (not every raw feature a policy variant might read, e.g.
rs_percentile/volume_ratio).

A snapshot_id is content-addressed: calling build_snapshot() twice with
identical inputs returns the SAME id (decision_freeze.freeze() hashes the
material and only mints a new id when the content differs; a genuine
content change under the same symbol+as_of correctly raises
DecisionIdentityCollision rather than silently overwriting), which is what
makes "the exact snapshot a policy saw" reproducible and restart-safe.
"""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from core.runtime_paths import logs_dir
from product import decision_context

def snapshot_freeze_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_EVOLUTION_SNAPSHOTS")
    if override:
        return Path(override)
    # Resolved fresh on every call -- never a frozen module-level constant --
    # so QT_RUNTIME_ROOT redirection actually takes effect.
    return logs_dir() / "product" / "evolution_snapshots.freeze.db"


def _payload_path(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path).with_suffix(".payload.db")
    override = os.environ.get("QT_EVOLUTION_SNAPSHOTS_PAYLOAD")
    if override:
        return Path(override)
    return logs_dir() / "product" / "evolution_snapshots_payload.db"


def _payload_connect(path: str | Path | None) -> sqlite3.Connection:
    from product.sqlite_runtime import connect

    target = _payload_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    con = connect(target)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS snapshot_payloads (
            market_snapshot_id TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL
        )
        """
    )
    return con


def build_snapshot(
    card: Mapping[str, Any],
    *,
    book=None,
    regime: str = "",
    domain: str = "EQUITY",
    as_of: str = "",
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze exactly what decision_context.snapshot() captured for this
    candidate at this moment. Never captures future bars -- it only ever
    reads what the caller's `card`/`book`/`regime` already represent, which
    are themselves point-in-time by construction upstream (the scanner card
    and the paper book state as of right now).
    """
    ctx = decision_context.snapshot(card, book=book, regime=regime)
    symbol = str(card.get("symbol") or "").upper()
    cutoff = str(
        as_of or card.get("as_of") or card.get("scan_scanned_at")
        or card.get("generated_at") or ""
    )[:19]

    from product.decision_freeze import freeze as freeze_canonical

    freeze_input = {
        # decision_id intentionally omitted: content-addressed by
        # (symbol, as_of, material) so the SAME snapshot content always
        # resolves to the SAME id, and different content never collides
        # onto the same id.
        "symbol": symbol,
        "as_of": cutoff,
        "decision": "SNAPSHOT",
        "reason_code": f"EVOLUTION_SNAPSHOT:{domain}",
        "setup_label": ctx.get("setup_label"),
        "sector": ctx.get("sector"),
        "regime": ctx.get("regime"),
        "entry": ctx.get("entry"),
        "stop": ctx.get("stop"),
        "target": ctx.get("target"),
        "entry_quality": ctx.get("entry_quality"),
        "chase_risk": ctx.get("chase_risk"),
        "extension_pct": ctx.get("extension_pct"),
        "family_confirms": ctx.get("family_confirms"),
        "missing_evidence": ctx.get("missing_evidence"),
        "empirical": ctx.get("empirical"),
        "dd_status": ctx.get("dd_status"),
        "portfolio": ctx.get("portfolio"),
        "methods": ctx.get("methods"),
        "source_scan_id": card.get("source_scan_id") or card.get("scan_scanned_at") or "",
        "data_snapshot_id": card.get("data_snapshot_id") or "",
        "evidence_class": f"EVOLUTION_{domain}_SNAPSHOT",
    }
    frozen = freeze_canonical(freeze_input, path=snapshot_freeze_path(path))
    snapshot_id = str(frozen["freeze_id"])

    record = {
        "market_snapshot_id": snapshot_id,
        "domain": domain,
        "symbol": symbol,
        "as_of": cutoff,
        "context": ctx,
        "card": dict(card),
        "frozen_at": frozen["frozen_at"],
        "versions": frozen.get("versions") or {},
        "fingerprint": frozen.get("fingerprint"),
    }

    con = _payload_connect(path)
    existing = con.execute(
        "SELECT 1 FROM snapshot_payloads WHERE market_snapshot_id=?", (snapshot_id,),
    ).fetchone()
    if existing is None:
        con.execute(
            "INSERT INTO snapshot_payloads (market_snapshot_id, payload_json) VALUES (?,?)",
            (snapshot_id, json.dumps(record, default=str)),
        )
        con.commit()
    con.close()
    return record


def get_snapshot_record(snapshot_id: str, *, path: str | Path | None = None) -> dict[str, Any] | None:
    """The FULL snapshot a policy saw -- not just the identity fields
    decision_freeze itself persists. Restart-safe: pure read from durable
    storage, no in-memory cache to go stale or need invalidating."""
    con = _payload_connect(path)
    row = con.execute(
        "SELECT payload_json FROM snapshot_payloads WHERE market_snapshot_id=?",
        (snapshot_id,),
    ).fetchone()
    con.close()
    return json.loads(row["payload_json"]) if row else None
