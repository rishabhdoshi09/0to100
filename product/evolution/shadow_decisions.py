"""Frozen shadow decisions: what a non-Champion policy WOULD have done.

Safety boundary (see product/evolution/policy_eval.py's docstring): this
module contains no broker call, no book mutation, no Telegram send. A shadow
decision is a RECORD, never an order. It is frozen via product.decision_freeze
BEFORE any future outcome exists -- decision_id = f"{policy_id}:{snapshot_id}"
makes re-evaluating the same (policy, snapshot) pair idempotent (restart-safe,
no duplicate rows) while a genuine content change under that same id raises
DecisionIdentityCollision instead of silently overwriting a frozen call.

The freeze gives an immutable, collision-safe identity. Outcome grading
(product/evolution/grading.py) is attached afterward in a SEPARATE, mutable
ledger -- exactly product.counterfactual_learning's freeze-then-settle
pattern, reused here rather than re-invented.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from core.runtime_paths import logs_dir

SCHEMA_VERSION = 1


def shadow_freeze_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_EVOLUTION_SHADOW_FREEZE")
    if override:
        return Path(override)
    # Resolved fresh on every call -- never a frozen module-level constant --
    # so QT_RUNTIME_ROOT redirection actually takes effect.
    return logs_dir() / "product" / "evolution_shadow_decisions.freeze.db"


def ledger_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_EVOLUTION_SHADOW_LEDGER")
    if override:
        return Path(override)
    return logs_dir() / "product" / "evolution_shadow_decisions.jsonl"


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                out.append(row)
    except Exception:
        return []
    return out


def _write_ledger(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
    tmp.replace(path)


def freeze_shadow_decision(
    snapshot: Mapping[str, Any], verdict: Mapping[str, Any],
    *, path: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze one policy's verdict against one snapshot. Idempotent: calling
    this again with the same (policy_id, snapshot_id) and unchanged verdict
    content returns the existing row rather than creating a duplicate."""
    policy_id = str(verdict.get("policy_id") or "")
    snapshot_id = str(verdict.get("market_snapshot_id") or snapshot.get("market_snapshot_id") or "")
    if not policy_id or not snapshot_id:
        raise ValueError("freeze_shadow_decision requires policy_id and market_snapshot_id")
    decision_id = f"{policy_id}:{snapshot_id}"
    symbol = str(verdict.get("symbol") or snapshot.get("symbol") or "").upper()
    as_of = str(snapshot.get("as_of") or "")

    ctx = dict(snapshot.get("context") or {})
    freeze_input = {
        "decision_id": decision_id,
        "symbol": symbol,
        "as_of": as_of,
        "decision": verdict.get("decision"),
        "reason_code": verdict.get("reason_code"),
        "entry": verdict.get("entry"),
        "stop": verdict.get("stop"),
        "target": verdict.get("target"),
        "setup_label": verdict.get("setup_label") or ctx.get("setup_label"),
        "sector": verdict.get("sector") or ctx.get("sector"),
        "regime": ctx.get("regime"),
        "selection_score": verdict.get("adjusted_score"),
        "evidence_class": "EVOLUTION_SHADOW",
        "versions": snapshot.get("versions"),
    }

    from product.decision_freeze import freeze as freeze_canonical

    frozen = freeze_canonical(freeze_input, path=shadow_freeze_path(path))
    shadow_id = str(frozen["freeze_id"])

    target = ledger_path(path)
    existing_rows = _read_ledger(target)
    for row in existing_rows:
        if row.get("shadow_id") == shadow_id:
            return row

    row = {
        "schema_version": SCHEMA_VERSION,
        "shadow_id": shadow_id,
        "policy_id": policy_id,
        "market_snapshot_id": snapshot_id,
        "domain": verdict.get("domain") or snapshot.get("domain"),
        "symbol": symbol,
        "as_of": as_of,
        "decision": verdict.get("decision"),
        "reason_code": verdict.get("reason_code"),
        "adjusted_score": verdict.get("adjusted_score"),
        "breakdown": verdict.get("breakdown"),
        "entry": verdict.get("entry"),
        "stop": verdict.get("stop"),
        "target": verdict.get("target"),
        "sector": verdict.get("sector"),
        "setup_label": verdict.get("setup_label"),
        "regime": ctx.get("regime"),
        "frozen_at": frozen["frozen_at"],
        "fingerprint": frozen.get("fingerprint"),
        "outcome": None,
        "classification": None,
        "graded_at": None,
        "not_pnl": True,
        "is_champion_decision": bool(verdict.get("is_champion_decision", False)),
        "grading_mode": str(
            verdict.get("grading_mode") or snapshot.get("grading_mode") or "UNDERLYING_FORWARD"
        ),
        "direction": verdict.get("direction"),
        "selected_contract": verdict.get("selected_contract"),
        "contract_symbol": verdict.get("contract_symbol"),
        "contract_context_key": verdict.get("contract_context_key"),
        "holding_days": verdict.get("holding_days"),
        "exit_policy": verdict.get("exit_policy"),
        "evidence_class": verdict.get("evidence_class") or "EVOLUTION_SHADOW",
    }
    existing_rows.append(row)
    _write_ledger(target, existing_rows)
    return row


def get_shadow_decision(shadow_id: str, *, path: str | Path | None = None) -> dict[str, Any] | None:
    for row in _read_ledger(ledger_path(path)):
        if row.get("shadow_id") == shadow_id:
            return row
    return None


def list_shadow_decisions(
    *, policy_id: str | None = None, market_snapshot_id: str | None = None,
    symbol: str | None = None, ungraded_only: bool = False,
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    rows = _read_ledger(ledger_path(path))
    if policy_id is not None:
        rows = [r for r in rows if r.get("policy_id") == policy_id]
    if market_snapshot_id is not None:
        rows = [r for r in rows if r.get("market_snapshot_id") == market_snapshot_id]
    if symbol is not None:
        rows = [r for r in rows if r.get("symbol") == symbol.upper()]
    if ungraded_only:
        rows = [r for r in rows if r.get("outcome") is None]
    return rows


def save_graded_decision(updated_row: Mapping[str, Any], *, path: str | Path | None = None) -> dict[str, Any]:
    """Persist a graded (outcome/classification filled in) shadow decision.
    Only the mutable grading fields change; shadow_id/decision/entry/stop/
    target are never rewritten here (they are immutable via the freeze)."""
    target = ledger_path(path)
    rows = _read_ledger(target)
    shadow_id = updated_row.get("shadow_id")
    found = False
    for i, row in enumerate(rows):
        if row.get("shadow_id") == shadow_id:
            rows[i] = dict(updated_row)
            found = True
            break
    if not found:
        rows.append(dict(updated_row))
    _write_ledger(target, rows)
    return dict(updated_row)
