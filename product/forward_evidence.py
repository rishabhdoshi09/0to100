"""Canonical Forward Evidence Ledger.

One append-only store for every real-market (or test) candidate the paper
path already decided. Historical decision fields are never overwritten by
later prices. Outcomes live in settlement fields only.

Provenance is mandatory so test fixtures cannot pollute promotion stats.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "logs" / "product" / "forward_evidence.jsonl"
SCHEMA_VERSION = 1  # previous: none. Additive research ledger.

REAL_FORWARD_MARKET = "REAL_FORWARD_MARKET"
TEST_FIXTURE = "TEST_FIXTURE"
BACKTEST = "BACKTEST"
WALK_FORWARD = "WALK_FORWARD"
COUNTERFACTUAL = "COUNTERFACTUAL"

PROVENANCES = {
    REAL_FORWARD_MARKET,
    TEST_FIXTURE,
    BACKTEST,
    WALK_FORWARD,
    COUNTERFACTUAL,
}


def ledger_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_FORWARD_LEDGER")
    if override:
        return Path(override)
    return DEFAULT_PATH


def current_provenance(explicit: str | None = None) -> str:
    if explicit and str(explicit) in PROVENANCES:
        return str(explicit)
    override = os.environ.get("QT_EVIDENCE_PROVENANCE")
    if override and override in PROVENANCES:
        return override
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return TEST_FIXTURE
    return REAL_FORWARD_MARKET


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(dict(r), default=str) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, path)


def load_ledger(path: str | Path | None = None) -> list[dict[str, Any]]:
    return _read_jsonl(ledger_path(path))


def real_forward_only(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Promotion statistics must use this filter."""
    return [dict(r) for r in rows if str(r.get("provenance") or "") == REAL_FORWARD_MARKET]


def decision_id(
    *,
    as_of: str,
    symbol: str,
    group: str,
    reason_code: str,
    cycle_id: str,
) -> str:
    return "|".join([
        str(as_of or "")[:10],
        str(symbol or "").upper(),
        str(group or ""),
        str(reason_code or ""),
        str(cycle_id or ""),
    ])


def freeze_observation(
    row: Mapping[str, Any],
    *,
    cycle_id: str,
    as_of: str,
    rules_hash: str = "",
    group: str = "",
    entered: bool = False,
    surfaced: bool = True,
    provenance: str | None = None,
    path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Freeze a PIT decision. Existing decision_id is left untouched."""
    symbol = str(row.get("symbol") or "").strip().upper()
    if not symbol:
        return None
    reason = str(row.get("reason_code") or row.get("decision") or "")
    did = str(row.get("decision_id") or "") or decision_id(
        as_of=as_of, symbol=symbol, group=group, reason_code=reason, cycle_id=cycle_id,
    )
    target = ledger_path(path)
    existing = load_ledger(target)
    for prev in existing:
        if str(prev.get("decision_id")) == did:
            return dict(prev)
    portfolio = row.get("portfolio_authority") if isinstance(row.get("portfolio_authority"), Mapping) else {}
    policy = str(row.get("policy_effect") or (row.get("policy") or {}).get("final_effect") or "NEUTRAL")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "decision_id": did,
        "cycle_id": str(cycle_id or ""),
        "provenance": current_provenance(provenance),
        "symbol": symbol,
        "decision_timestamp": _now(),
        "market_timestamp": str(as_of or "")[:32],
        "rules_hash": str(rules_hash or row.get("rules_hash") or ""),
        "policy_version": str(row.get("policy_version") or ""),
        "recommendation_tier": str(row.get("tier") or row.get("reco_tier") or ""),
        "selection_result": str(row.get("decision") or group or ""),
        "policy_result": policy,
        "portfolio_result": str((portfolio or {}).get("decision") or row.get("portfolio_result") or ""),
        "entry_state": str(row.get("entry_state") or ""),
        "dd_status": str(row.get("dd_status") or row.get("dd_verdict") or ""),
        "setup": str(row.get("setup_label") or row.get("setup") or row.get("primary_thesis") or ""),
        "sector": str(row.get("sector") or ""),
        "regime": str(row.get("regime") or ""),
        "entry": row.get("entry") or row.get("entry_fill"),
        "stop": row.get("stop"),
        "target": row.get("target"),
        "surfaced": bool(surfaced),
        "entered": bool(entered),
        "why": row.get("why") or row.get("detail") or "",
        "reason_code": reason,
        "group": str(group or row.get("group") or ""),
        "later_outcome": None,
        "gross_R": None,
        "execution_adjusted_R": None,
        "execution_coverage": None,
        "execution_charges": None,
        "spread_status": None,
        "slippage_status": None,
        "liquidity_assumptions": None,
        "counterfactual_classification": None,
        "pit_proof": {
            "as_of": str(as_of or "")[:10],
            "rules_hash": str(rules_hash or row.get("rules_hash") or ""),
            "scan_scanned_at": str(row.get("scan_scanned_at") or ""),
            "decision_frozen": True,
            "future_data_used_for_decision": False,
        },
        "not_pnl": not entered,
    }
    existing.append(payload)
    _write_jsonl(target, existing)
    return payload


def attach_settlement(
    decision_id_value: str,
    *,
    classification: str | None = None,
    forward_return_pct: float | None = None,
    gross_R: float | None = None,
    execution_adjusted_R: float | None = None,
    execution_coverage: float | None = None,
    execution_charges: float | None = None,
    spread_status: str | None = None,
    slippage_status: str | None = None,
    liquidity_assumptions: str | None = None,
    outcome_provenance: str | None = None,
    path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Write later outcome only. Never mutates PIT decision fields."""
    target = ledger_path(path)
    rows = load_ledger(target)
    found = None
    out: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("decision_id")) != str(decision_id_value):
            out.append(row)
            continue
        updated = dict(row)
        if updated.get("later_outcome") is None:
            updated["later_outcome"] = {
                "forward_return_pct": forward_return_pct,
                "settled_at": _now(),
                "provenance": current_provenance(outcome_provenance),
                "not_pnl": not bool(updated.get("entered")),
            }
            if classification and updated.get("counterfactual_classification") is None:
                updated["counterfactual_classification"] = classification
            if gross_R is not None and updated.get("gross_R") is None:
                updated["gross_R"] = gross_R
            if execution_adjusted_R is not None and updated.get("execution_adjusted_R") is None:
                updated["execution_adjusted_R"] = execution_adjusted_R
            if execution_coverage is not None and updated.get("execution_coverage") is None:
                updated["execution_coverage"] = execution_coverage
            if execution_charges is not None and updated.get("execution_charges") is None:
                updated["execution_charges"] = execution_charges
            if spread_status and updated.get("spread_status") is None:
                updated["spread_status"] = spread_status
            if slippage_status and updated.get("slippage_status") is None:
                updated["slippage_status"] = slippage_status
            if liquidity_assumptions and updated.get("liquidity_assumptions") is None:
                updated["liquidity_assumptions"] = liquidity_assumptions
        found = updated
        out.append(updated)
    if found is None:
        return None
    _write_jsonl(target, out)
    return found


def freeze_cycle(cycle: Mapping[str, Any], *, path: str | Path | None = None) -> dict[str, Any]:
    cycle = dict(cycle or {})
    as_of = str(cycle.get("as_of") or "")[:10]
    rules_hash = str(cycle.get("rules_hash") or "")
    cycle_id = str(cycle.get("cycle_id") or f"{as_of}:{rules_hash}:{cycle.get('recorded_at') or _now()}")
    written = 0
    skipped = 0
    for group, entered, surfaced in (
        ("taken", True, True),
        ("rejections", False, True),
        ("waits", False, True),
        ("not_surfaced", False, False),
    ):
        for row in list(cycle.get(group) or []):
            if not isinstance(row, Mapping):
                continue
            before = len(load_ledger(path))
            freeze_observation(
                row,
                cycle_id=cycle_id,
                as_of=as_of,
                rules_hash=rules_hash,
                group=group.upper() if group != "rejections" else "REJECTED",
                entered=entered,
                surfaced=surfaced,
                path=path,
            )
            after = len(load_ledger(path))
            if after > before:
                written += 1
            else:
                skipped += 1
    return {"cycle_id": cycle_id, "written": written, "skipped_duplicates": skipped}
