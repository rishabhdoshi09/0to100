"""Canonical Forward Evidence Ledger.

One append-only store for every real-market (or test) candidate the paper
path already decided. Historical decision fields are never overwritten by
later prices. Outcomes live in settlement fields only.

Provenance is mandatory so test fixtures cannot pollute promotion stats.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from core.runtime_paths import _require_existing_runtime, logs_dir

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = logs_dir() / "product" / "forward_evidence.jsonl"
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


def _read_jsonl(path: Path, *, strict: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    # Do not build a second full-size raw UTF-8 string AND a splitlines list
    # alongside parsed dicts. Stream one record at a time on low-memory Macs.
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception as exc:
                if strict:
                    # Writers must not rewrite a filtered ledger and silently
                    # erase earlier malformed point-in-time decisions.
                    raise RuntimeError(
                        f"forward evidence ledger has invalid JSON at line {number}"
                    ) from exc
                continue
            if isinstance(payload, dict):
                rows.append(payload)
            elif strict:
                raise RuntimeError(
                    f"forward evidence ledger has a non-object row at line {number}"
                )
    return rows


def _require_safe_ledger_destination(path: Path) -> None:
    """Never recreate a vanished installed runtime or escape its logs tree."""
    if _require_existing_runtime():
        expected_logs = logs_dir()
        try:
            path.resolve(strict=False).relative_to(expected_logs)
        except ValueError as exc:
            raise RuntimeError(
                "forward ledger destination is outside the installed persistent runtime"
            ) from exc


@contextmanager
def _ledger_write_lock(path: Path):
    """Serialize the WHOLE read-modify-write across Mac worker processes.

    The lock is on a stable sibling file, not the ledger inode (which atomic
    rename replaces). Every frozen decision and settlement observes the most
    recent authoritative JSONL under this lock.
    """
    _require_safe_ledger_destination(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    _require_safe_ledger_destination(lock)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(str(lock), flags, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Publish one complete JSONL snapshot, never an intermediate truncation."""
    _require_safe_ledger_destination(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent),
    )
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in rows:
                handle.write(json.dumps(dict(record), default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


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
    """Freeze one immutable PIT decision, serializing its ledger mutation."""
    target = ledger_path(path)
    with _ledger_write_lock(target):
        rows = _read_jsonl(target, strict=True)
        item, created = _freeze_observation_in_rows(
            row, cycle_id=cycle_id, as_of=as_of, rules_hash=rules_hash,
            group=group, entered=entered, surfaced=surfaced,
            provenance=provenance, existing=rows,
        )
        if created:
            _write_jsonl(target, rows)
        return item


def _freeze_observation_in_rows(
    row: Mapping[str, Any],
    *,
    cycle_id: str,
    as_of: str,
    rules_hash: str = "",
    group: str = "",
    entered: bool = False,
    surfaced: bool = True,
    provenance: str | None = None,
    existing: list[dict[str, Any]],
    existing_by_id: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    """Freeze a PIT decision. Existing decision_id is left untouched."""
    symbol = str(row.get("symbol") or "").strip().upper()
    if not symbol:
        return None, False
    try:
        from product.decision_taxonomy import is_non_judgment
        if is_non_judgment(row.get("decision") or row.get("selection_result"), row.get("reason_code")):
            return None, False
    except Exception:
        if str(row.get("decision") or "").upper() == "NO_JUDGMENT":
            return None, False
    reason = str(row.get("reason_code") or row.get("decision") or "")
    did = str(row.get("decision_id") or "") or decision_id(
        as_of=as_of, symbol=symbol, group=group, reason_code=reason, cycle_id=cycle_id,
    )
    # A cycle must not linearly scan its entire historical evidence ledger
    # once for every new candidate. Keep the FIRST record for duplicate ids,
    # preserving existing immutable PIT decision semantics.
    if existing_by_id is None:
        existing_by_id = {}
        for previous in existing:
            existing_by_id.setdefault(str(previous.get("decision_id")), previous)
    prev = existing_by_id.get(did)
    if prev is not None:
        return dict(prev), False
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
        "freeze_id": str(row.get("freeze_id") or ""),
        "decision_fingerprint": str(row.get("evidence_fingerprint") or row.get("decision_fingerprint") or ""),
        "thesis_hash": str(row.get("thesis_hash") or ""),
        "calibration_snapshot_id": str(row.get("calibration_snapshot_id") or ""),
        "data_snapshot_id": str(row.get("data_snapshot_id") or row.get("source_snapshot_id") or ""),
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
            "freeze_id": str(row.get("freeze_id") or ""),
            "decision_fingerprint": str(row.get("evidence_fingerprint") or row.get("decision_fingerprint") or ""),
            "decision_frozen": True,
            "future_data_used_for_decision": False,
        },
        "not_pnl": not entered,
    }
    existing.append(payload)
    existing_by_id[did] = payload
    return payload, True

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
    with _ledger_write_lock(target):
        rows = _read_jsonl(target, strict=True)
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
    """Freeze a full decision cycle with ONE locked ledger read and publication.

    Avoid N read/serialize/fsync cycles for a whole-market candidate list.
    Separate worker processes still interlock with standalone settlements.
    """
    cycle = dict(cycle or {})
    as_of = str(cycle.get("as_of") or "")[:10]
    rules_hash = str(cycle.get("rules_hash") or "")
    cycle_id = str(cycle.get("cycle_id") or f"{as_of}:{rules_hash}:{cycle.get('recorded_at') or _now()}")
    written = 0
    skipped = 0
    target = ledger_path(path)
    with _ledger_write_lock(target):
        rows = _read_jsonl(target, strict=True)
        # One O(historical decisions) index; duplicate checks for each card
        # are now O(1) rather than walking a growing JSONL row list.
        by_id: dict[str, dict[str, Any]] = {}
        for existing in rows:
            by_id.setdefault(str(existing.get("decision_id")), existing)
        for group, entered, surfaced in (
            ("taken", True, True),
            ("rejections", False, True),
            ("waits", False, True),
            ("not_surfaced", False, False),
        ):
            for row in list(cycle.get(group) or []):
                if not isinstance(row, Mapping):
                    continue
                _record, created = _freeze_observation_in_rows(
                    row, cycle_id=cycle_id, as_of=as_of, rules_hash=rules_hash,
                    group=group.upper() if group != "rejections" else "REJECTED",
                    entered=entered, surfaced=surfaced, existing=rows,
                    existing_by_id=by_id,
                )
                if created:
                    written += 1
                else:
                    skipped += 1
        if written:
            _write_jsonl(target, rows)
    return {"cycle_id": cycle_id, "written": written, "skipped_duplicates": skipped}
