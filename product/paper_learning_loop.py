"""Paper-forward learning loop: taken exits and counterfactuals update policies.

This is not a second scanner. It only records what the selection authority already
decided, then feeds measured evidence into the explicit Policy Layer so the *next*
recommendation cycle can SUPPORT / PENALIZE / BLOCK. It cannot invent a BUY and
cannot enable live money.

Hard product gates (stop, duplicate, capital, DD fail, chase extension) stay hard.
Learning may only add evidence overlays on top of those gates.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from product.counterfactual_learning import (
    AVOIDED_LOSER,
    CORRECT_REJECTION,
    FLAT,
    GOOD_WAIT,
    MISSED_WINNER,
    RAN_AWAY,
    ledger_path,
    settle,
)
from product.evidence_policy_engine import HARD_REASON_CODES
from product.learning_policy_store import record_measured_outcome

ROOT = Path(__file__).resolve().parents[1]
TAKEN_PATH = ROOT / "logs" / "product" / "taken_evidence.jsonl"
INGESTED_PATH = ROOT / "logs" / "product" / "learning_ingested.json"


def _env_path(name: str, default: Path) -> Path:
    override = os.environ.get(name)
    return Path(override) if override else default


def taken_path() -> Path:
    return _env_path("QT_TAKEN_EVIDENCE", TAKEN_PATH)


def ingested_path() -> Path:
    return _env_path("QT_LEARNING_INGESTED", INGESTED_PATH)


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(row), default=str) + "\n")


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


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def record_taken_evidence(taken: Sequence[Mapping[str, Any]], *, as_of: str) -> None:
    """Freeze the point-in-time evidence for every paper fill."""
    for row in taken:
        _append_jsonl(taken_path(), {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "as_of": as_of,
            "symbol": str(row.get("symbol") or "").upper(),
            "setup_label": row.get("setup_label") or row.get("primary_thesis") or "",
            "sector": row.get("sector") or "",
            "tier": row.get("tier") or row.get("reco_tier") or "",
            "entry_state": row.get("entry_state") or "",
            "policy_effect": row.get("policy_effect") or "NEUTRAL",
            "selection_score": row.get("selection_score"),
            "entry": row.get("entry"),
            "entry_fill": row.get("entry_fill"),
            "qty": row.get("qty"),
            "stop": row.get("stop"),
            "target": row.get("target"),
            "reason_code": row.get("reason_code") or "ELIGIBLE",
            "regime": row.get("regime") or "",
            "dd_status": row.get("dd_status") or "",
            "entry_quality": row.get("entry_quality") or "",
            "execution_reality_shadow": row.get("execution_reality_shadow"),
            "why": row.get("why") or {},
        })


def _trade_key(trade: Mapping[str, Any]) -> str:
    return "|".join([
        str(trade.get("symbol") or "").upper(),
        str(trade.get("entry_date") or ""),
        str(trade.get("exit_date") or ""),
        str(trade.get("exit_reason") or ""),
    ])


def _lookup_taken(symbol: str) -> dict[str, Any]:
    symbol = symbol.upper()
    rows = _read_jsonl(taken_path())
    for row in reversed(rows):
        if str(row.get("symbol") or "").upper() == symbol:
            return row
    return {}


def ingest_closed_trade(
    trade: Mapping[str, Any] | Any,
    *,
    path=None,
    floors: Mapping[str, int] | None = None,
) -> dict[str, Any] | None:
    """One settled paper trade updates setup (+ limited conditionals). Not a new BUY.

    Gross R is always preserved. When execution-adjusted evidence is available,
    the Policy Layer consumes the more conservative of gross and adjusted R.
    Paper fills themselves are never repriced here.
    """
    row = dict(trade.as_dict()) if hasattr(trade, "as_dict") else dict(trade)
    symbol = str(row.get("symbol") or "").upper()
    if not symbol:
        return None
    evidence = _lookup_taken(symbol)
    setup = str(evidence.get("setup_label") or row.get("setup_label") or "UNKNOWN_SETUP")
    sector = str(evidence.get("sector") or row.get("sector") or "")
    regime = str(evidence.get("regime") or row.get("regime") or "")
    entry_state = str(evidence.get("entry_state") or row.get("entry_state") or "")

    try:
        from product.evidence_integrity import settled_learning_result
        integrity = settled_learning_result(row, evidence)
    except Exception:
        integrity = {
            "gross_realized_R": float(row.get("realized_R") or 0.0),
            "execution_adjusted_R": None,
            "policy_realized_R": float(row.get("realized_R") or 0.0),
            "execution_adjusted_available": False,
            "execution_complete": False,
            "execution_coverage": 0.0,
            "quality": "GROSS_ONLY",
            "paper_fill_unchanged": True,
        }
    realized = float(integrity.get("policy_realized_R") if integrity.get("policy_realized_R") is not None else (row.get("realized_R") or 0.0))
    source = (
        "paper_forward_taken_execution_adjusted"
        if integrity.get("execution_adjusted_available")
        else "paper_forward_taken_gross_only"
    )
    extra_base = {
        "symbol": symbol,
        "exit_reason": row.get("exit_reason") or "",
        "regime": regime,
        "not_live": True,
        "gross_realized_R": integrity.get("gross_realized_R"),
        "execution_adjusted_R": integrity.get("execution_adjusted_R"),
        "policy_realized_R": realized,
        "execution_adjusted_available": bool(integrity.get("execution_adjusted_available")),
        "execution_complete": bool(integrity.get("execution_complete")),
        "execution_coverage": float(integrity.get("execution_coverage") or 0.0),
        "evidence_quality": integrity.get("quality") or "GROSS_ONLY",
        "paper_fill_unchanged": True,
    }
    last = record_measured_outcome(
        policy_id=f"SETUP::{setup}",
        dimension="setup",
        bucket=setup,
        realized_R=realized,
        source=source,
        path=path,
        floors=floors,
        extra=extra_base,
    )
    # At most two 2-way conditionals — never a combinatorial explosion.
    if setup and sector:
        record_measured_outcome(
            policy_id=f"SETUP_SECTOR::{setup}|{sector}",
            dimension="setup|sector",
            bucket=f"{setup}|{sector}",
            realized_R=realized,
            source=source,
            path=path,
            floors=floors,
            extra=extra_base,
        )
    if setup and regime:
        record_measured_outcome(
            policy_id=f"SETUP_REGIME::{setup}|{regime}",
            dimension="setup|regime",
            bucket=f"{setup}|{regime}",
            realized_R=realized,
            source=source,
            path=path,
            floors=floors,
            extra=extra_base,
        )
    if entry_state:
        record_measured_outcome(
            policy_id=f"ENTRY::{entry_state}",
            dimension="entry_state",
            bucket=entry_state,
            realized_R=realized,
            source=source,
            path=path,
            floors=floors,
            extra=extra_base,
        )
    # Exit quality is evidence-only until an owner promotes an exit policy.
    exit_reason = str(row.get("exit_reason") or "")
    if exit_reason:
        record_measured_outcome(
            policy_id=f"EXIT::{exit_reason}",
            dimension="exit_reason",
            bucket=exit_reason,
            realized_R=realized,
            source="paper_forward_exit_execution_aware",
            path=path,
            floors=floors,
            extra={**extra_base, "affects_selection": False},
        )
    return last


def ingest_closed_book(book, *, path=None, floors: Mapping[str, int] | None = None) -> dict[str, Any]:
    """Idempotent: only new ClosedTrade rows update policies."""
    ingested = _load_json(ingested_path())
    seen = set(ingested.get("keys") or [])
    closed = list(getattr(book, "closed", []) or [])
    applied = 0
    last = None
    for trade in closed:
        row = trade.as_dict() if hasattr(trade, "as_dict") else dict(trade)
        key = _trade_key(row)
        if key in seen:
            continue
        last = ingest_closed_trade(row, path=path, floors=floors)
        seen.add(key)
        applied += 1
    target = ingested_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "keys": sorted(seen)[-2000:],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "applied": applied,
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    return {"applied": applied, "last_policy": last}


def ingest_counterfactual(
    settled: Mapping[str, Any],
    *,
    path=None,
    floors: Mapping[str, int] | None = None,
) -> dict[str, Any] | None:
    """Rejected/waited names update reason-level statistics. Never booked as P&L.

    Hard gates stay hard: SECTOR_CAP / chase / DD misses are observed, not reversed.
    A MISSED_WINNER on a *learned* setup block may weaken that setup overlay.
    """
    classification = str(settled.get("classification") or "")
    reason = str(settled.get("reason_code") or "REJECTED")
    mapped = {
        CORRECT_REJECTION: 0.40,
        AVOIDED_LOSER: 0.40,
        MISSED_WINNER: -0.40,
        GOOD_WAIT: 0.20,
        RAN_AWAY: 0.0,
        FLAT: 0.0,
    }.get(classification, 0.0)
    hard = reason in HARD_REASON_CODES or reason in {"SECTOR_CAP", "CORRELATION_CAP", "MAX_PORTFOLIO_RISK"}
    last = record_measured_outcome(
        policy_id=f"REJECT::{reason}",
        dimension="reason_code",
        bucket=reason,
        realized_R=mapped,
        source="counterfactual_not_pnl",
        path=path,
        floors=floors,
        extra={
            "classification": classification,
            "not_pnl": True,
            "symbol": settled.get("symbol"),
            "affects_selection": not hard,
            "regime": (settled.get("evidence") or {}).get("regime") or settled.get("regime") or "",
        },
    )
    evidence = dict(settled.get("evidence") or {})
    setup = str(evidence.get("setup_label") or settled.get("setup") or "")
    # Learned setup overlay can be weakened by missed winners; hard gates cannot.
    if classification == MISSED_WINNER and setup and reason == "EVIDENCE_POLICY_BLOCK":
        last = record_measured_outcome(
            policy_id=f"SETUP::{setup}",
            dimension="setup",
            bucket=setup,
            realized_R=0.40,
            source="counterfactual_missed_winner",
            path=path,
            floors=floors,
            extra={"classification": classification, "not_pnl": True, "symbol": settled.get("symbol")},
        )
    if classification == GOOD_WAIT:
        last = record_measured_outcome(
            policy_id="ENTRY_QUALITY::GOOD_WAIT",
            dimension="entry_quality",
            bucket="good_wait",
            realized_R=0.20,
            source="counterfactual_good_wait",
            path=path,
            floors=floors,
            extra={"classification": GOOD_WAIT, "not_pnl": True, "symbol": settled.get("symbol")},
        )
    return last


def note_later_entry(symbol: str, *, path=None, floors: Mapping[str, int] | None = None) -> dict[str, Any]:
    """A waited name later received a valid fill → GOOD_WAIT. Not booked as extra P&L."""
    return settle_pending_counterfactuals(
        later_entered={str(symbol).upper(): True},
        path=path,
        floors=floors,
    )


def settle_pending_counterfactuals(
    *,
    forward_return_by_symbol: Mapping[str, float] | None = None,
    later_entered: Mapping[str, bool] | None = None,
    path=None,
    floors: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Classify frozen rejects once a forward return is known. Does not create P&L."""
    ledger = ledger_path()
    rows = _read_jsonl(ledger)
    returns = {str(k).upper(): float(v) for k, v in dict(forward_return_by_symbol or {}).items()}
    entered = {str(k).upper(): bool(v) for k, v in dict(later_entered or {}).items()}
    updated = 0
    classifications: dict[str, int] = {}
    rewritten: list[dict[str, Any]] = []
    for row in rows:
        if row.get("classification"):
            rewritten.append(row)
            continue
        symbol = str(row.get("symbol") or "").upper()
        if symbol not in returns and symbol not in entered:
            rewritten.append(row)
            continue
        settled = settle(
            row,
            forward_return_pct=returns.get(symbol),
            later_entered=entered.get(symbol, False),
        )
        ingest_counterfactual(settled, path=path, floors=floors)
        rewritten.append(settled)
        updated += 1
        cls = str(settled.get("classification") or FLAT)
        classifications[cls] = classifications.get(cls, 0) + 1
    if updated:
        tmp = ledger.with_suffix(ledger.suffix + ".tmp")
        tmp.write_text("".join(json.dumps(r, default=str) + "\n" for r in rewritten), encoding="utf-8")
        os.replace(tmp, ledger)
    return {"updated": updated, "classifications": classifications}


def learning_dashboard(
    *,
    policy_path=None,
) -> dict[str, Any]:
    """Operator view of explicit policies. No 'AI is learning' copy."""
    from product.learning_policy_store import load_policies
    from product.live_readiness import evaluate_live_readiness
    from product.autopilot_journal import load_journal

    store = load_policies(policy_path)
    policies = [dict(p) for p in (store.get("policies") or [])]
    by_status: dict[str, list[dict[str, Any]]] = {}
    for policy in policies:
        status = str(policy.get("production_status") or "OBSERVING")
        by_status.setdefault(status, []).append(policy)
    cf_rows = _read_jsonl(ledger_path())
    class_counts: dict[str, int] = {}
    for row in cf_rows:
        cls = str(row.get("classification") or "")
        if cls:
            class_counts[cls] = class_counts.get(cls, 0) + 1
    journal = load_journal()
    latest = dict(journal.get("latest") or {})
    taken_n = sum(len(list(c.get("taken") or [])) for c in (journal.get("cycles") or []))
    soak: dict[str, Any] | None
    try:
        from product.forward_soak import scoreboard
        soak = scoreboard()
    except Exception as exc:
        soak = {"error": str(exc)[:200], "live_locked": True}
    return {
        "schema_version": 1,
        "live_locked": True,
        "note": (
            "Policies are versioned evidence overlays. They never invent a BUY. "
            "Hard risk/DD/entry gates stay authoritative."
        ),
        "policies": policies,
        "active": by_status.get("ACTIVE", []),
        "eligible": by_status.get("ELIGIBLE", []),
        "observing": by_status.get("OBSERVING", []) + by_status.get("EXPERIMENTAL", []),
        "rejected_hypotheses": by_status.get("REJECTED", []) + by_status.get("DEMOTED", []),
        "counterfactuals": {
            "frozen": len(cf_rows),
            "classified": sum(class_counts.values()),
            "counts": class_counts,
        },
        "recent_learning": {
            "taken_fills": taken_n,
            "latest_as_of": latest.get("as_of") or "",
            "latest_taken": len(latest.get("taken") or []),
            "latest_rejected": len(latest.get("rejections") or []),
            "latest_waits": len(latest.get("waits") or []),
            "not_surfaced": len(latest.get("not_surfaced") or []),
            "correct_rejects": class_counts.get(CORRECT_REJECTION, 0),
            "missed_winners": class_counts.get(MISSED_WINNER, 0),
            "avoided_losers": class_counts.get(AVOIDED_LOSER, 0),
            "good_waits": class_counts.get(GOOD_WAIT, 0),
        },
        "explanations": {
            "taken": [dict(t.get("why") or {}, symbol=t.get("symbol")) for t in (latest.get("taken") or [])[:8]],
            "rejected": [dict(r.get("why") or {}, symbol=r.get("symbol"), reason_code=r.get("reason_code")) for r in (latest.get("rejections") or [])[:8]],
            "waits": [dict(w.get("why") or {}, symbol=w.get("symbol"), reason_code=w.get("reason_code")) for w in (latest.get("waits") or [])[:8]],
        },
        "live_readiness": evaluate_live_readiness(),
        "forward_soak": soak,
    }
