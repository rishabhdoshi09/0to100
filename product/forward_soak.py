"""Forward paper-trading soak: journey, settlement, health, scoreboard, daily report.

Does not add a scanner, recommendation engine, or trading path. It reads the
artifacts the existing money path already writes, freezes them into the Forward
Evidence Ledger, settles rejects when later official bars exist, and tells the
operator whether the loop is actually running.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from product.forward_evidence import (
    COUNTERFACTUAL,
    REAL_FORWARD_MARKET,
    TEST_FIXTURE,
    attach_settlement,
    attach_settlements,
    current_provenance,
    freeze_cycle,
    load_ledger,
    real_forward_only,
)
from product.live_safety import live_safety_projection
from product.promotion_governance import assess_component, promotion_board
from core.runtime_paths import logs_dir

ROOT = Path(__file__).resolve().parents[1]
JOURNEY_PATH = logs_dir() / "product" / "forward_journey.json"
DAILY_DIR = logs_dir() / "product" / "forward_daily"
VERIFY_PATH = logs_dir() / "product" / "forward_soak_verify.json"
SCHEMA_VERSION = 1
MIN_SCOREBOARD_N = 20
VERIFY_MIN_INTERVAL_S = 60
# Counterfactual rejections settle on the ONE rulebook (core.evaluation_spec),
# exactly like taken paper trades. Until 2026-11-20 this was a 5-session
# close-to-close move that ignored the stop (see counterfactual_learning).
from core.evaluation_spec import OUTCOME_HORIZON_SESSIONS as SETTLE_HORIZON  # noqa: E402

NOT_STARTED = "NOT_STARTED"
COLLECTING = "COLLECTING"
HEALTHY = "HEALTHY"
DEGRADED = "DEGRADED"
BLOCKED = "BLOCKED"

STAGE_ORDER = (
    "DATA_REFRESH",
    "MARKET_SCAN",
    "RECOMMENDATIONS",
    "SELECTION_AUTHORITY",
    "EVIDENCE_POLICY",
    "PORTFOLIO_SELECTION",
    "ENTRY_AUTHORITY",
    "PAPER_EXECUTION",
    "OPEN_POSITION",
    "EXIT_SUPERVISOR",
    "CLOSED_TRADE",
    "EXECUTION_ADJUSTED_EVIDENCE",
    "POLICY_LEARNING",
    "COUNTERFACTUAL_SETTLEMENT",
    "NEXT_CYCLE_POLICY_CONSUMPTION",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_path(name: str, default: Path) -> Path:
    override = os.environ.get(name)
    return Path(override) if override else default


def journey_path() -> Path:
    return _env_path("QT_FORWARD_JOURNEY", JOURNEY_PATH)


def daily_dir() -> Path:
    return _env_path("QT_FORWARD_DAILY", DAILY_DIR)


def verify_state_path() -> Path:
    return _env_path("QT_FORWARD_SOAK_VERIFY", VERIFY_PATH)


def load_latest_verification() -> dict[str, Any]:
    return _read_json(verify_state_path())


def persist_soak_verification(*, min_interval_s: int = VERIFY_MIN_INTERVAL_S, force: bool = False) -> dict[str, Any]:
    """Read-only verify. Throttled so the runtime does not spam disk."""
    existing = load_latest_verification()
    if existing and not force:
        stamp = str(existing.get("generated_at") or "")
        try:
            prev = datetime.fromisoformat(stamp)
            if prev.tzinfo is None:
                prev = prev.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - prev).total_seconds()
            if age < max(5, int(min_interval_s)):
                return existing
        except Exception:
            pass
    result = verify_persisted_soak()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "lanes": result.get("lanes") or {},
        "soak_status": result.get("soak_status"),
        "scoreboard_evidence": result.get("scoreboard_evidence"),
        "real_forward_n": result.get("real_forward_n"),
        "valid_no_trade": result.get("valid_no_trade"),
        "live_locked": result.get("live_locked"),
        "live_lock_verified": result.get("live_lock_verified") is True,
        "live_execution_authorized": result.get("live_execution_authorized"),
        "live_lock_status": result.get("live_lock_status") or "UNVERIFIED",
        "live_lock_reason": result.get("live_lock_reason") or "",
        "live_lock_source": result.get("live_lock_source") or "product.live_execution_interlock",
        "source": "verify_persisted_soak",
    }
    _write_json(verify_state_path(), payload)
    try:
        from product.runtime_capabilities import note_run
        note_run("forward_soak_verify", status=str(payload.get("soak_status") or ""), reason="automatic")
    except Exception:
        pass
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    from core.atomic_io import atomic_write_text
    atomic_write_text(path, json.dumps(dict(payload), indent=2, default=str))
    return path


def _stage(name: str, *, status: str, input_artifact: str = "", output_artifact: str = "",
           reason_code: str = "", cycle_id: str = "", decision_id: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "timestamp": _now(),
        "input_artifact": input_artifact,
        "output_artifact": output_artifact,
        "status": status,
        "reason_code": reason_code or "",
        "cycle_id": cycle_id,
        "decision_id": decision_id,
    }


def _scan_payload() -> dict[str, Any]:
    """Read-only view of the saved scan (shared parse; never mutate it)."""
    from core.json_cache import read_json_shared
    from product.scan_store import resolved_scan_path
    path = resolved_scan_path()
    try:
        payload = read_json_shared(path)
    except Exception:
        payload = {}
    return {"path": str(path), "payload": payload if isinstance(payload, dict) else {}}


def _reco_payload() -> dict[str, Any]:
    """Read-only view of the saved recommendations (shared parse; never mutate it)."""
    from core.json_cache import read_json_shared
    from product.recommendations_store import DEFAULT_RECO_PATH, load_recommendations
    path = Path(os.environ["QT_RECO_PATH"]) if os.environ.get("QT_RECO_PATH") else DEFAULT_RECO_PATH
    try:
        payload = read_json_shared(path, loader=load_recommendations) or {}
    except Exception:
        payload = {}
    return {"path": str(path), "payload": payload}


def record_cycle_evidence(cycle: Mapping[str, Any]) -> dict[str, Any]:
    """Called after Selection Authority finishes a paper cycle."""
    frozen = freeze_cycle(cycle)
    journey = build_runtime_journey(cycle=cycle)
    _write_json(journey_path(), journey)
    try:
        persist_soak_verification(min_interval_s=120)
    except Exception:
        pass
    return {"ledger": frozen, "journey": journey.get("summary")}


def _rulebook_outcome(symbol: str, as_of: str, row: Mapping[str, Any], horizon: int = SETTLE_HORIZON):
    """The rejection judged as if taken: same rulebook, stop/target path and costs."""
    try:
        from core.outcome_resolver import counterfactual_outcome
        return counterfactual_outcome(symbol, as_of, row.get("entry"), row.get("stop"), row.get("target"),
                                      horizon=horizon)
    except Exception:
        return None


# Counterfactual rows per forward-ledger rewrite (bounds redo work after a crash).
_SETTLE_FLUSH_EVERY = 250


def settle_pending_from_market(
    *,
    horizon: int = SETTLE_HORIZON,
    outcome_fn: Callable[[str, str, Mapping[str, Any]], Mapping[str, Any] | None] | None = None,
    later_entered: Mapping[str, bool] | None = None,
    floors: Mapping[str, int] | None = None,
    policy_path=None,
) -> dict[str, Any]:
    """Settle under ONE cross-process settlement lock (see _settle_pending_locked).

    Two settlers (the EOD job and the autonomous cycle) must not interleave: a
    stale one would compute another outcome for a row the other already
    committed, replace its still-pending learning event and then quarantine
    it, leaving a committed outcome with no replayable event (verification
    review QT-02). The pending rows are read fresh INSIDE the lock. Not
    re-entrant: never call it from inside a settlement.
    """
    import fcntl

    from product.forward_evidence import ledger_path as fwd_path

    ledger = Path(fwd_path())
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger.with_name(ledger.name + ".settle.lock"), "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            return _settle_pending_locked(horizon=horizon, outcome_fn=outcome_fn, later_entered=later_entered,
                                          floors=floors, policy_path=policy_path)
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _settle_pending_locked(
    *,
    horizon: int = SETTLE_HORIZON,
    outcome_fn: Callable[[str, str, Mapping[str, Any]], Mapping[str, Any] | None] | None = None,
    later_entered: Mapping[str, bool] | None = None,
    floors: Mapping[str, int] | None = None,
    policy_path=None,
) -> dict[str, Any]:
    """Rejected / waited / not-surfaced names get a classification when bars exist.

    Each rejection is judged as a taken trade would be (counterfactual_outcome:
    the rulebook horizon, stop/target path and costs), so a missed winner here
    means what a winner means for a taken trade. The settled row carries the
    rulebook's ``eval_spec``. Never books P&L. Missing bars stay pending.
    Duplicate settlement is a no-op. ``outcome_fn(symbol, as_of, row)`` is the
    test seam and must return a rulebook outcome dict (or None while pending).
    """
    from product.paper_learning_loop import ingest_counterfactual
    from product.counterfactual_learning import settle_rulebook, ledger_path as cf_path
    from product.forward_evidence import load_ledger as load_fwd

    getter = outcome_fn or (lambda symbol, as_of, row: _rulebook_outcome(symbol, as_of, row, horizon))
    entered = {str(k).upper(): bool(v) for k, v in dict(later_entered or {}).items()}
    from product.forward_evidence import is_forward_evidence
    # Simulated (BACKTEST / WALK_FORWARD) rows never become forward learning.
    # A row without its decision id cannot be attached or tied to a learning
    # event: it is never settled here (and never by a second settler, L-01).
    pending = [r for r in load_fwd() if not r.get("entered") and r.get("later_outcome") is None
               and is_forward_evidence(r) and str(r.get("decision_id") or "").strip()]
    updated = 0
    pending_n = 0
    classifications: dict[str, int] = {}
    chunk: list[tuple[dict[str, Any], dict[str, Any]]] = []

    from product import learning_observations as LO
    from product.learning_policy_store import policy_path as _policy_path

    outbox = LO.ledger_path_for(_policy_path(policy_path))
    failures: list[str] = []
    # Events a crash stranded (settled on the ledger, learning never confirmed)
    # are applied first; idempotent (cross-review A-3).
    replayed = replay_learning_outbox(policy_path=policy_path, floors=floors)
    failures.extend(replayed.get("errors") or [])
    conflicts: list[str] = []

    def _flush() -> None:
        # Exactly once across crashes (cross-review A-3). Order per chunk:
        #   1. every learning event is written to the durable OUTBOX;
        #   2. the forward ledger is rewritten once (attach_settlements);
        #   3. each event is applied and marked DONE.
        # A crash before 2 leaves the rows pending (settled again next run; the
        # outbox insert is idempotent). A crash after 2 leaves PENDING events
        # that replay_learning_outbox() applies on the next run. Applying is
        # idempotent per independent observation (two-phase claims), so a
        # replay of an already-applied event changes nothing. Until 2026-11-21
        # the ledger was written before the ingest with nothing durable in
        # between, so a crash permanently lost the rest of the chunk.
        if not chunk:
            return
        batch = list(chunk)
        chunk.clear()  # before ingesting: a raise here must not re-flush (re-ingest) it
        events = [(_learning_event_id(settled_row), settled_row) for _item, settled_row in batch
                  if settled_row.get("forward_decision_id")]
        for event_id, settled_row in events:
            # These rows are not settled yet, so a PENDING event under the same
            # id was never committed: this measurement supersedes it (L-02).
            state = LO.upsert_pending_event(outbox, event_id, LEARNING_EVENT_KIND, settled_row)
            if state not in ("NEW", "REPLACED"):
                conflicts.append(f"{event_id}: already {state}")
        committed = attach_settlements([item for item, _settled in batch])
        for event_id, settled_row in events:
            # Learn only what the ledger committed, exactly as committed (L-02).
            why = committed_mismatch(settled_row, committed.get(str(settled_row.get("forward_decision_id"))))
            if why:
                LO.quarantine_event(outbox, event_id, why)
                conflicts.append(f"{event_id}: {why}")
                continue
            err = _apply_learning_event(outbox, event_id, settled_row, policy_path=policy_path, floors=floors)
            if err:
                failures.append(err)

    try:
        for row in pending:
            symbol = str(row.get("symbol") or "").upper()
            as_of = str(row.get("market_timestamp") or row.get("pit_proof", {}).get("as_of") or "")[:10]
            outcome = getter(symbol, as_of, row) if as_of else None
            later = entered.get(symbol, False)
            if outcome is None and not later:
                pending_n += 1
                continue
            settled = settle_rulebook(
                {
                    "hypothetical_entry": row.get("entry"),
                    "hypothetical_stop": row.get("stop"),
                    "hypothetical_target": row.get("target"),
                    "reason_code": row.get("reason_code"),
                    "symbol": symbol,
                    # The forward-ledger row this settlement belongs to: the
                    # identity of its learning event (A-3).
                    "forward_decision_id": str(row.get("decision_id") or ""),
                    # The decision session: with symbol + reason it names the
                    # independent unit the learning ladder counts once.
                    "as_of": as_of,
                    "evidence": {
                        "setup_label": row.get("setup"),
                        "regime": row.get("regime"),
                        "rules_hash": row.get("rules_hash"),
                        # The decision's own id, so the label reaches the
                        # feature observation frozen under it (L-06). A label
                        # call for an id with no observation changes nothing.
                        "decision_id": str(row.get("decision_id") or ""),
                    },
                    "setup": row.get("setup"),
                    "regime": row.get("regime"),
                },
                outcome,
                later_entered=later,
            )
            chunk.append((
                {
                    "decision_id": str(row.get("decision_id")),
                    "classification": str(settled.get("classification") or ""),
                    "forward_return_pct": (settled.get("outcome") or {}).get("forward_return_pct"),
                    "outcome_provenance": COUNTERFACTUAL if later or outcome is not None else None,
                    "eval_spec": settled.get("eval_spec"),
                    "counterfactual_R": settled.get("counterfactual_R"),
                },
                settled,
            ))
            updated += 1
            cls = str(settled.get("classification") or "")
            classifications[cls] = classifications.get(cls, 0) + 1
            if len(chunk) >= _SETTLE_FLUSH_EVERY:
                _flush()
    finally:
        _flush()
    _ = cf_path
    result = {"updated": updated, "pending": pending_n, "classifications": classifications,
              "learning_conflicts": conflicts}
    if failures:
        # Settled and durably queued; learning retries from the outbox. Raise so
        # the outcome job is RETRYABLE, never a silent SUCCEEDED (A-3).
        raise LearningPending(f"{len(failures)} learning event(s) not yet applied (queued for replay): "
                              + "; ".join(failures[:3]), result=result)
    return result


class LearningPending(RuntimeError):
    """Outcomes were settled and their learning durably queued, but not all of it applied yet."""

    def __init__(self, message: str, *, result: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.result = dict(result or {})


def committed_mismatch(payload: Mapping[str, Any], row: Mapping[str, Any] | None) -> str:
    """'' when a learning payload is exactly the settlement the forward ledger committed, else why not.

    The outbox can hold a payload the ledger never committed (a crash before
    the write, then a different retry) or a damaged one. Learning applies a
    payload only when its identity, measurement and provenance equal the
    committed row's (overnight review L-02).
    """
    from product.forward_evidence import is_forward_evidence

    if not isinstance(row, Mapping):
        return "NOT_ON_LEDGER"
    later = row.get("later_outcome")
    if not isinstance(later, Mapping):
        return "NOT_SETTLED"
    if row.get("entered") or not is_forward_evidence(row):
        return "NOT_A_FORWARD_COUNTERFACTUAL"
    exact = {
        "FORWARD_DECISION_ID": (payload.get("forward_decision_id"), row.get("decision_id")),
        "SYMBOL": (str(payload.get("symbol") or "").upper(), str(row.get("symbol") or "").upper()),
        "REASON_CODE": (payload.get("reason_code"), row.get("reason_code")),
        "CLASSIFICATION": (payload.get("classification"), row.get("counterfactual_classification")),
        "EVAL_SPEC": (payload.get("eval_spec"), later.get("eval_spec")),
    }
    for name, (a, b) in exact.items():
        if str(a or "") != str(b or ""):
            return f"{name}_MISMATCH"
    numeric = {
        "COUNTERFACTUAL_R": (payload.get("counterfactual_R"), later.get("counterfactual_R")),
        "FORWARD_RETURN": ((payload.get("outcome") or {}).get("forward_return_pct")
                           if isinstance(payload.get("outcome"), Mapping) else None,
                           later.get("forward_return_pct")),
    }
    for name, (a, b) in numeric.items():
        try:
            same = (a is None and b is None) or (a is not None and b is not None and abs(float(a) - float(b)) <= 1e-9)
        except (TypeError, ValueError):
            same = False
        if not same:
            return f"{name}_MISMATCH"
    return ""


LEARNING_EVENT_KIND = "counterfactual"


def _learning_event_id(settled_row: Mapping[str, Any]) -> str:
    """Immutable identity of one settlement → learning event: the decision and the rulebook it was measured on."""
    return (f"{LEARNING_EVENT_KIND}:{settled_row.get('forward_decision_id') or ''}:"
            f"{settled_row.get('eval_spec') or 'no-spec'}")


def _settled_decision_ids(rows) -> set[str]:
    return {str(r.get("decision_id")) for r in rows or [] if r.get("later_outcome") is not None}


def _apply_learning_event(outbox, event_id: str, settled_row: Mapping[str, Any], *, policy_path=None,
                          floors=None) -> str:
    """Apply one queued event and mark it DONE. '' on success, else the error (the event stays queued)."""
    from product import learning_observations as LO
    from product.paper_learning_loop import ingest_counterfactual

    try:
        res = ingest_counterfactual(settled_row, path=policy_path, floors=floors) or {}
    except Exception as exc:
        err = f"{event_id}: {type(exc).__name__}: {exc}"[:240]
        LO.finish_event(outbox, event_id, error=err)
        return err
    result = ("duplicate" if res.get("duplicate_observation") else
              f"skipped:{res.get('skipped')}" if res.get("skipped") else "applied")
    LO.finish_event(outbox, event_id, result=result)
    return ""


def replay_learning_outbox(*, policy_path=None, floors=None, settled_ids: set[str] | None = None,
                           ledger_rows: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Apply every PENDING settlement → learning event (crash recovery; idempotent).

    Every event is checked against the forward ledger as committed (read here
    unless ``ledger_rows`` is given), whatever the arguments (L-02): an event
    whose row is not settled yet stays PENDING (its settlement will be redone
    and will supersede it); one that is not exactly the committed settlement,
    or is corrupt, is QUARANTINED, never guessed. ``settled_ids`` can only
    narrow what is applied. Policy claims a crash left half-applied are
    replayed first (learning_policy_store.recover_claims).
    """
    from product import learning_observations as LO
    from product.forward_evidence import load_ledger as load_fwd
    from product.learning_policy_store import policy_path as _policy_path, recover_claims

    outbox = LO.ledger_path_for(_policy_path(policy_path))
    out: dict[str, Any] = {"claims": recover_claims(policy_path, floors), "applied": 0, "waiting": 0,
                           "quarantined": 0, "errors": []}
    events = LO.pending_events(outbox, kind=LEARNING_EVENT_KIND)
    if not events:
        return out
    rows = load_fwd() if ledger_rows is None else ledger_rows
    by_id = {str(r.get("decision_id")): r for r in rows or [] if isinstance(r, Mapping) and r.get("decision_id")}
    for ev in events:
        payload = ev.get("payload")
        if not isinstance(payload, dict) or _learning_event_id(payload) != ev["event_id"]:
            LO.quarantine_event(outbox, ev["event_id"], "CORRUPT_OR_MISMATCHED_PAYLOAD")
            out["quarantined"] += 1
            continue
        fid = str(payload.get("forward_decision_id"))
        row = by_id.get(fid)
        if (row is None or row.get("later_outcome") is None
                or (settled_ids is not None and fid not in settled_ids)):
            out["waiting"] += 1
            continue
        why = committed_mismatch(payload, row)
        if why:
            LO.quarantine_event(outbox, ev["event_id"], why)
            out["quarantined"] += 1
            continue
        err = _apply_learning_event(outbox, ev["event_id"], payload, policy_path=policy_path, floors=floors)
        if err:
            out["errors"].append(err)
        else:
            out["applied"] += 1
    return out


def attach_closed_trades(book, *, path=None) -> dict[str, Any]:
    """Copy settled paper trades onto the ledger with execution-adjusted fields.

    Each close settles the ledger row of ITS OWN entry, once:

    * joined on the paper position's decision id (``paper_decision_id``) when
      the ledger row carries it;
    * for older rows without it, on the symbol AND the entry session, and only
      when exactly one unsettled entered row matches (ambiguity stays
      unresolved, never guessed);
    * a close already attached (``later_outcome.closed_trade_key``) is skipped.

    Until 2026-11-20 every historical close was re-matched by symbol alone to
    the newest unsettled entered row, so an old close settled a re-entry in the
    same symbol with the old R, and the write-once settlement then blocked the
    re-entry's real result (GPT cross-review A-2).
    """
    from product.forward_evidence import attach_settlements

    closed = list(getattr(book, "closed", []) or [])
    ledger = load_ledger(path)
    used = {str((r.get("later_outcome") or {}).get("closed_trade_key") or "") for r in ledger} - {""}
    claimed: set[str] = set()
    items: list[dict[str, Any]] = []
    ambiguous = 0
    for trade in closed:
        row = trade.as_dict() if hasattr(trade, "as_dict") else dict(trade)
        symbol = str(row.get("symbol") or "").upper()
        if not symbol:
            continue
        did = str(row.get("decision_id") or "")
        entry_day = str(row.get("entry_date") or "")[:10]
        key = did or "|".join([symbol, entry_day, str(row.get("exit_date") or "")[:10],
                               str(row.get("exit_reason") or "")])
        if key in used:
            continue
        open_rows = [r for r in ledger
                     if r.get("entered") and str(r.get("symbol")) == symbol and r.get("later_outcome") is None
                     and str(r.get("decision_id")) not in claimed]
        target = None
        if did:
            exact = [r for r in open_rows if str(r.get("paper_decision_id") or "") == did]
            target = exact[0] if exact else None
        if target is None:
            # Legacy rows (no paper id): the symbol's entered rows of that entry
            # session, settled OR pending, must be exactly one, and pending. A
            # close attached by the old code (no closed_trade_key) to one row of
            # the session must not settle a second (QT-08).
            def _day(r):
                return str(r.get("market_timestamp") or (r.get("pit_proof") or {}).get("as_of") or "")[:10]
            same_session = [r for r in ledger if r.get("entered") and str(r.get("symbol")) == symbol
                            and not r.get("paper_decision_id") and _day(r) == entry_day]
            if len(same_session) == 1 and same_session[0].get("later_outcome") is None \
                    and str(same_session[0].get("decision_id")) not in claimed:
                target = same_session[0]
            elif len(same_session) > 1:
                ambiguous += 1
        if target is None:
            continue
        try:
            from product.evidence_integrity import settled_learning_result
            from product.paper_learning_loop import _lookup_taken
            taken_ctx = _lookup_taken(symbol, decision_id=did, entry_date=entry_day)
            integrity = settled_learning_result(row, taken_ctx)
        except Exception:
            taken_ctx = {}
            integrity = {
                "gross_realized_R": row.get("realized_R"),
                "execution_adjusted_R": None,
                "execution_coverage": 0.0,
                "execution_complete": False,
            }
        shadow = dict((taken_ctx or {}).get("execution_reality_shadow") or {})
        charges = (shadow.get("execution_adjusted_result") or {}).get("charges_total")
        fill = dict(shadow.get("fill") or {})
        claimed.add(str(target.get("decision_id")))
        used.add(key)
        items.append({
            "decision_id": str(target.get("decision_id")),
            "classification": str(row.get("exit_reason") or "CLOSED"),
            "gross_R": integrity.get("gross_realized_R"),
            "execution_adjusted_R": integrity.get("execution_adjusted_R"),
            "execution_coverage": integrity.get("execution_coverage"),
            "execution_charges": charges,
            "spread_status": "measured" if any(
                f.get("name") == "bid_ask_spread" and f.get("measured")
                for f in (fill.get("fields") or [])
                if isinstance(f, Mapping)
            ) else "estimated_or_missing",
            "slippage_status": str(fill.get("reason") or "unknown"),
            "liquidity_assumptions": "volume participation labelled; no L2 book invented",
            "outcome_provenance": current_provenance(),
            "closed_trade_key": key,
        })
    if items:
        attach_settlements(items, path=path)
    return {"attached": len(items), "ambiguous_unresolved": ambiguous}


def settle_and_report(
    as_of: str,
    *,
    book=None,
    outcomes: Mapping[str, Mapping[str, Any]] | None = None,
    later_entered: Mapping[str, bool] | None = None,
    floors: Mapping[str, int] | None = None,
    policy_path=None,
) -> dict[str, Any]:
    """EOD hook: ingest is already done by the caller; this settles and reports."""
    closed = attach_closed_trades(book) if book is not None else {"attached": 0}
    by_symbol = {str(k).upper(): dict(v) for k, v in dict(outcomes or {}).items()}
    outcome_fn = (lambda symbol, _as_of, _row: by_symbol.get(symbol)) if outcomes is not None else None
    settled = settle_pending_from_market(
        outcome_fn=outcome_fn,
        later_entered=later_entered,
        floors=floors,
        policy_path=policy_path,
    )
    report = write_daily_report(as_of, book=book)
    journey = build_runtime_journey()
    _write_json(journey_path(), journey)
    status = soak_status()
    try:
        verified = persist_soak_verification(force=True)
    except Exception:
        verified = {}
    return {
        "closed_attached": closed,
        "counterfactuals": settled,
        "daily_report": str(report.get("json_path") or ""),
        "soak_status": status.get("status"),
        "soak_verification": verified.get("lanes") or {},
        **live_safety_projection(),
    }


def build_runtime_journey(*, cycle: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Root-to-leaf path from persisted artifacts, not injected unit rows."""
    from product.autopilot_journal import load_journal
    from product.paper_status import read_paper_status
    from product.learning_policy_store import load_policies

    scan = _scan_payload()
    reco = _reco_payload()
    journal = load_journal()
    latest = dict(cycle or journal.get("latest") or {})
    paper = read_paper_status()
    policies = load_policies()
    ledger = load_ledger()
    ingested = _read_json(Path(os.environ.get("QT_LEARNING_INGESTED") or logs_dir() / "product" / "learning_ingested.json"))
    cycle_id = str(latest.get("cycle_id") or "")
    scan_payload = dict(scan.get("payload") or {})
    scan_records = scan_payload.get("records")
    try:
        scanned_count = int(
            scan_payload.get("scanned")
            or scan_payload.get("universe_size")
            or scan_payload.get("approved_universe")
            or 0
        )
    except (TypeError, ValueError):
        scanned_count = 0
    # A completed market pass can truthfully produce zero qualifying rows.
    # Success is the persisted scan execution, not "at least one candidate".
    scan_ok = bool(
        isinstance(scan_records, list)
        and scan_payload.get("scanned_at")
        and (scanned_count > 0 or scan_payload.get("available") is True)
    )
    reco_ok = bool(reco["payload"])
    cycle_ok = bool(latest)

    # A fresh canonical decision-discovery projection can itself prove a
    # completed no-action selection for the current scan. This matters when
    # QuantTerm starts after the cash session: there is correctly no forward
    # paper-entry job to execute, but "zero actionable candidates" is still a
    # real Selection Authority result and must not be reported as a system
    # failure. Missing/stale discovery or any actionable candidate remains
    # fail-closed and still requires the paper-autopilot cycle.
    discovery_gate: dict[str, Any] = {}
    try:
        from product.decision_simulation_gate import status as decision_gate_status

        discovery_gate = dict(decision_gate_status() or {})
    except Exception:
        discovery_gate = {}
    current_scan_id = str(scan_payload.get("scanned_at") or "")
    gate_scan_id = str(discovery_gate.get("scan_scanned_at") or "")
    discovery_no_trade = bool(
        scan_ok
        and reco_ok
        and discovery_gate.get("discovery_ready")
        and discovery_gate.get("scan_fresh")
        and current_scan_id
        and gate_scan_id == current_scan_id
        and int(discovery_gate.get("actionable") or 0) == 0
    )
    selection_ok = bool(cycle_ok or discovery_no_trade)

    taken = list(latest.get("taken") or [])
    rejections = list(latest.get("rejections") or [])
    waits = list(latest.get("waits") or [])
    reasons = [str(x) for x in (latest.get("cycle_reasons") or []) if x]
    valid_no_trade = bool(
        (not taken) and (
            bool(rejections)
            or bool(waits)
            or bool(reasons)
            or discovery_no_trade
        )
    )
    opens = list(paper.open_positions or [])
    closed = list(paper.closed_trades or [])
    exec_rows = [r for r in ledger if r.get("entered") and r.get("execution_adjusted_R") is not None]
    settled_rows = [r for r in ledger if r.get("later_outcome") is not None]
    policy_consumed = any(
        str(row.get("policy_effect") or row.get("policy_result") or "") not in {"", "NEUTRAL"}
        for row in taken + rejections + waits
    ) or any(str(p.get("production_status")) in {"ACTIVE", "ELIGIBLE"} for p in (policies.get("policies") or []))

    stages = [
        _stage("DATA_REFRESH", status="PASS" if scan_ok or reco_ok else "PENDING",
               input_artifact="bhavcopy/official snapshot", output_artifact=scan["path"],
               reason_code="" if scan_ok or reco_ok else "NO_SAVED_SCAN", cycle_id=cycle_id),
        _stage("MARKET_SCAN", status="PASS" if scan_ok else "FAIL",
               input_artifact=scan["path"], output_artifact=scan["path"],
               reason_code="" if scan_ok else "SCAN_MISSING", cycle_id=cycle_id),
        _stage("RECOMMENDATIONS", status="PASS" if reco_ok else "FAIL",
               input_artifact=scan["path"], output_artifact=reco["path"],
               reason_code="" if reco_ok else "RECO_MISSING", cycle_id=cycle_id),
        _stage(
            "SELECTION_AUTHORITY",
            status="PASS" if selection_ok else "FAIL",
            input_artifact=reco["path"],
            output_artifact=(
                "logs/product/startup_trade_discovery.json"
                if discovery_no_trade and not cycle_ok
                else "logs/product/paper_autopilot_journal.json"
            ),
            reason_code=(
                "NO_ELIGIBLE_TRADE"
                if discovery_no_trade and not cycle_ok
                else ("" if cycle_ok else "NO_AUTOPILOT_CYCLE")
            ),
            cycle_id=cycle_id,
        ),
        _stage("EVIDENCE_POLICY", status="PASS" if cycle_ok else "PENDING",
               input_artifact="logs/product/learning_policies.json",
               output_artifact="paper_autopilot_journal.latest.policy_effect",
               cycle_id=cycle_id),
        _stage("PORTFOLIO_SELECTION", status="PASS" if cycle_ok else "PENDING",
               input_artifact="selection_authority eligible set",
               output_artifact="paper_autopilot_journal.latest.portfolio_authority",
               cycle_id=cycle_id),
        _stage("ENTRY_AUTHORITY", status="PASS" if cycle_ok else "PENDING",
               input_artifact="gates+windows", output_artifact="reason_code",
               reason_code=(reasons[0] if reasons and not taken else ""), cycle_id=cycle_id),
        _stage(
            "PAPER_EXECUTION",
            status="PASS" if taken else ("PASS" if valid_no_trade else "FAIL"),
            input_artifact="TradeIntent",
            output_artifact="logs/intelligence/intel_book.json",
            reason_code="" if taken else (reasons[0] if reasons else ("NO_ELIGIBLE_TRADE" if valid_no_trade else "NO_CYCLE")),
            cycle_id=cycle_id,
        ),
        _stage("OPEN_POSITION", status="PASS" if opens or taken else ("PASS" if valid_no_trade else "PENDING"),
               input_artifact="intel_book.open", output_artifact="intel_book.open",
               reason_code="" if opens or taken or valid_no_trade else "NO_OPEN", cycle_id=cycle_id),
        _stage("EXIT_SUPERVISOR", status="PASS" if closed or opens else "UNKNOWN",
               input_artifact="intel_book", output_artifact="intel_book.closed",
               reason_code="" if closed or opens else "NO_SUPERVISED_POSITION", cycle_id=cycle_id),
        _stage("CLOSED_TRADE", status="PASS" if closed else "PENDING",
               input_artifact="exit supervisor", output_artifact="intel_book.closed",
               reason_code="" if closed else "NO_CLOSED_TRADE", cycle_id=cycle_id),
        _stage("EXECUTION_ADJUSTED_EVIDENCE", status="PASS" if exec_rows else ("PARTIAL" if closed else "PENDING"),
               input_artifact="ExecutionRealityEngine shadow", output_artifact="forward_evidence.jsonl",
               reason_code="" if exec_rows else "EXECUTION_EVIDENCE_INCOMPLETE", cycle_id=cycle_id),
        _stage("POLICY_LEARNING", status="PASS" if ingested.get("keys") or closed else "PENDING",
               input_artifact="intel_book.closed + counterfactuals",
               output_artifact="logs/product/learning_policies.json",
               reason_code="" if ingested.get("keys") or closed else "NO_LEARNING_INGEST", cycle_id=cycle_id),
        _stage("COUNTERFACTUAL_SETTLEMENT", status="PASS" if settled_rows else "PENDING",
               input_artifact="official bhavcopy later bars", output_artifact="forward_evidence.jsonl",
               reason_code="" if settled_rows else "FORWARD_BARS_PENDING", cycle_id=cycle_id),
        _stage("NEXT_CYCLE_POLICY_CONSUMPTION", status="PASS" if policy_consumed or cycle_ok else "PENDING",
               input_artifact="learning_policies.json", output_artifact="evaluate_policies()",
               reason_code="" if cycle_ok else "NO_NEXT_CYCLE", cycle_id=cycle_id),
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "cycle_id": cycle_id,
        "rules_hash": latest.get("rules_hash") or "",
        "stages": stages,
        "summary": {s["name"]: s["status"] for s in stages},
        **live_safety_projection(),
        "valid_no_trade": valid_no_trade,
    }


def soak_status() -> dict[str, Any]:
    journey = build_runtime_journey()
    from product.autonomy_status import read_autonomy_status
    from product.paper_autopilot import reco_is_stale

    autonomy = {}
    try:
        autonomy = read_autonomy_status()
    except Exception:
        autonomy = {}
    scheduler = bool(autonomy.get("running"))
    reco = _reco_payload()["payload"]
    stale = False
    try:
        stale = bool(reco) and reco_is_stale(reco)
    except Exception:
        stale = False
    ledger = load_ledger()
    real = real_forward_only(ledger)
    summary = journey.get("summary") or {}
    blockers: list[str] = []
    if stale:
        blockers.append("STALE_RECOMMENDATION_ARTIFACT")
    if summary.get("MARKET_SCAN") == "FAIL" and summary.get("RECOMMENDATIONS") == "FAIL":
        blockers.append("NO_SCAN_OR_RECO")
    if summary.get("SELECTION_AUTHORITY") == "FAIL":
        blockers.append("NO_AUTOPILOT_CYCLE")

    if not ledger and summary.get("SELECTION_AUTHORITY") != "PASS":
        status = NOT_STARTED
        detail = "No forward soak artifacts yet"
    elif blockers:
        status = BLOCKED
        detail = "; ".join(blockers)
    elif scheduler and summary.get("SELECTION_AUTHORITY") == "PASS" and summary.get("PAPER_EXECUTION") == "PASS" and not stale:
        if summary.get("MARKET_SCAN") == "FAIL":
            status = DEGRADED
            detail = "Autopilot ran but saved scan artifact is missing"
        else:
            status = HEALTHY if (real or current_provenance() == TEST_FIXTURE) else COLLECTING
            detail = "Runtime artifacts present; process-alive is not the health signal"
            if status == HEALTHY and not real and current_provenance() != TEST_FIXTURE:
                status = COLLECTING
                detail = "Artifacts present but no REAL_FORWARD_MARKET rows yet"
    elif summary.get("SELECTION_AUTHORITY") == "PASS":
        status = COLLECTING
        detail = "Cycles recorded; scheduler not proven running"
    else:
        status = DEGRADED
        detail = "Partial runtime artifacts"

    if status == HEALTHY and not (
        summary.get("RECOMMENDATIONS") == "PASS"
        and summary.get("SELECTION_AUTHORITY") == "PASS"
        and summary.get("PAPER_EXECUTION") == "PASS"
    ):
        status = DEGRADED
        detail = "Scheduler up but required soak artifacts incomplete"

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "detail": detail,
        "scheduler_running": scheduler,
        "stale_recommendation": stale,
        "real_forward_n": len(real),
        "ledger_n": len(ledger),
        "stages": summary,
        "blockers": blockers,
        **live_safety_projection(),
        "process_alive_is_not_healthy": True,
    }


def _expectancy(values: Sequence[float]) -> float | None:
    if len(values) < MIN_SCOREBOARD_N:
        return None
    return round(sum(values) / len(values), 6)


def _group_expectancy(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        label = str(row.get(key) or "UNKNOWN")
        r = row.get("execution_adjusted_R")
        if r is None:
            r = row.get("gross_R")
        if r is None:
            continue
        buckets.setdefault(label, []).append(float(r))
    out = {}
    for label, vals in buckets.items():
        out[label] = {
            "n": len(vals),
            "expectancy": None if len(vals) < MIN_SCOREBOARD_N else round(sum(vals) / len(vals), 6),
            "evidence": "INSUFFICIENT EVIDENCE" if len(vals) < MIN_SCOREBOARD_N else "MEASURED",
        }
    return out


def _is_comparable_rejection(row: Mapping[str, Any]) -> bool:
    """A settled rejection measured under the CURRENT rulebook (or an event label like GOOD_WAIT)."""
    from product.counterfactual_learning import FORWARD_OUTCOME_CLASSES, is_rulebook
    cls = str(row.get("counterfactual_classification") or "")
    return cls not in FORWARD_OUTCOME_CLASSES or is_rulebook((row.get("later_outcome") or {}).get("eval_spec"))


def _counterfactual_r(row: Mapping[str, Any]) -> float | None:
    later_rb = dict(row.get("later_outcome") or {})
    if later_rb.get("counterfactual_R") is not None:          # rulebook net R, the taken-trade unit
        try:
            return float(later_rb["counterfactual_R"])
        except (TypeError, ValueError):
            return None
    try:
        entry = float(row.get("entry"))
        stop = float(row.get("stop"))
        later = dict(row.get("later_outcome") or {})
        move_pct = float(later.get("forward_return_pct"))
    except (TypeError, ValueError):
        return None
    risk = abs(entry - stop)
    if entry <= 0 or risk <= 0:
        return None
    return (entry * move_pct / 100.0) / risk



def rejection_gate_scorecard(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_n: int = MIN_SCOREBOARD_N,
) -> dict[str, Any]:
    """Descriptive counterfactual diagnostics by rejection gate; never P&L."""
    buckets: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw or {})
        if row.get("entered"):
            continue
        classification = str(row.get("counterfactual_classification") or "")
        if not classification or not _is_comparable_rejection(row):
            continue
        reason = str(row.get("reason_code") or "UNSPECIFIED_REJECTION")
        bucket = buckets.setdefault(
            reason,
            {
                "n": 0,
                "MISSED_WINNER": 0,
                "AVOIDED_LOSER": 0,
                "CORRECT_REJECTION": 0,
                "GOOD_WAIT": 0,
                "RAN_AWAY_WITHOUT_ENTRY": 0,
                "FLAT": 0,
                "counterfactual_R_values": [],
            },
        )
        bucket["n"] += 1
        bucket[classification] = int(bucket.get(classification) or 0) + 1
        r_value = _counterfactual_r(row)
        if r_value is not None:
            bucket["counterfactual_R_values"].append(float(r_value))

    out: dict[str, Any] = {}
    for reason, bucket in sorted(buckets.items()):
        vals = list(bucket.pop("counterfactual_R_values") or [])
        n = int(bucket.get("n") or 0)
        missed = int(bucket.get("MISSED_WINNER") or 0)
        protective = int(bucket.get("AVOIDED_LOSER") or 0) + int(
            bucket.get("CORRECT_REJECTION") or 0
        )
        out[reason] = {
            **bucket,
            "evidence": "MEASURED" if n >= int(min_n) else "INSUFFICIENT EVIDENCE",
            "missed_winner_rate": round(missed / n, 4) if n else None,
            "protective_rejection_rate": round(protective / n, 4) if n else None,
            "mean_rejected_move_R": (
                round(sum(vals) / len(vals), 6) if vals else None
            ),
            "sum_rejected_move_R": round(sum(vals), 6) if vals else None,
            "r_coverage": round(len(vals) / n, 4) if n else 0.0,
            "not_pnl": True,
            "interpretation": (
                "Positive rejected-move R means the gate rejected subsequent upside; "
                "negative R means it avoided subsequent downside. Counterfactual only."
            ),
        }
    return out



def scoreboard() -> dict[str, Any]:
    from product.learning_policy_store import load_policies
    from product.champion_challenger import load_store
    from product.live_readiness import evaluate_live_readiness

    from product.forward_evidence import independent_decisions

    ledger = load_ledger()
    real = real_forward_only(ledger)
    # Count decisions, not the per-cycle copies of a decision.
    use = independent_decisions(real) if real else []
    taken = [r for r in use if r.get("entered")]
    settled_taken = [r for r in taken if r.get("gross_R") is not None]
    all_rejected_settled = [r for r in use if not r.get("entered") and r.get("counterfactual_classification")]
    # Only rejections judged under the same rulebook as taken trades are counted;
    # legacy 5-session close-to-close labels are reported aside, never pooled.
    rejected_settled = [r for r in all_rejected_settled if _is_comparable_rejection(r)]
    adj = [r for r in settled_taken if r.get("execution_adjusted_R") is not None]
    coverage = (len(adj) / len(settled_taken)) if settled_taken else 0.0
    gross_vals = [float(r["gross_R"]) for r in settled_taken]
    adj_vals = [float(r["execution_adjusted_R"]) for r in adj]
    wins = [v for v in adj_vals or gross_vals if v > 0]
    losses = [v for v in adj_vals or gross_vals if v < 0]
    sample = adj_vals or gross_vals
    counts = {}
    for row in rejected_settled:
        cls = str(row.get("counterfactual_classification") or "")
        counts[cls] = counts.get(cls, 0) + 1
    policies = load_policies().get("policies") or []
    by_status: dict[str, int] = {}
    for policy in policies:
        st = str(policy.get("production_status") or "OBSERVING")
        by_status[st] = by_status.get(st, 0) + 1
    try:
        challengers = list((load_store() or {}).get("challengers") or [])
    except Exception:
        challengers = []
    eq = 0.0
    peak = 0.0
    dd = 0.0
    for v in sample:
        eq += v
        peak = max(peak, eq)
        dd = max(dd, peak - eq)
    # Trade statistics (win rate, drawdown, expectancy) are measured from
    # settled TAKEN trades. Thousands of settled rejections do not make an
    # unmeasured trade record "MEASURED".
    insufficient = len(settled_taken) < MIN_SCOREBOARD_N
    safety = live_safety_projection()
    live_component_status = (
        "LOCKED"
        if safety.get("live_lock_verified") is True and safety.get("live_locked") is True
        else "UNVERIFIED"
        if safety.get("live_lock_verified") is not True
        else "UNLOCKED"
    )
    board = promotion_board([
        {"component": "execution_reality_fills", "status": "SHADOW", "forward_n": len(adj),
         "gross_expectancy": _expectancy(gross_vals), "execution_adjusted_expectancy": _expectancy(adj_vals),
         "execution_adjusted_coverage": coverage if settled_taken else None, "notes": ["paper fills remain intended-price"]},
        {"component": "regime_intelligence_2", "status": "SHADOW", "forward_n": 0,
         "notes": ["shadow only; production still RISK_ON/RISK_OFF"]},
        {"component": "ml_challenger", "status": "SHADOW", "forward_n": 0,
         "notes": ["ML cannot execute"]},
        {"component": "live_money", "status": live_component_status, "forward_n": len(settled_taken),
         "gross_expectancy": _expectancy(gross_vals),
         "execution_adjusted_expectancy": _expectancy(adj_vals),
         "execution_adjusted_coverage": coverage if settled_taken else None,
         "notes": [safety.get("live_lock_reason") or "canonical broker-boundary safety"]},
    ])
    soak = soak_status()
    return {
        "schema_version": SCHEMA_VERSION,
        "FORWARD_SOAK_STATUS": soak.get("status"),
        "soak_detail": soak.get("detail"),
        **safety,
        "provenance_filter": REAL_FORWARD_MARKET,
        "real_forward_observations": len(use),
        "real_forward_ledger_rows": len(real),
        "observation_unit": "decision session x symbol x group x reason_code",
        "paper_trades_taken": len(taken),
        "settled_trades": len(settled_taken),
        "rejected_candidates_settled": len(rejected_settled),
        "rejected_ledger_rows_settled": sum(
            1 for r in real if not r.get("entered") and r.get("counterfactual_classification")
        ),
        "rejection_gate_evidence": rejection_gate_scorecard(rejected_settled),
        "rejected_legacy_eval_spec_excluded": len(all_rejected_settled) - len(rejected_settled),
        "missed_winners": counts.get("MISSED_WINNER", 0),
        "avoided_losers": counts.get("AVOIDED_LOSER", 0),
        "good_waits": counts.get("GOOD_WAIT", 0),
        "correct_rejections": counts.get("CORRECT_REJECTION", 0),
        "gross_expectancy": _expectancy(gross_vals),
        "execution_adjusted_expectancy": _expectancy(adj_vals),
        "execution_adjusted_coverage_pct": round(coverage * 100.0, 2) if settled_taken else None,
        "current_drawdown": None if insufficient else round(dd, 6),
        "win_rate": None if insufficient else round(len(wins) / len(sample), 4) if sample else None,
        "average_win": None if len(wins) < MIN_SCOREBOARD_N else round(sum(wins) / len(wins), 6),
        "average_loss": None if len(losses) < MIN_SCOREBOARD_N else round(sum(losses) / len(losses), 6),
        "setup_level_evidence": _group_expectancy(settled_taken, "setup"),
        "regime_level_evidence": _group_expectancy(settled_taken, "regime"),
        "sector_level_evidence": _group_expectancy(settled_taken, "sector"),
        "active_policies": by_status.get("ACTIVE", 0),
        "eligible_policies": by_status.get("ELIGIBLE", 0),
        "challengers_under_evaluation": sum(
            1 for c in challengers if str(c.get("status")) in {"SHADOW", "TESTING", "ELIGIBLE"}
        ),
        "promotion_blockers": board,
        "insufficient_evidence": insufficient,
        "evidence_label": "INSUFFICIENT EVIDENCE" if insufficient else "MEASURED",
        "live_readiness": evaluate_live_readiness(),
        "note": (
            "Promotion statistics use REAL_FORWARD_MARKET rows only. "
            "Test fixtures, backtests, and walk-forwards are excluded."
        ),
    }


def write_daily_report(as_of: str, *, book=None) -> dict[str, Any]:
    from product.autopilot_journal import load_journal, why_no_trade
    from product.strategy_catalog import ensemble_identity
    from product.learning_policy_store import load_policies

    day = str(as_of or "")[:10]
    journal = load_journal()
    latest = dict(journal.get("latest") or {})
    why = why_no_trade()
    scan = _scan_payload()["payload"]
    reco = _reco_payload()["payload"]
    reco_n = sum(len(list(c.get("cards") or [])) for c in (reco.get("categories") or []))
    ledger = load_ledger()
    day_rows = [r for r in ledger if str(r.get("market_timestamp") or "")[:10] == day]
    ident = ensemble_identity()
    closed_today = 0
    if book is not None:
        closed_today = sum(
            1 for t in (getattr(book, "closed", []) or [])
            if str(getattr(t, "exit_date", "") or (t.get("exit_date") if isinstance(t, Mapping) else ""))[:10] == day
        )
    policies = load_policies().get("policies") or []
    soak = soak_status()
    errors = [s for s in (build_runtime_journey().get("stages") or []) if s.get("status") in {"FAIL", "DEGRADED"}]
    safety = live_safety_projection()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "market_date": day,
        "generated_at": _now(),
        "scan_count": len(list(scan.get("records") or [])),
        "recommendation_count": reco_n,
        "eligible_count": int(latest.get("eligible_count") or 0),
        "trades_entered": len(latest.get("taken") or []),
        "trades_exited": closed_today,
        "reasons_no_trade": why.get("reasons") or latest.get("cycle_reasons") or [],
        "rejects_classified": sum(1 for r in day_rows if r.get("counterfactual_classification")),
        "policies_changed": [p.get("policy_id") for p in policies if p.get("updated_at")],
        "execution_adjusted_evidence_added": sum(
            1 for r in day_rows if r.get("execution_adjusted_R") is not None
        ),
        "errors_or_degraded_lanes": errors,
        "rules_hash": ident.get("rules_hash"),
        "strategy_id": ident.get("strategy_id"),
        "soak_status": soak.get("status"),
        **safety,
        "provenance": current_provenance(),
    }
    folder = daily_dir()
    folder.mkdir(parents=True, exist_ok=True)
    json_path = folder / f"{day}.json"
    md_path = folder / f"{day}.md"
    _write_json(json_path, payload)
    lines = [
        f"# Forward soak {day}",
        "",
        f"- scan_count: {payload['scan_count']}",
        f"- recommendation_count: {payload['recommendation_count']}",
        f"- eligible_count: {payload['eligible_count']}",
        f"- trades_entered: {payload['trades_entered']}",
        f"- trades_exited: {payload['trades_exited']}",
        f"- reasons_no_trade: {', '.join(str(x) for x in payload['reasons_no_trade']) or '—'}",
        f"- rejects_classified: {payload['rejects_classified']}",
        f"- execution_adjusted_evidence_added: {payload['execution_adjusted_evidence_added']}",
        f"- soak_status: {payload['soak_status']}",
        f"- rules_hash: {payload['rules_hash']}",
        f"- live_locked: {payload['live_locked']}",
        f"- live_lock_verified: {payload['live_lock_verified']}",
        f"- live_lock_status: {payload['live_lock_status']}",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json_path": str(json_path), "md_path": str(md_path), "report": payload}


def verify_persisted_soak() -> dict[str, Any]:
    """Operator + test entry: judge persisted artifacts and canonical safety only."""
    journey = build_runtime_journey()
    summary = journey.get("summary") or {}
    soak = soak_status()
    board = scoreboard()
    safety = live_safety_projection()
    verified_locked = safety.get("live_lock_verified") is True and safety.get("live_locked") is True
    if verified_locked:
        live_lane = "LOCKED"
    elif safety.get("live_lock_verified") is not True:
        live_lane = "UNVERIFIED"
    else:
        live_lane = "UNLOCKED"
    paper_exec = summary.get("PAPER_EXECUTION") or "FAIL"
    if paper_exec == "PASS" and journey.get("valid_no_trade"):
        paper_label = "NO_ELIGIBLE_TRADE"
    elif paper_exec == "PASS":
        paper_label = "PASS"
    else:
        paper_label = paper_exec
    lanes = {
        "SCAN": summary.get("MARKET_SCAN") or "FAIL",
        "RECOMMENDATIONS": summary.get("RECOMMENDATIONS") or "FAIL",
        "SELECTION": summary.get("SELECTION_AUTHORITY") or "FAIL",
        "AUTOPILOT": summary.get("SELECTION_AUTHORITY") or "FAIL",
        "PAPER EXECUTION": paper_label,
        "EXIT SUPERVISION": summary.get("EXIT_SUPERVISOR") or "UNKNOWN",
        "FORWARD SETTLEMENT": summary.get("COUNTERFACTUAL_SETTLEMENT") or "PENDING",
        "LEARNING INGESTION": summary.get("POLICY_LEARNING") or "PENDING",
        "EXECUTION REALITY": summary.get("EXECUTION_ADJUSTED_EVIDENCE") or "PENDING",
        "LIVE MONEY": live_lane,
    }
    return {
        "lanes": lanes,
        "soak_status": soak.get("status"),
        "scoreboard_evidence": board.get("evidence_label"),
        "real_forward_n": board.get("real_forward_observations"),
        "execution_adjusted_coverage_pct": board.get("execution_adjusted_coverage_pct"),
        "valid_no_trade": journey.get("valid_no_trade"),
        **safety,
        "journey": journey,
    }
