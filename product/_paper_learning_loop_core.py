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
    locked_ledger,
    settle,
)
from product.evidence_policy_engine import HARD_REASON_CODES
from product.learning_policy_store import record_measured_outcome
from core.runtime_paths import logs_dir

ROOT = Path(__file__).resolve().parents[1]
TAKEN_PATH = logs_dir() / "product" / "taken_evidence.jsonl"
INGESTED_PATH = logs_dir() / "product" / "learning_ingested.json"


def _env_path(name: str, default: Path) -> Path:
    override = os.environ.get(name)
    return Path(override) if override else default


def taken_path() -> Path:
    return _env_path("QT_TAKEN_EVIDENCE", TAKEN_PATH)


def ingested_path() -> Path:
    return _env_path("QT_LEARNING_INGESTED", INGESTED_PATH)


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Append one record durably (taken-trade evidence, which a later close joins back to).

    Synced before return (cycle 10: a plain append could vanish in a power cut while the
    durably saved paper position survived, leaving its close with no taken evidence). A
    file that ends without a newline (a torn append) gets one first, so this record is
    never fused onto the torn line and lost with it; the readers skip the torn line.
    """
    import fcntl
    import os

    from core.atomic_io import durable_sync

    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(dict(row), default=str) + "\n").encode("utf-8")
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        # One writer at a time, from the tail check through the sync, on THIS descriptor
        # (threads and processes alike): a concurrent append can never land inside a
        # record cut by a short write (GPT C10-02). Closing the descriptor releases it.
        fcntl.flock(fd, fcntl.LOCK_EX)
        size = os.fstat(fd).st_size
        if size and os.pread(fd, 1, size - 1) != b"\n":
            line = b"\n" + line
        view, written = memoryview(line), 0
        while written < len(line):
            written += os.write(fd, view[written:])
        durable_sync(fd)
    finally:
        os.close(fd)


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
            "paper_decision_id": str(row.get("paper_decision_id") or ""),
            "entry_date": str(row.get("entry_date") or as_of or "")[:10],
        })


def _trade_key(trade: Mapping[str, Any]) -> str:
    return "|".join([
        str(trade.get("symbol") or "").upper(),
        str(trade.get("entry_date") or ""),
        str(trade.get("exit_date") or ""),
        str(trade.get("exit_reason") or ""),
    ])


def _lookup_taken(symbol: str, *, decision_id: str = "", entry_date: str = "") -> dict[str, Any]:
    """The taken-entry context of a paper position.

    Joined on the position's decision id when both sides carry it, else on its
    entry session. A symbol-only match is used only when the trade names
    neither: a re-entry in the same symbol must not lend its setup and
    execution geometry to an older close (GPT cross-review A-2).
    """
    symbol = symbol.upper()
    rows = [r for r in _read_jsonl(taken_path()) if str(r.get("symbol") or "").upper() == symbol]
    if decision_id:
        for row in reversed(rows):
            if str(row.get("paper_decision_id") or "") == str(decision_id):
                return row
    if entry_date:
        day = str(entry_date)[:10]
        # A row naming ANOTHER position is never borrowed, and the date match
        # must be unambiguous (cross-review verification QT-08).
        dated = [r for r in rows if str(r.get("entry_date") or r.get("as_of") or "")[:10] == day
                 and not (decision_id and r.get("paper_decision_id")
                          and str(r.get("paper_decision_id")) != str(decision_id))]
        return dated[0] if len(dated) == 1 else {}
    return rows[-1] if rows else {}


def record_conditional_evidence(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Fold one settled paper trade into the conditional evidence it belongs to.

    Only a trade that can name the decision it came from counts. A position
    opened before the linkage existed, or by one of the research simulators,
    has no canonical decision behind it and therefore no cell it was ranked in
    — recording it anywhere would be inventing an attribution.

    The paper venue has no broker to acknowledge an order, so the intent IS the
    order: paper_order_id is the intent's id rather than a fabricated one. The
    position id is derived from the intent and the entry session, which are
    exactly what identify the position in the book.
    """
    decision_id = str(row.get("decision_id") or "")
    context_key = str(row.get("context_key") or "")
    intent_id = str(row.get("paper_intent_id") or "")
    if not decision_id or not context_key or not intent_id:
        return None
    if row.get("realized_R") is None:
        return None

    try:
        from product.conditional_evidence import record_outcome
        from product.decision_chain import Outcome, _derive
        from product.evidence_class import PAPER_FORWARD

        position_id = _derive("pos", intent_id, str(row.get("symbol") or ""),
                              str(row.get("entry_date") or ""))
        outcome = Outcome(
            position_id=position_id,
            paper_order_id=intent_id,
            paper_intent_id=intent_id,
            decision_id=decision_id,
            symbol=str(row.get("symbol") or ""),
            exit_reason=str(row.get("exit_reason") or ""),
            realized_R=float(row.get("realized_R")),
            entry_price=row.get("entry_price"),
            exit_price=row.get("exit_price"),
            entry_session=str(row.get("entry_date") or "")[:10],
            exit_session=str(row.get("exit_date") or "")[:10],
            evidence_class=PAPER_FORWARD,
        )
        update = record_outcome(outcome, context_key=context_key,
                                evidence_class=PAPER_FORWARD, strict_ladder=True)
        learning_observation = None
        try:
            from product.evidence_intelligence import record_resolved_prediction
            learning_observation = record_resolved_prediction(
                decision_id,
                float(row.get("realized_R")),
                evidence_class=PAPER_FORWARD,
                not_pnl=False,
                resolved_at=outcome.resolved_at,
            )
        except Exception as exc:
            learning_observation = {"error": str(exc)[:200]}
        return {
            "decision_id": decision_id,
            "outcome_id": outcome.outcome_id,
            "evidence_update_id": update.evidence_update_id,
            "context_key": context_key,
            "before": update.before,
            "after": update.after,
            "changed": update.changed,
            "learning_observation": learning_observation,
        }
    except Exception as exc:
        from product.conditional_evidence import LadderPending
        if isinstance(exc, LadderPending):
            raise                       # retryable: the close must not be checkpointed (QT-05)
        log_note = str(exc)[:200]
        return {"error": log_note, "decision_id": decision_id}


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
    evidence = _lookup_taken(symbol, decision_id=str(row.get("decision_id") or ""),
                             entry_date=str(row.get("entry_date") or ""))
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
    conditional = record_conditional_evidence(row)
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
        # The chain back to the decision, so a policy row months from now can
        # still name the trades behind it.
        "decision_id": str(row.get("decision_id") or ""),
        "context_key": str(row.get("context_key") or ""),
        "evidence_update_id": str((conditional or {}).get("evidence_update_id") or ""),
    }
    # One closed trade is one independent observation for every policy it feeds.
    obs = {
        "observation_key": f"trade:{_trade_key(row)}",
        "session": str(row.get("entry_date") or "")[:10] or None,
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
        **obs,
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
            **obs,
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
            **obs,
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
            **obs,
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
            **obs,
        )
    if isinstance(last, dict) and conditional:
        last = {**last, "conditional_evidence": conditional}
    return last


def ingest_closed_book(book, *, path=None, floors: Mapping[str, int] | None = None) -> dict[str, Any]:
    """Idempotent: only new ClosedTrade rows update policies."""
    ingested = _load_json(ingested_path())
    seen = set(ingested.get("keys") or [])
    closed = list(getattr(book, "closed", []) or [])
    applied = 0
    last = None
    failures: list[BaseException] = []
    for trade in closed:
        row = trade.as_dict() if hasattr(trade, "as_dict") else dict(trade)
        key = _trade_key(row)
        if key in seen:
            continue
        try:
            last = ingest_closed_trade(row, path=path, floors=floors)
        except Exception as exc:
            # Not checkpointed: the next call retries this close, and every sink
            # is idempotent per outcome. The others still land (QT-05).
            failures.append(exc)
            continue
        seen.add(key)
        applied += 1
    target = ingested_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        # Every key is kept. A cap (formerly sorted(seen)[-2000:]) dropped the
        # alphabetically-first keys, so those trades would be ingested again.
        "keys": sorted(seen),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "applied": applied,
    }
    from core.atomic_io import atomic_write_text
    atomic_write_text(target, json.dumps(payload, indent=2), durable=True)    # the dedupe keys (audit F2)
    if failures:
        raise failures[0]
    return {"applied": applied, "last_policy": last}


def verified_feature_link(settled: Mapping[str, Any], decision_id: str = "") -> tuple[str, str]:
    """(feature decision id, status) for the feature observation a research outcome may label.

    The outcome of a rejected or waited decision labels the feature observation
    THAT decision froze, and nothing else (verification review QT-06):

    * the explicit ``feature_decision_id`` the decision carried, else the
      decision's own id (an explicitly linked legacy row);
    * the observation must exist, belong to the same symbol, be a PAPER_FORWARD
      decision (a simulated or replayed observation is never labelled with a
      forward outcome), and come from the same scan when both name one.

    Anything else returns ('', reason): reported, never matched by guesswork.
    """
    evidence = dict(settled.get("evidence") or {})
    explicit = str(evidence.get("feature_decision_id") or settled.get("feature_decision_id") or "")
    candidate = explicit or str(decision_id or "")
    if not candidate:
        return "", "NO_DECISION_ID"
    from product.evidence_class import PAPER_FORWARD
    from product.evidence_intelligence import _observation_id
    from research.feature_store import get_observation_strict
    # A store that cannot be read RAISES (FeatureStoreUnavailable): the unit stays
    # CLAIMED and the event is retried, never completed as "absent" (QT-C2-06).
    obs = get_observation_strict(_observation_id(candidate))
    if not obs:
        return "", "UNLINKED" if explicit else "NO_FEATURE_OBSERVATION"
    if str(obs.get("symbol") or "").upper() != str(settled.get("symbol") or "").upper():
        return "", "LINK_CONFLICT_SYMBOL"
    meta = dict(obs.get("meta") or {})
    # Explicit forward provenance only: a missing class is not PAPER_FORWARD.
    if str(meta.get("evidence_class") or "") != PAPER_FORWARD:
        return "", "LINK_CONFLICT_EVIDENCE_CLASS"
    # The immutable binding: the observation must BE this decision, at these levels.
    if str(meta.get("decision_id") or "") != candidate:
        return "", "LINK_CONFLICT_DECISION_ID"
    for key, mine in (("entry", settled.get("hypothetical_entry", settled.get("entry"))),
                      ("stop", settled.get("hypothetical_stop", settled.get("stop"))),
                      ("target", settled.get("hypothetical_target", settled.get("target")))):
        theirs = meta.get(key)
        try:
            if mine is None or theirs is None or abs(float(mine) - float(theirs)) > 1e-6 * max(1.0, abs(float(mine))):
                return "", f"LINK_CONFLICT_{key.upper()}"
        except (TypeError, ValueError):
            return "", f"LINK_CONFLICT_{key.upper()}"
    scan = str(evidence.get("scan_scanned_at") or "")
    source = str(meta.get("source_scan_id") or "")
    if not scan or not source or scan != source:
        return "", "LINK_CONFLICT_SCAN"
    return candidate, "VERIFIED"


def counterfactual_observation_key(settled: Mapping[str, Any]) -> tuple[str, str] | None:
    """(key, session) for the independent unit a settled counterfactual is.

    Every paper cycle re-freezes the names it rejects, so one symbol rejected for
    one reason on one day appears once per cycle. It is one observation.
    """
    evidence = dict(settled.get("evidence") or {})
    day = str(
        settled.get("as_of")
        or evidence.get("as_of")
        or settled.get("market_timestamp")
        or (settled.get("pit_proof") or {}).get("as_of")
        or ""
    )[:10]
    symbol = str(settled.get("symbol") or "").upper()
    reason = str(settled.get("reason_code") or "REJECTED")
    if len(day) != 10 or not symbol:
        return None
    return f"cf:{day}|{symbol}|{reason}", day


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
    # One rulebook: a forward-outcome label counts only when it was measured
    # under the CURRENT rulebook (horizon, stop/target path, costs), the same as
    # taken trades. Legacy 5-session close-to-close labels, or any other spec,
    # never reach a policy, the feature store or a dedupe claim.
    from product.counterfactual_learning import FORWARD_OUTCOME_CLASSES, UNMEASURED, is_rulebook
    spec = settled.get("eval_spec") or (settled.get("outcome") or {}).get("eval_spec")
    if classification == UNMEASURED:
        return {"skipped": "UNMEASURED", "classification": classification}
    if classification == GOOD_WAIT and not str(settled.get("later_entry_verified") or ""):
        # GOOD_WAIT is a claim about a WAIT and a fill verified inside its own
        # horizon; without the verified fill session it is never learned (QT-C2-07).
        return {"skipped": "UNVERIFIED_GOOD_WAIT", "classification": classification}
    if classification in FORWARD_OUTCOME_CLASSES and not is_rulebook(spec):
        return {"skipped": "INCOMPATIBLE_EVAL_SPEC", "eval_spec": spec, "classification": classification}
    evidence = dict(settled.get("evidence") or {})
    decision_id = str(evidence.get("decision_id") or settled.get("decision_id")
                      or settled.get("forward_decision_id") or "")
    keyed = counterfactual_observation_key(settled)
    if keyed is None and decision_id:
        # No (day, symbol) unit: key it by its decision, so a replay of the same
        # event can never count twice (an "unkeyed" observation always counted).
        keyed = (f"cfd:{decision_id}", "")
    obs: dict[str, Any] = {}
    unit_ledger = None
    if keyed is not None:
        from product import learning_observations as LO
        from product.learning_policy_store import policy_path

        key, day = keyed
        unit_ledger = LO.ledger_path_for(policy_path(path))
        # Whole-ingest dedupe, resumable (cross-review A-3): an independent unit
        # applied before (or by the pre-two-phase code: LEGACY) updates nothing;
        # a unit a crash left CLAIMED is RESUMED. Every effect below is itself
        # idempotent per observation key, so resuming never double-counts.
        state = LO.claim_state(unit_ledger, "COUNTERFACTUAL", key)
        if state in (LO.APPLIED, LO.LEGACY):
            return {"duplicate_observation": True, "observation_key": key}
        if not state:
            # The claim stores the WHOLE delivery that won it (L-03).
            body = {"kind": "counterfactual_ingest", "decision_id": decision_id,
                    "settled": json.loads(json.dumps(dict(settled), default=str))}
            if not LO.claim(unit_ledger, "COUNTERFACTUAL", key, day, payload=body):
                state = LO.claim_state(unit_ledger, "COUNTERFACTUAL", key)   # another worker won
                if state in (LO.APPLIED, LO.LEGACY):
                    return {"duplicate_observation": True, "observation_key": key}
        if state == LO.CLAIMED:
            # Resume the unit with ITS OWN payload, never this delivery's: two
            # cycle copies of one unit can differ (target, outcome), and a unit
            # must not finish with effects from two decisions (overnight review
            # L-03).
            claimed = LO.claim_payload(unit_ledger, "COUNTERFACTUAL", key) or {}
            stored = claimed.get("settled")
            if not isinstance(stored, dict):
                # A claim from before payloads were stored names only its
                # decision: resume it only with a delivery OF THAT decision; any
                # other delivery waits as a duplicate, never mixes its effects
                # into the unit (verification review QT-03).
                owner = str(claimed.get("decision_id") or "")
                if owner and owner != decision_id:
                    return {"duplicate_observation": True, "observation_key": key,
                            "deferred_to_decision": owner}
            if isinstance(stored, dict) and (counterfactual_observation_key(stored) == keyed
                                             or keyed[0].startswith("cfd:")):
                settled = stored
                classification = str(settled.get("classification") or "")
                reason = str(settled.get("reason_code") or "REJECTED")
                spec = settled.get("eval_spec") or (settled.get("outcome") or {}).get("eval_spec")
                evidence = dict(settled.get("evidence") or {})
                decision_id = str(evidence.get("decision_id") or settled.get("decision_id")
                                  or settled.get("forward_decision_id") or "")
        obs = {"observation_key": key, "session": day}
    counterfactual_r = settled.get("counterfactual_R")
    mapped = {
        CORRECT_REJECTION: 0.40,
        AVOIDED_LOSER: 0.40,
        MISSED_WINNER: -0.40,
        GOOD_WAIT: 0.20,
        RAN_AWAY: 0.0,
        FLAT: 0.0,
    }.get(classification, 0.0)
    hard = reason in HARD_REASON_CODES or reason in {"SECTOR_CAP", "CORRELATION_CAP", "MAX_PORTFOLIO_RISK"}
    from product.learning_policy_store import load_policies
    policy_store = load_policies(path)
    hist = next(
        (
            dict(p)
            for p in (policy_store.get("policies") or [])
            if str(p.get("policy_id") or "") == f"HIST_REJECT::{reason}"
        ),
        {},
    )
    historical_bridge = {
        "historical_policy_id": str(hist.get("policy_id") or ""),
        "historical_generation_fingerprint": str(hist.get("generation_fingerprint") or ""),
        "historical_over_rejection_candidate": bool(hist.get("over_rejection_candidate")),
        "historical_missed_rate": hist.get("historical_missed_rate"),
        "historical_hypothesis_linked": bool(hist),
    }
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
            **historical_bridge,
        },
        **obs,
    )
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
            **obs,
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
            **obs,
        )
    feature_id, feature_link = verified_feature_link(settled, decision_id)
    if feature_id and counterfactual_r is not None:
        # The feature label is a sink of its own (L-06): idempotent (an equal
        # label is "exists"), so a failure is raised, the unit stays CLAIMED
        # and the event is retried; the policy effects above are idempotent per
        # observation key, so the retry changes only what did not land. It goes
        # only to a VERIFIED link (QT-06).
        from product.evidence_intelligence import record_resolved_prediction
        labelled = record_resolved_prediction(
            feature_id,
            float(counterfactual_r),
            evidence_class="FORWARD_COUNTERFACTUAL",
            not_pnl=True,
            classification=classification,
            eval_spec=str(spec or ""),
            resolved_at=str(
                (settled.get("outcome") or {}).get("resolved_at")
                or datetime.now(timezone.utc).isoformat()
            ),
        )
        label_status = str(((labelled or {}).get("feature_store") or {}).get("status") or "")
        if label_status == "error":
            raise RuntimeError(f"feature label for {feature_id} failed: "
                               f"{((labelled or {}).get('feature_store') or {}).get('reason')}"[:240])
        if isinstance(last, dict):
            last = {**last, "feature_label": label_status}
    if isinstance(last, dict):
        last = {**last, "feature_link": feature_link}
    if unit_ledger is not None:
        from product import learning_observations as LO
        LO.mark_applied(unit_ledger, "COUNTERFACTUAL", obs["observation_key"])
    return last


#: A WAIT is credited with a later entry only inside its own outcome horizon.
GOOD_WAIT_MAX_CALENDAR_DAYS = 14          # retained for callers; no longer a fallback (L-08: unverifiable = pending)


def _wait_is_live(row: Mapping[str, Any], entry_day: str) -> bool:
    """True when ``row`` is a WAIT whose own horizon still covers ``entry_day``.

    GOOD_WAIT means "we waited and a valid entry came": it is a claim about
    THAT wait. Crediting it to a hard rejection, or to a wait weeks older than
    its horizon, rewarded unrelated decisions (cross-review open item,
    2026-11-21).
    """
    # Unverifiable ordinal (no archive, a wait older than it, an uncovered
    # calendar year): the credit stays pending, never guessed (L-08).
    return _wait_ordinal_state(row, entry_day) == "LIVE"


def _wait_ordinal_state(row: Mapping[str, Any], entry_day: str) -> str:
    """LIVE / EXPIRED / UNVERIFIABLE for a WAIT and a later fill; NOT_A_WAIT otherwise."""
    if str(row.get("decision") or "").upper() != "WAIT":
        return "NOT_A_WAIT"
    day = str(row.get("as_of") or "")[:10]
    if len(day) != 10 or not entry_day or day > entry_day:
        return "NOT_A_WAIT"
    from core.evaluation_spec import OUTCOME_HORIZON_SESSIONS
    n = _sessions_through(day, str(entry_day)[:10])
    if n is None:
        return "UNVERIFIABLE"
    return "LIVE" if n <= int(OUTCOME_HORIZON_SESSIONS) else "EXPIRED"


#: A fill queued for verification is dropped (the WAIT settles on its own price
#: path) when the archive still cannot verify it this long after the fill.
LATER_ENTRY_VERIFY_MAX_DAYS = 45


def _queue_later_fills(symbol: str, entry_day: str) -> int:
    """Durably record a later fill on each pending WAIT of ``symbol`` it may credit, BEFORE any credit.

    The marker is the credit's durable intent (lifecycle audit F1, 24 Nov). The credit used
    to claim the unit, update two policies and rewrite this ledger in one pass, with nothing
    on disk recording the fill: a crash anywhere in it lost the credit for good (the fill is
    never seen again) and the next official settlement classified the filled WAIT on its
    price path, MISSED_WINNER beside a policy that had counted GOOD_WAIT. Now a WAIT whose
    horizon covers the fill (LIVE) or cannot be verified yet (UNVERIFIABLE, QT-07) is marked
    first, and every credit is made from a marker (retry_pending_good_waits), which every
    official settlement runs before any price path. The earliest fill is kept.
    """
    with locked_ledger(ledger_path()) as (ledger, rows, _):
        changed = 0
        for row in rows:
            if row.get("classification") or str(row.get("symbol") or "").upper() != symbol:
                continue
            if row.get("later_entry_pending"):
                continue
            if _wait_ordinal_state(row, entry_day) in ("LIVE", "UNVERIFIABLE"):
                row["later_entry_pending"] = {"fill_session": str(entry_day)[:10],
                                              "noted_at": datetime.now(timezone.utc).isoformat()}
                changed += 1
        if changed:
            from core.atomic_io import atomic_write_text
            atomic_write_text(ledger, "".join(json.dumps(r, default=str) + "\n" for r in rows), durable=True)
        return changed


def _claimed_good_wait_units(path=None) -> dict[str, str]:
    """{unit key: its verified fill} for every unit the GOOD_WAIT credit has claimed (CLAIMED or APPLIED).

    Such a unit's credit was decided when it was claimed; a ledger row of it that is still
    unclassified is a credit a crash cut short, including one cut short before fills were
    marked (lifecycle audit F1). It is finished with the unit's own fill, never re-decided.
    """
    from product import learning_observations as LO
    from product.learning_policy_store import policy_path

    units: dict[str, str] = {}
    for unit in LO.scope_payloads(LO.ledger_path_for(policy_path(path)), "COUNTERFACTUAL", contains=GOOD_WAIT):
        body = unit.get("payload")
        settled = body.get("settled") if isinstance(body, dict) else None
        if not isinstance(settled, dict) or str(settled.get("classification") or "") != GOOD_WAIT:
            continue
        fill = str(settled.get("later_entry_verified") or "")[:10]
        keyed = counterfactual_observation_key(settled)
        if len(fill) == 10 and keyed is not None and keyed[0] == unit["observation_key"]:
            units[keyed[0]] = fill
    return units


def good_wait_holds(rows: Sequence[Mapping[str, Any]], *, path=None) -> dict[str, str]:
    """{unit key: verified fill} of every unit a GOOD_WAIT credit holds (cycle-9 review A2, A3).

    A unit is held while one of its copies carries a queued fill (its fill session), or once
    the GOOD_WAIT credit has claimed it (its verified fill). No settler may price a held
    unit: the official one waits, and the forward one settles it GOOD_WAIT only from that
    fill. ``rows`` are the counterfactual ledger's rows as read under its lock.
    """
    holds: dict[str, str] = {}
    for row in rows:
        pending = row.get("later_entry_pending")
        if row.get("classification") or not isinstance(pending, Mapping):
            continue
        keyed = counterfactual_observation_key(row)
        if keyed is not None:
            holds.setdefault(keyed[0], str(pending.get("fill_session") or "")[:10])
    holds.update(_claimed_good_wait_units(path))
    return holds


def _unit_classification(key: str, path=None) -> str:
    """The classification a claimed COUNTERFACTUAL unit was decided with ('' when unknown)."""
    from product import learning_observations as LO
    from product.learning_policy_store import policy_path

    body = LO.claim_payload(LO.ledger_path_for(policy_path(path)), "COUNTERFACTUAL", key) or {}
    settled = body.get("settled") if isinstance(body, Mapping) else None
    return str((settled or {}).get("classification") or "") if isinstance(settled, Mapping) else ""


def retry_pending_good_waits(*, path=None, floors: Mapping[str, int] | None = None,
                             today: str | None = None, symbols=None) -> dict[str, Any]:
    """Credit queued later fills; GOOD_WAIT only when the ordinal verifies LIVE (QT-07).

    Every GOOD_WAIT credit is made here, from a durable marker (_queue_later_fills), so a
    crash at any point leaves a marker this retries (lifecycle audit F1). EXPIRED drops
    the fill (the WAIT settles on its own price path); still UNVERIFIABLE keeps it queued,
    until LATER_ENTRY_VERIFY_MAX_DAYS after the fill, then drops it as unverifiable. A row
    of a unit the credit already claimed is finished with that unit's own fill. Never
    fabricates a session. ``symbols`` limits the pass to those symbols.
    """
    from datetime import date
    wanted = None if symbols is None else {str(s).upper() for s in symbols}
    with locked_ledger(ledger_path()) as (ledger, rows, _):
        now = str(today or datetime.now(timezone.utc).date().isoformat())[:10]
        out = {"credited": 0, "expired": 0, "still_unverifiable": 0, "dropped_unverifiable": 0}
        open_waits = [r for r in rows if not r.get("classification")
                      and (wanted is None or str(r.get("symbol") or "").upper() in wanted)]
        claimed = _claimed_good_wait_units(path) if open_waits else {}
        rewritten: list[dict[str, Any]] = []
        changed = False
        credited: dict[str, str] = {}          # unit key -> fill, for the units credited in this pass
        siblings: list[int] = []               # unmarked copies of a unit; finished after the sweep
        for row in rows:
            if row.get("classification") or (wanted is not None
                                             and str(row.get("symbol") or "").upper() not in wanted):
                rewritten.append(row)
                continue
            keyed = counterfactual_observation_key(row)
            is_wait = str(row.get("decision") or "").upper() == "WAIT"
            adopted = claimed.get(keyed[0]) if keyed is not None and is_wait else None
            pending = row.get("later_entry_pending")
            if not isinstance(pending, Mapping):
                if not adopted:
                    if keyed is not None and is_wait:
                        siblings.append(len(rewritten))
                    rewritten.append(row)
                    continue
                # A credit claimed before its fill was marked: persist the marker so no
                # price path can settle the row while the credit is finished.
                pending = {"fill_session": adopted, "recovered_from": "observation_unit"}
                row = {**row, "later_entry_pending": pending}
                changed = True
            fill = adopted or str(pending.get("fill_session") or "")[:10]
            state = "LIVE" if adopted else _wait_ordinal_state(row, fill)
            if state == "LIVE":
                settled = settle({k: v for k, v in row.items() if k != "later_entry_pending"},
                                 forward_return_pct=None, later_entered=True)
                settled["later_entry_verified"] = fill
                try:
                    applied = ingest_counterfactual(settled, path=path, floors=floors)
                    superseded = (_unit_classification(keyed[0], path)
                                  if keyed is not None and isinstance(applied, Mapping)
                                  and applied.get("duplicate_observation") else GOOD_WAIT)
                except Exception as exc:
                    # This row stays queued for the next pass; the others are still
                    # verified, expired or dropped and persisted (QT-C2-08).
                    out.setdefault("errors", []).append(f"{row.get('symbol')}: {type(exc).__name__}: {exc}"[:200])
                    rewritten.append(row)
                    continue
                if superseded and superseded != GOOD_WAIT:
                    # Another settlement decided this unit first (cycle-9 review A2): the row
                    # never claims GOOD_WAIT beside it; it settles like its unit, on price.
                    out["superseded"] = out.get("superseded", 0) + 1
                    kept = {k: v for k, v in row.items() if k != "later_entry_pending"}
                    kept["later_entry_superseded_by"] = superseded
                    rewritten.append(kept)
                    changed = True
                    continue
                rewritten.append(settled)
                out["credited"] += 1
                if keyed is not None:
                    credited.setdefault(keyed[0], fill)
                changed = True
                continue
            if state == "UNVERIFIABLE":
                try:
                    age = (date.fromisoformat(now) - date.fromisoformat(fill)).days
                except ValueError:
                    age = LATER_ENTRY_VERIFY_MAX_DAYS + 1
                if age <= LATER_ENTRY_VERIFY_MAX_DAYS:
                    out["still_unverifiable"] += 1
                    rewritten.append(row)
                    continue
                out["dropped_unverifiable"] += 1
            else:
                out["expired"] += 1
            kept = {k: v for k, v in row.items() if k != "later_entry_pending"}
            kept["later_entry_unverified" if state == "UNVERIFIABLE" else "later_entry_expired"] = fill
            if keyed is not None and is_wait:
                siblings.append(len(rewritten))      # its unit may still be credited by another copy (A3)
            rewritten.append(kept)
            changed = True
        # Another copy of a unit credited in this sweep (one cycle's re-freeze of the same
        # day, symbol and reason, unmarked or with a fill of its own that expired) carries
        # the unit's label and verified fill, never a price-path one beside it (review A3).
        for i in siblings:
            row = rewritten[i]
            keyed = counterfactual_observation_key(row)
            fill = (credited.get(keyed[0]) or claimed.get(keyed[0])) if keyed is not None else None
            if not fill:
                continue
            row = {k: v for k, v in row.items() if k not in ("later_entry_expired", "later_entry_unverified")}
            settled = settle(row, forward_return_pct=None, later_entered=True)
            settled["later_entry_verified"] = fill
            try:
                ingest_counterfactual(settled, path=path, floors=floors)    # the unit is applied: a no-op
            except Exception as exc:
                out.setdefault("errors", []).append(f"{row.get('symbol')}: {type(exc).__name__}: {exc}"[:200])
                rewritten[i] = {**row, "later_entry_pending": {"fill_session": fill,
                                                               "recovered_from": "observation_unit"}}
            else:
                rewritten[i] = settled
            changed = True
        if changed:
            from core.atomic_io import atomic_write_text
            atomic_write_text(ledger, "".join(json.dumps(r, default=str) + "\n" for r in rewritten), durable=True)
        return out


def _sessions_through(day: str, entry_day: str) -> int | None:
    """Official sessions in (day, entry_day], counted THROUGH the fill session; None when unverifiable.

    The archive ends at the last completed session, so an intraday fill's own
    session is not in it yet: the sessions after the archive come from the
    verified holiday calendar, and a year it does not cover is unverifiable
    (overnight review L-08). Without an archive nothing is verifiable.
    """
    from datetime import date, timedelta
    try:
        from core.session_window import official_dates
        sessions = list(official_dates() or [])
    except Exception:
        return None
    if not sessions or day < sessions[0]:
        return None
    n = sum(1 for d in sessions if day < d <= entry_day)
    last = sessions[-1]
    if entry_day > last:
        try:
            from research.intelligence.data.nse_calendar import covered_years, is_session, load_holidays
            holidays = load_holidays()
            years = covered_years(holidays)
        except Exception:
            return None
        cur, end = date.fromisoformat(last) + timedelta(days=1), date.fromisoformat(entry_day)
        while cur <= end:
            if cur.year not in years:
                return None
            if is_session(cur, holidays):
                n += 1
            cur += timedelta(days=1)
    return n


def note_later_entry(symbol: str, *, path=None, floors: Mapping[str, int] | None = None,
                     as_of: str | None = None) -> dict[str, Any]:
    """A waited name later received a valid fill → GOOD_WAIT. Not booked as extra P&L.

    Only the symbol's pending WAIT decisions whose horizon still covers the
    entry day are credited; other pending rows keep waiting for their own
    rulebook outcome. The fill is marked durably first and credited from the
    mark (lifecycle audit F1): a crash leaves it to the next settlement.
    """
    entry_day = str(as_of or datetime.now(timezone.utc).date().isoformat())[:10]
    symbol = str(symbol).upper()
    queued = _queue_later_fills(symbol, entry_day)
    out = retry_pending_good_waits(path=path, floors=floors, today=entry_day, symbols={symbol})
    credited = int(out.get("credited") or 0)
    result: dict[str, Any] = {"updated": credited, "classifications": {GOOD_WAIT: credited} if credited else {},
                              "queued": queued}
    if out.get("errors"):
        result["errors"] = list(out["errors"])
        from product.counterfactual_learning import note_recording_failure
        note_recording_failure("later-entry credit (kept queued; the next settlement retries it)", symbol,
                               RuntimeError("; ".join(out["errors"])))
    return result


def settle_pending_counterfactuals(
    *,
    forward_return_by_symbol: Mapping[str, float] | None = None,
    later_entered: Mapping[str, bool] | None = None,
    path=None,
    floors: Mapping[str, int] | None = None,
    row_filter=None,
    verified_fill: str | None = None,
) -> dict[str, Any]:
    """Classify frozen rejects once a forward return is known. Does not create P&L.

    ``later_entered`` alone never makes a GOOD_WAIT: the caller must also name the
    ``verified_fill`` session it checked each row against (note_later_entry), and
    the settled row carries it (cycle-2 review QT-C2-07).
    """
    with locked_ledger(ledger_path()) as (ledger, rows, _):
        returns = {str(k).upper(): float(v) for k, v in dict(forward_return_by_symbol or {}).items()}
        entered = {str(k).upper(): bool(v) for k, v in dict(later_entered or {}).items()}
        updated = 0
        classifications: dict[str, int] = {}
        rewritten: list[dict[str, Any]] = []
        queued_units = {keyed[0] for r in rows if r.get("later_entry_pending") and not r.get("classification")
                        for keyed in [counterfactual_observation_key(r)] if keyed is not None}
        for row in rows:
            keyed = counterfactual_observation_key(row)
            if (row.get("classification") or row.get("later_entry_pending")
                    or (keyed is not None and keyed[0] in queued_units)):
                # A queued later fill is credited by retry_pending_good_waits only; a
                # price outcome here would erase the GOOD_WAIT it may have earned (QT-07).
                rewritten.append(row)
                continue
            symbol = str(row.get("symbol") or "").upper()
            if (symbol not in returns and symbol not in entered) or (row_filter is not None and not row_filter(row)):
                rewritten.append(row)
                continue
            later = bool(entered.get(symbol, False) and verified_fill)
            settled = settle(
                row,
                forward_return_pct=returns.get(symbol),
                later_entered=later,
            )
            if later:
                settled["later_entry_verified"] = str(verified_fill)[:10]
            ingest_counterfactual(settled, path=path, floors=floors)
            rewritten.append(settled)
            updated += 1
            cls = str(settled.get("classification") or FLAT)
            classifications[cls] = classifications.get(cls, 0) + 1
        if updated:
            from core.atomic_io import atomic_write_text
            atomic_write_text(ledger, "".join(json.dumps(r, default=str) + "\n" for r in rewritten), durable=True)
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
    out = {
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
        "autonomous_learning": None,
        "challenger_learning": None,
    }
    try:
        from product.autonomous_learning import dashboard as autonomous_dashboard
        out["autonomous_learning"] = autonomous_dashboard()
    except Exception as exc:
        out["autonomous_learning"] = {"available": False, "error": str(exc)[:200]}
    try:
        from product.challenger_learning import dashboard as challenger_dashboard
        out["challenger_learning"] = challenger_dashboard()
    except Exception as exc:
        out["challenger_learning"] = {"available": False, "error": str(exc)[:200]}
    return out
