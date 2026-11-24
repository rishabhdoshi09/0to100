"""Versioned, explicit learning-policy layer.

Policies are never silent weight mutations. Each rule has a status ladder:

  OBSERVING → EXPERIMENTAL → ELIGIBLE → ACTIVE
  and may be DEMOTED or REJECTED.

A policy can SUPPORT / NEUTRAL / PENALIZE / BLOCK a candidate. It cannot invent
a BUY. Insufficient sample returns INSUFFICIENT_EVIDENCE rather than a fake edge.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from core.runtime_paths import logs_dir

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = logs_dir() / "product" / "learning_policies.json"
SCHEMA_VERSION = 1

OBSERVING = "OBSERVING"
EXPERIMENTAL = "EXPERIMENTAL"
ELIGIBLE = "ELIGIBLE"
ACTIVE = "ACTIVE"
DEMOTED = "DEMOTED"
REJECTED = "REJECTED"

STATUSES = (OBSERVING, EXPERIMENTAL, ELIGIBLE, ACTIVE, DEMOTED, REJECTED)

# Conservative defaults — tests may inject smaller floors to prove the handoff.
MIN_SAMPLE_EXPERIMENTAL = 8
MIN_SAMPLE_ELIGIBLE = 20
MIN_SAMPLE_ACTIVE = 30
MIN_ABS_EDGE_R = 0.25
SHRINKAGE_K = 8.0
# Two-way conditionals need more samples to avoid combinatorial overfitting.
CONDITIONAL_SAMPLE_MULT = 2
# Session diversity. A large n from few sessions is one market path counted
# many times: outcomes are measured over a 5-session horizon, so decisions made
# within the same 5 sessions share most of their forward move. 20 distinct
# decision sessions ~ 4 non-overlapping horizons before a policy may influence
# paper selection (ELIGIBLE); 40 ~ 8, matching the 40-trading-day floor of the
# forward-soak live-readiness contract, before it may BLOCK (ACTIVE). Below the
# floor a policy stays EXPERIMENTAL: recorded, shown, never acted on.
MIN_SESSIONS_ELIGIBLE = 20
MIN_SESSIONS_ACTIVE = 40


def _today_ist() -> str:
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()


def policy_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_LEARNING_POLICIES")
    if override:
        return Path(override)
    return DEFAULT_PATH


def empty_store() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": "",
        "policies": [],
        "live_locked": True,
        "note": (
            "Policies affect paper selection only after sample floors. "
            "They never invent a BUY and cannot enable live money."
        ),
    }


def load_policies(path: str | Path | None = None) -> dict[str, Any]:
    target = policy_path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return empty_store()
    if not isinstance(payload, dict):
        return empty_store()
    payload.setdefault("policies", [])
    payload["live_locked"] = True
    return payload


def save_policies(payload: Mapping[str, Any], path: str | Path | None = None) -> Path:
    target = policy_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["schema_version"] = SCHEMA_VERSION
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    data["live_locked"] = True
    from core.atomic_io import atomic_write_text
    # durable: a barrier between the learning claim before it and the ledger rewrite after it
    atomic_write_text(target, json.dumps(data, indent=2, default=str), durable=True)
    return target


def _status_for(
    n: int,
    abs_edge: float,
    *,
    floors: Mapping[str, int] | None = None,
    source: str = "",
    conditional: bool = False,
    sessions: int | None = None,
) -> str:
    floors = dict(floors or {})
    n_exp = int(floors.get("experimental", MIN_SAMPLE_EXPERIMENTAL))
    n_eli = int(floors.get("eligible", MIN_SAMPLE_ELIGIBLE))
    n_act = int(floors.get("active", MIN_SAMPLE_ACTIVE))
    if conditional:
        n_exp *= int(floors.get("conditional_mult", CONDITIONAL_SAMPLE_MULT))
        n_eli *= int(floors.get("conditional_mult", CONDITIONAL_SAMPLE_MULT))
        n_act *= int(floors.get("conditional_mult", CONDITIONAL_SAMPLE_MULT))
    if n < n_exp:
        return OBSERVING
    if abs_edge < MIN_ABS_EDGE_R:
        return REJECTED if n >= n_eli else OBSERVING
    # Historical/replay/simulation-only evidence can generate hypotheses,
    # but it cannot become production-selection authority on its own. Keep the
    # source classification broad so a renamed historical lane (for example
    # historical_replay or decision_simulation) cannot accidentally promote
    # merely by accumulating samples.
    source_key = str(source or "").strip().lower()
    historical_only = any(
        token in source_key
        for token in ("backtest", "historical", "replay", "simulation")
    )
    if historical_only:
        if n >= n_eli:
            return EXPERIMENTAL
        return OBSERVING
    status = ACTIVE if n >= n_act else ELIGIBLE if n >= n_eli else EXPERIMENTAL
    # Forward evidence must also span enough distinct sessions. Unknown session
    # coverage (None) is treated as zero: fail closed, never promote on count.
    s_eli = int(floors.get("sessions_eligible", MIN_SESSIONS_ELIGIBLE))
    s_act = int(floors.get("sessions_active", MIN_SESSIONS_ACTIVE))
    covered = int(sessions or 0)
    if status == ACTIVE and covered < s_act:
        status = ELIGIBLE
    if status == ELIGIBLE and covered < s_eli:
        status = EXPERIMENTAL
    return status


def upsert_policy(
    *,
    policy_id: str,
    dimension: str,
    bucket: str,
    sample_size: int,
    expectancy_R: float,
    baseline_R: float = 0.0,
    source: str = "paper_forward",
    path: str | Path | None = None,
    floors: Mapping[str, int] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create or version a policy from measured evidence. Never silent."""
    store = load_policies(path)
    policies = [dict(p) for p in (store.get("policies") or []) if isinstance(p, Mapping)]
    edge = float(expectancy_R) - float(baseline_R)
    k = float((extra or {}).get("shrinkage_k") or SHRINKAGE_K)
    n = max(1, int(sample_size))
    shrunk = (n / (n + k)) * edge
    conditional = "|" in str(dimension)
    extra_payload = dict(extra or {})
    sessions_raw = extra_payload.pop("sessions_observed", None)
    sessions = None if sessions_raw is None else int(sessions_raw)
    status = _status_for(
        int(sample_size), abs(shrunk), floors=floors, source=source, conditional=conditional,
        sessions=sessions,
    )
    existing = next((p for p in policies if p.get("policy_id") == policy_id), None)
    version = int((existing or {}).get("version") or 0) + 1
    first_seen = str((existing or {}).get("first_seen") or datetime.now(timezone.utc).isoformat())
    regimes = list((existing or {}).get("regimes_tested") or [])
    regime = str(extra_payload.pop("regime", "") or "")
    if regime and regime not in regimes:
        regimes.append(regime)
    affects = extra_payload.pop("affects_selection", True)
    row = {
        "policy_id": policy_id,
        "version": version,
        "dimension": dimension,
        "bucket": bucket,
        "sample_size": int(sample_size),
        "expectancy_R": round(float(expectancy_R), 4),
        "baseline_R": round(float(baseline_R), 4),
        "expectancy_difference_R": round(edge, 4),
        "shrunk_expectancy_R": round(shrunk, 4),
        "shrinkage_k": k,
        "confidence": (
            "INSUFFICIENT_EVIDENCE" if status in {OBSERVING, EXPERIMENTAL}
            else "MEASURED"
        ),
        "production_status": status,
        "evidence_source": source,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "first_seen": first_seen,
        "date_range": {
            "first": first_seen,
            "last": datetime.now(timezone.utc).isoformat(),
        },
        "regimes_tested": regimes,
        "sessions_observed": sessions,
        "affects_selection": bool(affects),
        "live_locked": True,
        **extra_payload,
    }
    policies = [p for p in policies if p.get("policy_id") != policy_id]
    policies.append(row)
    store["policies"] = sorted(policies, key=lambda p: str(p.get("policy_id") or ""))
    save_policies(store, path)
    return row


def record_measured_outcome(
    *,
    policy_id: str,
    dimension: str,
    bucket: str,
    realized_R: float,
    source: str = "paper_forward",
    path: str | Path | None = None,
    floors: Mapping[str, int] | None = None,
    extra: Mapping[str, Any] | None = None,
    observation_key: str | None = None,
    session: str | None = None,
) -> dict[str, Any]:
    """Increment a policy from one settled observation. Never silent; never invents BUY.

    ``observation_key`` names the independent unit this observation is (for a
    counterfactual: decision day, symbol, reason). A key already recorded for
    this policy is a duplicate and changes nothing. Without a key the call is
    counted but cannot be deduplicated. ``session`` (the decision session,
    YYYY-MM-DD) feeds the session-diversity floor; it defaults to today's IST
    date, which can only under-state diversity when a backlog settles at once.

    Gross-only positive evidence is audit-only: without execution-adjusted evidence
    it cannot SUPPORT or promote a setup. Gross-only *negative* evidence is a
    conservative upper bound, so after sample/shrinkage gates it may only act as
    a veto/penalty. Realistic costs cannot turn an already-negative gross edge into
    a better conservative edge. This keeps old downside protection while preventing
    optimistic gross P&L from being mistaken for tradable alpha.
    """
    from product import learning_observations as LO

    ledger = LO.ledger_path_for(policy_path(path))
    key = str(observation_key or f"unkeyed:{uuid.uuid4().hex}")
    session_day = str(session or _today_ist())[:10]
    args = {"policy_id": policy_id, "dimension": dimension, "bucket": bucket,
            "realized_R": float(realized_R), "source": source, "extra": dict(extra or {})}
    # Exactly once (cross-review A-3): a TWO-PHASE claim carries everything the
    # apply needs, under one cross-process writer lock. A crash between the
    # claim and the policy write leaves a CLAIMED row that recover_claims()
    # replays; it no longer silently becomes a permanent duplicate.
    with LO.writer_lock(ledger):
        _recover_locked(path, floors, ledger)
        if not LO.claim(ledger, policy_id, key, session_day, payload=args):
            store = load_policies(path)
            existing = next((dict(p) for p in (store.get("policies") or [])
                             if p.get("policy_id") == policy_id), None)
            return {**(existing or {"policy_id": policy_id}), "duplicate_observation": True,
                    "observation_key": key}
        out = _apply_measured(path=path, floors=floors, ledger=ledger, observation_key=key, **args)
        LO.mark_applied(ledger, policy_id, key)
        return out


def update_policies_locked(mutate, path: str | Path | None = None) -> Path:
    """Read-modify-write the policy store under the learning writer lock (overnight review L-04).

    Every writer of the live store goes through the lock record_measured_outcome
    holds across its claim → write → mark, so no writer can replace the file
    with a copy read before another's committed update. ``mutate(store)``
    returns the new store; it is read fresh INSIDE the lock. Not re-entrant.
    """
    from product import learning_observations as LO

    target = policy_path(path)
    with LO.writer_lock(LO.ledger_path_for(target)):
        return save_policies(mutate(load_policies(target)), target)


def locked_upsert_policy(**kwargs) -> dict[str, Any]:
    """upsert_policy under the learning writer lock, for writers outside record_measured_outcome (L-04)."""
    from product import learning_observations as LO

    with LO.writer_lock(LO.ledger_path_for(policy_path(kwargs.get("path")))):
        return upsert_policy(**kwargs)


def recover_claims(path: str | Path | None = None, floors: Mapping[str, int] | None = None) -> dict[str, int]:
    """Replay policy claims a crash left CLAIMED (applied exactly once; idempotent)."""
    from product import learning_observations as LO

    ledger = LO.ledger_path_for(policy_path(path))
    if not ledger.exists():
        return {"replayed": 0, "confirmed": 0, "corrupt": 0}
    with LO.writer_lock(ledger):
        return _recover_locked(path, floors, ledger)


_CLAIM_FIELDS = ("policy_id", "dimension", "bucket", "realized_R", "source", "extra")


def _valid_policy_claim(scope: str, payload: Any) -> bool:
    """A replayable policy claim: every field present and typed, and its policy IS the claim's scope."""
    if not isinstance(payload, dict) or any(k not in payload for k in _CLAIM_FIELDS):
        return False
    if str(payload.get("policy_id") or "") != str(scope or "") or not str(scope or ""):
        return False
    try:
        r = float(payload["realized_R"])
    except (TypeError, ValueError):
        return False
    return (r == r and abs(r) != float("inf") and isinstance(payload.get("extra"), dict)
            and all(isinstance(payload.get(k), str) for k in ("dimension", "bucket", "source")))


def _recover_locked(path, floors, ledger) -> dict[str, int]:
    from product import learning_observations as LO

    out = {"replayed": 0, "confirmed": 0, "corrupt": 0}
    for row in LO.claimed_unapplied(ledger):
        payload = row.get("payload")
        if isinstance(payload, dict) and "policy_id" not in payload and payload.get("kind"):
            continue                           # not a policy claim (e.g. a whole-ingest claim)
        if not _valid_policy_claim(row["scope"], payload):
            # Malformed, or a payload for ANOTHER policy than the claim's scope:
            # applying it would bypass that policy's own claim and count one
            # observation twice (overnight review L-05). It stays CLAIMED and
            # visible, never guessed, and never blocks the other claims.
            out["corrupt"] += 1
            continue
        store = load_policies(path)
        existing = next((dict(p) for p in (store.get("policies") or [])
                         if p.get("policy_id") == row["scope"]), None)
        if existing and str(existing.get("last_observation_key") or "") == str(row["observation_key"]):
            out["confirmed"] += 1              # the apply landed before the crash
        else:
            _apply_measured(path=path, floors=floors, ledger=ledger, observation_key=row["observation_key"],
                            **{k: payload[k] for k in ("policy_id", "dimension", "bucket", "realized_R",
                                                       "source", "extra")})
            out["replayed"] += 1
        LO.mark_applied(ledger, row["scope"], row["observation_key"])
    return out


def _apply_measured(*, path, floors, ledger, observation_key: str, policy_id: str, dimension: str, bucket: str,
                    realized_R: float, source: str, extra: Mapping[str, Any] | None) -> dict[str, Any]:
    """One increment of a policy, stamped with the observation it applies (atomic JSON write)."""
    from product import learning_observations as LO

    store = load_policies(path)
    existing = next(
        (dict(p) for p in (store.get("policies") or []) if p.get("policy_id") == policy_id),
        None,
    )
    sessions = LO.distinct_sessions(ledger, policy_id)
    n_old = int((existing or {}).get("sample_size") or 0)
    mean_old = float((existing or {}).get("expectancy_R") or 0.0)
    n = n_old + 1
    mean = mean_old + (float(realized_R) - mean_old) / n
    payload = dict(extra or {})
    payload["last_observation_R"] = round(float(realized_R), 4)
    payload["last_observation_key"] = str(observation_key)   # the replay's "did it land?" marker
    payload["sessions_observed"] = sessions
    if str(source) == "paper_forward_taken_gross_only":
        if mean < 0.0:
            # Veto-only asymmetry: negative gross evidence is already an upper
            # bound on a cost-adjusted result. It may protect capital, never
            # create/support a BUY.
            payload["affects_selection"] = True
            payload["gross_only_veto_only"] = True
            payload["evidence_only_reason"] = "NEGATIVE_GROSS_CONSERVATIVE_BOUND"
        else:
            payload["affects_selection"] = False
            payload["gross_only_veto_only"] = True
            payload["evidence_only_reason"] = "EXECUTION_ADJUSTED_UNAVAILABLE"
    return upsert_policy(
        policy_id=policy_id,
        dimension=dimension,
        bucket=bucket,
        sample_size=n,
        expectancy_R=mean,
        baseline_R=float((existing or {}).get("baseline_R") or 0.0),
        source=source,
        path=path,
        floors=floors,
        extra=payload,
    )


def active_policies(path: str | Path | None = None) -> list[dict[str, Any]]:
    store = load_policies(path)
    return [
        dict(p) for p in (store.get("policies") or [])
        if str(p.get("production_status") or "") in {ACTIVE, ELIGIBLE, EXPERIMENTAL}
    ]
