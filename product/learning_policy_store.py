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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "logs" / "product" / "learning_policies.json"
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
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return target


def _status_for(
    n: int,
    abs_edge: float,
    *,
    floors: Mapping[str, int] | None = None,
    source: str = "",
    conditional: bool = False,
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
    # Historical backtest can become EXPERIMENTAL, never ACTIVE on its own.
    if str(source).startswith("backtest"):
        if n >= n_eli:
            return EXPERIMENTAL
        return OBSERVING
    if n >= n_act:
        return ACTIVE
    if n >= n_eli:
        return ELIGIBLE
    return EXPERIMENTAL


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
    status = _status_for(
        int(sample_size), abs(shrunk), floors=floors, source=source, conditional=conditional,
    )
    existing = next((p for p in policies if p.get("policy_id") == policy_id), None)
    version = int((existing or {}).get("version") or 0) + 1
    first_seen = str((existing or {}).get("first_seen") or datetime.now(timezone.utc).isoformat())
    regimes = list((existing or {}).get("regimes_tested") or [])
    extra_payload = dict(extra or {})
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
) -> dict[str, Any]:
    """Increment a policy from one settled observation. Never silent; never invents BUY.

    Gross-only positive evidence is audit-only: without execution-adjusted evidence
    it cannot SUPPORT or promote a setup. Gross-only *negative* evidence is a
    conservative upper bound, so after sample/shrinkage gates it may only act as
    a veto/penalty. Realistic costs cannot turn an already-negative gross edge into
    a better conservative edge. This keeps old downside protection while preventing
    optimistic gross P&L from being mistaken for tradable alpha.
    """
    store = load_policies(path)
    existing = next(
        (dict(p) for p in (store.get("policies") or []) if p.get("policy_id") == policy_id),
        None,
    )
    n_old = int((existing or {}).get("sample_size") or 0)
    mean_old = float((existing or {}).get("expectancy_R") or 0.0)
    n = n_old + 1
    mean = mean_old + (float(realized_R) - mean_old) / n
    payload = dict(extra or {})
    payload["last_observation_R"] = round(float(realized_R), 4)
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
