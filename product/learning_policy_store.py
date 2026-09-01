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


def _status_for(n: int, abs_edge: float, *, floors: Mapping[str, int] | None = None) -> str:
    floors = dict(floors or {})
    n_exp = int(floors.get("experimental", MIN_SAMPLE_EXPERIMENTAL))
    n_eli = int(floors.get("eligible", MIN_SAMPLE_ELIGIBLE))
    n_act = int(floors.get("active", MIN_SAMPLE_ACTIVE))
    if n < n_exp:
        return OBSERVING
    if abs_edge < MIN_ABS_EDGE_R:
        return REJECTED if n >= n_eli else OBSERVING
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
    status = _status_for(int(sample_size), abs(edge), floors=floors)
    existing = next((p for p in policies if p.get("policy_id") == policy_id), None)
    version = int((existing or {}).get("version") or 0) + 1
    row = {
        "policy_id": policy_id,
        "version": version,
        "dimension": dimension,
        "bucket": bucket,
        "sample_size": int(sample_size),
        "expectancy_R": round(float(expectancy_R), 4),
        "baseline_R": round(float(baseline_R), 4),
        "expectancy_difference_R": round(edge, 4),
        "confidence": (
            "INSUFFICIENT_EVIDENCE" if status in {OBSERVING, EXPERIMENTAL}
            else "MEASURED"
        ),
        "production_status": status,
        "evidence_source": source,
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "live_locked": True,
        **dict(extra or {}),
    }
    policies = [p for p in policies if p.get("policy_id") != policy_id]
    policies.append(row)
    store["policies"] = sorted(policies, key=lambda p: str(p.get("policy_id") or ""))
    save_policies(store, path)
    return row


def active_policies(path: str | Path | None = None) -> list[dict[str, Any]]:
    store = load_policies(path)
    return [
        dict(p) for p in (store.get("policies") or [])
        if str(p.get("production_status") or "") in {ACTIVE, ELIGIBLE, EXPERIMENTAL}
    ]
