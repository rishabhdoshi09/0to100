"""Durable Policy Registry for the Evolution Engine.

A "policy" here is a whole decision PROCESS -- a named, bounded, interpretable
variant of QuantTerm's real scoring (see policy_eval.py) -- not a single
evidence rule. Exactly one policy per domain may hold CHAMPION status at a
time; that is the only policy allowed to create real PAPER positions
(enforced in policy_eval.py / tournament.py, not here).

Every policy version is immutable once registered: re-registering the same
policy_id with different weights/hypothesis is a collision, not an update.
A change to a policy is a NEW policy_id (bump the version suffix), exactly
like product/decision_freeze.py refuses to silently rewrite a frozen
decision. Status transitions (CHALLENGER -> PROBATION -> CHAMPION -> ...)
are the only thing that may change after registration, and every transition
is appended to the policy's own history rather than overwriting it.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from core.runtime_paths import logs_dir

SCHEMA_VERSION = 1

# ── lifecycle status ladder (exact vocabulary from the product brief) ───────
CHAMPION = "CHAMPION"
CHALLENGER = "CHALLENGER"
SHADOW = "SHADOW"
PROBATION = "PROBATION"
RETIRED = "RETIRED"
REJECTED = "REJECTED"
STATUSES = (CHAMPION, CHALLENGER, SHADOW, PROBATION, RETIRED, REJECTED)

# ── domains: shared tournament engine, domain-specific policy manifests ────
EQUITY = "EQUITY"
FNO_UNDERLYING = "FNO_UNDERLYING"
FNO_CONTRACT = "FNO_CONTRACT"
DOMAINS = (EQUITY, FNO_UNDERLYING, FNO_CONTRACT)

_BASELINE_IDS = {
    EQUITY: "EQUITY_CHAMPION_BASELINE_V1",
    FNO_UNDERLYING: "FNO_UNDERLYING_CHAMPION_BASELINE_V1",
    FNO_CONTRACT: "FNO_CONTRACT_CHAMPION_BASELINE_V1",
}

_SEEDED_CHALLENGERS = {
    EQUITY: (
        (
            "EQUITY_RS_HEAVY_V1",
            "Test whether giving relative-strength leadership moderately more weight improves paired forward outcomes.",
            {"relative_strength_mult": 1.25},
        ),
        (
            "EQUITY_SECTOR_STRONG_V1",
            "Test whether stronger sector confirmation reduces false positives without starving opportunity capture.",
            {"sector_confirmation_bonus": 2.0},
        ),
        (
            "EQUITY_VOLUME_CONFIRM_V1",
            "Test whether stronger volume/liquidity confirmation improves forward trade quality.",
            {"volume_confirmation_bonus": 2.0},
        ),
        (
            "EQUITY_EVIDENCE_CONSERVATIVE_V1",
            "Test whether requiring a larger empirical sample before selection improves robustness.",
            {"min_empirical_sample": 30.0},
        ),
        (
            "EQUITY_REGIME_DEFENSIVE_V1",
            "Test whether stronger risk-off regime penalties reduce avoidable losses.",
            {"regime_standdown_mult": 1.5},
        ),
    ),
    FNO_UNDERLYING: (
        (
            "FNO_UNDERLYING_OI_HEAVY_V1",
            "Test whether stronger futures-OI confirmation improves directional F&O selection.",
            {"oi_confirmation_mult": 1.25},
        ),
        (
            "FNO_UNDERLYING_SECTOR_HEAVY_V1",
            "Test whether stronger sector/NIFTY confirmation improves F&O directional selection.",
            {"sector_confirmation_mult": 1.25},
        ),
        (
            "FNO_UNDERLYING_EXTENSION_DEFENSIVE_V1",
            "Test whether stronger extension penalties reduce failed F&O breakouts and breakdowns.",
            {"extension_penalty_mult": 1.25},
        ),
    ),
    FNO_CONTRACT: (
        (
            "FNO_CONTRACT_DELTA_CORE_V1",
            "Test a mild preference for liquid 0.55-0.70 delta contracts among already-eligible options.",
            {"delta_preference_mult": 1.15},
        ),
        (
            "FNO_CONTRACT_LIQUIDITY_HEAVY_V1",
            "Test stronger spread/OI/volume emphasis among already-eligible option contracts.",
            {"liquidity_mult": 1.25},
        ),
        (
            "FNO_CONTRACT_THETA_DEFENSIVE_V1",
            "Test stronger theta/DTE protection among already-eligible option contracts.",
            {"theta_penalty_mult": 1.25},
        ),
    ),
}


def policy_manifest_fingerprint(policy: Mapping[str, Any]) -> str:
    """Stable fingerprint of the immutable policy manifest."""
    material = json.dumps(
        _manifest_identity(policy), sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def ensure_seed_population(
    domain: str, *, path: str | Path | None = None,
) -> dict[str, Any]:
    """Idempotently ensure one Champion and a bounded, interpretable seed
    population. No random search, no duplicate versions on restart.

    Existing promoted Champions are preserved. Seed policies are only
    registered if their immutable IDs do not already exist.
    """
    if domain not in DOMAINS:
        raise ValueError(f"unknown policy domain {domain!r}, expected one of {DOMAINS}")
    champion = current_champion(domain, path=path)
    if champion is None:
        baseline_id = _BASELINE_IDS[domain]
        champion = register_policy(
            policy_id=baseline_id,
            domain=domain,
            hypothesis=(
                "Baseline Champion: current canonical production behavior with "
                "neutral Evolution weights; reference for paired Challenger evidence."
            ),
            weights={},
            status=CHAMPION,
            reason_created="bootstrap: no champion existed yet",
            path=path,
        )

    seeded: list[dict[str, Any]] = []
    for policy_id, hypothesis, weights in _SEEDED_CHALLENGERS.get(domain, ()):
        existing = get_policy(policy_id, path=path)
        if existing is None:
            existing = register_policy(
                policy_id=policy_id,
                domain=domain,
                hypothesis=hypothesis,
                weights=weights,
                parent_policy_id=champion["policy_id"],
                status=CHALLENGER,
                reason_created="deterministic initial Evolution seed",
                path=path,
            )
        seeded.append(existing)
    return {"champion": champion, "challengers": seeded}


class PolicyIdentityCollision(RuntimeError):
    """The same policy_id was re-registered with different manifest content."""


class MultipleChampionsError(RuntimeError):
    """A domain may have at most one CHAMPION at a time."""


def registry_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_EVOLUTION_POLICY_REGISTRY")
    if override:
        return Path(override)
    # Resolved fresh on every call (never cached as a module-level constant)
    # so QT_RUNTIME_ROOT redirection -- the whole point of core.runtime_paths
    # -- actually takes effect for whichever root is active right now.
    return logs_dir() / "product" / "evolution_policy_registry.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_store() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": "",
        "policies": {},
        "live_locked": True,
        "note": (
            "Policies are PAPER decision-process variants only. Only the "
            "domain's current CHAMPION may reach real PAPER execution. "
            "This registry cannot authorize live money."
        ),
    }


def load_registry(path: str | Path | None = None) -> dict[str, Any]:
    target = registry_path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return _empty_store()
    if not isinstance(payload, dict) or not isinstance(payload.get("policies"), dict):
        return _empty_store()
    payload["live_locked"] = True
    return payload


def save_registry(payload: Mapping[str, Any], path: str | Path | None = None) -> Path:
    target = registry_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["live_locked"] = True
    body["updated_at"] = _now()
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(body, indent=2, default=str, sort_keys=True), encoding="utf-8")
    tmp.replace(target)
    return target


def _manifest_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a policy that must never change once registered."""
    return {
        "policy_id": str(manifest.get("policy_id") or ""),
        "domain": str(manifest.get("domain") or ""),
        "version": int(manifest.get("version") or 1),
        "parent_policy_id": manifest.get("parent_policy_id"),
        "hypothesis": str(manifest.get("hypothesis") or ""),
        "weights": dict(manifest.get("weights") or {}),
        "code_build_sha": str(manifest.get("code_build_sha") or ""),
    }


def register_policy(
    *,
    policy_id: str,
    domain: str,
    hypothesis: str,
    weights: Mapping[str, float],
    parent_policy_id: str | None = None,
    status: str = SHADOW,
    reason_created: str = "",
    version: int = 1,
    code_build_sha: str = "",
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Register a new, immutable policy version, or return the existing one
    if this exact (policy_id, manifest) was already registered. Raises
    PolicyIdentityCollision if policy_id already exists with DIFFERENT
    manifest content -- a real change must use a new policy_id/version.
    """
    if domain not in DOMAINS:
        raise ValueError(f"unknown policy domain {domain!r}, expected one of {DOMAINS}")
    if status not in STATUSES:
        raise ValueError(f"unknown policy status {status!r}, expected one of {STATUSES}")
    if not hypothesis.strip():
        raise ValueError(
            "every policy must state the hypothesis it tests "
            "(\"what specific hypothesis is this variant testing?\")"
        )

    store = load_registry(path)
    manifest = {
        "policy_id": policy_id,
        "domain": domain,
        "version": int(version),
        "parent_policy_id": parent_policy_id,
        "hypothesis": hypothesis,
        "weights": dict(weights),
        "code_build_sha": code_build_sha,
    }
    identity = _manifest_identity(manifest)

    existing = store["policies"].get(policy_id)
    if existing is not None:
        if _manifest_identity(existing) != identity:
            raise PolicyIdentityCollision(
                f"policy {policy_id!r} already registered with different "
                "manifest content -- changes require a new policy_id/version"
            )
        return existing

    if status == CHAMPION:
        current = current_champion(domain, path=path)
        if current is not None and current["policy_id"] != policy_id:
            raise MultipleChampionsError(
                f"domain {domain} already has CHAMPION {current['policy_id']!r}; "
                "demote it before registering a new CHAMPION"
            )

    record = {
        **manifest,
        "created_at": _now(),
        "status": status,
        "reason_created": reason_created,
        "lifecycle_history": [
            {"at": _now(), "status": status, "reason": reason_created or "registered"},
        ],
        "promotion_history": [],
    }
    store["policies"][policy_id] = record
    save_registry(store, path)
    return record


def get_policy(policy_id: str, *, path: str | Path | None = None) -> dict[str, Any] | None:
    return load_registry(path)["policies"].get(policy_id)


def list_policies(
    *, domain: str | None = None, status: str | None = None,
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    out = list(load_registry(path)["policies"].values())
    if domain is not None:
        out = [p for p in out if p.get("domain") == domain]
    if status is not None:
        out = [p for p in out if p.get("status") == status]
    return sorted(out, key=lambda p: str(p.get("created_at") or ""))


def current_champion(domain: str, *, path: str | Path | None = None) -> dict[str, Any] | None:
    champions = list_policies(domain=domain, status=CHAMPION, path=path)
    if len(champions) > 1:
        raise MultipleChampionsError(
            f"domain {domain} has {len(champions)} CHAMPION policies: "
            f"{[p['policy_id'] for p in champions]}"
        )
    return champions[0] if champions else None


def set_status(
    policy_id: str, new_status: str, *, reason: str = "",
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Append a lifecycle transition. Never rewrites prior history."""
    if new_status not in STATUSES:
        raise ValueError(f"unknown policy status {new_status!r}, expected one of {STATUSES}")
    store = load_registry(path)
    record = store["policies"].get(policy_id)
    if record is None:
        raise KeyError(f"no such policy {policy_id!r}")
    if new_status == CHAMPION:
        current = current_champion(record["domain"], path=path)
        if current is not None and current["policy_id"] != policy_id:
            raise MultipleChampionsError(
                f"domain {record['domain']} already has CHAMPION "
                f"{current['policy_id']!r}; demote it first"
            )
    record = dict(record)
    record["status"] = new_status
    record["lifecycle_history"] = list(record.get("lifecycle_history") or []) + [
        {"at": _now(), "status": new_status, "reason": reason},
    ]
    store["policies"][policy_id] = record
    save_registry(store, path)
    return record


def active_challengers(domain: str, *, path: str | Path | None = None) -> list[dict[str, Any]]:
    """Policies currently eligible to run as shadows in the tournament:
    CHALLENGER, SHADOW and PROBATION -- never RETIRED/REJECTED, never the
    CHAMPION itself (it runs through the champion path, not the shadow one)."""
    return [
        p for p in list_policies(domain=domain, path=path)
        if p.get("status") in (CHALLENGER, SHADOW, PROBATION)
    ]
