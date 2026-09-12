"""What the system has actually learned, conditioned on the situation.

"This setup wins 61%" is almost never the useful number. The same setup in a
narrow tape, bought five percent above its pivot, with a confidence the system
has historically over-stated, is a different bet — and the only way to know is
to keep the statistics conditioned on those things rather than pooled.

Every cell is keyed by ``(evidence_class, context_key)``. The evidence class is
part of the key on purpose: a replay row and a settled paper trade must never
land in the same bucket, because the moment they do, nobody can say which of
them the number came from. A historical replay may prove the loop is wired.
It may not move a statistic that decides position size.

Reads used for ranking go through :func:`ranking_evidence`, which returns
nothing until the sample floor is met and can only ever demote. A system that
lets thin evidence promote a setup has invented a reason to trade.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from statistics import median
from typing import Any, Mapping

from core.runtime_paths import logs_path
from product.decision_chain import EvidenceUpdate, Outcome
from product.evidence_class import PAPER_FORWARD, is_market_evidence
from product.evidence_class import normalise as normalise_evidence_class

SCHEMA_VERSION = 1

#: Below this the cell has no vote. The repository's standing rule: under 30
#: settled observations, no claim.
MIN_SAMPLE = 30

#: Wilson z for a 95% interval.
_Z = 1.96

DEFAULT_PATH = logs_path("product", "conditional_evidence.json")


def store_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_CONDITIONAL_EVIDENCE")
    return Path(override) if override else DEFAULT_PATH


def empty_store() -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "cells": {}}


def load(path: str | Path | None = None) -> dict[str, Any]:
    target = store_path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return empty_store()
    if not isinstance(payload, dict) or not isinstance(payload.get("cells"), dict):
        return empty_store()
    return payload


def save(payload: Mapping[str, Any], path: str | Path | None = None) -> Path:
    target = store_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["schema_version"] = SCHEMA_VERSION
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return target


def cell_key(evidence_class: str, context_key: str) -> str:
    return f"{normalise_evidence_class(evidence_class) or 'UNKNOWN'}::{context_key}"


def wilson_lower_bound(wins: int, n: int, z: float = _Z) -> float:
    """The win rate we can defend, not the one we observed.

    Thirty trades at 60% and three hundred at 60% are not the same claim, and
    ranking should not treat them as one.
    """
    if n <= 0:
        return 0.0
    p = wins / n
    denominator = 1.0 + (z * z) / n
    centre = p + (z * z) / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + (z * z) / (4 * n)) / n)
    return max(0.0, (centre - margin) / denominator)


def empty_cell(context_key: str, evidence_class: str) -> dict[str, Any]:
    return {
        "context_key": context_key,
        "evidence_class": normalise_evidence_class(evidence_class),
        "count": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": None,
        "wilson_lower_bound": None,
        "expectancy_R": None,
        "median_R": None,
        "mae_R": None,
        "mfe_R": None,
        "r_values": [],
        "confidence_claimed": [],
        "calibration_gap": None,
        "contributing_outcome_ids": [],
        "updated_at": "",
    }


def read(context_key: str, *, evidence_class: str = PAPER_FORWARD,
         path: str | Path | None = None) -> dict[str, Any]:
    """The cell as it stands. Absent reads as an empty cell, never as zero edge."""
    store = load(path)
    key = cell_key(evidence_class, context_key)
    cell = store.get("cells", {}).get(key)
    return dict(cell) if isinstance(cell, Mapping) else empty_cell(context_key, evidence_class)


def _recompute(cell: dict[str, Any]) -> dict[str, Any]:
    r_values = [float(r) for r in cell.get("r_values") or []]
    n = len(r_values)
    wins = sum(1 for r in r_values if r > 0)
    losses = n - wins
    cell["count"] = n
    cell["wins"] = wins
    cell["losses"] = losses
    cell["win_rate"] = (wins / n) if n else None
    cell["wilson_lower_bound"] = wilson_lower_bound(wins, n) if n else None
    cell["expectancy_R"] = (sum(r_values) / n) if n else None
    cell["median_R"] = median(r_values) if n else None
    claimed = [float(c) for c in cell.get("confidence_claimed") or [] if c is not None]
    if claimed and cell["win_rate"] is not None:
        # Said 70%, did 70% happen? Positive means the system over-claimed.
        cell["calibration_gap"] = (sum(claimed) / len(claimed)) - cell["win_rate"]
    else:
        cell["calibration_gap"] = None
    return cell


def record_outcome(
    outcome: Outcome,
    *,
    context_key: str,
    evidence_class: str | None = None,
    calibrated_confidence: float | None = None,
    path: str | Path | None = None,
    also_update_policy_ladder: bool = True,
    policy_dimension: str = "setup",
    policy_bucket: str = "",
) -> EvidenceUpdate:
    """Fold one settled outcome into its cell and return the belief change.

    The returned :class:`EvidenceUpdate` carries the cell before and after, so
    the caller (and a reader months later) can see that this outcome is what
    moved the number, rather than taking it on trust.
    """
    if outcome.realized_R is None:
        raise ValueError(
            f"outcome {outcome.outcome_id} has not settled; an unresolved trade "
            "is not evidence"
        )
    klass = normalise_evidence_class(evidence_class or outcome.evidence_class or "")
    if not klass:
        raise ValueError(
            "an outcome must declare how it was produced before it can become "
            "evidence"
        )

    store = load(path)
    cells = store.setdefault("cells", {})
    key = cell_key(klass, context_key)
    before = dict(cells.get(key) or empty_cell(context_key, klass))

    cell = json.loads(json.dumps(before))
    if outcome.outcome_id in (cell.get("contributing_outcome_ids") or []):
        # Replaying the same settled outcome must not double-count it.
        return EvidenceUpdate(
            outcome_id=outcome.outcome_id,
            position_id=outcome.position_id,
            decision_id=outcome.decision_id,
            context_key=context_key,
            before=_public(before),
            after=_public(before),
            evidence_class=klass,
        )

    cell.setdefault("r_values", []).append(round(float(outcome.realized_R), 6))
    cell.setdefault("contributing_outcome_ids", []).append(outcome.outcome_id)
    if calibrated_confidence is not None:
        cell.setdefault("confidence_claimed", []).append(float(calibrated_confidence))
    if outcome.mae_R is not None:
        cell["mae_R"] = min(float(outcome.mae_R), float(cell.get("mae_R") or outcome.mae_R))
    if outcome.mfe_R is not None:
        cell["mfe_R"] = max(float(outcome.mfe_R), float(cell.get("mfe_R") or outcome.mfe_R))
    cell["updated_at"] = outcome.resolved_at
    cell = _recompute(cell)

    cells[key] = cell
    save(store, path)

    if also_update_policy_ladder and is_market_evidence(klass):
        # One settled outcome, one event, both stores. The policy ladder stays
        # the gate it always was; this is not a second source of the outcome.
        try:
            from product.learning_policy_store import record_measured_outcome

            record_measured_outcome(
                policy_id=f"cond::{context_key}",
                dimension=policy_dimension,
                bucket=policy_bucket or context_key,
                realized_R=float(outcome.realized_R),
                source="paper_forward",
            )
        except Exception:
            # The ladder is an additional consumer, never a gate on recording.
            pass

    return EvidenceUpdate(
        outcome_id=outcome.outcome_id,
        position_id=outcome.position_id,
        decision_id=outcome.decision_id,
        context_key=context_key,
        before=_public(before),
        after=_public(cell),
        evidence_class=klass,
    )


def _public(cell: Mapping[str, Any]) -> dict[str, Any]:
    """The summary worth storing in an update: statistics, not raw streams."""
    return {
        "count": cell.get("count", 0),
        "wins": cell.get("wins", 0),
        "losses": cell.get("losses", 0),
        "win_rate": cell.get("win_rate"),
        "wilson_lower_bound": cell.get("wilson_lower_bound"),
        "expectancy_R": cell.get("expectancy_R"),
        "median_R": cell.get("median_R"),
        "calibration_gap": cell.get("calibration_gap"),
    }


def ranking_evidence(
    context_key: str,
    *,
    evidence_class: str = PAPER_FORWARD,
    path: str | Path | None = None,
    min_sample: int = MIN_SAMPLE,
) -> dict[str, Any]:
    """What ranking is allowed to use. Demote-only, sample-gated.

    Returns a record with an explicit ``adjustment`` in score points and the
    reason for it, so a rank that moved can always say why.
    """
    klass = normalise_evidence_class(evidence_class)
    if not is_market_evidence(klass):
        return {
            "usable": False,
            "reason": "NOT_MARKET_EVIDENCE",
            "adjustment": 0.0,
            "evidence_class": klass,
            "context_key": context_key,
            "count": 0,
        }
    cell = read(context_key, evidence_class=klass, path=path)
    count = int(cell.get("count") or 0)
    if count < int(min_sample):
        return {
            "usable": False,
            "reason": "INSUFFICIENT_EVIDENCE",
            "adjustment": 0.0,
            "evidence_class": klass,
            "context_key": context_key,
            "count": count,
            "min_sample": int(min_sample),
        }

    expectancy = float(cell.get("expectancy_R") or 0.0)
    if expectancy >= 0:
        # Measured, and not a loser. That earns no bonus: a setup does not get
        # ranked higher for having survived its own history.
        return {
            "usable": True,
            "reason": "MEASURED_NOT_NEGATIVE",
            "adjustment": 0.0,
            "evidence_class": klass,
            "context_key": context_key,
            "count": count,
            "expectancy_R": expectancy,
            "wilson_lower_bound": cell.get("wilson_lower_bound"),
        }

    # Proven to leak in this context. Demote in proportion to how badly, with a
    # floor so one measured cell cannot erase a decision entirely.
    adjustment = max(-30.0, round(expectancy * 20.0, 4))
    return {
        "usable": True,
        "reason": "MEASURED_NEGATIVE_EXPECTANCY",
        "adjustment": adjustment,
        "evidence_class": klass,
        "context_key": context_key,
        "count": count,
        "expectancy_R": expectancy,
        "wilson_lower_bound": cell.get("wilson_lower_bound"),
    }
