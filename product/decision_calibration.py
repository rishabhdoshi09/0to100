"""DecisionCalibrationEngine — are confidence labels trustworthy?

Audit / learning input first. Does not rename High Conviction because
calibration is poor. One observation cannot change production. PIT-safe:
predicted confidence is frozen at decision time; outcomes arrive later.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from core.runtime_paths import logs_dir

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = logs_dir() / "product" / "calibration.json"
SCHEMA_VERSION = 1
MIN_SAMPLE = 20


def store_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_CALIBRATION")
    if override:
        return Path(override)
    return DEFAULT_PATH


def empty_store() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "observations": [],
        "affects_production": False,
        "live_locked": True,
        "note": "Calibration does not rename reco tiers and cannot enable live money.",
    }


def load_store(path: str | Path | None = None) -> dict[str, Any]:
    target = store_path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return empty_store()
    if not isinstance(payload, dict):
        return empty_store()
    payload.setdefault("observations", [])
    payload["affects_production"] = False
    payload["live_locked"] = True
    return payload


def save_store(payload: Mapping[str, Any], path: str | Path | None = None) -> Path:
    target = store_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["schema_version"] = SCHEMA_VERSION
    data["affects_production"] = False
    data["live_locked"] = True
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return target


def _bucket(tier: str) -> str:
    t = str(tier or "").lower()
    if t in {"high_conviction", "high"}:
        return "high_conviction"
    if t in {"good_setup", "good"}:
        return "good_setup"
    if t in {"watch", "avoid"}:
        return t
    return t or "unspecified"


def _implied_p(bucket: str) -> float | None:
    """Tiers are setup-quality labels, not estimated win probabilities.

    Until a bucket has a measured hit rate (MIN_SAMPLE settled outcomes),
    QuantTerm must not display an invented percentage.
    """
    return None


def _probability(value: Any) -> float | None:
    """Return an explicit probability only when it is finite and in [0, 1]."""
    try:
        if value is None or value == "":
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out) or out < 0.0 or out > 1.0:
        return None
    return out


def _wilson_interval(wins: int, n: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson interval for a measured binary hit rate."""
    if n <= 0:
        return None
    p = max(0.0, min(1.0, float(wins) / float(n)))
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(
        (p * (1.0 - p) / n) + (z2 / (4.0 * n * n))
    )
    return [
        round(max(0.0, centre - margin), 4),
        round(min(1.0, centre + margin), 4),
    ]


def display_confidence(*, tier: str = "", sample_size: int = 0, hit_rate: float | None = None) -> dict[str, Any]:
    """What the operator is allowed to see. No decorative percentages."""
    bucket = _bucket(tier)
    if sample_size >= MIN_SAMPLE and hit_rate is not None:
        return {
            "kind": "MEASURED_HIT_RATE",
            "label": bucket or "unspecified",
            "sample_size": sample_size,
            "hit_rate": hit_rate,
            "is_probability": True,
            "display": f"{bucket} · measured hit rate {round(hit_rate * 100, 1)}% (n={sample_size})",
        }
    return {
        "kind": "SETUP_QUALITY",
        "label": bucket or "unspecified",
        "sample_size": sample_size,
        "hit_rate": None,
        "is_probability": False,
        "display": f"{bucket or 'unspecified'} (setup quality — not a win probability)",
    }


class DecisionCalibrationEngine:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = store_path(path)
        self.store = load_store(self.path)

    def record(
        self,
        *,
        predicted_confidence: str,
        realized_win: bool | None,
        strategy: str = "",
        setup: str = "",
        regime: str = "",
        sector: str = "",
        decision_as_of: str,
        outcome_as_of: str,
        predicted_p: float | None = None,
    ) -> dict[str, Any]:
        """PIT-safe: decision_as_of must be <= outcome_as_of. Future data cannot rewrite the prediction."""
        if outcome_as_of and decision_as_of and str(outcome_as_of) < str(decision_as_of):
            raise ValueError("outcome cannot precede the point-in-time decision")
        bucket = _bucket(predicted_confidence)
        implied = (
            _probability(predicted_p)
            if predicted_p is not None
            else _implied_p(bucket)
        )
        row = {
            "predicted_confidence": bucket,
            "predicted_p": implied,
            "realized_win": realized_win,
            "strategy": strategy,
            "setup": setup,
            "regime": regime,
            "sector": sector,
            "decision_as_of": decision_as_of,
            "outcome_as_of": outcome_as_of,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "production_changed": False,
        }
        obs = list(self.store.get("observations") or [])
        obs.append(row)
        self.store["observations"] = obs
        save_store(self.store, self.path)
        return row

    def summary(
        self,
        *,
        bucket: str | None = None,
        setup: str = "",
        regime: str = "",
        sector: str = "",
    ) -> dict[str, Any]:
        """Summarize hit-rate evidence and probability calibration separately.

        Setup-quality tiers are not probabilities. Every settled observation can
        contribute to a measured hit rate, but Brier / expected-vs-realized
        calibration is computed only from rows that carried an explicit
        point-in-time predicted_p. Missing probabilities are never replaced
        with 0.5 or any other invented value.
        """
        rows = [
            r for r in (self.store.get("observations") or [])
            if r.get("realized_win") is not None
            and (not bucket or r.get("predicted_confidence") == _bucket(bucket))
            and (not setup or r.get("setup") == setup)
            and (not regime or r.get("regime") == regime)
            and (not sector or r.get("sector") == sector)
        ]
        n = len(rows)
        probability_rows = [
            r for r in rows
            if _probability(r.get("predicted_p")) is not None
        ]
        probability_n = len(probability_rows)

        base = {
            "sample_size": n,
            "min_sample": MIN_SAMPLE,
            "affects_production": False,
            "bucket": bucket,
            "setup": setup,
            "regime": regime,
            "sector": sector,
            "probability_sample_size": probability_n,
            "probability_min_sample": MIN_SAMPLE,
            "probability_status": (
                "MEASURED" if probability_n >= MIN_SAMPLE
                else "INSUFFICIENT_PROBABILITY_EVIDENCE" if probability_n > 0
                else "NO_EXPLICIT_PROBABILITIES"
            ),
            "brier": None,
            "expected_p": None,
            "probability_actual_hit_rate": None,
            "probability_confidence_interval": None,
            "calibration_gap": None,
            "calibration_actionable": False,
            "calibration_direction": "NONE",
            "calibration_adjustment": 0.0,
            "overconfidence": False,
            "underconfidence": False,
            "rename_tier": False,
            "probability_only_scoring": True,
            "live_locked": True,
        }

        if n < MIN_SAMPLE:
            return {
                **base,
                "status": "INSUFFICIENT_EVIDENCE",
                "actual_hit_rate": None,
                "confidence_interval": None,
                "wilson_lower_bound": None,
                "wilson_upper_bound": None,
            }

        hits = sum(1 for r in rows if r.get("realized_win"))
        actual = hits / n
        interval = _wilson_interval(hits, n) or [None, None]
        result = {
            **base,
            "status": "MEASURED",
            "actual_hit_rate": round(actual, 4),
            "confidence_interval": interval,
            "wilson_lower_bound": interval[0],
            "wilson_upper_bound": interval[1],
        }

        # Probability calibration is a different estimand. It needs its own
        # sample floor and uses only predictions that really existed at decision
        # time. Tier labels without predicted_p do not enter these statistics.
        if probability_n >= MIN_SAMPLE:
            probs = [_probability(r.get("predicted_p")) for r in probability_rows]
            probs = [float(p) for p in probs if p is not None]
            probability_hits = sum(1 for r in probability_rows if r.get("realized_win"))
            probability_actual = probability_hits / probability_n
            expected = sum(probs) / probability_n
            brier = sum(
                (p - (1.0 if r.get("realized_win") else 0.0)) ** 2
                for p, r in zip(probs, probability_rows)
            ) / probability_n
            probability_interval = _wilson_interval(probability_hits, probability_n)
            gap = expected - probability_actual
            lower = float(probability_interval[0]) if probability_interval else probability_actual
            upper = float(probability_interval[1]) if probability_interval else probability_actual
            if expected > upper:
                direction = "OVERCONFIDENT"
                adjustment = expected - upper
            elif expected < lower:
                direction = "UNDERCONFIDENT"
                adjustment = expected - lower
            else:
                direction = "NONE"
                adjustment = 0.0
            result.update({
                "brier": round(brier, 4),
                "expected_p": round(expected, 4),
                "probability_actual_hit_rate": round(probability_actual, 4),
                "probability_confidence_interval": probability_interval,
                "calibration_gap": round(gap, 4),
                "calibration_actionable": direction != "NONE",
                "calibration_direction": direction,
                # Positive adjustment means the model over-claimed and should
                # be shifted down; negative means it under-claimed.
                "calibration_adjustment": round(adjustment, 4),
                "overconfidence": direction == "OVERCONFIDENT",
                "underconfidence": direction == "UNDERCONFIDENT",
            })
        return result

    def dossier(self) -> dict[str, Any]:
        """Read-only calibration dossier for operator/research inspection."""
        rows = [
            r for r in (self.store.get("observations") or [])
            if r.get("realized_win") is not None
        ]
        explicit = sum(1 for r in rows if _probability(r.get("predicted_p")) is not None)
        return {
            "schema_version": SCHEMA_VERSION,
            "overall": self.summary(),
            "buckets": self.buckets(),
            "settled_observations": len(rows),
            "explicit_probability_observations": explicit,
            "probability_coverage": round(explicit / len(rows), 4) if rows else 0.0,
            "affects_production": False,
            "live_locked": True,
            "note": (
                "Hit-rate measurement and probability calibration are separate. "
                "Only explicit point-in-time probabilities enter Brier/calibration-gap metrics."
            ),
        }

    def buckets(self) -> dict[str, Any]:
        found = sorted({
            str(r.get("predicted_confidence"))
            for r in self.store.get("observations") or []
            if str(r.get("predicted_confidence") or "")
        })
        return {b: self.summary(bucket=b) for b in found}
