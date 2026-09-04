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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "logs" / "product" / "calibration.json"
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
        implied = predicted_p if predicted_p is not None else _implied_p(bucket)
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
        rows = [
            r for r in (self.store.get("observations") or [])
            if r.get("realized_win") is not None
            and (not bucket or r.get("predicted_confidence") == _bucket(bucket))
            and (not setup or r.get("setup") == setup)
            and (not regime or r.get("regime") == regime)
            and (not sector or r.get("sector") == sector)
        ]
        n = len(rows)
        if n < MIN_SAMPLE:
            return {
                "sample_size": n,
                "min_sample": MIN_SAMPLE,
                "status": "INSUFFICIENT_EVIDENCE",
                "affects_production": False,
                "brier": None,
                "expected_p": None,
                "actual_hit_rate": None,
                "confidence_interval": None,
                "overconfidence": False,
                "underconfidence": False,
                "bucket": bucket,
            }
        hits = sum(1 for r in rows if r.get("realized_win"))
        actual = hits / n
        expected = sum(float(r.get("predicted_p") or 0.5) for r in rows) / n
        brier = sum(
            (float(r.get("predicted_p") or 0.5) - (1.0 if r.get("realized_win") else 0.0)) ** 2
            for r in rows
        ) / n
        se = math.sqrt(actual * (1 - actual) / n) if n else 0.0
        over = actual < expected - 0.08
        under = actual > expected + 0.08
        return {
            "sample_size": n,
            "min_sample": MIN_SAMPLE,
            "status": "MEASURED",
            "affects_production": False,
            "brier": round(brier, 4),
            "expected_p": round(expected, 4),
            "actual_hit_rate": round(actual, 4),
            "confidence_interval": [round(actual - 1.96 * se, 4), round(actual + 1.96 * se, 4)],
            "overconfidence": over,
            "underconfidence": under,
            "bucket": bucket,
            "rename_tier": False,
        }

    def buckets(self) -> dict[str, Any]:
        found = sorted({str(r.get("predicted_confidence")) for r in self.store.get("observations") or []})
        return {b: self.summary(bucket=b) for b in found}
