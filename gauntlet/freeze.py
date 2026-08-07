"""
E6 — Evidence Freeze.

Once the experiment starts, the thing being measured must not move under the
measurement. This snapshots the knobs that determine the result — parameters,
thresholds, weights, the signal set, the feature-engineering version — into one
hash. `verify_unchanged()` re-hashes and fails loudly if anything drifted mid-run.
No silent tuning: a changed knob means the run is void, not quietly better.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

_FREEZE_FILE = Path(__file__).resolve().parent.parent / "logs" / "gauntlet" / "freeze.json"


def _snapshot() -> dict:
    """Collect the result-determining configuration. Fail-open per source: a knob
    we cannot read is recorded as an explicit 'unavailable' so the hash still
    changes if it later becomes readable — never silently omitted."""
    snap: dict = {}
    try:
        from scan.unified_scanner import _load_calibration
        snap["calibration"] = _load_calibration() or {}
    except Exception:
        snap["calibration"] = "unavailable"
    try:
        from research.feature_schema import SCHEMA_VERSION
        snap["feature_schema"] = SCHEMA_VERSION
    except Exception:
        snap["feature_schema"] = "unavailable"
    try:
        from core import costs as _c
        snap["costs"] = {"base": _c._BASE_PCT, "slippage": _c._SLIPPAGE_PCT,
                         "enabled": _c._ENABLED}
    except Exception:
        snap["costs"] = "unavailable"
    try:
        import scan.signal_backtest as _bt
        snap["breakeven_pct"] = _bt._BT_BREAKEVEN_PCT
    except Exception:
        snap["breakeven_pct"] = "unavailable"
    try:
        from research import harness as _h
        snap["harness_gates"] = {"promote_p": _h._PROMOTE_P, "min_n": _h._HARD_MIN_N,
                                 "min_detectable_r": _h._MIN_DETECTABLE_R,
                                 "power": _h._POWER, "alpha": _h._ALPHA}
    except Exception:
        snap["harness_gates"] = "unavailable"
    return snap


def config_hash(snapshot: dict | None = None) -> str:
    snap = snapshot if snapshot is not None else _snapshot()
    blob = json.dumps(snap, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def freeze() -> dict:
    """Take the snapshot, persist it, return {hash, at, snapshot}."""
    snap = _snapshot()
    rec = {"hash": config_hash(snap), "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "snapshot": snap}
    try:
        _FREEZE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _FREEZE_FILE.write_text(json.dumps(rec, indent=2, default=str))
    except Exception:
        pass
    return rec


def verify_unchanged(frozen_hash: str) -> dict:
    """Re-hash the live config and compare. {ok, expected, actual} — ok False
    means a knob moved during the experiment and the run must be discarded."""
    actual = config_hash()
    return {"ok": actual == frozen_hash, "expected": frozen_hash, "actual": actual}
