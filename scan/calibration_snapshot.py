"""Immutable calibration snapshots for scanner/replay reproducibility.

Calibration is recomputed only when one of its factual inputs changes:
data identity, thesis, feature/model/registry version, backtest evidence, or
forward-outcome evidence. Each immutable snapshot owns the effective signal
multipliers used by the scanner.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from core.runtime_paths import logs_path
from scan import signal_registry as SR

SCHEMA_VERSION = 1
FEATURE_VERSION = "unified-scanner-features-v1"
MODEL_VERSION = "signal-calibration-v1"
DEFAULT_DIR = logs_path("scan", "calibration_snapshots")


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _default_data_identity() -> str:
    try:
        from product.readiness import official_history
        h = dict(official_history() or {})
        explicit = str(h.get("source_snapshot_id") or h.get("snapshot_id") or "").strip()
        if explicit:
            return explicit
        session = str(h.get("available_session") or h.get("latest_date") or "")[:10]
        source = str(h.get("source") or "official_nse").strip().lower()
        if session:
            return f"{source}:{session}"
    except Exception:
        pass
    return "unknown"


def _default_thesis_hash() -> str:
    try:
        from product.trading_thesis import manifest
        return str(manifest().get("thesis_hash") or "unknown")
    except Exception:
        return "unknown"


def _load_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        from scan.signal_backtest import load_report
        backtest = dict(load_report() or {})
    except Exception:
        backtest = {}
    try:
        from scan.live_edge import profile_edge
        live = dict(profile_edge() or {})
    except Exception:
        live = {}
    return backtest, live


def _bucket(expectancy_r: float) -> float:
    if expectancy_r >= 0.30:
        return 1.25
    if expectancy_r >= 0.10:
        return 1.0
    if expectancy_r >= -0.10:
        return 0.75
    return 0.45


def backtest_multipliers(
    report: Mapping[str, Any] | None,
    *,
    min_n: int = SR.BACKTEST_MIN_N,
) -> dict[str, float]:
    out: dict[str, float] = {}
    signals = dict((report or {}).get("signals") or {})
    canonical = set(SR.signal_ids())
    for signal_id, row in signals.items():
        if signal_id not in canonical:
            continue
        if int((row or {}).get("trades") or 0) < int(min_n):
            continue
        out[signal_id] = _bucket(float((row or {}).get("expectancy_r") or 0.0))
    return out


def forward_multipliers(
    profile: Mapping[str, Any] | None,
    *,
    min_n: int = SR.FORWARD_MIN_N,
) -> dict[str, float]:
    out: dict[str, float] = {}
    signals = dict((profile or {}).get("signals") or {})
    canonical = set(SR.signal_ids())
    for signal_id, row in signals.items():
        if signal_id not in canonical:
            continue
        if int((row or {}).get("n") or 0) < int(min_n):
            continue
        out[signal_id] = _bucket(float((row or {}).get("expectancy_r") or 0.0))
    return out


def effective_multipliers(
    backtest: Mapping[str, Any] | None,
    live: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Preserve current policy: live may demote but never inflate backtest."""
    bt = backtest_multipliers(backtest)
    fw = forward_multipliers(live)
    out = dict(bt)
    for signal_id, multiplier in fw.items():
        out[signal_id] = min(out.get(signal_id, 1.0), float(multiplier))
    return out


def build_snapshot(
    *,
    backtest_report: Mapping[str, Any] | None = None,
    live_profile: Mapping[str, Any] | None = None,
    data_identity: str | None = None,
    thesis_hash: str | None = None,
    feature_version: str = FEATURE_VERSION,
    model_version: str = MODEL_VERSION,
) -> dict[str, Any]:
    backtest = dict(backtest_report or {})
    live = dict(live_profile or {})
    registry = SR.build_registry(backtest_report=backtest, live_profile=live)
    identities = {
        "data_identity": str(data_identity or _default_data_identity()),
        "thesis_hash": str(thesis_hash or _default_thesis_hash()),
        "feature_version": str(feature_version),
        "model_version": str(model_version),
        "signal_registry_version": str(registry.get("registry_version") or SR.registry_version()),
        "backtest_evidence_hash": _canonical_hash(backtest),
        "forward_evidence_hash": _canonical_hash(live),
    }
    snapshot_id = "cal_" + _canonical_hash(identities)[:20]
    return {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "immutable": True,
        "live_locked": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identities": identities,
        "multipliers": effective_multipliers(backtest, live),
        "backtest_multipliers": backtest_multipliers(backtest),
        "forward_multipliers": forward_multipliers(live),
        "signal_registry": registry,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return dict(payload) if isinstance(payload, dict) else {}
    except Exception:
        return {}


def get_or_create_snapshot(
    *,
    backtest_report: Mapping[str, Any] | None = None,
    live_profile: Mapping[str, Any] | None = None,
    data_identity: str | None = None,
    thesis_hash: str | None = None,
    feature_version: str = FEATURE_VERSION,
    model_version: str = MODEL_VERSION,
    directory: str | Path | None = None,
) -> dict[str, Any]:
    if backtest_report is None or live_profile is None:
        loaded_bt, loaded_live = _load_inputs()
        if backtest_report is None:
            backtest_report = loaded_bt
        if live_profile is None:
            live_profile = loaded_live
    candidate = build_snapshot(
        backtest_report=backtest_report,
        live_profile=live_profile,
        data_identity=data_identity,
        thesis_hash=thesis_hash,
        feature_version=feature_version,
        model_version=model_version,
    )
    root = Path(directory) if directory is not None else DEFAULT_DIR
    target = root / f"{candidate['snapshot_id']}.json"
    existing = _load(target)
    if existing:
        out = dict(existing)
        out["cache_hit"] = True
        return out

    _atomic_json(target, candidate)
    pointer = root / "current.json"
    _atomic_json(pointer, {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": candidate["snapshot_id"],
        "snapshot_path": str(target),
        "identities": candidate["identities"],
    })
    try:
        SR.save_registry(candidate["signal_registry"])
    except Exception:
        pass
    out = dict(candidate)
    out["cache_hit"] = False
    return out


def load_current(*, directory: str | Path | None = None) -> dict[str, Any]:
    root = Path(directory) if directory is not None else DEFAULT_DIR
    pointer = _load(root / "current.json")
    snapshot_id = str(pointer.get("snapshot_id") or "")
    return _load(root / f"{snapshot_id}.json") if snapshot_id else {}
