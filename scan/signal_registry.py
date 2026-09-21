"""Canonical QuantTerm signal registry and eligibility truth.

One signal catalog must explain every scanner/calibration/replay count. The
registry is descriptive and fail-closed: unknown legacy signal IDs may remain
in old evidence stores, but they cannot silently enter the current scanner.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from core.runtime_paths import logs_path

SCHEMA_VERSION = 1
REGISTRY_PATH = logs_path("scan", "signal_registry.json")
BACKTEST_MIN_N = 20
FORWARD_MIN_N = 30


def _def(label: str, category: str, score: int) -> dict[str, Any]:
    return {
        "version": 1,
        "label": label,
        "category": category,
        "base_score": int(score),
        "enabled": True,
        "scanner_eligible": True,
        "replay_eligible": True,
        "min_backtest_samples": BACKTEST_MIN_N,
        "min_forward_samples": FORWARD_MIN_N,
    }


SIGNAL_DEFINITIONS: dict[str, dict[str, Any]] = {
    "BREAKOUT_52W": _def("52-week high breakout", "Breakout", 30),
    "BREAKOUT_RES": _def("Resistance break on volume", "Breakout", 26),
    "GOLDEN_CROSS": _def("Golden cross (50/200 SMA)", "Breakout", 22),
    "VOL_SQUEEZE": _def("Squeeze breakout", "Breakout", 22),
    "VCP": _def("VCP — tightening base", "Pattern", 28),
    "FLAT_BASE": _def("Flat base near breakout", "Pattern", 24),
    "CUP_HANDLE": _def("Cup & handle", "Pattern", 24),
    "HIGH_TIGHT_FLAG": _def("High tight flag", "Pattern", 30),
    "ASC_TRIANGLE": _def("Ascending triangle", "Pattern", 24),
    "DOUBLE_BOTTOM": _def("Double bottom", "Pattern", 22),
    "PRE_BREAKOUT": _def("Breakout ke kareeb", "PreBreakout", 26),
    "ACCUMULATION": _def("Smart-money accumulation", "PreBreakout", 24),
    "DELIVERY_SPIKE": _def("Delivery buying rising", "PreBreakout", 18),
    "NR7_COIL": _def("Coiled — tightest day in 7", "PreBreakout", 14),
    "POCKET_PIVOT": _def("Pocket pivot volume", "PreBreakout", 20),
    "MOMENTUM": _def("Strong momentum", "Momentum", 20),
    "PULLBACK_SUPPORT": _def("Uptrend pullback to support", "Pullback", 26),
}


def signal_ids() -> tuple[str, ...]:
    return tuple(SIGNAL_DEFINITIONS)


def signal_meta() -> dict[str, tuple[str, str, int]]:
    return {
        key: (str(row["label"]), str(row["category"]), int(row["base_score"]))
        for key, row in SIGNAL_DEFINITIONS.items()
    }


def registry_version() -> str:
    raw = json.dumps(SIGNAL_DEFINITIONS, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _forward_reason(n: int, minimum: int) -> str:
    if n <= 0:
        return "NO_FORWARD_OUTCOMES"
    if n < minimum:
        return f"FORWARD_SAMPLE_{n}_LT_{minimum}"
    return ""


def _backtest_reason(n: int, minimum: int) -> str:
    if n <= 0:
        return "NO_BACKTEST_TRADES"
    if n < minimum:
        return f"BACKTEST_SAMPLE_{n}_LT_{minimum}"
    return ""


def build_registry(
    *,
    backtest_report: Mapping[str, Any] | None = None,
    live_profile: Mapping[str, Any] | None = None,
    backtest_min_n: int = BACKTEST_MIN_N,
    forward_min_n: int = FORWARD_MIN_N,
) -> dict[str, Any]:
    """Build one explainable snapshot of signal eligibility."""
    if backtest_report is None:
        try:
            from scan.signal_backtest import load_report
            backtest_report = load_report() or {}
        except Exception:
            backtest_report = {}
    if live_profile is None:
        try:
            from scan.live_edge import profile_edge
            live_profile = profile_edge() or {}
        except Exception:
            live_profile = {}

    bt = dict((backtest_report or {}).get("signals") or {})
    live = dict((live_profile or {}).get("signals") or {})
    rows: list[dict[str, Any]] = []
    for signal_id, definition in SIGNAL_DEFINITIONS.items():
        b = dict(bt.get(signal_id) or {})
        f = dict(live.get(signal_id) or {})
        bt_n = max(0, int(b.get("trades") or 0))
        fw_n = max(0, int(f.get("n") or 0))
        enabled = bool(definition.get("enabled", True))
        scanner_ok = enabled and bool(definition.get("scanner_eligible", True))
        replay_ok = enabled and bool(definition.get("replay_eligible", True))
        bt_ok = scanner_ok and bt_n >= int(backtest_min_n)
        fw_ok = scanner_ok and fw_n >= int(forward_min_n)
        sources = []
        if bt_ok:
            sources.append("BACKTEST")
        if fw_ok:
            sources.append("FORWARD_OUTCOMES")
        rows.append({
            "signal_id": signal_id,
            **dict(definition),
            "scanner_eligible": scanner_ok,
            "replay_eligible": replay_ok,
            "backtest_samples": bt_n,
            "forward_samples": fw_n,
            "backtest_calibration_eligible": bt_ok,
            "forward_calibration_eligible": fw_ok,
            "calibration_eligible": bool(bt_ok or fw_ok),
            "calibration_sources": sources,
            "backtest_exclusion_reason": "" if bt_ok else _backtest_reason(bt_n, int(backtest_min_n)),
            "forward_exclusion_reason": "" if fw_ok else _forward_reason(fw_n, int(forward_min_n)),
        })

    scanner_rows = [r for r in rows if r["scanner_eligible"]]
    backtest_rows = [r for r in rows if r["backtest_calibration_eligible"]]
    forward_rows = [r for r in rows if r["forward_calibration_eligible"]]
    effective_rows = [r for r in rows if r["calibration_eligible"]]
    scanner_without_forward = [
        {
            "signal_id": r["signal_id"],
            "forward_samples": r["forward_samples"],
            "reason": r["forward_exclusion_reason"],
        }
        for r in scanner_rows if not r["forward_calibration_eligible"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "registry_version": registry_version(),
        "signals": rows,
        "summary": {
            "scanner_catalog": len(scanner_rows),
            "replay_eligible": sum(1 for r in rows if r["replay_eligible"]),
            "backtest_calibrated": len(backtest_rows),
            "forward_calibrated": len(forward_rows),
            "effective_calibrated": len(effective_rows),
            "scanner_without_forward_calibration": scanner_without_forward,
            "count_difference_explained": True,
        },
        "unknown_backtest_signal_ids": sorted(set(bt) - set(SIGNAL_DEFINITIONS)),
        "unknown_forward_signal_ids": sorted(set(live) - set(SIGNAL_DEFINITIONS)),
    }


def save_registry(
    payload: Mapping[str, Any] | None = None,
    *,
    path: str | Path | None = None,
) -> dict[str, Any]:
    data = dict(payload or build_registry())
    target = Path(path) if path is not None else REGISTRY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, target)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
    return data


def load_registry(*, path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else REGISTRY_PATH
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        return dict(payload) if isinstance(payload, dict) else {}
    except Exception:
        return {}
