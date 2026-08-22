"""FEATURE-002 operational watchdog.

Detects logging / clock / ledger faults. Logs (and optional Telegram text).
Never places orders, never changes ranks, never mutates the experiment spec.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.feature002.constants import (
    FEATURE_SET_VERSION,
    LEDGER_DIR,
    protocol_hash,
)
from research.feature002.health import (
    HOOK_LOG,
    build_health,
    write_health,
    write_status_md,
)

WATCHDOG_LOG = LEDGER_DIR / "watchdog.jsonl"


def _append(path: Path, row: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except Exception:
        pass


def note_production_scan(*, n_cards: int, last_scan_ts: float | None = None) -> None:
    """Synchronous receipt from auto_scan. Must never raise into the worker."""
    try:
        from research.feature002.health import _ist_now, _unix_to_ist_iso
        _append(HOOK_LOG, {
            "kind": "production_scan_saved",
            "ts": _ist_now(),
            "n_cards": int(n_cards),
            "last_scan_ts_ist": _unix_to_ist_iso(last_scan_ts),
        })
    except Exception:
        pass


def evaluate(health: dict[str, Any] | None = None, *, ledger_path=None) -> dict[str, Any]:
    health = health or build_health(ledger_path=ledger_path)
    alerts: list[dict[str, Any]] = []

    empty = health.get("empty_primary") or {}
    if empty.get("is_bug"):
        alerts.append({
            "code": "SCANS_WITHOUT_SHADOW_ROWS",
            "severity": "error",
            "detail": empty.get("detail"),
        })

    ledger = health.get("ledger") or {}
    if int(ledger.get("observations") or 0) > 0 and int(ledger.get("candidate_sets") or 0) == 0:
        alerts.append({
            "code": "SHADOW_ROWS_WITHOUT_CANDIDATE_SETS",
            "severity": "error",
            "detail": "observations exist but candidate_sets is empty",
        })

    if ledger.get("corrupt"):
        alerts.append({
            "code": "LEDGER_CORRUPTION",
            "severity": "error",
            "detail": ledger.get("corrupt_reason"),
        })

    versions = list(ledger.get("feature_versions") or [])
    if versions and FEATURE_SET_VERSION not in versions:
        alerts.append({
            "code": "PROTOCOL_VERSION_MISMATCH",
            "severity": "error",
            "detail": f"ledger versions {versions} vs {FEATURE_SET_VERSION}",
        })
    hashes = list(ledger.get("protocol_hashes") or [])
    ph = protocol_hash()
    if hashes and ph not in hashes:
        alerts.append({
            "code": "PROTOCOL_HASH_MISMATCH",
            "severity": "warn",
            "detail": f"ledger hashes {hashes} vs {ph}",
        })

    clock = health.get("clock") or {}
    if clock.get("ok") is False:
        alerts.append({
            "code": "CLOCK_TIMEZONE_ERROR",
            "severity": "error",
            "detail": clock,
        })

    dups = int(health.get("duplicate_count") or 0)
    obs = int(ledger.get("observations") or 0)
    if dups >= 50 and obs and dups > obs * 3:
        alerts.append({
            "code": "DUPLICATE_EXPLOSION",
            "severity": "warn",
            "detail": f"duplicate_count={dups} observations={obs}",
        })

    n_pri = int((health.get("maturity") or {}).get("n_primary") or 0)
    n_unres = int(health.get("unresolved_outcomes") or 0)
    latest_obs = health.get("latest_feature002_observation_timestamp")
    if n_pri and n_unres and latest_obs:
        # Stale resolver: primary rows exist, none resolved, observation older than 10 sessions.
        if int(health.get("resolved_outcomes") or 0) == 0:
            alerts.append({
                "code": "STALE_OUTCOME_RESOLVER",
                "severity": "warn",
                "detail": (
                    f"{n_unres} unresolved; 0 resolved. Run "
                    "`python -m research.feature002 --resolve` after 5 official bars."
                ),
            })

    if int(health.get("logging_exception_count") or 0) > 0:
        alerts.append({
            "code": "LOGGING_EXCEPTIONS",
            "severity": "warn",
            "detail": f"{health.get('logging_exception_count')} hook/persist exceptions",
        })

    worst = "info"
    for a in alerts:
        if a["severity"] == "error":
            worst = "error"
        elif a["severity"] == "warn" and worst != "error":
            worst = "warn"

    return {
        "ok": worst != "error",
        "severity": worst,
        "alerts": alerts,
        "n_alerts": len(alerts),
        "is_logging_bug": bool(empty.get("is_bug")),
        "status": health.get("status"),
    }


def run(*, ledger_path=None, persist: bool = True, alert: bool = True) -> dict[str, Any]:
    """Build health, evaluate, persist, optionally notify. Never raises to caller."""
    try:
        health = write_health(ledger_path=ledger_path) if persist else build_health(ledger_path=ledger_path)
        if persist:
            write_status_md(health, ledger_path=ledger_path)
        verdict = evaluate(health, ledger_path=ledger_path)
        from research.feature002.health import _ist_now
        row = {"kind": "watchdog", "ts": _ist_now(), **verdict}
        _append(WATCHDOG_LOG, row)
        if alert and verdict["alerts"]:
            _emit(verdict)
        return verdict
    except Exception as exc:
        try:
            from logger import get_logger
            get_logger(__name__).debug("feature002_watchdog_failed", error=str(exc))
        except Exception:
            pass
        return {"ok": False, "severity": "error", "alerts": [{"code": "WATCHDOG_FAILED", "detail": str(exc)}]}


def _emit(verdict: dict[str, Any]) -> None:
    try:
        from logger import get_logger
        get_logger(__name__).warning(
            "feature002_watchdog",
            severity=verdict.get("severity"),
            codes=[a.get("code") for a in verdict.get("alerts") or []],
        )
    except Exception:
        pass
    if not verdict.get("is_logging_bug"):
        return
    try:
        from alerts.telegram_alerts import AlertEngine
        codes = ", ".join(a.get("code") or "?" for a in verdict.get("alerts") or [])
        AlertEngine().send(
            f"FEATURE-002 watchdog (logging only; no trade): {codes}"
        )
    except Exception:
        pass
