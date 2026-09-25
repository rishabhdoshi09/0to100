"""Persistent retail scan results and tomorrow-watchlist projection.

The scanner remains the source of truth. This module only serializes its output
so the UI opens instantly instead of rescanning the full market on every rerun.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from core.runtime_paths import logs_dir, logs_path

def default_scan_path() -> Path:
    return logs_path("product", "latest_momentum_scan.json")


def resolved_scan_path() -> Path:
    """One runtime scan-artifact identity for every reader."""
    override = str(os.environ.get("QT_SCAN_PATH") or "").strip()
    return Path(override) if override else default_scan_path()


DEFAULT_SCAN_PATH = default_scan_path()


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_float(raw: Any) -> float | None:
    """Coerce an already-persisted value, keeping missing as missing."""
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _opt_float(obj: Any, name: str) -> float | None:
    """Persist a number only when the scanner actually set it. Missing stays missing."""
    if isinstance(obj, Mapping) and name not in obj:
        return None
    if not isinstance(obj, Mapping) and not hasattr(obj, name):
        return None
    raw = _value(obj, name, None)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _opt_bool(obj: Any, name: str) -> bool | None:
    if isinstance(obj, Mapping):
        if name not in obj:
            return None
        return bool(obj.get(name))
    if not hasattr(obj, name):
        return None
    return bool(getattr(obj, name))


def _record(signal: Any, names: Mapping[str, str], fno_symbols: set[str]) -> dict[str, Any]:
    symbol = str(_value(signal, "symbol", "") or "").upper()
    signals = [str(x) for x in (_value(signal, "signals", []) or [])]
    reasons = [str(x) for x in (_value(signal, "reasons", []) or [])]
    chase = bool(_value(signal, "chase_risk", False))
    verdict = str(_value(signal, "verdict", "WATCH") or "WATCH")
    if chase:
        status = "Wait for pullback"
    elif verdict == "BUY":
        status = "Ready to trade"
    elif "PRE_BREAKOUT" in signals:
        status = "Watch for breakout"
    else:
        status = "Watch"
    cats_raw = _value(signal, "categories", None)
    if isinstance(cats_raw, (set, tuple, list)):
        categories = sorted({str(c) for c in cats_raw if c})
    else:
        categories = []
    row = {
        "symbol": symbol,
        "company": str(names.get(symbol, symbol)),
        "status": status,
        # Scanner status is technical setup state, never final execution authority.
        # Keep the legacy text for compatibility, but make its scope explicit so
        # every consumer can distinguish SETUP READY from PAPER ENTER_NOW.
        "status_scope": "SCANNER_SETUP",
        "selection_required": True,
        "verdict": verdict,
        # A trade level the scanner never produced is missing, not zero. Coercing
        # to 0.0 rendered "Stop Rs0.00" in the desk, which is a fabricated trade
        # plan rather than an absent one.
        "price": _opt_float(signal, "price"),
        "momentum_5d": float(_value(signal, "momentum_5d", 0.0) or 0.0),
        "score": float(_value(signal, "score", 0.0) or 0.0),
        "rsi": float(_value(signal, "rsi", 0.0) or 0.0),
        "volume_ratio": float(_value(signal, "volume_ratio", 0.0) or 0.0),
        "entry": _opt_float(signal, "entry"),
        "stop": _opt_float(signal, "stop"),
        "target": _opt_float(signal, "target"),
        **derive_trade_plan({
            "entry": _opt_float(signal, "entry"),
            "stop": _opt_float(signal, "stop"),
            "target": _opt_float(signal, "target"),
            "price": _opt_float(signal, "price"),
        }),
        "chase_risk": chase,
        "fno_available": symbol in fno_symbols,
        "signals": signals,
        "reasons": reasons,
        "why": reasons[0] if reasons else "No explanation recorded",
        "breakout_grade": str(_value(signal, "breakout_grade", "") or ""),
        "categories": categories,
    }
    for key in (
        "change_pct", "pivot_distance_pct", "breakout_conviction", "avg_vol20",
    ):
        value = _opt_float(signal, key)
        if value is not None:
            row[key] = value
    for key in ("above_sma50", "above_sma200"):
        flag = _opt_bool(signal, key)
        if flag is not None:
            row[key] = flag
    return row


SUPPORTED_SCHEMA_VERSIONS = (1, 2)
UNKNOWN_FRESHNESS = "UNKNOWN"
PRICE_SOURCE = "official_nse_bhavcopy"


def _history_freshness() -> tuple[dict[str, Any], str]:
    """Authoritative NSE session truth, or an explicit reason why it is unknown.

    Never guesses a session date. A caller that cannot reach the canonical
    bhavcopy store gets empty values plus the reason, so the UI can say
    "unavailable" instead of implying the scan's own clock is a data date.
    """
    try:
        from data.bhavcopy_runtime import official_history_freshness
        return dict(official_history_freshness()), ""
    except Exception as exc:  # canonical store unreadable in this process
        return {}, f"{type(exc).__name__}: official history freshness unavailable"


def scan_provenance(
    *,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    universe_failed: int | None = None,
    freshness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Separate WHEN THE SCAN RAN from WHICH MARKET SESSION IT READ.

    A recently executed job does not make its inputs current. These are two
    independent facts and the product must never collapse them into one date.
    """
    completed = completed_at or datetime.now(timezone.utc)
    started = started_at or completed
    if freshness is None:
        history, reason = _history_freshness()
    else:
        history, reason = dict(freshness), ""

    session = str(history.get("available_session") or "")[:10]
    expected = str(history.get("expected_latest_completed_session") or "")[:10]
    reason_code = str(history.get("reason_code") or "") or (UNKNOWN_FRESHNESS if not history else "")
    if not history and not reason:
        reason = "official history freshness returned no data"
    if not session and not reason:
        # History answered, but it has no usable session yet. Carry its own
        # machine-readable cause forward so the UI never shows a blank "why".
        reason = f"no official market session available ({reason_code or UNKNOWN_FRESHNESS})"

    # A start after the finish is incoherent input, not a zero-second scan.
    # Report the duration as unknown rather than emit a plausible-looking 0.0.
    elapsed = (completed - started).total_seconds()
    duration = round(elapsed, 3) if elapsed >= 0 else None
    scan_id = hashlib.sha256(
        f"{completed.isoformat()}|{session}|{expected}".encode("utf-8")
    ).hexdigest()[:16]

    return {
        "scan_id": scan_id,
        # when the job ran — never a data date
        "scan_started_at": started.isoformat(),
        "scan_completed_at": completed.isoformat(),
        "scan_duration_s": duration,
        # which market session the prices actually came from
        "market_session_date": session,
        "price_data_as_of": session,
        "expected_session_date": expected,
        "sessions_behind": history.get("stale_sessions"),
        "data_freshness": reason_code or UNKNOWN_FRESHNESS,
        "data_current": bool(history.get("current")) if history else False,
        "price_source": PRICE_SOURCE if session else "",
        "universe_failed": int(universe_failed) if universe_failed is not None else None,
        "provenance_available": bool(session),
        "provenance_reason": reason,
    }


def derive_trade_plan(row: Mapping[str, Any]) -> dict[str, Any]:
    """Deterministic plan geometry from levels the scanner actually produced.

    Every value here is arithmetic over real persisted levels. Nothing is
    invented: if a level is missing the derived field stays None and
    ``plan_missing`` names exactly which input was absent, so the desk can say
    why a plan is incomplete instead of printing a zero.
    """
    entry = _as_float(row.get("entry"))
    stop = _as_float(row.get("stop"))
    target = _as_float(row.get("target"))
    price = _as_float(row.get("price"))
    reference = entry if entry is not None else price

    missing = [name for name, value in
               (("entry", entry), ("stop", stop), ("target", target)) if value is None]

    risk_per_share = None
    if reference is not None and stop is not None and reference > stop:
        risk_per_share = reference - stop
    reward_per_share = None
    if reference is not None and target is not None and target > reference:
        reward_per_share = target - reference

    def _pct(numerator: float | None) -> float | None:
        if numerator is None or not reference:
            return None
        return round(numerator / reference * 100.0, 2)

    rr = None
    if risk_per_share and reward_per_share:
        rr = round(reward_per_share / risk_per_share, 2)

    return {
        "plan_reference_price": reference,
        "risk_per_share": round(risk_per_share, 2) if risk_per_share is not None else None,
        "reward_per_share": round(reward_per_share, 2) if reward_per_share is not None else None,
        "upside_pct": _pct(reward_per_share),
        "downside_pct": _pct(risk_per_share),
        "reward_risk": rr,
        "plan_complete": not missing,
        "plan_missing": missing,
    }


def build_scan_payload(
    names: Mapping[str, str],
    results: Iterable[Any],
    fno_symbols: Iterable[str] = (),
    *,
    scanned_at: datetime | None = None,
    scanned: int | None = None,
    approved_universe: int | None = None,
    started_at: datetime | None = None,
    universe_failed: int | None = None,
    freshness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fno = {str(s).upper() for s in fno_symbols}
    records = [_record(row, names, fno) for row in results]
    # deterministic ranking: score descending, symbol as the stable secondary key for ties
    records.sort(key=lambda row: (-float(row["score"] or 0.0), row["symbol"]))
    momentum = [r for r in records if "MOMENTUM" in r["signals"]]
    near = [r for r in records if "PRE_BREAKOUT" in r["signals"] and "MOMENTUM" not in r["signals"]]
    ready = [r for r in records if r["status"] == "Ready to trade"]
    now = scanned_at or datetime.now(timezone.utc)
    approved_n = int(approved_universe) if approved_universe is not None else len(names)
    normalized_n = len({str(s).strip().upper() for s in names if str(s).strip()})
    scanned_n = int(scanned) if scanned is not None else normalized_n
    qualified_n = len(records)
    return {
        "schema_version": 2,
        "scanned_at": now.isoformat(),
        "provenance": scan_provenance(
            started_at=started_at,
            completed_at=now,
            universe_failed=universe_failed,
            freshness=freshness,
        ),
        "approved_universe": approved_n,
        "scanned": scanned_n,
        "qualified_rows": qualified_n,
        "universe_size": scanned_n,
        "records": records,
        "summary": {
            "with_any_setup": qualified_n,
            "qualified": qualified_n,
            "momentum": len(momentum),
            "fno_momentum": sum(1 for r in momentum if r["fno_available"]),
            "near_breakout": len(near),
            # Canonical operator wording. ready_to_trade remains a legacy API
            # alias only; it does not mean the paper selection authority passed.
            "setup_ready": len(ready),
            "ready_to_trade": len(ready),
            "ready_to_trade_scope": "SCANNER_SETUP_ONLY",
            "extended": sum(1 for r in records if r["chase_risk"]),
        },
    }


def save_scan(payload: Mapping[str, Any], path: str | Path = DEFAULT_SCAN_PATH) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return target


def load_scan(path: str | Path = DEFAULT_SCAN_PATH) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) not in SUPPORTED_SCHEMA_VERSIONS:
            return None
        if not isinstance(payload.get("records"), list):
            return None
        if not isinstance(payload.get("provenance"), Mapping):
            # A pre-provenance artifact is still real data, but its market
            # session is genuinely unknown. Say so rather than let the UI read
            # scanned_at as if it were a data date.
            payload["provenance"] = {
                "scan_id": "",
                "scan_started_at": "",
                "scan_completed_at": str(payload.get("scanned_at") or ""),
                "scan_duration_s": None,
                "market_session_date": "",
                "price_data_as_of": "",
                "expected_session_date": "",
                "sessions_behind": None,
                "data_freshness": UNKNOWN_FRESHNESS,
                "data_current": False,
                "price_source": "",
                "universe_failed": None,
                "provenance_available": False,
                "provenance_reason": "scan predates session provenance (schema v1)",
            }
        return payload
    except Exception:
        return None


def watchlist_rows(payload: Mapping[str, Any] | None, limit: int = 25) -> list[dict[str, Any]]:
    if not payload:
        return []
    priority = {"Ready to trade": 0, "Watch for breakout": 1, "Wait for pullback": 2, "Watch": 3}
    rows = list(payload.get("records", []))
    rows.sort(key=lambda r: (priority.get(str(r.get("status")), 9), -float(r.get("score", 0) or 0),
                             str(r.get("symbol", ""))))       # symbol tiebreak → deterministic
    return rows[: max(0, int(limit))]


def scan_age_hours(payload: Mapping[str, Any] | None, *, now: datetime | None = None) -> float | None:
    if not payload or not payload.get("scanned_at"):
        return None
    try:
        stamp = datetime.fromisoformat(str(payload["scanned_at"]).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        age_seconds = (current - stamp).total_seconds()
        # Clock-corrupt/future timestamps are not fresh zero-age artifacts.
        # Their age is unknown, which makes every freshness gate fail closed.
        if age_seconds < 0:
            return None
        return age_seconds / 3600.0
    except Exception:
        return None


def scan_artifact_is_fresh(
    path: str | Path = DEFAULT_SCAN_PATH,
    *,
    max_age_s: float,
    now: datetime | None = None,
) -> bool:
    """True when the canonical scan JSON exists and scanned_at is within max_age_s."""
    age_h = scan_age_hours(load_scan(path), now=now)
    if age_h is None:
        return False
    try:
        return age_h * 3600.0 <= float(max_age_s)
    except (TypeError, ValueError):
        return False
