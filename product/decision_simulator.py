"""Historical decision simulator.

The button path runs a point-in-time replay of the production scanner,
recommendation workspace, and evaluate_candidate gates. Journal-only
classification remains a secondary overlay when paper cycles already exist.

This is research/backtest provenance. It never writes REAL_FORWARD_MARKET
rows and cannot change today's production policy.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence  # noqa: I001

from product.counterfactual_learning import (
    AVOIDED_LOSER,
    CORRECT_REJECTION,
    FLAT,
    GOOD_WAIT,
    MISSED_WINNER,
    RAN_AWAY,
    classify_forward,
)
from product.forward_evidence import BACKTEST

UNAVAILABLE = "UNAVAILABLE"
UNKNOWN = "UNKNOWN"
NOT_ENTERED = "NOT_ENTERED"
FAILED = "FAILED"
HISTORICAL_DECISION_UNAVAILABLE = "HISTORICAL_DECISION_UNAVAILABLE"
AMBIGUOUS_HISTORICAL_DECISION = "AMBIGUOUS_HISTORICAL_DECISION"
PIT_INTEGRITY_FAILED = "PIT_INTEGRITY_FAILED"
SUCCEEDED = "SUCCEEDED"
ENTRY_PERSISTED = "PERSISTED_DECISION"
ENTRY_CLOSE_AT_T = "OFFICIAL_CLOSE_AT_T_ASSUMPTION"
ENTRY_UNAVAILABLE = "UNAVAILABLE"

_ENTER_ACTIONS = frozenset({"BUY", "ENTER", "ENTER_NOW", "TAKEN", "TAKE"})
_VALID_ACTIONS = frozenset({"BUY", "WAIT", "AVOID", "REJECT", "NO_JUDGMENT"})

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "logs" / "product" / "decision_simulator.json"
SCHEMA_VERSION = 1

INCONCLUSIVE = "INCONCLUSIVE"

_REASON_HELPED = {
    "ENTRY_TOO_EXTENDED": "chase protection",
    "CHASE_RISK": "chase protection",
    "DD_GATE_FAILED": "due-diligence block",
    "LOW_QUALITY_SETUP": "weak setup rejection",
    "LIQUIDITY_FAILED": "liquidity filter",
    "PORTFOLIO_BLOCK": "portfolio concentration",
    "SECTOR_CAP": "sector cap",
    "CORRELATION_CAP": "correlation cap",
    "WAIT_FOR_ENTRY": "wait-for-price",
}

_REASON_HURT = {
    "SECTOR_WEAK": "sector filter",
    "EVIDENCE_POLICY_BLOCK": "evidence policy",
    "EMPIRICAL_GATE_FAILED": "empirical gate",
}


def report_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_DECISION_SIMULATOR")
    return Path(override) if override else DEFAULT_PATH


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    raw = json.dumps(
        [(r.get("symbol"), r.get("as_of"), r.get("decision"), r.get("reason_code")) for r in rows],
        default=str,
        sort_keys=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _decision_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        from product.autopilot_journal import load_journal
        journal = load_journal()
    except Exception:
        journal = {}
    for cycle in list(journal.get("cycles") or [])[-80:]:
        as_of = str(cycle.get("as_of") or cycle.get("recorded_at") or "")[:10]
        for taken in list(cycle.get("taken") or []):
            if not isinstance(taken, Mapping):
                continue
            rows.append({
                "symbol": str(taken.get("symbol") or "").upper(),
                "decision": "TAKEN",
                "reason_code": str(taken.get("reason_code") or "ENTER"),
                "as_of": as_of,
                "entry": taken.get("entry") or taken.get("entry_price"),
                "stop": taken.get("stop") or taken.get("stop_price"),
                "target": taken.get("target") or taken.get("target_price"),
                "sector": taken.get("sector") or "",
                "setup": taken.get("setup_label") or taken.get("setup") or "",
            })
        for skipped in list(cycle.get("rejections") or []) + list(cycle.get("waits") or []):
            if not isinstance(skipped, Mapping):
                continue
            decision = str(skipped.get("decision") or ("WAITED" if skipped.get("reason_code") == "WAIT_FOR_ENTRY" else "REJECTED"))
            rows.append({
                "symbol": str(skipped.get("symbol") or "").upper(),
                "decision": decision,
                "reason_code": str(skipped.get("reason_code") or skipped.get("reason") or ""),
                "as_of": as_of,
                "entry": skipped.get("entry") or skipped.get("hypothetical_entry"),
                "stop": skipped.get("stop") or skipped.get("hypothetical_stop"),
                "target": skipped.get("target") or skipped.get("hypothetical_target"),
                "sector": skipped.get("sector") or "",
                "setup": skipped.get("setup_label") or skipped.get("setup") or "",
            })
    try:
        from product.counterfactual_learning import ledger_path
        path = ledger_path()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines()[-400:]:
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if not isinstance(item, dict) or not item.get("symbol"):
                    continue
                rows.append({
                    "symbol": str(item.get("symbol") or "").upper(),
                    "decision": str(item.get("decision") or "REJECTED"),
                    "reason_code": str(item.get("reason_code") or ""),
                    "as_of": str(item.get("as_of") or "")[:10],
                    "entry": item.get("hypothetical_entry"),
                    "stop": item.get("hypothetical_stop"),
                    "target": item.get("hypothetical_target"),
                    "sector": item.get("sector") or "",
                    "setup": item.get("setup") or "",
                    "classification": item.get("classification"),
                    "forward_return_pct": (item.get("outcome") or {}).get("forward_return_pct") if isinstance(item.get("outcome"), Mapping) else None,
                })
    except Exception:
        pass
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("symbol") or not row.get("as_of"):
            continue
        key = (str(row["symbol"]), str(row["as_of"]), str(row["decision"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _forward_return(symbol: str, as_of: str, entry: Any) -> float | None:
    if not symbol or not as_of:
        return None
    try:
        from core.outcome_resolver import session_close_return
        result = session_close_return(symbol, as_of, horizon=5)
        if isinstance(result, tuple) and len(result) >= 2:
            return float(result[1])
        if isinstance(result, Mapping) and result.get("return_pct") is not None:
            return float(result["return_pct"])
        if isinstance(result, (int, float)):
            return float(result)
    except Exception:
        return None
    return None


def _classify(row: Mapping[str, Any]) -> str:
    existing = str(row.get("classification") or "")
    if existing:
        return existing
    fwd = row.get("forward_return_pct")
    if fwd is None:
        fwd = _forward_return(str(row.get("symbol") or ""), str(row.get("as_of") or ""), row.get("entry"))
    if fwd is None:
        return INCONCLUSIVE
    if str(row.get("decision") or "").upper() in {"TAKEN", "ENTER"}:
        return INCONCLUSIVE if fwd is None else FLAT
    return classify_forward(
        entry=row.get("entry") if row.get("entry") is not None else None,
        stop=row.get("stop") if row.get("stop") is not None else None,
        target=row.get("target") if row.get("target") is not None else None,
        forward_return_pct=float(fwd),
        later_entered=False,
    )


def _filter_scores(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    helped: dict[str, int] = {}
    hurt: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("reason_code") or "")
        klass = str(row.get("classification") or "")
        label = _REASON_HELPED.get(reason)
        if label and klass in {CORRECT_REJECTION, AVOIDED_LOSER, GOOD_WAIT}:
            helped[label] = helped.get(label, 0) + 1
        hurt_label = _REASON_HURT.get(reason)
        if hurt_label and klass == MISSED_WINNER:
            hurt[hurt_label] = hurt.get(hurt_label, 0) + 1
    top_helped = [name for name, _ in sorted(helped.items(), key=lambda kv: kv[1], reverse=True)[:4]]
    top_hurt = [name for name, _ in sorted(hurt.items(), key=lambda kv: kv[1], reverse=True)[:4]]
    return top_helped, top_hurt


def _sector_edge(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare high vs low sector scores among already-classified rows. Not a production change."""
    high: list[Mapping[str, Any]] = []
    low: list[Mapping[str, Any]] = []
    for row in rows:
        try:
            score = float(row.get("sector_leadership_score"))
        except (TypeError, ValueError):
            continue
        if score >= 60:
            high.append(row)
        elif score < 40:
            low.append(row)

    def _rate(items: Sequence[Mapping[str, Any]], klass: str) -> float | None:
        if len(items) < 8:
            return None
        return round(100.0 * sum(1 for r in items if r.get("classification") == klass) / len(items), 1)

    return {
        "available": bool(high or low),
        "high_n": len(high),
        "low_n": len(low),
        "high_missed_winner_pct": _rate(high, MISSED_WINNER),
        "low_missed_winner_pct": _rate(low, MISSED_WINNER),
        "high_correct_rejection_pct": _rate(high, CORRECT_REJECTION),
        "low_correct_rejection_pct": _rate(low, CORRECT_REJECTION),
        "insufficient_n": len(high) < 8 or len(low) < 8,
        "not_promotion_evidence": True,
        "note": (
            "BACKTEST only. Sector score must already be on the historical row. "
            "Missing scores stay out of this split. n<8 stays inconclusive."
        ),
    }


def load_latest(path: str | Path | None = None) -> dict[str, Any]:
    try:
        from product.historical_replay import load_latest as load_replay

        replay = load_replay()
    except Exception:
        replay = {}
    local = _read_json(report_path(path))
    if replay.get("status") == "RUNNING" or (replay and not local.get("engine")):
        merged = dict(local)
        merged.update(replay)
        merged["available"] = True
        merged["provenance"] = BACKTEST
        merged["live_locked"] = True
        return merged
    if local:
        local.setdefault("available", True)
        return local
    return replay


def run_decision_simulator(
    *,
    force: bool = False,
    path: str | Path | None = None,
    async_job: bool = False,
    sessions: int = 8,
    universe_limit: int = 40,
    symbols: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Replay production decisions at historical dates, then overlay journal outcomes."""
    from product.historical_replay import (
        load_latest as load_replay,
        run_historical_replay,
        start_replay_async,
    )

    if async_job:
        started = start_replay_async(
            force=force,
            sessions=sessions,
            universe_limit=universe_limit,
            symbols=symbols,
        )
        latest = load_replay()
        latest.update(started)
        latest["available"] = True
        return latest

    replay = run_historical_replay(
        force=force,
        sessions=sessions,
        universe_limit=universe_limit,
        symbols=symbols,
    )
    target = report_path(path)
    rows = _decision_rows()
    version = _fingerprint(rows)
    cached = _read_json(target)
    expected_n = int(replay.get("decisions_tested") or 0) or len(rows)
    if (
        not force
        and cached.get("version") == version
        and cached.get("engine")
        and cached.get("run_id") == replay.get("run_id")
        and int(cached.get("decisions_tested") or 0) == expected_n
    ):
        cached["cache_hit"] = True
        cached["provenance"] = BACKTEST
        cached["live_locked"] = True
        return cached

    classified: list[dict[str, Any]] = []
    counts = {
        CORRECT_REJECTION: 0,
        MISSED_WINNER: 0,
        AVOIDED_LOSER: 0,
        GOOD_WAIT: 0,
        RAN_AWAY: 0,
        FLAT: 0,
        INCONCLUSIVE: 0,
        "TAKEN": 0,
        "REJECTED": 0,
        "WAITED": 0,
    }
    for row in rows:
        klass = _classify(row)
        decision = str(row.get("decision") or "REJECTED").upper()
        if decision.startswith("TAKE") or decision == "ENTER":
            counts["TAKEN"] += 1
        elif decision.startswith("WAIT"):
            counts["WAITED"] += 1
        else:
            counts["REJECTED"] += 1
        if klass in counts:
            counts[klass] += 1
        classified.append({
            **row,
            "classification": klass,
            "not_pnl": True,
            "provenance": BACKTEST,
        })

    helped, hurt = _filter_scores(classified)
    simple = (
        f"{len(rows)} historical decisions tested. "
        f"Would take {counts['TAKEN']}. Rejected {counts['REJECTED']}. "
        f"Correct rejections {counts[CORRECT_REJECTION]}. "
        f"Missed winners {counts[MISSED_WINNER]}."
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "version": version,
        "provenance": BACKTEST,
        "cache_hit": False,
        "live_locked": True,
        "not_promotion_evidence": True,
        "engine": replay.get("engine"),
        "run_id": replay.get("run_id"),
        "status": replay.get("status"),
        "available": True,
        "period_start": replay.get("period_start"),
        "period_end": replay.get("period_end"),
        "trading_sessions": replay.get("trading_sessions"),
        "universe_observations": replay.get("universe_observations"),
        "stocks_evaluated": replay.get("stocks_evaluated"),
        "decision_candidates": replay.get("decision_candidates") or replay.get("decisions_tested"),
        "BUY": replay.get("BUY"),
        "WAIT": replay.get("WAIT"),
        "AVOID": replay.get("AVOID"),
        "REJECT": replay.get("REJECT"),
        "outcomes_matured": replay.get("outcomes_matured"),
        "open_unresolved": replay.get("open_unresolved"),
        "session_summaries": replay.get("session_summaries") or [],
        "decisions": replay.get("decisions") or replay.get("rows") or [],
        "journal_overlay": {
            "decisions_tested": len(rows),
            "would_take": counts["TAKEN"],
            "rejected": counts["REJECTED"],
            "waited": counts["WAITED"],
            "correct_rejections": counts[CORRECT_REJECTION],
            "missed_winners": counts[MISSED_WINNER],
            "avoided_losers": counts[AVOIDED_LOSER],
            "good_waits": counts[GOOD_WAIT],
            "ran_away": counts[RAN_AWAY],
            "flat": counts[FLAT],
            "inconclusive": counts[INCONCLUSIVE],
            "rows": classified[:250],
        },
        "decisions_tested": int(replay.get("decisions_tested") or len(classified)),
        "would_take": int(replay.get("would_take") or replay.get("BUY") or counts["TAKEN"]),
        "rejected": int(replay.get("rejected") or counts["REJECTED"]),
        "waited": int(replay.get("waited") or counts["WAITED"]),
        "correct_rejections": int(replay.get("correct_rejections") or counts[CORRECT_REJECTION]),
        "missed_winners": int(replay.get("missed_winners") or counts[MISSED_WINNER]),
        "avoided_losers": int(replay.get("avoided_losers") or counts[AVOIDED_LOSER]),
        "good_waits": int(replay.get("good_waits") or counts[GOOD_WAIT]),
        "ran_away": int(replay.get("ran_away") or counts[RAN_AWAY]),
        "flat": int(replay.get("flat") or counts[FLAT]),
        "inconclusive": int(replay.get("inconclusive") or counts[INCONCLUSIVE]),
        "filters_helped": helped,
        "filters_hurt": hurt,
        "simple": replay.get("simple") or simple,
        "note": (
            str(replay.get("note") or "This does not change REAL_FORWARD_MARKET promotion stats and does not open paper trades.")
            + " Journal overlay classifies already-recorded paper cycles separately."
        ).strip(),
        "sector_edge": _sector_edge(classified),
        "rows": replay.get("decisions") or classified[:250],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return payload


def _norm_action(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if raw in {"ENTER", "ENTER_NOW", "TAKEN", "TAKE"}:
        return "BUY"
    if raw in {"WAITED", "WAIT_FOR_ENTRY"}:
        return "WAIT"
    if raw in {"REJECTED", "BLOCK", "PORTFOLIO_BLOCK"}:
        return "REJECT"
    if raw in _VALID_ACTIONS:
        return raw
    return ""


def _is_enter(action: str) -> bool:
    return _norm_action(action) in {"BUY"} or str(action or "").upper() in _ENTER_ACTIONS


def _num(value: Any) -> float | None:
    try:
        if value in (None, "", UNAVAILABLE, UNKNOWN):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _honest(value: Any, *, empty: str = UNAVAILABLE) -> Any:
    if value in (None, "", [], {}):
        return empty
    return value


def _bar_date(idx: Any) -> str:
    return str(getattr(idx, "date", lambda: idx)())[:10]


def _frame_close_at(frame: Any, as_of: str) -> tuple[float | None, str | None, list[str]]:
    warnings: list[str] = []
    if frame is None or getattr(frame, "empty", True):
        return None, None, ["Official bars at T are UNAVAILABLE"]
    last = frame.index[-1]
    last_day = _bar_date(last)
    if last_day > str(as_of)[:10]:
        warnings.append("LOOKAHEAD: decision-time frame contained a bar after T; rejected")
        return None, None, warnings
    try:
        close = float(frame["close"].iloc[-1])
    except Exception:
        return None, last_day, ["Close at T is UNAVAILABLE"]
    return close, last_day, warnings


def _later_session_bars(symbol: str, as_of: str, *, ohlcv_fn=None, horizon: int = 10) -> list[dict[str, Any]]:
    """Bars strictly after T. Future prices are outcome-only."""
    from datetime import date, timedelta

    from product.historical_replay import ohlcv_as_of

    try:
        end = (date.fromisoformat(str(as_of)[:10]) + timedelta(days=max(21, int(horizon) * 3))).isoformat()
        frame = ohlcv_as_of(symbol, end, ohlcv_fn=ohlcv_fn)
    except Exception:
        return []
    if frame is None or getattr(frame, "empty", True):
        return []
    out: list[dict[str, Any]] = []
    cutoff = str(as_of)[:10]
    for idx, row in frame.iterrows():
        day = _bar_date(idx)
        if day <= cutoff:
            continue
        rec = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        rec["date"] = day
        out.append(rec)
        if len(out) >= int(horizon):
            break
    return out


def _reject_future_publication(snapshot: Mapping[str, Any] | None, as_of: str, label: str) -> dict[str, Any]:
    snap = dict(snapshot or {})
    pub = str(snap.get("latest_publication") or snap.get("available_from") or "")[:10]
    if pub and pub > str(as_of)[:10]:
        return {
            "available": False,
            "status": UNAVAILABLE,
            "reason": f"{label} publication {pub} is after T={as_of[:10]}; rejected as lookahead",
        }
    if not snap:
        return {"available": False, "status": UNAVAILABLE}
    snap.setdefault("available", True)
    return snap


def _pit_news(symbol: str, as_of: str) -> list[dict[str, Any]]:
    try:
        from product.pit_events import get_events

        rows = get_events(symbol, as_of=as_of)
    except Exception:
        return []
    kept: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        pub = str(row.get("available_from") or "")[:10]
        if pub and pub > str(as_of)[:10]:
            continue
        kept.append({
            "headline": row.get("headline") or UNKNOWN,
            "available_from": pub or UNAVAILABLE,
            "event_class": row.get("event_class") or UNKNOWN,
            "source": row.get("source") or UNKNOWN,
        })
    return kept


def _row_symbol(row: Mapping[str, Any] | None) -> str:
    return str((row or {}).get("symbol") or "").strip().upper()


def _row_day(row: Mapping[str, Any] | None) -> str:
    raw = (row or {}).get("market_as_of") or (row or {}).get("as_of") or (row or {}).get("decision_time") or ""
    return str(raw)[:10]


def _row_decision_id(row: Mapping[str, Any] | None) -> str:
    if not row:
        return ""
    explicit = str(row.get("decision_id") or row.get("freeze_id") or "").strip()
    if explicit:
        return explicit
    return "|".join([
        _row_day(row),
        _row_symbol(row),
        str(row.get("decision") or row.get("raw_decision") or ""),
        str(row.get("reason_code") or ""),
    ])


def _identity_error(row: Mapping[str, Any], symbol: str, as_of: str) -> str:
    name = str(symbol or "").strip().upper()
    session = str(as_of or "")[:10]
    found_symbol = _row_symbol(row)
    found_day = _row_day(row)
    if name and found_symbol and found_symbol != name:
        return (
            f"decision_id identity mismatch: persisted symbol {found_symbol} "
            f"does not match requested {name}"
        )
    if session and found_day and found_day != session:
        return (
            f"decision_id identity mismatch: persisted date {found_day} "
            f"does not match requested {session}"
        )
    return ""


def _collect_historical_rows(*, symbol: str, as_of: str) -> list[dict[str, Any]]:
    from product.decision_journal import list_for_symbol

    name = str(symbol or "").strip().upper()
    session = str(as_of or "")[:10]
    rows: list[dict[str, Any]] = []
    if name:
        for item in list_for_symbol(name, as_of=session, limit=50):
            if isinstance(item, Mapping):
                rows.append(dict(item))
    try:
        from product.autopilot_journal import load_journal

        journal = load_journal()
    except Exception:
        journal = {}
    for cycle in list(journal.get("cycles") or []):
        cycle_day = str(cycle.get("as_of") or cycle.get("recorded_at") or "")[:10]
        if session and cycle_day != session:
            continue
        for bucket in ("taken", "rejections", "waits"):
            for item in list(cycle.get(bucket) or []):
                if not isinstance(item, Mapping):
                    continue
                if name and str(item.get("symbol") or "").upper() != name:
                    continue
                row = dict(item)
                row.setdefault("symbol", name)
                row.setdefault("as_of", cycle_day)
                row.setdefault("market_as_of", cycle_day)
                if bucket == "taken":
                    row.setdefault("decision", "BUY")
                rows.append(row)
    try:
        from product.historical_replay import ledger_path

        path = ledger_path()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if not isinstance(item, Mapping):
                    continue
                if name and str(item.get("symbol") or "").upper() != name:
                    continue
                item_day = str(item.get("as_of") or item.get("market_as_of") or "")[:10]
                if session and item_day != session:
                    continue
                rows.append(dict(item))
    except Exception:
        pass
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = _row_decision_id(row)
        if not key or key in seen:
            continue
        seen.add(key)
        tagged = dict(row)
        tagged["decision_id"] = key
        unique.append(tagged)
    return unique


def _find_historical_decision(
    *,
    symbol: str,
    as_of: str,
    decision_id: str = "",
) -> dict[str, Any]:
    from product.decision_journal import get as dj_get

    wanted_id = str(decision_id or "").strip()
    if wanted_id:
        found = dj_get(wanted_id)
        if not found:
            for row in _collect_historical_rows(symbol=symbol, as_of=as_of):
                if _row_decision_id(row) == wanted_id:
                    found = row
                    break
        if not found and (symbol or as_of):
            for row in _collect_historical_rows(symbol=str(symbol or "").strip().upper(), as_of=""):
                if _row_decision_id(row) == wanted_id:
                    found = row
                    break
        if not found:
            return {"status": FAILED, "error": f"decision_id {wanted_id} does not match a persisted decision"}
        mismatch = _identity_error(found, symbol, as_of)
        if mismatch:
            return {"status": FAILED, "error": mismatch, "row": dict(found)}
        tagged = dict(found)
        tagged["decision_id"] = _row_decision_id(tagged)
        return {"status": "FOUND", "row": tagged}

    matches = _collect_historical_rows(symbol=symbol, as_of=as_of)
    if not matches:
        return {"status": "MISSING"}
    if len(matches) > 1:
        return {
            "status": AMBIGUOUS_HISTORICAL_DECISION,
            "matches": [
                {
                    "decision_id": _row_decision_id(row),
                    "symbol": _row_symbol(row),
                    "as_of": _row_day(row),
                    "decision": row.get("decision") or row.get("raw_decision") or UNKNOWN,
                    "reason_code": row.get("reason_code") or UNKNOWN,
                    "decision_time": row.get("decision_time") or _row_day(row),
                }
                for row in matches
            ],
        }
    return {"status": "FOUND", "row": matches[0]}


def _default_alternative(actual: str) -> str:
    return "WAIT" if _is_enter(actual) else "BUY"


def _outcome_for_action(
    *,
    action: str,
    entry: float | None,
    stop: float | None,
    target: float | None,
    later_bars: Sequence[Mapping[str, Any]],
    entry_source: str = ENTRY_UNAVAILABLE,
) -> dict[str, Any]:
    from product.decision_outcomes import path_metrics

    if not _is_enter(action):
        market_move = None
        if later_bars:
            first = _num((later_bars[0] or {}).get("close") if later_bars else None)
            last = _num((later_bars[-1] or {}).get("close") if later_bars else None)
            if first and last and first > 0:
                market_move = round((last - first) / first * 100.0, 3)
        return {
            "status": NOT_ENTERED,
            "methodology": "No position. Subsequent prices describe the market, not a trade.",
            "simulated_entry": UNAVAILABLE,
            "entry_source": ENTRY_UNAVAILABLE,
            "stop": _honest(stop),
            "hypothetical_return_pct": UNAVAILABLE,
            "hypothetical_r": UNAVAILABLE,
            "mfe_pct": UNAVAILABLE,
            "mae_pct": UNAVAILABLE,
            "subsequent_market_return_pct": _honest(market_move),
            "bars_used": len(list(later_bars)),
        }
    if entry is None:
        return {
            "status": UNAVAILABLE,
            "methodology": "BUY was simulated but no entry price existed at T.",
            "simulated_entry": UNAVAILABLE,
            "entry_source": ENTRY_UNAVAILABLE,
            "stop": _honest(stop),
            "hypothetical_return_pct": UNAVAILABLE,
            "hypothetical_r": UNAVAILABLE,
            "mfe_pct": UNAVAILABLE,
            "mae_pct": UNAVAILABLE,
            "subsequent_market_return_pct": UNAVAILABLE,
            "bars_used": 0,
        }
    if not later_bars:
        return {
            "status": UNAVAILABLE,
            "methodology": "No official bars after T. Outcome stays UNAVAILABLE.",
            "simulated_entry": entry,
            "stop": _honest(stop),
            "hypothetical_return_pct": UNAVAILABLE,
            "hypothetical_r": UNAVAILABLE,
            "mfe_pct": UNAVAILABLE,
            "mae_pct": UNAVAILABLE,
            "subsequent_market_return_pct": UNAVAILABLE,
            "bars_used": 0,
        }
    metrics = path_metrics(entry=entry, stop=stop, target=target, bars=list(later_bars))
    assumption = (
        " Entry is the official close at T, not a persisted QuantTerm entry."
        if entry_source == ENTRY_CLOSE_AT_T else
        " Entry is the persisted historical level."
        if entry_source == ENTRY_PERSISTED else
        ""
    )
    return {
        "status": "COMPUTED",
        "methodology": (
            "Official bars after T only. Frozen entry/stop from decision-time. "
            "Stop-before-target. Simulated first-touch / close path — not live execution."
            + assumption
        ),
        "simulated_entry": entry,
        "entry_source": entry_source,
        "stop": _honest(stop),
        "hypothetical_return_pct": _honest(metrics.get("return_pct")),
        "hypothetical_r": _honest(metrics.get("r_multiple")),
        "mfe_pct": _honest(metrics.get("mfe_pct")),
        "mae_pct": _honest(metrics.get("mae_pct")),
        "target_status": metrics.get("target_status") or UNKNOWN,
        "stop_status": metrics.get("stop_status") or UNKNOWN,
        "subsequent_market_return_pct": _honest(metrics.get("return_pct")),
        "bars_used": metrics.get("bars_used") or len(list(later_bars)),
        "path_note": metrics.get("path_note"),
    }


def _compare(actual_outcome: Mapping[str, Any], simulated_outcome: Mapping[str, Any], *, actual: str, alternative: str) -> dict[str, Any]:
    a_ret = actual_outcome.get("hypothetical_return_pct")
    s_ret = simulated_outcome.get("hypothetical_return_pct")
    if a_ret in {UNAVAILABLE, UNKNOWN, None} and s_ret in {UNAVAILABLE, UNKNOWN, None}:
        delta = UNAVAILABLE
    elif a_ret in {UNAVAILABLE, UNKNOWN, None} or s_ret in {UNAVAILABLE, UNKNOWN, None}:
        delta = UNAVAILABLE
    else:
        try:
            delta = round(float(s_ret) - float(a_ret), 3)
        except (TypeError, ValueError):
            delta = UNAVAILABLE
    return {
        "actual_action": actual,
        "simulated_action": alternative,
        "actual_return_pct": actual_outcome.get("hypothetical_return_pct"),
        "simulated_return_pct": simulated_outcome.get("hypothetical_return_pct"),
        "return_delta_pct": delta,
        "actual_status": actual_outcome.get("status"),
        "simulated_status": simulated_outcome.get("status"),
        "note": (
            "Counterfactual is hypothetical. It is not booked P&L and does not "
            "change REAL_FORWARD_MARKET promotion stats."
        ),
    }


def _replay_at_t(
    symbol: str,
    as_of: str,
    *,
    ohlcv_fn=None,
    analyzer=None,
    decide_fn=None,
) -> dict[str, Any]:
    from product.historical_replay import decide_session, scan_session

    scan = scan_session(as_of, [symbol], ohlcv_fn=ohlcv_fn, analyzer=analyzer)
    rows = decide_session(as_of, scan, decide_fn=decide_fn, persist_ledger=False)
    match = next((row for row in rows if str(row.get("symbol") or "").upper() == symbol), None)
    if not match:
        return {
            "status": UNAVAILABLE,
            "decision": UNAVAILABLE,
            "reason": "Committee replay produced no judgment at T",
            "pit": {"future_evidence_used": False},
        }
    pit = dict(match.get("pit") or {})
    if pit.get("future_evidence_used"):
        return {
            "status": FAILED,
            "decision": UNAVAILABLE,
            "reason": "LOOKAHEAD: reconstructed decision saw a bar after T",
            "pit": pit,
        }
    return {
        "status": SUCCEEDED,
        "decision": match.get("decision") or UNAVAILABLE,
        "reason_code": match.get("reason_code") or UNKNOWN,
        "reasons": match.get("reasons") or [],
        "entry": match.get("entry"),
        "stop": match.get("stop"),
        "target": match.get("target"),
        "effective_confirmation_count": match.get("effective_confirmation_count"),
        "evidence_family_votes": match.get("evidence_family_votes") or {},
        "pit": pit,
        "pit_financial": match.get("pit_financial"),
        "pit_research": match.get("pit_research"),
        "pit_sector": match.get("pit_sector"),
    }


def simulate_past_decision(
    *,
    symbol: str = "",
    as_of: str = "",
    alternative: str = "",
    decision_id: str = "",
    ohlcv_fn=None,
    analyzer=None,
    decide_fn=None,
    replay_engine: bool = True,
    horizon: int = 10,
) -> dict[str, Any]:
    """Replay one persisted decision at T and apply a counterfactual action.

    Decision-time features use only information available at T. Subsequent
    official bars are used only to measure the hypothetical path.
    """
    name = str(symbol or "").strip().upper()
    session = str(as_of or "")[:10]
    warnings: list[str] = []
    try:
        if not name and not str(decision_id or "").strip():
            return {
                "schema_version": 1,
                "kind": "PAST_DECISION_SIMULATION",
                "status": FAILED,
                "available": False,
                "provenance": BACKTEST,
                "live_locked": True,
                "error": "symbol and historical timestamp are required",
                "warnings": ["Missing symbol or decision_id"],
                "original": {"action": UNAVAILABLE},
                "simulated": {"action": UNAVAILABLE},
            }
        located = _find_historical_decision(symbol=name, as_of=session, decision_id=str(decision_id or ""))
        if located.get("status") == FAILED:
            return {
                "schema_version": 1,
                "kind": "PAST_DECISION_SIMULATION",
                "status": FAILED,
                "available": False,
                "provenance": BACKTEST,
                "live_locked": True,
                "symbol": name or UNAVAILABLE,
                "as_of": session or UNAVAILABLE,
                "decision_id": str(decision_id or "") or UNAVAILABLE,
                "error": located.get("error") or "decision identity mismatch",
                "warnings": ["Persisted decision identity did not match the requested symbol/date"],
                "original": {"action": UNAVAILABLE, "entry": UNAVAILABLE, "entry_source": ENTRY_UNAVAILABLE},
                "simulated": {"action": UNAVAILABLE, "entry": UNAVAILABLE, "entry_source": ENTRY_UNAVAILABLE},
            }
        if located.get("status") == AMBIGUOUS_HISTORICAL_DECISION:
            return {
                "schema_version": 1,
                "kind": "PAST_DECISION_SIMULATION",
                "status": AMBIGUOUS_HISTORICAL_DECISION,
                "available": False,
                "provenance": BACKTEST,
                "live_locked": True,
                "symbol": name,
                "as_of": session,
                "error": "Multiple persisted decisions match this symbol and date. Select a decision_id.",
                "matches": located.get("matches") or [],
                "warnings": ["Exact decision is ambiguous until decision_id is supplied"],
                "original": {"action": UNAVAILABLE, "entry": UNAVAILABLE, "entry_source": ENTRY_UNAVAILABLE},
                "simulated": {"action": UNAVAILABLE, "entry": UNAVAILABLE, "entry_source": ENTRY_UNAVAILABLE},
            }
        historical = located.get("row") if located.get("status") == "FOUND" else None
        if historical and not name:
            name = _row_symbol(historical)
        if historical and not session:
            session = _row_day(historical)
        if not name or not session:
            return {
                "schema_version": 1,
                "kind": "PAST_DECISION_SIMULATION",
                "status": FAILED,
                "available": False,
                "provenance": BACKTEST,
                "live_locked": True,
                "error": "symbol and historical timestamp are required",
                "warnings": ["Missing symbol or as_of"],
                "original": {"action": UNAVAILABLE, "entry": UNAVAILABLE, "entry_source": ENTRY_UNAVAILABLE},
                "simulated": {"action": UNAVAILABLE, "entry": UNAVAILABLE, "entry_source": ENTRY_UNAVAILABLE},
            }

        from product.historical_replay import ohlcv_as_of
        from product.pit_query import get_financial_snapshot, get_research_snapshot, get_sector_context

        frame = ohlcv_as_of(name, session, ohlcv_fn=ohlcv_fn)
        close_t, max_bar, bar_warnings = _frame_close_at(frame, session)
        warnings.extend(bar_warnings)

        financial = _reject_future_publication(
            get_financial_snapshot(name, as_of=session), session, "financials",
        )
        if financial.get("reason"):
            warnings.append(str(financial["reason"]))
        research = _reject_future_publication(
            get_research_snapshot(name, as_of=session), session, "research",
        )
        sector = get_sector_context(name, as_of=session)
        if str(sector.get("status") or "") in {"UNVERIFIED", "UNAVAILABLE", "SECTOR_MEMBERSHIP_APPROXIMATE"}:
            warnings.append(str(sector.get("note") or "Historical sector context is not versioned"))
        news = _pit_news(name, session)

        actual = _norm_action(
            (historical or {}).get("decision") or (historical or {}).get("raw_decision")
        ) if historical else ""
        if historical and not actual:
            actual = UNKNOWN
            warnings.append("Persisted decision action could not be normalized")
        alt = _norm_action(alternative) or (_default_alternative(actual) if actual and actual != UNKNOWN else "")
        if alternative and not _norm_action(alternative):
            return {
                "schema_version": 1,
                "kind": "PAST_DECISION_SIMULATION",
                "status": FAILED,
                "available": False,
                "symbol": name,
                "as_of": session,
                "provenance": BACKTEST,
                "live_locked": True,
                "error": f"Unknown alternative action {alternative!r}",
                "warnings": warnings,
            }

        reconstructed: dict[str, Any] = {
            "status": UNAVAILABLE,
            "decision": UNAVAILABLE,
            "reason": "Engine replay skipped" if not replay_engine else "Engine replay not run",
        }
        if replay_engine:
            # Reconstruction is the PIT authority for every caller, including
            # GET/POST which do not inject bars. Missing official close at T is
            # a reconstruction input, not a license to skip integrity.
            reconstructed = _replay_at_t(
                name, session, ohlcv_fn=ohlcv_fn, analyzer=analyzer, decide_fn=decide_fn,
            )
            if reconstructed.get("status") == FAILED:
                warnings.append(str(reconstructed.get("reason") or "Reconstructed decision failed"))

        persisted_entry = _num(
            (historical or {}).get("entry") if historical else None
        )
        if persisted_entry is None and historical:
            persisted_entry = _num(historical.get("hypothetical_entry"))
        original_entry_source = ENTRY_PERSISTED if persisted_entry is not None else ENTRY_UNAVAILABLE
        simulated_entry = persisted_entry
        simulated_entry_source = original_entry_source
        if simulated_entry is None and close_t is not None and historical:
            simulated_entry = close_t
            simulated_entry_source = ENTRY_CLOSE_AT_T
            warnings.append(
                "No persisted original entry. Counterfactual uses official close at T "
                f"({ENTRY_CLOSE_AT_T}); that is not a QuantTerm-recorded entry."
            )
        stop = _num((historical or {}).get("stop") or (historical or {}).get("hypothetical_stop")) if historical else None
        target = _num((historical or {}).get("target") or (historical or {}).get("hypothetical_target")) if historical else None
        if historical and persisted_entry is None and simulated_entry is None:
            warnings.append("No persisted entry and no official close at T; BUY path stays UNAVAILABLE")
        if historical and stop is None:
            warnings.append("No defined stop at T; R-multiple stays UNAVAILABLE")

        reconstructed_pit = dict(reconstructed.get("pit") or {})
        lookahead = bool(
            reconstructed_pit.get("future_evidence_used")
            or (
                reconstructed.get("status") == FAILED
                and "LOOKAHEAD" in str(reconstructed.get("reason") or "").upper()
            )
            or any("LOOKAHEAD" in str(item).upper() for item in bar_warnings)
        )
        if lookahead:
            pit_status = PIT_INTEGRITY_FAILED
        elif not replay_engine:
            pit_status = "PIT_OK" if historical else UNAVAILABLE
        elif reconstructed.get("status") == SUCCEEDED:
            pit_status = "PIT_OK" if historical else UNAVAILABLE
        else:
            # Replay was requested but produced no PIT-clean verdict.
            pit_status = UNAVAILABLE
        if lookahead:
            warnings.append("PIT look-ahead violation: counterfactual is not trustworthy")

        later = _later_session_bars(name, session, ohlcv_fn=ohlcv_fn, horizon=horizon)
        if any(str(bar.get("date") or "")[:10] <= session for bar in later):
            warnings.append("LOOKAHEAD: outcome series included T or earlier; those bars were dropped")
            later = [bar for bar in later if str(bar.get("date") or "")[:10] > session]
            lookahead = True
            pit_status = PIT_INTEGRITY_FAILED
        if not later and historical:
            warnings.append("No official subsequent bars after T")

        withheld = {
            "status": UNAVAILABLE,
            "methodology": "PIT integrity failed. Counterfactual withheld.",
            "simulated_entry": UNAVAILABLE,
            "entry_source": ENTRY_UNAVAILABLE,
            "stop": _honest(stop if historical else None),
            "hypothetical_return_pct": UNAVAILABLE,
            "hypothetical_r": UNAVAILABLE,
            "mfe_pct": UNAVAILABLE,
            "mae_pct": UNAVAILABLE,
            "subsequent_market_return_pct": UNAVAILABLE,
            "bars_used": 0,
            "trustworthy": False,
        }
        actual_action = actual or UNAVAILABLE
        simulated_action = alt or UNAVAILABLE
        if lookahead:
            actual_outcome = dict(withheld)
            actual_outcome["methodology"] = "PIT integrity failed. Actual path withheld."
            simulated_outcome = dict(withheld)
        elif historical:
            actual_outcome = _outcome_for_action(
                action=actual_action if actual_action != UNKNOWN else "",
                entry=persisted_entry,
                stop=stop,
                target=target,
                later_bars=later,
                entry_source=original_entry_source,
            )
            simulated_outcome = _outcome_for_action(
                action=simulated_action if simulated_action != UNAVAILABLE else "",
                entry=simulated_entry,
                stop=stop,
                target=target,
                later_bars=later,
                entry_source=simulated_entry_source,
            )
        else:
            actual_outcome = {
                "status": UNAVAILABLE,
                "methodology": "No persisted historical decision. Outcome is not invented.",
                "simulated_entry": UNAVAILABLE,
                "entry_source": ENTRY_UNAVAILABLE,
                "stop": UNAVAILABLE,
                "hypothetical_return_pct": UNAVAILABLE,
                "hypothetical_r": UNAVAILABLE,
                "mfe_pct": UNAVAILABLE,
                "mae_pct": UNAVAILABLE,
                "subsequent_market_return_pct": UNAVAILABLE,
                "bars_used": 0,
            }
            simulated_outcome = {
                "status": UNAVAILABLE,
                "methodology": "Counterfactual was not run because the historical decision is missing or the alternative is unknown.",
                "simulated_entry": UNAVAILABLE,
                "entry_source": ENTRY_UNAVAILABLE,
                "stop": UNAVAILABLE,
                "hypothetical_return_pct": UNAVAILABLE,
                "hypothetical_r": UNAVAILABLE,
                "mfe_pct": UNAVAILABLE,
                "mae_pct": UNAVAILABLE,
                "subsequent_market_return_pct": UNAVAILABLE,
                "bars_used": 0,
            }

        if lookahead:
            status = PIT_INTEGRITY_FAILED
            reconstructed = dict(reconstructed)
            pit_aligned = dict(reconstructed.get("pit") or {})
            pit_aligned["future_evidence_used"] = True
            reconstructed["pit"] = pit_aligned
            if reconstructed.get("status") != FAILED:
                reconstructed["status"] = PIT_INTEGRITY_FAILED
        elif historical:
            status = SUCCEEDED
        else:
            status = HISTORICAL_DECISION_UNAVAILABLE
        if lookahead:
            error = "PIT look-ahead violation; counterfactual is not trustworthy"
            comparison = {
                "actual_action": actual_action,
                "simulated_action": simulated_action,
                "actual_return_pct": UNAVAILABLE,
                "simulated_return_pct": UNAVAILABLE,
                "return_delta_pct": UNAVAILABLE,
                "trustworthy": False,
                "note": "PIT integrity failed. Comparison withheld.",
            }
        elif historical:
            error = None
            comparison = _compare(actual_outcome, simulated_outcome, actual=actual_action, alternative=simulated_action)
        else:
            error = "No persisted QuantTerm decision for this symbol at that timestamp"
            comparison = {
                "actual_action": UNAVAILABLE,
                "simulated_action": UNAVAILABLE,
                "return_delta_pct": UNAVAILABLE,
                "note": "No persisted decision to compare.",
            }
        payload = {
            "schema_version": 1,
            "kind": "PAST_DECISION_SIMULATION",
            "status": status,
            "available": bool(historical) and not lookahead,
            "provenance": BACKTEST,
            "live_locked": True,
            "not_promotion_evidence": True,
            "counterfactual_trustworthy": bool(historical) and not lookahead,
            "pit_status": pit_status,
            "symbol": name,
            "as_of": session,
            "decision_id": str((historical or {}).get("decision_id") or decision_id or "") or UNAVAILABLE,
            "historical_timestamp": (historical or {}).get("decision_time") or session,
            "original": {
                "action": actual_action if historical else UNAVAILABLE,
                "reason_code": _honest((historical or {}).get("reason_code")),
                "reason": _honest((historical or {}).get("reason")),
                "tier": _honest((historical or {}).get("tier")),
                "entry": _honest(persisted_entry if historical else None),
                "entry_source": original_entry_source if historical else ENTRY_UNAVAILABLE,
                "stop": _honest(stop if historical else None),
                "target": _honest(target if historical else None),
                "source": "decision_journal" if historical and (historical or {}).get("decision_id") else (
                    "historical_replay_ledger" if historical else UNAVAILABLE
                ),
            },
            "simulated": {
                "action": simulated_action if historical and not lookahead else UNAVAILABLE,
                "entry": _honest(simulated_entry if historical and not lookahead else None),
                "entry_source": simulated_entry_source if historical and not lookahead else ENTRY_UNAVAILABLE,
                "role": "COUNTERFACTUAL_ALTERNATIVE",
                "defaulted": not bool(str(alternative or "").strip()) and bool(historical),
                "trustworthy": bool(historical) and not lookahead,
            },
            "evidence_at_t": {
                "label": "Information known at decision time",
                "max_bar_date": max_bar or UNAVAILABLE,
                "close": _honest(close_t),
                "future_bars_used_for_decision": lookahead,
                "pit_status": pit_status,
                "financials": financial if financial.get("available") else {
                    "available": False,
                    "status": UNAVAILABLE,
                    "note": financial.get("reason") or financial.get("note") or "No financials with publication <= T",
                },
                "research": research if research.get("available") else {
                    "available": False,
                    "status": UNAVAILABLE,
                    "note": research.get("note") or "No research snapshot dated <= T",
                },
                "sector": {
                    "status": sector.get("status") or UNAVAILABLE,
                    "usable_as_family_confirm": bool(sector.get("usable_as_family_confirm")),
                    "note": sector.get("limitation") or sector.get("note") or SECTOR_NOTE,
                },
                "news": news,
                "news_status": "AVAILABLE" if news else UNAVAILABLE,
                "reconstructed_engine_decision": reconstructed,
            },
            "subsequent_outcome": {
                "label": "What happened after T (not known at decision time)",
                "actual": actual_outcome,
                "simulated": simulated_outcome,
                "horizon_sessions": horizon,
            },
            "comparison": comparison,
            "warnings": warnings,
            "error": error,
        }
        fingerprint_src = {
            k: payload[k]
            for k in ("symbol", "as_of", "decision_id", "original", "simulated", "evidence_at_t", "subsequent_outcome", "comparison")
        }
        payload["fingerprint"] = hashlib.sha256(
            json.dumps(fingerprint_src, default=str, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        payload["generated_at"] = _now()
        return payload
    except Exception as exc:
        return {
            "schema_version": 1,
            "kind": "PAST_DECISION_SIMULATION",
            "status": FAILED,
            "available": False,
            "provenance": BACKTEST,
            "live_locked": True,
            "symbol": name or UNAVAILABLE,
            "as_of": session or UNAVAILABLE,
            "error": f"{type(exc).__name__}: {exc}"[:240],
            "warnings": warnings,
            "original": {"action": UNAVAILABLE},
            "simulated": {"action": UNAVAILABLE},
        }


SECTOR_NOTE = (
    "Static/current sector labels are not historically verified sector context."
)
