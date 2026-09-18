"""Operator-visible, durable autonomous learning control.

This is not a licence to rewrite trading code. The loop may measure outcomes,
update calibrated statistics, test challenger policies, and run point-in-time
historical replay. It must never open live orders, never mix replay evidence
with forward paper evidence, and never invent dashboard numbers.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from core.runtime_paths import logs_dir

SCHEMA_VERSION = 1
DEFAULT_PATH = logs_dir() / "product" / "autonomous_learning.json"

MODE_AUTO = "AUTO"
MODE_FORWARD_PAPER = "FORWARD_PAPER"
MODE_HISTORICAL_REPLAY = "HISTORICAL_REPLAY"
MODE_PAUSED = "PAUSED"
MODES = (MODE_AUTO, MODE_FORWARD_PAPER, MODE_HISTORICAL_REPLAY, MODE_PAUSED)

ACTIVITY_IDLE = "idle"
ACTIVITY_SCANNING = "scanning"
ACTIVITY_SIMULATING = "simulating"
ACTIVITY_SETTLING = "settling"
ACTIVITY_LEARNING = "learning"
ACTIVITY_RESEARCHING = "researching"
ACTIVITY_VALIDATING = "validating_challenger"

EVIDENCE_FORWARD = "REAL_FORWARD_PAPER"
EVIDENCE_REPLAY = "HISTORICAL_REPLAY"
EVIDENCE_BACKTEST = "BACKTEST"


def store_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_AUTONOMOUS_LEARNING")
    return Path(override) if override else DEFAULT_PATH


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def _empty_control() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "enabled": True,
        "mode": MODE_AUTO,
        "updated_at": "",
        "last_cycle_at": "",
        "last_replay_at": "",
        "last_forward_cycle_at": "",
        "last_error": "",
        "note": (
            "AUTO uses REAL_FORWARD_PAPER while the cash session is open and "
            "HISTORICAL_REPLAY when it is closed. Replay never counts as forward evidence."
        ),
    }


def load_control(path: str | Path | None = None) -> dict[str, Any]:
    payload = {**_empty_control(), **_read_json(store_path(path))}
    mode = str(payload.get("mode") or MODE_AUTO).upper()
    payload["mode"] = mode if mode in MODES else MODE_AUTO
    payload["enabled"] = payload.get("enabled") is not False
    payload["schema_version"] = SCHEMA_VERSION
    return payload


def save_control(payload: Mapping[str, Any], path: str | Path | None = None) -> dict[str, Any]:
    current = load_control(path)
    mode = str(payload.get("mode") or current.get("mode") or MODE_AUTO).upper()
    if mode not in MODES:
        raise ValueError(f"Unsupported autonomous learning mode: {mode}")
    updated = {
        **current,
        **dict(payload),
        "mode": mode,
        "enabled": payload.get("enabled", current.get("enabled")) is not False,
        "updated_at": _now(),
        "schema_version": SCHEMA_VERSION,
        "live_locked": True,
    }
    _atomic_write(store_path(path), updated)
    return updated


def _market_closed(now: datetime | None = None) -> bool:
    try:
        from zoneinfo import ZoneInfo
        from research.autonomy import schedules as SCH

        clock = now or datetime.now(timezone.utc)
        if clock.tzinfo is None:
            clock = clock.replace(tzinfo=timezone.utc)
        ist = clock.astimezone(ZoneInfo("Asia/Kolkata"))
        holidays: set = set()
        try:
            from research.intelligence.data.nse_calendar import load_holidays
            holidays = load_holidays() or set()
        except Exception:
            holidays = set()
        return not SCH.market_is_open(ist, holidays)
    except Exception:
        return True


def intended_evidence_lane(control: Mapping[str, Any] | None = None, *, now: datetime | None = None) -> str:
    state = dict(control or load_control())
    if not state.get("enabled") or state.get("mode") == MODE_PAUSED:
        return ""
    mode = str(state.get("mode") or MODE_AUTO)
    if mode == MODE_FORWARD_PAPER:
        return EVIDENCE_FORWARD
    if mode == MODE_HISTORICAL_REPLAY:
        return EVIDENCE_REPLAY
    return EVIDENCE_FORWARD if not _market_closed(now) else EVIDENCE_REPLAY


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                n += 1
    except Exception:
        return 0
    return n


def _evidence_counts() -> dict[str, Any]:
    forward_n = 0
    replay_n = 0
    classes: dict[str, int] = {}
    try:
        from product.forward_evidence import REAL_FORWARD_MARKET, load_ledger
        from product.evidence_class import HISTORICAL_REPLAY, PAPER_FORWARD

        for row in load_ledger():
            provenance = str(row.get("provenance") or row.get("evidence_class") or "")
            cls = str(row.get("classification") or "")
            if provenance in {REAL_FORWARD_MARKET, PAPER_FORWARD, "PAPER_FORWARD", "REAL_FORWARD", EVIDENCE_FORWARD}:
                forward_n += 1
            elif provenance in {HISTORICAL_REPLAY, EVIDENCE_REPLAY}:
                replay_n += 1
            if cls:
                classes[cls] = classes.get(cls, 0) + 1
    except Exception:
        pass
    try:
        from product.historical_replay import ledger_path as replay_ledger
        replay_n = max(replay_n, _count_jsonl(replay_ledger()))
    except Exception:
        pass
    try:
        from product.counterfactual_learning import ledger_path as cf_path
        from product.counterfactual_learning import (
            AVOIDED_LOSER, CORRECT_REJECTION, GOOD_WAIT, MISSED_WINNER,
        )
        for line in (cf_path().read_text(encoding="utf-8").splitlines() if cf_path().exists() else []):
            try:
                item = json.loads(line)
            except Exception:
                continue
            if not isinstance(item, dict):
                continue
            cls = str(item.get("classification") or "")
            if cls:
                classes[cls] = classes.get(cls, 0)
                classes[cls] += 1
            lane = str(item.get("evidence_class") or item.get("provenance") or "")
            if lane in {EVIDENCE_REPLAY, "BACKTEST", "HISTORICAL_REPLAY"}:
                replay_n += 0  # already counted via replay ledger when present
        classes.setdefault(CORRECT_REJECTION, classes.get(CORRECT_REJECTION, 0))
        classes.setdefault(MISSED_WINNER, classes.get(MISSED_WINNER, 0))
        classes.setdefault(AVOIDED_LOSER, classes.get(AVOIDED_LOSER, 0))
        classes.setdefault(GOOD_WAIT, classes.get(GOOD_WAIT, 0))
    except Exception:
        pass
    return {
        "forward_evidence_count": forward_n,
        "replay_evidence_count": replay_n,
        "correct_rejects": classes.get("CORRECT_REJECTION", 0),
        "avoided_losers": classes.get("AVOIDED_LOSER", 0),
        "missed_winners": classes.get("MISSED_WINNER", 0),
        "good_waits": classes.get("GOOD_WAIT", 0),
        "false_positives": classes.get("FALSE_POSITIVE", 0),
        "false_negatives": classes.get("FALSE_NEGATIVE", 0),
        "classifications": classes,
    }


def _paper_counts() -> dict[str, Any]:
    opened = 0
    settled = 0
    try:
        from product.paper_status import read_paper_status
        paper = read_paper_status()
        opened = len(list(paper.open_positions or []))
        settled = len(list(paper.closed_trades or []))
    except Exception:
        pass
    return {
        "paper_trades_opened": opened,
        "paper_trades_settled": settled,
    }


def _policy_projection() -> dict[str, Any]:
    champion: dict[str, Any] = {}
    challengers: list[dict[str, Any]] = []
    promotion_log: list[dict[str, Any]] = []
    try:
        from product.champion_challenger import load_store
        store = load_store()
        champion = dict(store.get("champion") or {})
        challengers = [dict(c) for c in (store.get("challengers") or []) if isinstance(c, Mapping)]
        promotion_log = [dict(x) for x in (store.get("promotion_log") or []) if isinstance(x, Mapping)]
    except Exception as exc:
        return {
            "champion": {},
            "challenger": {},
            "challengers_under_evaluation": 0,
            "active_policies": 0,
            "rejected_policies": 0,
            "promotion_eligible": False,
            "promotion_blocked_reason": f"Champion/challenger store unavailable: {exc}"[:240],
            "promotion_log": [],
        }
    evaluating = [
        c for c in challengers
        if str(c.get("status") or "").upper() in {"SHADOW", "TESTING", "ELIGIBLE", "PROPOSED"}
    ]
    rejected = [c for c in challengers if str(c.get("status") or "").upper() in {"REJECTED", "RETIRED"}]
    eligible = [c for c in challengers if str(c.get("status") or "").upper() == "ELIGIBLE"]
    active = [c for c in challengers if str(c.get("status") or "").upper() == "PROMOTED"]
    blocked = "No challenger has met the promotion contract (OOS + forward evidence, explicit promote)."
    if eligible:
        blocked = str(eligible[0].get("promotion_block_reason") or blocked)
    return {
        "champion": champion,
        "challenger": evaluating[0] if evaluating else {},
        "challengers_under_evaluation": len(evaluating),
        "active_policies": len(active) + (1 if champion else 0),
        "rejected_policies": len(rejected),
        "promotion_eligible": bool(eligible),
        "promotion_blocked_reason": "" if eligible else blocked,
        "promotion_log": promotion_log[-12:],
    }


def _activity(auto: Mapping[str, Any], ops: Mapping[str, Any], replay: Mapping[str, Any]) -> str:
    if str(replay.get("status") or "").upper() == "RUNNING":
        return ACTIVITY_SIMULATING
    job = str((_as(auto.get("active_job")).get("job_type") or "")).lower()
    kinds = {str(o.get("kind") or "").upper() for o in list(ops.get("active") or []) if isinstance(o, Mapping)}
    if job in {"market_scan"} or "MARKET_SCAN" in kinds:
        return ACTIVITY_SCANNING
    if job in {"outcome_resolution"} or "OUTCOME_RESOLUTION" in kinds:
        return ACTIVITY_SETTLING
    if job in {"learning_cycle"}:
        return ACTIVITY_LEARNING
    if job in {"research_cycle"}:
        return ACTIVITY_RESEARCHING
    if job in {"paper_cycle"}:
        return ACTIVITY_SETTLING
    state = str(auto.get("state") or "").upper()
    if state == "RESEARCHING":
        return ACTIVITY_RESEARCHING
    if state == "PAPER_ACTIVE":
        return ACTIVITY_SETTLING
    return ACTIVITY_IDLE


def _as(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _next_action(control: Mapping[str, Any], *, market_closed: bool, activity: str) -> str:
    if not control.get("enabled") or control.get("mode") == MODE_PAUSED:
        return "Autonomous learning is paused. No cycle will start."
    if activity != ACTIVITY_IDLE:
        return f"Current activity is {activity}."
    lane = intended_evidence_lane(control)
    if lane == EVIDENCE_FORWARD:
        return "Next: REAL_FORWARD_PAPER cycle after the shared market scan (no live orders)."
    if lane == EVIDENCE_REPLAY:
        if not market_closed:
            return "Historical replay is deferred until the cash session closes."
        return "Next: historical virtual-paper batch from the next unprocessed PIT sessions."
    return "No next learning action while the control is off."


def dashboard(*, path: str | Path | None = None) -> dict[str, Any]:
    """Persisted operator view. Missing evidence stays missing."""
    from product.live_safety import live_safety_projection

    control = load_control(path)
    safety = live_safety_projection()
    auto: dict[str, Any] = {}
    ops: dict[str, Any] = {}
    replay: dict[str, Any] = {}
    historical_paper: dict[str, Any] = {}
    try:
        from product.autonomy_status import read_autonomy_status
        auto = read_autonomy_status()
    except Exception:
        auto = {}
    try:
        from operations.store import OperationStore
        from terminal_api import OPS_DB
        store = OperationStore(OPS_DB)
        ops = {"active": store.active(), "recent": store.recent(limit=8)}
    except Exception:
        ops = {"active": [], "recent": []}
    try:
        from product.historical_replay import load_latest, progress_path
        replay = dict(load_latest() or {})
        progress = _read_json(progress_path())
        if str(progress.get("status") or "").upper() == "RUNNING":
            replay = {**replay, **progress}
    except Exception:
        replay = {}
    try:
        from product.historical_paper_loop import DEFAULT_LEDGER, load_state

        historical_paper = load_state()
        ledger = Path(DEFAULT_LEDGER)
        historical_paper["virtual_trades"] = _count_jsonl(ledger)
    except Exception:
        historical_paper = {}
    market_closed = _market_closed()
    activity = ACTIVITY_IDLE if (not control.get("enabled") or control.get("mode") == MODE_PAUSED) else _activity(auto, ops, replay)
    historical_phase = str(historical_paper.get("phase") or "").upper()
    if control.get("enabled") and control.get("mode") != MODE_PAUSED:
        if historical_phase == "RUNNING":
            activity = ACTIVITY_SIMULATING
        elif historical_phase == "AWAITING_LEARNING":
            activity = ACTIVITY_LEARNING
        elif historical_phase == "AWAITING_RESEARCH":
            activity = ACTIVITY_RESEARCHING
    evidence = _evidence_counts()
    paper = _paper_counts()
    policies = _policy_projection()
    sim_n = 0
    try:
        sim_n = int(replay.get("decision_candidates") or replay.get("decisions_tested") or 0)
        if not sim_n:
            sim_n = len(list(replay.get("decisions") or replay.get("rows") or []))
    except Exception:
        sim_n = 0
    last_cycle = control.get("last_cycle_at") or replay.get("finished_at") or auto.get("heartbeat_ist") or ""
    missing = []
    if not evidence["forward_evidence_count"] and not evidence["replay_evidence_count"]:
        missing.append("No persisted learning evidence yet.")
    if not policies.get("champion"):
        missing.append("Champion identity has not been loaded.")
    return {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "enabled": bool(control.get("enabled")),
        "mode": control.get("mode"),
        "activity": activity,
        "evidence_lane": intended_evidence_lane(control) or "NONE",
        "market_closed": market_closed,
        "current_activity": activity,
        "counts": {
            "historical_decisions_simulated": sim_n,
            "historical_virtual_paper_trades": int(historical_paper.get("virtual_trades") or 0),
            "forward_paper_decisions": evidence["forward_evidence_count"],
            **paper,
            "correct_rejects": evidence["correct_rejects"],
            "avoided_losers": evidence["avoided_losers"],
            "missed_winners": evidence["missed_winners"],
            "good_waits": evidence["good_waits"],
            "false_positives": evidence["false_positives"],
            "false_negatives": evidence["false_negatives"],
            "challenger_policies_under_evaluation": policies["challengers_under_evaluation"],
            "active_policies": policies["active_policies"],
            "rejected_policies": policies["rejected_policies"],
            "forward_evidence_count": evidence["forward_evidence_count"],
            "replay_evidence_count": evidence["replay_evidence_count"],
        },
        "champion": policies["champion"] or {"status": "UNAVAILABLE", "detail": "No champion record persisted."},
        "challenger": policies["challenger"] or {"status": "NONE", "detail": "No challenger is under evaluation."},
        "challenger_evidence": policies["challenger"].get("metrics") if policies.get("challenger") else {},
        "promotion_eligible": policies["promotion_eligible"],
        "promotion_blocked_reason": policies["promotion_blocked_reason"],
        "last_learning_cycle": last_cycle or "No learning cycle has been recorded.",
        "next_learning_action": _next_action(control, market_closed=market_closed, activity=activity),
        "historical_virtual_paper": {
            "phase": historical_paper.get("phase") or "IDLE",
            "batch_id": historical_paper.get("current_batch_id") or "",
            "current_sessions": list(historical_paper.get("current_sessions") or []),
            "last_completed_session": historical_paper.get("last_completed_session") or "",
            "virtual_trades": int(historical_paper.get("virtual_trades") or 0),
            "last_result": dict(historical_paper.get("last_result") or {}),
            "last_error": historical_paper.get("last_error") or "",
            "provenance": EVIDENCE_REPLAY,
            "not_real_pnl": True,
        },
        "latest_persisted_evidence": {
            "replay_status": replay.get("status") or "NONE",
            "replay_period": (
                f"{replay.get('period_start') or '—'} → {replay.get('period_end') or '—'}"
                if replay else "No historical replay report on disk."
            ),
            "replay_note": replay.get("note") or "",
            "provenance": replay.get("provenance") or EVIDENCE_REPLAY,
        },
        "missing": missing,
        "updated_at": control.get("updated_at") or "",
        "note": control.get("note"),
        "live_locked": True,
        **safety,
    }


def set_control(*, enabled: bool | None = None, mode: str | None = None, path: str | Path | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if enabled is not None:
        payload["enabled"] = bool(enabled)
    if mode is not None:
        payload["mode"] = str(mode).upper()
    save_control(payload, path)
    return dashboard(path=path)


def maybe_run_closed_market_replay(*, now: datetime | None = None, force: bool = False) -> dict[str, Any]:
    """Manual report-only replay. Never opens the real/forward paper book."""
    control = load_control()
    if not control.get("enabled") or control.get("mode") == MODE_PAUSED:
        return {"skipped": True, "reason": "autonomous_learning_paused"}
    lane = intended_evidence_lane(control, now=now)
    if lane != EVIDENCE_REPLAY and not force:
        return {"skipped": True, "reason": "forward_paper_lane_active", "lane": lane}
    if not force and not _market_closed(now):
        return {
            "skipped": True,
            "reason": "market_open_replay_deferred",
            "lane": EVIDENCE_REPLAY,
            "next_action": "WAIT_FOR_MARKET_CLOSE",
            "opens_paper_trades": False,
            "not_promotion_evidence": True,
        }
    try:
        from product.historical_replay import start_replay_async, load_latest
        latest = load_latest()
        if str(latest.get("status") or "").upper() == "RUNNING" and not force:
            return {"skipped": True, "reason": "replay_already_running", "status": latest.get("status")}
        started = start_replay_async(
            force=force,
            sessions=8,
            universe_limit=40,
            persist_live_reco=False,
        )
        if not started.get("skipped"):
            stamp = _now()
            save_control({"last_replay_at": stamp, "last_cycle_at": stamp})
        started["evidence_class"] = EVIDENCE_REPLAY
        started["not_promotion_evidence"] = True
        started["opens_paper_trades"] = False
        if started.get("skipped"):
            started.setdefault("next_action", "WAIT_FOR_NEW_EVIDENCE")
        return started
    except Exception as exc:
        save_control({"last_error": str(exc)[:240]})
        return {"skipped": False, "error": str(exc)[:240], "evidence_class": EVIDENCE_REPLAY}
