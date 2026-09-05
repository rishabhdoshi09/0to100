"""Autonomous evidence evolution for QuantTerm.

Contract
--------
1. Historical point-in-time replay is the *prior*: the same production decision
   stack must reproduce a positive setup edge on independent historical slices.
2. Only a reproduced historical setup becomes eligible for autonomous PAPER
   exploration. Historical evidence can never unlock live money.
3. Real forward paper outcomes are the *confirmation*: they can strengthen or
   decay the historical prior and eventually dominate the confidence score.
4. The engine produces bounded research hypotheses; it never writes arbitrary
   Python, invents a BUY, or silently promotes live execution.

The module is intentionally storage-backed and idempotent. Re-running the same
history generation replaces the same split evidence instead of multiplying it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = ROOT / "logs" / "product" / "autonomous_evolution.json"
DEFAULT_RUN_DIR = ROOT / "logs" / "product" / "autonomous_evolution"
SCHEMA_VERSION = 1

DEFAULT_SPLITS = 3
DEFAULT_SESSIONS_PER_SPLIT = 8
DEFAULT_UNIVERSE_LIMIT = 40
DEFAULT_OUTCOME_BUFFER = 12
DEFAULT_MIN_HIST_N = 8
DEFAULT_MIN_POSITIVE_SPLITS = 2
DEFAULT_MIN_MEAN_R = 0.15
DEFAULT_REFRESH_SESSIONS = 5

_lock = threading.Lock()
_thread: threading.Thread | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _i(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return int(default)


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def state_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_AUTONOMOUS_EVOLUTION")
    return Path(override) if override else DEFAULT_STATE


def run_dir(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_AUTONOMOUS_EVOLUTION_DIR")
    return Path(override) if override else DEFAULT_RUN_DIR


def _read(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["schema_version"] = SCHEMA_VERSION
    data["updated_at"] = _now()
    data["live_locked"] = True
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def _hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _setup(row: Mapping[str, Any]) -> str:
    return str(
        row.get("setup")
        or row.get("setup_label")
        or row.get("primary_thesis")
        or ""
    ).strip()


def _usable_buy(row: Mapping[str, Any]) -> tuple[str, float] | None:
    if str(row.get("decision") or "").upper() != "BUY":
        return None
    setup = _setup(row)
    if not setup:
        return None
    pit = dict(row.get("pit") or {})
    grade = str(row.get("pit_grade") or pit.get("grade") or "")
    comparable = pit.get("comparable_to_forward")
    if comparable is False:
        return None
    if grade and grade not in {"PIT_STRONG", "PIT_PARTIAL"}:
        return None
    if str(row.get("outcome_status") or "") not in {"MATURED", ""}:
        return None
    value = row.get("r_multiple")
    try:
        r_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(r_value):
        return None
    return setup, r_value


def _reason_quality(row: Mapping[str, Any]) -> tuple[str, float, str] | None:
    decision = str(row.get("decision") or "").upper()
    if decision not in {"WAIT", "AVOID", "REJECT"}:
        return None
    reason = str(row.get("reason_code") or "").strip()
    classification = str(row.get("classification") or "").strip()
    if not reason or not classification:
        return None
    score = {
        "CORRECT_REJECTION": 1.0,
        "AVOIDED_LOSER": 1.0,
        "GOOD_WAIT": 0.75,
        "FLAT": 0.0,
        "RAN_AWAY_WITHOUT_ENTRY": -0.25,
        "MISSED_WINNER": -1.0,
    }.get(classification)
    if score is None:
        return None
    return reason, float(score), classification


def summarize_report(report: Mapping[str, Any], *, split_id: str) -> dict[str, Any]:
    """Pure summary used by both production and tests."""
    setups: dict[str, list[float]] = {}
    reasons: dict[str, list[tuple[float, str]]] = {}
    for raw in list(report.get("decisions") or report.get("rows") or []):
        if not isinstance(raw, Mapping):
            continue
        buy = _usable_buy(raw)
        if buy is not None:
            setup, r_value = buy
            setups.setdefault(setup, []).append(r_value)
        reason = _reason_quality(raw)
        if reason is not None:
            code, quality, classification = reason
            reasons.setdefault(code, []).append((quality, classification))

    setup_rows: dict[str, Any] = {}
    for setup, values in setups.items():
        n = len(values)
        setup_rows[setup] = {
            "n": n,
            "mean_R": round(sum(values) / n, 6) if n else None,
            "positive": sum(1 for value in values if value > 0),
            "negative": sum(1 for value in values if value < 0),
        }

    reason_rows: dict[str, Any] = {}
    for reason, values in reasons.items():
        n = len(values)
        counts: dict[str, int] = {}
        for _, classification in values:
            counts[classification] = counts.get(classification, 0) + 1
        reason_rows[reason] = {
            "n": n,
            "mean_quality": round(sum(value for value, _ in values) / n, 6) if n else None,
            "classifications": counts,
        }

    return {
        "split_id": split_id,
        "run_id": report.get("run_id"),
        "period_start": report.get("period_start"),
        "period_end": report.get("period_end"),
        "status": report.get("status"),
        "setups": setup_rows,
        "reasons": reason_rows,
    }


def _aggregate_splits(split_summaries: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    setup_parts: dict[str, list[dict[str, Any]]] = {}
    reason_parts: dict[str, list[dict[str, Any]]] = {}
    for split in split_summaries:
        sid = str(split.get("split_id") or "")
        for setup, raw in dict(split.get("setups") or {}).items():
            row = dict(raw or {})
            row["split_id"] = sid
            setup_parts.setdefault(str(setup), []).append(row)
        for reason, raw in dict(split.get("reasons") or {}).items():
            row = dict(raw or {})
            row["split_id"] = sid
            reason_parts.setdefault(str(reason), []).append(row)

    min_n = _i("QT_EVOLUTION_MIN_HIST_N", DEFAULT_MIN_HIST_N)
    min_positive_splits = _i("QT_EVOLUTION_MIN_POSITIVE_SPLITS", DEFAULT_MIN_POSITIVE_SPLITS)
    min_mean_r = _f("QT_EVOLUTION_MIN_MEAN_R", DEFAULT_MIN_MEAN_R)

    setups: dict[str, Any] = {}
    for setup, parts in setup_parts.items():
        n = sum(int(p.get("n") or 0) for p in parts)
        weighted = sum(float(p.get("mean_R") or 0.0) * int(p.get("n") or 0) for p in parts)
        mean_r = weighted / n if n else 0.0
        positive_splits = sum(1 for p in parts if int(p.get("n") or 0) > 0 and float(p.get("mean_R") or 0.0) > 0)
        tested_splits = sum(1 for p in parts if int(p.get("n") or 0) > 0)
        worst_split = min((float(p.get("mean_R") or 0.0) for p in parts if int(p.get("n") or 0) > 0), default=0.0)
        reproduced = (
            n >= min_n
            and tested_splits >= min_positive_splits
            and positive_splits >= min_positive_splits
            and mean_r >= min_mean_r
        )
        sample_component = min(1.0, n / max(1.0, float(min_n * 3)))
        reproduction_component = min(1.0, positive_splits / max(1.0, float(min_positive_splits)))
        edge_component = max(0.0, min(1.0, mean_r / 0.60))
        stability_component = max(0.0, min(1.0, (worst_split + 0.20) / 0.60))
        score = 100.0 * (
            0.25 * sample_component
            + 0.35 * reproduction_component
            + 0.25 * edge_component
            + 0.15 * stability_component
        )
        # Historical evidence is a prior, never a 90+ certainty claim.
        score = min(79.0 if reproduced else 49.0, max(0.0, score))
        setups[setup] = {
            "n": n,
            "mean_R": round(mean_r, 6),
            "tested_splits": tested_splits,
            "positive_splits": positive_splits,
            "worst_split_mean_R": round(worst_split, 6),
            "reproduced": bool(reproduced),
            "historical_confidence_score": round(score, 1),
            "split_metrics": parts,
        }

    reasons: dict[str, Any] = {}
    for reason, parts in reason_parts.items():
        n = sum(int(p.get("n") or 0) for p in parts)
        weighted = sum(float(p.get("mean_quality") or 0.0) * int(p.get("n") or 0) for p in parts)
        mean_quality = weighted / n if n else 0.0
        counts: dict[str, int] = {}
        for part in parts:
            for klass, count in dict(part.get("classifications") or {}).items():
                counts[str(klass)] = counts.get(str(klass), 0) + int(count or 0)
        reasons[reason] = {
            "n": n,
            "mean_quality": round(mean_quality, 6),
            "classifications": counts,
            "split_metrics": parts,
        }
    return setups, reasons


def _publish_history_policies(setups: Mapping[str, Any], reasons: Mapping[str, Any]) -> list[dict[str, Any]]:
    from product.learning_policy_store import upsert_policy

    published: list[dict[str, Any]] = []
    for setup, raw in sorted(setups.items()):
        row = dict(raw or {})
        reproduced = bool(row.get("reproduced"))
        published.append(upsert_policy(
            policy_id=f"HIST_SETUP::{setup}",
            dimension="setup",
            bucket=str(setup),
            sample_size=int(row.get("n") or 0),
            expectancy_R=float(row.get("mean_R") or 0.0),
            source="backtest_reproduced",
            extra={
                "production_status": "ELIGIBLE" if reproduced else "EXPERIMENTAL",
                "confidence": "REPRODUCED_BACKTEST" if reproduced else "INSUFFICIENT_EVIDENCE",
                "affects_selection": bool(reproduced),
                "historical_reproduced_positive": bool(reproduced),
                "historical_confidence_score": row.get("historical_confidence_score"),
                "historical_only": True,
                "splits_tested": row.get("tested_splits"),
                "positive_splits": row.get("positive_splits"),
                "worst_split_mean_R": row.get("worst_split_mean_R"),
                "split_metrics": row.get("split_metrics") or [],
                "live_locked": True,
            },
        ))
    for reason, raw in sorted(reasons.items()):
        row = dict(raw or {})
        # Historical rejection diagnostics are research-only. Forward
        # counterfactual evidence remains the authority for changing filters.
        published.append(upsert_policy(
            policy_id=f"HIST_REJECT::{reason}",
            dimension="reason_code",
            bucket=str(reason),
            sample_size=int(row.get("n") or 0),
            expectancy_R=float(row.get("mean_quality") or 0.0),
            source="backtest_reproduced_counterfactual",
            extra={
                "production_status": "EXPERIMENTAL",
                "confidence": "RESEARCH_ONLY",
                "affects_selection": False,
                "historical_only": True,
                "classification_counts": row.get("classifications") or {},
                "split_metrics": row.get("split_metrics") or [],
                "not_pnl": True,
                "live_locked": True,
            },
        ))
    return published


def _hypotheses(setups: Mapping[str, Any], reasons: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for setup, row in setups.items():
        if row.get("reproduced"):
            out.append({
                "kind": "FORWARD_VALIDATE_SETUP",
                "setup": setup,
                "claim": (
                    f"{setup} reproduced positive historical expectancy across "
                    f"{row.get('positive_splits')}/{row.get('tested_splits')} independent slices; "
                    "validate with autonomous forward paper trades."
                ),
                "historical_n": row.get("n"),
                "historical_mean_R": row.get("mean_R"),
                "paper_only": True,
            })
    for reason, row in reasons.items():
        n = int(row.get("n") or 0)
        counts = dict(row.get("classifications") or {})
        missed = int(counts.get("MISSED_WINNER") or 0)
        if n >= 8 and missed / max(1, n) >= 0.35:
            out.append({
                "kind": "OVER_REJECTION_CANDIDATE",
                "reason_code": reason,
                "claim": f"{reason} missed {missed}/{n} historical opportunities; challenge the filter before changing it.",
                "paper_only": True,
                "auto_changes_hard_gate": False,
            })
    return out[:40]


def _split_plan(all_sessions: Sequence[str]) -> list[dict[str, Any]]:
    splits = _i("QT_EVOLUTION_SPLITS", DEFAULT_SPLITS)
    width = _i("QT_EVOLUTION_SESSIONS_PER_SPLIT", DEFAULT_SESSIONS_PER_SPLIT)
    buffer_n = _i("QT_EVOLUTION_OUTCOME_BUFFER", DEFAULT_OUTCOME_BUFFER)
    sessions = [str(x)[:10] for x in all_sessions if str(x)[:10]]
    if len(sessions) < splits * width + buffer_n + 2:
        return []
    eligible_end = len(sessions) - buffer_n
    eligible = sessions[:eligible_end]
    chosen = eligible[-splits * width:]
    plan: list[dict[str, Any]] = []
    for index in range(splits):
        window = chosen[index * width:(index + 1) * width]
        if not window:
            continue
        end_idx = sessions.index(window[-1])
        if end_idx + 1 >= len(sessions):
            continue
        sentinel = sessions[end_idx + 1]
        sid = f"split-{index + 1}-{window[0]}-{window[-1]}"
        plan.append({"split_id": sid, "sessions": window, "sentinel": sentinel})
    return plan


def _latest_anchor(all_sessions: Sequence[str]) -> str:
    buffer_n = _i("QT_EVOLUTION_OUTCOME_BUFFER", DEFAULT_OUTCOME_BUFFER)
    sessions = [str(x)[:10] for x in all_sessions if str(x)[:10]]
    if len(sessions) <= buffer_n:
        return ""
    return sessions[-buffer_n - 1]


def bootstrap_status(path: str | Path | None = None) -> dict[str, Any]:
    payload = _read(state_path(path))
    if not payload:
        return {
            "required": True,
            "status": "NOT_STARTED",
            "analysis_complete": False,
            "paper_ready_setups": 0,
            "live_locked": True,
        }
    payload.setdefault("required", True)
    payload.setdefault("analysis_complete", False)
    payload["live_locked"] = True
    return payload


def _needs_refresh(state: Mapping[str, Any], all_sessions: Sequence[str]) -> bool:
    if not state.get("analysis_complete"):
        return True
    current = _latest_anchor(all_sessions)
    previous = str(state.get("history_anchor") or "")
    if not current or not previous or current <= previous:
        return False
    sessions = [str(x)[:10] for x in all_sessions]
    try:
        delta = sessions.index(current) - sessions.index(previous)
    except ValueError:
        return True
    return delta >= _i("QT_EVOLUTION_REFRESH_SESSIONS", DEFAULT_REFRESH_SESSIONS)


def run_bootstrap(*, force: bool = False, path: str | Path | None = None) -> dict[str, Any]:
    """Produce + reproduce historical evidence, then publish PAPER-only priors."""
    from product.historical_replay import official_sessions, run_historical_replay

    target = state_path(path)
    sessions = official_sessions()
    state = bootstrap_status(target)
    if not force and not _needs_refresh(state, sessions):
        return state

    plan = _split_plan(sessions)
    if not plan:
        payload = {
            **state,
            "required": True,
            "status": "WAITING_FOR_HISTORY",
            "analysis_complete": False,
            "history_anchor": _latest_anchor(sessions),
            "reason": "Not enough mature official sessions for independent reproduction slices.",
            "live_locked": True,
        }
        _write(target, payload)
        return payload

    working = {
        **state,
        "required": True,
        "status": "RUNNING",
        "analysis_complete": False,
        "started_at": _now(),
        "history_anchor": _latest_anchor(sessions),
        "plan": plan,
        "live_locked": True,
    }
    _write(target, working)

    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    universe_limit = _i("QT_EVOLUTION_UNIVERSE_LIMIT", DEFAULT_UNIVERSE_LIMIT)
    base_dir = run_dir()
    base_dir.mkdir(parents=True, exist_ok=True)

    for split in plan:
        sid = str(split["split_id"])
        window = list(split["sessions"])
        sentinel = str(split["sentinel"])
        dates = list(window) + [sentinel]
        try:
            report = run_historical_replay(
                sessions=len(window),
                universe_limit=universe_limit,
                force=force,
                directory=base_dir / sid,
                dates_fn=lambda dates=tuple(dates): list(dates),
                persist_live_reco=False,
            )
            summary = summarize_report(report, split_id=sid)
            summaries.append(summary)
        except Exception as exc:
            errors.append({"split_id": sid, "error": str(exc)[:240]})

    setups, reasons = _aggregate_splits(summaries)
    published = _publish_history_policies(setups, reasons) if summaries else []
    hypotheses = _hypotheses(setups, reasons)
    ready = sorted(setup for setup, row in setups.items() if row.get("reproduced"))
    complete = len(summaries) == len(plan) and not errors
    payload = {
        "schema_version": SCHEMA_VERSION,
        "required": True,
        "status": "SUCCEEDED" if complete else ("DEGRADED" if summaries else "FAILED"),
        "analysis_complete": bool(complete),
        "paper_ready_setups": len(ready),
        "ready_setups": ready,
        "history_anchor": _latest_anchor(sessions),
        "splits": summaries,
        "setups": setups,
        "reason_diagnostics": reasons,
        "hypotheses": hypotheses,
        "published_policies": [p.get("policy_id") for p in published if isinstance(p, Mapping)],
        "errors": errors,
        "finished_at": _now(),
        "live_locked": True,
        "note": (
            "Historical evidence is a PAPER prior only. Autonomous forward paper outcomes "
            "must confirm or decay it. Live money cannot be enabled here."
        ),
    }
    _write(target, payload)
    return payload


def ensure_started_async() -> dict[str, Any]:
    """Start one bounded historical bootstrap thread if evidence is missing/stale."""
    global _thread
    try:
        from product.historical_replay import official_sessions

        sessions = official_sessions()
    except Exception:
        sessions = []
    state = bootstrap_status()
    if sessions and not _needs_refresh(state, sessions):
        return state
    with _lock:
        if _thread is not None and _thread.is_alive():
            return {**state, "status": "RUNNING", "analysis_complete": False}

        def _runner() -> None:
            try:
                run_bootstrap(force=False)
            except Exception as exc:
                failed = bootstrap_status()
                failed.update({
                    "status": "FAILED",
                    "analysis_complete": False,
                    "error": str(exc)[:240],
                    "finished_at": _now(),
                })
                _write(state_path(), failed)

        _thread = threading.Thread(target=_runner, name="autonomous-evidence-evolution", daemon=True)
        _thread.start()
    return {**state, "status": "RUNNING", "analysis_complete": False}


def confidence_from_policies(
    candidate: Mapping[str, Any],
    policies: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Historical prior first; forward paper evidence then strengthens/decays it.

    ``evidence_confidence_score`` is an explicit composite evidence score, not a
    claimed win probability.
    """
    setup = _setup(candidate)
    hist = next(
        (dict(p) for p in policies if str(p.get("policy_id") or "") == f"HIST_SETUP::{setup}"),
        {},
    )
    forward = next(
        (dict(p) for p in policies if str(p.get("policy_id") or "") == f"SETUP::{setup}"),
        {},
    )
    hist_ready = bool(hist.get("historical_reproduced_positive"))
    hist_score = float(hist.get("historical_confidence_score") or 0.0)
    forward_n = int(forward.get("sample_size") or 0)
    forward_edge = float(forward.get("expectancy_difference_R") or 0.0)
    forward_sample = min(1.0, forward_n / 30.0)
    forward_edge_component = math.tanh(forward_edge / 0.50) if forward_n else 0.0
    forward_score = max(0.0, min(100.0, 50.0 + 30.0 * forward_edge_component + 20.0 * forward_sample))

    if not hist_ready:
        combined = min(49.0, hist_score)
        stage = "HISTORICAL_UNPROVEN"
    elif forward_n <= 0:
        combined = min(79.0, hist_score)
        stage = "HISTORICAL_BASE"
    else:
        forward_weight = min(0.70, 0.20 + 0.50 * forward_sample)
        combined = hist_score * (1.0 - forward_weight) + forward_score * forward_weight
        if forward_n < 8:
            stage = "FORWARD_EARLY"
        elif forward_edge <= -0.20:
            stage = "FORWARD_DECAYED"
        elif forward_n < 20:
            stage = "FORWARD_CALIBRATING"
        else:
            stage = "FORWARD_CONFIRMED" if forward_edge > 0 else "FORWARD_WEAK"
    combined = round(max(0.0, min(95.0, combined)), 1)

    paper_eligible = bool(hist_ready)
    if forward_n >= 5 and forward_edge <= -0.25:
        paper_eligible = False
    return {
        "setup": setup,
        "historical_ready": hist_ready,
        "historical_n": int(hist.get("sample_size") or 0),
        "historical_mean_R": hist.get("expectancy_R"),
        "historical_splits": int(hist.get("splits_tested") or 0),
        "historical_positive_splits": int(hist.get("positive_splits") or 0),
        "historical_confidence_score": round(hist_score, 1),
        "forward_n": forward_n,
        "forward_mean_R": forward.get("expectancy_R"),
        "forward_source": forward.get("evidence_source") or "",
        "forward_confidence_score": round(forward_score, 1) if forward_n else 0.0,
        "evidence_confidence_score": combined,
        "confidence_stage": stage,
        "paper_eligible": paper_eligible,
        "is_win_probability": False,
        "live_locked": True,
    }


if __name__ == "__main__":
    print(json.dumps(run_bootstrap(force=False), indent=2, default=str))
