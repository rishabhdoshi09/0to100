"""Connect scan → candidates → research → recommendation → paper → outcome → learning.

Uses existing engines. Does not start a fourth scheduler. Official bhavcopy is
enough for post-market work. Kite is required only for live paper entry.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from product import candidate_lifecycle as CL
from product import readiness as RDY

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "logs" / "product" / "autonomous_loop.json"
EVENTS_PATH = ROOT / "logs" / "product" / "autonomy_events.jsonl"
LEARNING_PATH = ROOT / "logs" / "product" / "learning_observations.jsonl"
MEMORY_PATH = ROOT / "logs" / "product" / "learning_memory.json"
METRICS_PATH = ROOT / "logs" / "product" / "operator_metrics.json"
FAILURES_PATH = ROOT / "logs" / "product" / "loop_failures.jsonl"
DEEP_RESEARCH_CAP = 6
SERIOUS_CANDIDATE_CAP = 15


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), default=str, indent=2), encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(row), default=str) + "\n")


def emit(kind: str, text: str, **extra: Any) -> None:
    _append_jsonl(EVENTS_PATH, {"at": _now(), "kind": kind, "text": text, **extra})


def session_date() -> str:
    try:
        from research.autonomy import schedules as SCH
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        return SCH.last_completed_session_date(now) or now.date().isoformat()
    except Exception:
        return datetime.now(timezone.utc).date().isoformat()


def _load_scan() -> dict[str, Any]:
    try:
        from product.scan_store import default_scan_path, load_scan

        return dict(load_scan(default_scan_path()) or {})
    except Exception:
        return {}


def _load_reco() -> dict[str, Any]:
    try:
        from product.recommendations_store import load_recommendations

        return dict(load_recommendations() or {})
    except Exception:
        return {}


def _ensure_reco(scan: Mapping[str, Any]) -> dict[str, Any]:
    reco = _load_reco()
    scan_at = str(scan.get("scanned_at") or "")
    if reco.get("from_saved_market_scan") and reco.get("scan_scanned_at") == scan_at and reco.get("categories") is not None:
        return reco
    if not scan.get("records"):
        return reco
    try:
        from product.recommendations_workspace import build_recommendations_workspace

        return dict(build_recommendations_workspace() or reco)
    except Exception as exc:
        reco = dict(reco)
        reco["build_error"] = str(exc)[:240]
        return reco


def _cards(reco: Mapping[str, Any]) -> list[dict[str, Any]]:
    try:
        from product.autopilot_journal import flatten_cards

        return [dict(c) for c in flatten_cards(reco) if isinstance(c, dict)]
    except Exception:
        out: list[dict[str, Any]] = []
        for cat in reco.get("categories") or []:
            if isinstance(cat, dict):
                out.extend(dict(c) for c in (cat.get("cards") or []) if isinstance(c, dict))
        return out


def _census(scan: Mapping[str, Any], reco: Mapping[str, Any], paper: Mapping[str, Any], session: str = "") -> dict[str, Any]:
    coverage = dict(scan.get("coverage") or {})
    reasons = dict(coverage.get("reason_counts") or {})
    ensemble = dict(reco.get("ensemble") or {})
    states = CL.state_counts(str(session or session_date())[:10])
    return {
        "universe": int(coverage.get("requested") or scan.get("requested_universe") or 0),
        "eligible": int(coverage.get("checked") or scan.get("scanned") or 0),
        "low_liquidity": int(reasons.get("LOW_LIQUIDITY") or 0),
        "no_setup": int(reasons.get("NO_SETUP") or 0),
        "technical_rejection": int(reasons.get("FALLING_KNIFE") or 0) + int(reasons.get("PRICE_BELOW_20") or 0),
        "extended": int((scan.get("summary") or {}).get("extended") or 0),
        "insufficient_evidence": int(reasons.get("INSUFFICIENT_HISTORY") or 0) + int(reasons.get("NO_OHLCV") or 0),
        "qualified": int(coverage.get("qualified") or (scan.get("summary") or {}).get("qualified") or 0),
        "high_conviction": int(ensemble.get("high_conviction_count") or 0),
        "good_setup": int(ensemble.get("good_setup_count") or 0),
        "watch": int(ensemble.get("watch_count") or 0),
        "paper_taken": len(paper.get("taken") or []),
        "paper_waits": len(paper.get("waits") or []),
        "paper_rejections": len(paper.get("rejections") or []),
        "researching": int(states.get(CL.RESEARCHING) or 0) + int(states.get(CL.WAIT_EVIDENCE) or 0),
        "ready": int(states.get(CL.READY) or 0),
        "entered": int(states.get(CL.ENTERED) or 0),
        "rejected": int(states.get(CL.REJECTED) or 0),
        "candidate_states": states,
        "reason_counts": reasons,
        "funnel": {
            "universe": int(coverage.get("requested") or scan.get("requested_universe") or 0),
            "eligible": int(coverage.get("checked") or scan.get("scanned") or 0),
            "basic_qualified": int(coverage.get("qualified") or (scan.get("summary") or {}).get("qualified") or 0),
            "recommendation_worthy": int(ensemble.get("high_conviction_count") or 0) + int(ensemble.get("good_setup_count") or 0),
            "serious_candidates": min(SERIOUS_CANDIDATE_CAP, int(ensemble.get("high_conviction_count") or 0) + int(ensemble.get("good_setup_count") or 0)),
            "deep_researched": int((paper.get("research_n") or 0) or 0),
            "ready": int(states.get(CL.READY) or 0),
            "wait": int(states.get(CL.WAIT) or 0),
        },
    }


def _ingest_candidates(scan: Mapping[str, Any], cards: list[dict[str, Any]], session: str) -> list[dict[str, Any]]:
    scan_run_id = str(scan.get("scanned_at") or scan.get("source_snapshot_id") or session)
    rows = []
    reco_symbols = {str(c.get("symbol") or "").upper() for c in cards if c.get("symbol")}
    for rec in scan.get("records") or []:
        if not isinstance(rec, dict):
            continue
        symbol = str(rec.get("symbol") or "").upper()
        if not symbol:
            continue
        status = str(rec.get("status") or "")
        ready = status.lower().startswith("ready")
        if not ready and symbol not in reco_symbols:
            continue
        verdict = str(rec.get("verdict") or status or "").upper()
        if verdict in {"WATCH", "WAIT"} or "wait" in status.lower() or "pullback" in status.lower():
            state = CL.WATCH
        elif ready:
            state = CL.SCREENED
        else:
            state = CL.SCREENED
        rows.append(CL.upsert(
            symbol=symbol, session_date=session, state=state,
            reason=str((rec.get("reasons") or ["scan"])[0])[:240],
            scan_run_id=scan_run_id, trigger="scan",
            payload={"verdict": rec.get("verdict"), "setup": rec.get("signals"), "status": status},
        ))
    for card in cards:
        symbol = str(card.get("symbol") or "").upper()
        if not symbol:
            continue
        tier = str(card.get("reco_tier") or "")
        rid = CL.recommendation_id(scan_run_id, symbol, tier)
        state = CL.QUALIFIED if tier in {"high_conviction", "good_setup"} else CL.WATCH
        rows.append(CL.upsert(
            symbol=symbol, session_date=session, state=state,
            reason=str(card.get("primary_thesis") or tier or "recommendation"),
            scan_run_id=scan_run_id, recommendation_id_value=rid, trigger="recommendation",
            payload={"tier": tier, "entry_state": card.get("entry_state")},
        ))
    return rows


def _research_shortlist(cards: list[dict[str, Any]]) -> list[str]:
    ranked = [
        c for c in cards
        if str(c.get("reco_tier") or "") in {"high_conviction", "good_setup"}
    ]
    ranked.sort(key=lambda c: (str(c.get("reco_tier")) != "high_conviction", str(c.get("symbol") or "")))
    seen: list[str] = []
    for card in ranked:
        symbol = str(card.get("symbol") or "").upper()
        if symbol and symbol not in seen:
            seen.append(symbol)
        if len(seen) >= DEEP_RESEARCH_CAP:
            break
    return seen


def _facts_present(symbol: str) -> dict[str, Any]:
    try:
        from product.due_diligence.acquire import load_autonomy_facts

        return dict(load_autonomy_facts(symbol) or {})
    except Exception:
        return {}


def _acquire(symbols: list[str], session: str, scan_run_id: str, *, download: bool) -> dict[str, Any]:
    acquired, cached, errors, waiting = [], [], [], []
    for symbol in symbols:
        CL.upsert(
            symbol=symbol, session_date=session, state=CL.RESEARCHING,
            reason="inspect evidence coverage", scan_run_id=scan_run_id, trigger="research",
        )
        facts = _facts_present(symbol)
        if facts:
            acquired.append({"symbol": symbol, "ok": True, "cached": True})
            cached.append(symbol)
            CL.upsert(
                symbol=symbol, session_date=session, state=CL.QUALIFIED,
                reason="evidence already on disk", scan_run_id=scan_run_id, trigger="research",
                payload={"acquire": "cached", "still_missing": facts.get("still_missing") or []},
            )
            continue
        if not download:
            waiting.append(symbol)
            CL.upsert(
                symbol=symbol, session_date=session, state=CL.WAIT_EVIDENCE,
                reason="queued for automatic acquire", scan_run_id=scan_run_id, trigger="research",
            )
            continue
        try:
            from product.due_diligence.acquire import acquire_symbol

            result = acquire_symbol(symbol, force=False)
            acquired.append({"symbol": symbol, "ok": True, "cached": False, "facts": (result or {}).get("symbol")})
            CL.upsert(
                symbol=symbol, session_date=session, state=CL.QUALIFIED,
                reason="evidence acquired", scan_run_id=scan_run_id, trigger="research",
                payload={"acquire": "ok"},
            )
        except Exception as exc:
            errors.append({"symbol": symbol, "error": str(exc)[:240]})
            _append_jsonl(FAILURES_PATH, {
                "at": _now(), "kind": "research_acquire", "symbol": symbol,
                "error": str(exc)[:240], "retry": True, "final": False,
            })
            CL.upsert(
                symbol=symbol, session_date=session, state=CL.WAIT_EVIDENCE,
                reason=str(exc)[:240], scan_run_id=scan_run_id, trigger="research",
            )
    return {
        "symbols": symbols, "acquired": acquired, "cached": cached, "waiting": waiting,
        "errors": errors, "n_ok": len(acquired), "n_failed": len(errors),
        "n_waiting": len(waiting), "downloaded": bool(download),
    }


def _consume_paper(cards: list[dict[str, Any]], reco: Mapping[str, Any], session: str, scan_run_id: str) -> dict[str, Any]:
    readiness = RDY.inspect_readiness()
    broker_ok = bool(readiness["capabilities"].get(RDY.BROKER_LIVE_DATA_READY))
    try:
        from research.autonomy import schedules as SCH
        from zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo("Asia/Kolkata"))
        window = SCH.entries_allowed_by_clock(now)
    except Exception:
        window = False
    from product.paper_autopilot import (
        BROKER_LOGIN_REQUIRED,
        ENTER_NOW,
        OUTSIDE_ENTRY_WINDOW,
        WAIT,
        evaluate_candidate,
    )

    taken, waits, rejections, intents = [], [], [], []
    for card in cards:
        symbol = str(card.get("symbol") or "").upper()
        if not symbol:
            continue
        decision = evaluate_candidate(card, book=None, entries_allowed=True, workspace=reco)
        execution_block = ""
        if decision.decision == ENTER_NOW:
            if not broker_ok:
                execution_block = BROKER_LOGIN_REQUIRED
            elif not window:
                execution_block = OUTSIDE_ENTRY_WINDOW
        row = decision.as_dict()
        row["execution_block"] = execution_block
        rid = CL.recommendation_id(scan_run_id, symbol, str(card.get("reco_tier") or ""))
        did = f"{session}|{symbol}|{decision.decision}|{decision.reason_code}|{scan_run_id}"
        intent_id = f"{did}:intent"
        if decision.decision == ENTER_NOW and not execution_block:
            state = CL.READY
            taken.append(row)
        elif decision.decision == ENTER_NOW and execution_block:
            state = CL.READY
            row["decision"] = "BLOCKED"
            row["reason_code"] = execution_block
            intents.append(row)
        elif decision.decision == WAIT:
            state = CL.WAIT
            waits.append(row)
        else:
            state = CL.REJECTED
            rejections.append(row)
        CL.upsert(
            symbol=symbol, session_date=session, state=state,
            reason=str(row.get("reason_code") or decision.reason_code),
            scan_run_id=scan_run_id, recommendation_id_value=rid,
            decision_id_value=did, paper_intent_id=intent_id, trigger="paper",
            payload={"paper": row},
        )
        prev = CL.get(CL.candidate_id(symbol, session))
        already = bool(prev and prev.get("decision_id") == did)
        if not already:
            try:
                from product.forward_evidence import freeze_observation

                freeze_observation(
                    {**row, "scan_scanned_at": scan_run_id, "reco_tier": card.get("reco_tier")},
                    cycle_id=scan_run_id, as_of=session, group=decision.group or decision.decision,
                    entered=False, surfaced=True,
                )
            except Exception:
                pass
            try:
                from product.counterfactual_learning import freeze_decision

                freeze_decision(
                    symbol=symbol, reason_code=str(row.get("reason_code") or ""),
                    decision=str(row.get("decision") or ""),
                    entry=card.get("entry"), stop=card.get("stop"), target=card.get("target"),
                    as_of=session,
                    evidence={"scan_run_id": scan_run_id, "recommendation_id": rid, "decision_id": did},
                )
            except Exception:
                pass
    return {
        "taken": taken, "waits": waits, "rejections": rejections, "intents": intents,
        "broker_ok": broker_ok, "entry_window": window,
        "eligibility": "TRADED" if taken else ("BLOCKED_BROKER" if intents and not broker_ok else "NO_ELIGIBLE_TRADE"),
    }


def settle_official_outcomes(session: str | None = None) -> dict[str, Any]:
    """Resolve frozen decisions from official completed bars. No Kite."""
    from core.outcome_resolver import session_close_return
    from product.counterfactual_learning import classify_forward, ledger_path as cf_path
    from product.forward_evidence import attach_settlement, load_ledger

    as_of = session or session_date()
    settled, pending, failed = [], [], []
    try:
        rows = []
        path = cf_path()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        rewritten = []
        for row in rows:
            if row.get("classification") or row.get("outcome"):
                rewritten.append(row)
                continue
            symbol = str(row.get("symbol") or "")
            day = str(row.get("as_of") or as_of)[:10]
            try:
                resolved = session_close_return(symbol, day, horizon=5)
            except Exception as exc:
                failed.append({"symbol": symbol, "error": str(exc)[:160]})
                rewritten.append(row)
                continue
            if resolved is None:
                pending.append({"symbol": symbol, "as_of": day, "reason": "HORIZON_NOT_REACHED"})
                rewritten.append(row)
                continue
            _exit, ret = resolved
            classification = classify_forward(
                entry=row.get("hypothetical_entry"),
                stop=row.get("hypothetical_stop"),
                target=row.get("hypothetical_target"),
                forward_return_pct=ret,
            )
            row = dict(row)
            row["outcome"] = {"forward_return_pct": ret, "exit": _exit, "not_pnl": True, "source": "official_bhavcopy"}
            row["classification"] = classification
            rewritten.append(row)
            settled.append({"symbol": symbol, "classification": classification, "return_pct": ret, "as_of": day})
            _append_jsonl(LEARNING_PATH, {
                "at": _now(), "symbol": symbol, "as_of": day,
                "classification": classification, "return_pct": ret,
                "reason_code": row.get("reason_code"),
                "provenance": "REAL_FORWARD_MARKET",
                "updates_policy": False,
                "observation": "counterfactual_settled",
            })
            cid = CL.candidate_id(symbol, day)
            if CL.get(cid):
                CL.upsert(
                    symbol=symbol, session_date=day, state=CL.CLOSED,
                    reason=classification, outcome_id=f"{cid}:outcome", trigger="outcome",
                )
        if rewritten:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(r, default=str) + "\n" for r in rewritten), encoding="utf-8")
    except Exception as exc:
        failed.append({"error": str(exc)[:200]})

    for row in load_ledger():
        if row.get("later_outcome") or not row.get("decision_id"):
            continue
        symbol = str(row.get("symbol") or "")
        day = str(row.get("market_timestamp") or row.get("pit_proof", {}).get("as_of") or as_of)[:10]
        try:
            resolved = session_close_return(symbol, day, horizon=5)
        except Exception:
            continue
        if resolved is None:
            pending.append({"symbol": symbol, "decision_id": row.get("decision_id"), "reason": "HORIZON_NOT_REACHED"})
            continue
        _exit, ret = resolved
        classification = classify_forward(
            entry=row.get("entry"), stop=row.get("stop"), target=row.get("target"),
            forward_return_pct=ret,
        )
        attach_settlement(str(row["decision_id"]), classification=classification, forward_return_pct=ret)
        settled.append({"symbol": symbol, "decision_id": row.get("decision_id"), "classification": classification, "return_pct": ret})
    return {"settled": settled, "pending": pending, "failed": failed, "n_settled": len(settled)}


def consume_learning_memory(session: str | None = None) -> dict[str, Any]:
    """Fold settled official events into calibrated memory. Never changes policy."""
    from collections import Counter

    as_of = session or session_date()
    by_class: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    by_setup: Counter[str] = Counter()
    observations = 0
    try:
        from product.counterfactual_learning import ledger_path

        path = ledger_path()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                classification = str(row.get("classification") or "")
                if not classification:
                    continue
                observations += 1
                by_class[classification] += 1
                if row.get("reason_code"):
                    by_reason[f"{row.get('reason_code')}:{classification}"] += 1
                if row.get("setup"):
                    by_setup[f"{row.get('setup')}:{classification}"] += 1
    except Exception as exc:
        _append_jsonl(FAILURES_PATH, {
            "at": _now(), "kind": "learning_observation", "error": str(exc)[:200],
            "retry": False, "final": True,
        })
    memory = {
        "schema_version": 1,
        "as_of": as_of,
        "generated_at": _now(),
        "observations": observations,
        "classification_counts": dict(by_class),
        "reason_code_stats": dict(by_reason),
        "setup_stats": dict(by_setup),
        "updates_policy": False,
        "provenance": "REAL_FORWARD_MARKET",
        "note": "Memory only. Production trading policy is unchanged until promotion evidence exists.",
    }
    _write_json(MEMORY_PATH, memory)
    if observations:
        _append_jsonl(LEARNING_PATH, {
            "at": _now(), "observation": "memory_refresh", "as_of": as_of,
            "observations": observations, "updates_policy": False,
            "provenance": "REAL_FORWARD_MARKET",
        })
    return memory


def _reassess_reasons(prev: Mapping[str, Any], scan_run_id: str, session: str, trigger: str) -> list[str]:
    reasons = []
    if str(prev.get("scan_run_id") or "") != str(scan_run_id):
        reasons.append("new_scan_run")
    if str(prev.get("session_date") or "") != str(session):
        reasons.append("new_completed_session")
    if trigger == "DUE_DILIGENCE_ACQUIRE":
        reasons.append("new_evidence_acquired")
    if trigger == "outcome_resolution":
        reasons.append("outcome_horizon")
    if trigger == "MARKET_SCAN":
        reasons.append("scan_complete")
    if not reasons:
        reasons.append("scheduled_recheck")
    return reasons


def _should_download(trigger: str, prev: Mapping[str, Any], scan_run_id: str) -> bool:
    if trigger == "MARKET_SCAN":
        return False
    if trigger == "outcome_resolution" and str(prev.get("scan_run_id") or "") == str(scan_run_id):
        return False
    return trigger in {"DUE_DILIGENCE_ACQUIRE", "research_cycle", "pipeline", "manual"}


def project_events(limit: int = 40) -> list[dict[str, Any]]:
    """Human-readable chronology from loop events plus existing job stores."""
    rows = list(event_log(max(limit, 24)))
    try:
        import sqlite3

        con = sqlite3.connect(str(ROOT / "logs" / "market_ops" / "jobs.db"))
        con.row_factory = sqlite3.Row
        for row in con.execute(
            "SELECT kind, status, message, finished_at, requested_by FROM operations "
            "WHERE finished_at IS NOT NULL ORDER BY finished_at DESC LIMIT 24"
        ):
            rows.append({
                "at": row["finished_at"],
                "kind": str(row["kind"] or "JOB"),
                "text": f"{row['kind']} {row['status']} — {str(row['message'] or '')[:160]}",
                "source": "market_ops",
                "requested_by": row["requested_by"],
            })
        con.close()
    except Exception:
        pass
    try:
        import sqlite3

        con = sqlite3.connect(str(ROOT / "logs" / "autonomy" / "jobs.db"))
        con.row_factory = sqlite3.Row
        for row in con.execute(
            "SELECT job_type, status, result_summary, finished_at FROM jobs "
            "WHERE finished_at IS NOT NULL ORDER BY finished_at DESC LIMIT 24"
        ):
            rows.append({
                "at": row["finished_at"],
                "kind": str(row["job_type"] or "JOB"),
                "text": f"{row['job_type']} {row['status']} — {str(row['result_summary'] or '')[:160]}",
                "source": "autonomy",
            })
        con.close()
    except Exception:
        pass
    rows.sort(key=lambda r: str(r.get("at") or ""))
    return rows[-int(limit):]


def replay_compatible_rows(limit: int = 8) -> list[dict[str, Any]]:
    path = ROOT / "logs" / "product" / "historical_replay" / "decisions.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines()[-int(limit):]:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        out.append({
            "symbol": row.get("symbol"),
            "decision": row.get("decision") or row.get("action"),
            "as_of": row.get("as_of") or row.get("session_date"),
            "provenance": row.get("provenance") or "BACKTEST",
            "not_real_forward": True,
        })
    return out


def _operator_metrics() -> dict[str, Any]:
    automated = 0
    human = 0
    try:
        import sqlite3

        con = sqlite3.connect(str(ROOT / "logs" / "market_ops" / "jobs.db"))
        for by, n in con.execute("SELECT requested_by, count(*) FROM operations GROUP BY 1"):
            if str(by) in {"pipeline", "bootstrap", "autonomy", "autonomous_loop"}:
                automated += int(n)
            elif str(by) in {"terminal", "user", "product_bootstrap"}:
                human += int(n)
            else:
                automated += int(n)
        con.close()
    except Exception:
        pass
    try:
        import sqlite3

        con = sqlite3.connect(str(ROOT / "logs" / "autonomy" / "jobs.db"))
        n = list(con.execute("SELECT count(*) FROM jobs"))[0][0]
        automated += int(n)
        con.close()
    except Exception:
        pass
    total = automated + human
    return {
        "automated_jobs": automated,
        "human_required_actions": human,
        "manual_fallbacks": 0,
        "automation_rate": (automated / total) if total else 1.0,
        "necessary_human": ["Kite authentication"] if not RDY.broker_live().get("ready") else [],
    }


def advance_loop(*, trigger: str = "pipeline") -> dict[str, Any]:
    """One idempotent step of the autonomous product loop."""
    started = time.time()
    readiness = RDY.inspect_readiness()
    session = session_date()
    scan = _load_scan()
    reco = _ensure_reco(scan)
    cards = _cards(reco)
    scan_run_id = str(scan.get("scanned_at") or session)
    prev = load_summary()
    reasons = _reassess_reasons(prev, scan_run_id, session, trigger)
    emit("LOOP", f"autonomous loop · trigger={trigger} · {','.join(reasons)}", trigger=trigger, session=session)

    ingested = _ingest_candidates(scan, cards, session)
    shortlist = _research_shortlist(cards)
    research = {"symbols": [], "acquired": [], "errors": [], "n_ok": 0, "n_failed": 0, "skipped": "no_shortlist"}
    if shortlist and readiness["capabilities"].get(RDY.RESEARCH_DATA_READY):
        download = _should_download(trigger, prev, scan_run_id)
        research = _acquire(shortlist, session, scan_run_id, download=download)
        if research.get("n_ok"):
            reco = _ensure_reco(scan)
            cards = _cards(reco)
        emit(
            "RESEARCH",
            f"deep research {research.get('n_ok')}/{len(shortlist)} cached={len(research.get('cached') or [])} waiting={research.get('n_waiting')}",
            n_ok=research.get("n_ok"), n_failed=research.get("n_failed"), downloaded=research.get("downloaded"),
        )

    paper = _consume_paper(cards, reco, session, scan_run_id)
    emit("PAPER", f"paper consume · taken={len(paper.get('taken') or [])} waits={len(paper.get('waits') or [])} blocked={len(paper.get('intents') or [])}")

    outcomes = {"settled": [], "pending": [], "failed": [], "n_settled": 0}
    if readiness["capabilities"].get(RDY.OUTCOME_DATA_READY):
        outcomes = settle_official_outcomes(session)
        emit("OUTCOME", f"official settlement · {outcomes.get('n_settled')} matured · {len(outcomes.get('pending') or [])} pending")

    census = _census(scan, reco, paper, session)
    funnel = dict((census.get("funnel") or {}))
    funnel["deep_researched"] = int(research.get("n_ok") or 0)
    census["funnel"] = funnel
    memory = consume_learning_memory(session) if (outcomes.get("n_settled") or trigger in {"outcome_resolution", "learning_cycle"}) else {}
    metrics = _operator_metrics()
    next_watch = [
        str(c.get("symbol") or "")
        for c in CL.list_candidates(session_date=session, states=(CL.WATCH, CL.WAIT, CL.READY, CL.QUALIFIED), limit=20)
    ]
    summary = {
        "schema_version": 1,
        "generated_at": _now(),
        "trigger": trigger,
        "session_date": session,
        "scan_run_id": scan_run_id,
        "readiness": readiness,
        "census": census,
        "reassess_reasons": reasons,
        "research": {k: research.get(k) for k in ("symbols", "n_ok", "n_failed", "n_waiting", "errors", "downloaded", "cached") if k in research},
        "learning_memory": {k: memory.get(k) for k in ("observations", "classification_counts", "updates_policy") if memory},
        "paper": {
            "eligibility": paper.get("eligibility"),
            "taken": len(paper.get("taken") or []),
            "waits": len(paper.get("waits") or []),
            "rejections": len(paper.get("rejections") or []),
            "intents": paper.get("intents") or [],
            "broker_ok": paper.get("broker_ok"),
            "entry_window": paper.get("entry_window"),
        },
        "outcomes": outcomes,
        "next_session_watch": [s for s in next_watch if s],
        "operator_metrics": metrics,
        "candidates_touched": len(ingested),
        "duration_s": round(time.time() - started, 3),
        "live_locked": True,
    }
    _write_json(SUMMARY_PATH, summary)
    _write_json(METRICS_PATH, metrics)
    return summary


def load_summary() -> dict[str, Any]:
    try:
        return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def event_log(limit: int = 40) -> list[dict[str, Any]]:
    if not EVENTS_PATH.exists():
        return []
    rows = []
    for line in EVENTS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows[-int(limit):]


def desk_projection() -> dict[str, Any]:
    summary = load_summary()
    readiness = summary.get("readiness") or RDY.inspect_readiness()
    return {
        "schema_version": 1,
        "generated_at": _now(),
        "what_quantterm_did": project_events(24),
        "what_it_found": summary.get("census") or {},
        "what_it_is_doing": {"trigger": summary.get("trigger"), "session": summary.get("session_date")},
        "what_is_blocked": {
            "broker": readiness.get("broker"),
            "paper": (summary.get("paper") or {}).get("intents") or [],
            "outcomes_pending": (summary.get("outcomes") or {}).get("pending") or [],
        },
        "what_needs_the_operator": readiness.get("broker"),
        "what_changed": {
            "scan_run_id": summary.get("scan_run_id"),
            "candidates_touched": summary.get("candidates_touched"),
            "research": summary.get("research"),
        },
        "what_it_learned": {
            "outcomes": summary.get("outcomes"),
            "policy_changed": False,
        },
        "census": summary.get("census") or {},
        "candidate_lifecycle": CL.state_counts(str(summary.get("session_date") or "")),
        "lineage": {
            "scan_run_id": summary.get("scan_run_id"),
            "candidates": [
                {
                    "candidate_id": r.get("candidate_id"),
                    "symbol": r.get("symbol"),
                    "state": r.get("state"),
                    "scan_run_id": r.get("scan_run_id"),
                    "recommendation_id": r.get("recommendation_id"),
                    "decision_id": r.get("decision_id"),
                    "paper_intent_id": r.get("paper_intent_id"),
                    "outcome_id": r.get("outcome_id"),
                }
                for r in CL.list_candidates(session_date=str(summary.get("session_date") or ""), limit=12)
            ],
        },
        "readiness": readiness,
        "operator_metrics": summary.get("operator_metrics") or _operator_metrics(),
        "next_session_watch": summary.get("next_session_watch") or [],
        "historical_replay": replay_compatible_rows(),
        "live_locked": True,
    }
