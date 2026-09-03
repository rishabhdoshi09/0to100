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


_TIER_RANK = {"high_conviction": 0, "good_setup": 1, "watch": 2}


def _cards(reco: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One card per symbol, keeping the strongest recommendation tier."""
    best: dict[str, dict[str, Any]] = {}
    for cat in reco.get("categories") or []:
        if not isinstance(cat, dict):
            continue
        for card in cat.get("cards") or []:
            if not isinstance(card, dict):
                continue
            symbol = str(card.get("symbol") or "").upper()
            if not symbol:
                continue
            row = dict(card)
            row["symbol"] = symbol
            prev = best.get(symbol)
            if prev is None or _TIER_RANK.get(str(row.get("reco_tier")), 9) < _TIER_RANK.get(str(prev.get("reco_tier")), 9):
                best[symbol] = row
    if best:
        return list(best.values())
    try:
        from product.autopilot_journal import flatten_cards

        return [dict(c) for c in flatten_cards(reco) if isinstance(c, dict)]
    except Exception:
        return []


def _census(
    scan: Mapping[str, Any],
    reco: Mapping[str, Any],
    paper: Mapping[str, Any],
    session: str = "",
    *,
    committee: list[dict[str, Any]] | None = None,
    scan_run_id: str = "",
    researched: list[str] | None = None,
    generated_at: str = "",
) -> dict[str, Any]:
    from product.judgment_census import build_census

    return build_census(
        scan=scan,
        reco=reco,
        committee=committee or paper.get("committee") or [],
        session=str(session or session_date())[:10],
        scan_run_id=scan_run_id or str(scan.get("scanned_at") or ""),
        generated_at=generated_at or _now(),
        researched_symbols=researched or [],
        candidate_states=CL.state_counts(str(session or session_date())[:10]),
    )


def _ingest_candidates(scan: Mapping[str, Any], cards: list[dict[str, Any]], session: str) -> list[dict[str, Any]]:
    scan_run_id = str(scan.get("scanned_at") or scan.get("source_snapshot_id") or session)
    rows = []
    remembered = set()
    try:
        from product.opportunity_memory import list_open

        remembered = {str(r.get("symbol") or "").upper() for r in list_open(limit=200)}
    except Exception:
        remembered = set()
    reco_symbols = {str(c.get("symbol") or "").upper() for c in cards if c.get("symbol")}
    keep = reco_symbols | remembered
    for rec in scan.get("records") or []:
        if not isinstance(rec, dict):
            continue
        symbol = str(rec.get("symbol") or "").upper()
        if not symbol or symbol not in keep:
            continue
        status = str(rec.get("status") or "")
        verdict = str(rec.get("verdict") or status or "").upper()
        if verdict in {"WATCH", "WAIT"} or "wait" in status.lower() or "pullback" in status.lower():
            state = CL.WATCH
        else:
            state = CL.SCREENED
        rows.append(CL.upsert(
            symbol=symbol, session_date=session, state=state,
            reason=str((rec.get("reasons") or ["scan"])[0])[:240],
            scan_run_id=scan_run_id, trigger="scan",
            opportunity_id_value=symbol,
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


def _research_shortlist(cards: list[dict[str, Any]], committee: list[dict[str, Any]] | None = None) -> list[str]:
    """Deep-research only when extra evidence can change a meaningful decision."""
    by_sym = {str(r.get("symbol") or "").upper(): r for r in (committee or [])}
    ranked: list[tuple[int, str, str]] = []
    for card in cards:
        symbol = str(card.get("symbol") or "").upper()
        if not symbol:
            continue
        rec = by_sym.get(symbol) or {}
        info = str(rec.get("information_value") or "")
        if info == "NONE":
            continue
        if rec.get("decision") == "AVOID" and info != "HIGH":
            continue
        rank = 0 if str(card.get("reco_tier")) == "high_conviction" else 1
        if info == "HIGH":
            rank -= 1
        if rec.get("research_required"):
            rank -= 1
        if _facts_present(symbol):
            continue
        ranked.append((rank, symbol, info))
    ranked.sort()
    seen: list[str] = []
    for _rank, symbol, _info in ranked:
        if symbol not in seen:
            seen.append(symbol)
        if len(seen) >= DEEP_RESEARCH_CAP:
            break
    if seen:
        return seen
    # Fallback: high-conviction names still missing facts.
    for card in cards:
        symbol = str(card.get("symbol") or "").upper()
        if str(card.get("reco_tier")) == "high_conviction" and symbol and not _facts_present(symbol):
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


def _committee_cards(cards: list[dict[str, Any]], extra_symbols: set[str] | None = None) -> list[dict[str, Any]]:
    extra = {str(s).upper() for s in (extra_symbols or set()) if s}
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for card in cards:
        symbol = str(card.get("symbol") or "").upper()
        if not symbol or symbol in seen:
            continue
        tier = str(card.get("reco_tier") or "")
        if tier in {"high_conviction", "good_setup"} or symbol in extra:
            seen.add(symbol)
            out.append(card)
    return out


def _entry_window() -> bool:
    try:
        from research.autonomy import schedules as SCH
        from zoneinfo import ZoneInfo

        return bool(SCH.entries_allowed_by_clock(datetime.now(ZoneInfo("Asia/Kolkata"))))
    except Exception:
        return False


def _evaluate_committee(
    cards: list[dict[str, Any]],
    reco: Mapping[str, Any],
    *,
    extra_symbols: set[str] | None = None,
    broker_ok: bool | None = None,
    entry_window: bool | None = None,
) -> list[dict[str, Any]]:
    from product.decision_committee import evaluate_many

    readiness = RDY.inspect_readiness()
    if broker_ok is None:
        broker_ok = bool(readiness["capabilities"].get(RDY.BROKER_LIVE_DATA_READY))
    if entry_window is None:
        entry_window = _entry_window()
    selected = _committee_cards(cards, extra_symbols)
    return [
        rec.as_dict()
        for rec in evaluate_many(
            selected,
            book=None,
            broker_ok=bool(broker_ok),
            entry_window=bool(entry_window),
            workspace=reco,
        )
    ]


def _consume_paper(
    cards: list[dict[str, Any]],
    reco: Mapping[str, Any],
    session: str,
    scan_run_id: str,
    *,
    committee: list[dict[str, Any]] | None = None,
    extra_symbols: set[str] | None = None,
) -> dict[str, Any]:
    """Persist committee judgment. Broker login is execution-only."""
    readiness = RDY.inspect_readiness()
    broker_ok = bool(readiness["capabilities"].get(RDY.BROKER_LIVE_DATA_READY))
    window = _entry_window()
    records = list(committee or [])
    if not records:
        records = _evaluate_committee(
            cards, reco, extra_symbols=extra_symbols, broker_ok=broker_ok, entry_window=window,
        )
    by_card = {str(c.get("symbol") or "").upper(): c for c in cards}
    taken, waits, rejections, intents = [], [], [], []
    from product import decision_journal as DJ
    from product import opportunity_memory as OM

    for rec in records:
        symbol = str(rec.get("symbol") or "").upper()
        if not symbol:
            continue
        card = by_card.get(symbol) or {"symbol": symbol}
        if not rec.get("sector"):
            rec["sector"] = card.get("sector") or ""
        try:
            from product.portfolio_committee import apply_overlay, evaluate_portfolio

            overlay = evaluate_portfolio(rec, book=None, as_of=session)
            rec = apply_overlay(rec, overlay)
        except Exception:
            pass
        for i, prev_rec in enumerate(records):
            if str(prev_rec.get("symbol") or "").upper() == symbol:
                records[i] = rec
                break
        rid = CL.recommendation_id(scan_run_id, symbol, str(rec.get("tier") or card.get("reco_tier") or ""))
        did = f"{session}|{symbol}|{rec.get('decision')}|{rec.get('reason_code')}|{scan_run_id}"
        intent_id = f"{did}:intent"
        state = str(rec.get("candidate_state") or CL.WATCH)
        demote = state in {CL.WAIT, CL.WAIT_EVIDENCE, CL.REJECTED, CL.WATCH}
        row = dict(rec)
        row["decision_id"] = did
        row["recommendation_id"] = rid
        row["candidate_id"] = CL.candidate_id(symbol, session)
        row["execution_block"] = rec.get("execution_state")
        prev = CL.get(CL.candidate_id(symbol, session))
        already = bool(prev and prev.get("decision_id") == did)
        from product.decision_taxonomy import is_judgment_row

        judgment = is_judgment_row(rec)
        if rec.get("decision") == "BUY" and rec.get("candidate_state") == CL.READY:
            if str(rec.get("execution_state") or "").startswith("BLOCKED"):
                intents.append(row)
                if str(rec.get("execution_state") or "") == "BLOCKED_BROKER_AUTH":
                    try:
                        from product.shadow_execution import SHADOW_NOT_EXECUTED, freeze_shadow

                        shadow = freeze_shadow({**row, "shadow_status": SHADOW_NOT_EXECUTED})
                        row["shadow_status"] = shadow.get("status")
                        rec["shadow_status"] = shadow.get("status")
                    except Exception:
                        pass
            else:
                taken.append(row)
        elif rec.get("decision") == "WAIT":
            waits.append(row)
        elif judgment:
            rejections.append(row)
        CL.upsert(
            symbol=symbol, session_date=session, state=state,
            reason=str(rec.get("reason_code") or rec.get("reason") or ""),
            scan_run_id=scan_run_id, recommendation_id_value=rid,
            decision_id_value=did, paper_intent_id=intent_id, trigger="committee",
            decision=str(rec.get("decision") or ""),
            entry_state=str(rec.get("entry_state") or ""),
            execution_state=str(rec.get("execution_state") or ""),
            wait_trigger=rec.get("wait_trigger") or {},
            opportunity_id_value=symbol,
            payload={"committee": rec},
            demote=demote,
        )
        DJ.persist({
            **rec,
            "decision_id": did,
            "candidate_id": CL.candidate_id(symbol, session),
            "opportunity_id": symbol,
            "scan_run_id": scan_run_id,
            "recommendation_id": rid,
            "decision_time": _now(),
            "market_as_of": session,
            "evidence_cutoff": session,
        })
        OM.remember(
            symbol=symbol, session_date=session, scan_run_id=scan_run_id,
            state=state, decision=str(rec.get("decision") or ""),
            entry_state=str(rec.get("entry_state") or ""),
            execution_state=str(rec.get("execution_state") or ""),
            reason=str(rec.get("reason_code") or ""),
            setup=str((rec.get("references") or {}).get("primary_thesis") or ""),
            tier=str(rec.get("tier") or ""),
            wait_trigger=rec.get("wait_trigger") or {},
            research_completed=bool(rec.get("evidence_coverage_pct")),
            payload={"decision_id": did, "wait_trigger": rec.get("wait_trigger") or {}},
        )
        if not already and judgment:
            try:
                from product.forward_evidence import freeze_observation

                freeze_observation(
                    {**row, "scan_scanned_at": scan_run_id, "reco_tier": rec.get("tier")},
                    cycle_id=scan_run_id, as_of=session,
                    group=str(rec.get("decision") or ""),
                    entered=False, surfaced=True,
                )
            except Exception:
                pass
            try:
                from product.counterfactual_learning import freeze_decision

                freeze_decision(
                    symbol=symbol,
                    reason_code=str(rec.get("reason_code") or ""),
                    decision=str(rec.get("decision") or ""),
                    entry=rec.get("entry") or card.get("entry"),
                    stop=rec.get("stop") or card.get("stop"),
                    target=rec.get("target") or card.get("target"),
                    as_of=session,
                    evidence={
                        "scan_run_id": scan_run_id,
                        "recommendation_id": rid,
                        "decision_id": did,
                        "entry_state": rec.get("entry_state"),
                        "execution_state": rec.get("execution_state"),
                    },
                )
            except Exception:
                pass
    return {
        "taken": taken, "waits": waits, "rejections": rejections, "intents": intents,
        "committee": records,
        "broker_ok": broker_ok, "entry_window": window,
        "eligibility": (
            "TRADED" if taken else (
                "BLOCKED_BROKER" if intents and not broker_ok else "NO_ELIGIBLE_TRADE"
            )
        ),
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
            from product.missed_winner import analyze_decision_quality

            quality = analyze_decision_quality(
                row, classification=classification, forward_return_pct=ret,
            )
            row["decision_quality"] = quality
            _append_jsonl(LEARNING_PATH, {
                "at": _now(), "symbol": symbol, "as_of": day,
                "classification": classification, "return_pct": ret,
                "reason_code": row.get("reason_code"),
                "decision_quality": quality,
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
    by_reason_detail: dict[str, dict[str, int]] = {}
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
                    detail = by_reason_detail.setdefault(str(row.get("reason_code")), {
                        "decisions": 0, "later_rallied": 0, "valid_entry_after_wait": 0,
                        "missed_winner": 0, "correct_rejection": 0, "good_wait": 0,
                    })
                    detail["decisions"] += 1
                    if classification == "MISSED_WINNER":
                        detail["later_rallied"] += 1
                        detail["missed_winner"] += 1
                    if classification == "CORRECT_REJECTION":
                        detail["correct_rejection"] += 1
                    if classification == "GOOD_WAIT":
                        detail["valid_entry_after_wait"] += 1
                        detail["good_wait"] += 1
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
        "reason_aggregations": by_reason_detail,
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


def _operator_metrics(session: str = "") -> dict[str, Any]:
    from product.operator_metrics import build_operator_metrics

    return build_operator_metrics(session=session)


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

    from product import opportunity_memory as OM
    from product.due_diligence.acquire import write_research_queue

    ingested = _ingest_candidates(scan, cards, session)
    woken = []
    try:
        woken = OM.wake_candidates(scan)
        if trigger == "DUE_DILIGENCE_ACQUIRE":
            for row in OM.list_open(states=(CL.WAIT_EVIDENCE,)):
                symbol = str(row.get("symbol") or "").upper()
                if symbol and _facts_present(symbol):
                    woken.append({
                        "symbol": symbol, "wake_event": "EVIDENCE_ACQUIRED",
                        "old_state": row.get("last_state"),
                    })
        for item in woken:
            emit("WAKE", f"{item.get('symbol')} woke · {item.get('wake_event')}", **item)
            OM.remember(
                symbol=str(item.get("symbol") or ""), session_date=session,
                scan_run_id=scan_run_id, wake_event=str(item.get("wake_event") or "WAKE"),
                state=str(item.get("old_state") or ""),
                reason=str(item.get("wake_event") or ""),
            )
    except Exception as exc:
        emit("WAKE", f"wake pass skipped · {str(exc)[:120]}")

    extra = {str(w.get("symbol") or "").upper() for w in woken if w.get("symbol")}
    try:
        extra |= {
            str(r.get("symbol") or "").upper()
            for r in OM.list_open(states=(CL.WAIT, CL.WAIT_EVIDENCE, CL.WATCH, CL.READY), limit=200)
            if r.get("symbol")
        }
    except Exception:
        pass

    committee = _evaluate_committee(cards, reco, extra_symbols=extra)
    committee_before = {str(r.get("symbol") or "").upper(): dict(r) for r in committee}
    shortlist = _research_shortlist(cards, committee=committee)
    write_research_queue(
        shortlist, scan_run_id=scan_run_id, session=session,
        reasons={r.get("symbol"): r.get("information_value") for r in committee if r.get("symbol")},
    )
    research = {"symbols": [], "acquired": [], "errors": [], "n_ok": 0, "n_failed": 0, "skipped": "no_shortlist"}
    if shortlist and readiness["capabilities"].get(RDY.RESEARCH_DATA_READY):
        download = _should_download(trigger, prev, scan_run_id)
        research = _acquire(shortlist, session, scan_run_id, download=download)
        if research.get("n_ok"):
            reco = _ensure_reco(scan)
            cards = _cards(reco)
            researched = {str(x.get("symbol") or "").upper() for x in (research.get("acquired") or []) if x.get("symbol")}
            committee = _evaluate_committee(cards, reco, extra_symbols=extra | researched)
            try:
                from product.research_value import record_research_effect

                for rec in committee:
                    sym = str(rec.get("symbol") or "").upper()
                    if sym not in researched:
                        continue
                    record_research_effect(
                        symbol=sym,
                        before=committee_before.get(sym) or {},
                        after=rec,
                        missing_before=list((committee_before.get(sym) or {}).get("missing_critical") or []),
                        research_type="DUE_DILIGENCE_ACQUIRE",
                    )
            except Exception:
                pass
        emit(
            "RESEARCH",
            f"deep research {research.get('n_ok')}/{len(shortlist)} cached={len(research.get('cached') or [])} waiting={research.get('n_waiting')}",
            n_ok=research.get("n_ok"), n_failed=research.get("n_failed"), downloaded=research.get("downloaded"),
        )

    paper = _consume_paper(
        cards, reco, session, scan_run_id, committee=committee, extra_symbols=extra,
    )
    emit(
        "COMMITTEE",
        f"committee · buy={len(paper.get('taken') or [])} wait={len(paper.get('waits') or [])} "
        f"avoid={len(paper.get('rejections') or [])} exec_blocked={len(paper.get('intents') or [])}",
    )

    outcomes = {"settled": [], "pending": [], "failed": [], "n_settled": 0}
    if readiness["capabilities"].get(RDY.OUTCOME_DATA_READY):
        outcomes = settle_official_outcomes(session)
        emit("OUTCOME", f"official settlement · {outcomes.get('n_settled')} matured · {len(outcomes.get('pending') or [])} pending")

    researched_names = list(research.get("symbols") or shortlist or [])
    census = _census(
        scan, reco, paper, session,
        committee=paper.get("committee") or committee,
        scan_run_id=scan_run_id,
        researched=researched_names,
        generated_at=_now(),
    )
    memory = consume_learning_memory(session) if (outcomes.get("n_settled") or trigger in {"outcome_resolution", "learning_cycle"}) else {}
    metrics = _operator_metrics(session)
    next_set = OM.next_session_set(session)
    _write_json(ROOT / "logs" / "product" / "next_session_set.json", {
        "session": session, "scan_run_id": scan_run_id, "generated_at": _now(), "buckets": next_set,
    })
    next_watch = [
        str(item.get("symbol") or "")
        for bucket in next_set.values()
        for item in bucket
        if item.get("symbol")
    ]
    summary = {
        "schema_version": 2,
        "generated_at": _now(),
        "trigger": trigger,
        "session_date": session,
        "scan_run_id": scan_run_id,
        "readiness": readiness,
        "census": census,
        "reassess_reasons": reasons,
        "woken": woken,
        "research": {k: research.get(k) for k in ("symbols", "n_ok", "n_failed", "n_waiting", "errors", "downloaded", "cached") if k in research},
        "learning_memory": {k: memory.get(k) for k in ("observations", "classification_counts", "reason_aggregations", "updates_policy") if memory},
        "paper": {
            "eligibility": paper.get("eligibility"),
            "taken": len(paper.get("taken") or []),
            "waits": len(paper.get("waits") or []),
            "rejections": len(paper.get("rejections") or []),
            "intents": paper.get("intents") or [],
            "broker_ok": paper.get("broker_ok"),
            "entry_window": paper.get("entry_window"),
            "committee": paper.get("committee") or [],
        },
        "outcomes": outcomes,
        "next_session_set": next_set,
        "next_session_watch": [s for s in next_watch if s],
        "operator_metrics": metrics,
        "candidates_touched": len(ingested),
        "duration_s": round(time.time() - started, 3),
        "live_locked": True,
    }
    if trigger in {"outcome_resolution", "learning_cycle"} and not _entry_window():
        try:
            from product.pit_backfill import consume_data_debt

            debt = consume_data_debt(limit=2, entry_window=False)
            summary["pit_data_debt"] = {
                "acquired": debt.get("acquired"),
                "failed": debt.get("failed"),
                "skipped": debt.get("skipped"),
            }
        except Exception as exc:
            summary["pit_data_debt"] = {"skipped": True, "error": str(exc)[:160]}
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


def _what_it_learned(summary: Mapping[str, Any] | None = None) -> dict[str, Any]:
    summary = summary or {}
    try:
        from product.learning_ledger import learned_today

        today = learned_today(str(summary.get("session_date") or "")[:10] or "")
    except Exception:
        today = {
            "summary": "Nothing statistically meaningful.",
            "policy_changed": False,
            "statistically_meaningful": [],
        }
    return {
        "outcomes": summary.get("outcomes"),
        "reason_aggregations": (summary.get("learning_memory") or {}).get("reason_aggregations"),
        "policy_changed": False,
        "production_impact": "none",
        "learning_level": 1,
        "today": today.get("summary") or "Nothing statistically meaningful.",
        "entries": today.get("entries") or [],
        "statistically_meaningful": today.get("statistically_meaningful") or [],
        "research_value": _safe_research_value(),
    }


def _safe_research_value() -> dict[str, Any]:
    try:
        from product.research_value import summary as research_summary

        return research_summary()
    except Exception:
        return {"n": 0, "note": "No research-value observations yet."}


def desk_projection() -> dict[str, Any]:
    summary = load_summary()
    readiness = summary.get("readiness") or RDY.inspect_readiness()
    session = str(summary.get("session_date") or "")
    committee = list((summary.get("paper") or {}).get("committee") or [])
    return {
        "schema_version": 2,
        "generated_at": _now(),
        "what_quantterm_did": project_events(24),
        "what_it_found": summary.get("census") or {},
        "what_it_is_doing": {"trigger": summary.get("trigger"), "session": session},
        "what_is_blocked": {
            "broker": readiness.get("broker"),
            "execution": [
                {
                    "symbol": r.get("symbol"),
                    "decision": r.get("decision"),
                    "candidate_state": r.get("candidate_state"),
                    "entry_state": r.get("entry_state"),
                    "execution_state": r.get("execution_state"),
                    "reason_code": r.get("reason_code"),
                }
                for r in committee
                if str(r.get("execution_state") or "").startswith("BLOCKED")
            ],
            "paper": (summary.get("paper") or {}).get("intents") or [],
            "outcomes_pending": (summary.get("outcomes") or {}).get("pending") or [],
        },
        "what_needs_the_operator": readiness.get("broker"),
        "what_changed": {
            "scan_run_id": summary.get("scan_run_id"),
            "candidates_touched": summary.get("candidates_touched"),
            "research": summary.get("research"),
            "woken": summary.get("woken") or [],
        },
        "what_it_learned": _what_it_learned(summary),
        "census": summary.get("census") or {},
        "candidate_lifecycle": CL.state_counts(session),
        "judgments": {
            r.get("symbol"): {
                "candidate_state": r.get("candidate_state"),
                "decision": r.get("decision"),
                "entry_state": r.get("entry_state"),
                "execution_state": r.get("execution_state"),
                "reason_code": r.get("reason_code"),
            }
            for r in committee
            if r.get("symbol")
        },
        "lineage": {
            "scan_run_id": summary.get("scan_run_id"),
            "candidates": [
                {
                    "candidate_id": r.get("candidate_id"),
                    "symbol": r.get("symbol"),
                    "state": r.get("state"),
                    "decision": r.get("decision"),
                    "entry_state": r.get("entry_state"),
                    "execution_state": r.get("execution_state"),
                    "scan_run_id": r.get("scan_run_id"),
                    "recommendation_id": r.get("recommendation_id"),
                    "decision_id": r.get("decision_id"),
                    "paper_intent_id": r.get("paper_intent_id"),
                    "outcome_id": r.get("outcome_id"),
                    "opportunity_id": r.get("opportunity_id"),
                }
                for r in CL.list_candidates(session_date=session, limit=12)
            ],
        },
        "readiness": readiness,
        "operator_metrics": summary.get("operator_metrics") or _operator_metrics(session),
        "next_session_set": summary.get("next_session_set") or {},
        "next_session_watch": summary.get("next_session_watch") or [],
        "historical_replay": replay_compatible_rows(),
        "live_locked": True,
    }
