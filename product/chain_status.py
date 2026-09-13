"""Walk the full decision chain and report where it is actually blocked.

QuantTerm has many correct gates. The failure mode is not that a gate fires --
it is that the operator cannot see WHICH gate fired, so an intentionally
blocked desk and a broken desk look identical: both are simply empty.

This module walks every link from official history through to learning and
reports, for each one, whether it is FLOWING, BLOCKED, WAITING or UNKNOWN,
with the real reason and the concrete action that would unblock it. Every value
is read from live product state; nothing here simulates or assumes.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

SCHEMA_VERSION = 1

FLOWING = "FLOWING"
BLOCKED = "BLOCKED"
WAITING = "WAITING"
UNKNOWN = "UNKNOWN"


def _link(name: str, state: str, detail: str, *, unblock: str = "", **evidence: Any) -> dict[str, Any]:
    return {"link": name, "state": state, "detail": detail,
            "unblock": unblock, "evidence": evidence}


def _safe(fn: Callable[[], dict[str, Any]], name: str) -> dict[str, Any]:
    try:
        return fn()
    except Exception as exc:
        return _link(name, UNKNOWN, f"check failed: {type(exc).__name__}",
                     unblock="Read the service logs for this subsystem.")


def _official_history() -> dict[str, Any]:
    from data.bhavcopy_runtime import official_history_freshness

    f = official_history_freshness()
    if not f.get("ready"):
        return _link("Official history", BLOCKED,
                     f"No canonical NSE store ({f.get('reason_code')}).",
                     unblock="Run the first history build; it needs market egress to NSE.",
                     sessions=f.get("sessions"), symbols=f.get("symbols"))
    if not f.get("current"):
        return _link("Official history", WAITING,
                     f"Store ends {f.get('available_session')}, expected "
                     f"{f.get('expected_latest_completed_session')} "
                     f"({f.get('stale_sessions')} sessions behind).",
                     unblock="Let the data-refresh job run, or trigger Refresh market data.",
                     available_session=f.get("available_session"))
    return _link("Official history", FLOWING,
                 f"{f.get('sessions')} sessions to {f.get('available_session')}.",
                 sessions=f.get("sessions"), symbols=f.get("symbols"))


def _raw_sessions_for_replay() -> dict[str, Any]:
    """The historical bootstrap reads per-session CSVs, not the consolidated cache."""
    from product.autonomous_evolution import (
        DEFAULT_OUTCOME_BUFFER, DEFAULT_SESSIONS_PER_SPLIT, DEFAULT_SPLITS,
    )
    from product.historical_replay import official_sessions

    needed = DEFAULT_SPLITS * DEFAULT_SESSIONS_PER_SPLIT + DEFAULT_OUTCOME_BUFFER + 2
    have = len(official_sessions())
    if have < needed:
        return _link("Replay sessions on disk", BLOCKED,
                     f"{have} raw session files; the reproduction plan needs {needed}.",
                     unblock=("Keep raw bhavcopy CSVs in logs/bhav. The consolidated cache "
                              "alone is not enough for independent PIT slices."),
                     have=have, needed=needed)
    return _link("Replay sessions on disk", FLOWING,
                 f"{have} raw session files available (need {needed}).",
                 have=have, needed=needed)


def _latest_scan() -> dict[str, Any]:
    from product.scan_store import load_scan

    scan = load_scan()
    if not scan:
        return _link("Whole-market scan", BLOCKED, "No saved scan artifact.",
                     unblock="Run a market scan, or wait for the scheduled one.")
    prov = scan.get("provenance") or {}
    records = len(scan.get("records") or [])
    ready = int((scan.get("summary") or {}).get("ready_to_trade") or 0)
    if not records:
        return _link("Whole-market scan", WAITING,
                     "Scan ran but produced no qualifying rows.",
                     unblock="Normal in a weak tape. Check universe coverage if it persists.",
                     scanned=scan.get("scanned"))
    return _link("Whole-market scan", FLOWING,
                 f"{records} rows from {scan.get('scanned')} scanned "
                 f"(session {prov.get('market_session_date') or 'unknown'}); "
                 f"{ready} ready to trade.",
                 records=records, ready_to_trade=ready,
                 market_session_date=prov.get("market_session_date"),
                 data_freshness=prov.get("data_freshness"))


def _candidates() -> dict[str, Any]:
    from collections import Counter

    from product.recommendations_store import load_recommendations

    ws = load_recommendations() or {}
    cards = [c for cat in (ws.get("categories") or []) for c in (cat.get("cards") or [])]
    if not cards:
        return _link("Candidates", WAITING, "No recommendation cards published yet.",
                     unblock="Cards are built from the latest scan; run one if the scan is stale.")
    tiers = Counter(str(c.get("reco_tier")) for c in cards)
    eligible = tiers.get("high_conviction", 0) + tiers.get("good_setup", 0)
    if not eligible:
        return _link("Candidates", WAITING,
                     f"{len(cards)} cards, none above watch tier.",
                     unblock="Only high_conviction and good_setup are auto-enterable.",
                     tiers=dict(tiers))
    return _link("Candidates", FLOWING,
                 f"{eligible} of {len(cards)} cards are auto-enterable.",
                 tiers=dict(tiers))


def _historical_gate() -> dict[str, Any]:
    from product.autonomous_evolution import bootstrap_status

    s = bootstrap_status()
    if not s.get("analysis_complete"):
        return _link("Historical reproduction", WAITING,
                     f"{s.get('status')}: {s.get('reason') or 'bootstrap has not finished'}",
                     unblock="The bootstrap self-starts; it needs enough raw sessions on disk.",
                     status=s.get("status"))
    ready = int(s.get("paper_ready_setups") or 0)
    if not ready:
        return _link("Historical reproduction", BLOCKED,
                     "Bootstrap complete, but no setup reproduced across independent PIT slices.",
                     unblock=("This is the evidence gate working. A setup must prove itself in "
                              "history before it may auto-enter paper. Nothing to fix."),
                     paper_ready_setups=0, history_anchor=s.get("history_anchor"))
    return _link("Historical reproduction", FLOWING,
                 f"{ready} setups reproduced and are paper-eligible.",
                 paper_ready_setups=ready)


def _paper_capability() -> dict[str, Any]:
    from product.autonomy_status import read_autonomy_status
    from research.autonomy import health as H

    status = read_autonomy_status() or {}
    failures = set(status.get("active_failures") or [])
    caps = H.capabilities(failures)
    level = caps["new_paper_entries"]
    if level == H.BLOCKED:
        notes = caps.get("notes") or []
        return _link("Paper entry capability", BLOCKED,
                     notes[0] if notes else "New paper entries are blocked.",
                     unblock="Clear the failing subsystem, or resume if you paused it.",
                     active_failures=sorted(failures))
    return _link("Paper entry capability", FLOWING,
                 f"New paper entries {level}.",
                 active_failures=sorted(failures))


def _open_positions() -> dict[str, Any]:
    from product.forward_evidence_board import _open_positions as positions

    rows = positions(None)
    if not rows:
        return _link("Open paper positions", WAITING, "No open paper positions.",
                     unblock="Positions appear once a candidate clears every gate above.")
    return _link("Open paper positions", FLOWING, f"{len(rows)} open.", count=len(rows))


def _forward_evidence() -> dict[str, Any]:
    from product.forward_evidence_board import build_forward_evidence_board

    board = build_forward_evidence_board()
    state = str(board.get("state") or "")
    settled = int((board.get("totals") or {}).get("settled") or 0)
    if state == "NO_MARKET_EVIDENCE":
        return _link("Forward evidence", WAITING,
                     "No paper trade has settled yet.",
                     unblock="Settled outcomes accrue only after positions close.",
                     board_state=state)
    return _link("Forward evidence", FLOWING, f"{settled} settled observations.",
                 board_state=state, settled=settled)


def build_chain_status() -> dict[str, Any]:
    """Every link in the decision chain, in order, with its real state."""
    links = [
        _safe(_official_history, "Official history"),
        _safe(_raw_sessions_for_replay, "Replay sessions on disk"),
        _safe(_latest_scan, "Whole-market scan"),
        _safe(_candidates, "Candidates"),
        _safe(_historical_gate, "Historical reproduction"),
        _safe(_paper_capability, "Paper entry capability"),
        _safe(_open_positions, "Open paper positions"),
        _safe(_forward_evidence, "Forward evidence"),
    ]
    first_stop = next((l for l in links if l["state"] in {BLOCKED, WAITING}), None)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "links": links,
        "flowing": sum(1 for l in links if l["state"] == FLOWING),
        "total": len(links),
        "first_stop": first_stop["link"] if first_stop else "",
        "first_stop_detail": first_stop["detail"] if first_stop else "",
        "first_stop_unblock": first_stop["unblock"] if first_stop else "",
        "chain_complete": first_stop is None,
    }


def render_text(payload: dict[str, Any]) -> str:
    mark = {FLOWING: "ok  ", BLOCKED: "STOP", WAITING: "wait", UNKNOWN: "????"}
    width = max(len(l["link"]) for l in payload["links"])
    out = [f"QUANTTERM DECISION CHAIN   {payload['flowing']}/{payload['total']} links flowing", ""]
    for l in payload["links"]:
        out.append(f"  [{mark[l['state']]}] {l['link']:<{width}}  {l['detail']}")
    if payload["first_stop"]:
        out += ["", f"FIRST STOP: {payload['first_stop']}",
                f"  {payload['first_stop_detail']}",
                f"  -> {payload['first_stop_unblock']}"]
    else:
        out += ["", "Every link is flowing."]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Show where the decision chain is blocked")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    payload = build_chain_status()
    print(json.dumps(payload, indent=2, default=str) if args.json else render_text(payload))
    return 0 if payload["chain_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
