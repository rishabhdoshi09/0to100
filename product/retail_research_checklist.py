"""Retail research checklist — what a cash trader still needs before trusting edge.

Pure projection over already-persisted status dicts. Starts no workers, invents
no data, and never claims LIVE readiness.
"""
from __future__ import annotations

from typing import Any, Mapping


def _item(
    *,
    key: str,
    label: str,
    status: str,
    why_it_matters: str,
    next_action: str,
    evidence: str,
) -> dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "status": status,
        "why_it_matters": why_it_matters,
        "next_action": next_action,
        "evidence": evidence,
    }


def build_retail_research_checklist(
    *,
    data: Mapping[str, Any] | None = None,
    ca: Mapping[str, Any] | None = None,
    universe: Mapping[str, Any] | None = None,
    pit_valuations: Mapping[str, Any] | None = None,
    live_edge: Mapping[str, Any] | None = None,
    book_correlation: Mapping[str, Any] | None = None,
    options_eod: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    data = dict(data or {})
    bhav = dict(data.get("bhavcopy", {}) or {})
    snapshot = dict(data.get("snapshot", {}) or {})
    options_eod = dict(options_eod or data.get("options_eod", {}) or {})
    ca = dict(ca or {})
    universe = dict(universe or {})
    pit_valuations = dict(pit_valuations or {})
    live_edge = dict(live_edge or {})
    book_correlation = dict(book_correlation or {})
    overall = dict(live_edge.get("overall", {}) or {})

    sessions = int(bhav.get("sessions", 0) or 0)
    min_sessions = int(bhav.get("minimum_sessions", 60) or 60)
    bhav_ok = bool(bhav.get("ready")) and sessions >= min_sessions
    snap_ok = bool(snapshot.get("ready")) and bool(snapshot.get("snapshot_id"))
    ca_ok = bool(ca.get("research_grade")) and int(ca.get("events", 0) or 0) > 0
    uni_ok = bool(universe.get("research_grade"))
    uni_present = bool(universe.get("survivorship_complete"))
    pit_ok = bool(pit_valuations.get("research_grade")) and int(pit_valuations.get("rows", 0) or 0) > 0
    live_n = int(overall.get("n", 0) or 0)
    live_ok = live_n >= 30
    live_partial = 1 <= live_n < 30
    n_pos = int(book_correlation.get("n_positions", 0) or 0)
    n_bets = int(book_correlation.get("n_bets", 0) or 0)
    book_ok = n_pos >= 2
    opts_ok = bool(options_eod.get("available")) and int(options_eod.get("snapshots", 0) or 0) > 0

    items = [
        _item(
            key="bhav_history",
            label="Official price history",
            status="READY" if bhav_ok else "MISSING",
            why_it_matters="Charts, scanners and risk math need enough NSE EOD sessions.",
            next_action="REFRESH_DATA_NOW" if not bhav_ok else "NONE",
            evidence=f"{sessions} sessions · {int(bhav.get('symbols', 0) or 0):,} symbols",
        ),
        _item(
            key="verified_snapshot",
            label="Verified research snapshot",
            status="READY" if snap_ok else "MISSING",
            why_it_matters="Paper autonomy and reproducible research need a pinned bar set.",
            next_action="python main.py snapshot-certify" if not snap_ok else "NONE",
            evidence=str(snapshot.get("snapshot_id") or "none"),
        ),
        _item(
            key="corporate_actions",
            label="Corporate-action ledger",
            status="READY" if ca_ok else "MISSING",
            why_it_matters="Without splits/bonuses, historical moves can look like fake crashes or breakouts.",
            next_action=(
                "python main.py ca-ingest --source logs/ca_events.incoming.json"
                if not ca_ok else "NONE"
            ),
            evidence=(
                f"{int(ca.get('symbols', 0) or 0)} symbols / {int(ca.get('events', 0) or 0)} events"
                if ca else "ledger empty — QuantTerm will not invent events"
            ),
        ),
        _item(
            key="universe_history",
            label="Listing / delisting history",
            status="READY" if uni_ok else ("PARTIAL" if uni_present else "MISSING"),
            why_it_matters="Today's survivors alone make past strategies look better than they were.",
            next_action=(
                "python main.py universe-history --source logs/universe_history.incoming.csv"
                if not uni_ok else "NONE"
            ),
            evidence=(
                f"source={universe.get('source') or 'none'} · "
                f"research_grade={bool(universe.get('research_grade'))}"
            ),
        ),
        _item(
            key="pit_valuations",
            label="Point-in-time valuations",
            status="READY" if pit_ok else "MISSING",
            why_it_matters="Current PE on past bars is look-ahead. Only publication-dated figures are honest.",
            next_action=(
                "python main.py pit-valuations --source logs/pit_valuations.incoming.json"
                if not pit_ok else "NONE"
            ),
            evidence=f"{int(pit_valuations.get('rows', 0) or 0)} rows · "
                     f"{int(pit_valuations.get('symbols', 0) or 0)} symbols",
        ),
        _item(
            key="live_edge",
            label="Live edge after costs",
            status="READY" if live_ok else ("PARTIAL" if live_partial else "MISSING"),
            why_it_matters="Gross backtest R is not take-home money. Need enough closed outcomes net of costs.",
            next_action="Keep tracking closed paper/live outcomes until n≥30" if not live_ok else "NONE",
            evidence=f"n={live_n} · expectancy_r={overall.get('expectancy_r', 'n/a')}",
        ),
        _item(
            key="book_correlation",
            label="Book concentration lens",
            status="READY" if book_ok else ("EMPTY_BOOK" if n_pos == 0 else "SINGLE_NAME"),
            why_it_matters="Five correlated names can be one bet. See positions vs independent bets.",
            next_action="Open ≥2 positions or inspect /api/book-correlation" if not book_ok else "NONE",
            evidence=f"{n_pos} positions → {n_bets} effective bets",
        ),
        _item(
            key="options_eod",
            label="Options OI / IV history",
            status="READY" if opts_ok else "MISSING",
            why_it_matters="One live chain is a snapshot; multi-day PCR/IV needs a saved EOD store.",
            next_action="python main.py options-eod" if not opts_ok else "NONE",
            evidence=f"{int(options_eod.get('snapshots', 0) or 0)} snapshots",
        ),
    ]
    missing = [i for i in items if i["status"] not in {"READY", "EMPTY_BOOK", "SINGLE_NAME"}]
    return {
        "schema_version": 1,
        "summary": (
            "Research inputs look complete enough for careful cash research."
            if not missing
            else f"{len(missing)} research gap(s) still limit how much edge you can trust."
        ),
        "ready_count": sum(1 for i in items if i["status"] == "READY"),
        "gap_count": len(missing),
        "items": items,
        "gaps": missing,
    }
