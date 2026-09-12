"""One report per session that says what the desk actually did.

Unattended operation is only trustworthy if it produces evidence nobody had to
ask for. This assembles that evidence from durable state — provenance files,
coverage ledgers, the paper book, the interlock — and it is built around one
rule: every section must be able to say it does not know.

That rule is the whole design. A report that renders a missing scan as zero
symbols scanned, or a missing regime as a neutral one, is worse than no report,
because it converts an operational failure into a calm-looking number. So each
section carries its own availability, and an absent input produces UNKNOWN
rather than a plausible default.

The report never computes market truth. It reads what the running system
recorded and repeats it, so a disagreement between this and the desk is a
defect in one of them rather than a third opinion.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from core.runtime_paths import logs_dir, logs_path

SCHEMA_VERSION = 1

UNKNOWN = "UNKNOWN"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def production_sha() -> str:
    """The SHA this desk is running. Empty when it cannot be established."""
    override = os.environ.get("QT_BUILD_SHA", "").strip()
    if override:
        return override
    try:
        from core.runtime_paths import REPO_ROOT

        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except Exception:
        return ""


def _system_section() -> dict[str, Any]:
    try:
        from product.runtime_lifecycle import inspect_runtime

        runtime = inspect_runtime(api_serving=False)
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"[:160]}

    components = list(runtime.get("components") or [])
    degraded = [
        {"name": c.get("name"), "status": c.get("status"), "detail": c.get("detail")}
        for c in components
        if str(c.get("status")) not in ("READY",)
    ]
    ops = _read_json(logs_path("market_ops", "runtime.json"), {}) or {}
    return {
        "available": True,
        "lifecycle": runtime.get("lifecycle"),
        "operational_ready": runtime.get("operational_ready"),
        "evidence_ready": runtime.get("evidence_ready"),
        "reasons": list(runtime.get("reasons") or []),
        "components": [
            {"name": c.get("name"), "status": c.get("status"), "detail": c.get("detail")}
            for c in components
        ],
        "degraded_components": degraded,
        "worker_started_at": ops.get("started_at") or "",
        "worker_heartbeat": ops.get("heartbeat") or ops.get("heartbeat_at") or "",
        "recoveries": ops.get("recoveries"),
        "crashes": ops.get("crashes"),
    }


def _provenance_rows() -> list[dict[str, Any]]:
    directory = logs_dir() / "provenance"
    rows: list[dict[str, Any]] = []
    try:
        files = sorted(directory.glob("*.json"))
    except Exception:
        return rows
    for path in files:
        payload = _read_json(path)
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def _data_section() -> dict[str, Any]:
    """Read straight off the acquisition contract's provenance."""
    rows = _provenance_rows()
    if not rows:
        return {
            "available": False,
            "reason": "no acquisition has recorded provenance yet",
            "datasets": [],
        }
    datasets = []
    fresh, stale, failed, fallback = [], [], [], []
    for row in rows:
        name = str(row.get("dataset") or "")
        state = str(row.get("state") or UNKNOWN)
        level = int(row.get("fallback_level") or 0)
        entry = {
            "dataset": name,
            "state": state,
            "selected_source": row.get("source") or "",
            "tier": row.get("tier"),
            "fallback_level": level,
            "record_count": row.get("record_count"),
            "fetched_at": row.get("fetched_at") or "",
            "content_hash": row.get("content_hash") or "",
            "parser_changed_somewhere": bool(row.get("parser_changed_somewhere")),
            "attempts": [
                {"source": a.get("source"), "outcome": a.get("outcome"),
                 "detail": str(a.get("detail") or "")[:160]}
                for a in (row.get("attempts") or [])
            ],
        }
        datasets.append(entry)
        if state == "ACQUIRED":
            fresh.append(name)
        elif state == "LAST_KNOWN_GOOD":
            stale.append(name)
        else:
            failed.append(name)
        if level > 0:
            fallback.append(name)
    return {
        "available": True,
        "datasets": datasets,
        "fresh_sources": sorted(set(fresh)),
        "stale_sources": sorted(set(stale)),
        "unresolved_source_failures": sorted(set(failed)),
        "used_a_fallback": sorted(set(fallback)),
        "parser_changes_seen": sorted(
            {d["dataset"] for d in datasets if d["parser_changed_somewhere"]}
        ),
    }


def _market_section() -> dict[str, Any]:
    try:
        from product.market_view import peek_cached_market_view

        view = peek_cached_market_view()
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"[:160]}
    if view is None:
        return {"available": False, "reason": "no market view has been assembled",
                "regime": UNKNOWN}
    health = str(getattr(view, "health", "") or "")
    if not health or health == "Unavailable":
        return {"available": False, "reason": "market view is unavailable",
                "regime": UNKNOWN}
    return {
        "available": True,
        "regime": health.upper(),
        "breadth": getattr(view, "breadth", ""),
        "leaders": list(getattr(view, "leaders", []) or [])[:5],
        "laggards": list(getattr(view, "laggards", []) or [])[:5],
        "summary": getattr(view, "summary", ""),
    }


def _scan_section() -> dict[str, Any]:
    audit = _read_json(logs_path("scan_coverage_audit.json"), {}) or {}
    summary = dict(audit.get("summary") or {})
    if not summary:
        try:
            from scan.scan_coverage import load_audit

            summary = dict((load_audit() or {}).get("summary") or {})
        except Exception:
            summary = {}
    if not summary:
        return {"available": False, "reason": "no scan coverage audit on disk",
                "intended_universe": None, "analysed": None, "coverage_pct": None}
    return {
        "available": True,
        "state": summary.get("state"),
        "intended_universe": summary.get("requested"),
        "analysed": summary.get("checked"),
        "qualified": summary.get("qualified"),
        "skipped": summary.get("not_observed"),
        "data_unavailable": summary.get("data_unavailable"),
        "analysis_errors": summary.get("analysis_errors"),
        "coverage_pct": summary.get("coverage_pct"),
        "history_coverage_pct": summary.get("history_coverage_pct"),
        "skip_reasons": dict(summary.get("reason_counts") or {}),
        "history_repair": dict(summary.get("history_repair") or {}),
        "universe_provenance": dict(summary.get("universe_provenance") or {}),
    }


def _decisions_section() -> dict[str, Any]:
    try:
        from product.decision_service import decision_board

        board = decision_board()
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"[:160]}
    if not board.get("available"):
        return {
            "available": False,
            "state": board.get("state"),
            "reason": board.get("reason"),
            "counts": {},
        }
    counts = dict(board.get("counts") or {})
    return {
        "available": True,
        "state": board.get("state"),
        "scan_scanned_at": board.get("scan_scanned_at"),
        "counts": counts,
        "actionable": board.get("actionable", 0),
        "no_trade": counts.get("NO_TRADE", 0) or (0 if board.get("actionable") else 1),
        "evidence_gaps": dict(board.get("evidence_gaps") or {}),
        "top_rejection_reasons": _rejection_reasons(),
    }


def _rejection_reasons() -> list[dict[str, Any]]:
    """Why candidates were turned down, counted. Absent reads as absent."""
    ledger = _read_json(logs_path("product", "rejection_reasons.json"), {}) or {}
    counts = dict(ledger.get("counts") or {})
    return [
        {"reason": reason, "count": int(count)}
        for reason, count in sorted(counts.items(), key=lambda kv: -int(kv[1]))
    ][:10]


def _paper_section() -> dict[str, Any]:
    book = _read_json(logs_path("intelligence", "intel_book.json"), {}) or {}
    if not book:
        return {"available": False, "reason": "no paper book on disk",
                "open": 0, "closed": 0}
    opens = list(book.get("open") or [])
    closed = list(book.get("closed") or [])
    return {
        "available": True,
        "currently_open": len(opens),
        "closed_total": len(closed),
        "unresolved": len(opens),
        "attributable_open": sum(1 for p in opens if p.get("decision_id")),
        "unattributable_open": sum(1 for p in opens if not p.get("decision_id")),
        "capital": book.get("capital"),
    }


def _forward_evidence_section() -> dict[str, Any]:
    try:
        from product.forward_evidence_board import build_forward_evidence_board

        book = _read_json(logs_path("intelligence", "intel_book.json"), {}) or {}
        board = build_forward_evidence_board(book=book)
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"[:160]}

    section = {
        "available": True,
        "state": board.get("state"),
        "settled_sample": board.get("settled_trades", 0),
        "headline": board.get("headline"),
        "cells_usable_for_ranking": board.get("cells_usable_for_ranking", 0),
    }
    # Statistically meaningful metrics ONLY. Below the floor these numbers
    # exist but mean nothing, and printing them invites someone to read them.
    if board.get("cells_usable_for_ranking"):
        section["by_setup"] = [
            row for row in board.get("by_setup", []) if row.get("usable_for_ranking")
        ]
        section["risk_shape"] = board.get("risk_shape")
        section["r_distribution"] = board.get("r_distribution")
    else:
        section["metrics_withheld"] = (
            "no context has reached the sample floor; per-setup statistics "
            "would be noise presented as measurement"
        )
    return section


def _capital_safety_section() -> dict[str, Any]:
    try:
        from product.live_execution_interlock import get_live_execution_state

        state = get_live_execution_state()
        return {
            "available": True,
            "live_locked": bool(state.locked),
            "authorized": bool(state.authorized),
            "verified": bool(state.verified),
            "status": state.status,
            "reason": state.reason,
            "broker_mutations": 0,
            "source": state.source,
        }
    except Exception as exc:
        # Fail closed in the report exactly as the interlock fails closed.
        return {
            "available": False,
            "live_locked": True,
            "broker_mutations": 0,
            "reason": f"interlock unreadable: {type(exc).__name__}: {exc}"[:160],
        }


def _silent_wrongness_section() -> dict[str, Any]:
    """Assert against plausible-looking wrong states, now, on live state.

    Not a ledger someone remembers to update: the checks run every time the
    report is built, so a state that has quietly gone wrong since the last
    session shows up in this session's report.
    """
    try:
        from product.silent_wrongness_checks import run_silent_wrongness_checks

        result = run_silent_wrongness_checks()
    except Exception as exc:
        return {
            "available": False,
            "reason": f"checks could not run: {type(exc).__name__}: {exc}"[:160],
            "discovered": None,
        }
    return {
        "available": True,
        "checks_run": result["checks_run"],
        "clean": result["clean"],
        "discovered": len(result["findings"]),
        "critical": result["critical"],
        "warnings": result["warnings"],
        "unknown": result["unknown"],
        "findings": result["findings"],
    }


def build_daily_operating_report() -> dict[str, Any]:
    """Assemble the session report. Reads state; never runs a scan."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "production_sha": production_sha(),
        "system": _system_section(),
        "data": _data_section(),
        "market": _market_section(),
        "scan": _scan_section(),
        "decisions": _decisions_section(),
        "paper": _paper_section(),
        "forward_evidence": _forward_evidence_section(),
        "capital_safety": _capital_safety_section(),
        "silent_wrongness": _silent_wrongness_section(),
    }


def unmet_operating_proofs(report: Mapping[str, Any]) -> list[str]:
    """Which daily proofs this report cannot evidence.

    Returned rather than raised: a session with unmet proofs is a session to
    look at, not a crash. An empty list means every proof was evidenced.
    """
    missing: list[str] = []
    if not report.get("production_sha"):
        missing.append("production SHA could not be established")
    system = report.get("system") or {}
    if not system.get("available"):
        missing.append("runtime state unavailable")
    elif system.get("degraded_components"):
        names = ", ".join(str(c.get("name")) for c in system["degraded_components"])
        missing.append(f"components not READY: {names}")
    if not (report.get("data") or {}).get("available"):
        missing.append("no acquisition provenance recorded")
    if not (report.get("market") or {}).get("available"):
        missing.append("market regime not populated")
    if not (report.get("scan") or {}).get("available"):
        missing.append("no scan coverage evidence")
    if not (report.get("decisions") or {}).get("available"):
        missing.append("no decisions published")
    capital = report.get("capital_safety") or {}
    if not capital.get("live_locked"):
        missing.append("LIVE LOCK NOT CONFIRMED")
    if int(capital.get("broker_mutations") or 0) != 0:
        missing.append("broker mutations recorded")
    findings = report.get("silent_wrongness") or {}
    if not findings.get("available"):
        missing.append("silent-wrongness checks did not run")
    elif findings.get("critical"):
        missing.append(f"{findings['critical']} critical silent-wrongness finding(s)")
    return missing


def render_text(report: Mapping[str, Any]) -> str:
    """The concise operator-facing form."""
    lines: list[str] = []
    add = lines.append
    add(f"PRODUCTION_SHA {report.get('production_sha') or UNKNOWN}")
    add(f"generated      {report.get('generated_at')}")

    system = report.get("system") or {}
    add("")
    add("SYSTEM")
    if not system.get("available"):
        add(f"  unavailable — {system.get('reason')}")
    else:
        add(f"  lifecycle {system.get('lifecycle')} · operational_ready "
            f"{system.get('operational_ready')} · evidence_ready {system.get('evidence_ready')}")
        for component in system.get("degraded_components") or []:
            add(f"  degraded: {component['name']} [{component['status']}] {component['detail']}")
        if not system.get("degraded_components"):
            add("  every component READY")

    data = report.get("data") or {}
    add("")
    add("DATA")
    if not data.get("available"):
        add(f"  {data.get('reason')}")
    else:
        add(f"  fresh {len(data['fresh_sources'])} · stale {len(data['stale_sources'])} "
            f"· failed {len(data['unresolved_source_failures'])} "
            f"· used a fallback {len(data['used_a_fallback'])}")
        for row in data["datasets"]:
            add(f"  {row['dataset']}: {row['state']} via {row['selected_source'] or '—'} "
                f"(level {row['fallback_level']}, n={row['record_count']})")
        if data["parser_changes_seen"]:
            add(f"  PARSER CHANGED: {', '.join(data['parser_changes_seen'])}")

    market = report.get("market") or {}
    add("")
    add(f"MARKET  regime {market.get('regime') or UNKNOWN}"
        + (f" · breadth {market.get('breadth')}" if market.get("available") else ""))

    scan = report.get("scan") or {}
    add("")
    add("SCAN")
    if not scan.get("available"):
        add(f"  {scan.get('reason')}")
    else:
        add(f"  intended {scan.get('intended_universe')} · analysed {scan.get('analysed')} "
            f"· coverage {scan.get('coverage_pct')}%")
        provenance = scan.get("universe_provenance") or {}
        if provenance:
            add(f"  universe from {provenance.get('source') or UNKNOWN} "
                f"[{provenance.get('state') or UNKNOWN}, level {provenance.get('fallback_level')}]")

    decisions = report.get("decisions") or {}
    add("")
    add("DECISIONS")
    if not decisions.get("available"):
        add(f"  {decisions.get('state')} — {decisions.get('reason')}")
    else:
        counts = decisions.get("counts") or {}
        add("  " + " · ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "  none")
        for row in decisions.get("top_rejection_reasons") or []:
            add(f"  rejected {row['reason']}: {row['count']}")

    paper = report.get("paper") or {}
    add("")
    if paper.get("available"):
        add(f"PAPER   open {paper['currently_open']} · closed {paper['closed_total']} "
            f"· unattributable open {paper['unattributable_open']}")
    else:
        add(f"PAPER   {paper.get('reason')}")

    evidence = report.get("forward_evidence") or {}
    add("")
    add(f"FORWARD EVIDENCE  {evidence.get('state') or UNKNOWN} "
        f"· settled {evidence.get('settled_sample')}")
    if evidence.get("metrics_withheld"):
        add(f"  {evidence['metrics_withheld']}")

    capital = report.get("capital_safety") or {}
    add("")
    add(f"CAPITAL SAFETY  LIVE_LOCKED={str(bool(capital.get('live_locked'))).upper()} "
        f"· BROKER_MUTATIONS={int(capital.get('broker_mutations') or 0)}")

    findings = report.get("silent_wrongness") or {}
    add("")
    if not findings.get("available"):
        add(f"SILENT WRONGNESS  {findings.get('reason')}")
    else:
        add(f"SILENT WRONGNESS  {findings['checks_run']} checks · "
            f"critical {findings['critical']} · warnings {findings['warnings']} "
            f"· unknown {findings['unknown']}")
        for row in findings.get("findings") or []:
            add(f"  [{row['severity']}] {row['check']}: {row['summary']}")

    unmet = unmet_operating_proofs(report)
    add("")
    if unmet:
        add("UNMET OPERATING PROOFS")
        for item in unmet:
            add(f"  - {item}")
    else:
        add("UNMET OPERATING PROOFS  none")
    return "\n".join(lines)


def write_daily_operating_report(report: Mapping[str, Any] | None = None) -> Path:
    payload = dict(report or build_daily_operating_report())
    day = str(payload.get("generated_at") or _now())[:10]
    target = logs_path("product", "operating_reports", f"{day}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return target


def main(argv: list[str] | None = None) -> int:
    """`python -m product.daily_operating_report` — the session report.

    Exit code is the operator-facing signal, so a scheduler can act on it:

        0  every daily proof was evidenced
        1  the report was produced and some proofs were unmet
        2  a critical silent-wrongness finding, or the live lock unconfirmed

    A non-zero exit is not a crash; it means read the report. The report is
    always written, because a session that went badly is exactly the one whose
    evidence must survive.
    """
    import argparse

    parser = argparse.ArgumentParser(description="QuantTerm daily operating report")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    parser.add_argument("--no-write", action="store_true", help="do not persist the report")
    args = parser.parse_args(argv)

    report = build_daily_operating_report()
    if not args.no_write:
        try:
            write_daily_operating_report(report)
        except Exception as exc:  # persisting must never lose the report
            print(f"# could not persist the report: {type(exc).__name__}: {exc}")

    print(json.dumps(report, indent=2, default=str) if args.json else render_text(report))

    capital = report.get("capital_safety") or {}
    findings = report.get("silent_wrongness") or {}
    if not capital.get("live_locked") or int(capital.get("broker_mutations") or 0):
        return 2
    if findings.get("available") and findings.get("critical"):
        return 2
    return 1 if unmet_operating_proofs(report) else 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
