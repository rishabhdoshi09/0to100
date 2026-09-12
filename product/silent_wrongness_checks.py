"""Assertions against plausible-looking wrong states, run on live state.

Crashes announce themselves. The dangerous failures on a trading desk are the
ones that look fine: a scan that reports complete over a third of the universe,
an artifact that says fresh while its source failed, fallback data wearing the
label of primary, a green dot over a sentence describing an outage.

Every check here was written because that shape of wrongness either happened in
this system or is one step away from it. They run against whatever the desk has
on disk right now, so they are an operating instrument rather than a test — the
tests pin the checks themselves.

A check returns a finding or nothing. Nothing means the state passed, not that
the check could not run: a check that cannot evaluate says so as a finding with
severity UNKNOWN, because a silent detector is the thing being guarded against.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from core.runtime_paths import logs_dir, logs_path

SCHEMA_VERSION = 1

CRITICAL = "CRITICAL"
WARNING = "WARNING"
UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class Finding:
    check: str
    severity: str
    summary: str
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


# ── individual checks ──────────────────────────────────────────────────────
def check_scan_complete_over_thin_coverage() -> Finding | None:
    """A scan calling itself FULL while a slice of the universe went unseen."""
    audit = _read(logs_path("scan_coverage_audit.json"), {}) or {}
    summary = dict(audit.get("summary") or {})
    if not summary:
        return None
    state = str(summary.get("state") or "")
    coverage = summary.get("coverage_pct")
    if state == "FULL" and coverage is not None and float(coverage) < 95.0:
        return Finding(
            "scan_complete_over_thin_coverage", CRITICAL,
            "the scan reports FULL coverage while a material slice went unseen",
            f"state=FULL but coverage_pct={coverage}",
            {"summary": {k: summary.get(k) for k in
                         ("state", "requested", "checked", "coverage_pct")}},
        )
    return None


def check_universe_claimed_without_provenance() -> Finding | None:
    """A market-wide claim standing on a universe nobody can trace."""
    audit = _read(logs_path("scan_coverage_audit.json"), {}) or {}
    summary = dict(audit.get("summary") or {})
    if not summary:
        return None
    provenance = dict(summary.get("universe_provenance") or {})
    if not provenance or provenance.get("state") == "UNRECORDED":
        return Finding(
            "universe_claimed_without_provenance", WARNING,
            "a scan claims a universe whose source was never recorded",
            "coverage summary has no usable universe_provenance",
            {"requested": summary.get("requested")},
        )
    return None


def check_fallback_data_presented_as_primary() -> Finding | None:
    """Stale or fallback acquisitions must never read as fresh primary."""
    offenders = []
    directory = logs_dir() / "provenance"
    try:
        files = sorted(directory.glob("*.json"))
    except Exception:
        return None
    for path in files:
        row = _read(path, {}) or {}
        state = str(row.get("state") or "")
        level = int(row.get("fallback_level") or 0)
        if state == "ACQUIRED" and row.get("tier") == 8:
            offenders.append({"dataset": row.get("dataset"), "source": row.get("source"),
                              "tier": row.get("tier"), "state": state})
        elif level > 0 and state not in ("ACQUIRED", "LAST_KNOWN_GOOD",
                                         "DATA_UNAVAILABLE", "SOURCE_CONFLICT"):
            offenders.append({"dataset": row.get("dataset"), "state": state,
                              "fallback_level": level})
    if offenders:
        return Finding(
            "fallback_data_presented_as_primary", CRITICAL,
            "an acquisition from a cache tier is labelled as freshly acquired",
            "a last-known-good tier must report LAST_KNOWN_GOOD",
            {"datasets": offenders[:5]},
        )
    return None


def check_zero_row_successful_acquisition() -> Finding | None:
    """A source that returned nothing must never be recorded as a success."""
    offenders = []
    try:
        files = sorted((logs_dir() / "provenance").glob("*.json"))
    except Exception:
        return None
    for path in files:
        row = _read(path, {}) or {}
        if str(row.get("state")) in ("ACQUIRED", "LAST_KNOWN_GOOD"):
            count = row.get("record_count")
            if count is not None and int(count) == 0:
                offenders.append({"dataset": row.get("dataset"),
                                  "source": row.get("source")})
    if offenders:
        return Finding(
            "zero_row_successful_acquisition", CRITICAL,
            "an acquisition succeeded with zero records",
            "zero rows is a claim about the world and needs a source to survive",
            {"datasets": offenders[:5]},
        )
    return None


def check_health_lane_contradictions() -> Finding | None:
    """A lane whose own detail describes an outage may not read HEALTHY."""
    try:
        from product.system_health_contract import (
            build_system_health_contract,
            detail_contradicts_healthy,
        )

        contract = build_system_health_contract()
    except Exception as exc:
        return Finding(
            "health_lane_contradictions", UNKNOWN,
            "the health contract could not be evaluated",
            f"{type(exc).__name__}: {exc}"[:160],
        )
    offenders = [
        {"lane": lane.get("key"), "detail": lane.get("detail")}
        for lane in contract.get("lanes", [])
        if lane.get("status") == "HEALTHY"
        and detail_contradicts_healthy(str(lane.get("detail") or ""))
    ]
    if offenders:
        return Finding(
            "health_lane_contradictions", CRITICAL,
            "a health lane reads HEALTHY over a detail describing a problem",
            "the operator reads the dot, not the sentence",
            {"lanes": offenders},
        )
    return None


def check_component_contradictions() -> Finding | None:
    """The same rule for runtime components, which are a separate surface."""
    try:
        from product.runtime_lifecycle import inspect_runtime
        from product.system_health_contract import detail_contradicts_healthy

        runtime = inspect_runtime(api_serving=False)
    except Exception as exc:
        return Finding(
            "component_contradictions", UNKNOWN,
            "runtime components could not be evaluated",
            f"{type(exc).__name__}: {exc}"[:160],
        )
    offenders = [
        {"component": c.get("name"), "detail": c.get("detail")}
        for c in runtime.get("components", [])
        if c.get("status") == "READY"
        and detail_contradicts_healthy(str(c.get("detail") or ""))
    ]
    if offenders:
        return Finding(
            "component_contradictions", CRITICAL,
            "a runtime component reads READY over a detail describing a problem",
            "liveness is not capability",
            {"components": offenders},
        )
    return None


def check_non_market_evidence_in_paper_cells() -> Finding | None:
    """Replay or fixture rows must never sit in a PAPER_FORWARD cell."""
    try:
        from product.conditional_evidence import load
        from product.evidence_class import PAPER_FORWARD

        store = load()
    except Exception as exc:
        return Finding(
            "non_market_evidence_in_paper_cells", UNKNOWN,
            "the conditional evidence store could not be read",
            f"{type(exc).__name__}: {exc}"[:160],
        )
    offenders = []
    for key, cell in (store.get("cells") or {}).items():
        if not str(key).startswith(f"{PAPER_FORWARD}::"):
            continue
        recorded = str((cell or {}).get("evidence_class") or "")
        if recorded != PAPER_FORWARD:
            offenders.append({"cell": key, "evidence_class": recorded})
    if offenders:
        return Finding(
            "non_market_evidence_in_paper_cells", CRITICAL,
            "a paper-forward cell holds a row from another evidence class",
            "replay proves plumbing; it may never become market evidence",
            {"cells": offenders[:5]},
        )
    return None


def check_unattributable_open_positions() -> Finding | None:
    """Open paper positions that cannot name the decision that opened them."""
    book = _read(logs_path("intelligence", "intel_book.json"), {}) or {}
    if not book:
        return None
    orphans = [p for p in (book.get("open") or []) if not p.get("decision_id")]
    if orphans:
        return Finding(
            "unattributable_open_positions", WARNING,
            "open paper positions cannot name the decision that opened them",
            "their outcomes will be refused as evidence when they settle",
            {"count": len(orphans),
             "symbols": [p.get("symbol") for p in orphans[:8]]},
        )
    return None


def check_decisions_published_without_a_scan() -> Finding | None:
    """A recommendation surface must never outlive the scan behind it."""
    try:
        from product.recommendations_store import load_recommendations

        saved = load_recommendations()
    except Exception:
        return None
    if not saved:
        return None
    if not str(saved.get("scan_scanned_at") or "").strip():
        return Finding(
            "decisions_published_without_a_scan", CRITICAL,
            "recommendations exist with no scan timestamp behind them",
            "a projection file is not evidence that the market was looked at",
            {"categories": len(saved.get("categories") or [])},
        )
    return None


CHECKS: tuple[Callable[[], Finding | None], ...] = (
    check_scan_complete_over_thin_coverage,
    check_universe_claimed_without_provenance,
    check_fallback_data_presented_as_primary,
    check_zero_row_successful_acquisition,
    check_health_lane_contradictions,
    check_component_contradictions,
    check_non_market_evidence_in_paper_cells,
    check_unattributable_open_positions,
    check_decisions_published_without_a_scan,
)


def run_silent_wrongness_checks() -> dict[str, Any]:
    """Run every check against current state. A check that raises is a finding."""
    findings: list[dict[str, Any]] = []
    for check in CHECKS:
        try:
            result = check()
        except Exception as exc:
            result = Finding(
                getattr(check, "__name__", "unknown_check"), UNKNOWN,
                "the check itself failed",
                f"{type(exc).__name__}: {exc}"[:160],
            )
        if result is not None:
            findings.append(result.to_dict())
    return {
        "schema_version": SCHEMA_VERSION,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "checks_run": len(CHECKS),
        "findings": findings,
        "critical": sum(1 for f in findings if f["severity"] == CRITICAL),
        "warnings": sum(1 for f in findings if f["severity"] == WARNING),
        "unknown": sum(1 for f in findings if f["severity"] == UNKNOWN),
        "clean": not findings,
    }
