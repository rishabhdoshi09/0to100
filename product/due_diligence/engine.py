"""Assemble an evidence-backed due-diligence report for one scanner candidate."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from product.due_diligence.classify import classify_company
from product.due_diligence.coverage import availability_state_for_kpi, inspect_research_coverage
from product.due_diligence.dashboard import (
    cache_schedule,
    company_snapshot,
    confirmation_label,
    first_screen,
)
from product.due_diligence.evidence import (
    confirmation_from_evidence,
    critical_metrics_missing,
    decision_coverage,
    missing_evidence,
    score_evidence,
    sector_kpi_verdict,
)
from product.due_diligence.extract import extract_guidance, extract_kpis_from_raw, merge_kpi_maps, _in_bounds
from product.due_diligence.frameworks import KpiSpec, get_framework
from product.due_diligence.news_layer import material_events, news_verdict
from product.due_diligence.peers import rank_peers
from product.due_diligence.provenance import conflict_record, material_disagreement, provenance
from product.due_diligence.quality_rules import balance_sheet_quality, cash_flow_quality, growth_quality
from product.due_diligence.red_flags import collect_red_flags, partition_flags
from product.due_diligence.score_breakdown import score_breakdown
from product.due_diligence.series import dated_series, direction, find_row, snapshot
from product.due_diligence.thesis import compose_thesis
from product.due_diligence.wiring import apply_autonomy_pack, load_evidence_pack

ROOT = Path(__file__).resolve().parents[2]


def _f(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        number = float(value)
        return number if number == number else None
    except (TypeError, ValueError):
        return None


def _find(payload: Mapping[str, Any] | None, symbol: str) -> dict[str, Any]:
    for row in list((payload or {}).get("records") or []):
        if isinstance(row, Mapping) and str(row.get("symbol") or "").upper() == symbol:
            return dict(row)
    return {}


def _quality_label(score: int | None, coverage: float, *, min_coverage: float = 0.40) -> str:
    if score is None or coverage < min_coverage:
        return "Unmeasured"
    if score >= 80:
        return "Strong"
    if score >= 60:
        return "Adequate"
    if score >= 40:
        return "Mixed"
    return "Weak"


def _pillar_label(directions: Sequence[str]) -> str:
    useful = [d for d in directions if d != "unknown"]
    if not useful:
        return "Unmeasured"
    if all(d == "improving" for d in useful):
        return "Improving"
    if all(d == "deteriorating" for d in useful):
        return "Deteriorating"
    if useful.count("deteriorating") > useful.count("improving"):
        return "Weakening"
    if useful.count("improving") > 0:
        return "Improving"
    return "Stable"


def _formula(spec: KpiSpec, snap: Mapping[str, Any]) -> str:
    current = snap.get("current")
    if current is None:
        return "Calculation not possible"
    if spec.kind == "rate":
        if snap.get("year_ago") is not None and snap.get("yoy_change") is not None:
            return f"{current} − {snap['year_ago']} = {snap['yoy_change']:+.2f} pt (YoY)"
        if snap.get("previous") is not None and snap.get("qoq_change") is not None:
            return f"{current} − {snap['previous']} = {snap['qoq_change']:+.2f} pt (QoQ)"
        return "Only the latest print is on file — no comparable period."
    year = snap.get("year_ago")
    if year not in (None, 0) and snap.get("yoy_change") is not None:
        return f"({current} − {year}) / {year} = {snap['yoy_change']:+.1f}% YoY"
    prev = snap.get("previous")
    if prev not in (None, 0) and snap.get("qoq_change") is not None:
        return f"({current} − {prev}) / {prev} = {snap['qoq_change']:+.1f}% QoQ"
    return "Only the latest print is on file — no comparable period."


def _kpi_points(spec: KpiSpec, snap: Mapping[str, Any], trend: str) -> float | None:
    if snap.get("current") is None:
        return None
    if trend == "improving":
        return 88.0
    if trend == "stable":
        return 62.0
    if trend == "deteriorating":
        return 28.0
    return 55.0  # current exists, no comparable period


def _fact_line(spec: KpiSpec, snap: Mapping[str, Any]) -> str:
    current = snap.get("current")
    if current is None:
        return "Data unavailable"
    period = snap.get("current_period") or "latest period"
    unit = spec.unit
    bits = [f"{spec.label}: {current} {unit} ({period})"]
    basis = str(snap.get("reporting_basis") or "")
    period_type = str(snap.get("period_type") or "")
    extra = []
    if basis:
        extra.append(basis)
    if period_type and period_type not in {"unknown", ""}:
        extra.append(period_type)
    if extra:
        bits[0] += f" [{' · '.join(extra)}]"
    prev = snap.get("previous")
    if prev is not None and snap.get("previous_period"):
        bits.append(f"previous {snap['previous_period']}: {prev} {unit}")
    year = snap.get("year_ago")
    if year is not None and snap.get("year_ago_period"):
        bits.append(f"year-ago {snap['year_ago_period']}: {year} {unit}")
    return "; ".join(bits)


def _interpretation(spec: KpiSpec, snap: Mapping[str, Any], trend: str) -> str:
    if snap.get("current") is None:
        return "Data unavailable"
    if trend == "unknown":
        return f"Only the latest {spec.label.lower()} print is on file — no comparable period."
    yoy = snap.get("yoy_change")
    qoq = snap.get("qoq_change")
    move = yoy if yoy is not None else qoq
    horizon = "YoY" if yoy is not None else "QoQ"
    if spec.kind == "rate":
        verb = "improved" if trend == "improving" else "worsened" if trend == "deteriorating" else "was little changed"
        return f"{spec.label} {verb} {horizon} ({move:+.2f} pt)."
    verb = "grew" if (move or 0) > 0 else "fell" if (move or 0) < 0 else "was unchanged"
    return f"{spec.label} {verb} {horizon} ({move:+.1f}%). Trend is {trend}."


def _implication(trend: str, vs_setup: str) -> str:
    if trend == "improving":
        return "This strengthens the fundamental case for the technical setup." if vs_setup != "UNMEASURED" else "This is a constructive business trend."
    if trend == "deteriorating":
        return "This reduces conviction in the technical setup until the series turns."
    if trend == "unknown":
        return "No trend implication — the series is too short."
    return "This leaves conviction unchanged on this metric."


def _apply_overlay(
    findings: list[dict[str, Any]],
    specs: Sequence[KpiSpec],
    measured: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Fill Data unavailable KPIs from extra tables, key-ratios or downloaded filings.

    If a second source disagrees with an already-filled print, keep the first and log a conflict.
    """
    spec_by_id = {spec.id: spec for spec in specs}
    conflicts: list[dict[str, Any]] = []
    for finding in findings:
        spec = spec_by_id.get(str(finding.get("id")))
        snap = dict(measured.get(str(finding.get("id"))) or {})
        if spec is None or snap.get("current") is None:
            continue
        if finding.get("available"):
            existing = (finding.get("snapshot") or {}).get("current")
            if material_disagreement(existing, snap.get("current"), kind=spec.kind):
                preferred = dict(finding.get("provenance") or {})
                other = provenance(
                    value=snap.get("current"),
                    period=str(snap.get("current_period") or ""),
                    source=str(snap.get("source") or "Secondary source"),
                    source_url=str(snap.get("source_url") or ""),
                    confidence="medium",
                    raw_reference=str(snap.get("raw_reference") or spec.table),
                )
                conflicts.append(conflict_record(str(finding.get("id")), preferred, other))
                finding["source_consensus"] = "conflict"
                finding["conflicting_sources"] = list(finding.get("conflicting_sources") or []) + [{
                    "value": snap.get("current"),
                    "source": str(snap.get("source") or "Secondary source"),
                    "period": snap.get("current_period"),
                }]
            else:
                agreeing = list(finding.get("agreeing_sources") or [])
                extra = str(snap.get("source") or "")
                if extra and extra not in agreeing:
                    agreeing.append(extra)
                    finding["agreeing_sources"] = agreeing
                    finding["source_count"] = len(agreeing)
                    finding["source_consensus"] = "confirmed" if len(agreeing) >= 2 else "single"
            continue
        trend = direction(
            higher_is_better=spec.higher_is_better,
            qoq=snap.get("qoq_change"),
            yoy=snap.get("yoy_change"),
            current_period_type=str(snap.get("period_type") or ""),
            compare_period_type=str(snap.get("year_ago_period_type") or snap.get("previous_period_type") or ""),
        )
        source = str(snap.get("source") or "Downloaded filing / extra table")
        finding.update({
            "available": True,
            "trend": trend,
            "points": _kpi_points(spec, snap, trend),
            "snapshot": {
                "current": snap.get("current"),
                "current_period": snap.get("current_period") or "",
                "previous": snap.get("previous"),
                "previous_period": snap.get("previous_period") or "",
                "year_ago": snap.get("year_ago"),
                "year_ago_period": snap.get("year_ago_period") or "",
                "qoq_change": snap.get("qoq_change"),
                "yoy_change": snap.get("yoy_change"),
                "points": list(snap.get("points") or []),
            },
            "fact": _fact_line(spec, snap),
            "interpretation": _interpretation(spec, snap, trend),
            "implication": _implication(trend, "pending"),
            "formula": _formula(spec, snap),
            "source": source,
            "source_url": str(snap.get("source_url") or finding.get("source_url") or ""),
            "source_date": str(snap.get("source_date") or snap.get("current_period") or finding.get("source_date") or ""),
            "confidence": "medium",
            "period_type": snap.get("period_type") or finding.get("period_type") or "",
            "reporting_basis": snap.get("reporting_basis") or finding.get("reporting_basis") or "",
            "source_count": int(snap.get("source_count") or 1),
            "source_consensus": str(snap.get("source_consensus") or "single"),
            "agreeing_sources": list(snap.get("agreeing_sources") or ([source] if source else [])),
            "provenance": provenance(
                value=snap.get("current"),
                period=str(snap.get("current_period") or ""),
                source=source,
                source_url=str(snap.get("source_url") or finding.get("source_url") or ""),
                retrieved_at=str(finding.get("retrieved_at") or ""),
                published_at=str(snap.get("current_period") or ""),
                confidence="medium",
                raw_reference=str(snap.get("raw_reference") or spec.table),
            ),
        })
    return conflicts


def _evaluate_kpis(raw: Mapping[str, Any], specs: Sequence[KpiSpec], source_url: str, fetched_at: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    tables = {
        "quarterly_results": list(raw.get("quarterly_results") or []),
        "profit_loss": list(raw.get("profit_loss") or []),
        "cash_flow": list(raw.get("cash_flow") or []),
        "shareholding": list(raw.get("shareholding") or []),
        "balance_sheet": list(raw.get("balance_sheet") or []),
        "key_ratios": list(raw.get("key_ratios") or []),
    }
    for spec in specs:
        row = find_row(tables.get(spec.table), spec.needles)
        series = dated_series(row)
        if not series:
            for other in tables.values():
                row = find_row(other, spec.needles)
                series = dated_series(row)
                if series:
                    break
        year_steps = 1 if spec.table in {"profit_loss", "balance_sheet", "cash_flow"} else 4
        snap = snapshot(series, kind=spec.kind, year_steps=year_steps, table=spec.table)
        if spec.kind == "rate" and snap.get("current") is not None:
            if not _in_bounds(spec.id, snap["current"]):
                series = []
                snap = snapshot([], kind=spec.kind, table=spec.table)
        compare_type = str(snap.get("year_ago_period_type") or snap.get("previous_period_type") or "")
        trend = direction(
            higher_is_better=spec.higher_is_better,
            qoq=snap.get("qoq_change"),
            yoy=snap.get("yoy_change"),
            current_period_type=str(snap.get("period_type") or ""),
            compare_period_type=compare_type,
        )
        available = snap.get("current") is not None
        source = "Screener.in cache / company results table"
        findings.append({
            "id": spec.id,
            "label": spec.label,
            "pillar": spec.pillar,
            "weight": spec.weight,
            "unit": spec.unit,
            "table": spec.table,
            "missing_ok": spec.missing_ok,
            "importance": spec.importance,
            "available": available,
            "higher_is_better": spec.higher_is_better,
            "trend": trend if available else "unknown",
            "points": _kpi_points(spec, snap, trend) if available else None,
            "snapshot": snap,
            "fact": _fact_line(spec, snap),
            "interpretation": _interpretation(spec, snap, trend) if available else "Data unavailable",
            "implication": _implication(trend, "pending") if available else "No implication without a measured value.",
            "formula": _formula(spec, snap),
            "source": source if available else "Source unavailable",
            "source_url": source_url,
            "source_date": snap.get("current_period") or fetched_at,
            "retrieved_at": fetched_at,
            "confidence": "high" if len(series) >= 5 else "medium" if len(series) >= 2 else "low",
            "period_type": snap.get("period_type") or "",
            "reporting_basis": snap.get("reporting_basis") or "",
            "source_count": 1 if available else 0,
            "source_consensus": "single" if available else "",
            "agreeing_sources": [source] if available else [],
            "provenance": provenance(
                value=snap.get("current"),
                period=str(snap.get("current_period") or ""),
                source=source if available else "Source unavailable",
                source_url=source_url,
                retrieved_at=fetched_at,
                published_at=str(snap.get("current_period") or ""),
                confidence="high" if len(series) >= 5 else "medium" if len(series) >= 2 else "low",
                raw_reference=f"{spec.table}:{','.join(spec.needles)}",
            ) if available else provenance(
                value=None,
                source="Source unavailable",
                retrieved_at=fetched_at,
                confidence="none",
                raw_reference="Data unavailable",
            ),
        })
    return findings


def _score(findings: Sequence[Mapping[str, Any]], *, min_score_coverage: float = 0.40) -> tuple[int | None, float, dict[str, Any]]:
    meta = score_evidence(findings, min_score_coverage=min_score_coverage)
    return meta["score"], meta["coverage"], meta


def _red_flags(
    findings: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    extra: Sequence[Mapping[str, Any]] | None = None,
    *,
    lending: bool = False,
) -> list[dict[str, Any]]:
    return collect_red_flags(findings, events, extra, lending=lending)


def _technical_context(scan_row: Mapping[str, Any], long_row: Mapping[str, Any]) -> dict[str, Any]:
    sepa = _f(scan_row.get("sepa_score"))
    grade = str(scan_row.get("breakout_grade") or "")
    status = str(scan_row.get("status") or scan_row.get("verdict") or "")
    score = _f(scan_row.get("score"))
    chase = bool(scan_row.get("chase_risk"))
    present = bool(scan_row)
    return {
        "available": present,
        "scanner_status": status or "Data unavailable",
        "scanner_score": score,
        "sepa_score": sepa,
        "breakout_grade": grade or None,
        "breakout_quality": (
            "Strong" if grade in {"A", "B"} else "Watch" if grade else ("Extended" if chase else "Data unavailable")
        ),
        "chase_risk": chase,
        "signals": list(scan_row.get("signals") or [])[:8],
        "reasons": list(scan_row.get("reasons") or [])[:4],
        "long_term_class": str(long_row.get("classification") or "") or None,
        "detail": (
            "SEPA / breakout fields come from the last saved scan overlay — this engine does not rescan."
            if present else
            "This symbol is not on the current scan file. Investigate still runs; vs-setup stays UNMEASURED."
        ),
    }


def _vs_setup(
    *,
    technical: Mapping[str, Any],
    score: int | None,
    coverage: float,
    trend_label: str,
    flags: Sequence[Mapping[str, Any]],
    news_label: str,
    decision_coverage_pct: float = 100.0,
) -> tuple[str, str]:
    if not technical.get("available"):
        return "UNMEASURED", "No current scanner setup to compare against."
    if flags:
        severe = any(
            f.get("kind") in {"regulatory_action", "governance", "asset_quality"}
            or f.get("severity") == "critical"
            for f in flags
        )
        if severe:
            return "STRONGLY CONTRADICTS" if technical.get("chase_risk") else "CONTRADICTS", flags[0]["title"]
    if score is None or coverage < 0.40 or decision_coverage_pct < 40.0:
        return "NEUTRAL", "Insufficient fundamental evidence to raise or cut conviction."
    if news_label == "Negative" and score < 60:
        return "CONTRADICTS", "Material negative news plus mixed fundamentals."
    warnings = [f for f in flags if f.get("severity") == "warning"]
    if score >= 80 and trend_label == "Improving" and news_label != "Negative":
        if warnings:
            return "CAUTION", warnings[0].get("title") or "Quality is strong but warnings are on file."
        label = "STRONGLY SUPPORTS" if not technical.get("chase_risk") else "SUPPORTS"
        return label, "Quality, trend and news are aligned with the setup."
    if score >= 60 and trend_label in {"Improving", "Stable"} and news_label != "Negative":
        if warnings:
            return "CAUTION", warnings[0].get("title") or "Fundamentals are adequate but warnings are on file."
        return "SUPPORTS", "Fundamentals do not contradict the setup."
    if score < 40 or trend_label == "Deteriorating":
        return "CONTRADICTS", "Business trend or quality is against the setup."
    if warnings:
        return "CAUTION", warnings[0].get("title") or "Evidence is mixed with warnings on file."
    return "NEUTRAL", "Evidence is mixed — conviction unchanged."


def _defaults(symbol: str) -> dict[str, Any]:
    from product.scan_store import load_scan
    from product.long_term_store import load_long_term_scan
    from reporting.evidence_intake import load_raw_fundamentals

    news: list[dict[str, Any]] = []
    try:
        from news.curator_store import NewsCuratorStore
        store = NewsCuratorStore(ROOT / "logs" / "news_curator.sqlite3")
        try:
            news = [item.as_dict() for item in store.recent(hours=24 * 90, limit=40, symbol=symbol)]
        finally:
            store.close()
    except Exception:
        news = []
    return {
        "scan": load_scan() or {},
        "long_term": load_long_term_scan() or {},
        "raw": load_raw_fundamentals(symbol),
        "news": news,
    }


def build_due_diligence(
    symbol: str,
    *,
    scan_payload: Mapping[str, Any] | None = None,
    long_term_payload: Mapping[str, Any] | None = None,
    raw_fundamentals: Mapping[str, Any] | None = None,
    news: Sequence[Mapping[str, Any]] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Vertical slice: classify → sector KPIs → news → SUPPORTS/NEUTRAL/CONTRADICTS."""
    from product.stock_workspace import clean_symbol

    symbol = clean_symbol(symbol)
    now = now or datetime.now(timezone.utc)
    if scan_payload is None or long_term_payload is None or raw_fundamentals is None or news is None:
        defaults = _defaults(symbol)
        scan_payload = defaults["scan"] if scan_payload is None else scan_payload
        long_term_payload = defaults["long_term"] if long_term_payload is None else long_term_payload
        raw_fundamentals = defaults["raw"] if raw_fundamentals is None else raw_fundamentals
        news = defaults["news"] if news is None else news

    scan_row = _find(scan_payload, symbol)
    long_row = _find(long_term_payload, symbol)
    raw_record = dict(raw_fundamentals or {})
    raw = dict(raw_record.get("data") or {})
    company = str(scan_row.get("company") or long_row.get("company") or symbol)
    sector = str(long_row.get("sector") or scan_row.get("sector") or "")
    profile = classify_company(
        symbol,
        sector=sector,
        about=str(raw.get("about") or ""),
        quarterly_rows=list(raw.get("quarterly_results") or []),
    )
    framework = get_framework(profile["framework_id"])
    source_url = str(raw.get("url") or "")
    fetched_at = str(raw_record.get("fetched_at") or "")
    findings = _evaluate_kpis(raw, framework["kpis"], source_url, fetched_at)
    autonomy: dict[str, Any] = {}
    try:
        from product.due_diligence.acquire import load_autonomy_facts
        autonomy = load_autonomy_facts(symbol)
    except Exception:
        autonomy = {}
    measured = merge_kpi_maps(extract_kpis_from_raw(raw), dict(autonomy.get("kpis") or {}))
    conflicts = _apply_overlay(findings, framework["kpis"], measured)
    min_score_coverage = float(framework.get("min_score_coverage") or 0.40)
    score, coverage, score_meta = _score(findings, min_score_coverage=min_score_coverage)
    decision = decision_coverage(findings)
    by_id = {str(f.get("id")): f for f in findings}

    def _kpi_current(kpi_id: str) -> float | None:
        return _f((by_id.get(kpi_id) or {}).get("snapshot", {}).get("current"))

    events = material_events(
        list(news or []),
        symbol,
        framework_id=str(framework.get("id") or ""),
        context={
            "revenue_cr": _kpi_current("sales") or _kpi_current("nii"),
            "pat_cr": _kpi_current("pat"),
            "promoter_pct": _kpi_current("promoter"),
            "market_cap_cr": _f(scan_row.get("market_cap") or raw.get("market_cap")),
            "debt_cr": _kpi_current("borrowings"),
        },
    )
    news_label, news_detail = news_verdict(events)
    try:
        research_coverage = inspect_research_coverage(
            symbol=symbol,
            raw=raw,
            autonomy=autonomy,
            news=list(news or []),
            framework_id=framework["id"],
            findings=findings,
            events=events,
            fetched_at=fetched_at,
            now=now,
        )
    except Exception:
        research_coverage = {
            "coverage_pct": 0.0,
            "available_n": 0,
            "required_n": 0,
            "summary": "Coverage inspect failed",
            "needs_acquire": False,
            "to_fetch": [],
            "datasets": [],
            "not_a_quality_score": True,
        }
    _STATE_LABEL = {
        "reported": "Reported",
        "not_yet_acquired": "Not yet acquired",
        "acquisition_failed": "Acquisition failed",
        "source_unavailable": "Source unavailable",
        "metric_not_reported": "Metric not reported",
        "not_applicable": "Not applicable",
    }
    for finding in findings:
        state = availability_state_for_kpi(
            kpi_id=str(finding.get("id") or ""),
            has_value=bool(finding.get("available")),
            missing_ok=bool(finding.get("missing_ok")),
            coverage=research_coverage,
        )
        finding["availability_state"] = state
        finding["availability_label"] = _STATE_LABEL.get(state, "Data unavailable")
    pack = apply_autonomy_pack(
        load_evidence_pack(
            symbol,
            raw=raw,
            scan_as_of=str((scan_payload or {}).get("scanned_at") or ""),
            long_term_as_of=str((long_term_payload or {}).get("scanned_at") or ""),
            news_as_of=str(events[0]["published_at"] if events else ""),
            long_row=long_row,
        ),
        autonomy,
    )
    if pack.get("revenue_drivers") and pack["revenue_drivers"] != "Data unavailable — no segment table on file.":
        profile["revenue_drivers"] = pack["revenue_drivers"]
    # Classifier owns sector / sub-sector / business_model. Pack may fill about
    # text only when the company description is empty — never overwrite the model.
    if not profile.get("about") and pack.get("business_model") and pack["business_model"] != "Data unavailable":
        profile["about"] = pack["business_model"]
    cash = cash_flow_quality(raw, framework_id=framework["id"])
    balance_rules = balance_sheet_quality(raw, framework_id=framework["id"])
    growth = growth_quality(findings)
    extra_flags = list(pack.get("flags") or []) + list(cash.get("flags") or []) + list(balance_rules.get("flags") or [])
    flags = collect_red_flags(
        findings, events, extra_flags, lending=bool(framework.get("lending")),
    )
    flag_groups = partition_flags(flags)
    technical = _technical_context(scan_row, long_row)
    trend_label = _pillar_label([str(f.get("trend")) for f in findings if f.get("available")])
    sector_verdict = sector_kpi_verdict(
        findings,
        min_critical=int(framework.get("min_critical") or 2),
        min_decision_coverage=float(framework.get("min_decision_coverage") or 0.45),
        cycle_aware=bool(framework.get("cycle_aware")),
    )
    sector_kpi_label = sector_verdict["label"]
    missing_rows = missing_evidence(findings)
    missing_critical = critical_metrics_missing(findings)
    vs_setup, vs_detail = _vs_setup(
        technical=technical, score=score, coverage=coverage,
        trend_label=trend_label, flags=flags, news_label=news_label,
        decision_coverage_pct=decision["coverage_pct"],
    )
    confirmation = confirmation_from_evidence(
        vs_setup=vs_setup,
        vs_detail=vs_detail,
        score=score,
        decision_coverage_pct=decision["coverage_pct"],
    )
    vs_setup = confirmation["vs_setup"]
    vs_detail = confirmation["vs_detail"]
    for finding in findings:
        finding["implication"] = _implication(str(finding.get("trend")), vs_setup)

    strengths = [
        f["interpretation"] for f in findings
        if f.get("available") and f.get("trend") == "improving"
    ][:5]
    concerns = [
        f["interpretation"] for f in findings
        if f.get("available") and f.get("trend") == "deteriorating"
    ][:5]
    unavailable = [f["label"] for f in findings if not f.get("available")]
    if not strengths and coverage < 0.30:
        strengths = []
    changed = []
    if findings:
        latest_period = next((f.get("snapshot", {}).get("current_period") for f in findings if f.get("available")), "")
        if latest_period:
            changed.append(f"Latest financial period on file: {latest_period}.")
    if events:
        changed.append(events[0]["headline"])
    if not changed:
        changed.append("No material quarter-to-quarter change could be measured from files on disk.")

    quality_label = _quality_label(score, coverage, min_coverage=min_score_coverage)
    governance = "Low"
    if any(f.get("kind") == "governance" or f.get("severity") == "critical" for f in flags if f.get("kind") == "governance"):
        governance = "Elevated"
    elif any(f.get("kind") == "governance" for f in flags):
        governance = "Elevated"
    elif any(f.get("id") == "promoter" and f.get("trend") == "deteriorating" for f in findings):
        governance = "Watch"

    earnings = growth.get("label") or _pillar_label(
        [str(f.get("trend")) for f in findings if f.get("pillar") == "profitability" and f.get("available")]
    )
    balance = balance_rules.get("label") or _pillar_label([
        str(f.get("trend")) for f in findings
        if f.get("pillar") in {"asset_quality", "cash", "governance", "leverage"} and f.get("available")
    ])
    financial = quality_label if quality_label != "Unmeasured" else "Unmeasured"

    extracted_guidance: list[dict[str, Any]] = []
    for row in pack.get("management_commentary") or []:
        extracted_guidance.extend(
            extract_guidance(
                " ".join(str(row.get(k) or "") for k in ("commentary", "guidance_metric", "guidance_value", "topic")),
                source=f"Research Data commentary ({row.get('speaker') or 'Management'})",
                source_url=str(row.get("source_url") or ""),
                source_date=str(row.get("event_date") or ""),
            )
        )
    for item in list(autonomy.get("guidance") or []):
        if isinstance(item, dict):
            extracted_guidance.append(item)
    seen_g = set()
    unique_guidance: list[dict[str, Any]] = []
    for item in extracted_guidance:
        key = (item.get("excerpt"), item.get("source"))
        if key in seen_g:
            continue
        seen_g.add(key)
        unique_guidance.append(item)
        if len(unique_guidance) >= 8:
            break

    watch = list(framework["watch"])
    if unique_guidance:
        watch = [f"Filing/commentary tone on file is {unique_guidance[0].get('tone')}."] + watch[:3]
    elif any(gap.get("key") == "management_commentary" for gap in pack.get("gaps") or []):
        watch = ["Upload a concall / results commentary in Research Data, or run Acquire — tone is not guessed."] + watch[:3]
    if pack.get("order_book"):
        watch = [f"Order-book / guidance on file: {pack['order_book'][0]['fact']}"] + list(watch[:3])

    as_of = {
        "latest_financial_period": next(
            (f.get("snapshot", {}).get("current_period") for f in findings if f.get("available")),
            "Data unavailable",
        ),
        "fundamentals_fetched_at": fetched_at or "Data unavailable",
        "fundamentals_freshness": str(raw_record.get("freshness") or "MISSING"),
        "latest_material_news": events[0]["published_at"] if events else "Data unavailable",
        "scan_scanned_at": str((scan_payload or {}).get("scanned_at") or ""),
        "generated_at": now.isoformat(),
        "evidence_pack_coverage_pct": pack.get("coverage_pct"),
        "autonomy_acquired_at": str(autonomy.get("acquired_at") or ""),
        "latest_annual_report": next(
            (
                str(item.get("title") or item.get("path") or "")
                for item in list(autonomy.get("downloads") or [])
                if "annual" in str(item.get("url") or item.get("path") or "").lower()
            ),
            "Data unavailable",
        ),
        "latest_data_refresh": research_coverage.get("latest_data_refresh") or autonomy.get("acquired_at") or fetched_at or "Data unavailable",
        "research_coverage_pct": research_coverage.get("coverage_pct"),
    }

    breakdown = score_breakdown(
        findings, framework_id=framework["id"], coverage=coverage, overall=score,
    )
    peers = rank_peers(
        list(pack.get("peers") or []),
        company=company,
        symbol=symbol,
        framework_id=str(framework.get("id") or ""),
        peer_note=str(framework.get("peer_note") or ""),
    )
    filings = []
    for item in list(autonomy.get("downloads") or [])[:20]:
        url = str(item.get("url") or "")
        path = str(item.get("path") or "")
        title = str(item.get("title") or path or url or "Filing")
        kind = "others"
        blob = f"{title} {url} {path}".lower()
        if "result" in blob or "financial" in blob:
            kind = "quarterly results"
        elif "annual" in blob:
            kind = "annual report"
        elif "presentation" in blob or "investor" in blob:
            kind = "investor presentation"
        elif "shareholding" in blob:
            kind = "shareholding"
        elif "credit" in blob or "rating" in blob:
            kind = "credit rating"
        filings.append({
            "title": title,
            "category": kind,
            "url": url,
            "path": path,
            "ok": bool(item.get("ok")),
            "source": "Acquire archive",
        })
    for event in events:
        if event.get("official") or event.get("verified"):
            filings.append({
                "title": event.get("headline"),
                "category": event.get("category") or event.get("event_type") or "others",
                "url": event.get("url") or "",
                "path": "",
                "ok": True,
                "source": event.get("source") or "Exchange / news curator",
                "published_at": event.get("published_at"),
            })
    sources = []
    seen_src = set()
    for finding in findings:
        prov = dict(finding.get("provenance") or {})
        key = (prov.get("source"), prov.get("source_url"), prov.get("period"))
        if key in seen_src or not finding.get("available"):
            continue
        seen_src.add(key)
        sources.append(prov)
    for event in events:
        key = (event.get("source"), event.get("url"))
        if key in seen_src:
            continue
        seen_src.add(key)
        sources.append(provenance(
            value=event.get("headline"),
            period=str(event.get("published_at") or ""),
            source=str(event.get("source") or ""),
            source_url=str(event.get("url") or ""),
            published_at=str(event.get("published_at") or ""),
            source_type="exchange_filing" if event.get("official") else "financial_media",
            confidence="high" if event.get("official") else "medium",
            raw_reference=str(event.get("category") or event.get("event_type") or ""),
        ))

    valuation = [
        item for item in list(pack.get("snapshot_metrics") or [])
        if item.get("id") in {"pe", "pb", "ev_ebitda", "dividend_yield"} or item.get("pillar") == "valuation"
    ]

    report = {
        "schema_version": 6,
        "engine": "StockResearchEngine",
        "symbol": symbol,
        "company": company,
        "profile": profile,
        "framework": {
            "id": framework["id"],
            "label": framework["label"],
            "blurb": framework["blurb"],
            "sub_sector": profile.get("sub_sector") or framework.get("default_sub_sector") or "",
            "business_model": profile.get("business_model") or framework.get("default_business_model") or "",
            "peer_note": framework.get("peer_note") or "",
        },
        "technical_context": technical,
        "long_term_overlay": pack.get("long_term_overlay") or {},
        "fundamental_quality": {
            "score": score,
            "label": quality_label,
            "coverage_pct": round(coverage * 100.0, 1),
            "score_coverage_pct": score_meta.get("coverage_pct"),
            "raw_awarded": score_meta.get("raw_awarded"),
            "evaluated_weight": score_meta.get("evaluated_weight"),
            "scoring_weight": score_meta.get("scoring_weight"),
            "explain": (
                (
                    f"{score_meta['n']} measurable KPIs scored {score_meta.get('raw_awarded')} "
                    f"/ {score_meta.get('evaluated_weight')} evaluated weight "
                    f"(score coverage {score_meta.get('coverage_pct')}%). "
                    "Missing KPIs are skipped, never scored as zero."
                )
                if score is not None else
                (
                    score_meta.get("unmeasured_because")
                    or "Fundamental quality is Unmeasured — evaluated weight is too thin to display a /100 score."
                )
            ),
            "breakdown": breakdown,
        },
        "research_coverage": research_coverage,
        "decision_coverage": decision,
        "decision_coverage_pct": decision.get("coverage_pct"),
        "missing_evidence": missing_rows,
        "critical_metrics_missing": missing_critical,
        "deeper_acquire_available": bool(research_coverage.get("needs_acquire")),
        "sector_kpi_label": sector_kpi_label,
        "sector_kpi_detail": sector_verdict.get("detail"),
        "business_trend": trend_label,
        "financial_strength": financial,
        "earnings_quality": earnings,
        "balance_sheet_quality": balance,
        "cash_flow_quality": cash,
        "growth_quality": growth,
        "balance_sheet_rules": balance_rules,
        "governance_risk": governance,
        "news_event_impact": news_label,
        "vs_technical_setup": vs_setup,
        "fundamental_confirmation": confirmation.get("display") or confirmation_label(vs_setup),
        "confirmation_reason": confirmation.get("reason"),
        "confirmation_qualifier": confirmation.get("qualifier") or "",
        "vs_detail": vs_detail,
        "strengths": strengths,
        "concerns": concerns,
        "unavailable": unavailable,
        "what_changed": changed[:4],
        "red_flags": flags,
        "flag_groups": flag_groups,
        "watch_next": watch,
        "kpis": findings,
        "events": events,
        "extracted_guidance": unique_guidance,
        "evidence_pack": pack,
        "peers": peers,
        "filings": filings[:24],
        "valuation": valuation,
        "sources": sources[:40],
        "source_conflicts": conflicts,
        "cache_schedule": [],
        "autonomy": {
            "acquired_at": autonomy.get("acquired_at") or None,
            "steps": list(autonomy.get("steps") or []),
            "downloads": list(autonomy.get("downloads") or [])[:12],
            "still_missing": list(autonomy.get("still_missing") or []),
            "files_on_disk": list(autonomy.get("files_on_disk") or []),
            "option_chain": dict(autonomy.get("option_chain") or {}) or None,
            "dataset_meta": dict(autonomy.get("dataset_meta") or {}),
            "mode": autonomy.get("mode"),
            "to_fetch": list(autonomy.get("to_fetch") or []),
            "not_an_llm": True,
        },
        "as_of": as_of,
        "places_orders": False,
        "uses_llm": False,
        "disclaimer": (
            "Due diligence on a scanner candidate — not a buy list, not a new scan, "
            "not a broker recommendation. Empty stays empty. The desk synthesis is "
            "rule-based, not a language model."
        ),
        "question": (
            "QuantTerm identified a technical setup. After sector KPIs, filings-on-file "
            "and material news, does the evidence raise conviction, leave it unchanged, "
            "or argue to reduce it?"
        ),
    }
    report["company_snapshot"] = company_snapshot(
        symbol=symbol, company=company, profile=profile, findings=findings,
        technical=technical, as_of=as_of, scan_row=scan_row, raw=raw,
    )
    report["cache_schedule"] = cache_schedule(as_of)
    report["first_screen"] = first_screen(report)
    report["thesis"] = compose_thesis(report)
    return report
