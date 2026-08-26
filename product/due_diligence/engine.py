"""Assemble an evidence-backed due-diligence report for one scanner candidate."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from product.due_diligence.classify import classify_company
from product.due_diligence.extract import extract_guidance, extract_kpis_from_raw, merge_kpi_maps
from product.due_diligence.frameworks import KpiSpec, get_framework
from product.due_diligence.news_layer import material_events, news_verdict
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


def _quality_label(score: int | None, coverage: float) -> str:
    if score is None or coverage < 0.30:
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
    line = f"{spec.label}: {current} {unit} ({period})"
    prev = snap.get("previous")
    if prev is not None and snap.get("previous_period"):
        line += f"; previous {snap['previous_period']}: {prev} {unit}"
    year = snap.get("year_ago")
    if year is not None and snap.get("year_ago_period"):
        line += f"; year-ago {snap['year_ago_period']}: {year} {unit}"
    return line


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
) -> None:
    """Fill Data unavailable KPIs from extra tables, key-ratios or downloaded filings."""
    spec_by_id = {spec.id: spec for spec in specs}
    for finding in findings:
        if finding.get("available"):
            continue
        spec = spec_by_id.get(str(finding.get("id")))
        snap = dict(measured.get(str(finding.get("id"))) or {})
        if spec is None or snap.get("current") is None:
            continue
        trend = direction(
            higher_is_better=spec.higher_is_better,
            qoq=snap.get("qoq_change"),
            yoy=snap.get("yoy_change"),
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
            "source": source,
            "source_url": str(snap.get("source_url") or finding.get("source_url") or ""),
            "source_date": str(snap.get("source_date") or snap.get("current_period") or finding.get("source_date") or ""),
            "confidence": "medium",
        })


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
        if not series and spec.id in {"gnpa", "nnpa", "pledge", "promoter"}:
            for other in tables.values():
                row = find_row(other, spec.needles)
                series = dated_series(row)
                if series:
                    break
        snap = snapshot(series, kind=spec.kind)
        trend = direction(
            higher_is_better=spec.higher_is_better,
            qoq=snap.get("qoq_change"),
            yoy=snap.get("yoy_change"),
        )
        available = snap.get("current") is not None
        findings.append({
            "id": spec.id,
            "label": spec.label,
            "pillar": spec.pillar,
            "weight": spec.weight,
            "unit": spec.unit,
            "available": available,
            "higher_is_better": spec.higher_is_better,
            "trend": trend if available else "unknown",
            "points": _kpi_points(spec, snap, trend) if available else None,
            "snapshot": snap,
            "fact": _fact_line(spec, snap),
            "interpretation": _interpretation(spec, snap, trend) if available else "Data unavailable",
            "implication": _implication(trend, "pending") if available else "No implication without a measured value.",
            "source": "Screener.in cache / company results table",
            "source_url": source_url,
            "source_date": snap.get("current_period") or fetched_at,
            "confidence": "high" if len(series) >= 5 else "medium" if len(series) >= 2 else "low",
        })
    return findings


def _score(findings: Sequence[Mapping[str, Any]]) -> tuple[int | None, float, dict[str, Any]]:
    usable = [f for f in findings if f.get("points") is not None]
    total_w = sum(float(f.get("weight") or 0) for f in findings)
    used_w = sum(float(f.get("weight") or 0) for f in usable)
    coverage = (used_w / total_w) if total_w else 0.0
    if not usable or coverage < 0.30:
        return None, coverage, {"used_weight": used_w, "total_weight": total_w, "n": len(usable)}
    weighted = sum(float(f["points"]) * float(f["weight"]) for f in usable)
    score = int(round(weighted / used_w))
    return score, coverage, {"used_weight": used_w, "total_weight": total_w, "n": len(usable)}


def _red_flags(findings: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for finding in findings:
        if finding.get("id") in {"gnpa", "nnpa"} and finding.get("trend") == "deteriorating":
            flags.append({
                "id": f"flag-{finding['id']}",
                "title": f"{finding['label']} is deteriorating",
                "kind": "asset_quality",
                "fact": finding["fact"],
                "source": finding.get("source"),
                "source_date": finding.get("source_date"),
            })
        if finding.get("id") == "pledge" and finding.get("available"):
            current = (finding.get("snapshot") or {}).get("current")
            if finding.get("trend") == "deteriorating" or (current is not None and current > 20):
                flags.append({
                    "id": "flag-pledge",
                    "title": "Promoter pledge is elevated or rising",
                    "kind": "governance",
                    "fact": finding["fact"],
                    "source": finding.get("source"),
                    "source_date": finding.get("source_date"),
                })
        if finding.get("id") == "promoter" and finding.get("trend") == "deteriorating":
            qoq = (finding.get("snapshot") or {}).get("qoq_change")
            if qoq is not None and qoq <= -3:
                flags.append({
                    "id": "flag-promoter-drop",
                    "title": "Promoter holding fell by more than 3 percentage points last quarter",
                    "kind": "governance",
                    "fact": finding["fact"],
                    "source": finding.get("source"),
                    "source_date": finding.get("source_date"),
                })
    for event in events:
        if event.get("event_type") in {"regulatory_action", "pledge", "governance"} or (
            event.get("impact") == "negative" and event.get("thesis_change")
        ):
            flags.append({
                "id": f"flag-news-{event.get('event_type')}",
                "title": event.get("headline"),
                "kind": event.get("event_type") or "news",
                "fact": event.get("headline"),
                "source": event.get("source"),
                "source_date": event.get("published_at"),
                "url": event.get("url"),
            })
    return flags


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
) -> tuple[str, str]:
    if not technical.get("available"):
        return "UNMEASURED", "No current scanner setup to compare against."
    if flags:
        severe = any(f.get("kind") in {"regulatory_action", "governance", "asset_quality"} for f in flags)
        if severe:
            return "STRONGLY CONTRADICTS" if technical.get("chase_risk") else "CONTRADICTS", flags[0]["title"]
    if score is None or coverage < 0.30:
        return "NEUTRAL", "Fundamental coverage is too thin to raise or cut conviction."
    if news_label == "Negative" and score < 60:
        return "CONTRADICTS", "Material negative news plus mixed fundamentals."
    if score >= 80 and trend_label == "Improving" and news_label != "Negative":
        label = "STRONGLY SUPPORTS" if not technical.get("chase_risk") else "SUPPORTS"
        return label, "Quality, trend and news are aligned with the setup."
    if score >= 60 and trend_label in {"Improving", "Stable"} and news_label != "Negative":
        return "SUPPORTS", "Fundamentals do not contradict the setup."
    if score < 40 or trend_label == "Deteriorating":
        return "CONTRADICTS", "Business trend or quality is against the setup."
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
    _apply_overlay(findings, framework["kpis"], measured)
    score, coverage, score_meta = _score(findings)
    events = material_events(list(news or []), symbol)
    news_label, news_detail = news_verdict(events)
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
    if pack.get("business_model") and pack["business_model"] != "Data unavailable":
        profile["business_model"] = pack["business_model"]
        if not profile.get("about"):
            profile["about"] = pack["business_model"]
    flags = _red_flags(findings, events)
    for extra in pack.get("flags") or []:
        if extra.get("id") not in {item.get("id") for item in flags}:
            flags.append(extra)
    technical = _technical_context(scan_row, long_row)
    trend_label = _pillar_label([str(f.get("trend")) for f in findings if f.get("available")])
    vs_setup, vs_detail = _vs_setup(
        technical=technical, score=score, coverage=coverage,
        trend_label=trend_label, flags=flags, news_label=news_label,
    )
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

    quality_label = _quality_label(score, coverage)
    governance = "Low"
    if any(f.get("kind") == "governance" for f in flags):
        governance = "Elevated"
    elif any(f.get("id") == "promoter" and f.get("trend") == "deteriorating" for f in findings):
        governance = "Watch"

    earnings = _pillar_label([str(f.get("trend")) for f in findings if f.get("pillar") == "profitability" and f.get("available")])
    balance = _pillar_label([
        str(f.get("trend")) for f in findings
        if f.get("pillar") in {"asset_quality", "cash", "governance"} and f.get("available")
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
    }

    report = {
        "schema_version": 3,
        "symbol": symbol,
        "company": company,
        "profile": profile,
        "framework": {
            "id": framework["id"],
            "label": framework["label"],
            "blurb": framework["blurb"],
        },
        "technical_context": technical,
        "long_term_overlay": pack.get("long_term_overlay") or {},
        "fundamental_quality": {
            "score": score,
            "label": quality_label,
            "coverage_pct": round(coverage * 100.0, 1),
            "explain": (
                f"{score_meta['n']} sector KPIs with values, "
                f"{score_meta['used_weight']:.0f}/{score_meta['total_weight']:.0f} weight. "
                "Missing KPIs are skipped, never filled."
            ),
        },
        "business_trend": trend_label,
        "financial_strength": financial,
        "earnings_quality": earnings,
        "balance_sheet_quality": balance,
        "governance_risk": governance,
        "news_event_impact": news_label,
        "vs_technical_setup": vs_setup,
        "vs_detail": vs_detail,
        "strengths": strengths,
        "concerns": concerns,
        "unavailable": unavailable,
        "what_changed": changed[:4],
        "red_flags": flags,
        "watch_next": watch,
        "kpis": findings,
        "events": events,
        "extracted_guidance": unique_guidance,
        "evidence_pack": pack,
        "autonomy": {
            "acquired_at": autonomy.get("acquired_at") or None,
            "steps": list(autonomy.get("steps") or []),
            "downloads": list(autonomy.get("downloads") or [])[:12],
            "still_missing": list(autonomy.get("still_missing") or []),
            "files_on_disk": list(autonomy.get("files_on_disk") or []),
            "option_chain": dict(autonomy.get("option_chain") or {}) or None,
            "not_an_llm": True,
        },
        "as_of": as_of,
        "places_orders": False,
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
    report["thesis"] = compose_thesis(report)
    return report
