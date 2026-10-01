"""Phase 3: a compact daily Evolution health report.

Not a performance dashboard -- a plumbing/evidence-quality check, reusing
only already-persisted stores (the shadow-decision ledger, the policy
registry, scorecard's existing aggregations). Never scans the market, never
mutates anything. A quiet day (no decisions, no grading) is reported as
healthy, not as a failure: a fresh domain, an off-hours run, or a market
holiday are all legitimate zero states.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.runtime_paths import logs_dir
from product.evidence_class import PAPER_FORWARD
from product.evolution import policy_registry as PR
from product.evolution import scorecard
from product.evolution import shadow_decisions as SD

REGIME_DOMINANCE_THRESHOLD = 0.85
SECTOR_DOMINANCE_THRESHOLD = 0.85
DUPLICATE_AGREEMENT_THRESHOLD = 0.98
DUPLICATE_MIN_PAIRED = 10
REQUIRED_PROVENANCE_FIELDS = ("market_snapshot_id", "frozen_at", "fingerprint", "policy_id")


def health_report_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_EVOLUTION_HEALTH_REPORTS")
    if override:
        return Path(override)
    return logs_dir() / "product" / "evolution_health_reports.jsonl"


def _read_reports(path: str | Path | None = None) -> list[dict[str, Any]]:
    target = health_report_path(path)
    if not target.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        for line in target.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                out.append(row)
    except Exception:
        return []
    return out


def _persist_report(report: dict[str, Any], *, path: str | Path | None = None) -> None:
    target = health_report_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = _read_reports(path)
    rows = [r for r in rows if not (r.get("domain") == report.get("domain") and r.get("as_of") == report.get("as_of"))]
    rows.append(report)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows[-2000:]:
            fh.write(json.dumps(row, default=str, sort_keys=True) + "\n")
    tmp.replace(target)


def _storage_growth(domain: str, total_rows_today: int, *, path: str | Path | None = None) -> dict[str, Any]:
    """Compare today's total row count against the recent trailing average of
    this module's OWN prior daily reports -- the only durable day-by-day
    trend record available without duplicating the shadow ledger itself."""
    history = [
        r for r in _read_reports(path)
        if r.get("domain") == domain and "total_shadow_rows_this_domain" in r
    ]
    history.sort(key=lambda r: str(r.get("as_of") or ""))
    prior = [int(r["total_shadow_rows_this_domain"]) for r in history[-8:-1]]
    if len(prior) < 3:
        return {"baseline_days": len(prior), "abnormal_growth": False, "note": "not enough history yet"}
    avg_prior = sum(prior) / len(prior)
    delta = total_rows_today - prior[-1]
    avg_daily_delta = (prior[-1] - prior[0]) / max(1, len(prior) - 1)
    abnormal = avg_daily_delta > 0 and delta > 5 * avg_daily_delta and delta > 50
    return {
        "baseline_days": len(prior),
        "avg_prior_total": round(avg_prior, 1),
        "today_total": total_rows_today,
        "delta_vs_yesterday": delta,
        "abnormal_growth": bool(abnormal),
    }


def _today_rows(rows: list[dict[str, Any]], today: str, field: str) -> list[dict[str, Any]]:
    return [r for r in rows if str(r.get(field) or "")[:10] == today]


def _provenance_gaps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gaps = []
    for row in rows:
        missing = [k for k in REQUIRED_PROVENANCE_FIELDS if not row.get(k)]
        if missing:
            gaps.append({"shadow_id": row.get("shadow_id"), "missing": missing})
    return gaps


def _dominance(rows: list[dict[str, Any]], field: str, threshold: float) -> dict[str, Any] | None:
    values = [str(r.get(field) or "UNKNOWN") for r in rows]
    if not values:
        return None
    counts = Counter(values)
    top_value, top_count = counts.most_common(1)[0]
    share = top_count / len(values)
    if share >= threshold:
        return {"value": top_value, "share": round(share, 3), "n": len(values)}
    return None


def _duplicate_pairs(
    domain: str, challengers: list[dict[str, Any]], *, ledger_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Two Challengers that nearly always agree on the same snapshots are
    not adding distinct hypotheses to the population -- flagged, never
    auto-retired (an operator decides whether to deduplicate)."""
    dups = []
    for i, a in enumerate(challengers):
        for b in challengers[i + 1:]:
            paired = scorecard.paired_comparison(
                a["policy_id"], b["policy_id"], domain=domain, path=ledger_path,
            )
            n = paired["paired_snapshots"]
            if n < DUPLICATE_MIN_PAIRED:
                continue
            agreement = paired["agreement"]
            agree_n = agreement["both_selected"] + agreement["both_rejected"]
            rate = agree_n / n if n else 0.0
            if rate >= DUPLICATE_AGREEMENT_THRESHOLD:
                dups.append({
                    "policy_a": a["policy_id"], "policy_b": b["policy_id"],
                    "agreement_rate": round(rate, 3), "paired_snapshots": n,
                })
    return dups


def _fingerprint_consistency(domain: str) -> dict[str, Any]:
    """The discovery-cache fingerprint check only exists for EQUITY today
    (Home/Best Trades reads the EQUITY Champion only) -- reported as not
    applicable for other domains rather than silently always-consistent."""
    champion = PR.current_champion(domain)
    if champion is None:
        return {"applicable": domain == PR.EQUITY, "consistent": True, "note": "no champion yet"}
    champion_fp = PR.policy_manifest_fingerprint(champion)
    if domain != PR.EQUITY:
        return {"applicable": False, "consistent": True, "champion_fingerprint": champion_fp}
    try:
        from product.decision_discovery_store import current_evolution_policy_fingerprint
        discovery_fp = current_evolution_policy_fingerprint()
    except Exception:
        discovery_fp = None
    return {
        "applicable": True,
        "consistent": not discovery_fp or discovery_fp == champion_fp,
        "champion_fingerprint": champion_fp,
        "discovery_fingerprint": discovery_fp,
    }


def _historical_leak_check(domain: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Every row promotion/scorecard actually reads comes from THIS ledger.
    A genuine historical-evidence leak would show up as a non-forward
    evidence_class on a graded row here -- it never has, by construction
    (historical_priors.py writes to a physically separate store), but this
    is the continuous check rather than a one-time code-reading assertion."""
    graded = [r for r in rows if r.get("outcome") is not None]
    leaked = [
        r for r in graded
        if str(r.get("evidence_class") or "") not in ("EVOLUTION_SHADOW", PAPER_FORWARD)
    ]
    return {
        "clean": not leaked, "leaked_count": len(leaked),
        "leaked_shadow_ids": [r.get("shadow_id") for r in leaked[:10]],
    }


def _contract_evidence_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    contract_rows = [
        r for r in rows
        if r.get("outcome") is not None
        and str(r.get("grading_mode") or "") == "PAPER_FORWARD_CONTRACT_ONLY"
    ]
    bad = [r for r in contract_rows if str(r.get("evidence_class") or "") != PAPER_FORWARD]
    return {
        "clean": not bad, "checked": len(contract_rows), "bad_count": len(bad),
        "bad_shadow_ids": [r.get("shadow_id") for r in bad[:10]],
    }


def daily_health_report(
    domain: str, *, as_of: str | None = None,
    registry_path: str | Path | None = None, ledger_path: str | Path | None = None,
    persist: bool = True, path: str | Path | None = None,
) -> dict[str, Any]:
    today = as_of or datetime.now(timezone.utc).date().isoformat()
    champion = PR.current_champion(domain, path=registry_path)
    champion_id = champion["policy_id"] if champion else None

    all_rows = SD.list_shadow_decisions(path=ledger_path)
    domain_rows = [r for r in all_rows if r.get("domain") == domain]
    today_rows = _today_rows(domain_rows, today, "frozen_at")

    champion_today = [r for r in today_rows if r.get("is_champion_decision")]
    challenger_today = [r for r in today_rows if not r.get("is_champion_decision")]
    taken_today = [r for r in today_rows if r.get("decision") == "ENTER_NOW"]
    wait_today = [r for r in today_rows if r.get("decision") == "WAIT"]
    rejected_today = [r for r in today_rows if r.get("decision") not in ("ENTER_NOW", "WAIT")]
    graded_today = _today_rows(domain_rows, today, "graded_at")
    pending = [r for r in domain_rows if r.get("outcome") is None]

    challengers = PR.active_challengers(domain, path=registry_path)
    paired_counts = {}
    if champion_id:
        for c in challengers:
            paired = scorecard.paired_comparison(
                champion_id, c["policy_id"], domain=domain, path=ledger_path,
            )
            paired_counts[c["policy_id"]] = paired["paired_snapshots"]

    never_produced = [
        c["policy_id"] for c in challengers
        if int(scorecard.scorecard(c["policy_id"], domain=domain, path=ledger_path).get("decision_snapshots") or 0) == 0
    ]

    graded_rows_all = [r for r in domain_rows if r.get("outcome") is not None]

    report: dict[str, Any] = {
        "schema_version": 1,
        "domain": domain,
        "as_of": today,
        "champion_policy_id": champion_id,
        "champion_made_decisions_today": len(champion_today) > 0,
        "champion_decisions_today": len(champion_today),
        "challenger_shadows_today": len(challenger_today),
        "taken_today": len(taken_today),
        "rejected_today": len(rejected_today),
        "wait_today": len(wait_today),
        "graded_today": len(graded_today),
        "pending_ungraded_total": len(pending),
        "paired_comparisons_by_challenger": paired_counts,
        "challengers_with_zero_output_ever": never_produced,
        "regime_dominance": _dominance(graded_rows_all, "regime", REGIME_DOMINANCE_THRESHOLD),
        "sector_dominance": _dominance(graded_rows_all, "sector", SECTOR_DOMINANCE_THRESHOLD),
        "duplicate_policy_pairs": _duplicate_pairs(domain, challengers, ledger_path=ledger_path),
        "provenance_gaps_today": _provenance_gaps(today_rows)[:10],
        "provenance_gap_count_all_time": len(_provenance_gaps(domain_rows)),
        "fingerprint_consistency": _fingerprint_consistency(domain),
        "historical_evidence_leak_check": _historical_leak_check(domain, domain_rows),
        "fno_contract_evidence_check": (
            _contract_evidence_check(domain_rows) if domain == PR.FNO_CONTRACT else None
        ),
        "total_shadow_rows_this_domain": len(domain_rows),
    }

    if persist:
        report["storage_growth"] = _storage_growth(
            domain, report["total_shadow_rows_this_domain"], path=path,
        )
        _persist_report(report, path=path)
    else:
        report["storage_growth"] = _storage_growth(
            domain, report["total_shadow_rows_this_domain"], path=path,
        )
    return report
