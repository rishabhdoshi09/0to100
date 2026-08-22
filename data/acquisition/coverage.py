"""Machine-readable Phase II coverage. Validation stats only — no strategy metrics."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from data.acquisition.cache import write_manifest
from research.data_foundation.quality import event_quality, fundamental_quality

ROOT = Path(__file__).resolve().parents[2]


def _fund_rows() -> list[dict]:
    from data.pit_fundamentals import _load_rows
    return _load_rows()


def _event_rows() -> list[dict]:
    from data.pit_events import ledger_path, _coerce_rows
    p = ledger_path()
    if not p.exists():
        return []
    try:
        return _coerce_rows(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return []


def fundamentals_coverage(rows: list[dict] | None = None) -> dict[str, Any]:
    rows = rows if rows is not None else _fund_rows()
    by_sym: dict[str, list] = {}
    years = Counter()
    q_vs_a = Counter()
    quality = Counter()
    for r in rows:
        by_sym.setdefault(r["symbol"], []).append(r)
        pe = str(r.get("period_end") or r.get("available_at") or "")[:4]
        if pe.isdigit():
            years[pe] += 1
        kind = str(r.get("period") or "").lower()
        q_vs_a["annual" if "annual" in kind else "quarterly"] += 1
        quality[fundamental_quality(r)] += 1
    n_q = {s: sum(1 for r in rs if r.get("quarterly_usable") is True
                  or str(r.get("period") or "").lower() == "quarterly")
           for s, rs in by_sym.items()}
    counts = list(n_q.values())
    counts.sort()
    median = counts[len(counts) // 2] if counts else 0
    dates = [r.get("available_at") for r in rows if r.get("available_at")]
    return {
        "rows": len(rows),
        "symbols_with_ge1": len(by_sym),
        "symbols_with_ge4_quarters": sum(1 for n in n_q.values() if n >= 4),
        "symbols_with_ge8_quarters": sum(1 for n in n_q.values() if n >= 8),
        "symbols_with_ge12_quarters": sum(1 for n in n_q.values() if n >= 12),
        "median_quarters_per_company": median,
        "earliest_filing": min(dates) if dates else None,
        "latest_filing": max(dates) if dates else None,
        "annual_vs_quarterly": dict(q_vs_a),
        "by_year": dict(sorted(years.items())),
        "quality": dict(quality),
    }


def events_coverage(rows: list[dict] | None = None) -> dict[str, Any]:
    rows = rows if rows is not None else _event_rows()
    years = Counter()
    tq = Counter()
    for r in rows:
        y = str(r.get("available_at") or "")[:4]
        if y.isdigit():
            years[y] += 1
        tq[event_quality(r)] += 1
    dates = [r.get("available_at") for r in rows if r.get("available_at")]
    return {
        "event_count": len(rows),
        "symbols": len({r.get("symbol") for r in rows}),
        "earliest": min(dates) if dates else None,
        "latest": max(dates) if dates else None,
        "by_year": dict(sorted(years.items())),
        "timestamp_quality": dict(tq),
        "consensus_series": False,
        "may_compute_surprise": False,
        "date_only_causal_policy": "NEXT_SESSION",
    }


def build_report() -> dict[str, Any]:
    from data.benchmarks import file_coverage, load_index
    from data.ca_research import research_status
    from data.listing_archive import universe_pit_class
    from data.sector_map import coverage as sector_coverage
    from data.universe_history import ledger_status
    from research.feature002.acceptance import evaluate_first_real_scan, operational_state

    fund = fundamentals_coverage()
    ev = events_coverage()
    uni = universe_pit_class()
    bhav = ledger_status(ROOT / "logs" / "universe_history_bhav_inferred.json")
    official = ledger_status(ROOT / "logs" / "universe_history_v2.json")
    sec = sector_coverage()
    ca = research_status()
    bench_files = file_coverage()
    nifty50 = load_index("Nifty 50")
    nifty500 = load_index("Nifty 500")
    ntm = load_index("Nifty Total Market")
    f002 = operational_state()
    acc = evaluate_first_real_scan()
    out = {
        "generated_from": "data.acquisition.coverage.build_report",
        "fundamentals": fund,
        "earnings_events": ev,
        "universe": {
            "default": uni,
            "official_v2": {
                "rows": official.get("rows"),
                "source": official.get("source"),
                "research_grade": official.get("research_grade"),
                "completeness": official.get("completeness"),
            },
            "bhav_inferred_sidecar": {
                "rows": bhav.get("rows"),
                "source": bhav.get("source"),
                "research_grade": bhav.get("research_grade"),
            },
        },
        "sector": sec,
        "ca": ca,
        "benchmarks": {
            "files": bench_files,
            "nifty50": {k: nifty50.get(k) for k in ("first", "last", "n", "return_kind", "available")},
            "nifty500": {k: nifty500.get(k) for k in ("first", "last", "n", "return_kind", "available")},
            "nifty_total_market": {k: ntm.get(k) for k in ("first", "last", "n", "return_kind", "available")},
        },
        "feature002": {
            "operational": f002,
            "first_real_scan_accepted": acc.get("accepted"),
            "note": acc.get("note"),
        },
    }
    write_manifest("data_coverage", out)
    dest = ROOT / "docs" / "data_program" / "DATA_COVERAGE_REPORT.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out
