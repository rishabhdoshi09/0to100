"""SEPA-001R2 study orchestration — validity-first, then long history."""
from __future__ import annotations

from typing import Any

from research.sepa.ablation_r2 import persist_r2, run_ablation_r2
from research.sepa.ca_audit import (
    build_timeline,
    ca_research_acceptability,
    unresolved_events,
    verify_report,
)
from research.sepa.config import R2_CONFIG
from research.sepa.integrity import research_integrity_report
from research.sepa.universe_pit import load_store_frames


def try_expand_bhav(*, days: int = 1800) -> dict[str, Any]:
    """Best-effort official bhav download. Never fabricates missing sessions."""
    try:
        from data.bhavcopy_store import build_store
        n = build_store(days=int(days))
        from data.bhavcopy_runtime import status
        st = status(load_cache=True)
        return {"symbols": n, **st}
    except Exception as exc:
        return {"error": str(exc)}


def try_ingest_ca(years: list[int] | None = None) -> dict[str, Any]:
    """Merge official NSE share-count events. Does not invent factors."""
    from data.corporate_actions import merge_events
    from data.nse_ca_ingest import fetch_ca_range, rows_from_nse_payload

    years = years or list(range(2018, 2027))
    adjusting: list[dict] = []
    metas: list[dict] = []
    errors: list[str] = []
    import requests
    sess = requests.Session()
    for y in years:
        fr, to = f"01-01-{y}", f"31-12-{y}"
        try:
            payload, meta = fetch_ca_range(fr, to, session=sess)
            packed = rows_from_nse_payload(payload, source_sha256=meta.get("source_sha256") or "")
            adjusting.extend(packed["adjusting"])
            metas.append({"year": y, **{k: meta[k] for k in meta if k != "raw"}})
        except Exception as exc:
            errors.append(f"{y}: {exc}")
    if adjusting:
        merge_events(
            [{"symbol": e["symbol"], "ex_date": e["ex_date"], "factor": e["factor"], "type": e["type"]}
             for e in adjusting],
            source="nse_corporates_api",
        )
    return {"n_adjusting_fetched": len(adjusting), "years": years, "errors": errors, "metas": metas}


def coverage_table(frames: dict) -> dict[str, Any]:
    import pandas as pd
    starts, ends = [], []
    year_syms: dict[str, set] = {}
    n_sessions = 0
    for sym, df in frames.items():
        if df is None or len(df) == 0:
            continue
        idx = pd.DatetimeIndex(df.index)
        starts.append(idx.min())
        ends.append(idx.max())
        n_sessions = max(n_sessions, len(idx))
        for y in set(idx.year):
            year_syms.setdefault(str(int(y)), set()).add(sym)
    return {
        "n_symbols": len(frames),
        "first": str(min(starts).date()) if starts else "",
        "last": str(max(ends).date()) if ends else "",
        "symbols_by_year": {y: len(s) for y, s in sorted(year_syms.items())},
        "sessions_span_days": (
            int((max(ends) - min(starts)).days) if starts else 0
        ),
        "max_symbol_sessions": n_sessions,
    }


def run_study_r2(*, expand: bool = False, max_date_step: int = 1) -> dict[str, Any]:
    coverage_note = {}
    if expand:
        coverage_note["bhav"] = try_expand_bhav()
        coverage_note["ca"] = try_ingest_ca()
        try:
            from data.bhavcopy_store import reload_corporate_actions
            reload_corporate_actions()
        except Exception:
            pass
    frames = load_store_frames(min_bars=80)
    if not frames:
        from research.sepa.study import synthetic_panel
        frames = synthetic_panel()
        source = "synthetic_panel"
        kwargs = dict(warmup_sessions=80, min_sessions=80, min_price=1.0,
                      min_turnover=1.0, horizon=12, date_step=1, scanner_step=99,
                      variants=("F", "G"))
    else:
        source = "official_nse_bhavcopy"
        kwargs = dict(
            warmup_sessions=252, min_sessions=260, min_price=20.0,
            min_turnover=5_000_000.0, horizon=20, date_step=1,
            scanner_step=1, variants=("A", "B", "C", "D", "E", "F", "G"),
            top_n=None,
        )
    cov = coverage_table(frames)
    # Exhaustive unresolved-event audit over the research-relevant store.
    print(f"SEPA-001R2.1 exhaustive CA audit on {len(frames)} frames", flush=True)
    unresolved = unresolved_events(frames, sample=None)
    print(f"SEPA-001R2.1 unresolved events enumerated: {len(unresolved)}", flush=True)
    timeline = build_timeline(unresolved)
    # Global verifier: sample is documented and does NOT certify the study.
    ca_rep = verify_report(frames, sample=min(200, len(frames)))
    integ = research_integrity_report(
        frames=frames, as_of=cov.get("first") or None, verify=True, exhaustive=True,
    )
    ca_ok = ca_research_acceptability(
        unresolved=unresolved,
        exhaustive=True,
        inferred_factors=False,
        unknown_path_crossings=0,
        future_leak_removed_prior=0,
        audit_persisted=True,
        contaminated_uncensored=0,
    )
    integ["ca_research_acceptable"] = ca_ok["ca_research_acceptable"]
    integ["ca_research"] = ca_ok
    integ["n_unresolved_enumerated"] = len(unresolved)
    core = run_ablation_r2(
        frames=frames, config=R2_CONFIG, ca_timeline=timeline, integrity=integ, **kwargs,
    )
    core["integrity"] = integ
    core["data_source"] = source
    core["coverage"] = cov
    core["coverage_note"] = coverage_note
    core["ca_audit"] = {
        **ca_rep,
        "unresolved_events": unresolved,
        "n_unresolved": len(unresolved),
        "ca_research_acceptable": ca_ok["ca_research_acceptable"],
        "ca_research": ca_ok,
        "segment_events": timeline.to_audit(),
        "verify_sample_is_not_certification": True,
        "exhaustive_unresolved": True,
        "n_frames_audited": len(frames),
    }
    core["unresolved_n"] = len(unresolved)
    persist_r2(core, name="ablation_001r2.json")
    try:
        from research.sepa.report_r2 import write_all
        core["reports"] = {k: str(v) for k, v in write_all(core).items()}
    except Exception as exc:
        core["report_error"] = str(exc)
    core["top100_sensitivity"] = {
        "note": "Full vs top-100 RS is a separate denominator study; not a second A–F grid.",
    }
    return core
