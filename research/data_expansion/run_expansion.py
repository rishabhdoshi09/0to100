"""One-shot research data expansion runner (D1–D10; no alpha experiments)."""
from __future__ import annotations

import json
from pathlib import Path

from research.data_expansion.assess import (
    assess_fundamentals_events,
    assess_sector_history,
    future_research_families,
    low_vol_retest_readiness,
    research_power,
)
from research.data_expansion.classify import (
    WINDOW_END,
    WINDOW_START,
    classify_universe,
    write_classification,
)
from research.data_expansion.snapshot import build_expanded_snapshot
from research.phase_a5.scoped_certification import FROZEN_PANEL

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "logs" / "research_expansion"


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    print("D1 classify…", flush=True)
    cls = classify_universe(run_ca_audit=True, progress_every=50)
    write_classification(cls)
    print("counts", cls.counts, "certifiable", len(cls.certifiable_symbols), flush=True)

    print("D7 snapshot…", flush=True)
    snap = build_expanded_snapshot(cls)
    print("snapshot", snap["snapshot_id"], "verify", snap["verify_ok"], flush=True)

    cov = snap.get("coverage") or {}
    power = research_power(
        n_securities=snap["n_symbols"],
        n_sessions=snap["n_sessions"],
        security_sessions=int(cov.get("total_security_sessions") or 0),
        date_start=(snap.get("date_range") or [WINDOW_START])[0],
        date_end=(snap.get("date_range") or [WINDOW_END])[1],
        prior_n=29,
        prior_sessions=764,
    )
    sector = assess_sector_history()
    funds = assess_fundamentals_events()
    lowvol = low_vol_retest_readiness(
        n_securities=snap["n_symbols"],
        n_sessions=snap["n_sessions"],
    )
    families = future_research_families(
        n_certifiable=len(cls.certifiable_symbols),
        low_vol_verdict=lowvol["verdict"],
    )

    # Prior 29 verification reminder
    prior_path = REPO_ROOT / "logs" / "phase_a5_scoped" / "snapshots" / "a7a9828ec37e09e4"
    prior_ok = prior_path.exists()
    prior29_in = sorted(set(cls.certifiable_symbols) & set(FROZEN_PANEL))

    payload = {
        "classification_counts": cls.counts,
        "certifiable_n": len(cls.certifiable_symbols),
        "partial_n": len(cls.partial_symbols),
        "snapshot": {
            "snapshot_id": snap["snapshot_id"],
            "verify_ok": snap["verify_ok"],
            "verify_fails": snap["verify_fails"],
            "n_symbols": snap["n_symbols"],
            "n_sessions": snap["n_sessions"],
            "date_range": snap["date_range"],
            "coverage": cov,
            "root": snap["root"],
        },
        "prior_29_snapshot_verified_path": str(prior_path),
        "prior_29_snapshot_present": prior_ok,
        "prior_29_in_expanded_certifiable": prior29_in,
        "research_power": power,
        "sector_history": sector,
        "fundamentals_events": funds,
        "low_vol_retest": lowvol,
        "future_families": families,
        "window": [WINDOW_START, WINDOW_END],
        "global_trust_class": "OPERATIONAL_ONLY",
        "hashes": cls.hashes,
        "git_sha": cls.git_sha,
    }
    (OUT / "expansion_result.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    return payload


if __name__ == "__main__":
    run()
