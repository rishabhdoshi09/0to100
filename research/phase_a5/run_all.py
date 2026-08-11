"""Orchestrate Phase A.5 evidence activation (research only)."""
from __future__ import annotations

import json
from pathlib import Path

from research.phase_a5.dataset import commit_exploratory_snapshot, load_closes, load_sectors, load_manifest
from research.phase_a5.exp_structure import run_exp_a5_01
from research.phase_a5.exp_network import run_exp_a6_01
from research.phase_a5.exp_horizons import run_exp_a2_01
from research.phase_a5.exp_challenger import run_exp_a3_01
from research.phase_a5.exp_interaction import run_exp_a5a6_01

OUT = Path(__file__).resolve().parents[2] / "logs" / "phase_a5" / "results.json"


def run_phase_a5() -> dict:
    closes = load_closes()
    sectors = load_sectors()
    sid, pit, manifest = commit_exploratory_snapshot(closes)
    # Smoke: PitContract must serve bars without future leakage
    sample_sym = list(closes.columns)[0]
    mid = closes.index[len(closes) // 2]
    as_of = mid.strftime("%Y-%m-%d")
    read = pit.as_of("bars", when=as_of, symbol=sample_sym)
    assert read.usable, f"PitContract failed: {read}"
    assert all(b.date <= as_of for b in read.data)

    results = {
        "manifest": manifest,
        "pit_smoke": {
            "status": read.status,
            "symbol": sample_sym,
            "as_of": as_of,
            "n_bars": len(read.data),
            "snapshot_id": sid,
        },
        "production_behaviour_changed": False,
        "experiments": {},
    }

    results["experiments"]["EXP-A5-01"] = run_exp_a5_01(
        closes=closes, sectors=sectors, manifest=manifest
    )
    results["experiments"]["EXP-A6-01"] = run_exp_a6_01(
        closes=closes, sectors=sectors, manifest=manifest
    )
    results["experiments"]["EXP-A2-01"] = run_exp_a2_01(
        closes=closes, manifest=manifest
    )
    results["experiments"]["EXP-A3-01"] = run_exp_a3_01(
        closes=closes, manifest=manifest
    )
    results["experiments"]["EXP-A5A6-01"] = run_exp_a5a6_01(
        closes=closes, sectors=sectors, manifest=manifest
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return results


if __name__ == "__main__":
    out = run_phase_a5()
    print(json.dumps({
        "snapshot_id": out["manifest"].get("snapshot_id"),
        "trust_class": out["manifest"].get("trust_class"),
        "verdicts": {k: v.get("verdict") for k, v in out["experiments"].items()},
        "hypothesis_ids": {k: v.get("hypothesis_id") for k, v in out["experiments"].items()},
        "production_behaviour_changed": out["production_behaviour_changed"],
        "results_path": str(OUT),
    }, indent=2))
