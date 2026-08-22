"""EDGE-002 inference. Reuses EDGE-001 metric helpers; own primary key."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from research.edge001.analyze import (
    _ann_from_monthly,
    _arr,
    _beta,
    _block_slice,
    _by_year,
    _capacity,
    _crash,
    _decile_table,
    _moments,
    _sharpe,
    _spec_metrics,
    classify,
)
from research.edge002.constants import DSR_N_TRIALS, LOG_DIR, OUT_DIR
from research.harness import block_bootstrap_mean_ci, deflated_sharpe_ratio, evaluate, probabilistic_sharpe_ratio


def analyse(artifacts: Path | None = None) -> dict[str, Any]:
    artifacts = Path(artifacts or LOG_DIR)
    manifest = json.loads((artifacts / "experiment_manifest.json").read_text())
    periods = json.loads((artifacts / "portfolio_periods.json").read_text())
    ew = json.loads((artifacts / "ew_universe.json").read_text())
    bench = json.loads((artifacts / "benchmark_comparison.json").read_text())
    dec_raw = json.loads((artifacts / "decile_returns.json").read_text())
    regime = json.loads((artifacts / "regime_periods.json").read_text()) if (artifacts / "regime_periods.json").exists() else []
    snaps = json.loads((artifacts / "universe_snapshots.json").read_text()) if (artifacts / "universe_snapshots.json").exists() else []

    prim = periods.get("primary_V1_top20_monthly") or []
    pm = _spec_metrics(prim, "net")
    pm.update({"by_year_net": _by_year(prim, "net"), "by_year_gross": _by_year(prim, "gross")})
    nifty_rets = _arr([b.get("nifty") for b in bench])
    ew_rets = _arr([b.get("ew_universe") for b in bench])
    net_rets = _arr([p.get("net") for p in prim])
    if nifty_rets.size == net_rets.size and net_rets.size:
        pm["nifty_cagr"] = _ann_from_monthly(nifty_rets)
        pm["excess_cagr_nifty"] = pm["cagr_net"] - pm["nifty_cagr"]
        pm["beta_nifty"] = _beta(net_rets, nifty_rets)
        pm["hit_vs_nifty"] = float((net_rets > nifty_rets).mean())
    if ew_rets.size == net_rets.size and net_rets.size:
        pm["ew_cagr"] = _ann_from_monthly(ew_rets)
        pm["excess_cagr_ew"] = pm["cagr_net"] - pm["ew_cagr"]
        pm["hit_vs_ew"] = float((net_rets > ew_rets).mean())

    excess_ew = _arr([b.get("excess_net_vs_ew") for b in bench])
    inference = {}
    if excess_ew.size >= 3:
        sk, ku = _moments(excess_ew)
        sh = _sharpe(excess_ew)
        inference["excess_ew"] = {
            "mean": float(excess_ew.mean()),
            "ci": block_bootstrap_mean_ci(excess_ew, n_boot=2000, seed=21),
            "sharpe": sh,
            "psr": probabilistic_sharpe_ratio(sh, int(excess_ew.size), sk, ku, 0.0),
            "dsr": deflated_sharpe_ratio(sh, int(excess_ew.size), sk, ku, n_trials=DSR_N_TRIALS),
            "p_beat_ew": float((excess_ew > 0).mean()),
        }
        ev = evaluate(excess_ew, n_trials=DSR_N_TRIALS, require_block_ci=True, block_ci_seed=21)
        inference["harness_excess_ew"] = {
            "verdict": ev.verdict, "n": ev.n, "mean_r": ev.mean_r,
            "sharpe": ev.sharpe, "psr": ev.psr, "dsr": ev.dsr, "insight": ev.insight,
        }

    blocks = {}
    for name in ("development", "validation", "confirmation"):
        sl = _block_slice(prim, name)
        bsl = _block_slice(bench, name)
        m = _spec_metrics(sl, "net")
        nr = _arr([p.get("net") for p in sl])
        er = _arr([b.get("ew_universe") for b in bsl])
        nf = _arr([b.get("nifty") for b in bsl])
        if nr.size and er.size == nr.size:
            m["excess_cagr_ew"] = _ann_from_monthly(nr) - _ann_from_monthly(er)
        if nr.size and nf.size == nr.size:
            m["excess_cagr_nifty"] = _ann_from_monthly(nr) - _ann_from_monthly(nf)
        blocks[name] = m

    sensitivities = {k: _spec_metrics(v, "net") for k, v in periods.items() if k != "primary_V1_top20_monthly"}
    formula_excess = {}
    for key, col in (("V1", "primary_V1_top20_monthly"), ("V2", "sens_V2_top20_monthly"),
                     ("V3", "sens_V3_top20_monthly"), ("V0", "diag_V0_top20_monthly")):
        sl = periods.get(col) or []
        mp = {p["rebalance"]: p for p in sl}
        aligned, ewa = [], []
        for e in ew:
            if e["rebalance"] in mp:
                aligned.append(mp[e["rebalance"]]["net"])
                ewa.append(e["gross"])
        if aligned:
            formula_excess[key] = _ann_from_monthly(_arr(aligned)) - _ann_from_monthly(_arr(ewa))

    v0 = _spec_metrics(periods.get("diag_V0_top20_monthly") or [], "net")
    v0["excess_cagr_ew"] = formula_excess.get("V0")

    deciles = {
        "M1": _decile_table(dec_raw.get("V1_decile") or []),  # classify() reads deciles.M1
        "V1": _decile_table(dec_raw.get("V1_decile") or []),
        "V0": _decile_table(dec_raw.get("V0_decile") or []),
        "V2": _decile_table(dec_raw.get("V2_decile") or []),
        "V3": _decile_table(dec_raw.get("V3_decile") or []),
    }

    by_reg: dict[str, list[float]] = defaultdict(list)
    for r in regime:
        st = str(r.get("regime") or "UNKNOWN")
        bucket = {"STRONG_BULL": "bull", "BULL": "bull", "SIDEWAYS": "sideways",
                  "CORRECTION": "correction", "BEAR": "bear"}.get(st, "unknown")
        if r.get("net") is not None:
            by_reg[bucket].append(float(r["net"]))
    regime_out = {}
    for k, xs in by_reg.items():
        a = _arr(xs)
        from research.edge001.analyze import _ann_from_monthly as ann, _max_dd_from_rets
        regime_out[k] = {"months": int(a.size), "mean": float(a.mean()) if a.size else float("nan"),
                         "cagr": ann(a) if a.size else float("nan"),
                         "max_dd": _max_dd_from_rets(a) if a.size else float("nan")}

    stats = {
        "manifest": manifest,
        "primary": pm,
        "blocks": blocks,
        "sensitivities": sensitivities,
        "formula_excess_ew": formula_excess,
        "h3": v0,  # 20d diagnostic occupies the classify() consumed-lookback slot
        "deciles": deciles,
        "inference": inference,
        "crash": _crash(prim),
        "capacity": _capacity(prim),
        "regime": regime_out,
        "universe": {
            "n_snapshots": len(snaps),
            "avg_candidates": float(np.mean([s.get("candidate_count") or 0 for s in snaps])) if snaps else None,
            "avg_investable": float(np.mean([s.get("investable_count") or 0 for s in snaps])) if snaps else None,
            "avg_ranked": float(np.mean([s.get("ranked_count") or 0 for s in snaps])) if snaps else None,
        },
    }
    stats["decision"] = classify(stats)
    if formula_excess.get("V1", 0) <= 0 and formula_excess.get("V0", 0) > 0:
        stats["decision"]["failures"] = list(stats["decision"].get("failures") or []) + ["only_consumed_20d_lookback"]
        if stats["decision"]["label"] == "PROMISING — FORWARD VALIDATION WARRANTED":
            stats["decision"]["label"] = "RESEARCH-ONLY"
            stats["decision"].setdefault("notes", []).append("Only V0 (20d, EXP-NEXT-02 overlap) beat EW.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (artifacts / "edge_002_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    (OUT_DIR / "edge_002_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    (OUT_DIR / "edge_002_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return stats
