"""EDGE-003 inference. Classification follows the inclusion protocol, not EDGE-001 deciles."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.edge001.analyze import (
    _ann_from_monthly,
    _arr,
    _beta,
    _block_slice,
    _by_year,
    _crash,
    _moments,
    _sharpe,
    _spec_metrics,
)
from research.edge001.constants import block_of
from research.edge003.constants import CAPITALS, DSR_N_TRIALS, LOG_DIR, OUT_DIR, PARTICIPATION_FLAG
from research.harness import block_bootstrap_mean_ci, deflated_sharpe_ratio, evaluate, probabilistic_sharpe_ratio


def _inclusion_table(rows: list[dict]) -> dict[str, Any]:
    by: dict[str, list[float]] = defaultdict(list)
    nobs: dict[str, list[int]] = defaultdict(list)
    share: dict[str, list[float]] = defaultdict(list)
    paired: list[tuple[float, float, float, str]] = []  # t1, ext1, share_t1, rebalance
    by_reb: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        lab = str(r.get("bucket") or "")
        if r.get("mean") is None:
            continue
        by[lab].append(float(r["mean"]))
        nobs[lab].append(int(r.get("n") or 0))
        share[lab].append(float(r.get("share") or float("nan")))
        by_reb[str(r.get("rebalance") or "")][lab] = float(r["mean"])
    for reb, d in by_reb.items():
        if "T1" in d and "exT1" in d:
            sh = next((float(r["share"]) for r in rows if r.get("rebalance") == reb and r.get("bucket") == "T1"), float("nan"))
            paired.append((d["T1"], d["exT1"], sh, reb))
    diffs = _arr([a - b for a, b, _, _ in paired])
    shares = _arr([s for _, _, s, _ in paired if s == s])
    spearman = float("nan")
    if len(paired) >= 8 and shares.size == len(paired) and float(np.std(shares)) > 0:
        xs = pd.Series([s for _, _, s, _ in paired])
        ys = pd.Series([a - b for a, b, _, _ in paired])
        spearman = float(xs.corr(ys, method="spearman"))
    t1 = _arr(by.get("T1", []))
    ex = _arr(by.get("exT1", []))
    return {
        "n_paired": len(paired),
        "t1_mean": float(t1.mean()) if t1.size else float("nan"),
        "ext1_mean": float(ex.mean()) if ex.size else float("nan"),
        "t1_minus_ext1": float(diffs.mean()) if diffs.size else float("nan"),
        "t1_beats_ext1_share": float((diffs > 0).mean()) if diffs.size else float("nan"),
        "mean_t1_share": float(shares.mean()) if shares.size else float("nan"),
        "share_vs_spread_spearman": spearman,
        "t1_n_mean": float(np.mean(nobs["T1"])) if nobs.get("T1") else float("nan"),
        "ext1_n_mean": float(np.mean(nobs["exT1"])) if nobs.get("exT1") else float("nan"),
    }


def _capacity_from_medians(primary: list[dict]) -> dict[str, Any]:
    """Variable-N book: flag if (capital / n) ≥ 5% of median ADV."""
    flags = {int(c): 0 for c in CAPITALS}
    n_reb = 0
    for p in primary:
        n = max(int(p.get("n_picks") or 0), 1)
        adv = float(p.get("median_adv20") or 0.0)
        n_reb += 1
        for cap in CAPITALS:
            pos = float(cap) / n
            if adv <= 0 or pos / adv >= PARTICIPATION_FLAG:
                flags[int(cap)] += 1
    n_reb = max(n_reb, 1)
    return {
        "participation_flag": PARTICIPATION_FLAG,
        "method": "median_adv20_times_equal_weight_clip",
        "capitals": {
            str(c): {
                "flagged_rebalances": flags[int(c)],
                "flagged_share": flags[int(c)] / n_reb,
            }
            for c in CAPITALS
        },
    }


def classify(stats: dict[str, Any]) -> dict[str, Any]:
    """Exactly one label. Rules written against EDGE-003 protocol, not the tape."""
    prim = stats.get("primary") or {}
    blocks = stats.get("blocks") or {}
    years = prim.get("by_year_net") or {}
    formulas = stats.get("formula_excess_ew") or {}
    incl = stats.get("inclusion") or {}
    failures: list[str] = []
    notes: list[str] = []

    val = blocks.get("validation") or {}
    conf = blocks.get("confirmation") or {}
    valconf_n = int(val.get("n") or 0) + int(conf.get("n") or 0)
    later_ex_ew = None
    later_ex_nifty = None
    if valconf_n:
        parts_ew = [x for x in (val.get("excess_cagr_ew"), conf.get("excess_cagr_ew")) if x is not None and x == x]
        parts_nf = [x for x in (val.get("excess_cagr_nifty"), conf.get("excess_cagr_nifty")) if x is not None and x == x]
        later_ex_ew = float(np.mean(parts_ew)) if parts_ew else None
        later_ex_nifty = float(np.mean(parts_nf)) if parts_nf else None

    beat_ew = (later_ex_ew is not None and later_ex_ew > 0) or (prim.get("excess_cagr_ew") or -1) > 0
    beat_nifty = (later_ex_nifty is not None and later_ex_nifty > 0) or (prim.get("excess_cagr_nifty") or -1) > 0
    if not beat_ew and not beat_nifty:
        failures.append("net_does_not_beat_ew_or_nifty")

    share = prim.get("mean_qualifier_share")
    if share is None:
        share = incl.get("mean_t1_share")
    excess = prim.get("excess_cagr_ew")
    if share is not None and share == share and share >= 0.90 and (
        excess is None or excess != excess or abs(float(excess)) < 0.01
    ):
        failures.append("qualifier_is_the_market")

    spread = incl.get("t1_minus_ext1")
    if spread is not None and spread == spread and spread <= 0:
        failures.append("included_do_not_beat_excluded")

    if years:
        vals = list(years.values())
        total_abs = sum(abs(v) for v in vals) or 1.0
        if max(vals) / total_abs > 0.80 and len(vals) >= 3:
            failures.append("one_year_dominates")

    if (prim.get("cagr_gross") or 0) > (prim.get("ew_cagr") or 0) and (prim.get("cagr_net") or 0) <= (prim.get("ew_cagr") or 0):
        failures.append("costs_destroy_edge")

    calmar = prim.get("calmar")
    if calmar is not None and calmar == calmar and calmar < 0.15 and (excess is None or excess <= 0):
        failures.append("drawdown_unacceptable_vs_excess")

    dev_ex = (blocks.get("development") or {}).get("excess_cagr_ew")
    conf_ex = (blocks.get("confirmation") or {}).get("excess_cagr_ew")
    conf_reverse = (
        dev_ex is not None and conf_ex is not None
        and dev_ex == dev_ex and conf_ex == conf_ex
        and dev_ex > 0.02 and conf_ex < -0.02
    )
    if conf_reverse:
        failures.append("confirmation_reverses_development")

    t1_ex = formulas.get("T1")
    top20_ex = formulas.get("T1_TOP20")
    only_top20 = (
        t1_ex is not None and top20_ex is not None
        and t1_ex == t1_ex and top20_ex == top20_ex
        and t1_ex <= 0 and top20_ex > 0.02
    )
    if only_top20:
        failures.append("only_top20_distance_works")
        notes.append("Inclusion failed; distance-rank Top20 is a different hypothesis. Do not switch inside EDGE-003.")

    label = "RESEARCH-ONLY"
    hard = {
        "net_does_not_beat_ew_or_nifty",
        "qualifier_is_the_market",
        "included_do_not_beat_excluded",
        "costs_destroy_edge",
    }
    if only_top20 and not beat_ew:
        label = "MODIFY HYPOTHESIS"
    elif len(hard & set(failures)) >= 2:
        label = "REJECT"
    elif (
        beat_ew
        and not conf_reverse
        and "costs_destroy_edge" not in failures
        and "qualifier_is_the_market" not in failures
        and "included_do_not_beat_excluded" not in failures
        and "only_top20_distance_works" not in failures
        and "one_year_dominates" not in failures
        and valconf_n >= 12
        and share is not None and share == share and share < 0.90
    ):
        label = "PROMISING — FORWARD VALIDATION WARRANTED"
    elif failures:
        label = "RESEARCH-ONLY"

    return {
        "label": label,
        "failures": failures,
        "notes": notes,
        "later_excess_ew": later_ex_ew,
        "later_excess_nifty": later_ex_nifty,
        "live_trading_authorised": False,
        "paper_trading_authorised": False,
        "feature002_change_authorised": False,
    }


def analyse(artifacts: Path | None = None) -> dict[str, Any]:
    artifacts = Path(artifacts or LOG_DIR)
    manifest = json.loads((artifacts / "experiment_manifest.json").read_text())
    periods = json.loads((artifacts / "portfolio_periods.json").read_text())
    ew = json.loads((artifacts / "ew_universe.json").read_text())
    bench = json.loads((artifacts / "benchmark_comparison.json").read_text())
    buckets = json.loads((artifacts / "inclusion_buckets.json").read_text())
    regime = json.loads((artifacts / "regime_periods.json").read_text()) if (artifacts / "regime_periods.json").exists() else []
    snaps = json.loads((artifacts / "universe_snapshots.json").read_text()) if (artifacts / "universe_snapshots.json").exists() else []

    prim = periods.get("primary_T1_all_monthly") or []
    pm = _spec_metrics(prim, "net")
    pm.update({"by_year_net": _by_year(prim, "net"), "by_year_gross": _by_year(prim, "gross")})
    shares = _arr([p.get("qualifier_share") for p in prim])
    pm["mean_qualifier_share"] = float(shares.mean()) if shares.size else float("nan")
    pm["median_qualifier_share"] = float(np.median(shares)) if shares.size else float("nan")
    pm["min_qualifier_share"] = float(shares.min()) if shares.size else float("nan")
    pm["max_qualifier_share"] = float(shares.max()) if shares.size else float("nan")
    pm["mean_n_univ"] = float(np.mean([p.get("n_univ") or 0 for p in prim])) if prim else float("nan")

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
        bsl = [b for b in bench if block_of(str(b.get("rebalance") or "")) == name]
        m = _spec_metrics(sl, "net")
        m["mean_qualifier_share"] = float(np.mean([p.get("qualifier_share") or 0 for p in sl])) if sl else float("nan")
        nr = _arr([p.get("net") for p in sl])
        er = _arr([b.get("ew_universe") for b in bsl])
        nf = _arr([b.get("nifty") for b in bsl])
        if nr.size and er.size == nr.size:
            m["excess_cagr_ew"] = _ann_from_monthly(nr) - _ann_from_monthly(er)
        if nr.size and nf.size == nr.size:
            m["excess_cagr_nifty"] = _ann_from_monthly(nr) - _ann_from_monthly(nf)
        blocks[name] = m

    sensitivities = {k: _spec_metrics(v, "net") for k, v in periods.items() if k != "primary_T1_all_monthly"}
    formula_excess = {}
    keymap = {
        "T1": "primary_T1_all_monthly",
        "T2": "sens_T2_all_monthly",
        "T3": "sens_T3_all_monthly",
        "T1_TOP20": "sens_T1_top20_dist",
        "T1_4W": "sens_T1_all_4week",
        "T1_2M": "sens_T1_all_2month",
        "T1_Q": "sens_T1_all_quarterly",
    }
    for key, col in keymap.items():
        sl = periods.get(col) or []
        mp = {p["rebalance"]: p for p in sl}
        aligned, ewa = [], []
        for e in ew:
            if e["rebalance"] in mp:
                aligned.append(mp[e["rebalance"]]["net"])
                ewa.append(e["gross"])
        if aligned:
            formula_excess[key] = _ann_from_monthly(_arr(aligned)) - _ann_from_monthly(_arr(ewa))

    by_reg: dict[str, list[float]] = defaultdict(list)
    for r in regime:
        st = str(r.get("regime") or "UNKNOWN")
        bucket = {"STRONG_BULL": "bull", "BULL": "bull", "SIDEWAYS": "sideways",
                  "CORRECTION": "correction", "BEAR": "bear"}.get(st, "unknown")
        if r.get("net") is not None:
            by_reg[bucket].append(float(r["net"]))
    from research.edge001.analyze import _max_dd_from_rets
    regime_out = {}
    for k, xs in by_reg.items():
        a = _arr(xs)
        regime_out[k] = {
            "months": int(a.size),
            "mean": float(a.mean()) if a.size else float("nan"),
            "cagr": _ann_from_monthly(a) if a.size else float("nan"),
            "max_dd": _max_dd_from_rets(a) if a.size else float("nan"),
        }

    stats = {
        "manifest": manifest,
        "primary": pm,
        "blocks": blocks,
        "sensitivities": sensitivities,
        "formula_excess_ew": formula_excess,
        "inclusion": _inclusion_table(buckets),
        "inference": inference,
        "crash": _crash(prim),
        "capacity": _capacity_from_medians(prim),
        "regime": regime_out,
        "universe": {
            "n_snapshots": len(snaps),
            "avg_candidates": float(np.mean([s.get("candidate_count") or 0 for s in snaps])) if snaps else None,
            "avg_investable": float(np.mean([s.get("investable_count") or 0 for s in snaps])) if snaps else None,
            "avg_ranked": float(np.mean([s.get("ranked_count") or 0 for s in snaps])) if snaps else None,
            "avg_n_t1": float(np.mean([s.get("n_t1") or 0 for s in snaps])) if snaps else None,
            "avg_t1_share": float(np.mean([s.get("t1_share") or 0 for s in snaps])) if snaps else None,
        },
    }
    stats["decision"] = classify(stats)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (artifacts / "edge_003_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    (OUT_DIR / "edge_003_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    (OUT_DIR / "edge_003_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return stats
