"""EDGE-004 inference. Reversal slope + §17 robustness bar (predeclared)."""
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
)
from research.edge001.constants import block_of
from research.edge004.constants import DSR_N_TRIALS, LOG_DIR, OUT_DIR
from research.harness import block_bootstrap_mean_ci, deflated_sharpe_ratio, evaluate, probabilistic_sharpe_ratio


def classify(stats: dict[str, Any]) -> dict[str, Any]:
    prim = stats.get("primary") or {}
    dec = stats.get("deciles", {}).get("R1") or stats.get("deciles", {}).get("M1") or {}
    blocks = stats.get("blocks") or {}
    years = prim.get("by_year_net") or {}
    formulas = stats.get("formula_excess_ew") or {}
    inf = stats.get("inference") or {}
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

    spearman = dec.get("spearman")
    # D10 = losers. Reversal ⇒ positive decile-vs-next Spearman.
    if spearman is None or spearman != spearman or spearman < 0.20:
        failures.append("deciles_not_inverse")
    d10m1 = dec.get("d10_minus_d1")
    if d10m1 is not None and d10m1 == d10m1 and d10m1 <= 0:
        failures.append("losers_do_not_beat_winners")

    if years:
        vals = list(years.values())
        total_abs = sum(abs(v) for v in vals) or 1.0
        if max(vals) / total_abs > 0.80 and len(vals) >= 3:
            failures.append("one_year_dominates")

    if (prim.get("cagr_gross") or 0) > (prim.get("ew_cagr") or 0) and (prim.get("cagr_net") or 0) <= (prim.get("ew_cagr") or 0):
        failures.append("costs_destroy_edge")

    calmar = prim.get("calmar")
    excess = prim.get("excess_cagr_ew")
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

    r2 = formulas.get("R2")
    r1 = formulas.get("R1")
    if r2 is not None and r1 is not None and r2 == r2 and r1 == r1 and r1 <= 0 and r2 > 0.02:
        failures.append("only_10session_works")
        notes.append("10-session is nearer EXP-NEXT-01. Do not switch inside EDGE-004.")

    win = formulas.get("WIN")
    if win is not None and win == win and win > 0 and (r1 is None or r1 <= 0):
        failures.append("only_winners_work")
        notes.append("WIN book is continuation, already studied as EDGE-001 family.")

    ci_excludes = bool(((inf.get("excess_ew") or {}).get("ci") or {}).get("excludes_zero"))
    harness = ((inf.get("harness_excess_ew") or {}).get("verdict") or "")
    conf_flat = conf_ex is not None and conf_ex == conf_ex and abs(float(conf_ex)) < 0.01

    label = "RESEARCH-ONLY"
    hard = {
        "net_does_not_beat_ew_or_nifty",
        "deciles_not_inverse",
        "losers_do_not_beat_winners",
        "costs_destroy_edge",
    }
    if "only_10session_works" in failures and not beat_ew:
        label = "MODIFY HYPOTHESIS"
    elif "only_winners_work" in failures and not beat_ew:
        label = "MODIFY HYPOTHESIS"
    elif len(hard & set(failures)) >= 2:
        label = "REJECT"
    elif (
        beat_ew
        and not conf_reverse
        and "costs_destroy_edge" not in failures
        and "deciles_not_inverse" not in failures
        and "losers_do_not_beat_winners" not in failures
        and "one_year_dominates" not in failures
        and valconf_n >= 12
    ):
        if (ci_excludes or harness in ("PROMOTE", "PASS")) and not conf_flat:
            label = "PROMISING — FORWARD VALIDATION WARRANTED"
        else:
            label = "RESEARCH-ONLY"
            notes.append("§17 robustness bar: CI includes 0 and/or confirmation economically flat.")
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
    dec_raw = json.loads((artifacts / "decile_returns.json").read_text())
    regime = json.loads((artifacts / "regime_periods.json").read_text()) if (artifacts / "regime_periods.json").exists() else []
    snaps = json.loads((artifacts / "universe_snapshots.json").read_text()) if (artifacts / "universe_snapshots.json").exists() else []

    prim = periods.get("primary_R1_top20_monthly") or []
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
        bsl = [b for b in bench if block_of(str(b.get("rebalance") or "")) == name]
        m = _spec_metrics(sl, "net")
        nr = _arr([p.get("net") for p in sl])
        er = _arr([b.get("ew_universe") for b in bsl])
        nf = _arr([b.get("nifty") for b in bsl])
        if nr.size and er.size == nr.size:
            m["excess_cagr_ew"] = _ann_from_monthly(nr) - _ann_from_monthly(er)
        if nr.size and nf.size == nr.size:
            m["excess_cagr_nifty"] = _ann_from_monthly(nr) - _ann_from_monthly(nf)
        blocks[name] = m

    sensitivities = {k: _spec_metrics(v, "net") for k, v in periods.items() if k != "primary_R1_top20_monthly"}
    formula_excess = {}
    keymap = {
        "R1": "primary_R1_top20_monthly",
        "R0": "sens_R0_top20_monthly",
        "R2": "sens_R2_top20_monthly",
        "R3": "sens_R3_top20_monthly",
        "WIN": "diag_WIN_top20_monthly",
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

    deciles = {
        "M1": _decile_table(dec_raw.get("R1_decile") or []),
        "R1": _decile_table(dec_raw.get("R1_decile") or []),
        "R0": _decile_table(dec_raw.get("R0_decile") or []),
        "R2": _decile_table(dec_raw.get("R2_decile") or []),
        "R3": _decile_table(dec_raw.get("R3_decile") or []),
    }

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
        "h3": _spec_metrics(periods.get("sens_R2_top20_monthly") or [], "net"),
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
    stats["h3"]["excess_cagr_ew"] = formula_excess.get("R2")
    stats["decision"] = classify(stats)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (artifacts / "edge_004_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    (OUT_DIR / "edge_004_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    (OUT_DIR / "edge_004_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return stats
