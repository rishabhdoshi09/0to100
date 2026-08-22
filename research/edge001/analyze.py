"""EDGE-001 inference. Classification rules locked to the protocol, not the tape."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.edge001.constants import (
    CAPITALS,
    DSR_N_TRIALS,
    LOG_DIR,
    OUT_DIR,
    PARTICIPATION_FLAG,
    PRIMARY_N,
    block_of,
)
from research.harness import (
    block_bootstrap_mean_ci,
    deflated_sharpe_ratio,
    evaluate,
    probabilistic_sharpe_ratio,
)


def _arr(xs) -> np.ndarray:
    return np.asarray([float(x) for x in xs if x is not None and x == x], dtype=float)


def _ann_from_monthly(rets: np.ndarray) -> float:
    r = _arr(rets)
    if r.size == 0:
        return float("nan")
    nav = float(np.prod(1.0 + r))
    years = r.size / 12.0
    if years <= 0 or nav <= 0:
        return float("nan")
    return nav ** (1.0 / years) - 1.0


def _vol(rets: np.ndarray) -> float:
    r = _arr(rets)
    if r.size < 2:
        return float("nan")
    return float(r.std(ddof=1) * math.sqrt(12))


def _sharpe(rets: np.ndarray) -> float:
    r = _arr(rets)
    if r.size < 3:
        return float("nan")
    sd = float(r.std(ddof=1))
    if sd <= 0:
        return float("nan")
    return float(r.mean() / sd * math.sqrt(12))


def _sortino(rets: np.ndarray) -> float:
    r = _arr(rets)
    if r.size < 3:
        return float("nan")
    dn = r[r < 0]
    dd = float(dn.std(ddof=1)) if dn.size > 1 else 0.0
    if dd <= 0:
        return float("nan")
    return float(r.mean() / dd * math.sqrt(12))


def _max_dd_from_rets(rets: np.ndarray) -> float:
    r = _arr(rets)
    if r.size == 0:
        return float("nan")
    nav = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1.0).min())


def _calmar(cagr: float, mdd: float) -> float:
    if mdd != mdd or abs(mdd) < 1e-12:
        return float("nan")
    return float(cagr / abs(mdd))


def _moments(rets: np.ndarray) -> tuple[float, float]:
    r = _arr(rets)
    if r.size < 4:
        return 0.0, 3.0
    s = pd.Series(r)
    return float(s.skew()), float(s.kurtosis() + 3.0)


def _spec_metrics(periods: list[dict], key: str = "net") -> dict[str, Any]:
    rets = _arr([p.get(key) for p in periods])
    gross = _arr([p.get("gross") for p in periods])
    costs = _arr([p.get("cost") for p in periods])
    turns = _arr([p.get("one_way_turnover") for p in periods])
    if rets.size == 0:
        return {"n": 0}
    years = rets.size / 12.0
    cagr = _ann_from_monthly(rets)
    cagr_g = _ann_from_monthly(gross)
    mdd = _max_dd_from_rets(rets)
    return {
        "n": int(rets.size),
        "years": years,
        "cagr_net": cagr,
        "cagr_gross": cagr_g,
        "vol": _vol(rets),
        "sharpe": _sharpe(rets),
        "sortino": _sortino(rets),
        "max_dd": mdd,
        "calmar": _calmar(cagr, mdd),
        "win_months": float((rets > 0).mean()),
        "best_month": float(rets.max()),
        "worst_month": float(rets.min()),
        "turnover_per_year": float(turns.mean() * 12) if turns.size else float("nan"),
        "cost_drag_per_year": float(costs.mean() * 12) if costs.size else float("nan"),
        "avg_names": float(np.mean([p.get("n_filled") or p.get("n_picks") or 0 for p in periods])),
        "avg_max_sector": float(np.mean([p.get("max_sector_weight") or 0 for p in periods])),
        "max_max_sector": float(np.max([p.get("max_sector_weight") or 0 for p in periods])),
        "mean_monthly": float(rets.mean()),
    }


def _by_year(periods: list[dict], key: str = "net") -> dict[str, float]:
    bags: dict[str, list[float]] = defaultdict(list)
    for p in periods:
        y = str(p.get("rebalance") or "")[:4]
        if p.get(key) is not None:
            bags[y].append(float(p[key]))
    out = {}
    for y, rs in sorted(bags.items()):
        nav = float(np.prod(1.0 + np.asarray(rs)))
        out[y] = nav - 1.0
    return out


def _block_slice(periods: list[dict], block: str) -> list[dict]:
    return [p for p in periods if block_of(str(p.get("rebalance") or "")) == block]


def _align(a: list[dict], b: list[dict], ka="net", kb="gross") -> tuple[np.ndarray, np.ndarray]:
    mb = {x["rebalance"]: x for x in b}
    xs, ys = [], []
    for p in a:
        q = mb.get(p["rebalance"])
        if q is None or p.get(ka) is None or q.get(kb) is None:
            continue
        xs.append(float(p[ka]))
        ys.append(float(q[kb]))
    return _arr(xs), _arr(ys)


def _beta(y: np.ndarray, x: np.ndarray) -> float:
    if y.size < 3 or x.size != y.size:
        return float("nan")
    vx = float(np.var(x, ddof=1))
    if vx <= 0:
        return float("nan")
    return float(np.cov(y, x, ddof=1)[0, 1] / vx)


def _decile_table(rows: list[dict]) -> dict[str, Any]:
    by: dict[int, list[float]] = defaultdict(list)
    nobs: dict[int, int] = defaultdict(int)
    excess: dict[int, list[float]] = defaultdict(list)
    meds: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        b = int(r["bucket"])
        by[b].append(float(r["mean"]))
        nobs[b] += int(r.get("n") or 0)
        if r.get("excess_vs_universe") is not None:
            excess[b].append(float(r["excess_vs_universe"]))
        if r.get("median") is not None:
            meds[b].append(float(r["median"]))
    table = []
    means = []
    for b in range(1, 11):
        xs = _arr(by.get(b, []))
        if xs.size == 0:
            continue
        ci = block_bootstrap_mean_ci(xs, n_boot=1000, seed=7 + b)
        means.append((b, float(xs.mean())))
        table.append({
            "decile": b,
            "n_rebalances": int(xs.size),
            "n_observations": int(nobs.get(b, 0)),
            "mean": float(xs.mean()),
            "median": float(np.median(meds[b])) if meds.get(b) else float("nan"),
            "excess_vs_universe": float(np.mean(excess[b])) if excess.get(b) else float("nan"),
            "ci_lower": ci["ci_lower"],
            "ci_upper": ci["ci_upper"],
            "win_rate": float((xs > 0).mean()),
        })
    spearman = float("nan")
    if len(means) >= 5:
        ranks = np.arange(len(means), dtype=float)
        vals = np.asarray([m for _, m in means], dtype=float)
        # decile number vs mean: expect positive if stronger ranks earn more
        dec = np.asarray([b for b, _ in means], dtype=float)
        if float(np.std(vals)) > 0:
            spearman = float(pd.Series(dec).corr(pd.Series(vals), method="spearman"))
    d10 = next((t for t in table if t["decile"] == 10), None)
    d1 = next((t for t in table if t["decile"] == 1), None)
    mid = [t for t in table if t["decile"] in (8, 9)]
    d10_only = bool(
        d10 and d10["mean"] > 0
        and all(t["mean"] <= 0 for t in mid)
        and (spearman != spearman or spearman < 0.40)
    )
    by_year: dict[str, dict[int, float]] = defaultdict(dict)
    bags: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in rows:
        bags[(str(r["rebalance"])[:4], int(r["bucket"]))].append(float(r["mean"]))
    for (y, b), xs in bags.items():
        by_year[y][b] = float(np.mean(xs))
    return {
        "table": table,
        "spearman": spearman,
        "d10_minus_d1": (None if not (d10 and d1) else d10["mean"] - d1["mean"]),
        "d10_only": d10_only,
        "by_year": {y: dict(v) for y, v in sorted(by_year.items())},
    }


def _capacity(primary: list[dict]) -> dict[str, Any]:
    rows = []
    flags = {int(c): 0 for c in CAPITALS}
    n_pos = 0
    for p in primary:
        advs = p.get("advs") or []
        picks = p.get("picks") or []
        n = max(len(picks), int(p.get("n_picks") or PRIMARY_N), 1)
        for cap in CAPITALS:
            pos = float(cap) / n
            for adv in advs:
                n_pos += 1
                if not adv or adv <= 0:
                    flags[int(cap)] += 1
                    continue
                part = pos / float(adv)
                if part >= PARTICIPATION_FLAG:
                    flags[int(cap)] += 1
        if advs:
            rows.append({
                "rebalance": p["rebalance"],
                "min_adv20": float(min(advs)),
                "median_adv20": float(np.median(advs)),
            })
    n_reb = max(len(primary), 1)
    # n_pos counts cap × name × rebalance; per-cap count is names*rebalances
    names_reb = max(sum(len(p.get("advs") or []) for p in primary), 1)
    return {
        "participation_flag": PARTICIPATION_FLAG,
        "capitals": {
            str(c): {
                "flagged_positions": flags[int(c)],
                "flagged_share": flags[int(c)] / names_reb,
            }
            for c in CAPITALS
        },
        "adv_summary": rows,
    }


def _crash(periods: list[dict]) -> dict[str, Any]:
    nets = [(p["rebalance"], float(p["net"])) for p in periods if p.get("net") is not None]
    if not nets:
        return {}
    worst = min(nets, key=lambda x: x[1])
    # rolling 3-month
    worst_q = None
    if len(nets) >= 3:
        best_q = 99
        for i in range(len(nets) - 2):
            q = float(np.prod([1 + nets[i + k][1] for k in range(3)]) - 1)
            if q < best_q:
                best_q = q
                worst_q = {"start": nets[i][0], "ret": q}
    nav = np.cumprod([1 + x[1] for x in nets])
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    trough_i = int(np.argmin(dd))
    peak_i = int(np.argmax(nav[: trough_i + 1])) if trough_i >= 0 else 0
    return {
        "worst_month": {"rebalance": worst[0], "net": worst[1]},
        "worst_quarter": worst_q,
        "max_dd": float(dd.min()),
        "dd_peak": nets[peak_i][0] if nets else None,
        "dd_trough": nets[trough_i][0] if nets else None,
    }


def classify(stats: dict[str, Any]) -> dict[str, Any]:
    """Exactly one label. Rules written against the protocol, not the result tape."""
    prim = stats.get("primary") or {}
    dec = stats.get("deciles", {}).get("M1") or {}
    blocks = stats.get("blocks") or {}
    years = prim.get("by_year_net") or {}
    h3 = stats.get("h3") or {}
    formulas = stats.get("formula_excess_ew") or {}
    failures: list[str] = []
    notes: list[str] = []

    val = blocks.get("validation") or {}
    conf = blocks.get("confirmation") or {}
    valconf_n = int(val.get("n") or 0) + int(conf.get("n") or 0)
    # Combined later-block excess
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
    if spearman is None or spearman != spearman or spearman < 0.35:
        failures.append("deciles_not_ordered")
    if dec.get("d10_only"):
        failures.append("effect_concentrated_in_d10")

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

    pos_formulas = sum(1 for v in formulas.values() if v is not None and v == v and v > 0)
    if formulas and pos_formulas <= 1 and (formulas.get("M1") or 0) > 0:
        failures.append("single_formula_only")

    h3_better = (
        h3.get("excess_cagr_ew") is not None
        and prim.get("excess_cagr_ew") is not None
        and h3["excess_cagr_ew"] > (prim["excess_cagr_ew"] + 0.03)
        and (prim["excess_cagr_ew"] or 0) <= 0
    )

    label = "RESEARCH-ONLY"
    if conf_reverse or h3_better:
        label = "MODIFY HYPOTHESIS"
        notes.append("Structural issue noted; do not retune M1/Top20/monthly inside EDGE-001.")
    hard = {
        "net_does_not_beat_ew_or_nifty",
        "deciles_not_ordered",
        "costs_destroy_edge",
        "confirmation_reverses_development",
    }
    if len(hard & set(failures)) >= 2 or (
        "net_does_not_beat_ew_or_nifty" in failures and "deciles_not_ordered" in failures
    ):
        label = "REJECT"
    elif (
        beat_ew
        and spearman is not None and spearman == spearman and spearman >= 0.50
        and not conf_reverse
        and "costs_destroy_edge" not in failures
        and "one_year_dominates" not in failures
        and not dec.get("d10_only")
        and valconf_n >= 12
    ):
        label = "PROMISING — FORWARD VALIDATION WARRANTED"
    elif failures:
        label = "RESEARCH-ONLY" if label != "MODIFY HYPOTHESIS" and label != "REJECT" else label

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
    mom = json.loads((artifacts / "prod_momentum_compare.json").read_text())
    regime = []
    rp = artifacts / "regime_periods.json"
    if rp.exists():
        regime = json.loads(rp.read_text())
    snaps = []
    sp = artifacts / "universe_snapshots.json"
    if sp.exists():
        snaps = json.loads(sp.read_text())

    prim = periods.get("primary_M1_top20_monthly") or []
    pm = _spec_metrics(prim, "net")
    pm.update({
        "by_year_net": _by_year(prim, "net"),
        "by_year_gross": _by_year(prim, "gross"),
    })
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
    excess_nf = _arr([b.get("excess_net_vs_nifty") for b in bench])
    inference = {}
    if excess_ew.size >= 3:
        sk, ku = _moments(excess_ew)
        sh = _sharpe(excess_ew)
        inference["excess_ew"] = {
            "mean": float(excess_ew.mean()),
            "ci": block_bootstrap_mean_ci(excess_ew, n_boot=2000, seed=17),
            "sharpe": sh,
            "psr": probabilistic_sharpe_ratio(sh, int(excess_ew.size), sk, ku, 0.0),
            "dsr": deflated_sharpe_ratio(sh, int(excess_ew.size), sk, ku, n_trials=DSR_N_TRIALS),
            "p_beat_ew": float((excess_ew > 0).mean()),
        }
        ev = evaluate(excess_ew, n_trials=DSR_N_TRIALS, require_block_ci=True, block_ci_seed=17)
        inference["harness_excess_ew"] = {
            "verdict": ev.verdict, "n": ev.n, "mean_r": ev.mean_r,
            "sharpe": ev.sharpe, "psr": ev.psr, "dsr": ev.dsr, "insight": ev.insight,
        }
    if excess_nf.size >= 3:
        inference["excess_nifty"] = {
            "mean": float(excess_nf.mean()),
            "ci": block_bootstrap_mean_ci(excess_nf, n_boot=2000, seed=19),
            "p_beat_nifty": float((excess_nf > 0).mean()),
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

    sensitivities = {k: _spec_metrics(v, "net") for k, v in periods.items() if k != "primary_M1_top20_monthly"}
    formula_excess = {}
    for key, col in (("M1", "primary_M1_top20_monthly"), ("M2", "sens_M2_top20_monthly"),
                     ("M3", "sens_M3_top20_monthly"), ("M4", "sens_M4_top20_monthly")):
        sl = periods.get(col) or []
        nr = _arr([p.get("net") for p in sl])
        # align to ew via rebalance
        mp = {p["rebalance"]: p for p in sl}
        aligned = []
        ewa = []
        for e in ew:
            if e["rebalance"] in mp:
                aligned.append(mp[e["rebalance"]]["net"])
                ewa.append(e["gross"])
        if aligned:
            formula_excess[key] = _ann_from_monthly(_arr(aligned)) - _ann_from_monthly(_arr(ewa))

    h3_periods = periods.get("sens_H3_top20_monthly") or []
    h3 = _spec_metrics(h3_periods, "net")
    h3_aligned, ew_h3 = [], []
    mh = {p["rebalance"]: p for p in h3_periods}
    for e in ew:
        if e["rebalance"] in mh:
            h3_aligned.append(mh[e["rebalance"]]["net"])
            ew_h3.append(e["gross"])
    if h3_aligned:
        h3["excess_cagr_ew"] = _ann_from_monthly(_arr(h3_aligned)) - _ann_from_monthly(_arr(ew_h3))

    mom_m = _spec_metrics(mom, "net")
    mm = {p["rebalance"]: p for p in mom}
    ma, ea = [], []
    for e in ew:
        if e["rebalance"] in mm:
            ma.append(mm[e["rebalance"]]["net"])
            ea.append(e["gross"])
    if ma:
        mom_m["excess_cagr_ew"] = _ann_from_monthly(_arr(ma)) - _ann_from_monthly(_arr(ea))
        mom_m["turnover_per_year"] = float(np.mean([p.get("one_way_turnover") or 0 for p in mom]) * 12)

    deciles = {
        "M1": _decile_table(dec_raw.get("M1_decile") or []),
        "M2": _decile_table(dec_raw.get("M2_decile") or []),
        "M3": _decile_table(dec_raw.get("M3_decile") or []),
        "M1_quintile": _decile_table(dec_raw.get("M1_quintile") or []),
    }

    by_reg: dict[str, list[float]] = defaultdict(list)
    for r in regime:
        st = str(r.get("regime") or "UNKNOWN")
        if st in ("STRONG_BULL", "BULL"):
            bucket = "bull"
        elif st == "SIDEWAYS":
            bucket = "sideways"
        elif st == "CORRECTION":
            bucket = "correction"
        elif st == "BEAR":
            bucket = "bear"
        else:
            bucket = "unknown"
        if r.get("net") is not None:
            by_reg[bucket].append(float(r["net"]))
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
        "h3": h3,
        "deciles": deciles,
        "inference": inference,
        "crash": _crash(prim),
        "capacity": _capacity(prim),
        "prod_momentum": mom_m,
        "regime": regime_out,
        "universe": {
            "n_snapshots": len(snaps),
            "avg_candidates": float(np.mean([s.get("candidate_count") or 0 for s in snaps])) if snaps else None,
            "avg_investable": float(np.mean([s.get("investable_count") or 0 for s in snaps])) if snaps else None,
            "avg_ranked": float(np.mean([s.get("ranked_count") or 0 for s in snaps])) if snaps else None,
        },
    }
    stats["decision"] = classify(stats)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (artifacts / "edge_001_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    (OUT_DIR / "edge_001_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    (OUT_DIR / "edge_001_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return stats
