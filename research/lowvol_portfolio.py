"""Executable definition of PR-2026-11-21-LOWVOL: a low-volatility portfolio (research only).

The spec (``research/preregistered/pr_2026_11_21_lowvol.json``) was frozen and
committed BEFORE any return of this score was computed; its SHA-256 is pinned
here and an edited spec refuses to evaluate. Shares the point-in-time panel,
rebalance calendar, holding-return rule and missing-data treatment of
``research.momentum_portfolio``; differs only in the score (lowest trailing
volatility) and the PRIMARY outcome (Newey-West alpha versus the equal-weight
market). No BUY, no order, no gate change can come from this module.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.momentum_portfolio import _is_member, build_panel, holding_return, rebalance_indices

SPEC_PATH = Path(__file__).resolve().parent / "preregistered" / "pr_2026_11_21_lowvol.json"
#: SHA-256 of the frozen spec at its pre-registration commit.
FROZEN_SHA256 = "e1e21ebb8dfb63866b52a352efcb8ad09b7cd09feb73b3b2c04cd58cc232f78d"


def spec_sha256(path: Path = SPEC_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def eligible_scores(panel: Mapping[str, Any], i: int, spec: Mapping[str, Any],
                    member: Mapping[str, tuple[str, str | None]] | None) -> dict[int, float]:
    """{column: -trailing volatility} for the rebalance at index i, from bars up to i-1 only."""
    u = spec["universe"]
    c = i - 1
    look = int(u["formation_sessions"])
    if c - look < 0:
        return {}
    close, value, traded = panel["close"], panel["value"], panel["traded"]
    day = panel["dates"][c]
    out: dict[int, float] = {}
    for j, sym in enumerate(panel["symbols"]):
        if not _is_member(member, sym, day):
            continue
        window = close[c - look:c + 1, j]
        with np.errstate(divide="ignore", invalid="ignore"):
            rets = np.diff(np.log(window))
        rets = rets[np.isfinite(rets)]
        if rets.size < int(u["min_bars_in_formation"]):
            continue
        if not (np.isfinite(close[c, j]) and traded[c, j] >= float(u["min_traded_price_inr"])):
            continue
        tv = value[max(0, c - 19):c + 1, j]
        tv = tv[np.isfinite(tv)]
        if tv.size < 10 or float(np.median(tv)) < float(u["min_median_traded_value_20d_inr"]):
            continue
        out[j] = -float(np.std(rets, ddof=1))
    return out


def newey_west_alpha(y: np.ndarray, x: np.ndarray, lag: int = 1) -> dict[str, float]:
    """OLS y = a + b x with a Newey-West (Bartlett, ``lag``) covariance: alpha, beta, se, t."""
    n = len(y)
    if n < 6 or np.var(x) <= 0:
        return {"alpha": math.nan, "beta": math.nan, "se_alpha": math.nan, "t": math.nan, "n": n}
    X = np.column_stack([np.ones(n), x])
    xtx_inv = np.linalg.inv(X.T @ X)
    coef = xtx_inv @ X.T @ y
    u = y - X @ coef
    g = X * u[:, None]
    s = g.T @ g
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        gamma = g[k:].T @ g[:-k]
        s += w * (gamma + gamma.T)
    v = xtx_inv @ s @ xtx_inv
    se = float(math.sqrt(v[0, 0])) if v[0, 0] > 0 else math.nan
    return {"alpha": float(coef[0]), "beta": float(coef[1]), "se_alpha": se,
            "t": float(coef[0] / se) if se and np.isfinite(se) else math.nan, "n": n}


def evaluate(panel: Mapping[str, Any], spec: Mapping[str, Any], *, member, coverage: tuple[str | None, str | None],
             period: str = "discovery", single_look: bool = False, remainder_loss: float = 0.0) -> dict[str, Any]:
    if period not in ("discovery", "held_out"):
        raise ValueError("period must be 'discovery' or 'held_out'")
    if period == "held_out" and not single_look:
        raise PermissionError("the held-out year is evaluated ONCE, after discovery is committed (single_look=True)")
    per = spec["periods"][period]
    cover_from, cover_to = coverage
    dates, close = panel["dates"], panel["close"]
    per_side = float(spec["costs"]["per_side_pct"]) / 100.0
    look = int(spec["universe"]["formation_sessions"])
    rows, prev_w, stopped = [], {}, 0
    for a, b in zip(rebalance_indices(dates), rebalance_indices(dates)[1:]):
        entry_day = dates[a]
        lo = per.get("entry_on_or_after")
        if lo and lo[:4].isdigit() and entry_day < lo:
            continue
        if entry_day >= per["entry_before"] or a - 1 - look < 0:
            continue
        if not cover_from or dates[a - 1 - look] < cover_from or (cover_to and dates[b] > cover_to):
            continue
        scores = {j: s for j, s in eligible_scores(panel, a, spec, member).items() if np.isfinite(close[a, j])}
        if len(scores) < int(spec["portfolio"]["min_names"]) * 5:
            continue
        ranked = sorted(scores, key=lambda j: (-scores[j], panel["symbols"][j]))     # least volatile first
        picks = ranked[:max(int(spec["portfolio"]["min_names"]), int(len(ranked) * 0.2))]
        rets = {}
        for j in scores:
            r, was_stopped = holding_return(close, j, a, b, remainder_loss=remainder_loss)
            if np.isfinite(r):
                rets[j] = r
                stopped += int(was_stopped and j in picks)
        held = [j for j in picks if j in rets]
        if not held:
            continue
        bench = float(np.mean(list(rets.values())))
        w = {j: 1.0 / len(held) for j in held}
        turnover = sum(abs(w.get(j, 0.0) - prev_w.get(j, 0.0)) for j in set(w) | set(prev_w))
        gross = float(np.mean([rets[j] for j in held]))
        net = gross - per_side * turnover
        grown = {j: w[j] * (1.0 + rets[j]) for j in held}
        tot = sum(grown.values()) or 1.0
        prev_w = {j: v / tot for j, v in grown.items()}
        rows.append({"entry": entry_day, "exit": dates[b], "names": len(held), "universe": len(rets),
                     "net": net, "benchmark": bench, "excess": net - bench, "turnover_one_way": turnover / 2.0})
    net = np.array([r["net"] for r in rows], dtype=float)
    bm = np.array([r["benchmark"] for r in rows], dtype=float)
    reg = newey_west_alpha(net, bm)
    from scipy.stats import t as student_t
    p = float(student_t.sf(reg["t"], max(1, len(net) - 2))) if np.isfinite(reg["t"]) else math.nan
    equity = np.cumprod(1.0 + net) if len(net) else np.array([1.0])
    sharpe = lambda s: float(np.mean(s) / np.std(s, ddof=1) * math.sqrt(12)) if len(s) > 2 and np.std(s) > 0 else math.nan
    return {
        "spec_id": spec["id"], "spec_sha256": spec_sha256(), "period": period, "months": len(rows),
        "first_entry": rows[0]["entry"] if rows else None, "last_exit": rows[-1]["exit"] if rows else None,
        "primary": {"alpha_monthly": reg["alpha"], "nw_se": reg["se_alpha"], "t": reg["t"], "p_one_sided": p,
                    "ci95": ([reg["alpha"] - 1.96 * reg["se_alpha"], reg["alpha"] + 1.96 * reg["se_alpha"]]
                             if np.isfinite(reg["se_alpha"]) else None),
                    "verdict": "PASS" if np.isfinite(p) and p < 0.05 and reg["alpha"] > 0 else "NOT_SIGNIFICANT"},
        "secondary": {"beta": reg["beta"],
                      "mean_monthly_net": float(np.mean(net)) if len(net) else math.nan,
                      "mean_monthly_benchmark": float(np.mean(bm)) if len(bm) else math.nan,
                      "mean_monthly_net_excess": float(np.mean(net - bm)) if len(net) else math.nan,
                      "sharpe_net": sharpe(net), "sharpe_benchmark": sharpe(bm),
                      "max_drawdown": float(np.min(equity / np.maximum.accumulate(equity) - 1.0)) if len(net) else math.nan,
                      "worst_month_net": float(np.min(net)) if len(net) else math.nan,
                      "mean_turnover_one_way": float(np.mean([r["turnover_one_way"] for r in rows])) if rows else math.nan,
                      "mean_names": float(np.mean([r["names"] for r in rows])) if rows else math.nan,
                      "held_names_stopped_trading": stopped},
        "remainder_loss_assumed": remainder_loss,
        "coverage": {"from": cover_from, "to": cover_to},
        "rows": rows,
    }


def run(root: str | Path, *, period: str = "discovery", single_look: bool = False,
        accept_unbound_coverage: bool = False) -> dict[str, Any]:
    from core.runtime_paths import logs_path
    from data.corporate_actions import load_events
    from research.edge_diagnosis import coverage_span, load_copy, membership

    if spec_sha256() != FROZEN_SHA256:
        raise RuntimeError("the pre-registered spec changed after freezing: refusing to evaluate")
    copy = load_copy(root, accept_unbound_coverage=accept_unbound_coverage)
    spec = load_spec()
    panel = build_panel(copy["frames"], load_events(logs_path("ca_events.json")))
    member = membership(copy["universe_rows"])
    cov = coverage_span(copy["coverage"])
    out = evaluate(panel, spec, member=member, coverage=cov, period=period, single_look=single_look)
    out["sensitivity_stopped_names"] = {
        str(x): evaluate(panel, spec, member=member, coverage=cov, period=period, single_look=single_look,
                         remainder_loss=x)["primary"] for x in (0.5, 1.0)}
    out["ca_coverage_binding"] = copy.get("coverage_binding")
    return out


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--period", default="discovery", choices=("discovery", "held_out"))
    ap.add_argument("--single-look", action="store_true")
    ap.add_argument("--accept-unbound-coverage", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rep = run(a.root, period=a.period, single_look=a.single_look, accept_unbound_coverage=a.accept_unbound_coverage)
    Path(a.out).write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps({k: rep[k] for k in ("spec_id", "period", "months", "first_entry", "last_exit", "primary",
                                          "secondary")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
