"""Executable definition of PR-2026-11-21-MOM12-1: a 12-1 momentum portfolio (research only).

``research/preregistered/pr_2026_11_21_mom12_1.json`` is the frozen authority,
and its SHA-256 is pinned here (the test suite fails on any edit). This module
computes only what the spec says. It runs on an OFFLINE SSD copy, refuses the
live runtime, writes no report into any runtime, spends no alpha look, and has
no path to the BUY gate, the paper book or a broker.

Point-in-time:

* Each rebalance ranks on bars up to the information cutoff (the session
  before the rebalance) and executes at the rebalance close.
* Membership is as of the cutoff, and unknown membership is excluded.
* The price floor uses the traded price, not the back-adjusted one.
* A rebalance is evaluated only when its formation window and holding period
  lie inside the verified corporate-action coverage.

The held-out year is refused unless ``period="held_out"`` is passed together
with ``single_look=True``: it is evaluated once, after discovery is committed.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from core.live_volume import on_live_volume

SPEC_PATH = Path(__file__).resolve().parent / "preregistered" / "pr_2026_11_21_mom12_1.json"
#: SHA-256 of the spec bytes when it was frozen (2026-11-21). Never update it to "fix" a test.
FROZEN_SHA256 = "86a959d2dde8424537e7e6e028a426f80fddaa61c3338dbe347bc9e84fdc106f"


def spec_sha256(path: Path = SPEC_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _multipliers(dates: Sequence[str], events: Sequence[tuple[str, float]] | None) -> np.ndarray:
    """Traded price = adjusted price x multiplier[d] (product of factors of events after d)."""
    m = np.ones(len(dates))
    for ex, factor in events or ():
        m[np.asarray(dates) < str(ex)[:10]] *= float(factor)
    return m


def build_panel(frames: Mapping[str, pd.DataFrame], events: Mapping[str, Sequence] | None = None) -> dict[str, Any]:
    """Close, traded value and traded-price panels on the union of official session dates."""
    dates = sorted({d for df in frames.values() for d in pd.DatetimeIndex(df.index).strftime("%Y-%m-%d")})
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    syms = sorted(frames)
    close = np.full((len(dates), len(syms)), np.nan)
    value = np.full_like(close, np.nan)
    traded = np.full_like(close, np.nan)
    for j, s in enumerate(syms):
        df = frames[s]
        c = df["close"].astype(float).reindex(idx)
        close[:, j] = c.to_numpy()
        if "volume" in df.columns:
            value[:, j] = (df["close"].astype(float) * df["volume"].astype(float)).reindex(idx).to_numpy()
        ev = [(str(pd.Timestamp(e["ex_date"]).date()), float(e["factor"])) for e in (events or {}).get(s, [])]
        traded[:, j] = close[:, j] * _multipliers(dates, ev)
    return {"dates": dates, "symbols": syms, "close": close, "value": value, "traded": traded}


def rebalance_indices(dates: Sequence[str]) -> list[int]:
    """Index of the first official session of every calendar month."""
    out, month = [], ""
    for i, d in enumerate(dates):
        if d[:7] != month:
            out.append(i)
            month = d[:7]
    return out


def _is_member(member: Mapping[str, tuple[str, str | None]] | None, sym: str, day: str) -> bool:
    row = (member or {}).get(sym)
    if row is None:
        return False
    listed, delisted = row
    return listed <= day and (delisted is None or delisted > day)


def eligible_scores(panel: Mapping[str, Any], i: int, spec: Mapping[str, Any],
                    member: Mapping[str, tuple[str, str | None]] | None) -> dict[int, float]:
    """{symbol column: 12-1 score} for the rebalance at index i, using bars up to i-1 only."""
    u, sc = spec["universe"], spec["score"]
    c = i - 1
    look = int(u["formation_sessions"])
    if c - look < 0:
        return {}
    close, value, traded = panel["close"], panel["value"], panel["traded"]
    day = panel["dates"][c]
    out: dict[int, float] = {}
    _ = sc
    for j, sym in enumerate(panel["symbols"]):
        if not _is_member(member, sym, day):
            continue
        window = close[c - look:c + 1, j]
        if np.count_nonzero(~np.isnan(window)) < int(u["min_bars_in_formation"]):
            continue
        p_now, p_skip, p_start = close[c, j], close[c - 21, j], close[c - look, j]
        if not (np.isfinite(p_now) and np.isfinite(p_skip) and np.isfinite(p_start) and p_start > 0):
            continue
        if not (traded[c, j] >= float(u["min_traded_price_inr"])):
            continue
        tv = value[max(0, c - 19):c + 1, j]
        tv = tv[np.isfinite(tv)]
        if tv.size < 10 or float(np.median(tv)) < float(u["min_median_traded_value_20d_inr"]):
            continue
        out[j] = float(p_skip / p_start - 1.0)
    return out


def holding_return(close: np.ndarray, j: int, i: int, k: int, *, remainder_loss: float = 0.0) -> tuple[float, bool]:
    """(return from close[i] to close[k], stopped) for column j; NaN when not executable at i.

    A name with no bar at k stopped trading during the holding: its return to
    its last traded close is used, then cash (primary), or a further
    ``remainder_loss`` on what is left (sensitivity). Never priced at a bar
    that did not trade.
    """
    entry = close[i, j]
    if not (np.isfinite(entry) and entry > 0):
        return math.nan, False
    seg = close[i:k + 1, j]
    if np.isfinite(seg[-1]):
        return float(seg[-1] / entry - 1.0), False
    last = seg[np.isfinite(seg)][-1]
    gross = float(last / entry)
    return float(gross * (1.0 - remainder_loss) - 1.0), True


def _hac_t(x: np.ndarray, lag: int = 1) -> tuple[float, float, float]:
    """(mean, Newey-West SE, t) of a series."""
    n = len(x)
    if n < 3:
        return float(np.mean(x)) if n else math.nan, math.nan, math.nan
    mu = float(np.mean(x))
    u = x - mu
    s = float(np.dot(u, u)) / n
    for L in range(1, lag + 1):
        s += 2.0 * (1.0 - L / (lag + 1)) * float(np.dot(u[L:], u[:-L])) / n
    se = math.sqrt(max(s, 0.0) / n)
    return mu, se, (mu / se if se > 0 else math.nan)


def evaluate(panel: Mapping[str, Any], spec: Mapping[str, Any], *, member, coverage: tuple[str | None, str | None],
             period: str = "discovery", single_look: bool = False, remainder_loss: float = 0.0,
             sectors: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Monthly portfolio vs equal-weight benchmark over the spec's period. Research only."""
    if period not in ("discovery", "held_out"):
        raise ValueError("period must be 'discovery' or 'held_out'")
    if period == "held_out" and not single_look:
        raise PermissionError("the held-out year is evaluated ONCE, after discovery is committed (single_look=True)")
    per = spec["periods"][period]
    cover_from, cover_to = coverage
    dates, close = panel["dates"], panel["close"]
    rebs = rebalance_indices(dates)
    top_frac = 0.2
    per_side = float(spec["costs"]["per_side_pct"]) / 100.0
    rows, prev_w = [], {}
    stopped_names = 0
    stopped_events: list[dict[str, Any]] = []
    look = int(spec["universe"]["formation_sessions"])
    for a, b in zip(rebs, rebs[1:]):
        entry_day = dates[a]
        lo = per.get("entry_on_or_after")
        if lo and lo[:4].isdigit() and entry_day < lo:
            continue
        if entry_day >= per["entry_before"]:
            continue
        if a - 1 - look < 0:
            continue
        if not cover_from or dates[a - 1 - look] < cover_from or (cover_to and dates[b] > cover_to):
            continue                       # formation or holding outside verified CA coverage
        scores = eligible_scores(panel, a, spec, member)
        execu = {j: s for j, s in scores.items() if np.isfinite(close[a, j])}
        if len(execu) < int(spec["portfolio"]["min_names"]) * 5:
            continue
        ranked = sorted(execu, key=lambda j: (-execu[j], panel["symbols"][j]))
        picks = ranked[:max(int(spec["portfolio"]["min_names"]), int(len(ranked) * top_frac))]
        rets = {}
        for j in execu:
            r, stopped = holding_return(close, j, a, b, remainder_loss=remainder_loss)
            if np.isfinite(r):
                rets[j] = r
                if stopped and j in picks:
                    stopped_names += 1
                    later = close[b + 1:, j]
                    stopped_events.append({"symbol": panel["symbols"][j], "entry": entry_day, "exit": dates[b],
                                           "resumes_later": bool(np.isfinite(later).any())})
        held = [j for j in picks if j in rets]
        bench = float(np.mean([rets[j] for j in rets])) if rets else math.nan
        w = {j: 1.0 / len(held) for j in held}
        turnover = sum(abs(w.get(j, 0.0) - prev_w.get(j, 0.0)) for j in set(w) | set(prev_w))
        gross = float(np.mean([rets[j] for j in held])) if held else math.nan
        net = gross - per_side * turnover
        # Drifted weights carried into the next rebalance's turnover.
        grown = {j: w[j] * (1.0 + rets[j]) for j in held}
        tot = sum(grown.values()) or 1.0
        prev_w = {j: v / tot for j, v in grown.items()}
        sector_share = 0.0
        if sectors and held:
            counts: dict[str, int] = {}
            for j in held:
                s = sectors.get(panel["symbols"][j], "UNMAPPED")
                counts[s] = counts.get(s, 0) + 1
            sector_share = max(counts.values()) / len(held)
        rows.append({"entry": entry_day, "exit": dates[b], "names": len(held), "universe": len(rets),
                     "gross": gross, "net": net, "benchmark": bench, "excess": net - bench,
                     "turnover_one_way": turnover / 2.0, "max_sector_share": round(sector_share, 3)})
    ex = np.array([r["excess"] for r in rows], dtype=float)
    net = np.array([r["net"] for r in rows], dtype=float)
    bm = np.array([r["benchmark"] for r in rows], dtype=float)
    mu, se, t = _hac_t(ex) if len(ex) else (math.nan, math.nan, math.nan)
    from scipy.stats import t as student_t
    p = float(student_t.sf(t, max(1, len(ex) - 1))) if np.isfinite(t) else math.nan
    beta = alpha = math.nan
    if len(net) >= 3 and np.var(bm) > 0:
        beta = float(np.cov(net, bm, ddof=1)[0, 1] / np.var(bm, ddof=1))
        alpha = float(np.mean(net) - beta * np.mean(bm))
    equity = np.cumprod(1.0 + net) if len(net) else np.array([1.0])
    dd = float(np.min(equity / np.maximum.accumulate(equity) - 1.0)) if len(net) else math.nan
    return {
        "spec_id": spec["id"], "spec_sha256": spec_sha256(), "period": period, "months": len(rows),
        "first_entry": rows[0]["entry"] if rows else None, "last_exit": rows[-1]["exit"] if rows else None,
        "primary": {"mean_monthly_net_excess": mu, "hac_se": se, "t": t, "p_one_sided": p,
                    "ci95": [mu - 1.96 * se, mu + 1.96 * se] if np.isfinite(se) else None,
                    "verdict": ("PASS" if np.isfinite(p) and p < 0.05 and mu > 0 else "NOT_SIGNIFICANT")},
        "secondary": {"mean_monthly_net": float(np.mean(net)) if len(net) else math.nan,
                      "mean_monthly_benchmark": float(np.mean(bm)) if len(bm) else math.nan,
                      "alpha_monthly": alpha, "beta": beta, "max_drawdown": dd,
                      "worst_month_net": float(np.min(net)) if len(net) else math.nan,
                      "hit_rate_excess": float(np.mean(ex > 0)) if len(ex) else math.nan,
                      "mean_turnover_one_way": float(np.mean([r["turnover_one_way"] for r in rows])) if rows else math.nan,
                      "mean_names": float(np.mean([r["names"] for r in rows])) if rows else math.nan,
                      "mean_max_sector_share": float(np.mean([r["max_sector_share"] for r in rows])) if rows else math.nan,
                      "held_names_stopped_trading": stopped_names,
                      # A series that resumes later was a suspension or a data gap; one that never
                      # resumes is a delisting OR a symbol rename (no rename map in the store).
                      "stopped_resumed_later": sum(1 for e in stopped_events if e["resumes_later"]),
                      "stopped_never_resumed": sum(1 for e in stopped_events if not e["resumes_later"])},
        "stopped_events": stopped_events,
        "remainder_loss_assumed": remainder_loss,
        "coverage": {"from": cover_from, "to": cover_to},
        "rows": rows,
    }


def run(root: str | Path, *, period: str = "discovery", single_look: bool = False,
        accept_unbound_coverage: bool = False) -> dict[str, Any]:
    """Evaluate the frozen spec on an SSD copy (primary plus the 50% / 100% stopped-name sensitivities)."""
    from research.edge_diagnosis import coverage_span, load_copy, membership

    if spec_sha256() != FROZEN_SHA256:
        raise RuntimeError("the pre-registered spec changed after freezing: refusing to evaluate")
    copy = load_copy(root, accept_unbound_coverage=accept_unbound_coverage)
    from data.corporate_actions import load_events
    from core.runtime_paths import logs_path
    events = load_events(logs_path("ca_events.json"))
    spec = load_spec()
    panel = build_panel(copy["frames"], events)
    member = membership(copy["universe_rows"])
    try:
        from scan.sector_heat import _load_map
        sectors = _load_map()
    except Exception:
        sectors = {}
    cov = coverage_span(copy["coverage"])
    out = evaluate(panel, spec, member=member, coverage=cov, period=period, single_look=single_look,
                   sectors=sectors)
    out["sensitivity_stopped_names"] = {
        str(x): evaluate(panel, spec, member=member, coverage=cov, period=period, single_look=single_look,
                         remainder_loss=x, sectors=sectors)["primary"]
        for x in (0.5, 1.0)}
    out["ca_coverage_binding"] = copy.get("coverage_binding")
    out["ca_coverage_accepted_unbound"] = bool(accept_unbound_coverage and copy.get("coverage_binding") == "UNBOUND")
    return out


def main(argv=None) -> int:
    import argparse
    import os

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--period", default="discovery", choices=("discovery", "held_out"))
    ap.add_argument("--single-look", action="store_true", help="required for the held-out year (evaluated once)")
    ap.add_argument("--accept-unbound-coverage", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if on_live_volume(args.root):
        raise SystemExit(f"refusing {args.root}: copy the store to the internal SSD first")
    root = Path(args.root).resolve()
    if on_live_volume(root):
        raise SystemExit(f"refusing {root}: copy the store to the internal SSD first")
    os.environ["QT_RUNTIME_ROOT"] = str(root)
    rep = run(root, period=args.period, single_look=args.single_look,
              accept_unbound_coverage=args.accept_unbound_coverage)
    text = json.dumps(rep, indent=1, default=str)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("months", "first_entry", "last_exit", "primary", "secondary",
                                          "sensitivity_stopped_names", "ca_coverage_binding")}, indent=1,
                     default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
