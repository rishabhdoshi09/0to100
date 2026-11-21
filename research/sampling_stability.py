"""QT-08: is the walk-forward's decision sample stable when data is APPENDED? (research only)

The production walk-forward (``scan.signal_backtest.run_backtest``) measures
every ``sample_step``-th decision day from a start that depends on each
symbol's bar count (``DECISION_GRID = rolling``). Appending bars moves that
start, so the next report measures a DIFFERENT subset of the same historical
decisions. This module measures what that does, with the real production
walk-forward on an offline SSD copy:

* cut the copy k = 0..K sessions before its last session (every earlier bar is
  identical, the cut only removes the future: an append-only history);
* run the production walk-forward on each cut with the production rolling
  lookback (250) and stride (5), and with the session-anchored grid;
* compare, on the decision window every cut can measure, which decisions were
  selected, whether a decision measured by two cuts has the identical outcome
  (point-in-time consistency), and how much the per-signal estimates move from
  phase alone against their own standard error.

The symbol set is held fixed (``fixed_symbols=True``) to isolate the grid. With
``fixed_symbols=False`` the cut drops symbols that list later, as production
would see, which also moves the even-stride symbol sample.

No report is written, no alpha look is reserved (edge_diagnosis.collect), no
network. Refuses the live runtime.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


def cut_frames(frames: Mapping[str, pd.DataFrame], last_day: str, *, fixed_symbols: bool = True) -> dict:
    """Every bar up to and including ``last_day``; nothing after. Earlier bars are untouched."""
    out = {}
    for sym, df in frames.items():
        kept = df.loc[:pd.Timestamp(last_day)]
        if len(kept) or fixed_symbols:
            out[sym] = kept
    return out


def _cluster_mean_se(rows: list[dict]) -> tuple[float | None, float | None, int]:
    """Mean net R and a decision-day clustered SE (reference only: windows overlap)."""
    if not rows:
        return None, None, 0
    by_day: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_day[str(r["decision"])].append(float(r["net_r"]))
    mean = sum(float(r["net_r"]) for r in rows) / len(rows)
    sums = [sum(v) - len(v) * mean for v in by_day.values()]
    n, g = len(rows), len(by_day)
    se = math.sqrt(sum(x * x for x in sums) * g / max(1, g - 1)) / n if g > 1 else None
    return mean, se, n


def _net_r(trade: Mapping[str, Any]) -> float | None:
    for key in ("net_r", "r_multiple", "realized_R", "net_R"):
        if trade.get(key) is not None:
            return float(trade[key])
    return None


def summarise(runs: Mapping[int, list[dict]], *, min_trades: int = 30) -> dict[str, Any]:
    """Compare the cut runs on the common decision window (see the module doc)."""
    keyed = {k: {(t["symbol"], t["decision"]): t for t in rows} for k, rows in runs.items()}
    starts = [min(d for _s, d in m) for m in keyed.values() if m]
    ends = [max(d for _s, d in m) for m in keyed.values() if m]
    lo, hi = (max(starts), min(ends)) if starts and ends else (None, None)
    window = {k: {key: t for key, t in m.items() if lo <= key[1] <= hi} for k, m in keyed.items()}
    base = window.get(0) or {}
    overlap, outcome_diffs = {}, {}
    for k, m in window.items():
        both = set(base) & set(m)
        union = set(base) | set(m)
        overlap[k] = {"decisions": len(m), "shared_with_k0": len(both),
                      "jaccard_vs_k0": round(len(both) / len(union), 4) if union else None}
        diffs = [abs((_net_r(base[x]) or 0.0) - (_net_r(m[x]) or 0.0)) for x in both]
        outcome_diffs[k] = round(max(diffs), 9) if diffs else 0.0
    per_signal: dict[str, dict[int, tuple]] = defaultdict(dict)
    for k, m in window.items():
        by_sig: dict[str, list[dict]] = defaultdict(list)
        for t in m.values():
            r = _net_r(t)
            if r is None:
                continue
            for sig in t.get("signals") or []:
                by_sig[sig].append({"decision": t["decision"], "net_r": r})
            by_sig["ANY_SIGNAL"].append({"decision": t["decision"], "net_r": r})
        for sig, rows in by_sig.items():
            per_signal[sig][k] = _cluster_mean_se(rows)
    stability = {}
    for sig, by_k in sorted(per_signal.items()):
        means = [m for (m, _se, n) in by_k.values() if m is not None and n >= min_trades]
        ses = [se for (_m, se, n) in by_k.values() if se is not None and n >= min_trades]
        if len(means) < 2 or not ses:
            continue
        sd = statistics.pstdev(means)
        se = statistics.mean(ses)
        lowers = [m - 1.645 * s for (m, s, n) in by_k.values() if m is not None and s is not None and n >= min_trades]
        stability[sig] = {"phases": len(means), "mean_of_means": round(statistics.mean(means), 4),
                          "between_phase_sd": round(sd, 4), "mean_se": round(se, 4),
                          "sd_over_se": round(sd / se, 3) if se else None,
                          "lower_bound_sign_flips": bool(lowers and min(lowers) < 0 < max(lowers)),
                          "trades_by_phase": {str(k): n for k, (_m, _s, n) in sorted(by_k.items())}}
    return {"common_window": [lo, hi], "overlap": {str(k): v for k, v in sorted(overlap.items())},
            "max_outcome_difference_on_shared_decisions": {str(k): v for k, v in sorted(outcome_diffs.items())},
            "per_signal": stability}


def run(root: str | Path, *, symbols: int, cuts: Iterable[int] = range(5), grid: str = "rolling",
        lookback: int = 250, step: int = 5, fixed_symbols: bool = True,
        accept_unbound_coverage: bool = False) -> dict[str, Any]:
    from research.edge_diagnosis import collect, load_copy
    from scan import signal_backtest as SB

    copy = load_copy(root, accept_unbound_coverage=accept_unbound_coverage)
    frames = copy["frames"]
    days = sorted({d for df in frames.values() for d in pd.DatetimeIndex(df.index).strftime("%Y-%m-%d")})
    runs, sampled = {}, {}
    for k in cuts:
        fr = cut_frames(frames, days[-1 - int(k)], fixed_symbols=fixed_symbols)
        col = collect(fr, symbols=symbols, sample_step=step, lookback_sessions=lookback, decision_grid=grid)
        runs[int(k)] = [dict(t, net_r=_net_r(t)) for t in col["trades"]]
        sampled[int(k)] = set(col["sampled_symbols"])
    base = sampled.get(0) or set()
    return {"grid": grid, "lookback_sessions": lookback, "sample_step": step, "symbols": symbols,
            "fixed_symbols": fixed_symbols, "cuts": sorted(runs), "last_session": days[-1],
            "symbol_sample_jaccard_vs_k0": {str(k): round(len(base & s) / len(base | s), 4) if (base | s) else None
                                           for k, s in sorted(sampled.items())},
            "decision_grid_constant_in_code": SB.DECISION_GRID,
            **summarise(runs)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--symbols", type=int, default=200)
    ap.add_argument("--cuts", type=int, default=5)
    ap.add_argument("--grid", default="rolling", choices=["rolling", "session_anchored"])
    ap.add_argument("--lookback", type=int, default=250)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--realistic-symbols", action="store_true", help="let the cut drop later listings")
    ap.add_argument("--accept-unbound-coverage", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rep = run(a.root, symbols=a.symbols, cuts=range(a.cuts), grid=a.grid, lookback=a.lookback, step=a.step,
              fixed_symbols=not a.realistic_symbols, accept_unbound_coverage=a.accept_unbound_coverage)
    Path(a.out).write_text(json.dumps(rep, indent=2, default=str))
    print(json.dumps({k: rep[k] for k in ("grid", "common_window", "overlap", "symbol_sample_jaccard_vs_k0")},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
