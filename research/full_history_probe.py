"""Bounded, offline probe of the full-history walk-forward on an SSD COPY of the store.

Never touches the live runtime. It reads a copied ``store_cache.pkl`` (and
the copied ``ca_events.json`` via QT_RUNTIME_ROOT), runs the PRODUCTION
walk-forward (``scan.signal_backtest.run_backtest``) on a small evenly spaced
symbol sample at several lookbacks, and measures:

* wall-clock and scanner evaluations (so the 800-symbol cost can be extrapolated);
* per-signal trades and independent 10-session blocks (overlap-robust);
* windows skipped because they were not exactly 10 official sessions;
* peak resident memory.

    QT_RUNTIME_ROOT=<copy root> python -m research.full_history_probe --store <copy>/logs/bhav/store_cache.pkl \
        --symbols 40 --lookbacks 250,1288,1690 --out docs/research/full_history_probe_2026-11-20.json

It spends research alpha-ledger looks in the COPY's runtime only (QT_RUNTIME_ROOT).
"""

from __future__ import annotations

import argparse
import json
import pickle
import resource
import time
from datetime import date


def run(store_path: str, symbols: int, lookbacks: list[int]) -> dict:
    import data.bhavcopy_store as BS
    import scan.signal_backtest as SB
    from data.corporate_actions import adjust_frame, load_events

    store = pickle.load(open(store_path, "rb"))["store"]
    events = load_events()
    all_dates = sorted({d for df in store.values() for d in df.index.strftime("%Y-%m-%d")})

    def get_ohlcv(sym):
        df = store.get(str(sym).upper())
        if df is None:
            return None
        df = df.copy()
        ev = events.get(str(sym).upper())
        return adjust_frame(df, ev, copy=False) if ev else df

    BS.store_symbols = lambda: sorted(store)
    BS.get_ohlcv = get_ohlcv
    BS._dates_on_disk = lambda: [date.fromisoformat(d) for d in all_dates]
    SB.MIN_SYMBOLS_TO_SAVE = 10 ** 9                     # never write a report into any runtime

    from scan.unified_scanner import UnifiedScanner
    calls = {"n": 0}
    real = UnifiedScanner._analyze

    def counting(self, *a, **k):
        calls["n"] += 1
        return real(self, *a, **k)

    UnifiedScanner._analyze = counting
    out = {"store_sessions": len(all_dates), "first_session": all_dates[0], "last_session": all_dates[-1],
           "store_symbols": len(store), "sample_symbols": symbols, "ca_event_symbols": len(events), "runs": {}}
    for lb in lookbacks:
        calls["n"] = 0
        t0 = time.perf_counter()
        rep = SB.run_backtest(lookback_sessions=lb, max_symbols=symbols)
        elapsed = time.perf_counter() - t0
        prov = rep.get("evidence_provenance") or {}
        per_signal = {k: {"trades": v.get("trades"), "blocks": v.get("independent_sessions"),
                          "mean_r": v.get("expectancy_r_precise"), "lower_r": v.get("lower_r"),
                          "inference": v.get("inference")}
                      for k, v in sorted((rep.get("signals") or {}).items(), key=lambda kv: -(kv[1].get("trades") or 0))}
        out["runs"][str(lb)] = {
            "elapsed_s": round(elapsed, 1), "evaluations": calls["n"],
            "ms_per_evaluation": round(1000 * elapsed / max(1, calls["n"]), 2),
            "extrapolated_800_symbols_min": round(elapsed * 800 / symbols / 60, 1),
            "horizon_gap_windows_skipped": prov.get("horizon_gap_windows_skipped"),
            "data_first_session": prov.get("data_first_session"), "sessions_indexed": prov.get("sessions_indexed"),
            "signals": per_signal,
        }
    out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)   # macOS: bytes
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", required=True)
    ap.add_argument("--symbols", type=int, default=40)
    ap.add_argument("--lookbacks", default="250,1288,1690")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    res = run(args.store, args.symbols, [int(x) for x in args.lookbacks.split(",")])
    text = json.dumps(res, indent=2, default=str)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "signals"} for k, v in res["runs"].items()}, indent=2))
    print("peak_rss_mb", res["peak_rss_mb"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
