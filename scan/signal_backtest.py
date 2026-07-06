"""
Signal Backtest — measures how accurate each signal type ACTUALLY is.

Walk-forward over the bhavcopy store (no lookahead):
  For sampled historical days t, run the unified detector on data[:t]
  only, then check what happened over the next `horizon` sessions:
    WIN  — target (entry + 2×risk) hit before stop
    LOSS — stop hit first
    FLAT — neither hit within the horizon (counted, excluded from WR)

Output per signal type: trades, win rate, avg win/loss, expectancy (R).
Saved to logs/signal_backtest.json so the scanner can show
"historically X% accurate" next to each signal — measured, not promised.

Run from the UI (background thread) or:  python -m scan.signal_backtest
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

from logger import get_logger

log = get_logger(__name__)

_OUT = Path(__file__).resolve().parent.parent / "logs" / "signal_backtest.json"

_state = {"running": False, "progress": 0, "total": 0}
_state_lock = threading.Lock()


def _simulate(entry: float, stop: float, target: float,
              fwd_high: np.ndarray, fwd_low: np.ndarray,
              fwd_close: np.ndarray) -> tuple[str, float]:
    """
    Realistic first-touch simulation: the trade only exists once price
    actually reaches the entry (breakout orders sit above the market).
    Returns (outcome, r_multiple). FLAT trades are marked-to-market at
    the horizon close so expectancy is honest, not survivorship-biased.
    """
    risk = entry - stop
    filled = False
    for h, l, c in zip(fwd_high, fwd_low, fwd_close):
        if not filled:
            if h >= entry:
                filled = True
            else:
                continue
        if l <= stop:
            return "LOSS", -1.0
        if h >= target:
            return "WIN", (target - entry) / risk
    if filled:
        return "FLAT", (float(fwd_close[-1]) - entry) / risk
    return "NO_FILL", 0.0


def run_backtest(sample_step: int = 5, lookback_sessions: int = 250,
                 horizon: int = 10, max_symbols: int = 800) -> dict:
    """
    Walk-forward backtest across the bhav store. Returns and persists
    per-signal stats. Needs ≥ (60 + lookback + horizon) sessions of data.
    """
    from data.bhavcopy_store import store_symbols, get_ohlcv
    from scan.unified_scanner import UnifiedScanner

    sc = UnifiedScanner()
    stats: dict[str, dict] = {}
    symbols = store_symbols()[:max_symbols]

    with _state_lock:
        _state.update(running=True, progress=0, total=len(symbols))

    t0 = time.time()
    for si, sym in enumerate(symbols):
        df = get_ohlcv(sym)
        if df is None or len(df) < 60 + horizon + 20:
            continue
        n = len(df)
        start = max(60, n - horizon - lookback_sessions)
        for t in range(start, n - horizon, sample_step):
            hist = df.iloc[:t]
            try:
                r = sc._analyze(sym, hist)
            except Exception:
                continue
            if r is None or not r.signals:
                continue
            risk = r.entry - r.stop
            if risk <= 0:
                continue
            fwd = df.iloc[t:t + horizon]
            outcome, r_mult = _simulate(r.entry, r.stop, r.target,
                                        fwd["high"].values, fwd["low"].values,
                                        fwd["close"].values)
            if outcome == "NO_FILL":
                continue          # order never triggered — not a trade
            for sig in r.signals:
                s = stats.setdefault(sig, {"trades": 0, "wins": 0, "losses": 0,
                                           "flat": 0, "r_sum": 0.0})
                s["trades"] += 1
                s["r_sum"] += r_mult
                if outcome == "WIN":
                    s["wins"] += 1
                elif outcome == "LOSS":
                    s["losses"] += 1
                else:
                    s["flat"] += 1
        with _state_lock:
            _state["progress"] = si + 1

    # Derive rates & expectancy — true average R across ALL filled trades
    # (wins, losses, AND flats marked at horizon), no survivorship bias.
    report: dict[str, dict] = {}
    for sig, s in stats.items():
        closed = s["wins"] + s["losses"]
        wr = s["wins"] / closed if closed else 0.0
        report[sig] = {
            "trades": s["trades"], "closed": closed,
            "win_rate": round(wr * 100, 1),
            "expectancy_r": round(s["r_sum"] / s["trades"], 2) if s["trades"] else 0.0,
            "flat": s["flat"],
        }

    out = {"generated_at": time.strftime("%Y-%m-%d %H:%M"),
           "symbols": len(symbols), "horizon_days": horizon,
           "signals": report,
           "elapsed_s": round(time.time() - t0, 1)}
    try:
        _OUT.parent.mkdir(parents=True, exist_ok=True)
        _OUT.write_text(json.dumps(out, indent=2))
    except Exception as exc:
        log.warning("backtest_save_failed", error=str(exc))

    with _state_lock:
        _state["running"] = False
    log.info("signal_backtest_done", symbols=len(symbols),
             signals=len(report), elapsed_s=out["elapsed_s"])
    return out


def run_in_background(**kwargs) -> None:
    with _state_lock:
        if _state["running"]:
            return
    threading.Thread(target=run_backtest, kwargs=kwargs,
                     name="signal-backtest", daemon=True).start()


def get_state() -> dict:
    with _state_lock:
        return dict(_state)


def load_report() -> dict | None:
    try:
        if _OUT.exists():
            return json.loads(_OUT.read_text())
    except Exception:
        pass
    return None


def combo_edge(signal_keys: list[str], min_trades: int = 30) -> float | None:
    """
    Measured expectancy (avg R/trade) for a stock's signal combo —
    mean of each signal's backtested expectancy, using only signals
    with enough evidence. None when no backtest / no evidenced signal.
    """
    rep = load_report()
    if not rep:
        return None
    vals = []
    for k in signal_keys:
        s = rep.get("signals", {}).get(k)
        if s and s.get("trades", 0) >= min_trades:
            vals.append(float(s.get("expectancy_r", 0)))
    return round(sum(vals) / len(vals), 2) if vals else None


def accuracy_for(signal_key: str) -> str:
    """'62% WR (145 trades)' for a signal, or '' if no backtest yet."""
    rep = load_report()
    if not rep:
        return ""
    s = rep.get("signals", {}).get(signal_key)
    if not s or s.get("closed", 0) < 20:
        return ""
    return f"{s['win_rate']:.0f}% WR ({s['closed']} trades)"


if __name__ == "__main__":
    from data.bhavcopy_store import build_store
    build_store()
    print(json.dumps(run_backtest(), indent=2))
