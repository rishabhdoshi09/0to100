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
import os as _os
import threading
import time
from pathlib import Path

import numpy as np

from logger import get_logger

log = get_logger(__name__)

_OUT = Path(__file__).resolve().parent.parent / "logs" / "signal_backtest.json"

# Breakeven-trail level for the per-signal expectancy sim — mirrors the live
# autopilot's breakeven_trigger_pct (default 2.0). Backtesting WITH the trail
# measures the strategy the system actually trades, not a naked-stop strawman.
# Env-tunable so it can track a changed autopilot setting.
_BT_BREAKEVEN_PCT = float(_os.getenv("QT_BT_BREAKEVEN_PCT", "2.0") or 2.0)

# The slippage assumption baked into every ledger row (a MODELED cost, not a
# measured one — an E0-evidence value, reconciled against real fills once paper
# trading runs). Recorded per trade so the report can state its assumptions.
try:
    from core.costs import _SLIPPAGE_PCT as _SLIPPAGE_NOTE
except Exception:
    _SLIPPAGE_NOTE = 0.10

_state = {"running": False, "progress": 0, "total": 0}
_state_lock = threading.Lock()


def _simulate(entry: float, stop: float, target: float,
              fwd_high: np.ndarray, fwd_low: np.ndarray,
              fwd_close: np.ndarray, be_pct: float = 0.0) -> tuple[str, float]:
    """
    Realistic first-touch simulation: the trade only exists once price
    actually reaches the entry (breakout orders sit above the market).
    Returns (outcome, r_multiple). FLAT trades are marked-to-market at
    the horizon close so expectancy is honest, not survivorship-biased.

    be_pct > 0 arms a BREAKEVEN TRAIL — once a day's high reaches
    entry × (1 + be_pct/100), the stop moves up to entry. This models the
    live system's actual risk discipline (trailing_enabled +
    breakeven_trigger_pct): a trade that pops in profit then fades exits at
    ~breakeven (SCRATCH, 0R) instead of being logged as a FLAT loss. Without
    it the backtest tests a strategy the system doesn't trade (naked 2×ATR
    stop, no trail) and systematically understates expectancy — full losses
    are captured while faded winners bleed to a FLAT loss. be_pct=0 keeps the
    original behaviour (used by the target-geometry sweep)."""
    o, r, _, _ = _simulate_timed(entry, stop, target, fwd_high, fwd_low,
                                 fwd_close, be_pct)
    return o, r


def _simulate_timed(entry: float, stop: float, target: float,
                    fwd_high: np.ndarray, fwd_low: np.ndarray,
                    fwd_close: np.ndarray, be_pct: float = 0.0):
    """The simulation core, also returning (entry_offset, exit_offset) — the bar
    indices into the forward window where the trade FILLED and EXITED (exit = the
    bar of the stop/target/scratch touch, or the last bar for FLAT). `_simulate`
    delegates here and drops the offsets, so its (outcome, r) contract is
    byte-identical. The offsets are what the Historical Trade Ledger needs to date
    each trade and pair a benchmark/factor return over its exact holding window —
    without them the alpha-vs-beta gate cannot be computed. Offsets are -1 on
    NO_FILL."""
    risk = entry - stop
    if risk <= 0:
        return "NO_FILL", 0.0, -1, -1
    filled = False
    entry_off = -1
    cur_stop = stop
    be_level = entry * (1 + be_pct / 100) if be_pct > 0 else None
    for i, (h, l, c) in enumerate(zip(fwd_high, fwd_low, fwd_close)):
        if not filled:
            if h >= entry:
                filled = True
                entry_off = i
            else:
                continue
        if l <= cur_stop:                         # stop first (gap-conservative)
            # LOSS only if the stop is still below entry; a trailed-to-entry
            # stop that gets tagged is a SCRATCH (0R), not a loss.
            if cur_stop < entry:
                return "LOSS", -1.0, entry_off, i
            return "SCRATCH", 0.0, entry_off, i
        if h >= target:
            return "WIN", (target - entry) / risk, entry_off, i
        if be_level is not None and cur_stop < entry and h >= be_level:
            cur_stop = entry                      # arm the breakeven trail
    if filled:
        return ("FLAT", (float(fwd_close[-1]) - entry) / risk,
                entry_off, len(fwd_close) - 1)
    return "NO_FILL", 0.0, -1, -1


def _emit_trade_record(on_trade, df, t, sym, r, outcome, gross_r, net_r,
                       cost_r, e_off, x_off, regime, calib_version):
    """Assemble the immutable core of one ledger row and hand it to the sink. Only
    the fields KNOWN at the trade level are set here (dates, prices, R, cost,
    regime, signals); the gauntlet's sink enriches with benchmark/factor returns
    over [entry_dt, exit_dt] so nothing is recomputed downstream. `calib_version`
    is hashed once per run and passed in (not recomputed per trade)."""
    risk = r.entry - r.stop
    entry_dt = df.index[t + e_off] if e_off >= 0 else df.index[min(t, len(df) - 1)]
    exit_dt = df.index[t + x_off] if x_off >= 0 else entry_dt
    exit_price = r.entry + gross_r * risk            # exact for WIN/LOSS/SCRATCH/FLAT
    sigs = tuple(r.signals) if getattr(r, "signals", None) else ()
    on_trade({
        "symbol": sym,
        "signals": sigs,
        "signal_id": sigs[0] if sigs else "",
        "entry_datetime": entry_dt,
        "exit_datetime": exit_dt,
        "entry_price": float(r.entry),
        "exit_price": float(exit_price),
        "holding_period": int(x_off - e_off + 1) if x_off >= e_off >= 0 else 0,
        "stop_price": float(r.stop),
        "target_price": float(r.target),
        "gross_R": round(float(gross_r), 4),
        "net_R": round(float(net_r), 4),
        "costs": round(float(cost_r), 4),
        "slippage_used": _SLIPPAGE_NOTE,
        "exit_reason": outcome,
        "regime": regime or "UNKNOWN",
        "confidence": float(getattr(r, "confidence", 0.0) or 0.0),
        "calibration_version": calib_version,
    })


def _calibration_version() -> str:
    """A short fingerprint of the score-calibration in force — pinned into every
    ledger row so a result is tied to the exact weights that produced it."""
    try:
        from scan.unified_scanner import _load_calibration
        import hashlib
        cal = _load_calibration() or {}
        return hashlib.sha1(json.dumps(cal, sort_keys=True,
                                       default=str).encode()).hexdigest()[:12]
    except Exception:
        return "none"


# ── Regime classification — SAME rule for history and today ──────────────────
# Signals behave differently in bull/chop/bear tape. Classifying every
# historical sample lets the report say "VCP earns +0.4R in BULL and
# −0.2R in CHOP" — and because today's regime uses the identical rule,
# the number is directly actionable.

def classify_regime(close) -> "pd.Series":
    """BULL / CHOP / BEAR per day: price vs 50DMA + 20-day return."""
    import pandas as pd
    close = pd.Series(close) if not isinstance(close, pd.Series) else close
    sma50 = close.rolling(50).mean()
    ret20 = close.pct_change(20)
    out = pd.Series("CHOP", index=close.index)
    out[(close > sma50) & (ret20 > 0.02)] = "BULL"
    out[(close < sma50) & (ret20 < -0.02)] = "BEAR"
    return out


def _nifty_regime_series():
    try:
        from data.index_store import get_index_ohlcv
        df = get_index_ohlcv("^NSEI")
        if df is None or len(df) < 70:
            return None
        col = next((c for c in df.columns if c.lower() == "close"), None)
        return classify_regime(df[col]) if col else None
    except Exception as exc:
        log.debug("backtest_regime_series_failed", error=str(exc))
        return None


def current_regime_simple() -> str:
    """Today's regime by the SAME rule the backtest buckets used."""
    s = _nifty_regime_series()
    return str(s.iloc[-1]) if s is not None and len(s) else "UNKNOWN"


# ── Statistics helpers ────────────────────────────────────────────────────────

def wilson_ci_pp(wins: int, n: int, z: float = 1.96) -> float:
    """± half-width of the win-rate CI in percentage points. Honest
    claims: '62% WR' on 12 trades is really '62% ± 27' — say so."""
    if n <= 0:
        return 0.0
    p = wins / n
    return round(z * ((p * (1 - p) / n) ** 0.5) * 100, 1)


def signal_verdict(closed: int, expectancy_r: float) -> str:
    """PROVEN / POSITIVE / NEUTRAL / LOSER / THIN — one usable word."""
    if closed < 30:
        return "THIN"
    if expectancy_r >= 0.25:
        return "PROVEN"
    if expectancy_r >= 0.05:
        return "POSITIVE"
    if expectancy_r <= -0.05:
        return "LOSER"
    return "NEUTRAL"


def sweep_targets(entry: float, stop: float, fwd_high, fwd_low, fwd_close,
                  pcts=(2.0, 3.0, 4.0)) -> dict[str, tuple]:
    """{'+2%': (outcome, r)} — same entry/stop, different profit targets.
    Measures which target geometry actually earns; feeds the autopilot's
    target_pct recommendation with evidence instead of a hunch."""
    out = {}
    for p in pcts:
        tgt = entry * (1 + p / 100)
        out[f"+{p:.0f}%"] = _simulate(entry, stop, tgt,
                                      fwd_high, fwd_low, fwd_close)
    return out


def run_backtest(sample_step: int = 5, lookback_sessions: int = 250,
                 horizon: int = 10, max_symbols: int | None = None,
                 symbols: list[str] | None = None, on_trade=None) -> dict:
    """
    Walk-forward backtest across the bhav store. Returns and persists
    per-signal stats. Needs ≥ (60 + lookback + horizon) sessions of data.

    Universe selection:
      • ``symbols`` provided → use that list exactly
      • else ``max_symbols is None`` / ``<= 0`` → **every** bhav store symbol
      • else first ``max_symbols`` only (legacy capped runs; marked truncated)

    `on_trade`, when given, is called with a full per-trade record dict for every
    FILLED trade (the Historical Trade Ledger hook). Default None → the aggregate
    path is byte-for-byte unchanged; the ledger is opt-in for the gauntlet.
    """
    from data.bhavcopy_store import store_symbols
    from scan.unified_scanner import UnifiedScanner

    sc = UnifiedScanner()
    stats: dict[str, dict] = {}
    available = list(store_symbols() or [])
    if symbols is not None:
        wanted = [str(s).strip().upper() for s in symbols if str(s).strip()]
        avail_set = set(available)
        run_symbols = [s for s in wanted if s in avail_set] if avail_set else wanted
        truncated = len(run_symbols) < len(wanted)
    elif max_symbols is None or int(max_symbols) <= 0:
        run_symbols = available
        truncated = False
    else:
        cap = int(max_symbols)
        run_symbols = available[:cap]
        truncated = len(available) > len(run_symbols)

    with _state_lock:
        _state.update(running=True, progress=0, total=len(run_symbols))

    t0 = time.time()
    try:
        out = _run_backtest_inner(sc, stats, run_symbols, sample_step,
                                   lookback_sessions, horizon, t0, on_trade)
        out["universe"] = {
            "source": "bhavcopy_store",
            "available": len(available),
            "run": len(run_symbols),
            "truncated": truncated,
            "max_symbols": max_symbols,
            "note": (
                "Full bhav EQ universe"
                if not truncated and symbols is None and (max_symbols is None or int(max_symbols or 0) <= 0)
                else (
                    f"Explicit list · {len(run_symbols)} symbols"
                    if symbols is not None
                    else f"Capped at {len(run_symbols)} of {len(available)} available symbols"
                )
            ),
        }
        try:
            _OUT.parent.mkdir(parents=True, exist_ok=True)
            _OUT.write_text(json.dumps(out, indent=2))
        except Exception as exc:
            log.warning("backtest_save_failed", error=str(exc))
        return out
    finally:
        with _state_lock:
            _state["running"] = False


def _run_backtest_inner(sc, stats, symbols, sample_step, lookback_sessions,
                        horizon, t0, on_trade=None):
    from data.bhavcopy_store import get_ohlcv

    regime_series = _nifty_regime_series()      # None → regime split skipped
    # the calibration is frozen for the whole run — hash it ONCE, not per trade
    _calib_version = _calibration_version() if on_trade is not None else ""
    tgt_stats: dict[str, dict] = {}             # target label → aggregate

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
            outcome, r_mult, _e_off, _x_off = _simulate_timed(
                r.entry, r.stop, r.target,
                fwd["high"].values, fwd["low"].values, fwd["close"].values,
                be_pct=_BT_BREAKEVEN_PCT)
            if outcome == "NO_FILL":
                continue          # order never triggered — not a trade
            gross_r = r_mult      # before costs — the ledger keeps both
            # NET of round-trip trading costs — the edge you'd actually keep, not
            # the gross move. Same cost hits the target-geometry sweep below.
            try:
                from core.costs import cost_in_r
                _cost_r = cost_in_r(risk / r.entry, "CNC")
            except Exception:
                _cost_r = 0.0
            r_mult -= _cost_r

            regime = ""
            if regime_series is not None:
                try:
                    reg = regime_series.asof(df.index[t - 1])
                    regime = str(reg) if isinstance(reg, str) else ""
                except Exception:
                    regime = ""

            if on_trade is not None:
                try:
                    _emit_trade_record(on_trade, df, t, sym, r, outcome,
                                       gross_r, r_mult, _cost_r, _e_off, _x_off,
                                       regime, _calib_version)
                except Exception:
                    pass          # ledger emission must never break the backtest

            # Target geometry sweep — same fill, +2/+3/+4% targets
            for label, (o2, r2) in sweep_targets(
                    r.entry, r.stop, fwd["high"].values,
                    fwd["low"].values, fwd["close"].values).items():
                ts = tgt_stats.setdefault(label, {"trades": 0, "wins": 0,
                                                  "closed": 0, "r_sum": 0.0})
                ts["trades"] += 1
                ts["r_sum"] += r2 - _cost_r          # net of costs, same as above
                if o2 == "WIN":
                    ts["wins"] += 1
                    ts["closed"] += 1
                elif o2 == "LOSS":
                    ts["closed"] += 1

            for sig in r.signals:
                s = stats.setdefault(sig, {"trades": 0, "wins": 0, "losses": 0,
                                           "flat": 0, "r_sum": 0.0,
                                           "regimes": {}})
                s["trades"] += 1
                s["r_sum"] += r_mult
                if outcome == "WIN":
                    s["wins"] += 1
                elif outcome == "LOSS":
                    s["losses"] += 1
                else:
                    s["flat"] += 1
                if regime:
                    rg = s["regimes"].setdefault(
                        regime, {"trades": 0, "wins": 0, "closed": 0,
                                 "r_sum": 0.0})
                    rg["trades"] += 1
                    rg["r_sum"] += r_mult
                    if outcome == "WIN":
                        rg["wins"] += 1
                        rg["closed"] += 1
                    elif outcome == "LOSS":
                        rg["closed"] += 1
        with _state_lock:
            _state["progress"] = si + 1

    # Derive rates & expectancy — true average R across ALL filled trades
    # (wins, losses, AND flats marked at horizon), no survivorship bias.
    report: dict[str, dict] = {}
    for sig, s in stats.items():
        closed = s["wins"] + s["losses"]
        wr = s["wins"] / closed if closed else 0.0
        exp = round(s["r_sum"] / s["trades"], 2) if s["trades"] else 0.0
        by_regime = {}
        for reg, rg in (s.get("regimes") or {}).items():
            by_regime[reg] = {
                "trades": rg["trades"],
                "win_rate": round(rg["wins"] / rg["closed"] * 100, 1)
                            if rg["closed"] else 0.0,
                "expectancy_r": round(rg["r_sum"] / rg["trades"], 2)
                                if rg["trades"] else 0.0,
            }
        report[sig] = {
            "trades": s["trades"], "closed": closed,
            "win_rate": round(wr * 100, 1),
            "wr_ci_pp": wilson_ci_pp(s["wins"], closed),
            "expectancy_r": exp,
            "verdict": signal_verdict(closed, exp),
            "flat": s["flat"],
            "by_regime": by_regime,
        }

    target_sweep = {}
    for label, ts in tgt_stats.items():
        target_sweep[label] = {
            "trades": ts["trades"],
            "hit_rate": round(ts["wins"] / ts["closed"] * 100, 1)
                        if ts["closed"] else 0.0,
            "expectancy_r": round(ts["r_sum"] / ts["trades"], 2)
                            if ts["trades"] else 0.0,
        }
    recommended = None
    evidenced = {k: v for k, v in target_sweep.items()
                 if v["trades"] >= 100}
    if evidenced:
        best = max(evidenced.items(), key=lambda kv: kv[1]["expectancy_r"])
        recommended = float(best[0].strip("+%"))

    out = {"generated_at": time.strftime("%Y-%m-%d %H:%M"),
           "symbols": len(symbols), "horizon_days": horizon,
           "signals": report,
           "target_sweep": target_sweep,
           "recommended_target_pct": recommended,
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


def report_is_actionable(report: dict | None = None) -> bool:
    """True only when the persisted signal backtest is usable for product decisions."""
    rep = dict(report or load_report() or {})
    if not rep or not rep.get("signals"):
        return False
    uni = dict(rep.get("universe") or {})
    run = int(uni.get("run") or rep.get("symbols") or 0)
    if run < 100:
        return False
    available = int(uni.get("available") or uni.get("available_in_store") or 0)
    truncated = bool(uni.get("truncated"))
    if truncated and run < 500:
        return False
    if available and run < max(100, int(available * 0.35)):
        return False
    return True


def universe_evidence_note(report: dict | None = None) -> str:
    rep = dict(report or load_report() or {})
    uni = dict(rep.get("universe") or {})
    run = int(uni.get("run") or rep.get("symbols") or 0)
    available = int(uni.get("available") or uni.get("available_in_store") or run)
    if run <= 0:
        return "no measured backtest yet"
    if uni.get("truncated"):
        return f"{run} of {available} stocks (truncated sample)"
    return f"{run} stocks (full-universe backtest)"


def combo_edge(signal_keys: list[str], min_trades: int = 30) -> float | None:
    """
    Measured expectancy (avg R/trade) for a stock's signal combo —
    mean of each signal's backtested expectancy, using only signals
    with enough evidence. None when no backtest / no evidenced signal /
    report too thin or truncated to trust for product decisions.
    """
    rep = load_report()
    if not report_is_actionable(rep):
        return None
    vals = []
    for k in signal_keys:
        s = rep.get("signals", {}).get(k)
        if s and s.get("trades", 0) >= min_trades:
            vals.append(float(s.get("expectancy_r", 0)))
    return round(sum(vals) / len(vals), 2) if vals else None


def edge_in_regime(signal_keys: list[str], regime: str,
                   min_trades: int = 20) -> float | None:
    """combo_edge, but conditioned on a regime bucket. Falls back to the
    overall number per signal when its regime bucket is too thin —
    honest specificity, never thin-slice noise."""
    rep = load_report()
    if not report_is_actionable(rep):
        return None
    vals = []
    for k in signal_keys:
        s = rep.get("signals", {}).get(k)
        if not s:
            continue
        rg = (s.get("by_regime") or {}).get(regime)
        if rg and rg.get("trades", 0) >= min_trades:
            vals.append(float(rg["expectancy_r"]))
        elif s.get("trades", 0) >= 30:
            vals.append(float(s.get("expectancy_r", 0)))
    return round(sum(vals) / len(vals), 2) if vals else None


def trading_playbook() -> dict | None:
    """The backtest as ACTION, not a table: aaj ka regime, usme kaunse
    signals earn karte hain, kaunse avoid, aur kaunsa target geometry
    sabse zyada expectancy deta hai. None until a backtest has run."""
    rep = load_report()
    if not rep:
        return None
    regime = current_regime_simple()
    best: list[dict] = []
    for k, s in rep.get("signals", {}).items():
        rg = (s.get("by_regime") or {}).get(regime)
        if rg and rg.get("trades", 0) >= 20:
            best.append({"signal": k, "expectancy_r": rg["expectancy_r"],
                         "trades": rg["trades"], "basis": "regime"})
        elif s.get("trades", 0) >= 30:
            best.append({"signal": k, "expectancy_r": s["expectancy_r"],
                         "trades": s["trades"], "basis": "overall"})
    best.sort(key=lambda d: -d["expectancy_r"])
    avoid = sorted(k for k, s in rep.get("signals", {}).items()
                   if s.get("verdict") == "LOSER")
    return {"regime": regime,
            "best": [b for b in best if b["expectancy_r"] > 0][:3],
            "avoid": avoid,
            "target_sweep": rep.get("target_sweep") or {},
            "recommended_target_pct": rep.get("recommended_target_pct"),
            "generated_at": rep.get("generated_at")}


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
    # Default CLI module run = full bhav universe (not the old 800 cap).
    print(json.dumps(run_backtest(max_symbols=None), indent=2))
