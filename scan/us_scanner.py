"""
US scanner — the SAME signal engine, pointed at US equities.

UnifiedScanner._analyze() is data-source agnostic: it takes a daily
OHLCV DataFrame and returns a graded setup. So the entire edge —
confirmed breakouts, breakout conviction, base quality, chart patterns,
the falling-knife filter — carries over to the US market for free. Only
two things change:

  • the relative-strength benchmark is the S&P 500, not Nifty
  • delivery % does not exist for US equities → the conviction model
    already treats it as neutral (no penalty)

Results are shaped exactly like the NSE store so the UI can reuse cards.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from logger import get_logger

log = get_logger(__name__)

_lock = threading.Lock()
_results: list[dict] = []
_last_ts: float = 0.0
_status: str = "idle"        # idle | scanning | ready | error


def _serialize(r) -> dict:
    return {
        "symbol": r.symbol, "price": r.price, "change_pct": r.change_pct,
        "momentum_5d": r.momentum_5d, "volume_ratio": r.volume_ratio,
        "signals": r.signal_labels, "categories": sorted(r.categories),
        "reasons": r.reasons, "score": r.score, "verdict": r.verdict,
        "entry": r.entry, "stop": r.stop, "target": r.target,
        "rr": round(r.risk_reward, 1),
        "pivot_distance_pct": getattr(r, "pivot_distance_pct", 0.0),
        "breakout_grade": getattr(r, "breakout_grade", ""),
        "breakout_conviction": getattr(r, "breakout_conviction", 0.0),
    }


def scan_us(max_workers: int = 8) -> list[dict]:
    """Run the unified engine over the US universe. Returns serialized,
    conviction-ranked results and caches them."""
    global _results, _last_ts, _status
    from data.us_universe import get_us_universe
    from data.us_data import get_us_daily, sp500_return_30d
    from scan.unified_scanner import UnifiedScanner

    with _lock:
        _status = "scanning"
    try:
        symbols = get_us_universe()
        sc = UnifiedScanner(max_workers=max_workers)
        sc._nifty_ret30 = sp500_return_30d()      # RS benchmark = S&P 500

        raw = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(get_us_daily, s): s for s in symbols}
            dfs = {}
            for f in as_completed(futs):
                df = None
                try:
                    df = f.result()
                except Exception:
                    df = None
                if df is not None and len(df) >= 60:
                    dfs[futs[f]] = df
        for sym, df in dfs.items():
            try:
                r = sc._analyze(sym, df)
                if r and r.signals:
                    raw.append(r)
            except Exception as exc:
                log.debug("us_analyze_failed", symbol=sym, error=str(exc)[:80])

        serialized = [_serialize(r) for r in raw]
        try:
            from scan.auto_scan import tag_conviction
            tag_conviction(serialized)
        except Exception:
            pass
        _vrank = {"STRONG BUY": 2, "BUY": 1}
        serialized.sort(
            key=lambda r: (_vrank.get(r.get("verdict"), 0),
                           float(r.get("score", 0))), reverse=True)
        with _lock:
            _results = serialized
            _last_ts = time.time()
            _status = "ready"
        # 🇺🇸🤖 US autopilot — same signals, paper-only, additive
        try:
            from execution.us_autopilot import on_setups, review_cycle
            review_cycle()
            on_setups(serialized)
        except Exception as exc:
            log.debug("us_autopilot_feed_skip", error=str(exc))
        log.info("us_scan_done", scanned=len(dfs), with_signals=len(serialized))
        return serialized
    except Exception as exc:
        log.warning("us_scan_failed", error=str(exc))
        with _lock:
            _status = "error"
        return []


def get_us_results() -> tuple[list[dict], float, str]:
    with _lock:
        return list(_results), _last_ts, _status


_loop_started = False


def start_us_loop() -> None:
    """Background daemon: during US market hours, scan every 15 min so the
    US autopilot gets a live signal feed. Idle (5-min re-check) otherwise.
    Started once from app.py alongside the other daemons."""
    global _loop_started
    with _lock:
        if _loop_started:
            return
        _loop_started = True

    def _worker():
        while True:
            try:
                from data.us_data import us_market_open
                if us_market_open():
                    scan_us()
                    time.sleep(900)          # 15 min during US hours
                else:
                    time.sleep(300)          # 5 min re-check off-hours
            except Exception as exc:
                log.debug("us_loop_error", error=str(exc))
                time.sleep(300)

    threading.Thread(target=_worker, name="us-scan-loop", daemon=True).start()
    log.info("us_scan_loop_started")
