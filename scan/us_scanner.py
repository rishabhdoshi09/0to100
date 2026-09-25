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

import json
import os
import threading
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from logger import get_logger
from core.runtime_paths import logs_path

log = get_logger(__name__)

_lock = threading.Lock()
_results: list[dict] = []
_last_ts: float = 0.0
_status: str = "idle"        # idle | scanning | ready | error
_progress: int = 0           # symbols processed so far (live)
_total: int = 0              # universe size for this run
_scan_running: bool = False  # only one scan at a time
_scope: str = "All"          # "All" | "S&P 500" | "NASDAQ-100" | "Dow 30"
_pushed: dict[str, set] = {}  # {YYYY-MM-DD: symbols alerted} — one/stock/day
_STORE = logs_path("product", "us_scan.json")


def _persist_results(results: list[dict], *, scope: str, status: str = "ready") -> None:
    payload = {
        "schema_version": 1,
        "market": "US",
        "status": status,
        "scope": scope,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "saved_epoch": time.time(),
        "records": list(results),
        "count": len(results),
        "paper_only": True,
        "live_locked": True,
    }
    _STORE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _STORE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, _STORE)


def _load_persisted() -> dict:
    try:
        payload = json.loads(_STORE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def persisted_us_scan() -> dict:
    """Durable US scan truth across UI/API/host restarts."""
    payload = _load_persisted()
    payload.setdefault("records", [])
    payload.setdefault("status", "idle")
    payload.setdefault("scope", "All")
    payload.setdefault("market", "US")
    return payload


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
        "above_sma50": bool(getattr(r, "above_sma50", False)),
        "above_sma200": bool(getattr(r, "above_sma200", False)),
        "avg_vol20": float(getattr(r, "avg_vol20", 0.0)),
        "chase_risk": bool(getattr(r, "chase_risk", False)),
    }


# ── 🏦 US quality floor — "naam jaana-pehchana ho" ka quant proxy ─────────────
# Market cap directly listing file mein nahi aata, par uska sabse imaandaar
# proxy humare paas FREE hai: dollar turnover (price × 20-day avg volume).
# Unknown micro-caps $10M/day kabhi nahi chhoote; har naam jo tum jaante ho
# (AAPL se PLTR tak) isse 100× upar hai. Penny floor alag se ($5 — SEC ki
# penny-stock line). .env se tunable; 0 = off.
_US_MIN_TURNOVER_M = float(os.getenv("QT_US_MIN_TURNOVER_M", "10") or 10)
_US_MIN_PRICE = float(os.getenv("QT_US_MIN_PRICE", "5") or 5)


def _quality_floor(rows: list[dict]) -> list[dict]:
    """Drop penny/micro-cap junk: price < $5 ya turnover < $10M/day.
    Institutional-grade names hi bache — breakout list ab pehchaane huye
    naamon ki hogi."""
    if _US_MIN_TURNOVER_M <= 0 and _US_MIN_PRICE <= 0:
        return rows
    kept = []
    for r in rows:
        px = float(r.get("price") or 0)
        turnover = px * float(r.get("avg_vol20") or 0)
        if px < _US_MIN_PRICE:
            continue
        if _US_MIN_TURNOVER_M > 0 and turnover < _US_MIN_TURNOVER_M * 1e6:
            continue
        r["turnover_m"] = round(turnover / 1e6, 1)   # card display ke liye
        kept.append(r)
    if len(kept) < len(rows):
        log.info("us_quality_floor", dropped=len(rows) - len(kept),
                 kept=len(kept))
    return kept


def _liquid_first(symbols: list[str]) -> list[str]:
    """Put the mega/large-cap liquid names at the FRONT so the first
    batches surface real, tradeable setups within seconds — the long
    illiquid tail streams in behind them."""
    try:
        from data.us_universe import _CURATED
        liquid = [s for s in _CURATED if s in symbols]
        seen = set(liquid)
        return liquid + [s for s in symbols if s not in seen]
    except Exception:
        return symbols


def _rank(raw: list) -> list[dict]:
    serialized = _quality_floor([_serialize(r) for r in raw])
    try:
        from scan.auto_scan import tag_conviction
        tag_conviction(serialized)
    except Exception:
        pass
    _vr = {"STRONG BUY": 2, "BUY": 1}
    serialized.sort(key=lambda r: (_vr.get(r.get("verdict"), 0),
                                   float(r.get("score", 0))), reverse=True)
    return serialized


def _index_universe(index: str | None) -> tuple[list[str], str]:
    """Symbols to scan for a requested scope. index=None → full US listing;
    an index name → just that index's members (S&P 500 / NASDAQ-100 / Dow 30).
    Returns (symbols, scope_label)."""
    from data.us_universe import get_us_universe
    if not index or index in ("All", "all"):
        return get_us_universe(), "All"
    try:
        from data.us_indices import get_index_members
        members, _src = get_index_members(index)
        if members:
            return sorted(members), index
    except Exception as exc:
        log.debug("us_index_scope_failed", index=index, error=str(exc)[:80])
    # unknown/empty index → fail safe to the full universe (never scan nothing)
    return get_us_universe(), "All"


def scan_us(max_workers: int = 8, index: str | None = None) -> list[dict]:
    """Run the unified engine over the US universe. Returns serialized,
    conviction-ranked results and caches them. HEAVY when scanning the full
    listing — run it via start_us_scan() so it never blocks the UI thread.
    `index` scopes the scan to one index (S&P 500 / NASDAQ-100 / Dow 30)."""
    global _results, _last_ts, _status, _progress, _total, _scan_running, _scope
    from data.us_data import get_us_daily, sp500_return_30d
    from scan.unified_scanner import UnifiedScanner

    with _lock:
        if _scan_running:               # already scanning — don't double-run
            return list(_results)
        _scan_running = True
        _status = "scanning"
        _progress = 0
    try:
        _syms, scope = _index_universe(index)
        symbols = _liquid_first(_syms)
        with _lock:
            _total = len(symbols)
            _scope = scope
        try:
            from core.eco import workers as _eco_workers
            max_workers = _eco_workers(max_workers)
        except Exception:
            pass
        sc = UnifiedScanner(max_workers=max_workers)
        sc._nifty_ret30 = sp500_return_30d()      # RS benchmark = S&P 500

        from data.us_data import get_us_daily_batch

        def _run_batch(chunk: list[str]) -> list:
            out = []
            try:
                for sym, df in get_us_daily_batch(chunk).items():
                    try:
                        r = sc._analyze(sym, df)
                        if r and r.signals:
                            out.append(r)
                    except Exception:
                        pass
            except Exception as exc:
                log.debug("us_batch_chunk_failed", error=str(exc)[:80])
            return out

        # PARALLEL batches (not one-by-one), liquid names first, and results
        # STREAM into the store as each batch lands — good setups show in
        # seconds instead of waiting for all ~5,000 names.
        _BATCH = 100
        chunks = [symbols[i:i + _BATCH] for i in range(0, len(symbols), _BATCH)]
        raw: list = []
        done = 0
        try:
            from core.eco import workers as _eco_pool
            _n_pool = _eco_pool(6)
        except Exception:
            _n_pool = 6
        with ThreadPoolExecutor(max_workers=_n_pool) as pool:
            futs = {pool.submit(_run_batch, c): c for c in chunks}
            for fut in as_completed(futs):
                raw.extend(fut.result() or [])
                done += len(futs[fut])
                serialized = _rank(raw)          # progressive: publish as we go
                with _lock:
                    _results = serialized
                    _progress = min(done, len(symbols))
                    _last_ts = time.time()

        serialized = _rank(raw)
        with _lock:
            _results = serialized
            _last_ts = time.time()
            _status = "ready"
        _persist_results(serialized, scope=scope, status="ready")
        # 📲 Telegram push — US setups bhi phone pe (NSE push untouched)
        try:
            _push_us_setups(serialized)
        except Exception as exc:
            log.debug("us_push_skip", error=str(exc))
        # 🇺🇸🤖 US autopilot — same signals, paper-only, additive
        try:
            from execution.us_autopilot import on_setups, review_cycle
            review_cycle()
            on_setups(serialized)
        except Exception as exc:
            log.debug("us_autopilot_feed_skip", error=str(exc))
        log.info("us_scan_done", scanned=len(symbols),
                 with_signals=len(serialized))
        return serialized
    except Exception as exc:
        log.warning("us_scan_failed", error=str(exc))
        with _lock:
            _status = "error"
        return []
    finally:
        with _lock:
            _scan_running = False


def start_us_scan(index: str | None = None) -> bool:
    """Kick off scan_us in a BACKGROUND daemon thread and return
    immediately. The UI polls get_us_results()/get_us_progress() — the
    heavy full-universe scan never blocks the Streamlit thread (which is
    why Ctrl+C works and the page stays responsive). No-op if already
    running. `index` scopes the scan (None = full listing)."""
    with _lock:
        if _scan_running:
            return False
    threading.Thread(target=scan_us, kwargs={"index": index},
                     name="us-scan", daemon=True).start()
    return True


def get_us_progress() -> dict:
    with _lock:
        return {"status": _status, "progress": _progress, "total": _total,
                "running": _scan_running, "have": len(_results), "scope": _scope}


def get_us_scope() -> str:
    """Which universe the current results cover: 'All' or an index name."""
    with _lock:
        return _scope


def _push_us_setups(results: list[dict]) -> None:
    """Fresh US BUY/STRONG-BUY setups → Telegram, highest conviction first.
    One alert per stock per day. Silent if Telegram not configured."""
    from datetime import datetime

    from core.market_clock import IST
    try:
        from alerts.telegram_alerts import AlertEngine
        engine = AlertEngine()
        if not engine.is_configured():
            return
    except Exception:
        return
    today = datetime.now(IST).strftime("%Y-%m-%d")
    with _lock:
        _pushed.setdefault(today, set())
        for k in list(_pushed):
            if k != today:
                del _pushed[k]
        seen = _pushed[today]
    ranked = sorted(
        [r for r in results if r.get("verdict") in ("STRONG BUY", "BUY")
         and r["symbol"] not in seen],
        key=lambda r: float(r.get("conviction_rank") or r.get("score", 0) or 0),
        reverse=True)[:5]
    if not ranked:
        return
    lines = ["🇺🇸 <b>US setups mile</b>"]
    for r in ranked:
        emoji = "🔥" if r["verdict"] == "STRONG BUY" else "⚡"
        hc = "🎯 " if r.get("high_conviction") else ""
        conv = r.get("breakout_conviction") or 0
        conv_bit = f" · conviction {conv:.0f}" if conv else ""
        why = (r.get("reasons") or [""])[0]
        lines.append(
            f"\n{emoji} {hc}<b>{r['symbol']}</b> ${float(r.get('price') or 0):,.2f}"
            f" — {r['verdict']}{conv_bit}\n"
            f"   {why[:120]}\n"
            f"   Entry ${float(r.get('entry') or 0):,.2f} · "
            f"Stop ${float(r.get('stop') or 0):,.2f} · "
            f"Target ${float(r.get('target') or 0):,.2f}")
    lines.append("\n<i>US market · paper autopilot in 🇺🇸 tab · "
                 "live ke liye US broker chahiye</i>")
    try:
        engine.send("\n".join(lines))
        with _lock:
            _pushed[today].update(r["symbol"] for r in ranked)
        log.info("us_setups_pushed", count=len(ranked))
    except Exception as exc:
        log.debug("us_push_send_failed", error=str(exc))


def get_us_results() -> tuple[list[dict], float, str]:
    global _results, _last_ts, _status, _scope
    with _lock:
        if not _results:
            saved = persisted_us_scan()
            records = [dict(r) for r in (saved.get("records") or []) if isinstance(r, dict)]
            if records:
                _results = records
                _last_ts = float(saved.get("saved_epoch") or 0.0)
                _status = str(saved.get("status") or "ready")
                _scope = str(saved.get("scope") or "All")
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

    # Autopilot feed scope: a liquid index by default (fast + tradeable) —
    # 6,900+ names every 15 min is slow and rate-limit-prone, and the paper
    # autopilot trades liquid names anyway. Override with QT_US_SCAN_SCOPE
    # (e.g. "NASDAQ-100", "Dow 30", or "All" to scan the whole listing).
    import os
    _feed_scope = os.getenv("QT_US_SCAN_SCOPE", "S&P 500").strip()
    _feed_index = None if _feed_scope.lower() in ("all", "") else _feed_scope

    def _worker():
        while True:
            try:
                from data.us_data import us_market_open
                if us_market_open():
                    scan_us(index=_feed_index)
                    time.sleep(900)          # 15 min during US hours
                else:
                    time.sleep(300)          # 5 min re-check off-hours
            except Exception as exc:
                log.debug("us_loop_error", error=str(exc))
                time.sleep(300)

    threading.Thread(target=_worker, name="us-scan-loop", daemon=True).start()
    log.info("us_scan_loop_started")
