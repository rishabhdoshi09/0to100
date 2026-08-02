"""
Unified Smart Scanner — one engine, every signal type.

One pass over the bulk OHLCV cache (no per-stock HTTP). Signals:

BREAKOUT (already happened — confirm & ride):
  BREAKOUT_52W    — new 52-week high
  BREAKOUT_RES    — 20-day resistance break on volume
  GOLDEN_CROSS    — 50 SMA crossing above 200 SMA
  VOL_SQUEEZE     — Bollinger squeeze expanding upward

PATTERN (structure forming — breakout likely ahead):
  VCP             — volatility contraction pattern near pivot
  FLAT_BASE       — tight 6+ week base near ceiling
  CUP_HANDLE      — cup with handle proxy
  HIGH_TIGHT_FLAG — 80%+ run then tight flag
  ASC_TRIANGLE    — flat top + rising lows
  DOUBLE_BOTTOM   — W-shape, neckline overhead

PRE-BREAKOUT (catch it before it happens):
  PRE_BREAKOUT    — within 2.5% of pivot with volume building
  ACCUMULATION    — volume dry-up + up-day volume dominance near highs
  DELIVERY_SPIKE  — NSE delivery %% rising (buy-and-hold money entering)
  NR7_COIL        — narrowest range in 7 sessions inside an uptrend
  POCKET_PIVOT    — up-day volume beats every down-day volume of 10 days

MOMENTUM:
  MOMENTUM        — RSI + price momentum + volume surge

Every stock gets ONE row with all detected signals, a composite score,
and an entry/stop/target plan. Plain-English reasons included.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from logger import get_logger

log = get_logger(__name__)

# ── Falling-knife filter (system-wide) ────────────────────────────────────────
# Stocks down more than this from their 52-week (250-session) high are
# ignored EVERYWHERE — scanner signals, pulse, movers, autopilot, all of
# it. A −60% name is a delisting/insolvency risk or a knife still falling;
# the system does not chase it. Tunable via .env, never below 40%.
import os as _os
_MAX_DROP = max(0.40, float(_os.getenv("QT_MAX_DROP_FROM_HIGH", "0.60") or 0.60))
_DROP_LOOKBACK = 250


# ── Breakout quality (the pro filter) ────────────────────────────────────────
# A level being *touched* is not a breakout — the false-break is the single
# biggest killer of breakout traders. A CONFIRMED breakout clears the level
# by a slice of ATR (real room, not a 0.1% poke) AND carries volume (real
# conviction). Marginal breaks are demoted to a WATCH, never a buy.
_BREAKOUT_ATR_BUFFER = float(_os.getenv("QT_BREAKOUT_ATR_BUFFER", "0.25") or 0.25)
_BREAKOUT_MIN_VOL = float(_os.getenv("QT_BREAKOUT_MIN_VOL", "1.5") or 1.5)
_BREAKOUT_MAX_GAP = 8.0        # >8% gap up = exhaustion risk, not a clean break


def _market_is_live() -> bool:
    """Kya abhi koi market (NSE ya US) OPEN hai? Off-hours = aaj ka close
    aa chuka hai, toh 'wait' nahi bolna — data humare paas hai."""
    try:
        from core.market_session import in_market_open, us_market_open
        return in_market_open() or us_market_open()
    except Exception:
        return False


# Close Location Value (Wyckoff): kis end pe close hua din ki range mein.
# 1.0 = din ki high pe band (buyers ne close tak control rakha) · 0.0 = din
# ki low pe band (sellers ne breakout ke din hi vaapas le liya — bada upper
# wick, bull-trap ka classic pehla sanket). ATR/volume clear kar sakte hain
# aur phir bhi din khatam hote-hote seller le jaye — yehi cheez ATR+volume
# akele nahi pakadte, isliye alag check zaroori.
_CLV_WEAK = 0.40

# RSI overbought — a breakout with NO room left to run is a blow-off-top
# risk, not a leader continuing higher. Soft = demote A→B; Hard = reject
# outright (same tier as the gap-exhaustion check).
_RSI_OVERBOUGHT_SOFT = float(_os.getenv("QT_RSI_OVERBOUGHT_SOFT", "72") or 72)
_RSI_OVERBOUGHT_HARD = float(_os.getenv("QT_RSI_OVERBOUGHT_HARD", "82") or 82)

# Falling-knife / momentum-rollover — the OPPOSITE failure mode from
# extension. A fresh breakout/momentum BUY is a STRENGTH signal; a stock that
# is red today or whose RSI has rolled over (momentum turning down) is not
# breaking out — it's falling. Recommending it is catching a knife. A pullback-
# to-support (buys weakness INTO a rising trend, by design) is exempt.
_FALLING_DAY_PCT = float(_os.getenv("QT_FALLING_DAY_PCT", "-1.0") or -1.0)
_RSI_ROLLOVER_DROP = float(_os.getenv("QT_RSI_ROLLOVER_DROP", "5") or 5)

# Absolute extension above the 50-DMA — the chase the 20-EMA check MISSES.
# A stock grinding steadily higher stays within ~5-10% of its (rising) 20-EMA
# even after a big cumulative run, so the 20-EMA extension guard never fires —
# yet buying a name 20%+ above its 50-DMA is a late-stage/climax entry (mean
# reversion pending), not a base breakout (which sits ~0-10% above the 50-DMA).
# This is the ADANIENSOL case: +18% off the base, RSI rolling over, but only
# ~5% above the 20-EMA so chase_risk stayed False. Env-tunable; 0 = off.
_EXT_ABOVE_SMA50 = float(_os.getenv("QT_EXT_ABOVE_SMA50_PCT", "20") or 20)


def close_location_value(close: float, high: float, low: float) -> float:
    """(close-low)/(high-low), [0,1] mein clamp. Flat/no-range din → 0.5
    neutral (koi wick-info nahi, na positive na negative)."""
    rng = high - low
    if rng <= 0:
        return 0.5
    return max(0.0, min(1.0, (close - low) / rng))


def grade_breakout(price: float, level: float, atr: float, vratio: float,
                   day_change: float, clv: float = 1.0, rsi: float = 0.0
                   ) -> tuple[bool, str, str]:
    """(confirmed, grade, note). Grade A = clears by ≥1×ATR on ≥2× volume;
    B = clears by buffer on ≥min volume; not confirmed = marginal poke.

    clv (Close Location Value, default 1.0 = no penalty): weak close (<0.40
    — closed in bottom 40% of the day's range) flags a possible bull-trap —
    sellers took the day back despite the clearance.
    rsi (default 0.0 = no penalty): ≥82 rejects outright (blow-off-top, no
    room to run — same tier as the gap-exhaustion check); ≥72 flags as
    extended. Both flags DEMOTE only — A→B, or B→unconfirmed if either
    fires alongside a merely-marginal clearance. Never upgrades a grade."""
    if level <= 0 or price <= level:
        return False, "", ""
    clearance = price - level
    atr_mult = clearance / atr if atr > 0 else 0.0
    if day_change > _BREAKOUT_MAX_GAP:
        return False, "", f"gap {day_change:+.0f}% — exhaustion risk, chase nahi"
    if rsi >= _RSI_OVERBOUGHT_HARD:
        return False, "", f"RSI {rsi:.0f} — blow-off-top overbought, chase nahi"
    vol_ok = vratio >= _BREAKOUT_MIN_VOL
    room_ok = atr_mult >= _BREAKOUT_ATR_BUFFER
    if not (vol_ok and room_ok):
        why = ("volume kam" if not vol_ok else f"clearance sirf {atr_mult:.2f}×ATR")
        # Time-aware: market LIVE hai toh close ka wait; band ho chuka hai toh
        # close aa gaya aur us par bhi confirm nahi hua (data humare paas hai).
        tail = ("close confirm ka wait" if _market_is_live()
                else "close pe bhi confirm nahi hua")
        return False, "", f"marginal break ({why}) — {tail}"
    flags = []
    if clv < _CLV_WEAK:
        flags.append(f"weak close (CLV {clv:.2f})")
    if rsi >= _RSI_OVERBOUGHT_SOFT:
        flags.append(f"RSI {rsi:.0f} extended")
    if atr_mult >= 1.0 and vratio >= 2.0:
        if flags:
            return (True, "B", f"clean clearance ({atr_mult:.1f}×ATR, "
                    f"{vratio:.1f}×vol) PAR {', '.join(flags)} — A se B demote")
        return True, "A", f"clean break: {atr_mult:.1f}×ATR upar, {vratio:.1f}× volume"
    if flags:
        return (False, "", f"confirmed clearance PAR {', '.join(flags)} — "
                f"bull-trap/exhaustion risk, chase nahi")
    return True, "B", f"confirmed: {atr_mult:.2f}×ATR clear, {vratio:.1f}× volume"


# ── Breakout conviction (the full vault) ──────────────────────────────────────
# A confirmed break clears the level; CONVICTION asks who is behind it. We
# aggregate the signals retail ignores and pros live by, especially DELIVERY %
# (NSE bhavcopy tells us whether buyers took delivery — real ownership — or
# just punted intraday). Below the threshold a breakout is a WATCH, not a buy.
_MIN_BREAKOUT_CONVICTION = float(
    _os.getenv("QT_BREAKOUT_MIN_CONVICTION", "50") or 50)


def base_tightness(high, low, price: float, sma50: float) -> tuple[float, str]:
    """0-1 base quality. Traders prize BASE breakouts — a spring-loaded
    move out of tight consolidation, not a late break after a vertical
    run. Rewards a tight 6-week base; penalises extension above the
    50-DMA (already extended = late, wide stop, trap-prone)."""
    try:
        import numpy as _np
        if len(high) < 32:
            return 0.5, ""
        base_hi = float(_np.max(high[-31:-1]))      # base BEFORE today's bar
        base_lo = float(_np.min(low[-31:-1]))
        rng = (base_hi - base_lo) / base_hi if base_hi > 0 else 1.0
        tight = 1.0 - _norm(rng, 0.06, 0.25)        # 6% range→1.0, 25%+→0
        ext = (price / sma50 - 1) if sma50 > 0 else 0.0
        ext_ok = 1.0 - _norm(ext, 0.05, 0.20)       # ≤5% above→ok, ≥20%→extended
        q = 0.6 * tight + 0.4 * ext_ok
        note = ""
        if tight >= 0.7 and ext_ok >= 0.6:
            note = f"tight {rng*100:.0f}% base se nikla (spring-loaded)"
        elif ext_ok < 0.4:
            note = f"50-DMA se {ext*100:.0f}% upar — extended, late break"
        return round(q, 2), note
    except Exception:
        return 0.5, ""


def breakout_conviction(vratio: float, deliv_now, deliv_base,
                        rs_outperf: float, above_50: bool,
                        above_200: bool, base_q: float = 0.5,
                        clv: float = 1.0, pct_below_high: float = 0.0
                        ) -> tuple[float, list[str]]:
    """0-100 conviction behind a confirmed breakout + plain-English factors.
      Volume 25 · Delivery 20 · Relative strength 20 · Trend stage 15 ·
      Base quality 20 (base breakouts > extended breakouts).

    Two demote-only passes AFTER the weight budget (never inflate — a
    strong reading never adds points, it just avoids a cut):
    clv (default 1.0): weak close (<0.5) — sellers took the day back —
      cuts up to 40% at the worst (closed at the day's low).
    pct_below_high (default 0.0 — exempt by construction for a fresh
      52-week-high breakout): how far below the 52-week high this break
      is happening. >30% below = 'laggard, not leader' zone — a genuine
      Minervini/institutional selectivity criterion (jyada gira hua stock
      ek genuine leader nahi hota, chahe aaj resistance clear kar de).
      Deliberately gentle (protection gear, not a brake): a strong break
      (raw conviction ~70+) still clears the BUY conviction floor even
      deep in laggard territory — this should thin out mediocre laggard
      setups, never blanket-veto the category. Cuts linearly past 30%,
      floors at a 30% cut (mult 0.70) by 60%+ below the high — a harsher
      floor would combine with the BUY conviction gate to make DEEP
      laggards un-BUYable regardless of quality, which is a brake, not a
      filter."""
    factors: list[str] = []
    score = 0.0
    # Volume — is there force behind the break?
    score += _norm(vratio, 1.5, 3.0) * 25
    if vratio >= 2.0:
        factors.append(f"{vratio:.1f}× volume")
    # Delivery % — institutional ownership vs intraday punting (India edge)
    if deliv_now is not None:
        d = _norm(float(deliv_now), 35, 70) * 12
        if deliv_base is not None and deliv_now > deliv_base * 1.1:
            d += 8
            factors.append(f"delivery {deliv_now:.0f}% & rising (real buying)")
        elif deliv_now >= 55:
            factors.append(f"delivery {deliv_now:.0f}% (strong ownership)")
        score += min(20, d)
    else:
        score += 10                     # unknown → neutral, not penalised
    # Relative strength — leading the index or lagging it?
    score += _norm(rs_outperf, -3, 12) * 20
    if rs_outperf >= 3:
        factors.append(f"Nifty se {rs_outperf:+.0f}% aage (RS leader)")
    # Trend stage — a breakout in a Stage-2 uptrend, not a dead-cat bounce
    if above_50:
        score += 7.5
    if above_200:
        score += 7.5
        factors.append("200-DMA ke upar (Stage 2)")
    # Base quality — the trader's edge: base breakout > extended breakout
    score += max(0.0, min(1.0, base_q)) * 20
    score = min(100.0, score)
    # Close Location Value — demote-only, applied AFTER the weight budget
    # (keeps Volume/Delivery/RS/Trend/Base weights untouched at 100).
    if clv < 0.5:
        score *= 0.6 + 0.8 * clv           # clv 0.5→no cut · clv 0.0→40% cut
        factors.append(f"weak close (CLV {clv:.2f}) — sellers ne din "
                       f"wapas liya, conviction demote")
    if pct_below_high > 30:
        mult = max(0.70, 1.0 - 0.01 * (pct_below_high - 30))  # →60%→floor 0.70
        score *= mult
        factors.append(f"52-week high se {pct_below_high:.0f}% neeche — "
                       f"laggard zone, genuine leader nahi")
    return round(score, 0), factors


def _nifty_return_30d() -> float:
    try:
        from data.index_store import get_index_ohlcv
        df = get_index_ohlcv("^NSEI")
        if df is None or len(df) < 31:
            return 0.0
        col = next((c for c in df.columns if c.lower() == "close"), None)
        if not col:
            return 0.0
        c = df[col].values
        return float((c[-1] / c[-31] - 1) * 100)
    except Exception:
        return 0.0


def is_beaten_down_arr(highs, close: float, max_drop: float = _MAX_DROP) -> bool:
    """Array fast-path for the scan hot loop: close vs 52-week peak high."""
    try:
        import numpy as _np
        h = _np.asarray(highs, dtype=float)
        if len(h) < 30 or close <= 0:
            return False
        peak = float(_np.nanmax(h))
        return peak > 0 and close <= peak * (1.0 - max_drop)
    except Exception:
        return False


def is_beaten_down(df, max_drop: float = _MAX_DROP) -> bool:
    """True if the latest close is more than max_drop below the 52-week
    high — i.e. price <= (1-max_drop) × high. Needs some history; too
    little data → not flagged (never guess). Accepts a DataFrame with a
    'close'/'high' column or a close array."""
    try:
        import numpy as _np
        if hasattr(df, "columns"):
            hi_src = df["high"] if "high" in df.columns else df["close"]
            close = float(df["close"].values[-1])
            highs = _np.asarray(hi_src.values[-_DROP_LOOKBACK:], dtype=float)
        else:
            arr = _np.asarray(df, dtype=float)
            close = float(arr[-1])
            highs = arr[-_DROP_LOOKBACK:]
        if len(highs) < 30 or close <= 0:
            return False
        peak = float(_np.nanmax(highs))
        if peak <= 0:
            return False
        return close <= peak * (1.0 - max_drop)
    except Exception:
        return False


# Signal metadata: label shown to user, category for filtering, base score
SIGNAL_META = {
    "BREAKOUT_52W":    ("52-week high breakout",       "Breakout",    30),
    "BREAKOUT_RES":    ("Resistance break on volume",  "Breakout",    26),
    "GOLDEN_CROSS":    ("Golden cross (50/200 SMA)",   "Breakout",    22),
    "VOL_SQUEEZE":     ("Squeeze breakout",            "Breakout",    22),
    "VCP":             ("VCP — tightening base",       "Pattern",     28),
    "FLAT_BASE":       ("Flat base near breakout",     "Pattern",     24),
    "CUP_HANDLE":      ("Cup & handle",                "Pattern",     24),
    "HIGH_TIGHT_FLAG": ("High tight flag",             "Pattern",     30),
    "ASC_TRIANGLE":    ("Ascending triangle",          "Pattern",     24),
    "DOUBLE_BOTTOM":   ("Double bottom",               "Pattern",     22),
    "PRE_BREAKOUT":    ("Breakout ke kareeb",          "PreBreakout", 26),
    "ACCUMULATION":    ("Smart-money accumulation",    "PreBreakout", 24),
    "DELIVERY_SPIKE":  ("Delivery buying rising",      "PreBreakout", 18),
    "NR7_COIL":        ("Coiled — tightest day in 7",  "PreBreakout", 14),
    "POCKET_PIVOT":    ("Pocket pivot volume",         "PreBreakout", 20),
    "MOMENTUM":        ("Strong momentum",             "Momentum",    20),
    "PULLBACK_SUPPORT": ("Uptrend pullback to support", "Pullback",   26),
}


@dataclass
class StockSignal:
    symbol: str
    price: float
    change_pct: float          # 1-day change
    momentum_5d: float
    rsi: float
    volume_ratio: float
    signals: list[str]                     # signal keys from SIGNAL_META
    reasons: list[str] = field(default_factory=list)   # plain-English evidence
    score: float = 0.0                     # composite 0-100
    entry: float = 0.0
    stop: float = 0.0
    target: float = 0.0
    verdict: str = "WATCH"                 # BUY | WATCH
    pivot_distance_pct: float = 0.0        # how far below the pivot (0 = at/above)
    breakout_grade: str = ""               # A | B | "" — confirmed-breakout quality
    breakout_conviction: float = 0.0       # 0-100 — force behind the break
    avg_vol20: float = 0.0                  # 20-day avg volume — sniper vol pacing
    above_sma50: bool = False              # trend flags — breadth ka raw data
    above_sma200: bool = False
    chase_risk: bool = False               # extension guard fired — WATCH is a
                                            # safety demotion, not a low-score
                                            # miss; downstream enrichment (buzz/
                                            # earnings) must never promote it back

    @property
    def categories(self) -> set[str]:
        return {SIGNAL_META[s][1] for s in self.signals if s in SIGNAL_META}

    @property
    def signal_labels(self) -> list[str]:
        return [SIGNAL_META[s][0] for s in self.signals if s in SIGNAL_META]

    @property
    def risk_reward(self) -> float:
        risk = self.entry - self.stop
        return (self.target - self.entry) / risk if risk > 0 else 0.0


def _load_calibration() -> dict[str, float]:
    """
    Score multipliers from the walk-forward backtest — signals that
    historically lose get their weight cut automatically; proven ones
    get a boost. 1.0 for everything until a backtest has been run.
    """
    mult: dict[str, float] = {}
    try:
        from scan.signal_backtest import load_report
        rep = load_report()
        if not rep:
            return mult
        for sig, s in rep.get("signals", {}).items():
            if s.get("trades", 0) < 20:
                continue                       # not enough evidence — leave at 1.0
            exp = s.get("expectancy_r", 0.0)
            if exp >= 0.30:
                mult[sig] = 1.25
            elif exp >= 0.10:
                mult[sig] = 1.0
            elif exp >= -0.10:
                mult[sig] = 0.75
            else:
                mult[sig] = 0.45               # proven loser — heavily discounted
    except Exception:
        pass
    return mult


class UnifiedScanner:
    """Scans the full universe from the bulk cache. Compute-only, no network."""

    def __init__(self, max_workers: int = 8):
        self._max_workers = max_workers
        self._calib = _load_calibration()
        # Live edge — blend our OWN forward-tested outcomes into the backtest
        # calibration. CONSERVATIVE: take the lower multiplier of the two, so
        # live data can DEMOTE a signal that's leaking money but never inflate
        # one the backtest already distrusts. Evidence-gated (≥30 outcomes);
        # no tracked data → no change. This is what actually lifts expectancy:
        # proven-negative signals stop earning BUY weight.
        try:
            from scan.live_edge import live_calibration
            for sig, m in live_calibration().items():
                self._calib[sig] = min(self._calib.get(sig, 1.0), m)
        except Exception as exc:
            log.debug("live_calibration_skip", error=str(exc))
        self._nifty_ret30 = 0.0        # index benchmark for relative strength
        # Regime-conditional calibration is set per-scan in scan() (NSE tape) —
        # left empty here so US/search paths (which call _analyze directly)
        # never inherit an NSE regime or pay for computing one.
        self._regime = ""
        self._regime_calib: dict[str, float] = {}
        if self._calib:
            log.info("scanner_calibrated", signals=len(self._calib))

    def scan(self, symbols: list[str], progress=None, *, skip_prefetch: bool = False) -> list[StockSignal]:
        from scan.bulk_fetcher import prefetch, get_cached, cached_symbols
        import time as _time

        def _report(current: int, total: int) -> None:
            if not callable(progress):
                return
            try:
                progress(int(current), int(total))
            except Exception:
                pass

        if skip_prefetch:
            # Caller already warmed official history (market-ops). Ensure the
            # in-process bhav flag is on so get_cached/cached_symbols use it.
            try:
                from scan import bulk_fetcher as bf

                with bf._lock:
                    if not bf._bhav_ok:
                        try:
                            from data.bhavcopy_store import store_symbols as _store_symbols

                            if _store_symbols():
                                bf._bhav_ok = True
                        except Exception:
                            pass
            except Exception:
                pass
        else:
            prefetch(symbols, progress=progress)

        available = [s for s in symbols if s in set(cached_symbols())]
        total = max(1, len(available) or len(symbols) or 1)
        _report(0, total)

        self._nifty_ret30 = _nifty_return_30d()      # RS benchmark, once/scan
        # Current market tape → regime-conditional demotion for this scan only.
        # compute_regime is cached (15 min) + streamlit-free; one call/scan.
        try:
            from core.regime_engine import compute_regime
            self._regime = str(getattr(compute_regime(), "market_regime", "")
                               or "")
            from scan.live_edge import regime_calibration
            self._regime_calib = regime_calibration(self._regime)
            if self._regime_calib:
                log.info("scanner_regime_calibrated", regime=self._regime,
                         signals=len(self._regime_calib))
        except Exception as exc:
            log.debug("regime_calib_skip", error=str(exc))
            self._regime, self._regime_calib = "", {}
        log.info("unified_scan_start", requested=len(symbols), with_data=len(available))
        _report(0, total)

        results: list[StockSignal] = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(self._analyze, sym, get_cached(sym)): sym
                       for sym in available}
            done = 0
            total = len(futures) or 1
            last_report = _time.monotonic()
            for fut in as_completed(futures):
                done += 1
                now = _time.monotonic()
                # Report often enough that a 2k-universe scan never looks frozen.
                if done == 1 or done == total or done % 10 == 0 or (now - last_report) >= 1.5:
                    _report(done, total)
                    last_report = now
                try:
                    r = fut.result()
                    if r and r.signals:
                        results.append(r)
                except Exception as exc:
                    log.debug("unified_analyze_failed", symbol=futures[fut], error=str(exc))

        results.sort(key=lambda r: r.score, reverse=True)
        log.info("unified_scan_done", scanned=len(available), with_signals=len(results))
        _report(total, total)
        return results

    # ── Per-stock analysis ────────────────────────────────────────────────────

    def _analyze(self, symbol: str, df: Optional[pd.DataFrame]) -> Optional[StockSignal]:
        if df is None or len(df) < 60:
            return None
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float) if "high" in df.columns else close
        low = df["low"].values.astype(float) if "low" in df.columns else close
        vol = df["volume"].values.astype(float) if "volume" in df.columns else None
        deliv = (df["deliv_per"].values.astype(float)
                 if "deliv_per" in df.columns else None)

        price = float(close[-1])
        if price < 20:               # skip penny stocks
            return None
        if vol is not None and np.nanmean(vol[-20:]) * price < 1e7:
            return None              # < ₹1 cr avg daily turnover — illiquid
        # Falling-knife guard: down >60% from 52-week high → ignore entirely
        hi = high if high is not None else close
        if is_beaten_down_arr(hi[-_DROP_LOOKBACK:], price):
            return None

        # Honest dating — signals are computed on the latest EOD session
        try:
            session = pd.Timestamp(df.index[-1]).strftime("%d %b")
        except Exception:
            session = ""
        sess_tag = f" ({session} close)" if session else ""

        chg = (close[-1] / close[-2] - 1) * 100 if len(close) > 1 else 0.0
        mom5 = (close[-1] / close[-6] - 1) * 100 if len(close) > 5 else 0.0
        rsi = _rsi(close)
        vratio = 1.0
        avg_vol20 = 0.0
        if vol is not None and len(vol) > 21:
            avg = np.nanmean(vol[-21:-1])
            vratio = float(vol[-1] / avg) if avg > 0 else 1.0
            avg_vol20 = float(np.nanmean(vol[-20:]))    # sniper vol pacing
        atr = _atr(high, low, close)
        sma50 = close[-50:].mean() if len(close) >= 50 else price

        signals: list[str] = []
        reasons: list[str] = []

        # ── Momentum ──────────────────────────────────────────────────────────
        mom_score = (_norm(mom5, -5, 10) * 0.40 + _norm(rsi, 30, 70) * 0.35
                     + _norm(vratio, 0.5, 3.0) * 0.25) * 100
        if mom_score >= 65 and 50 <= rsi < 78:
            signals.append("MOMENTUM")
            reasons.append(f"Up {mom5:+.1f}% in 5 days on {vratio:.1f}× volume")

        # ── Breakouts — CONFIRMED only (ATR-cleared + volume-backed) ──────────
        # A marginal poke is not a buy; it becomes a PRE_BREAKOUT watch so the
        # trader waits for the close to confirm instead of chasing a false break.
        breakout_grade = ""
        breakout_conv = 0.0
        if len(high) > 60:
            hi52 = float(np.max(high[:-1]))
            res20 = float(np.max(high[-21:-1]))
            level, tag = (hi52, "52-week high") if price > hi52 else (res20, "resistance")
            clv = close_location_value(close[-1], high[-1], low[-1])
            pct_below_high = max(0.0, (hi52 - price) / hi52 * 100) if hi52 > 0 else 0.0
            ok, grade, note = grade_breakout(level=level, price=price, atr=atr,
                                             vratio=vratio, day_change=chg,
                                             clv=clv, rsi=rsi)
            if ok:
                # Conviction — who is behind the break? (the full vault)
                d_now = float(np.nanmean(deliv[-5:])) if (
                    deliv is not None and len(deliv) >= 5) else None
                d_base = float(np.nanmean(deliv[-30:-5])) if (
                    deliv is not None and len(deliv) >= 30) else None
                stk_ret30 = ((close[-1] / close[-31] - 1) * 100
                             if len(close) > 31 else 0.0)
                rs_outperf = stk_ret30 - self._nifty_ret30
                above_200 = len(close) >= 200 and price > close[-200:].mean()
                base_q, base_note = base_tightness(high, low, price, sma50)
                conv, conv_factors = breakout_conviction(
                    vratio=vratio, deliv_now=d_now, deliv_base=d_base,
                    rs_outperf=rs_outperf, above_50=price > sma50,
                    above_200=above_200, base_q=base_q, clv=clv,
                    pct_below_high=pct_below_high)
                if base_note:
                    conv_factors.append(base_note)
                if conv >= _MIN_BREAKOUT_CONVICTION:
                    breakout_grade = grade
                    breakout_conv = conv
                    key = "BREAKOUT_52W" if level == hi52 else "BREAKOUT_RES"
                    signals.append(key)
                    fac = (" · " + ", ".join(conv_factors)) if conv_factors else ""
                    reasons.append(f"[{grade} · conviction {conv:.0f}] Broke "
                                   f"₹{level:,.0f} {tag} — {note}{fac}{sess_tag}")
                else:
                    # Confirmed break but weak conviction → watch, not buy
                    if "PRE_BREAKOUT" not in signals:
                        signals.append("PRE_BREAKOUT")
                        reasons.append(f"₹{level:,.0f} {tag} toda par conviction "
                                       f"kam ({conv:.0f}/100) — pack ka wait{sess_tag}")
            elif price > level and note:
                # Cleared the level but not cleanly → watch, not buy
                if "PRE_BREAKOUT" not in signals:
                    signals.append("PRE_BREAKOUT")
                    reasons.append(f"₹{level:,.0f} {tag} pe hai par {note}{sess_tag}")

        if len(close) >= 201:
            s50, s200 = close[-50:].mean(), close[-200:].mean()
            s50p, s200p = close[-55:-5].mean(), close[-205:-5].mean()
            if s50 > s200 and s50p <= s200p:
                signals.append("GOLDEN_CROSS")
                reasons.append("50-day average just crossed above 200-day")

        if len(close) >= 40:
            width_now = _bb_width(close, -1)
            width_prev = np.mean([_bb_width(close, i) for i in range(-11, -1)])
            upper = close[-20:].mean() + 2 * close[-20:].std()
            if width_prev > 0 and width_now > width_prev * 1.3 and price > upper:
                signals.append("VOL_SQUEEZE")
                reasons.append("Tight squeeze just expanded upward")

        # ── Chart patterns (structure — breakout ahead) ───────────────────────
        patterns = _detect_patterns(close, high, low, vol)
        pivot = 0.0
        for key, reason, pv in patterns[:2]:          # max 2 patterns per stock
            signals.append(key)
            reasons.append(reason)
            if pv and not pivot:
                pivot = pv

        # ── Pre-breakout evidence ─────────────────────────────────────────────
        pivot_ref = pivot or (float(np.max(high[-41:-1])) if len(high) > 41 else 0.0)
        pivot_dist = ((pivot_ref - price) / price * 100) if pivot_ref > price else 0.0

        # About to break: just below pivot with volume building
        if pivot_ref > price and pivot_dist <= 2.5 and vol is not None and len(vol) >= 20:
            vol_building = np.nanmean(vol[-5:]) > np.nanmean(vol[-20:-5]) * 1.15
            has_structure = any(k in signals for k in
                                ("VCP", "FLAT_BASE", "CUP_HANDLE", "ASC_TRIANGLE"))
            if vol_building and (has_structure or price > sma50):
                signals.append("PRE_BREAKOUT")
                reasons.append(f"Sirf {pivot_dist:.1f}% neeche hai ₹{pivot_ref:,.0f} "
                               f"ke breakout se — volume badh raha hai")

        # Accumulation: volume dry-up + up-day volume dominance near highs
        if vol is not None and len(vol) >= 60 and len(close) >= 60:
            dryup = np.nanmean(vol[-10:]) < np.nanmean(vol[-60:-10]) * 0.70
            rets = np.diff(close[-16:])
            v15 = vol[-15:]
            up_vol = float(np.nansum(v15[rets > 0])) if len(v15) == len(rets) else 0.0
            dn_vol = float(np.nansum(v15[rets < 0])) if len(v15) == len(rets) else 1.0
            hi60, lo60 = float(np.max(high[-60:])), float(np.min(low[-60:]))
            in_top_third = price >= lo60 + (hi60 - lo60) * 0.66 if hi60 > lo60 else False
            if dryup and dn_vol > 0 and up_vol / dn_vol >= 1.4 and in_top_third:
                signals.append("ACCUMULATION")
                reasons.append(f"Sellers sookh gaye — buying volume {up_vol/dn_vol:.1f}× "
                               f"selling volume, price highs ke paas tikka hua")

        # Delivery spike: NSE delivery %% rising (only bhavcopy data has this)
        if deliv is not None and len(deliv) >= 30:
            d_recent = float(np.nanmean(deliv[-5:]))
            d_base = float(np.nanmean(deliv[-30:-5]))
            if d_base > 0 and d_recent > d_base * 1.25 and d_recent >= 40:
                signals.append("DELIVERY_SPIKE")
                reasons.append(f"Delivery {d_recent:.0f}% (normal {d_base:.0f}%) — "
                               f"log hold karne ke liye khareed rahe hain")

        # NR7: narrowest daily range in 7 sessions inside an uptrend near highs
        if len(high) >= 45 and price > sma50:
            ranges = high[-7:] - low[-7:]
            near_high = price >= float(np.max(high[-41:-1])) * 0.95
            if ranges[-1] > 0 and ranges[-1] == ranges.min() and near_high:
                signals.append("NR7_COIL")
                reasons.append("Aaj 7 dinon ka sabse tight din — spring coil ho chuki hai")

        # Pocket pivot: up-day volume beats every down-day volume of last 10
        if vol is not None and len(vol) >= 12 and chg > 0 and price > sma50:
            rets10 = np.diff(close[-11:])
            down_vols = vol[-10:][rets10 < 0]
            if len(down_vols) > 0 and vol[-1] > float(np.max(down_vols)) \
                    and price >= float(np.max(high[-41:-1])) * 0.92:
                signals.append("POCKET_PIVOT")
                reasons.append("Aaj ki buying volume ne pichhle 10 din ki har "
                               "selling volume ko hara diya")

        # ── Buyable pullback (NOT at 52w high — the other kind of setup) ──────
        pb_ok, pb_reason, pb_pivot = detect_pullback_support(
            close, high, low, vol, atr)
        if pb_ok:
            signals.append("PULLBACK_SUPPORT")
            reasons.append(pb_reason + sess_tag)
            if pb_pivot and not pivot:
                pivot = pb_pivot

        if not signals:
            return None
        signals = signals[:6]
        reasons = reasons[:6]

        # ── Trade plan ────────────────────────────────────────────────────────
        # Refresh pivot_ref: the pullback-support block (above) can set a
        # more precise swing-high `pivot` when the chart-pattern pivot came
        # up empty, but that happens AFTER pivot_ref was first derived — so
        # without this, a PULLBACK_SUPPORT-only setup's entry/stop/target
        # silently fell back to the generic 41-day-high instead of the
        # pullback's own (tighter, more relevant) pivot.
        pivot_ref = pivot or pivot_ref
        entry = pivot_ref if pivot_ref > price else price
        stop = round(entry - 2 * atr, 1) if atr > 0 else round(entry * 0.95, 1)
        target = round(entry + 4 * atr, 1) if atr > 0 else round(entry * 1.10, 1)

        # ── Composite score (backtest + live + regime-conditional weights) ────
        # Each signal weight is scaled by (a) global calibration (backtest∧live)
        # and (b) a DEMOTE-ONLY regime factor: if this signal leaks in the
        # current tape, cut it — but never let the regime factor inflate a
        # weight (min 1.0), so it can only add caution, never chase.
        base = sum(SIGNAL_META[s][2] * self._calib.get(s, 1.0)
                   * min(1.0, self._regime_calib.get(s, 1.0)) for s in signals)
        _above50 = price > sma50
        _above200 = len(close) >= 200 and price > close[-200:].mean()
        trend_bonus = 10 if _above200 else 0
        score = min(100.0, base + trend_bonus + mom_score * 0.2)

        verdict = "BUY" if (score >= 55 and len(signals) >= 2) or any(
            s in ("BREAKOUT_52W", "HIGH_TIGHT_FLAG") for s in signals) else "WATCH"

        # ── Extension guard — DON'T CHASE ─────────────────────────────────────
        # A stock already up big in 5 days and stretched far above its 20-EMA,
        # WITHOUT a confirmed breakout, is a chase (wide stop, poor entry) —
        # exactly TATVA-type: +14% in 5d, only a marginal break. Downgrade it
        # to a WATCH and say wait for the pullback (which PULLBACK_SUPPORT then
        # catches). A CONFIRMED breakout is exempt — that's a real trigger.
        # Two views of "extended" — a stock only needs to fail ONE:
        #   • 20-EMA: short-term stretched (the fast, mean-reverting view)
        #   • 50-DMA: absolute/late-stage extension — catches the STEADY
        #     grinder that stays glued to its 20-EMA the whole way up (the
        #     ADANIENSOL case: +18% off the base but only ~5% above the
        #     20-EMA, so the 20-EMA view alone said "fine").
        ema20_now = _ema_np(close, 20)
        ext_pct = (price / ema20_now - 1) * 100 if ema20_now else 0.0
        ext50_pct = (price / sma50 - 1) * 100 if sma50 else 0.0
        chase_risk = False
        _short_stretched = ext_pct > 10 and mom5 > 10
        _far_from_base = _EXT_ABOVE_SMA50 > 0 and ext50_pct > _EXT_ABOVE_SMA50
        if (_short_stretched or _far_from_base) and not breakout_grade:
            why = (f"{ext_pct:.0f}% above 20-EMA, +{mom5:.0f}% in 5 din"
                   if _short_stretched
                   else f"{ext50_pct:.0f}% above 50-DMA — late-stage, base se door")
            reasons.insert(
                0, f"⚠ Extended ({why}) — pullback ka wait, abhi chase mat karo")
            # Flag it regardless of the current verdict — a PRE_BREAKOUT that's
            # already WATCH still needs chase_risk set so the sniper skips it
            # (the ADANIENSOL path: extended pre-breakout, never a "BUY" to
            # demote, but must NOT be sniped). chase_risk is a don't-chase
            # SAFETY flag downstream (sniper skip + conviction WATCH-floor) —
            # buzz/earnings enrichment must never promote it back. It also
            # means reasons[0] is now an unpaired leading warning, so
            # zip(signals, reasons) consumers skip reasons[0] when it's set.
            chase_risk = True
            if verdict == "BUY":
                verdict = "WATCH"

        # ── Falling-knife / momentum-rollover guard — DON'T CATCH A KNIFE ─────
        # A breakout/momentum entry happens on strength. If the stock is red
        # today, OR its RSI has rolled over (turning down from a few sessions
        # ago), it is NOT breaking out — demote any BUY to WATCH and say wait
        # for it to stabilise. A pullback-to-support setup is exempt: buying a
        # controlled dip to a rising EMA is exactly its (evidence-backed) job.
        rsi_prev = _rsi(close[:-3]) if len(close) > 17 else rsi
        _falling_today = chg <= _FALLING_DAY_PCT
        _rsi_rolling = rsi < rsi_prev - _RSI_ROLLOVER_DROP
        _is_pullback = "PULLBACK_SUPPORT" in signals
        if (_falling_today or _rsi_rolling) and not _is_pullback:
            tag = (f"aaj {chg:+.1f}%" if _falling_today
                   else f"RSI roll-over {rsi_prev:.0f}→{rsi:.0f}")
            note = (f"⚠ Girta hua ({tag}) — falling knife, fresh BUY nahi; "
                    f"pehle rukna/base banna dekho")
            # Keep exactly ONE leading warning at reasons[0] so the single
            # reasons[0]-skip in conviction.py stays aligned: if the extension
            # guard already put a headline there, append instead of insert.
            if chase_risk:
                reasons.append(note)
            else:
                reasons.insert(0, note)
                chase_risk = True
            if verdict == "BUY":
                verdict = "WATCH"

        return StockSignal(
            symbol=symbol, price=round(price, 2), change_pct=round(chg, 2),
            momentum_5d=round(mom5, 2), rsi=round(rsi, 1),
            volume_ratio=round(vratio, 2), signals=signals, reasons=reasons,
            score=round(score, 1), entry=round(entry, 1), stop=stop,
            target=target, verdict=verdict, chase_risk=chase_risk,
            pivot_distance_pct=round(pivot_dist, 2),
            breakout_grade=breakout_grade,
            breakout_conviction=breakout_conv,
            avg_vol20=round(avg_vol20, 0),
            above_sma50=bool(_above50), above_sma200=bool(_above200),
        )


# ── Pattern detection ─────────────────────────────────────────────────────────

def _detect_patterns(close, high, low, vol) -> list[tuple[str, str, float]]:
    """All structural patterns found, strongest first: [(key, reason, pivot)]."""
    found: list[tuple[str, str, float]] = []
    price = float(close[-1])

    # High tight flag: 80%+ run in ~8 weeks, then <25% pullback flag
    if len(close) >= 60:
        run = close[-15] / close[-55] - 1
        flag_hi = float(np.max(high[-15:]))
        flag_lo = float(np.min(low[-15:]))
        if run >= 0.8 and flag_hi > 0 and (flag_hi - flag_lo) / flag_hi <= 0.25 \
                and price >= flag_hi * 0.93:
            found.append(("HIGH_TIGHT_FLAG",
                          f"Ran {run*100:.0f}% then tight flag — rare leader", flag_hi))

    # VCP: successive pullbacks contracting, price within 5% of pivot
    if len(close) >= 120:
        pullbacks = _pullback_depths(high, low, lookback=120)
        if len(pullbacks) >= 2 and all(
                pullbacks[i] > pullbacks[i + 1] for i in range(len(pullbacks) - 1)) \
                and pullbacks[-1] <= 12:
            pivot = float(np.max(high[-40:]))
            if price >= pivot * 0.95:
                seq = " → ".join(f"{p:.0f}%" for p in pullbacks)
                found.append(("VCP",
                              f"Pullbacks shrinking ({seq}) — coiling near ₹{pivot:,.0f}",
                              pivot))

    # Ascending triangle: flat ceiling (3+ touches) + rising lows
    if len(high) >= 45:
        h40, l40 = high[-40:], low[-40:]
        ceiling = float(np.max(h40))
        touches = int(np.sum(h40 >= ceiling * 0.985))
        lows_rising = float(np.min(l40[:20])) < float(np.min(l40[20:]))
        if touches >= 3 and lows_rising and price >= ceiling * 0.96:
            found.append(("ASC_TRIANGLE",
                          f"Har dip pe buyers upar aa rahe — ceiling ₹{ceiling:,.0f} "
                          f"ko {touches} baar test kiya", ceiling))

    # Flat base: 30+ days within a 12% range, price in top third
    if len(close) >= 45:
        base_hi = float(np.max(high[-35:]))
        base_lo = float(np.min(low[-35:]))
        depth = (base_hi - base_lo) / base_hi if base_hi > 0 else 1
        if depth <= 0.12 and price >= base_hi * 0.96:
            found.append(("FLAT_BASE",
                          f"7-week flat base, sitting at ₹{base_hi:,.0f} ceiling",
                          base_hi))

    # Double bottom: two lows within 3%, ≥15 sessions apart, price near neckline
    if len(low) >= 90:
        l90 = low[-90:]
        i1 = int(np.argmin(l90[:45]))
        i2 = 45 + int(np.argmin(l90[45:]))
        lo1, lo2 = float(l90[i1]), float(l90[i2])
        if lo1 > 0 and abs(lo1 - lo2) / lo1 <= 0.03 and (i2 - i1) >= 15:
            neckline = float(np.max(high[-90:][i1:i2 + 1]))
            depth_ok = neckline > max(lo1, lo2) * 1.08
            # price near the neckline only — far above = pattern already done
            if depth_ok and neckline * 0.95 <= price <= neckline * 1.05:
                found.append(("DOUBLE_BOTTOM",
                              f"W-pattern bana — do baar ₹{lo1:,.0f} se wapas uchhla, "
                              f"neckline ₹{neckline:,.0f}", neckline))

    # Cup & handle proxy: prior high, rounded 15-35% cup, small handle near rim.
    # Price must be NEAR the rim (±8%) — far above means the pattern already
    # played out long ago (stale signal).
    if len(close) >= 130:
        left_rim = float(np.max(high[-130:-65]))
        cup_low = float(np.min(low[-90:-20]))
        cup_depth = (left_rim - cup_low) / left_rim if left_rim > 0 else 1
        handle_lo = float(np.min(low[-15:]))
        handle_ok = handle_lo >= cup_low + (left_rim - cup_low) * 0.5
        if 0.15 <= cup_depth <= 0.35 and handle_ok \
                and left_rim * 0.92 <= price <= left_rim * 1.08:
            found.append(("CUP_HANDLE",
                          f"Cup formed, handle near ₹{left_rim:,.0f} rim", left_rim))

    return found


def _pullback_depths(high, low, lookback: int = 120) -> list[float]:
    """Depths (%) of successive swing pullbacks over the lookback window."""
    h = high[-lookback:]
    l = low[-lookback:]
    depths, seg = [], max(20, lookback // 4)
    for i in range(0, lookback - seg + 1, seg):
        hi = float(np.max(h[i:i + seg]))
        lo = float(np.min(l[i:i + seg]))
        if hi > 0:
            depths.append((hi - lo) / hi * 100)
    return depths


# ── Math helpers ──────────────────────────────────────────────────────────────

def _rsi(close: np.ndarray, period: int = 14) -> float:
    if len(close) < period + 1:
        return 50.0
    delta = np.diff(close[-(period + 1):])
    gains, losses = np.where(delta > 0, delta, 0.0), np.where(delta < 0, -delta, 0.0)
    ag, al = gains.mean(), losses.mean()
    return 100.0 if al == 0 else 100 - 100 / (1 + ag / al)


def _atr(high, low, close, period: int = 14) -> float:
    if len(high) < period + 1:
        return 0.0
    tr = [max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
          for i in range(-period, 0)]
    return float(np.mean(tr))


def _bb_width(close: np.ndarray, idx: int, period: int = 20) -> float:
    seg = close[idx - period:idx] if idx != -1 else close[-period:]
    if len(seg) < period:
        return 0.0
    m, s = seg.mean(), seg.std()
    return (4 * s) / m if m > 0 else 0.0


def _norm(val: float, lo: float, hi: float) -> float:
    return max(0.0, min(1.0, (val - lo) / (hi - lo))) if hi != lo else 0.5


def _ema_np(arr, span: int) -> float:
    """Latest EMA value of a numpy array (no pandas in the hot loop)."""
    a = arr[-(span * 3):] if len(arr) > span * 3 else arr
    if len(a) == 0:
        return 0.0
    k = 2.0 / (span + 1)
    e = float(a[0])
    for x in a[1:]:
        e = float(x) * k + e * (1 - k)
    return e


def detect_pullback_support(close, high, low, vol, atr: float) -> tuple[bool, str, float]:
    """Buyable pullback: a Stage-2 uptrend stock that has PULLED BACK to a
    rising 20/50-EMA on drying volume and is tightening — a low-risk
    continuation entry that is NOT at a 52-week high. (The other setups
    hug the highs; this one deliberately does not.)

    Returns (found, reason, pivot). Pivot = the recent minor swing high
    the next leg would clear."""
    try:
        import numpy as _np
        if len(close) < 60:
            return False, "", 0.0
        price = float(close[-1])
        ema20 = _ema_np(close, 20)
        ema50 = _ema_np(close, 50)
        ema50_prior = _ema_np(close[:-10], 50)
        sma200 = float(_np.mean(close[-200:])) if len(close) >= 200 else 0.0
        # Stage-2 uptrend: RISING 50-EMA + above the 200-SMA. Price may dip
        # slightly BELOW the 50-EMA on the pullback (that's the point) — allow
        # up to 3% below; a deeper break isn't a pullback.
        if not (ema50 > ema50_prior and price > ema50 * 0.97):
            return False, "", 0.0
        if sma200 and price < sma200:
            return False, "", 0.0
        # Pulled back from a recent high — a real dip, not extended, not a break
        recent_high = float(_np.max(high[-30:]))
        off_high = (recent_high - price) / recent_high if recent_high else 0.0
        if not (0.03 <= off_high <= 0.18):
            return False, "", 0.0
        # Sitting ON support: within ~3.5% of the 20- or 50-EMA
        near = (abs(price - ema20) / ema20 <= 0.035 if ema20 else False) or \
               (abs(price - ema50) / ema50 <= 0.035 if ema50 else False)
        if not near:
            return False, "", 0.0
        # Healthy pullback: volume drying up + range tightening
        vol_dry = True
        if vol is not None and len(vol) >= 20:
            vol_dry = float(_np.mean(vol[-5:])) < float(_np.mean(vol[-20:]))
        rng_now = float(_np.mean(high[-5:] - low[-5:]))
        rng_prior = float(_np.mean(high[-20:-5] - low[-20:-5]))
        tightening = rng_prior > 0 and rng_now < rng_prior
        if not (vol_dry and tightening):
            return False, "", 0.0
        ema_lbl = "20-EMA" if (ema20 and abs(price - ema20) / ema20 <= 0.035) \
            else "50-EMA"
        return True, (f"Uptrend mein {off_high*100:.0f}% pullback, {ema_lbl} pe "
                      f"support, volume dry + tightening — buyable pullback"), recent_high
    except Exception:
        return False, "", 0.0
