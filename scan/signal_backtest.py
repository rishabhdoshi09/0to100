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

import hashlib
import json
import os as _os
from datetime import datetime, timezone
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from logger import get_logger
from core.runtime_paths import logs_dir

log = get_logger(__name__)

_OUT = logs_dir() / "signal_backtest.json"
MIN_SYMBOLS_TO_SAVE = 100

# Breakeven-trail level for the per-signal expectancy sim — mirrors the live
# autopilot's breakeven_trigger_pct (default 2.0). Backtesting WITH the trail
# measures the strategy the system actually trades, not a naked-stop strawman.
# Env-tunable so it can track a changed autopilot setting.
from core.evaluation_spec import BREAKEVEN_TRIGGER_PCT as _BT_BREAKEVEN_PCT  # noqa: E402
from core.evaluation_spec import HORIZON_SESSIONS as _EVAL_HORIZON  # noqa: E402
from core.evaluation_spec import simulate_trade  # noqa: E402

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
    res = simulate_trade(entry, stop, target, fwd_high, fwd_low, fwd_close, be_pct=be_pct)
    return res.outcome, res.r, res.entry_offset, res.exit_offset


def _emit_trade_record(on_trade, df, t, sym, r, outcome, gross_r, net_r,
                       cost_r, e_off, x_off, regime, calib_version,
                       *, fill_price=None, exit_price=None):
    """Assemble the immutable core of one ledger row and hand it to the sink. Only
    the fields KNOWN at the trade level are set here (dates, prices, R, cost,
    regime, signals); the gauntlet's sink enriches with benchmark/factor returns
    over [entry_dt, exit_dt] so nothing is recomputed downstream. `calib_version`
    is hashed once per run and passed in (not recomputed per trade)."""
    risk = r.entry - r.stop
    entry_dt = df.index[t + e_off] if e_off >= 0 else df.index[min(t, len(df) - 1)]
    exit_dt = df.index[t + x_off] if x_off >= 0 else entry_dt
    if exit_price is None:
        exit_price = r.entry + gross_r * risk        # exact when the fill was at entry
    sigs = tuple(r.signals) if getattr(r, "signals", None) else ()
    on_trade({
        "symbol": sym,
        "signals": sigs,
        "signal_id": sigs[0] if sigs else "",
        "entry_datetime": entry_dt,
        "exit_datetime": exit_dt,
        "entry_price": float(fill_price if fill_price else r.entry),
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


#: Bumped when the report gains fields the evidence gate relies on; the
#: supervisor treats an older report as due for a rerun.
# 3: adds exit_variants + feature_confirmation (pre-registered 2026-11-19). A
# schema change makes the supervisor rerun the walk-forward in the next
# closed-market window, so the out-of-sample verdicts arrive the first night
# after deploy instead of the following Saturday.
# 4: evidence_provenance (code SHA, eval spec, horizon, family-correction
# method) and per-row family correction; the live gate refuses anything older.
# 5: alpha_accounting - every report reserves a look in research.alpha_ledger
# before its family correction; the gate verifies the attestation.
REPORT_SCHEMA = 5
#: Weekly walk-forward reports re-test overlapping data about 52 times a year.
#: Holm bounds false admissions per REPORT; the annual union bound spends the
#: 5% family-wise budget across a year's reports (alpha / 52 each), so
#: P(any false admission in a calendar year) <= 5% under any dependence.
ANNUAL_FAMILY_ALPHA = 0.05
REPORTS_PER_YEAR = 52
REPORT_ALPHA = ANNUAL_FAMILY_ALPHA / REPORTS_PER_YEAR
#: The predeclared multiple-testing procedure for the live gate's family.
#: Holm (family-wise) rather than BH: a BUY admission is an individual
#: decision. BH controls the expected share of false discoveries, needs
#: positive dependence, and says nothing about one admitted row. Holm bounds
#: P(any false admission) at alpha under arbitrary dependence
#: (research/admission_stress.py; 2026-11-20 review).
FAMILY_METHOD = "Holm-Bonferroni (family-wise), one-sided edge > 0, alpha 0.05/52 per weekly report"
#: Exact signal combinations need their own sample before they make a claim.
COMBO_MIN_TRADES = 30
#: One-sided 95% lower bound: mean - 1.645 * clustered standard error.
LOWER_BOUND_Z = 1.645


def combo_key(signals) -> str:
    """Order-insensitive identity of an exact signal combination."""
    return "|".join(sorted({str(s) for s in (signals or ()) if s}))


def _add_cluster(clusters: dict, day: str, r: float) -> None:
    row = clusters.setdefault(day, [0.0, 0])
    row[0] += float(r)
    row[1] += 1


def clustered_mean_se(clusters: dict) -> tuple[float, float | None, int]:
    """(mean R, cluster-robust standard error, n clusters).

    Trades signalled on the same day share the market's next ten sessions, so
    they are not independent: the naive standard error would overstate
    certainty several-fold. Clusters are decision dates; the variance of the
    mean sums each date's residual total (CR1 small-sample factor G/(G-1)).
    """
    n = sum(int(v[1]) for v in clusters.values())
    g = len(clusters)
    if n <= 0:
        return 0.0, None, g
    mean = sum(float(v[0]) for v in clusters.values()) / n
    if g < 2:
        return mean, None, g
    resid = sum((float(v[0]) - mean * int(v[1])) ** 2 for v in clusters.values())
    var = resid / (n * n) * g / (g - 1)
    return mean, var ** 0.5, g


def _evidence_fields(stats: dict, session_index: dict[str, int] | None = None,
                     horizon: int = _EVAL_HORIZON) -> dict:
    """The BUY gate's evidence: mean R and a one-sided 95% lower bound.

    Trades on different days are NOT independent when their 10-session
    windows overlap, so the day-clustered error alone is several-fold too
    narrow. With the realised NSE session index the bound comes from
    research.overlap_inference (offset-subsample Minkowski bound, lag-h
    residual widening, Student t on independent blocks). Without an index the
    day-clustered error is widened by sqrt(horizon), the worst case for
    windows of that length; it is never left at the clustered value.
    """
    import math

    clusters = stats.get("clusters") or {}
    mean, se_cl, g = clustered_mean_se(clusters)
    out = {"expectancy_r_precise": round(mean, 4), "clusters": g,
           "se_r_day_clustered": None if se_cl is None else round(se_cl, 4)}
    indexed = session_index is not None and all(str(d)[:10] in session_index for d in clusters)
    if indexed and g >= 2:
        from research.overlap_inference import cells_from_clusters, overlap_robust_test

        rep = overlap_robust_test(cells_from_clusters(clusters, session_index), horizon)
        se, lower = rep["se"], rep["lower_95_one_sided"]
        out.update({"inference": "OVERLAP_ROBUST", "independent_sessions": rep["independent_sessions"],
                    "df": rep["df"], "variance_inflation": rep["variance_inflation"]})
        if rep["reason"] == "DEGENERATE_ZERO_DISPERSION":
            # Identical outcomes are a degenerate sample, not certainty: no bound.
            se, lower = None, None
            out["degenerate"] = True
        elif rep["t"] is not None and rep["df"]:
            from research.overlap_inference import _t_sf
            p_raw = _t_sf(float(rep["t"]), int(rep["df"]))           # unrounded, coherent with lower_r
            out["p_one_sided_raw"] = p_raw if math.isfinite(p_raw) else 1.0
    else:
        se = None if se_cl is None else se_cl * math.sqrt(max(1, int(horizon)))
        lower = None if se is None else mean - LOWER_BOUND_Z * se
        out["inference"] = "WORST_CASE_NO_SESSION_INDEX"
    out["se_r"] = None if se is None else round(se, 4)
    out["lower_r"] = None if lower is None else round(lower, 4)
    return out


# Pre-declared exit-construction experiment (research only; never changes the
# calibration, the evidence gate or any live/paper rule). The forward sample
# (2026-11-19, 24 sessions) pointed at the breakeven trail: V0 -0.086R,
# V3 +0.102R, none significant after BH. These four are fixed BEFORE the
# large-sample run so the weekly walk-forward can confirm or reject them.
# (name, breakeven trigger % or None = no trail, horizon multiple)
EXIT_VARIANTS = (
    ("V0_prod_be_h", "trail", 1),
    ("V1_no_trail_h", None, 1),
    ("V2_be_2h", "trail", 2),
    ("V3_no_trail_2h", None, 2),
)


def exit_variant_results(entry: float, stop: float, target: float, fwd, horizon: int) -> list[tuple[str, str, float]]:
    """Net R of the same signal under each pre-declared exit construction.

    ``fwd`` must hold at least 2 * horizon bars after the signal so every
    variant is judged on the same trades. NO_FILL trades are omitted.
    """
    out: list[tuple[str, str, float]] = []
    from core.costs import cost_in_r

    for name, trail, mult in EXIT_VARIANTS:
        window = fwd.iloc[: int(horizon) * int(mult)]
        sim = simulate_trade(
            entry, stop, target,
            window["high"].values, window["low"].values, window["close"].values,
            opens=window["open"].values if "open" in window.columns else None,
            be_pct=_BT_BREAKEVEN_PCT if trail else 1e9,
        )
        if sim.outcome == "NO_FILL":
            continue
        cost = cost_in_r((sim.fill_price - stop) / sim.fill_price, "CNC") if sim.fill_price > stop else 0.0
        out.append((name, sim.outcome, float(sim.r) - cost))
    return out


#: The production walk-forward configuration (the supervisor's weekly
#: run_in_background() defaults). It is part of the gate program, so a run at
#: any other configuration (a preview at another sample step, lookback or
#: universe size) reserves its look under a DIFFERENT program and can never
#: certify a BUY: configurations cannot be chosen after peeking at results
#: (GPT cross-review C-4).
PRODUCTION_RUN_CONFIG = {"sample_step": 5, "lookback_sessions": 250, "max_symbols": 800}


def gate_program(run_config: dict | None = None) -> dict:
    """The predeclared hypothesis program whose annual alpha budget the gate spends.

    Any change here (method, rulebook, horizon, sampling, run configuration,
    membership rule, report schema) is a NEW program, attested as such; all
    programs share the one global annual budget. The live gate verifies
    against the PRODUCTION configuration only.
    """
    cfg = dict(PRODUCTION_RUN_CONFIG if run_config is None else run_config)
    return {"family_method": FAMILY_METHOD, "eval_spec": _eval_spec_tag(_EVAL_HORIZON),
            "horizon_sessions": int(_EVAL_HORIZON), "sampling": SAMPLING, "report_schema": REPORT_SCHEMA,
            "run_config": {k: int(cfg[k]) for k in sorted(PRODUCTION_RUN_CONFIG) if k in cfg},
            "membership": (f"every catalogue signal (untested ones at p = 1) and every exact combination "
                           f"with >= {COMBO_MIN_TRADES} trades")}


def apply_family_correction(signals: dict, combos: dict, *, alpha: float = REPORT_ALPHA,
                            untested: int = 0) -> dict:
    """Holm-Bonferroni (family-wise) across every row the live BUY gate can use.

    The gate judges each signal and each exact combination at a one-sided 5%;
    with ~15 signals and ~30 combinations a global null would still "prove"
    one or two. Each row gets its one-sided p for "edge > 0" (overlap-robust t
    and df), and family_significant only when it survives Holm's step-down
    across the whole family: P(any false admission) <= alpha under ARBITRARY
    dependence. Rows that cannot be tested (not OVERLAP_ROBUST, zero or
    missing error) count with p = 1, so they enlarge the family and never
    pass. ``untested`` catalogue signals that produced no filled trade also
    count at p = 1: the family is fixed before outcomes are seen, so it cannot
    shrink to the signals that happened to trade (GPT cross-review C-4).
    Mutates and returns a summary.
    """
    from research.overlap_inference import _t_sf

    import math

    rows = [r for r in list((signals or {}).values()) + list((combos or {}).values()) if isinstance(r, dict)]
    pvals = []
    for r in rows:
        try:
            mean, se, df = float(r.get("expectancy_r_precise")), float(r.get("se_r") or 0.0), int(r.get("df") or 0)
        except (TypeError, ValueError):
            mean, se, df = 0.0, 0.0, 0
        ok = (r.get("inference") == "OVERLAP_ROBUST" and math.isfinite(mean) and math.isfinite(se)
              and se > 0 and df > 0 and not r.get("degenerate"))
        raw = r.get("p_one_sided_raw")
        if not ok:
            p = 1.0
        elif isinstance(raw, (int, float)) and math.isfinite(raw) and 0.0 <= raw <= 1.0:
            p = float(raw)                      # unrounded: coherent with the row's lower bound
        else:
            p = _t_sf(mean / se, df)
        if not math.isfinite(p):
            p = 1.0
        r["p_one_sided"] = round(float(p), 8)
        pvals.append(float(p))
    pvals.extend([1.0] * max(0, int(untested)))
    m = len(pvals)
    rejected = [False] * m
    for rank, i in enumerate(sorted(range(m), key=lambda i: pvals[i])):      # Holm step-down
        if pvals[i] < alpha / (m - rank):
            rejected[i] = True
        else:
            break
    for r, rej in zip(rows, rejected):
        r["family_significant"] = bool(rej)
        r["family_size"] = m
    return {"family_size": m, "alpha": alpha, "method": FAMILY_METHOD, "untested_members": max(0, int(untested)),
            "survivors": int(sum(bool(x) for x in rejected))}


def _alpha_ledger_lost() -> bool:
    """True when the alpha ledger is missing although looks were already spent through it.

    Evidence of earlier spending: the ledger's own sentinel, the last saved
    report's attestation, or the active calibration snapshot's provenance.
    """
    try:
        from research import alpha_ledger
        path = alpha_ledger.ledger_path()
        if path.exists():
            return False
        if alpha_ledger.provisioned_marker(path).exists():
            return True
        prior = json.loads(Path(_OUT).read_text(encoding="utf-8")) if Path(_OUT).exists() else {}
        acct = dict((prior.get("evidence_provenance") or {}).get("alpha_accounting") or {})
        if int(acct.get("look_index") or 0) >= 1:
            return True
        try:
            from scan.calibration_snapshot import load_current
            snap = dict(load_current() or {})
            sacct = dict((snap.get("backtest_provenance") or {}).get("alpha_accounting") or {})
            return int(sacct.get("look_index") or 0) >= 1
        except Exception:
            return False
    except Exception:
        return False


def _code_sha() -> str:
    """The build that produced the evidence: QT_BUILD_SHA on the host, else git HEAD."""
    sha = _os.environ.get("QT_BUILD_SHA", "").strip()
    if sha:
        return sha
    try:
        import subprocess
        out = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _day_clustered_reference(stats: dict) -> dict:
    mean, se, g = clustered_mean_se(stats.get("clusters") or {})
    return {"expectancy_r_precise": round(mean, 4), "se_r": None if se is None else round(se, 4),
            "clusters": g, "note": "reference only: ignores overlapping outcome windows"}


def exit_variant_report(exit_stats: dict, *, horizon: int = _EVAL_HORIZON,
                        session_index: dict[str, int] | None = None) -> dict:
    """Net expectancy per pre-declared exit variant, with honest inference.

    Same inference as confirmation_report: trade-weighted mean R with the
    overlap-robust error of research.overlap_inference (trades a few sessions
    apart share most of their 10-session window, and adjacent blocks still
    overlap), Student t on independent blocks, Benjamini-Hochberg across the
    variants. Fewer than MIN_INDEPENDENT_SESSIONS blocks, or zero dispersion,
    is INSUFFICIENT EVIDENCE. The day-clustered fields are reference only.
    """
    all_days = sorted({d for st in exit_stats.values() for d in (st.get("clusters") or {})})
    ordinals = session_index if session_index is not None else _session_ordinals(all_days)
    rows: dict = {}
    pvals, names = [], []
    for name, _trail, _mult in EXIT_VARIANTS:
        st = exit_stats.get(name)
        if not st or not st.get("trades"):
            continue
        row = {"trades": st["trades"], "win_rate": round(st["wins"] / st["trades"] * 100, 1),
               "scratch_rate": round(st.get("scratches", 0) / st["trades"] * 100, 1),
               "day_clustered": _day_clustered_reference(st)}
        rows[name] = row
        if ordinals is None:
            row.update({"verdict": "INSUFFICIENT EVIDENCE", "reason": "EXCHANGE_CALENDAR_UNAVAILABLE"})
            continue
        from research.overlap_inference import cells_from_clusters, overlap_robust_test

        rep = overlap_robust_test(cells_from_clusters(st.get("clusters") or {}, ordinals), horizon)
        row["independent_sessions"] = rep["independent_sessions"]
        if rep["mean"] is not None:
            row["block_mean_r"] = round(rep["mean"], 4)
        if rep["status"] != "OK":
            row.update({"verdict": "INSUFFICIENT EVIDENCE", "reason": rep["reason"]})
            continue
        row.update({"t": round(rep["t"], 3), "df": rep["df"], "p_two_sided": round(rep["p_two_sided"], 6),
                    "se_r": round(rep["se"], 4), "variance_inflation": rep["variance_inflation"],
                    "residual_dependence": rep["residual_dependence"],
                    "block_ci95_r": [round(x, 4) for x in rep["ci95"]]})
        pvals.append(rep["p_two_sided"])
        names.append(name)
    if pvals:
        from research.harness import benjamini_hochberg

        bh = benjamini_hochberg(pvals)
        rejected = [bool(x) for x in bh.get("rejected", [])]
        qvalues = [float(x) for x in bh.get("qvalues", [])]
        for i, name in enumerate(names):
            ok = rejected[i] if i < len(rejected) else False
            rows[name]["bh_significant"] = ok
            rows[name]["bh_q"] = round(qvalues[i], 6) if i < len(qvalues) else None
            rows[name]["verdict"] = ("POSITIVE" if ok and rows[name]["block_mean_r"] > 0 else
                                     "NEGATIVE" if ok and rows[name]["block_mean_r"] < 0 else
                                     "NOT_DISTINGUISHABLE_FROM_ZERO")
    return {
        "variants": rows,
        "status": "RESEARCH_ONLY",
        "inference": (f"trade-weighted mean R; overlap-robust error for {int(horizon)}-session windows "
                      f"(research.overlap_inference); Student t (df=blocks-1); BH across variants; "
                      f">= {MIN_INDEPENDENT_SESSIONS} independent blocks"),
        "note": ("Same trades, same fills, four pre-declared exits; costs included. A "
                 "construction change reaches PAPER only through a champion/challenger gate."),
    }


# Pre-registered confirmation (2026-11-19). On the forward tracked sample
# (4,804 BUY signals, 20 sessions, May-Jul 2026) three point-in-time features
# ranked 10-session excess returns after Benjamini-Hochberg: lower ATR %
# (IC +0.078, t 4.76), closer to the 52-week high (+0.089, t 3.36), less
# extended above the 50-DMA (-0.060, t -2.96). quality_score had no skill
# (+0.040, t 0.94). The walk-forward re-tests exactly these, plus one
# pre-declared composite, on sessions OUTSIDE that discovery window.
DISCOVERY_WINDOW = ("2026-05-26", "2026-07-30")
CONFIRM_FEATURES = ("low_atr", "near_high", "not_extended", "composite")
MIN_TRADES_PER_DAY_FOR_IC = 5


def confirmation_features(hist) -> dict[str, float] | None:
    """PIT features from bars up to (not including) the trade's first session.

    Signs are oriented so that HIGHER = predicted better, as pre-registered.
    """
    try:
        close = hist["close"].astype(float)
        high = hist["high"].astype(float)
        low = hist["low"].astype(float)
    except Exception:
        return None
    if len(close) < 60 or float(close.iloc[-1]) <= 0:
        return None
    last = float(close.iloc[-1])
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr_pct = float(tr.iloc[-14:].mean()) / last
    hi = float(close.iloc[-252:].max())
    sma50 = float(close.iloc[-50:].mean())
    if hi <= 0 or sma50 <= 0:
        return None
    return {"low_atr": -atr_pct, "near_high": -(hi - last) / hi, "not_extended": -(last / sma50 - 1.0)}


#: A verdict other than INSUFFICIENT EVIDENCE needs at least this many
#: non-overlapping horizon blocks, i.e. ~independent outcome windows.
from research.overlap_inference import MIN_INDEPENDENT_BLOCKS as MIN_INDEPENDENT_SESSIONS  # noqa: E402


def _average_ranks(values: list[float]) -> list[float]:
    """Fractional ranks: tied values share the mean of their positions."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = mean_rank
        i = j + 1
    return ranks


def spearman_ic(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation with tie-corrected (average) ranks.

    None when either side has no rank variation: a constant feature or a
    constant outcome on a day carries no ranking information, so it must not
    contribute an IC of 0 or ±1.
    """
    import math

    if len(xs) != len(ys) or len(xs) < 3:
        return None
    rx, ry = _average_ranks(list(xs)), _average_ranks(list(ys))
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx <= 1e-12 or syy <= 1e-12:
        return None
    return sxy / math.sqrt(sxx * syy)


def _session_ordinals(days: list[str]) -> dict[str, int] | None:
    """NSE session number of each day (holidays and weekends skipped).

    None unless the exchange holiday calendar is available AND covers every
    year the days span (nse_calendar.strict_session_ordinals): an uncovered
    year would count its holidays as sessions and inflate the independent-
    block count, so the caller must not reach a verdict.
    """
    try:
        from research.intelligence.data.nse_calendar import load_holidays, strict_session_ordinals

        return strict_session_ordinals(days, load_holidays())
    except Exception:
        return None


def window_is_exact(day_keys, t: int, horizon: int, session_index: dict[str, int] | None) -> bool:
    """True when bars t .. t+horizon-1 are exactly the next ``horizon`` official sessions after bar t-1.

    Positional bars silently stretch the outcome window over a symbol's
    suspension or missing bars (and an unofficial bar would shorten it). With
    the realised session index, a trade's window must be consecutive official
    sessions starting the session after its signal bar; otherwise the trade is
    not measured. Without an index (tests, no store) nothing is checked.
    """
    if not session_index:
        return True
    try:
        prev = session_index[day_keys[t - 1]]
        return all(session_index.get(day_keys[t + k]) == prev + 1 + k for k in range(int(horizon)))
    except (KeyError, IndexError):
        return False


def realised_session_index(dates) -> dict[str, int]:
    """Ordinals from the official bars actually present (the walk-forward's index).

    Every NSE session publishes a bhavcopy, so the union of bar dates across
    the sampled universe is the realised session list. A session missing from
    the data can only make two days look CLOSER than they are (more overlap
    assumed, fewer blocks counted): conservative. Dates the covered holiday
    calendar rules out (weekends, declared holidays) are dropped.
    """
    from datetime import date as _date

    try:
        from research.intelligence.data.nse_calendar import covered_years, is_session, load_holidays

        holidays = load_holidays()
        years = covered_years(holidays)
    except Exception:
        holidays, years, is_session = set(), set(), None
    keep = []
    for d in {str(x)[:10] for x in dates}:
        try:
            day = _date.fromisoformat(d)
        except ValueError:
            continue
        if day.weekday() >= 5:
            continue
        if is_session is not None and day.year in years and not is_session(day, holidays):
            continue
        keep.append(d)
    return {d: i for i, d in enumerate(sorted(keep))}


def confirmation_report(rows: list[dict], *, horizon: int = _EVAL_HORIZON,
                        session_index: dict[str, int] | None = None) -> dict:
    """Out-of-sample test of the pre-registered features (discovery window excluded).

    Inference unit: daily ICs (tie-corrected Spearman, feature vs net R),
    keyed by NSE session. Days a few sessions apart share most of their
    outcome window, and consecutive non-overlapping blocks still share the
    sessions at their boundary, so the error is the overlap-robust bound of
    research.overlap_inference (Student t, df = independent blocks - 1),
    Benjamini-Hochberg across features. A verdict other than INSUFFICIENT
    EVIDENCE needs >= MIN_INDEPENDENT_SESSIONS blocks and non-zero
    dispersion: identical daily ICs are a degenerate sample, never decisive.
    """
    import math

    lo, hi = DISCOVERY_WINDOW
    usable = [r for r in rows if not (lo <= str(r["day"]) <= hi)]
    by_day: dict[str, list[dict]] = {}
    for r in usable:
        by_day.setdefault(str(r["day"])[:10], []).append(r)
    days = {d: rs for d, rs in by_day.items() if len(rs) >= MIN_TRADES_PER_DAY_FOR_IC}
    ordinals = session_index if session_index is not None else _session_ordinals(sorted(days))

    results, pvals, names = {}, [], []
    for feat in CONFIRM_FEATURES:
        daily: list[tuple[str, float]] = []
        for day, rs in sorted(days.items()):
            if feat == "composite":
                parts = [_average_ranks([float(r[f]) for r in rs]) for f in ("low_atr", "near_high", "not_extended")]
                xs = [sum(part[i] for part in parts) for i in range(len(rs))]
            else:
                xs = [float(r[feat]) for r in rs]
            ic = spearman_ic(xs, [float(r["r"]) for r in rs])
            if ic is not None and math.isfinite(ic):
                daily.append((day, ic))
        res: dict = {"days_with_ic": len(daily)}
        if ordinals is None:
            res.update({"independent_sessions": 0, "verdict": "INSUFFICIENT EVIDENCE",
                        "reason": "EXCHANGE_CALENDAR_UNAVAILABLE"})
            results[feat] = res
            continue
        from research.overlap_inference import cells_from_pairs, overlap_robust_test

        rep = overlap_robust_test(cells_from_pairs((ordinals[d], ic) for d, ic in daily if d in ordinals), horizon)
        res["independent_sessions"] = rep["independent_sessions"]
        if rep["mean"] is not None:
            res["mean_ic"] = round(rep["mean"], 4)
        if rep["status"] != "OK":
            res.update({"verdict": "INSUFFICIENT EVIDENCE", "reason": rep["reason"]})
            results[feat] = res
            continue
        res.update({"t": round(rep["t"], 3), "df": rep["df"], "p_two_sided": round(rep["p_two_sided"], 6),
                    "se": round(rep["se"], 5), "variance_inflation": rep["variance_inflation"],
                    "residual_dependence": rep["residual_dependence"]})
        results[feat] = res
        pvals.append(rep["p_two_sided"])
        names.append(feat)
    if pvals:
        from research.harness import benjamini_hochberg

        bh = benjamini_hochberg(pvals)
        rejected = [bool(x) for x in bh.get("rejected", [])]
        qvalues = [float(x) for x in bh.get("qvalues", [])]
        for i, feat in enumerate(names):
            res = results[feat]
            ok = rejected[i] if i < len(rejected) else False
            res["bh_q"] = round(qvalues[i], 6) if i < len(qvalues) else None
            res["bh_significant"] = ok
            res["verdict"] = ("CONFIRMED" if ok and res["mean_ic"] > 0 else
                              "CONTRADICTED" if ok and res["mean_ic"] < 0 else "NOT_CONFIRMED")
    return {
        "status": "RESEARCH_ONLY",
        "discovery_window_excluded": list(DISCOVERY_WINDOW),
        "inference": (f"tie-corrected Spearman IC per day; overlap-robust error for {int(horizon)}-session "
                      f"windows (research.overlap_inference); Student t (df=blocks-1); BH across features; "
                      f">= {MIN_INDEPENDENT_SESSIONS} independent blocks and non-zero dispersion required"),
        "trades_used": len(usable),
        "days_used": len(days),
        "features": results,
        "note": "Pre-registered 2026-11-19. CONFIRMED is required before a challenger may use a feature; "
                "even then it reaches PAPER only through the champion/challenger gate.",
    }


def _eval_spec_tag(horizon: int) -> str:
    return f"sim:h{int(horizon)}:be{_BT_BREAKEVEN_PCT:g}:gap"


#: How run_backtest picks its symbols; recorded in every saved report so a
#: calibration built by the old alphabetical prefix is recognisably legacy.
SAMPLING = "even_stride"


def _spread_sample(symbols: list[str], limit: int) -> list[str]:
    """Deterministic, evenly spaced ``limit`` symbols across the whole store.

    store_symbols() is alphabetical (the store is built with groupby), so the
    old ``[:max_symbols]`` prefix calibrated every signal on roughly the A-D
    names only (first weekly run on the owner's Mac, 2026-11-15: 800 of 3,466
    symbols). An even stride keeps the same runtime budget, covers the whole
    alphabet, and picks the same names for the same store.
    """
    names = list(symbols)
    n = len(names)
    if limit <= 0 or n <= limit:
        return names
    return [names[(i * n) // limit] for i in range(limit)]


def run_backtest(sample_step: int = 5, lookback_sessions: int = 250,
                 horizon: int = _EVAL_HORIZON, max_symbols: int = 800, on_trade=None) -> dict:
    """
    Walk-forward backtest across the bhav store. Returns and persists
    per-signal stats. Needs ≥ (60 + lookback + horizon) sessions of data.

    `on_trade`, when given, is called with a full per-trade record dict for every
    FILLED trade (the Historical Trade Ledger hook). Default None → the aggregate
    path is byte-for-byte unchanged; the ledger is opt-in for the gauntlet.
    """
    from data.bhavcopy_store import store_symbols, get_ohlcv
    from scan.unified_scanner import UnifiedScanner

    # Measure raw detections: the backtest is the evidence the live gate uses,
    # so it must never be filtered by that gate (verdicts are unused here).
    sc = UnifiedScanner(evidence_gate=False)
    stats: dict[str, dict] = {}
    symbols = _spread_sample(store_symbols(), max_symbols)

    with _state_lock:
        _state.update(running=True, progress=0, total=len(symbols))

    t0 = time.time()
    try:
        return _run_backtest_inner(sc, stats, symbols, sample_step,
                                   lookback_sessions, horizon, t0, on_trade,
                                   run_config={"sample_step": sample_step, "lookback_sessions": lookback_sessions,
                                               "max_symbols": max_symbols})
    finally:
        # Crash-proof: 'running' can never stick True and freeze the UI button
        with _state_lock:
            _state["running"] = False


def _run_backtest_inner(sc, stats, symbols, sample_step, lookback_sessions,
                        horizon, t0, on_trade=None, run_config=None):
    from data.bhavcopy_store import get_ohlcv

    regime_series = _nifty_regime_series()      # None → regime split skipped
    # the calibration is frozen for the whole run — hash it ONCE, not per trade
    _calib_version = _calibration_version() if on_trade is not None else ""
    tgt_stats: dict[str, dict] = {}             # target label → aggregate
    exit_stats: dict[str, dict] = {}            # exit variant → aggregate (research only)
    confirm_rows: list[dict] = []               # pre-registered feature confirmation (research only)
    combo_stats: dict[str, dict] = {}           # exact signal combination → aggregate
    seen_dates: set[str] = set()                # realised NSE sessions (official bars present)
    try:
        # Exact windows are checked against the realised archive sessions PLUS
        # the verified calendar's sessions in covered years, minus uncovered
        # future years (core.session_window): a market-wide missing archive
        # leaves a hole instead of compressing two days into one.
        from core.session_window import official_session_index
        from data.bhavcopy_store import _dates_on_disk
        official_index = official_session_index(d.isoformat() for d in _dates_on_disk())
    except Exception:
        official_index = {}
    horizon_gaps = 0                            # trades whose window is not exactly `horizon` sessions

    evaluated = 0                               # symbols that had >= 1 analysed window (the save threshold)
    # A symbol whose bars END before the last official session stopped trading
    # (delisted, suspended, merged). Its final signals used to be skipped for
    # want of a full window, so an observable early stop-out of a dying stock
    # silently left the evidence: optimistic survivorship (cross-review B-6).
    # Such a window now counts when the trade RESOLVES inside the bars that
    # exist (stop, target, scratch); one still open when prices stop is
    # censored: counted, reported, never invented as a win or a loss.
    global_last = max(official_index) if official_index else ""
    ended_terminal = 0                          # partial windows of stopped symbols that resolved
    ended_censored = 0                          # ... that were still open when prices stopped
    for si, sym in enumerate(symbols):
        df = get_ohlcv(sym)
        if df is None or len(df) < 60 + horizon + 20 or not {"high", "low", "close"} <= set(df.columns):
            continue
        n = len(df)
        start = max(60, n - horizon - lookback_sessions)
        day_keys = list(pd.DatetimeIndex(df.index).strftime("%Y-%m-%d")) if official_index else []
        ended = bool(global_last and day_keys and day_keys[-1] < global_last)
        symbol_used = False
        for t in range(start, n if ended else n - horizon, sample_step):
            avail = min(horizon, n - t)
            partial = avail < horizon
            if official_index and not window_is_exact(day_keys, t, avail, official_index):
                horizon_gaps += 1
                continue
            if not symbol_used:
                # Counted (save threshold, data span) only once it contributes a
                # real analysed window: an unusable or all-gap frame must not
                # pad a thin run into a saved report, nor date its evidence
                # (cross-review verification QT-11).
                symbol_used = True
                evaluated += 1
                try:
                    seen_dates.update(pd.DatetimeIndex(df.index).strftime("%Y-%m-%d"))
                except Exception:
                    seen_dates.update(str(x)[:10] for x in df.index)
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
            # The same rulebook as the forward tracker and the paper book
            # (core.evaluation_spec): buy-stop fill at the open when it gaps
            # above entry, gap-aware stop/target exits, breakeven trail.
            sim = simulate_trade(
                r.entry, r.stop, r.target,
                fwd["high"].values, fwd["low"].values, fwd["close"].values,
                opens=fwd["open"].values if "open" in fwd.columns else None,
                be_pct=_BT_BREAKEVEN_PCT)
            outcome, r_mult = sim.outcome, sim.r
            _e_off, _x_off = sim.entry_offset, sim.exit_offset
            if outcome == "NO_FILL":
                continue          # order never triggered — not a trade
            if partial:
                if outcome not in ("WIN", "LOSS", "SCRATCH"):
                    ended_censored += 1     # open when prices stopped: outcome unobservable
                    continue
                ended_terminal += 1
            gross_r = r_mult      # before costs — the ledger keeps both
            # NET of round-trip trading costs — the edge you'd actually keep, not
            # the gross move. Same cost hits the target-geometry sweep below.
            try:
                from core.costs import cost_in_r
                _cost_r = cost_in_r((sim.fill_price - r.stop) / sim.fill_price, "CNC")
            except Exception:
                _cost_r = 0.0
            r_mult -= _cost_r
            day_key = str(df.index[t])[:10]   # decision cluster: same-day trades co-move

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
                                       regime, _calib_version,
                                       fill_price=sim.fill_price, exit_price=sim.exit_price)
                except Exception:
                    pass          # ledger emission must never break the backtest

            feats = confirmation_features(hist)
            if feats is not None:
                confirm_rows.append({"day": day_key, "r": r_mult, **feats})

            # Pre-declared exit-construction experiment on the same trade.
            if t + 2 * horizon <= n:
                for vname, voutcome, vr in exit_variant_results(
                        r.entry, r.stop, r.target, df.iloc[t:t + 2 * horizon], horizon):
                    es = exit_stats.setdefault(vname, {"trades": 0, "wins": 0, "scratches": 0,
                                                       "r_sum": 0.0, "clusters": {}})
                    es["trades"] += 1
                    es["r_sum"] += vr
                    es["wins"] += voutcome == "WIN"
                    es["scratches"] += voutcome == "SCRATCH"
                    _add_cluster(es["clusters"], day_key, vr)

            # Target geometry sweep — same fill, +2/+3/+4% targets (full windows only)
            for label, (o2, r2) in ({} if partial else sweep_targets(
                    r.entry, r.stop, fwd["high"].values,
                    fwd["low"].values, fwd["close"].values)).items():
                ts = tgt_stats.setdefault(label, {"trades": 0, "wins": 0,
                                                  "closed": 0, "r_sum": 0.0})
                ts["trades"] += 1
                ts["r_sum"] += r2 - _cost_r          # net of costs, same as above
                if o2 == "WIN":
                    ts["wins"] += 1
                    ts["closed"] += 1
                elif o2 == "LOSS":
                    ts["closed"] += 1

            combo = combo_key(r.signals)
            if combo:
                cs = combo_stats.setdefault(combo, {"trades": 0, "wins": 0, "losses": 0,
                                                    "r_sum": 0.0, "clusters": {}})
                cs["trades"] += 1
                cs["r_sum"] += r_mult
                cs["wins"] += outcome == "WIN"
                cs["losses"] += outcome == "LOSS"
                _add_cluster(cs["clusters"], day_key, r_mult)
            for sig in r.signals:
                s = stats.setdefault(sig, {"trades": 0, "wins": 0, "losses": 0,
                                           "flat": 0, "r_sum": 0.0,
                                           "regimes": {}, "clusters": {}})
                s["trades"] += 1
                s["r_sum"] += r_mult
                _add_cluster(s.setdefault("clusters", {}), day_key, r_mult)
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

    # Overlapping 10-session windows: inference needs the session index.
    session_index = realised_session_index(seen_dates)

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
            **_evidence_fields(s, session_index, horizon),
        }

    combos: dict[str, dict] = {}
    for key, cs in combo_stats.items():
        if cs["trades"] < COMBO_MIN_TRADES:
            continue
        combos[key] = {"trades": cs["trades"], "wins": cs["wins"], "losses": cs["losses"],
                       **_evidence_fields(cs, session_index, horizon)}

    # Spend one look of the annual alpha budget BEFORE testing: a forced rerun,
    # a crash after this point, or the same data re-run all count.
    import uuid
    from research import alpha_ledger

    try:
        from scan.unified_scanner import SIGNAL_META
        untested = sorted(set(SIGNAL_META) - set(report))
    except Exception:
        untested = []
    members = sorted(list(report) + [f"combo:{k}" for k in combos] + [f"untested:{k}" for k in untested])
    members_sha = hashlib.sha256("\n".join(members).encode()).hexdigest()
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    binding = {"generated_at": generated_at, "code_sha": _code_sha(),
               "universe_sha256": hashlib.sha256("\n".join(sorted(map(str, symbols))).encode()).hexdigest(),
               "data_first_session": min(seen_dates) if seen_dates else "",
               "data_last_session": max(seen_dates) if seen_dates else "",
               "members_sha256": members_sha, "family_size": len(members)}
    # A report that will be discarded (too few symbols, e.g. a fresh process whose
    # store is not loaded, or no signal at all) can admit nothing, so it spends no
    # look. That is decided from counts alone, before any test sees an outcome.
    # The threshold counts symbols that actually had a usable frame, not the
    # requested list: padding a preview with an empty symbol must not turn it
    # into a saved, charged report (GPT cross-review C-4).
    will_save = evaluated >= MIN_SYMBOLS_TO_SAVE and bool(report)
    program = gate_program(run_config)
    lost = will_save and _alpha_ledger_lost()
    if lost:
        # The ledger is gone although looks were spent through it: re-initialising
        # it would open a fresh annual budget (GPT cross-review C-3). The run is
        # NOT saved either, so the last spent report (and the loss) stays on disk
        # and every later run detects it again: nothing certifies until the
        # ledger is restored (cross-review verification QT-02).
        log.warning("alpha_ledger_missing_with_prior_looks", path=str(alpha_ledger.ledger_path()))
        will_save = False
        attestation = {"program_id": alpha_ledger.program_id(program), "look_id": "", "look_index": 0,
                       "alpha": 0.0, "unreserved": "LEDGER_LOST_WITH_PRIOR_LOOKS"}
    elif will_save:
        attestation = alpha_ledger.reserve(program, f"look-{uuid.uuid4().hex}", family_size=len(members),
                                           members_sha256=members_sha, binding=binding)
    else:
        attestation = {"program_id": alpha_ledger.program_id(program), "look_id": "", "look_index": 0,
                       "alpha": 0.0, "unreserved": "REPORT_NOT_SAVED"}
    family = apply_family_correction(report, combos, alpha=attestation["alpha"], untested=len(untested))
    family["members_sha256"] = members_sha

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
           "sampling": SAMPLING,
           "report_schema": REPORT_SCHEMA,
           "eval_spec": _eval_spec_tag(horizon),
           "signals": report,
           "combos": combos,
           "target_sweep": target_sweep,
           "exit_variants": exit_variant_report(exit_stats, horizon=horizon, session_index=session_index),
           "feature_confirmation": confirmation_report(confirm_rows, horizon=horizon,
                                                       session_index=session_index),
           "evidence_inference": {"method": "research.overlap_inference", "session_index": "realised official bars",
                                  "sessions_indexed": len(session_index), "family_correction": family},
           "evidence_provenance": {"generated_at": binding["generated_at"],
                                   "report_schema": REPORT_SCHEMA, "eval_spec": _eval_spec_tag(horizon),
                                   "horizon_sessions": int(horizon), "code_sha": binding["code_sha"],
                                   "members_sha256": members_sha,
                                   "family_method": family.get("method"), "family_size": family.get("family_size"),
                                   "sessions_indexed": len(session_index), "sampling": SAMPLING,
                                   "exact_horizon_check": bool(official_index),
                                   "horizon_gap_windows_skipped": horizon_gaps,
                                   # Symbols that stopped trading before the last session (B-6):
                                   "stopped_symbol_trades_resolved": ended_terminal,
                                   "stopped_symbol_trades_censored": ended_censored,
                                   # Data version and universe: hashed into the snapshot id with
                                   # the rest of the report, so any change re-keys the evidence.
                                   "universe_sha256": binding["universe_sha256"],
                                   "symbols": len(symbols), "symbols_evaluated": evaluated,
                                   "run_config": program.get("run_config"),
                                   "data_first_session": binding["data_first_session"],
                                   "data_last_session": binding["data_last_session"],
                                   "family_alpha": family.get("alpha"),
                                   "alpha_accounting": attestation},
           "recommended_target_pct": recommended,
           "elapsed_s": round(time.time() - t0, 1)}
    if not will_save:
        # Never replace a real calibration with an empty/thin one (e.g. a
        # process whose bhavcopy store was not loaded).
        log.warning("backtest_too_thin_not_saved", symbols=len(symbols), signals=len(report))
    else:
        try:
            from core.atomic_io import atomic_write_text
            atomic_write_text(_OUT, json.dumps(out, indent=2))
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


def edge_in_regime(signal_keys: list[str], regime: str,
                   min_trades: int = 20) -> float | None:
    """combo_edge, but conditioned on a regime bucket. Falls back to the
    overall number per signal when its regime bucket is too thin —
    honest specificity, never thin-slice noise."""
    rep = load_report()
    if not rep:
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
    print(json.dumps(run_backtest(), indent=2))
