"""
🧭 Detector — turns point-in-time price/volume history into MomentumBreakout
observations.

For each bar it asks: is there a candidate breakout event here (a first confirmed
close through a pre-existing base pivot)? If so it builds the full observation —
raw features across all groups, transparent component scores, structural stop and
initial risk, sector/valuation context, data-quality flags — then applies the
transparent primary eligibility contract, recording every rejection reason while
KEEPING the continuous feature values. Point-in-time safety is checked and fails
closed. Events are deduplicated so one breakout is counted once.

Research-only. No output of this module touches autopilot, the broker, Telegram,
GTT, or strategy graduation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research.momentum_breakout import pit as P
from research.momentum_breakout import features as F
from research.momentum_breakout import pit_safety as PS
from research.momentum_breakout import observation as OBS
from research.momentum_breakout.observation import MomentumBreakoutObservation
from research.momentum_breakout.config import MomentumBreakoutConfig, DETECTOR_VERSION


@dataclass
class BarSeries:
    """Aligned OHLCV for one symbol, chronological. `bench_close` is the benchmark
    (Nifty) close aligned to the SAME bar index — the caller reindexes before
    constructing this, so relative strength is point-in-time and pure."""
    symbol: str
    exchange: str
    dates: list           # ISO date strings, one per bar
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    bench_close: np.ndarray

    def __len__(self) -> int:
        return len(self.dates)


# rejection reason codes (structured, stable)
R_INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
R_NO_BASE = "NO_BASE"
R_WEAK_PRIOR_RS = "WEAK_PRIOR_RS"
R_NOT_ABOVE_TREND = "NOT_ABOVE_RISING_200DMA"
R_NO_CONTRACTION = "NO_BASE_CONTRACTION"
R_UNCONFIRMED_BREAKOUT = "UNCONFIRMED_BREAKOUT"
R_OVEREXTENDED = "OVEREXTENDED_CHASE"
R_LOW_RVOL = "LOW_BREAKOUT_RVOL"
R_RISK_TOO_HIGH = "STRUCTURAL_RISK_TOO_HIGH"
R_NO_STOP = "NO_STRUCTURAL_STOP"
R_WEAK_SECTOR = "WEAK_SECTOR"
R_ILLIQUID = "ILLIQUID"
R_PIT_VIOLATION = "PIT_VIOLATION"


def _detect_base(s: BarSeries, i: int, cfg: MomentumBreakoutConfig):
    """Deterministic base + pivot at breakout bar `i`. The base is the LONGEST
    window [i-L, i-1] (L in [min,max]) whose pivot (its max high) was NEVER closed
    through inside the base, and whose depth is acceptable — i.e. a genuine ceiling
    the bar-`i` close is the FIRST to confirm-break. Returns (base_start_i, pivot)
    or None. Uses only bars ≤ i (no future decides the base)."""
    c_i = float(s.close[i])
    if not np.isfinite(c_i):
        return None                         # missing session — never a candidate
    buffer = 1.0 + cfg.breakout_buffer_pct / 100.0
    # Incremental leftward scan: extend the window one bar at a time (L = 1..max),
    # maintaining running max(high)/max(close)/min(low) in O(1). Track the LARGEST
    # valid base — identical semantics to a max→min rescan, but O(base_max) not
    # O(base_max²), which is what makes a whole-market historical run tractable.
    run_hi = -np.inf; run_lo = np.inf; run_maxclose = -np.inf
    best = None
    lo_bound = max(0, i - cfg.base_max_len)
    for start in range(i - 1, lo_bound - 1, -1):
        h = float(s.high[start]); c = float(s.close[start]); l = float(s.low[start])
        if not (np.isfinite(h) and np.isfinite(c) and np.isfinite(l)):
            break                            # a gap → no clean base extends past it
        run_hi = max(run_hi, h); run_lo = min(run_lo, l); run_maxclose = max(run_maxclose, c)
        L = i - start
        if L < cfg.base_min_len:
            continue
        pivot = run_hi
        if pivot <= 0:
            continue
        if run_maxclose >= pivot * buffer:   # a confirmed close already cleared the pivot
            continue
        if c_i < pivot * buffer:             # bar i is not the confirming close vs this pivot
            continue
        if (pivot - run_lo) / pivot * 100.0 > cfg.max_base_depth_pct:
            continue
        best = (start, float(pivot))         # larger L overwrites → keeps the LONGEST base
    return best


def consider(s: BarSeries, i: int, cfg: MomentumBreakoutConfig,
             sector_ctx: dict | None = None, valuation_record: dict | None = None,
             provenance: dict | None = None) -> MomentumBreakoutObservation | None:
    """Evaluate bar `i` as a candidate. Returns a full observation (ELIGIBLE or
    REJECTED with reasons) if a breakout event exists here, else None (not a
    candidate — no base/breakout). Never mutates inputs."""
    provenance = provenance or {}
    n = len(s)
    min_hist = max(252, cfg.base_max_len + cfg.trend_ma) + 5
    if i < min_hist or i >= n:
        return None
    base = _detect_base(s, i, cfg)
    if base is None:
        return None
    base_start_i, pivot = base
    base_end_i = i - 1
    base_len = base_end_i - base_start_i + 1

    atr_val = P.atr(s.high, s.low, s.close, i, 14)

    # ── feature groups (raw, always computed even when a gate fails) ──
    prior = F.prior_upmove_features(s.close, s.high, s.low, s.bench_close, i, cfg)
    basef = F.base_features(s.open, s.high, s.low, s.close, s.volume,
                            base_start_i, base_end_i, pivot, cfg)
    brk = F.breakout_features(s.open, s.high, s.low, s.close, s.volume, i, pivot, atr_val, cfg)
    part = F.participation_features(s.volume, s.close, i, base_end_i, base_len, cfg,
                                    delivery_pct=(sector_ctx or {}).get("delivery_pct")
                                    if sector_ctx else None)
    trext = F.trend_extension_features(s.close, s.high, i, pivot, atr_val, cfg)
    valf, vflags, val_ts = F.valuation_features(valuation_record, s.dates[i], cfg)

    # ── structural stop / initial risk ──
    entry_ref = float(s.close[i])           # reference; actual entry is next bar (sim)
    cands = F.stop_candidates(s.high, s.low, s.close, i, pivot, basef["base_low"], atr_val, cfg)
    stop, stop_rule = F.select_structural_stop(cands, entry_ref, cfg)
    if stop is not None and entry_ref > 0:
        init_risk_pct = (entry_ref - stop) / entry_ref * 100.0
        init_risk_atr = ((entry_ref - stop) / atr_val) if (atr_val and np.isfinite(atr_val)
                                                           and atr_val > 0) else float("nan")
    else:
        init_risk_pct = float("nan"); init_risk_atr = float("nan")

    # ── data-quality flags ──
    flags: list[str] = list(vflags)
    if sector_ctx is None or not sector_ctx.get("membership_pit", False):
        flags.append(OBS.FLAG_SECTOR_MEMBERSHIP_NOT_PIT)
    if not part.get("delivery_available"):
        flags.append(OBS.FLAG_DELIVERY_UNAVAILABLE)
    if provenance.get("survivorship_complete") is False:
        flags.append(OBS.FLAG_SURVIVORSHIP_INCOMPLETE)

    # ── component scores (transparent) ──
    scores = {
        "leadership": F.leadership_score(prior, cfg),
        "base_quality": F.base_quality_score(basef, cfg),
        "breakout_quality": F.breakout_quality_score(brk, cfg),
        "sector_strength": F.sector_strength_score(sector_ctx or {}, cfg),
        "participation": F.participation_score(part, cfg),
        "risk_efficiency": F.risk_efficiency_score(init_risk_pct, cfg),
        "extension_risk": F.extension_risk_score(trext, cfg),
    }
    # combined score for RANKING only — never hides rejection reasons; versioned weights
    combined = round(
        0.25 * scores["leadership"] + 0.20 * scores["base_quality"]
        + 0.20 * scores["breakout_quality"] + 0.15 * scores["sector_strength"]
        + 0.10 * scores["participation"] + 0.10 * scores["risk_efficiency"]
        - 0.10 * scores["extension_risk"], 1)

    # ── primary eligibility contract (each failure recorded; features kept) ──
    reasons: list[str] = []
    rel = prior.get("rel_to_bench_pct")
    if rel is None or not np.isfinite(rel) or rel < cfg.min_prior_rs_pct:
        reasons.append(R_WEAK_PRIOR_RS)
    if cfg.require_above_200dma:
        slope = prior.get("dma200_slope_pct")
        if not prior.get("above_200dma") or slope is None or not np.isfinite(slope) or slope <= 0:
            reasons.append(R_NOT_ABOVE_TREND)
    if cfg.require_contraction:
        rc = basef.get("range_contraction_ratio"); ac = basef.get("atr_contraction_ratio")
        contracted = ((rc is not None and np.isfinite(rc) and rc < 1.0)
                      or (ac is not None and np.isfinite(ac) and ac < 1.0))
        if not contracted:
            reasons.append(R_NO_CONTRACTION)
    if cfg.require_confirmed_close and not brk.get("confirmed_close"):
        reasons.append(R_UNCONFIRMED_BREAKOUT)
    if brk.get("overextended"):
        reasons.append(R_OVEREXTENDED)
    rvol = brk.get("breakout_rvol")
    if rvol is None or not np.isfinite(rvol) or rvol < cfg.min_breakout_rvol:
        reasons.append(R_LOW_RVOL)
    if stop is None:
        reasons.append(R_NO_STOP)
    elif not np.isfinite(init_risk_pct) or init_risk_pct > cfg.max_initial_risk_pct:
        reasons.append(R_RISK_TOO_HIGH)
    if cfg.require_sector_strength:
        ok_sector = bool(sector_ctx) and (
            (sector_ctx.get("sector_rs_pct") is not None
             and np.isfinite(sector_ctx["sector_rs_pct"])
             and sector_ctx["sector_rs_pct"] >= cfg.min_sector_rs_pct)
            or (sector_ctx.get("breadth_pct_above_50dma") is not None
                and sector_ctx["breadth_pct_above_50dma"] >= cfg.min_sector_breadth_pct))
        if not ok_sector:
            reasons.append(R_WEAK_SECTOR)
    liq = (sector_ctx or {}).get("turnover_cr")
    if liq is not None and np.isfinite(liq) and liq < cfg.min_turnover_cr:
        reasons.append(R_ILLIQUID)

    # ── point-in-time safety: entry is the NEXT bar; fail closed on any violation ──
    entry_index = i + 1
    pit_reasons: list[str] = []
    if entry_index < n:
        ec = PS.check_entry_not_signal_bar(entry_index, i)
        if not ec.ok:
            pit_reasons += list(ec.violations)
    pc = PS.check_pivot_pre_existing(pivot, basef["base_high"], float(s.high[i]))
    if not pc.ok:
        pit_reasons += list(pc.violations)
    if stop is not None:
        sc = PS.check_stop_below_entry(stop, entry_ref)
        if not sc.ok:
            pit_reasons += list(sc.violations)
    if val_ts is not None:
        tc = PS.check_timestamps(market_ts=s.dates[i], signal_ts=s.dates[i],
                                 data_avail_ts=s.dates[i],
                                 entry_ts=s.dates[entry_index] if entry_index < n else s.dates[i],
                                 valuation_ts=val_ts)
        if not tc.ok:
            pit_reasons += list(tc.violations)
    if pit_reasons:
        reasons.append(R_PIT_VIOLATION + ":" + "|".join(pit_reasons))

    eligibility = OBS.ELIGIBLE if not reasons else OBS.REJECTED

    return MomentumBreakoutObservation(
        symbol=s.symbol, exchange=s.exchange,
        observation_ts=s.dates[i], data_availability_ts=s.dates[i],
        candidate_date=s.dates[i],
        pivot=round(pivot, 4),
        base_start_date=s.dates[base_start_i], base_end_date=s.dates[base_end_i],
        base_duration=base_len,
        entry_reference_price=round(entry_ref, 4),
        structural_stop=round(stop, 4) if stop is not None else float("nan"),
        initial_risk_pct=round(init_risk_pct, 4) if np.isfinite(init_risk_pct) else float("nan"),
        initial_risk_atr=round(init_risk_atr, 4) if np.isfinite(init_risk_atr) else float("nan"),
        prior_upmove=prior, base_quality=basef, breakout_quality=brk,
        sector_strength=dict(sector_ctx or {}), participation=part,
        trend_extension=trext, valuation=valf, valuation_data_ts=val_ts,
        stop_candidates={**cands, "selected_rule": stop_rule},
        component_scores=scores, combined_score=combined,
        eligibility=eligibility, rejection_reasons=tuple(reasons),
        data_quality_flags=tuple(sorted(set(flags))),
        experiment_id=provenance.get("experiment_id", ""),
        strategy_id=cfg.strategy_id, config_version=cfg.config_version,
        config_hash=provenance.get("config_hash", cfg.config_hash()),
        dataset_snapshot_id=provenance.get("dataset_snapshot_id", ""),
        code_commit=provenance.get("code_commit", ""),
        detector_version=DETECTOR_VERSION,
    )


def scan_symbol(s: BarSeries, cfg: MomentumBreakoutConfig,
                sector_ctx_fn=None, valuation_fn=None, provenance: dict | None = None,
                registry: PS.EventRegistry | None = None,
                eligible_only: bool = False) -> list[MomentumBreakoutObservation]:
    """Walk the series, emit one observation per DISTINCT breakout event. The base
    validity rule (no confirmed close inside the base) means consecutive closes
    above one pivot cannot mint a second candidate — dedup is structural — and the
    EventRegistry is a belt-and-suspenders guard plus a re-eligibility cooldown."""
    registry = registry if registry is not None else PS.EventRegistry()
    out: list[MomentumBreakoutObservation] = []
    last_event_i = -10**9
    n = len(s)
    for i in range(n):
        if i - last_event_i < cfg.reeligible_after_bars:
            # inside the cooldown window: still allow a genuinely NEW base/pivot,
            # but skip anything that is not truly new (dedup)
            pass
        obs = consider(s, i, cfg,
                       sector_ctx=sector_ctx_fn(s, i) if sector_ctx_fn else None,
                       valuation_record=valuation_fn(s, i) if valuation_fn else None,
                       provenance=provenance)
        if obs is None:
            continue
        eid = obs.event_id()
        if not registry.register(eid):
            continue                        # duplicate event — never double-count
        if i - last_event_i < cfg.reeligible_after_bars and \
                out and abs(out[-1].pivot - obs.pivot) < 1e-9:
            continue                        # same pivot within cooldown — not new
        last_event_i = i
        if eligible_only and obs.eligibility != OBS.ELIGIBLE:
            continue
        out.append(obs)
    return out
