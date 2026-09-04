"""
🧪 EXP-006 — Institutional Momentum Breakout v1 (pre-registration + runner).

This wires the detector's observations into a realistic historical trade simulation
and hands the resulting R-stream to the EXISTING evidence gate (`research.harness`)
and the EXISTING immutable ledger (`gauntlet.ledger.TradeRecord`). No new statistics
are invented; the gauntlet's DSR / alpha / block-bootstrap machinery judges the claim.

Nothing here promotes a strategy or touches execution. The output is EVIDENCE:
a verdict of PASS / FAIL / INCONCLUSIVE on the pre-registered hypothesis.

Entry convention (pre-registered, PIT-safe): the signal is known only after the
breakout bar CLOSES; entry is the NEXT tradable bar's open with explicit slippage;
gap-through-stop fills at the (worse) open, never optimistically at the stop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research.momentum_breakout import pit as P
from research.momentum_breakout import features as F
from research.momentum_breakout.config import MomentumBreakoutConfig
from research.momentum_breakout.detector import BarSeries

EXPERIMENT_ID = "EXP-006"
EXPERIMENT_TITLE = "Institutional Momentum Breakout v1"

# Pre-registered exit variants. The PRIMARY hypothesis uses exactly one; the others
# are labelled secondary analysis and must not be swapped in after seeing results.
EXIT_STRUCTURAL_TREND = "structural_trend"          # PRIMARY: hard stop + close<50DMA
EXIT_STRUCTURAL_EMA_TRAIL = "structural_ema_trail"  # secondary: hard stop + 20-EMA trail
EXIT_STRUCTURAL_MAXHOLD = "structural_maxhold"      # secondary: hard stop + time stop
PRIMARY_EXIT = EXIT_STRUCTURAL_TREND
EXIT_VARIANTS = (EXIT_STRUCTURAL_TREND, EXIT_STRUCTURAL_EMA_TRAIL, EXIT_STRUCTURAL_MAXHOLD)


def spec(cfg: MomentumBreakoutConfig) -> dict:
    """The machine-readable pre-registration (frozen alongside the config hash)."""
    return {
        "experiment_id": EXPERIMENT_ID,
        "title": EXPERIMENT_TITLE,
        "hypothesis": (
            "Stocks with prior leadership, a long contracting base, a confirmed "
            "breakout, small structural risk and strong sector support have positive "
            "forward expectancy after realistic Indian cash-equity costs."),
        "primary_exit": PRIMARY_EXIT,
        "exit_variants": list(EXIT_VARIANTS),
        "entry_convention": {
            "signal_known": "after confirmed breakout close",
            "entry": "next tradable bar open",
            "slippage_pct_per_side": cfg.slippage_pct,
            "roundtrip_cost_pct": cfg.cost_pct_roundtrip,
            "same_bar_ordering": "stop checked before target (pessimistic)",
            "gap_through_stop": "fills at open, not at the stop price",
        },
        "primary_comparisons": [
            "Nifty benchmark over equivalent holding periods",
            "all eligible liquid stocks",
            "cross-sectional momentum without base+sector conditions",
            "breakout candidates without the strong-sector requirement",
        ],
        "decision_metrics": [
            "n_candidates", "n_trades", "expectancy_R", "expectancy_CI", "profit_factor",
            "win_rate", "avg_win", "avg_loss", "MAE", "MFE", "max_drawdown", "turnover",
            "cost_drag", "sharpe", "benchmark_relative", "regime_breakdown",
            "sector_concentration", "reality_check_multiple_testing",
        ],
        "config_hash": cfg.config_hash(),
        "no_post_result_optimisation": True,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Realistic historical trade simulation (PIT, gap-aware, conservative)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimTrade:
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    stop_price: float
    holding_period: int
    gross_R: float
    net_R: float
    exit_reason: str
    mae_R: float
    mfe_R: float
    benchmark_return: float | None


def simulate_trade(s: BarSeries, signal_i: int, initial_stop: float,
                   cfg: MomentumBreakoutConfig, variant: str = PRIMARY_EXIT) -> SimTrade | None:
    """Simulate one trade from the pre-registered entry convention. Returns None
    when no realistic fill was possible (e.g. the next bar gapped below the stop,
    invalidating the setup) — no fabricated fill."""
    n = len(s)
    entry_i = signal_i + 1
    if entry_i >= n:
        return None
    if not (np.isfinite(s.open[entry_i]) and initial_stop is not None
            and np.isfinite(initial_stop)):
        return None                                        # no tradable next bar (gap/delisted)
    slp = cfg.slippage_pct / 100.0
    entry_fill = float(s.open[entry_i]) * (1.0 + slp)     # buy → pay up
    if entry_fill <= initial_stop:
        return None                                        # gap opened below stop
    risk = entry_fill - initial_stop
    if risk <= 0:
        return None
    exit_fill = None; exit_reason = "eod"; exit_i = n - 1
    mae = 0.0; mfe = 0.0
    for j in range(entry_i, n):
        lo = float(s.low[j]); hi = float(s.high[j]); op = float(s.open[j]); cl = float(s.close[j])
        if not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(op) and np.isfinite(cl)):
            continue                                       # missing session during hold — skip
        # running MAE/MFE in R
        mae = min(mae, (lo - entry_fill) / risk)
        mfe = max(mfe, (hi - entry_fill) / risk)
        # ── hard structural stop, gap-aware, checked BEFORE any upside (pessimistic) ──
        if j > entry_i and op < initial_stop:
            exit_fill = op * (1.0 - slp); exit_reason = "gap_stop"; exit_i = j; break
        if lo <= initial_stop:
            exit_fill = initial_stop * (1.0 - slp); exit_reason = "stop"; exit_i = j; break
        # ── variant weakening exit on the CLOSE (only after the entry bar) ──
        if j > entry_i:
            if variant == EXIT_STRUCTURAL_TREND:
                ma = P.sma(s.close, j, 50)
                if np.isfinite(ma) and cl < ma:
                    exit_fill = cl * (1.0 - slp); exit_reason = "trend_break"; exit_i = j; break
            elif variant == EXIT_STRUCTURAL_EMA_TRAIL:
                e = P.ema(s.close, j, cfg.trail_ema)
                if np.isfinite(e) and cl < e:
                    exit_fill = cl * (1.0 - slp); exit_reason = "ema_trail"; exit_i = j; break
            elif variant == EXIT_STRUCTURAL_MAXHOLD:
                if j - entry_i >= cfg.max_hold_days:
                    exit_fill = cl * (1.0 - slp); exit_reason = "time_stop"; exit_i = j; break
    if exit_fill is None:
        exit_fill = float(s.close[n - 1]) * (1.0 - slp); exit_i = n - 1
    gross_R = (exit_fill - entry_fill) / risk
    risk_frac = risk / entry_fill
    net_R = gross_R - (cfg.cost_pct_roundtrip / 100.0) / risk_frac   # charges in R units
    bench = None
    if np.isfinite(s.bench_close[entry_i]) and s.bench_close[entry_i] > 0:
        bench = float(s.bench_close[exit_i] / s.bench_close[entry_i] - 1.0)
    return SimTrade(
        symbol=s.symbol, entry_date=s.dates[entry_i], exit_date=s.dates[exit_i],
        entry_price=round(entry_fill, 4), exit_price=round(exit_fill, 4),
        stop_price=round(initial_stop, 4), holding_period=exit_i - entry_i,
        gross_R=round(gross_R, 4), net_R=round(net_R, 4), exit_reason=exit_reason,
        mae_R=round(mae, 4), mfe_R=round(mfe, 4), benchmark_return=bench)


# ══════════════════════════════════════════════════════════════════════════════
# Run → evidence verdict (reuses research.harness)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_trades(trades: list[SimTrade], n_trials: int = 1,
                    require_alpha: bool = True, require_block_ci: bool = True) -> dict:
    """Hand the R-stream to the existing evidence gate. Returns the harness verdict
    plus descriptive money metrics. PASS ⇔ harness PROMOTE; FAIL ⇔ REJECT;
    otherwise INCONCLUSIVE/UNDERPOWERED pass through."""
    from research import harness as H
    rs = [t.net_R for t in trades]
    if not rs:
        return {"verdict": "INCONCLUSIVE", "n_trades": 0,
                "insight": "no trades generated"}
    bench = None
    if require_alpha and all(t.benchmark_return is not None for t in trades):
        # convert per-trade benchmark % into that trade's R units (beta in R)
        bench = []
        for t in trades:
            risk_frac = (t.entry_price - t.stop_price) / t.entry_price if t.entry_price else 0.0
            bench.append((t.benchmark_return / risk_frac) if risk_frac > 0 else 0.0)
    res = H.evaluate(
        rs,
        n_trials=n_trials,
        benchmark_returns=bench,
        require_block_ci=require_block_ci,
        block_ci_seed=1,
    )
    win_R = [r for r in rs if r > 0]
    loss_R = [r for r in rs if r <= 0]
    gross_win = sum(win_R) if win_R else 0.0
    gross_loss = -sum(loss_R) if loss_R else 0.0
    return {
        "verdict": res.verdict, "insight": res.insight,
        "n_trades": len(trades), "expectancy_R": res.mean_r, "sharpe": res.sharpe,
        "p_value": res.p_value, "psr": res.psr, "dsr": res.dsr,
        "win_rate": round(len(win_R) / len(trades) * 100, 1),
        "avg_win_R": round(float(np.mean(win_R)), 3) if win_R else 0.0,
        "avg_loss_R": round(float(np.mean(loss_R)), 3) if loss_R else 0.0,
        "profit_factor": round(gross_win / gross_loss, 3) if gross_loss > 0 else float("inf"),
        "avg_mae_R": round(float(np.mean([t.mae_R for t in trades])), 3),
        "avg_mfe_R": round(float(np.mean([t.mfe_R for t in trades])), 3),
        "avg_hold": round(float(np.mean([t.holding_period for t in trades])), 1),
        "stats": res.stats,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Pre-registered ablations (fixed comparisons — diagnostic, FDR-controlled)
# ══════════════════════════════════════════════════════════════════════════════

def ablation_configs(base: MomentumBreakoutConfig) -> dict:
    """Each ablation RELAXES the full framework to isolate one component's
    contribution. Pre-registered; run together and FDR-corrected. `dataclasses.replace`
    keeps every other knob identical so the comparison is clean."""
    from dataclasses import replace
    return {
        "prior_only": replace(base, require_contraction=False, require_confirmed_close=False,
                              require_sector_strength=False, base_min_len=1,
                              min_breakout_rvol=0.0, max_extension_atr=1e9),
        "prior_plus_breakout": replace(base, require_contraction=False,
                                       require_sector_strength=False, base_min_len=1),
        "prior_plus_base": replace(base, require_sector_strength=False,
                                   require_confirmed_close=False, min_breakout_rvol=0.0),
        "prior_base_risk": replace(base, require_sector_strength=False),
        "prior_base_risk_sector": replace(base),           # == full minus participation weight
        "full_framework": replace(base),
    }
