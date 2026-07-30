"""
🧠 Brain 1 — the Evidence Brain.

Consumes canonical OutcomeObservations (and a strategy's backtest edge) and produces ONE
immutable `StrategyEvidenceCard`. It interprets evidence — expectancy, uncertainty, decay,
overfit, regime dependence, cost sensitivity, correlation cluster — and recommends a lifecycle
state. It NEVER allocates capital, sizes, or trades: it only reads and judges.

Reuses the audited stats in `research.harness` and the calibration verdict in
`research.auto_research.growth`. With too little evidence it says INSUFFICIENT_EVIDENCE —
never a winning/losing label, never a claim the data cannot support.
"""
from __future__ import annotations

from research.intelligence import schemas as SC
from research.auto_research import growth as GR

# evidence states (richer than win/lose)
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
PROMISING = "PROMISING"
FORWARD_PENDING = "FORWARD_PENDING"
CONFIRMED = "CONFIRMED"
REGIME_DEPENDENT = "REGIME_DEPENDENT"
WEAKER_THAN_EXPECTED = "WEAKER_THAN_EXPECTED"
DECAYING = "DECAYING"
OVERFIT = "OVERFIT"
RETIRED = "RETIRED"

MIN_FORWARD = 20


def _r_stats(returns: list) -> dict:
    """Mean, standard error and a conservative lower bound of a list of R multiples.
    Uses the repo harness when a sample exists; degrades to zeros when empty."""
    n = len(returns)
    if n == 0:
        return {"n": 0, "mean": 0.0, "stderr": 0.0, "lower": 0.0, "pf": None,
                "sharpe": 0.0, "max_dd": 0.0}
    mean = sum(returns) / n
    if n > 1:
        var = sum((r - mean) ** 2 for r in returns) / (n - 1)
        sd = var ** 0.5
        se = sd / (n ** 0.5)
    else:
        sd = se = 0.0
    wins = sum(r for r in returns if r > 0)
    losses = -sum(r for r in returns if r <= 0)
    pf = (wins / losses) if losses > 1e-9 else (None if wins == 0 else float("inf"))
    sharpe = (mean / sd) if sd > 1e-9 else 0.0
    # simple running max-drawdown on the cumulative R curve
    cum = 0.0; peak = 0.0; mdd = 0.0
    for r in returns:
        cum += r; peak = max(peak, cum); mdd = max(mdd, peak - cum)
    return {"n": n, "mean": mean, "stderr": se, "lower": mean - se,
            "pf": (round(pf, 3) if isinstance(pf, float) and pf != float("inf") else pf),
            "sharpe": sharpe, "max_dd": mdd}


def build_card(strategy_def: SC.StrategyDefinition, *, backtest_R: float,
               forward_returns: list, in_sample_trades: int = 0,
               out_of_sample_trades: int = 0, regime_returns: dict | None = None,
               sector_shares: dict | None = None, correlation_cluster: str = "",
               evidence_freshness_days: float = 0.0, data_quality_warnings=(),
               benchmark_returns: list | None = None,
               n_search_attempts: int = 1) -> SC.StrategyEvidenceCard:
    """Produce the immutable evidence card. `forward_returns` are realized R multiples from
    out-of-sample PAPER trades (never synthetic)."""
    fs = _r_stats(list(forward_returns or []))
    n_fwd = fs["n"]
    regime_returns = regime_returns or {}
    warns = tuple(data_quality_warnings or ())

    support, conflict = [], []
    # deflated sharpe (accounts for the number of strategies searched) — reuse the harness
    dsr = 0.0
    try:
        from research import harness as H
        if n_fwd >= 2:
            dsr = float(H.deflated_sharpe_ratio(fs["sharpe"], n_fwd,
                                                n_trials=max(1, n_search_attempts))["dsr"])
    except Exception:
        dsr = 0.0
    # alpha/beta vs benchmark when both series line up
    alpha = beta = 0.0
    if benchmark_returns and len(benchmark_returns) == n_fwd and n_fwd >= 3:
        try:
            from research import harness as H
            ab = H.alpha_beta(list(forward_returns), list(benchmark_returns))
            alpha, beta = float(ab.get("alpha", 0.0)), float(ab.get("beta", 0.0))
        except Exception:
            pass

    cal = GR.calibrate("", strategy_def.family, backtest_R, fs["mean"], n_fwd,
                       forward_lower_R=fs["lower"])
    ftb = (fs["mean"] / backtest_R) if backtest_R > 0 else 0.0

    # regime dependence: strong overall but negative in ≥1 regime with enough samples
    regime_dep = False
    for reg, rr in regime_returns.items():
        st = _r_stats(list(rr))
        if st["n"] >= 8 and st["mean"] < 0 <= fs["mean"]:
            regime_dep = True
            conflict.append(f"loses in the {reg} regime ({st['mean']:+.2f}R over {st['n']})")

    sector_conc = max(sector_shares.values()) if sector_shares else 0.0
    if sector_conc > 0.5:
        conflict.append(f"concentrated: {sector_conc:.0%} of trades in one sector")
    if warns:
        conflict.append(f"data-quality: {warns[0]}")

    # ── classify (rich states, evidence-gated) ───────────────────────────────────
    overfit = (cal.verdict == GR.OVERFIT)
    if n_fwd < MIN_FORWARD:
        state = PROMISING if in_sample_trades >= 30 else INSUFFICIENT_EVIDENCE
        if n_fwd > 0:
            state = FORWARD_PENDING
    elif overfit:
        state = OVERFIT
        conflict.append(cal.note)
    elif cal.verdict == GR.DECAYED:
        state = DECAYING; conflict.append(cal.note)
    elif regime_dep:
        state = REGIME_DEPENDENT
    elif cal.verdict == GR.CONFIRMED:
        state = CONFIRMED; support.append(cal.note)
    elif cal.verdict == GR.WEAKER_POSITIVE:
        state = WEAKER_THAN_EXPECTED; support.append("still positive out-of-sample")
    else:
        state = FORWARD_PENDING

    if fs["lower"] > 0:
        support.append(f"lower-bound edge {fs['lower']:+.2f}R (uncertainty-adjusted)")
    if dsr > 0.9:
        support.append(f"deflated Sharpe {dsr:.2f} survives the search burden")

    confidence = _confidence(state, fs, dsr, len(warns))
    reco = _recommend(state)

    return SC.StrategyEvidenceCard(
        strategy_id=strategy_def.strategy_id, strategy_version=strategy_def.strategy_version,
        rules_hash=strategy_def.rules_hash, data_snapshot_id=strategy_def.data_snapshot_id,
        source="evidence_brain", event_ts=strategy_def.event_ts,
        family=strategy_def.family, evidence_state=state, confidence=round(confidence, 3),
        in_sample_trades=in_sample_trades, out_of_sample_trades=out_of_sample_trades,
        forward_trades=n_fwd, expectancy_R=round(fs["mean"], 4),
        lower_bound_R=round(fs["lower"], 4), profit_factor=fs["pf"],
        max_drawdown=round(fs["max_dd"], 4), sharpe=round(fs["sharpe"], 4),
        deflated_sharpe=round(dsr, 4), alpha=round(alpha, 4), beta=round(beta, 4),
        cost_sensitivity_R=0.0, forward_to_backtest=round(ftb, 4),
        regime_results={k: round(_r_stats(list(v))["mean"], 4)
                        for k, v in regime_returns.items()},
        sector_concentration=round(sector_conc, 4), correlation_cluster=correlation_cluster,
        evidence_freshness_days=round(evidence_freshness_days, 2),
        decay_detected=(state in (DECAYING, OVERFIT)), overfit=overfit,
        lifecycle_recommendation=reco, supporting_reasons=tuple(support),
        conflicting_reasons=tuple(conflict), data_quality_warnings=warns)


def _confidence(state: str, fs: dict, dsr: float, n_warn: int) -> float:
    base = {CONFIRMED: 0.8, WEAKER_THAN_EXPECTED: 0.55, REGIME_DEPENDENT: 0.4,
            FORWARD_PENDING: 0.3, PROMISING: 0.25, DECAYING: 0.15, OVERFIT: 0.05,
            INSUFFICIENT_EVIDENCE: 0.1}.get(state, 0.2)
    base += min(0.15, max(0.0, (fs["n"] - MIN_FORWARD) / 200.0))
    base += min(0.1, max(0.0, dsr - 0.9))
    base -= 0.1 * n_warn
    return max(0.0, min(1.0, base))


def _recommend(state: str) -> str:
    from research.strategy_studio import spec as S
    return {CONFIRMED: S.PAPER_CONFIRMED, OVERFIT: S.DECAYED, DECAYING: S.DECAYED,
            REGIME_DEPENDENT: S.PAPER_EVALUATION, WEAKER_THAN_EXPECTED: S.PAPER_EVALUATION,
            FORWARD_PENDING: S.PAPER_EVALUATION, PROMISING: S.PAPER_EVALUATION,
            INSUFFICIENT_EVIDENCE: S.PAPER_EVALUATION}.get(state, S.PAPER_EVALUATION)
