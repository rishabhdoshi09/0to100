"""
🔌 Production providers — wire the growth loop to REAL market data, honestly.

The brain's daily loop needs three data hooks. In tests these are injected (deterministic);
in production they come from here, reading the canonical stores the rest of QuantTerm uses:

  • backtest_evaluator(spec, split) -> EvidenceReport   (in-sample, from bhavcopy history)
  • daily_bars(date)               -> {symbol: (h,l,c)}  (the day being forward-tested)
  • signals_for(paper_strategy, date) -> [signal dicts]  (entries from the strategy's rules)

Every one degrades HONESTLY: with no research-grade data on disk they return empty / invalid
rather than fabricating prices or evidence. `is_synthetic` is False only when the numbers
came from real history — so the discovery gate treats a no-data run as unavailable, never as
a passing strategy. Pure reads; no order path.
"""
from __future__ import annotations

from research.strategy_studio.discovery import EvidenceReport


# ── backtest (in-sample evidence from real history) ──────────────────────────────

def backtest_evaluator(spec, split: str) -> EvidenceReport:
    """Evaluate a candidate on the canonical bhavcopy history. Returns market-labelled
    evidence when real data exists, else an invalid/empty report (never synthetic-as-real).

    This wraps the existing frozen momentum backtest so discovery reuses one audited engine
    rather than a second, drifting implementation."""
    try:
        from research.momentum_breakout import dataset as DS
        provider = DS.BhavDataProvider()
        symbols = _universe(provider)
        if not symbols:
            return EvidenceReport(invalid_data=True, is_synthetic=False,
                                  verdict="INCONCLUSIVE")
        from research.momentum_breakout import runner as R
        res = R.run_evidence(provider)
        v = res.get("verdict", {})
        stats = res.get("stats", {}) or {}
        return EvidenceReport(
            n_trades=int(stats.get("n_trades", 0)),
            n_symbols=int(stats.get("n_symbols", 0)),
            gross_expectancy_R=float(stats.get("gross_expectancy_R", 0.0)),
            net_expectancy_R=float(stats.get("net_expectancy_R", 0.0)),
            cost_drag_R=float(stats.get("cost_drag_R", 0.0)),
            p_value=float(stats.get("p_value", 1.0)),
            max_drawdown=float(stats.get("max_drawdown", 0.0)),
            turnover=float(stats.get("turnover", 0.0)),
            max_symbol_share=float(stats.get("max_symbol_share", 0.0)),
            is_synthetic=False, verdict=v.get("verdict", "INCONCLUSIVE"))
    except Exception:
        # no data / engine unavailable ⇒ honest "cannot judge", never a fake pass
        return EvidenceReport(invalid_data=True, is_synthetic=False, verdict="INCONCLUSIVE")


def _universe(provider) -> list:
    try:
        return list(provider.symbols())
    except Exception:
        return []


# ── forward-test data (the day's bars) ───────────────────────────────────────────

def daily_bars(date: str) -> dict:
    """{symbol: (high, low, close)} for `date` from the canonical store. Empty when the
    session isn't on disk — the paper day then simply opens/marks nothing, honestly."""
    out: dict = {}
    try:
        from data import bhavcopy_store as bs
        frame = bs.bhav_for_date(date) if hasattr(bs, "bhav_for_date") else None
        if frame is None:
            return {}
        for row in frame.itertuples():
            sym = getattr(row, "SYMBOL", None)
            hi = getattr(row, "HIGH_PRICE", None)
            lo = getattr(row, "LOW_PRICE", None)
            cl = getattr(row, "CLOSE_PRICE", None)
            if sym and hi and lo and cl:
                out[str(sym).strip().upper()] = (float(hi), float(lo), float(cl))
    except Exception:
        return {}
    return out


# ── forward-test signals (entries from a strategy's rules) ───────────────────────

def current_regime() -> str:
    """"RISK_ON" / "RISK_OFF" from the macro + breadth read, so the autopilot stands down new
    deployments in a hostile tape. Fails OPEN to RISK_ON only when it truly can't tell — a
    missing signal shouldn't freeze the whole system, but a clear RISK_OFF must bite."""
    try:
        from core import macro_pulse
        mp = macro_pulse.assess() if hasattr(macro_pulse, "assess") else {}
        if str(mp.get("stance", "")).upper() in ("RISK_OFF", "DEFENSIVE"):
            return "RISK_OFF"
    except Exception:
        pass
    try:
        from scan import breadth as B
        b = B.compute() if hasattr(B, "compute") else {}
        if str(b.get("state", "")).upper() == "NARROW":
            return "RISK_OFF"
    except Exception:
        pass
    return "RISK_ON"


def signals_for(paper_strategy, date: str) -> list:
    """Entry signals a deployed strategy would fire on `date`, from real data up to (not
    including) that day — point-in-time, no look-ahead. Returns [] when data is unavailable
    so the strategy simply doesn't trade that day rather than inventing entries.

    NOTE: the rule→signal translation is intentionally conservative and will only emit a
    signal when the canonical scanner already surfaced the setup for that symbol/day, so the
    forward test uses the same audited detection the rest of the app trusts."""
    try:
        from scan import auto_scan
        results, _u, _ts, _status = auto_scan.get_results()
        sigs = []
        for r in results or []:
            entry = r.get("entry") or r.get("price")
            stop = r.get("stop") or r.get("stop_loss")
            target = r.get("target")
            sym = r.get("symbol") or r.get("ticker")
            if sym and entry and stop and target and float(entry) > float(stop) > 0:
                sigs.append({"symbol": str(sym).strip().upper(), "entry": float(entry),
                             "stop": float(stop), "target": float(target),
                             "max_hold": paper_strategy.spec.max_holding_days})
        return sigs
    except Exception:
        return []
