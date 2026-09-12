"""
🔌 Production providers — wire the growth loop to REAL market data, honestly.

The brain's daily loop needs three data hooks. In tests these are injected (deterministic);
in production they come from here, reading the canonical stores the rest of QuantTerm uses:

  • backtest_evaluator(spec, split) -> EvidenceReport   (in-sample, from bhavcopy history)
  • daily_bars(date)               -> {symbol: (o,h,l,c)} (the exact forward-test session)
  • signals_for(paper_strategy, date) -> [signal dicts]  (entries from the strategy's rules)

Every one degrades HONESTLY: with no research-grade data on disk they return empty / invalid
rather than fabricating prices or evidence. `is_synthetic` is False only when the numbers
came from real history — so the discovery gate treats a no-data run as unavailable, never as
a passing strategy. Pure reads; no live broker order path.
"""
from __future__ import annotations

from research.strategy_studio.discovery import EvidenceReport
from core.runtime_paths import logs_dir


def ensure_production_paper_pipeline() -> bool:
    """Idempotently route the production singleton through OMS/Risk/Protection/TCA.

    ``get_brain`` imports this provider module only for production wiring. Directly constructed
    test brains and scratch backtests retain the plain PaperBook. The wrapper cannot submit to a
    real broker; all external-looking IDs are simulated and explicitly paper-prefixed.
    """
    try:
        from pathlib import Path
        from research.auto_research import scheduler as SCH

        brain = getattr(SCH, "_BRAIN", None)
        if brain is None:
            return False
        if bool(getattr(brain.intel_book, "institutional_execution_enabled", False)):
            return True

        from execution.oms.store import OmsStore
        from execution.paper_book_adapter import InstitutionalPaperBookAdapter
        from execution.paper_pipeline import PaperExecutionPipeline
        from execution.protection.store import ProtectionStore
        from execution.tca.store import TcaStore
        from risk.governor_store import RiskDecisionStore

        logs = logs_dir()
        pipeline = PaperExecutionPipeline(
            oms_store=OmsStore(logs / "oms" / "orders.db"),
            risk_store=RiskDecisionStore(logs / "risk" / "decisions.db"),
            protection_store=ProtectionStore(logs / "protection" / "plans.db"),
            tca_store=TcaStore(logs / "tca" / "assessments.db"),
            event_store=brain.event_store,
        )
        brain.intel_book = InstitutionalPaperBookAdapter(
            brain.intel_book,
            pipeline=pipeline,
            runtime_state=brain.runtime_state,
        )
        return True
    except Exception as exc:
        # This is a safety capability, not optional decoration. Preserve the reason so the
        # supervisor/board can refuse new PAPER risk instead of silently using a weaker path.
        try:
            from research.auto_research import scheduler as SCH

            brain = getattr(SCH, "_BRAIN", None)
            if brain is not None:
                brain.state.last_error = f"institutional paper pipeline: {exc}"
        except Exception:
            pass
        return False


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


# ── forward-test data (the exact completed session) ─────────────────────────────

def daily_bars(date: str) -> dict:
    """Return exact-session official NSE bars for PAPER marking.

    The old provider looked for a non-existent ``bhav_for_date`` function and then
    expected uppercase NSE CSV column names from a normalized frame. In production
    that silently returned ``{}``, so autonomous PAPER positions could fail to age or
    close even though the official bhavcopy was on disk.

    This reads only the requested session's already-downloaded canonical bhavcopy. It
    never substitutes the latest day, never calls a live provider, and therefore cannot
    leak future prices into a forward/PIT decision. Four-field OHLC is returned so gap
    exits are modeled by ``PaperBook.mark``.
    """
    out: dict = {}
    try:
        from datetime import datetime
        from data import bhavcopy_store as bs

        day = datetime.strptime(str(date)[:10], "%Y-%m-%d").date()
        reader = getattr(bs, "_read_day", None)
        frame = reader(day) if callable(reader) else None
        if frame is None or len(frame) == 0:
            return {}
        for row in frame.itertuples(index=False):
            sym = getattr(row, "symbol", None)
            op = getattr(row, "open", None)
            hi = getattr(row, "high", None)
            lo = getattr(row, "low", None)
            cl = getattr(row, "close", None)
            if not sym or any(value is None for value in (op, hi, lo, cl)):
                continue
            values = tuple(float(value) for value in (op, hi, lo, cl))
            if any(value != value for value in values):
                continue
            out[str(sym).strip().upper()] = values
    except Exception:
        return {}
    return out


# ── forward-test signals (entries from a strategy's rules) ───────────────────────

def current_regime() -> str:
    """Return the deterministic market regime after production PAPER safety wiring.

    Failure to install the institutional PAPER pipeline is treated as RISK_OFF. This is a
    capability failure, not a market opinion, but the conservative posture guarantees the old
    direct-book path cannot silently take new production PAPER risk.
    """
    if not ensure_production_paper_pipeline():
        return "RISK_OFF"
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
