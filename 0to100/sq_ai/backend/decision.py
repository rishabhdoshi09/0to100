"""Claude decision engine – runs every 5 min during market hours.

Reads top-10 from ``screener_results``, builds the per-symbol decision
prompt (technical + news + portfolio context), asks Claude, applies the
ensemble veto for size > threshold, executes via ``Executor``, and logs
each decision row to SQLite ``signals``.

Falls back to the deterministic ML+regime composite when Claude is
unavailable (or returns invalid JSON).
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import pytz

from sq_ai.backend.claude_client import ClaudeClient as ClaudeBriefClient  # legacy hi-level
from sq_ai.backend.data_fetcher import fetch_news, fetch_yf_history
from sq_ai.backend.ensemble import (
    CLAUDE_DECISION_SYSTEM,
    EnsembleVeto,
    build_decision_prompt,
    parse_decision,
)
from sq_ai.backend.executor import Executor, Order
from sq_ai.backend.llm_clients import ClaudeClient, DeepSeekClient
from sq_ai.backend.risk_manager import RiskManager
from sq_ai.portfolio.tracker import PortfolioTracker
from sq_ai.signals.composite_signal import CompositeSignal


IST = pytz.timezone(os.environ.get("SQ_TIMEZONE", "Asia/Kolkata"))


class DecisionEngine:
    def __init__(self,
                 tracker: PortfolioTracker | None = None,
                 composite: CompositeSignal | None = None,
                 claude: ClaudeClient | None = None,
                 deepseek: DeepSeekClient | None = None,
                 executor: Executor | None = None,
                 risk: RiskManager | None = None,
                 ensemble_threshold_pct: float = 10.0) -> None:
        self.tracker = tracker or PortfolioTracker()
        self.composite = composite or CompositeSignal()
        self.claude = claude or ClaudeClient()
        self.deepseek = deepseek or DeepSeekClient()
        self.executor = executor or Executor(self.tracker)
        self.risk = risk or RiskManager()
        self.ensemble = EnsembleVeto(
            tracker=self.tracker, claude=self.claude, deepseek=self.deepseek,
            threshold_pct=ensemble_threshold_pct,
        )
        # legacy brief-builder used only for the snapshot prompt – ok if absent
        self._brief = ClaudeBriefClient()

    # ------------------------------------------------------------- snapshot
    def _snapshot(self, last_prices: dict[str, float]) -> dict[str, Any]:
        positions = self.tracker.open_positions()
        eq_curve = self.tracker.equity_curve()
        cash = eq_curve[-1]["cash"] if eq_curve else float(
            os.environ.get("SQ_INITIAL_EQUITY", 1_000_000)
        )
        gross = unreal = 0.0
        for p in positions:
            ltp = last_prices.get(p["symbol"], p["entry_price"])
            gross += ltp * p["qty"]
            unreal += (ltp - p["entry_price"]) * p["qty"]
        equity = cash + gross
        equity_open = eq_curve[-1]["equity"] if eq_curve else equity
        return {
            "cash": cash, "equity": equity, "gross_exposure": gross,
            "exposure_pct": (gross / equity * 100) if equity else 0.0,
            "unrealized_pnl": unreal,
            "daily_pnl_pct": ((equity - equity_open) / equity_open * 100)
                             if equity_open else 0.0,
            "positions": positions,
        }

    # ------------------------------------------------------------- one cycle
    def run(self) -> dict[str, Any]:
        ts = datetime.now(IST)
        cycle: dict[str, Any] = {
            "timestamp": ts.isoformat(), "decisions": [], "events": [],
            "used_claude": False, "vetoes": 0,
        }
        screener = self.tracker.latest_screener()
        if not screener:
            cycle["note"] = "no screener results yet"
            return cycle
        watchlist = [r["symbol"] for r in screener[:10]]

        # 1. fetch features per watchlist symbol
        features_per_sym: list[dict] = []
        last_prices: dict[str, float] = {}
        composite_results: dict[str, dict] = {}
        for sym in watchlist:
            try:
                df = fetch_yf_history(sym, period="6mo", interval="1d")
                if len(df) < 60:
                    continue
                feats = self.composite.compute_indicators(df)
                feats["symbol"] = sym
                if self.composite.ml.loaded:
                    order = self.composite.ml.feature_names or self.composite.DEFAULT_FEATURES
                    x = [float(feats.get(n, 1 if n == "regime" else 0.0)) for n in order]
                    feats["ml_proba_up"] = self.composite.ml.predict_proba_up(x)
                else:
                    feats["ml_proba_up"] = 0.5
                features_per_sym.append(feats)
                last_prices[sym] = feats["close"]
                composite_results[sym] = self.composite.compute(feats)
            except Exception as exc:                       # pragma: no cover
                cycle["events"].append({"symbol": sym, "error": str(exc)})

        # 2. exits before new entries
        cycle["events"].extend(self.executor.check_exits(last_prices))

        # 3. portfolio snapshot
        snap = self._snapshot(last_prices)
        cycle["snapshot"] = snap

        # 4. per-symbol Claude decision (with fallback) + ensemble veto
        for f in features_per_sym:
            sym = f["symbol"]
            news = fetch_news(sym.replace(".NS", ""), top_n=3)
            prompt = build_decision_prompt(sym, f, news, snap)
            raw = self.claude.generate(
                prompt, max_tokens=400, temperature=0.2,
                system=CLAUDE_DECISION_SYSTEM,
            )
            decision = parse_decision(raw)

            if decision is None:
                # Fallback: use composite signal directly
                comp = composite_results.get(sym, {})
                direction = comp.get("direction", 0)
                conf = comp.get("confidence", 0.0) / 100.0
                regime = int(f.get("regime", 1))
                if direction == 1 and regime != 0 and conf >= 0.5:
                    action = "BUY"
                else:
                    action = "HOLD"
                price = f.get("close", 0)
                atr_v = f.get("atr", 0)
                decision = {
                    "action": action,
                    "size_pct": min(5.0, conf * 10) if action == "BUY" else 0.0,
                    "stop": price - 2 * atr_v if action == "BUY" else 0.0,
                    "target": price + 3 * atr_v if action == "BUY" else 0.0,
                    "confidence": conf,
                    "reasoning": "fallback ml+regime",
                }
            else:
                cycle["used_claude"] = True

            # 5. ensemble veto for big trades
            veto = self.ensemble.maybe_veto(sym, decision, prompt)
            decision["veto"] = veto
            if veto.get("vetoed"):
                cycle["vetoes"] += 1
                decision["action"] = veto["final_action"]

            # 6. log signal
            self.tracker.log_signal(
                symbol=sym, action=decision["action"],
                confidence=float(decision.get("confidence", 0)),
                regime=int(f.get("regime", 1)),
                reasoning=decision.get("reasoning", ""),
                extra={"size_pct": decision.get("size_pct", 0),
                       "veto": veto},
            )

            # 7. execute BUY only
            if decision["action"] == "BUY":
                price = last_prices.get(sym)
                atr_v = f.get("atr", 0.0)
                if not price or atr_v <= 0:
                    continue
                qty = self.risk.position_size(
                    equity=snap["equity"], price=price, atr_value=atr_v,
                    confidence=float(decision.get("confidence", 0)),
                )
                if qty <= 0:
                    continue
                stop = float(decision.get("stop") or (price - 2 * atr_v))
                target = float(decision.get("target") or (price + 3 * atr_v))
                exec_res = self.executor.buy(
                    Order(sym, "BUY", qty, price, stop, target),
                    reasoning=decision.get("reasoning", ""),
                    regime=int(f.get("regime", 1)),
                )
                decision["execution"] = exec_res
            cycle["decisions"].append({"symbol": sym, **decision})

        return cycle


__all__ = ["DecisionEngine"]
