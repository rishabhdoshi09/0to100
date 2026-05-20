"""
Intelligence Hub — The system's brain. Connects regime, setups, news, and
your personal edge into a single unified intelligence layer.

Think: JARVIS. Always aware. Proactive. Contextual.

Computes the Opportunity Score: one number (0-100) that answers
"How good is today for trading?" by synthesising all market dimensions.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ── Trading Rules (score → hard behaviour) ───────────────────────────────────

@dataclass
class TradingRules:
    """Hard constraints derived from the opportunity score. Not advisory — enforced."""
    max_positions: int
    size_multiplier: float     # 1.0 = full, 0.5 = half
    min_tier: str              # "ELITE_A_PLUS" | "A" | "B" | "WATCHLIST"
    allow_new_trades: bool
    label: str
    color: str
    rationale: str             # one-line reason

    def display_size(self) -> str:
        return f"{int(self.size_multiplier * 100)}% of normal"

    def one_liner(self) -> str:
        if not self.allow_new_trades:
            return f"🚫 NO NEW TRADES — {self.rationale}"
        return (
            f"Max {self.max_positions} position{'s' if self.max_positions != 1 else ''} · "
            f"{self.display_size()} size · {self.min_tier}+ setups only"
        )


def get_trading_rules(opp_score: "OpportunityScore") -> TradingRules:
    """Hard rules enforced by the system based on opportunity score."""
    total = opp_score.total
    if total >= 75:
        return TradingRules(
            max_positions=3, size_multiplier=1.0, min_tier="B",
            allow_new_trades=True, label="Full Aggression", color="#00d4a0",
            rationale="Prime conditions across all dimensions",
        )
    elif total >= 60:
        return TradingRules(
            max_positions=2, size_multiplier=0.8, min_tier="A",
            allow_new_trades=True, label="Standard", color="#00d4ff",
            rationale="Good conditions, standard approach",
        )
    elif total >= 45:
        return TradingRules(
            max_positions=2, size_multiplier=0.6, min_tier="A",
            allow_new_trades=True, label="Cautious", color="#f59e0b",
            rationale="Mixed conditions — reduce exposure, be selective",
        )
    elif total >= 30:
        return TradingRules(
            max_positions=1, size_multiplier=0.5, min_tier="ELITE_A_PLUS",
            allow_new_trades=True, label="Defensive", color="#f97316",
            rationale="Poor conditions — only the very best setups",
        )
    else:
        return TradingRules(
            max_positions=0, size_multiplier=0.0, min_tier="ELITE_A_PLUS",
            allow_new_trades=False, label="STAND DOWN", color="#ff4b4b",
            rationale=f"Score {total:.0f} — conditions too poor to trade",
        )


# ── Opportunity Score breakdown ───────────────────────────────────────────────

@dataclass
class OpportunityScore:
    total: float                    # 0-100, the one number
    grade: str                      # S / A / B / C / D
    label: str                      # "Prime Hunting Ground" etc.
    color: str                      # hex

    regime_score: float             # 0-25
    breadth_score: float            # 0-20
    volatility_score: float         # 0-20
    setup_quality_score: float      # 0-20  (based on top setups available)
    personal_edge_score: float      # 0-15  (your historical edge in this regime)

    regime_note: str
    breadth_note: str
    volatility_note: str
    setup_note: str
    edge_note: str

    recommended_aggression: str     # "FULL (1.0x–1.2x)", "STANDARD (0.8x)", "REDUCED (0.5x)", "CASH"
    max_open_trades: int
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M"))


@dataclass
class MarketAlert:
    level: str              # INFO / WARNING / CRITICAL
    category: str           # REGIME_SHIFT / SETUP_READY / RISK / OPPORTUNITY
    title: str
    body: str
    symbol: Optional[str]
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M"))
    dismissed: bool = False

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ── Core computation ──────────────────────────────────────────────────────────

def compute_opportunity_score(
    regime_state=None,
    top_setups: list[dict] | None = None,
    personal_edge=None,     # AdaptiveEngine instance or None
) -> OpportunityScore:
    """
    Compute the unified Opportunity Score from all available intelligence.
    Degrades gracefully — works with partial data.
    """
    # ── 1. Regime dimension (0-25) ────────────────────────────────────────────
    if regime_state is not None:
        try:
            raw_regime_score = getattr(regime_state, "regime_score", 50.0)
            market_regime    = getattr(regime_state, "market_regime", "UNKNOWN")
            volatility       = getattr(regime_state, "volatility_regime", "NORMAL")
            breadth_label    = getattr(regime_state, "breadth_label", "NEUTRAL")
            breadth_strength = getattr(regime_state, "breadth_strength", 50)
            vix              = getattr(regime_state, "vix", 16.0)
            risk_mode        = getattr(regime_state, "risk_mode", "NEUTRAL")
            inst_activity    = getattr(regime_state, "institutional_activity", "NEUTRAL")
        except Exception:
            raw_regime_score = 50.0
            market_regime = volatility = breadth_label = "UNKNOWN"
            breadth_strength = 50
            vix = 16.0
            risk_mode = inst_activity = "NEUTRAL"
    else:
        raw_regime_score = 50.0
        market_regime = volatility = breadth_label = "UNKNOWN"
        breadth_strength = 50
        vix = 16.0
        risk_mode = inst_activity = "NEUTRAL"

    _REGIME_POINTS = {
        "TRENDING_BULL": 25, "EXPANSION": 23,
        "CHOPPY": 12, "COMPRESSION": 10,
        "DISTRIBUTION": 6, "TRENDING_BEAR": 3, "UNKNOWN": 10,
    }
    regime_pts = float(_REGIME_POINTS.get(market_regime, 10))

    if inst_activity == "ACCUMULATION":
        regime_pts = min(25, regime_pts + 2)
    elif inst_activity == "DISTRIBUTION":
        regime_pts = max(0, regime_pts - 3)

    _REGIME_NOTES = {
        "TRENDING_BULL": "Strong bull trend — ideal for breakout and momentum strategies",
        "EXPANSION": "Market expanding — maximum aggression warranted",
        "CHOPPY": "Choppy market — reduce size, trade mean-reversion only",
        "COMPRESSION": "Market compressing — wait for breakout, no new trades",
        "DISTRIBUTION": "Distribution detected — protect capital, reduce exposure",
        "TRENDING_BEAR": "Bear trend — cash is a position, strict stop discipline",
        "UNKNOWN": "Regime unclear — trade minimum size until signal emerges",
    }
    regime_note = _REGIME_NOTES.get(market_regime, "")

    # ── 2. Breadth dimension (0-20) ───────────────────────────────────────────
    breadth_pts = {"STRONG": 20, "NEUTRAL": 12, "WEAK": 4}.get(breadth_label, 10)
    breadth_note = {
        "STRONG": f"Broad participation ({breadth_strength}/100) — rally has legs",
        "NEUTRAL": f"Mixed breadth ({breadth_strength}/100) — selective environment",
        "WEAK":    f"Thin breadth ({breadth_strength}/100) — narrow leaders only",
    }.get(breadth_label, "Breadth data unavailable")

    # ── 3. Volatility dimension (0-20) ───────────────────────────────────────
    if vix is None or vix != vix:  # NaN check
        vix = 16.0

    if vix < 12:
        vol_pts = 18
        vol_note = f"VIX {vix:.1f} — ultra-low volatility, tighten stops, targets easy"
    elif vix < 16:
        vol_pts = 20
        vol_note = f"VIX {vix:.1f} — ideal volatility window, standard sizing"
    elif vix < 20:
        vol_pts = 15
        vol_note = f"VIX {vix:.1f} — elevated, widen stops 10-15%"
    elif vix < 25:
        vol_pts = 8
        vol_note = f"VIX {vix:.1f} — high volatility, reduce size 30%"
    else:
        vol_pts = 2
        vol_note = f"VIX {vix:.1f} — PANIC zone, no new longs, protect capital"

    # ── 4. Setup quality dimension (0-20) ─────────────────────────────────────
    if top_setups:
        tier_weights = {"ELITE_A_PLUS": 20, "A": 16, "B": 11, "WATCHLIST": 6, "AVOID": 0}
        top_tiers = [s.get("quality_tier", "WATCHLIST") for s in top_setups[:5]]
        if top_tiers:
            setup_pts = float(sum(tier_weights.get(t, 6) for t in top_tiers) / len(top_tiers))
        else:
            setup_pts = 8.0

        elite_count = sum(1 for t in top_tiers if t == "ELITE_A_PLUS")
        a_count = sum(1 for t in top_tiers if t == "A")

        if elite_count >= 2:
            setup_note = f"{elite_count} ELITE setups available — high-conviction day"
        elif elite_count == 1:
            setup_note = f"1 ELITE + {a_count} A-tier setups — selective opportunity"
        elif a_count >= 2:
            setup_note = f"{a_count} A-tier setups — decent opportunity, standard sizing"
        elif top_tiers:
            setup_note = f"{len(top_tiers)} B/Watchlist setups — trade small, be patient"
        else:
            setup_note = "No quality setups found — sit on hands"
    else:
        setup_pts = 8.0
        setup_note = "Run scan to evaluate setup quality"

    # ── 5. Personal edge dimension (0-15) ─────────────────────────────────────
    if personal_edge is not None:
        try:
            stats = personal_edge.get_summary_stats()
            total_trades = stats.get("total_trades", 0)
            personal_win_rate = stats.get("win_rate", 0.5)
            if total_trades >= 20:
                edge_pts = min(15, int(personal_win_rate * 20))
                edge_note = f"Your edge: {personal_win_rate*100:.0f}% win rate over {total_trades} trades"
            elif total_trades >= 5:
                edge_pts = 8
                edge_note = f"Building edge data ({total_trades} trades) — keep journaling"
            else:
                edge_pts = 8
                edge_note = "No personal trade history yet — using baseline rates"
        except Exception:
            edge_pts = 8
            edge_note = "Personal edge data unavailable"
    else:
        edge_pts = 8
        edge_note = "Journal trades to personalise edge scoring"

    # ── Aggregate ─────────────────────────────────────────────────────────────
    total = round(regime_pts + breadth_pts + vol_pts + setup_pts + edge_pts, 1)
    total = max(0.0, min(100.0, total))

    if total >= 82:
        grade, label, color = "S", "Prime Hunting Ground", "#f5c518"
    elif total >= 68:
        grade, label, color = "A", "Strong Opportunity", "#00d4a0"
    elif total >= 52:
        grade, label, color = "B", "Selective Environment", "#00d4ff"
    elif total >= 38:
        grade, label, color = "C", "Cautious Mode", "#f59e0b"
    else:
        grade, label, color = "D", "Preserve Capital", "#ff4b4b"

    # Aggression level
    if total >= 75 and market_regime in ("TRENDING_BULL", "EXPANSION"):
        aggression = "FULL (1.0×–1.2×)"
        max_trades = 5
    elif total >= 55:
        aggression = "STANDARD (0.8×)"
        max_trades = 4
    elif total >= 38:
        aggression = "REDUCED (0.5×)"
        max_trades = 2
    else:
        aggression = "CASH — No new trades"
        max_trades = 0

    return OpportunityScore(
        total=total, grade=grade, label=label, color=color,
        regime_score=regime_pts, breadth_score=breadth_pts,
        volatility_score=vol_pts, setup_quality_score=setup_pts,
        personal_edge_score=edge_pts,
        regime_note=regime_note, breadth_note=breadth_note,
        volatility_note=vol_note, setup_note=setup_note, edge_note=edge_note,
        recommended_aggression=aggression, max_open_trades=max_trades,
    )


# ── Alert generation ──────────────────────────────────────────────────────────

def generate_alerts(
    regime_state=None,
    prev_regime: str = "",
    top_setups: list[dict] | None = None,
    opp_score: OpportunityScore | None = None,
) -> list[MarketAlert]:
    """Generate proactive alerts based on current market state."""
    alerts: list[MarketAlert] = []
    now = datetime.now().strftime("%H:%M")

    if regime_state is None:
        return alerts

    market_regime  = getattr(regime_state, "market_regime", "UNKNOWN")
    vix            = getattr(regime_state, "vix", 16.0) or 16.0
    breadth        = getattr(regime_state, "breadth_label", "NEUTRAL")
    nifty_chg      = getattr(regime_state, "nifty_change_1d", 0.0) or 0.0

    # Regime shift
    if prev_regime and prev_regime != market_regime and market_regime != "UNKNOWN":
        if market_regime in ("TRENDING_BULL", "EXPANSION"):
            alerts.append(MarketAlert(
                level="INFO", category="REGIME_SHIFT",
                title=f"Regime upgraded: {prev_regime} → {market_regime}",
                body="Market has shifted to a favourable regime. Consider increasing exposure.",
                symbol=None, timestamp=now,
            ))
        elif market_regime in ("DISTRIBUTION", "TRENDING_BEAR"):
            alerts.append(MarketAlert(
                level="CRITICAL", category="REGIME_SHIFT",
                title=f"Regime deteriorated: {prev_regime} → {market_regime}",
                body="Market structure has broken down. Tighten stops, reduce new entries.",
                symbol=None, timestamp=now,
            ))
        elif market_regime in ("CHOPPY", "COMPRESSION"):
            alerts.append(MarketAlert(
                level="WARNING", category="REGIME_SHIFT",
                title=f"Regime weakened: {prev_regime} → {market_regime}",
                body="Breakout environment unfavourable. Switch to mean-reversion or wait.",
                symbol=None, timestamp=now,
            ))

    # VIX spike
    if vix > 22:
        alerts.append(MarketAlert(
            level="WARNING" if vix < 27 else "CRITICAL",
            category="RISK",
            title=f"VIX elevated at {vix:.1f}",
            body=f"Volatility spike — widen stops by 15-20%, reduce position size to 0.6×.",
            symbol=None, timestamp=now,
        ))

    # Nifty sharp move
    if abs(nifty_chg) > 1.5:
        direction = "up" if nifty_chg > 0 else "down"
        alerts.append(MarketAlert(
            level="INFO" if nifty_chg > 0 else "WARNING",
            category="OPPORTUNITY" if nifty_chg > 0 else "RISK",
            title=f"Nifty {direction} {abs(nifty_chg):.1f}% today",
            body=("Strong breadth move — momentum setups likely to follow through."
                  if nifty_chg > 0
                  else "Market selling off — check open stop levels, avoid new longs today."),
            symbol=None, timestamp=now,
        ))

    # Elite setup alerts
    if top_setups:
        elite = [s for s in top_setups if s.get("quality_tier") == "ELITE_A_PLUS"]
        for s in elite[:2]:
            sym = s.get("symbol", "")
            ev = s.get("expected_value_r", 0)
            arch = s.get("archetype", "").replace("_", " ")
            alerts.append(MarketAlert(
                level="INFO", category="SETUP_READY",
                title=f"ELITE setup: {sym} ({arch})",
                body=f"Expected value +{ev:.1f}R. Check entry zone and set alerts on pivot.",
                symbol=sym, timestamp=now,
            ))

    # Opportunity score extremes
    if opp_score is not None:
        if opp_score.total >= 80:
            alerts.append(MarketAlert(
                level="INFO", category="OPPORTUNITY",
                title=f"Grade S day — Opportunity Score {opp_score.total:.0f}/100",
                body=f"{opp_score.label}. {opp_score.recommended_aggression}. Max {opp_score.max_open_trades} concurrent positions.",
                symbol=None, timestamp=now,
            ))
        elif opp_score.total <= 30:
            alerts.append(MarketAlert(
                level="WARNING", category="RISK",
                title=f"Low opportunity — Score {opp_score.total:.0f}/100",
                body="Poor trading conditions. Sit in cash or reduce to minimum size.",
                symbol=None, timestamp=now,
            ))

    return alerts


# ── JARVIS Context Builder ────────────────────────────────────────────────────

def build_jarvis_context(
    regime_state=None,
    top_setups: list[dict] | None = None,
    opp_score: OpportunityScore | None = None,
    personal_edge=None,
) -> str:
    """
    Builds the full system context string injected into every JARVIS conversation.
    This is what makes JARVIS aware of the current market state.
    """
    lines = ["=== QUANTTERM SYSTEM CONTEXT ===", f"Time: {datetime.now().strftime('%H:%M IST')}"]

    if regime_state is not None:
        lines += [
            f"\nMARKET REGIME: {getattr(regime_state, 'market_regime', 'UNKNOWN')}",
            f"Regime Score: {getattr(regime_state, 'regime_score', 50):.0f}/100",
            f"Nifty 50: ₹{getattr(regime_state, 'nifty_price', 0):,.0f} "
            f"({getattr(regime_state, 'nifty_change_1d', 0):+.2f}%)",
            f"India VIX: {getattr(regime_state, 'vix', 16):.1f}",
            f"Breadth: {getattr(regime_state, 'breadth_label', 'NEUTRAL')} "
            f"({getattr(regime_state, 'breadth_strength', 50)}/100)",
            f"Institutional Activity: {getattr(regime_state, 'institutional_activity', 'NEUTRAL')}",
            f"Risk Mode: {getattr(regime_state, 'risk_mode', 'NEUTRAL')}",
            f"Leading Sectors: {', '.join(getattr(regime_state, 'leading_sectors', []))}",
            f"Lagging Sectors: {', '.join(getattr(regime_state, 'lagging_sectors', []))}",
            f"Recommended Playbooks: {', '.join(getattr(regime_state, 'recommended_playbooks', []))}",
        ]

    if opp_score is not None:
        lines += [
            f"\nOPPORTUNITY SCORE: {opp_score.total:.0f}/100 (Grade {opp_score.grade} — {opp_score.label})",
            f"Recommended Aggression: {opp_score.recommended_aggression}",
            f"Max Concurrent Trades: {opp_score.max_open_trades}",
        ]

    if top_setups:
        lines.append(f"\nTOP SETUPS ({len(top_setups)} in queue):")
        for s in top_setups[:5]:
            sym   = s.get("symbol", "")
            arch  = s.get("archetype", "").replace("_", " ")
            tier  = s.get("quality_tier", "")
            ev    = s.get("expected_value_r", 0)
            price = s.get("price", 0)
            lines.append(f"  • {sym}: {arch} | {tier} | EV +{ev:.1f}R | ₹{price:,.0f}")

    if personal_edge is not None:
        try:
            stats = personal_edge.get_summary_stats()
            if stats.get("total_trades", 0) > 0:
                lines += [
                    f"\nPERSONAL EDGE:",
                    f"  Trades: {stats['total_trades']} | Win Rate: {stats.get('win_rate',0)*100:.0f}%",
                    f"  Best Regime: {stats.get('best_regime', 'N/A')}",
                    f"  Best Playbook: {stats.get('best_playbook', 'N/A')}",
                    f"  Current Streak: {stats.get('streak_count', 0)} {stats.get('streak_current', '')}",
                ]
        except Exception:
            pass

    if opp_score is not None:
        rules = get_trading_rules(opp_score)
        lines += [
            f"\nACTIVE TRADING RULES (score {opp_score.total:.0f}/100):",
            f"  {rules.one_liner()}",
            f"  Rationale: {rules.rationale}",
        ]

    lines.append("\n=== END CONTEXT ===")
    return "\n".join(lines)
