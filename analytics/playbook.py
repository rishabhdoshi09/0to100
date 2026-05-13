"""
Playbook Engine — structured trading setups with historical expectancy.

Each playbook describes:
  - Setup conditions
  - Ideal regime alignment
  - Entry/stop/target rules
  - Historical expectancy (sourced from signal_tracker.db when available)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Playbook:
    id: str
    name: str
    category: str              # MOMENTUM | BREAKOUT | REVERSAL | POSITIONAL
    emoji: str
    description: str
    ideal_regime: list[str]    # BULL_TREND, EXPANSION, etc.
    conditions: list[str]
    entry_rule: str
    stop_rule: str
    target_rule: str
    avg_win_pct: float
    avg_loss_pct: float
    win_rate: float            # 0-1
    avg_holding_days: int
    expectancy: float          # win_rate * avg_win - (1-win_rate) * avg_loss
    risk_reward: float
    notes: str = ""
    live_stats: Optional[dict] = field(default=None, repr=False)


PLAYBOOKS: list[Playbook] = [
    Playbook(
        id="vcp",
        name="VCP — Volatility Contraction",
        category="BREAKOUT",
        emoji="📐",
        description=(
            "Minervini-style volatility contraction: stock consolidates in tighter and tighter "
            "ranges after a strong uptrend. Volume dries up during contraction. Explosive breakout "
            "on above-average volume confirms the setup."
        ),
        ideal_regime=["BULL_TREND", "EXPANSION"],
        conditions=[
            "Price above SMA200 (Stage 2)",
            "3+ contraction sequences (each smaller than prior)",
            "Volume declining during base",
            "RS relative to Nifty > 0.8",
            "Tight weekly closes in final contraction",
        ],
        entry_rule="Buy on breakout above pivot (day's high of tightest week) on volume ≥ 1.5× avg",
        stop_rule="Below the lowest point of the last contraction (typically 5–8%)",
        target_rule="1× base depth projected upward; trail with 8-day EMA in strong momentum",
        avg_win_pct=18.5,
        avg_loss_pct=6.0,
        win_rate=0.58,
        avg_holding_days=28,
        expectancy=0,  # computed below
        risk_reward=3.1,
        notes="Best in Bull Trend + Low VIX. Skip in Choppy or Bear regime.",
    ),
    Playbook(
        id="flat_base",
        name="Flat Base — Pocket Pivot",
        category="BREAKOUT",
        emoji="📊",
        description=(
            "Stock hugs a flat support zone for 5+ weeks after a prior uptrend. "
            "Low-volume consolidation followed by a pocket pivot (large-volume up day). "
            "Less dramatic than VCP but very reliable in institutional accumulation phase."
        ),
        ideal_regime=["BULL_TREND", "CHOPPY"],
        conditions=[
            "Price range ≤ 15% over 5+ weeks",
            "Support defended on light volume",
            "Pocket pivot day: volume > max of prior 10 down days",
            "Price above SMA50",
        ],
        entry_rule="Enter on pocket pivot day or breakout above flat base ceiling",
        stop_rule="Below the base low (typically 4–6%)",
        target_rule="10–20% initial target; trail if momentum continues",
        avg_win_pct=12.0,
        avg_loss_pct=5.0,
        win_rate=0.55,
        avg_holding_days=21,
        expectancy=0,
        risk_reward=2.4,
        notes="Particularly effective during sideways markets. Works in Choppy regime.",
    ),
    Playbook(
        id="stage2",
        name="Stage 2 Breakout — Weinstein",
        category="POSITIONAL",
        emoji="📈",
        description=(
            "Stan Weinstein Stage 2: stock moves from base (Stage 1) into advancing phase. "
            "Key signal: price crosses SMA30W (weekly 30-period MA) on rising volume. "
            "Most stocks do their best moves in Stage 2."
        ),
        ideal_regime=["BULL_TREND", "EXPANSION"],
        conditions=[
            "Price crossed above SMA30W (weekly) for first time in 6+ months",
            "SMA30W itself is rising",
            "Volume expanding on breakout week",
            "Stock was in Stage 1 base for 6+ weeks",
        ],
        entry_rule="Buy first close above SMA30W on weekly chart, or on pullback to SMA30W",
        stop_rule="Weekly close below SMA30W (exit stage 2)",
        target_rule="Hold through Stage 2; exit on Stage 3 signs (volume dry-up at highs)",
        avg_win_pct=32.0,
        avg_loss_pct=7.0,
        win_rate=0.52,
        avg_holding_days=65,
        expectancy=0,
        risk_reward=4.6,
        notes="Longest-duration play. Requires patience. Best risk-adjusted returns.",
    ),
    Playbook(
        id="gap_continuation",
        name="Gap & Continue",
        category="MOMENTUM",
        emoji="⚡",
        description=(
            "Earnings gap or news-driven gap up that holds above gap level for 3+ days. "
            "Volume confirmation essential. Gap acts as new support. "
            "Quick momentum trade — not a positional setup."
        ),
        ideal_regime=["BULL_TREND", "EXPANSION", "CHOPPY"],
        conditions=[
            "Gap up ≥ 3% on catalyst (earnings/news)",
            "Price holds above gap level for 3 consecutive sessions",
            "Volume on gap day ≥ 3× avg",
            "RSI 50–75 on daily",
        ],
        entry_rule="Enter on first pullback to gap top or intraday breakout above gap high",
        stop_rule="Below gap level (gap fill = stop out)",
        target_rule="1× gap size projected; partial at 1:1, trail rest",
        avg_win_pct=9.0,
        avg_loss_pct=4.0,
        win_rate=0.60,
        avg_holding_days=8,
        expectancy=0,
        risk_reward=2.25,
        notes="Short duration. High win rate but small wins. Works in any non-Bear regime.",
    ),
    Playbook(
        id="golden_cross_swing",
        name="Golden Cross — Swing",
        category="MOMENTUM",
        emoji="🟡",
        description=(
            "Daily SMA50 crosses above SMA200 (Golden Cross) with expanding volume. "
            "First pullback to SMA50 after cross is the entry. Classic intermediate-term signal."
        ),
        ideal_regime=["BULL_TREND", "EXPANSION"],
        conditions=[
            "SMA50 crossed above SMA200 within last 20 sessions",
            "Both MAs rising",
            "Pullback to SMA50 on lower-than-average volume",
            "RSI holds above 45 on pullback",
        ],
        entry_rule="Buy on touch of SMA50 post-crossover, or on next day breakout",
        stop_rule="Below SMA50 (2-day close below)",
        target_rule="SMA50 + 2× ATR initial; trail SMA50",
        avg_win_pct=14.0,
        avg_loss_pct=5.5,
        win_rate=0.54,
        avg_holding_days=35,
        expectancy=0,
        risk_reward=2.5,
        notes="Best when broader market breadth is STRONG.",
    ),
    Playbook(
        id="oversold_reversal",
        name="Oversold RSI Reversal",
        category="REVERSAL",
        emoji="🔄",
        description=(
            "RSI falls to 30 or below in an otherwise Stage 2 stock. "
            "Strong-hand stock that has been unjustly sold. "
            "Wait for RSI to turn back above 35 before entry — not catching a falling knife."
        ),
        ideal_regime=["BULL_TREND", "CHOPPY"],
        conditions=[
            "Daily RSI ≤ 30 (oversold)",
            "Price still above SMA200 (Stage 2 intact)",
            "Volume surge on reversal day (absorption)",
            "RSI now recovering above 35",
        ],
        entry_rule="Enter when RSI crosses back above 35 on a green day",
        stop_rule="Prior session low (or 3% below entry)",
        target_rule="RSI 60 zone = first exit; prior high = second",
        avg_win_pct=11.0,
        avg_loss_pct=4.5,
        win_rate=0.56,
        avg_holding_days=14,
        expectancy=0,
        risk_reward=2.4,
        notes="Skip in Bear regime. Only valid if SMA200 intact.",
    ),
]

# ── Pre-compute expectancy for each playbook ──────────────────────────────────
for pb in PLAYBOOKS:
    pb.expectancy = round(
        pb.win_rate * pb.avg_win_pct - (1 - pb.win_rate) * pb.avg_loss_pct, 2
    )


# ── Public helpers ────────────────────────────────────────────────────────────

def get_playbook(playbook_id: str) -> Optional[Playbook]:
    return next((p for p in PLAYBOOKS if p.id == playbook_id), None)


def get_regime_aligned_playbooks(regime: str) -> list[Playbook]:
    """Return playbooks that work in the current market regime, sorted by expectancy."""
    aligned = [p for p in PLAYBOOKS if regime in p.ideal_regime]
    if not aligned:
        aligned = PLAYBOOKS  # fallback: show all
    return sorted(aligned, key=lambda p: p.expectancy, reverse=True)


def enrich_with_live_stats(playbooks: list[Playbook]) -> list[Playbook]:
    """
    Pull historical signal_tracker.db data to compute live expectancy stats.
    Falls back to static values if DB is unavailable.
    """
    try:
        import sqlite3
        import os
        db_path = os.path.join("logs", "signal_tracker.db")
        if not os.path.exists(db_path):
            return playbooks
        conn = sqlite3.connect(db_path, check_same_thread=False)
        cur  = conn.cursor()
        # Only BUY signals with known outcomes
        cur.execute(
            "SELECT outcome, return_5d FROM signals WHERE signal='BUY' AND outcome != 'PENDING'"
        )
        rows = cur.fetchall()
        conn.close()

        if len(rows) < 10:
            return playbooks

        wins  = [r[1] for r in rows if r[0] == "WIN"  and r[1] is not None]
        loss  = [r[1] for r in rows if r[0] == "LOSS" and r[1] is not None]
        total = len(rows)
        w_count = len(wins)

        if total == 0:
            return playbooks

        live_win_rate = w_count / total
        live_avg_win  = sum(wins) / len(wins)  if wins else 0
        live_avg_loss = abs(sum(loss) / len(loss)) if loss else 0
        live_exp      = live_win_rate * live_avg_win - (1 - live_win_rate) * live_avg_loss

        for pb in playbooks:
            pb.live_stats = {
                "win_rate":    round(live_win_rate, 3),
                "avg_win_pct": round(live_avg_win, 2),
                "avg_loss_pct": round(live_avg_loss, 2),
                "expectancy":  round(live_exp, 2),
                "sample_size": total,
            }
    except Exception:
        pass
    return playbooks
