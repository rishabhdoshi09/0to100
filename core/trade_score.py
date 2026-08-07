"""
Composite Trade Score — the Infinity Gauntlet.

Every signal in the system feeds this single engine.
Output: one score (0-100), one action, one position size.

This is the final answer to: "Should I take this trade?"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── CompositeScore ─────────────────────────────────────────────────────────────

@dataclass
class CompositeScore:
    symbol: str
    final_score: float          # 0-100
    action: str                 # "BUY" | "WAIT" | "SKIP" | "BLOCKED"
    confidence: str             # "HIGH" | "MEDIUM" | "LOW"
    position_size_pct: float    # suggested % of capital

    # Component scores (each 0-100, None if unavailable)
    quality_score: Optional[float]
    regime_score: Optional[float]
    timing_score: Optional[float]
    rs_score: Optional[float]
    freshness_score: Optional[float]
    personal_edge_score: Optional[float]

    # Multipliers applied (each 0.0-1.0)
    whipsaw_mult: float
    earnings_mult: float
    breakout_memory_mult: float

    # What helped / what hurt
    boosters: list[str]         # factors that increased score
    detractors: list[str]       # factors that decreased score
    blockers: list[str]         # hard blocks (iron lock, flash crash)

    # Human-readable verdict
    verdict_line: str           # "Strong VCP with RS 92. Buy above ₹2450."
    risk_line: str              # "Risk 1.8% of capital. Stop ₹2380."


# ── Archetype plain-English map ────────────────────────────────────────────────

_ARCHETYPE_PLAIN = {
    "VCP_BREAKOUT":           "tight coil",
    "MOMENTUM_EXPANSION":     "momentum surge",
    "EARLY_LEADER":           "early stage leader",
    "ACCUMULATION_BREAKOUT":  "quiet accumulation",
    "EARNINGS_CONTINUATION":  "earnings follow-through",
    "FAILED_BREAKOUT":        "failed breakout",
    "MEAN_REVERSION":         "mean reversion",
    "TREND_CONTINUATION":     "trend continuation",
    "HIGH_TIGHT_FLAG":        "high tight flag",
}


# ── Main scoring function ──────────────────────────────────────────────────────

def compute_composite_score(
    setup: dict,
    regime: dict,
    capital: float = 1_000_000,
) -> CompositeScore:
    """
    Combine every system signal into one composite score and one actionable output.

    Parameters
    ----------
    setup   : dict produced by _run_pipeline() in institutional_terminal.
    regime  : dict from compute_regime() (has 'market', 'confidence', etc.)
    capital : total capital in INR (used for position sizing display).
    """
    symbol   = setup.get("symbol", "UNKNOWN")
    archetype = setup.get("archetype", "UNKNOWN")

    boosters: list[str] = []
    detractors: list[str] = []
    blockers: list[str] = []

    # ------------------------------------------------------------------
    # Step 1: Base score from quality engine
    # ------------------------------------------------------------------
    score: float = float(setup.get("score", 50))
    quality_score: Optional[float] = score

    # ------------------------------------------------------------------
    # Step 2: Regime alignment bonus / penalty
    # ------------------------------------------------------------------
    reg_align = setup.get("reg_align", "NEUTRAL")
    regime_bonus_map = {
        "STRONG":  15,
        "ALIGNED": 10,
        "NEUTRAL":  0,
        "WEAK":   -10,
        "COUNTER": -20,
    }
    regime_bonus = regime_bonus_map.get(reg_align, 0)
    score += regime_bonus
    regime_score: Optional[float] = max(0.0, min(100.0, 50.0 + regime_bonus * 2.5))
    if regime_bonus > 0:
        boosters.append(f"Regime alignment: {reg_align} ({regime_bonus:+d} pts)")
    elif regime_bonus < 0:
        detractors.append(f"Regime misalignment: {reg_align} ({regime_bonus:+d} pts)")

    # ------------------------------------------------------------------
    # Step 3: Timing multiplier
    # ------------------------------------------------------------------
    timing_score: Optional[float] = None
    timing_mult: float = 1.0
    try:
        from core.market_timing import get_timing_score
        timing = get_timing_score()
        timing_raw = float(timing.get("score", 100))
        timing_mult = timing_raw / 100.0
        timing_score = timing_raw
        if timing_raw < 40:
            reason = timing.get("reason", "")
            detractors.append(
                f"Bad timing: {timing.get('label', '')} ({reason[:40]})"
            )
        elif timing_raw >= 85:
            boosters.append(f"Optimal entry timing: {timing.get('label', '')} ({timing_raw:.0f}/100)")
    except Exception:
        timing_mult = 1.0

    score = score * timing_mult

    # ------------------------------------------------------------------
    # Step 4: Personal edge adjustment
    # ------------------------------------------------------------------
    personal_edge_score: Optional[float] = None
    try:
        from core.learning_loop import get_personal_edge
        edge = get_personal_edge()
        arch_edge = edge.get(archetype, {})
        if arch_edge.get("trades", 0) >= 5:
            vs_base = arch_edge.get("vs_baseline", 0.0)
            personal_edge_score = max(0.0, min(100.0, 50.0 + vs_base * 100))
            delta = vs_base * 30
            score += delta
            if vs_base > 0.1:
                boosters.append(
                    f"Your personal edge in {archetype}: +{vs_base * 100:.0f}% vs baseline"
                )
            elif vs_base < -0.1:
                detractors.append(
                    f"Below-average personal edge in {archetype}"
                )
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Step 5: RS score bonus
    # ------------------------------------------------------------------
    rs_score: Optional[float] = None
    rs: float = 50.0
    try:
        import streamlit as st  # type: ignore
        cached_rs = st.session_state.get(f"rs_{symbol}", None)
        if cached_rs is not None:
            rs = float(cached_rs)
        rs_score = rs
        if rs >= 90:
            score += 8
            boosters.append(f"RS {rs:.0f} — top 10% strength")
        elif rs < 40:
            score -= 8
            detractors.append(f"RS {rs:.0f} — underperforming")
        elif rs >= 75:
            boosters.append(f"RS {rs:.0f} — above-average strength")
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Step 6: Freshness score
    # ------------------------------------------------------------------
    freshness_score: Optional[float] = None
    try:
        import streamlit as st  # type: ignore  # noqa: F811
        fresh = st.session_state.get(f"fresh_{symbol}", {})
        status = fresh.get("status", "UNKNOWN")
        freshness_map = {
            "OPTIMAL": 5,
            "FRESH":   2,
            "MATURE":  0,
            "EARLY":  -3,
            "STALE": -10,
        }
        _freshness_score_map = {
            "OPTIMAL": 95,
            "FRESH":   80,
            "MATURE":  60,
            "EARLY":   40,
            "STALE":   15,
        }
        delta = freshness_map.get(status, 0)
        freshness_score = float(_freshness_score_map.get(status, 50))
        score += delta
        if status == "STALE":
            detractors.append(f"Setup is stale ({fresh.get('note', '')})")
        elif status == "OPTIMAL":
            boosters.append("Setup in optimal freshness window")
        elif status == "EARLY":
            detractors.append(f"Setup too early ({fresh.get('note', '')})")
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Step 7: Apply multipliers
    # ------------------------------------------------------------------

    # Whipsaw
    whipsaw_mult: float = 1.0
    try:
        from risk.whipsaw_guard import compute_whipsaw_score
        ws = compute_whipsaw_score()
        whipsaw_mult = float(ws.size_multiplier)
        if whipsaw_mult < 1.0:
            detractors.append(
                f"Whipsaw market: {ws.verdict} (size ×{whipsaw_mult:.2f})"
            )
    except Exception:
        whipsaw_mult = 1.0

    # Earnings
    earnings_mult: float = 1.0
    try:
        from data.earnings_calendar import get_earnings_risk
        er = get_earnings_risk(symbol)
        earnings_mult = float(er.get("size_multiplier", 1.0))
        if earnings_mult < 1.0:
            detractors.append(f"Earnings risk: {er.get('note', '')}")
    except Exception:
        earnings_mult = 1.0

    # Breakout memory
    bm_mult: float = 1.0
    try:
        from scan.breakout_memory import confidence_penalty
        bm_mult = float(confidence_penalty(symbol))
        if bm_mult < 1.0:
            detractors.append(f"False breakout history: confidence ×{bm_mult:.2f}")
    except Exception:
        bm_mult = 1.0

    # ------------------------------------------------------------------
    # Step 8: Hard blocks (score → 0, action → BLOCKED)
    # ------------------------------------------------------------------

    # Flash crash guard
    try:
        from risk.flash_crash_guard import is_flash_crash_active
        active, reason = is_flash_crash_active()
        if active:
            blockers.append(f"Flash crash guard: {reason}")
    except Exception:
        pass

    # Kill switch
    try:
        import streamlit as st  # type: ignore  # noqa: F811
        if st.session_state.get("kill_switch_active"):
            blockers.append("Kill switch active")
    except Exception:
        pass

    if blockers:
        return CompositeScore(
            symbol=symbol,
            final_score=0.0,
            action="BLOCKED",
            confidence="LOW",
            position_size_pct=0.0,
            quality_score=quality_score,
            regime_score=regime_score,
            timing_score=timing_score,
            rs_score=rs_score,
            freshness_score=freshness_score,
            personal_edge_score=personal_edge_score,
            whipsaw_mult=whipsaw_mult,
            earnings_mult=earnings_mult,
            breakout_memory_mult=bm_mult,
            boosters=boosters,
            detractors=detractors,
            blockers=blockers,
            verdict_line="Trade blocked — see blockers.",
            risk_line="Position size: 0% (blocked).",
        )

    # ------------------------------------------------------------------
    # Step 9: Final score and action
    # ------------------------------------------------------------------
    final = max(0.0, min(100.0, score * whipsaw_mult * earnings_mult * bm_mult))

    if final >= 75:
        action = "BUY"
    elif final >= 55:
        action = "WAIT"
    else:
        action = "SKIP"

    confidence = "HIGH" if final >= 80 else "MEDIUM" if final >= 60 else "LOW"

    # ------------------------------------------------------------------
    # Step 10: Position sizing
    # ------------------------------------------------------------------
    base_size = float(setup.get("pos_pct", 2.0))
    final_size = base_size * whipsaw_mult * earnings_mult
    final_size = min(final_size, 10.0)  # hard cap 10%

    # ------------------------------------------------------------------
    # Step 11: Generate verdict lines
    # ------------------------------------------------------------------
    plain = _ARCHETYPE_PLAIN.get(archetype, archetype.replace("_", " ").lower())
    pivot = setup.get("pivot", 0)
    stop  = setup.get("stop", 0)
    risk_pct = float(setup.get("risk_pct", 2.0))

    verb = action if action != "SKIP" else "Avoid"
    verdict_line = (
        f"{plain.title()} with RS {rs:.0f}. "
        f"{verb} above ₹{pivot:,.0f}."
    )

    risk_pct_capital = final_size * risk_pct / 100.0
    risk_line = (
        f"Risk {risk_pct_capital:.1f}% of capital. "
        f"Stop ₹{stop:,.0f}."
    )

    return CompositeScore(
        symbol=symbol,
        final_score=final,
        action=action,
        confidence=confidence,
        position_size_pct=final_size,
        quality_score=quality_score,
        regime_score=regime_score,
        timing_score=timing_score,
        rs_score=rs_score,
        freshness_score=freshness_score,
        personal_edge_score=personal_edge_score,
        whipsaw_mult=whipsaw_mult,
        earnings_mult=earnings_mult,
        breakout_memory_mult=bm_mult,
        boosters=boosters,
        detractors=detractors,
        blockers=blockers,
        verdict_line=verdict_line,
        risk_line=risk_line,
    )


# ── Streamlit render component ─────────────────────────────────────────────────

def render_composite_score(cs: CompositeScore) -> None:
    """
    Render the composite score as a styled Streamlit card.

    Layout:
      ┌──────────────────────────────────────────────┐
      │  COMPOSITE SCORE: 84/100  ●●●●●●●●○○  BUY   │
      │  Strong conviction · High confidence          │
      │                                              │
      │  ✅ Quality 78  ✅ RS 92  ✅ Regime +15      │
      │  ✅ Personal edge +12%  ⚠️ Timing 65         │
      │  📅 Earnings mult ×0.75                      │
      │                                              │
      │  Buy above ₹2,450 · Risk 1.4% of capital    │
      └──────────────────────────────────────────────┘
    """
    import streamlit as st  # type: ignore

    score = cs.final_score
    action = cs.action

    # Border / glow based on action
    if action == "BUY":
        border_col = "#00d4a0"
        badge_bg   = "#00d4a022"
        glow       = "0 0 24px #00d4a033"
        action_col = "#00d4a0"
    elif action == "WAIT":
        border_col = "#f59e0b"
        badge_bg   = "#f59e0b22"
        glow       = "0 0 20px #f59e0b22"
        action_col = "#f59e0b"
    elif action == "BLOCKED":
        border_col = "#ff4b4b"
        badge_bg   = "#ff4b4b22"
        glow       = "0 0 20px #ff4b4b33"
        action_col = "#ff4b4b"
    else:  # SKIP
        border_col = "#4a5568"
        badge_bg   = "#4a556822"
        glow       = "none"
        action_col = "#4a5568"

    # Dot progress bar (10 dots)
    filled = int(round(score / 10))
    dots = "●" * filled + "○" * (10 - filled)

    # Conviction label
    if score >= 80:
        conviction = "Strong conviction"
    elif score >= 65:
        conviction = "Moderate conviction"
    elif score >= 50:
        conviction = "Weak conviction"
    else:
        conviction = "Low conviction"

    # ── Build component rows ───────────────────────────────────────────
    components_html = ""

    # Quality score
    if cs.quality_score is not None:
        q_icon = "✅" if cs.quality_score >= 68 else "⚠️" if cs.quality_score >= 50 else "❌"
        q_col  = "#00d4a0" if cs.quality_score >= 68 else "#f59e0b" if cs.quality_score >= 50 else "#ff4b4b"
        components_html += (
            f"<span style='color:{q_col}'>{q_icon} Quality {cs.quality_score:.0f}</span>&nbsp;&nbsp;"
        )

    # RS score
    if cs.rs_score is not None:
        rs_icon = "✅" if cs.rs_score >= 75 else "⚠️" if cs.rs_score >= 50 else "❌"
        rs_col  = "#00d4a0" if cs.rs_score >= 75 else "#f59e0b" if cs.rs_score >= 50 else "#ff4b4b"
        components_html += (
            f"<span style='color:{rs_col}'>{rs_icon} RS {cs.rs_score:.0f}</span>&nbsp;&nbsp;"
        )

    # Regime score
    if cs.regime_score is not None:
        r_icon = "✅" if cs.regime_score >= 60 else "⚠️"
        r_col  = "#00d4a0" if cs.regime_score >= 60 else "#f59e0b"
        components_html += (
            f"<span style='color:{r_col}'>{r_icon} Regime {cs.regime_score:.0f}</span>&nbsp;&nbsp;"
        )

    # Timing
    if cs.timing_score is not None:
        t_icon = "✅" if cs.timing_score >= 70 else "⚠️" if cs.timing_score >= 40 else "❌"
        t_col  = "#00d4a0" if cs.timing_score >= 70 else "#f59e0b" if cs.timing_score >= 40 else "#ff4b4b"
        components_html += (
            f"<span style='color:{t_col}'>{t_icon} Timing {cs.timing_score:.0f}</span>&nbsp;&nbsp;"
        )

    # Personal edge
    if cs.personal_edge_score is not None:
        pe_icon = "✅" if cs.personal_edge_score >= 60 else "⚠️"
        pe_col  = "#00d4a0" if cs.personal_edge_score >= 60 else "#f59e0b"
        components_html += (
            f"<span style='color:{pe_col}'>{pe_icon} Edge {cs.personal_edge_score:.0f}</span>&nbsp;&nbsp;"
        )

    # Multiplier badges
    mults_html = ""
    if cs.earnings_mult < 1.0:
        mults_html += (
            f"<span style='color:#f59e0b'>"
            f"📅 Earnings ×{cs.earnings_mult:.2f}</span>&nbsp;&nbsp;"
        )
    if cs.whipsaw_mult < 1.0:
        mults_html += (
            f"<span style='color:#fb923c'>"
            f"🌀 Whipsaw ×{cs.whipsaw_mult:.2f}</span>&nbsp;&nbsp;"
        )
    if cs.breakout_memory_mult < 1.0:
        mults_html += (
            f"<span style='color:#a78bfa'>"
            f"⚡ Breakout hist ×{cs.breakout_memory_mult:.2f}</span>&nbsp;&nbsp;"
        )

    # ── Boosters ──────────────────────────────────────────────────────
    boosters_html = ""
    for b in cs.boosters:
        boosters_html += (
            f"<div style='color:#00d4a0;font-size:.72rem'>▲ {b}</div>"
        )

    # ── Detractors ────────────────────────────────────────────────────
    detractors_html = ""
    for d in cs.detractors:
        detractors_html += (
            f"<div style='color:#f59e0b;font-size:.72rem'>▼ {d}</div>"
        )

    # ── Blockers ──────────────────────────────────────────────────────
    blockers_html = ""
    for bl in cs.blockers:
        blockers_html += (
            f"<div style='color:#ff4b4b;font-size:.72rem'>🚫 {bl}</div>"
        )

    # ── Assemble card ─────────────────────────────────────────────────
    st.markdown(
        f"<div style='"
        f"background:linear-gradient(135deg,#0d1117 0%,#111827 100%);"
        f"border:1.5px solid {border_col};"
        f"border-radius:14px;"
        f"padding:18px 22px;"
        f"box-shadow:{glow};"
        f"margin-bottom:14px"
        f"'>"

        # ── Header row ─────────────────────────────────────────────────
        f"<div style='display:flex;justify-content:space-between;align-items:flex-start;"
        f"margin-bottom:12px'>"

        f"<div>"
        f"<div style='font-size:.55rem;color:#4a5568;text-transform:uppercase;"
        f"letter-spacing:.14em;margin-bottom:2px'>Composite Score</div>"
        f"<div style='font-family:JetBrains Mono,monospace;font-size:2rem;"
        f"font-weight:900;color:{border_col};line-height:1'>"
        f"{score:.0f}<span style='font-size:.9rem;color:#4a5568'>/100</span>"
        f"</div>"
        f"<div style='color:#8892a4;font-size:.7rem;margin-top:3px'>"
        f"{conviction} &nbsp;·&nbsp; {cs.confidence} confidence"
        f"</div>"
        f"</div>"

        f"<div style='text-align:right'>"
        f"<div style='background:{badge_bg};color:{action_col};"
        f"font-size:1rem;font-weight:900;padding:6px 16px;"
        f"border-radius:8px;border:1.5px solid {border_col}44;"
        f"font-family:JetBrains Mono,monospace;letter-spacing:.12em'>"
        f"{action}"
        f"</div>"
        f"<div style='font-family:JetBrains Mono,monospace;font-size:.65rem;"
        f"color:{border_col}55;margin-top:6px;letter-spacing:.04em'>"
        f"{dots}"
        f"</div>"
        f"</div>"

        f"</div>"

        # ── Component scores row ───────────────────────────────────────
        f"<div style='font-size:.72rem;margin-bottom:8px;line-height:1.9'>"
        f"{components_html}"
        f"</div>"

        # ── Multipliers row ────────────────────────────────────────────
        + (
            f"<div style='font-size:.72rem;margin-bottom:8px;line-height:1.9'>"
            f"{mults_html}"
            f"</div>"
            if mults_html else ""
        ) +

        # ── Boosters / detractors / blockers ──────────────────────────
        + (
            f"<div style='margin:8px 0 4px 0'>{boosters_html}{detractors_html}{blockers_html}</div>"
            if (cs.boosters or cs.detractors or cs.blockers) else ""
        ) +

        # ── Verdict footer ────────────────────────────────────────────
        f"<div style='background:#0d1117;border-radius:8px;padding:10px 14px;"
        f"margin-top:10px;border-left:3px solid {border_col}'>"
        f"<div style='font-size:.8rem;color:#c9d1e0;font-weight:600'>"
        f"{cs.verdict_line}"
        f"</div>"
        f"<div style='font-size:.72rem;color:#8892a4;margin-top:3px'>"
        f"{cs.risk_line}"
        f"</div>"
        f"</div>"

        f"</div>",
        unsafe_allow_html=True,
    )
