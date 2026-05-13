"""
Market Narrative Engine — generates structured daily market intelligence summaries.

NOT a chatbot. NOT motivational text.
Produces compressed institutional-grade market state summaries.

Output format mirrors a prop-desk morning brief:
  - Market State block
  - Regime shift detection
  - Best playbooks for today
  - Sectors to watch/avoid
  - Breakout environment signal
  - AI-enhanced narrative (DeepSeek, optional, cached)
"""
from __future__ import annotations

import streamlit as st


@st.cache_data(ttl=900, show_spinner=False)
def _get_full_narrative() -> dict:
    """Build complete narrative data. Cached 15 min."""
    out: dict = {}

    # ── Regime ────────────────────────────────────────────────────────────────
    try:
        from core.regime_engine import compute_regime
        r = compute_regime()
        out["regime"] = {
            "market":      r.market_regime,
            "volatility":  r.volatility_regime,
            "breadth":     r.breadth_label,
            "breadth_score": r.breadth_strength,
            "risk_mode":   r.risk_mode,
            "inst_activity": r.institutional_activity,
            "leaders":     r.leading_sectors,
            "laggards":    r.lagging_sectors,
            "rotation":    r.rotation_mode,
            "breakout_env": r.breakout_environment,
            "playbooks":   r.recommended_playbooks,
            "avoid":       r.avoid_patterns,
            "regime_score": r.regime_score,
            "quality_mult": r.quality_multiplier,
            "nifty":       r.nifty_price,
            "nifty_1d":    r.nifty_change_1d,
            "nifty_5d":    r.nifty_change_5d,
            "vix":         r.vix,
            "vix_state":   r.volatility_regime,
            "sector_returns": r.sector_returns,
            "timestamp":   r.timestamp,
        }
    except Exception:
        out["regime"] = None

    # ── Breakout Health ───────────────────────────────────────────────────────
    try:
        from scan.breakout_health import get_health
        bh = get_health()
        out["breakout_health"] = {
            "success_rate":   bh.success_rate,
            "failure_rate":   bh.failure_rate,
            "avg_extension":  bh.avg_extension_pct,
            "score":          bh.follow_through_score,
            "environment":    bh.environment,
            "signal":         bh.signal,
            "sample_size":    bh.sample_size,
            "recent_fail":    bh.recent_failures[:3],
            "recent_win":     bh.recent_successes[:3],
        }
    except Exception:
        out["breakout_health"] = None

    return out


def render_market_narrative() -> None:
    """
    Renders the full institutional market narrative.
    Compact, dense, professional — no chat bubbles.
    """
    data = _get_full_narrative()
    r    = data.get("regime")
    bh   = data.get("breakout_health")

    # ── Top strip: regime state ───────────────────────────────────────────────
    _render_regime_block(r)
    st.divider()

    col_l, col_r = st.columns([3, 2])

    with col_l:
        _render_market_state(r, bh)
        st.divider()
        _render_best_playbooks_today(r)

    with col_r:
        _render_sector_rotation(r)
        st.divider()
        _render_breakout_health_block(bh)

    st.divider()
    _render_deepseek_narrative(r, bh)


def _render_regime_block(r: dict | None) -> None:
    if not r:
        st.warning("Regime data unavailable — check yfinance connectivity.")
        return

    regime_colors = {
        "TRENDING_BULL":  ("#00d4a0", "#001a12"),
        "EXPANSION":      ("#00d4ff", "#001520"),
        "CHOPPY":         ("#f59e0b", "#1a1200"),
        "COMPRESSION":    ("#60a5fa", "#001020"),
        "DISTRIBUTION":   ("#fb923c", "#1a0800"),
        "TRENDING_BEAR":  ("#ff4b4b", "#1a0000"),
    }
    fg, bg = regime_colors.get(r["market"], ("#8892a4", "#111827"))

    nifty_chg = r["nifty_1d"]
    chg_col   = "#00d4a0" if nifty_chg >= 0 else "#ff4b4b"
    chg_arrow = "▲" if nifty_chg >= 0 else "▼"
    breadth_col = {"STRONG": "#00d4a0", "NEUTRAL": "#f59e0b", "WEAK": "#ff4b4b"}.get(r["breadth"], "#8892a4")
    vix_col     = {"LOW_VOL_COMPRESSION": "#00d4a0", "NORMAL": "#f59e0b",
                   "TREND_VOLATILITY": "#fb923c", "ELEVATED": "#ff4b4b", "PANIC": "#ff0000"}.get(r["vix_state"], "#8892a4")
    bk_col      = {"FAVORABLE": "#00d4a0", "NEUTRAL": "#f59e0b", "UNFAVORABLE": "#ff4b4b"}.get(r["breakout_env"], "#8892a4")
    rm_col      = {"RISK_ON": "#00d4a0", "NEUTRAL": "#f59e0b", "RISK_OFF": "#ff4b4b"}.get(r["risk_mode"], "#8892a4")

    score_bar = int(r.get("regime_score", 50))

    st.markdown(
        f"""<div style='background:{bg};border:1px solid {fg}33;border-radius:12px;
                padding:12px 18px;font-family:JetBrains Mono,monospace'>
          <div style='display:flex;align-items:center;gap:6px;margin-bottom:8px'>
            <span style='font-size:.62rem;color:#4a5568;letter-spacing:.12em;text-transform:uppercase'>
              Market Intelligence Brief</span>
            <span style='font-size:.58rem;color:#4a5568;margin-left:auto'>
              Updated {r.get('timestamp','—')}</span>
          </div>
          <div style='display:flex;gap:24px;flex-wrap:wrap;align-items:center'>

            <div>
              <div style='font-size:.56rem;color:#4a5568;text-transform:uppercase;letter-spacing:.1em'>Regime</div>
              <div style='font-size:1.1rem;color:{fg};font-weight:800;letter-spacing:.04em'>
                {r["market"].replace("_"," ")}</div>
              <div style='height:4px;width:100px;background:#1e293b;border-radius:2px;margin-top:3px'>
                <div style='height:4px;width:{score_bar}px;background:{fg};border-radius:2px'></div>
              </div>
            </div>

            <div>
              <div style='font-size:.56rem;color:#4a5568;text-transform:uppercase;letter-spacing:.1em'>Nifty 50</div>
              <div style='font-size:.95rem;color:#e8eaf0;font-weight:700'>
                {r["nifty"]:,.0f}
                <span style='font-size:.75rem;color:{chg_col}'> {chg_arrow}{abs(nifty_chg):.2f}%</span>
              </div>
              <div style='font-size:.6rem;color:#4a5568'>5d: {r["nifty_5d"]:+.2f}%</div>
            </div>

            <div>
              <div style='font-size:.56rem;color:#4a5568;text-transform:uppercase;letter-spacing:.1em'>VIX</div>
              <div style='font-size:.95rem;color:{vix_col};font-weight:700'>{r["vix"]:.1f}</div>
              <div style='font-size:.6rem;color:{vix_col}'>{r["vix_state"].replace("_"," ")}</div>
            </div>

            <div>
              <div style='font-size:.56rem;color:#4a5568;text-transform:uppercase;letter-spacing:.1em'>Breadth</div>
              <div style='font-size:.95rem;color:{breadth_col};font-weight:700'>{r["breadth"]}</div>
              <div style='font-size:.6rem;color:{breadth_col}'>{r["breadth_score"]:.0f}/100</div>
            </div>

            <div>
              <div style='font-size:.56rem;color:#4a5568;text-transform:uppercase;letter-spacing:.1em'>Breakout Env</div>
              <div style='font-size:.95rem;color:{bk_col};font-weight:700'>{r["breakout_env"]}</div>
            </div>

            <div>
              <div style='font-size:.56rem;color:#4a5568;text-transform:uppercase;letter-spacing:.1em'>Risk Mode</div>
              <div style='font-size:.95rem;color:{rm_col};font-weight:700'>{r["risk_mode"].replace("_"," ")}</div>
            </div>

            <div>
              <div style='font-size:.56rem;color:#4a5568;text-transform:uppercase;letter-spacing:.1em'>Institutional</div>
              <div style='font-size:.82rem;color:#8892a4;font-weight:600'>{r["inst_activity"]}</div>
            </div>

            <div style='margin-left:auto;text-align:right'>
              <div style='font-size:.56rem;color:#4a5568;text-transform:uppercase;letter-spacing:.1em'>Setup Multiplier</div>
              <div style='font-size:1.2rem;color:{fg};font-weight:800'>×{r["quality_mult"]:.2f}</div>
            </div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )


def _render_market_state(r: dict | None, bh: dict | None) -> None:
    st.markdown(
        "<span style='font-size:.65rem;color:#8892a4;text-transform:uppercase;"
        "letter-spacing:.1em;font-family:JetBrains Mono,monospace'>MARKET STATE</span>",
        unsafe_allow_html=True,
    )
    if not r:
        st.caption("Regime unavailable")
        return

    items = []
    items.append(f"{'✅' if r['breadth'] == 'STRONG' else '⚠️' if r['breadth'] == 'NEUTRAL' else '🔴'} "
                 f"Breadth: **{r['breadth']}** ({r['breadth_score']:.0f}/100)")
    items.append(f"{'✅' if r['inst_activity'] == 'ACCUMULATION' else '⚠️'} "
                 f"Institutional: **{r['inst_activity']}**")
    items.append(f"{'✅' if r['rotation'] == 'OFFENSIVE' else '🔄'} "
                 f"Rotation: **{r['rotation']}**")
    if r["leaders"]:
        items.append(f"🏆 Leaders: **{', '.join(r['leaders'][:3])}**")
    if bh:
        env_icon = "✅" if bh["environment"] == "FAVORABLE" else "⚠️" if bh["environment"] == "NEUTRAL" else "🔴"
        items.append(f"{env_icon} Breakout env: **{bh['environment']}** "
                     f"({bh['success_rate']*100:.0f}% follow-through)")

    for item in items:
        st.markdown(f"<div style='font-size:.78rem;padding:3px 0;color:#c9d1e0'>{item}</div>",
                    unsafe_allow_html=True)


def _render_best_playbooks_today(r: dict | None) -> None:
    st.markdown(
        "<span style='font-size:.65rem;color:#00d4a0;text-transform:uppercase;"
        "letter-spacing:.1em;font-family:JetBrains Mono,monospace'>BEST PLAYBOOKS TODAY</span>",
        unsafe_allow_html=True,
    )
    if not r:
        return

    try:
        from playbooks import get_playbooks_for_regime, REGISTRY
        pbs = get_playbooks_for_regime(
            r["market"], r.get("volatility", "NORMAL"), r["breadth"]
        )[:4]

        for i, pb in enumerate(pbs, 1):
            ev_str   = f"{pb.baseline_expectancy*100:+.1f}%"
            wr_str   = f"{pb.baseline_win_rate*100:.0f}%"
            st.markdown(
                f"<div style='background:#111827;border:1px solid rgba(0,212,160,.15);"
                f"border-radius:8px;padding:8px 12px;margin-bottom:5px'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<span style='color:#00d4a0;font-size:.75rem;font-weight:700'>"
                f"{pb.emoji} {i}. {pb.name}</span>"
                f"<span style='font-size:.6rem;color:#4a5568'>{pb.category}</span>"
                f"</div>"
                f"<div style='font-size:.65rem;color:#8892a4;margin-top:3px'>"
                f"EV {ev_str} · WR {wr_str} · R:R {pb.baseline_risk_reward:.1f}× · "
                f"Hold ~{pb.baseline_avg_hold_days}d</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    except Exception:
        st.caption("Playbook data unavailable")

    # Avoid list
    if r.get("avoid"):
        st.markdown(
            "<div style='margin-top:6px'><span style='font-size:.62rem;color:#ff4b4b;text-transform:uppercase;"
            "letter-spacing:.08em'>AVOID TODAY: </span>"
            f"<span style='font-size:.67rem;color:#8892a4'>"
            f"{' · '.join(a.replace('_',' ') for a in r['avoid'][:3])}"
            f"</span></div>",
            unsafe_allow_html=True,
        )


def _render_sector_rotation(r: dict | None) -> None:
    st.markdown(
        "<span style='font-size:.65rem;color:#00d4ff;text-transform:uppercase;"
        "letter-spacing:.1em;font-family:JetBrains Mono,monospace'>SECTOR ROTATION</span>",
        unsafe_allow_html=True,
    )
    if not r:
        return

    sector_rets = r.get("sector_returns", {})
    if not sector_rets:
        st.caption("Sector data unavailable")
        return

    sorted_secs = sorted(sector_rets.items(), key=lambda x: x[1], reverse=True)
    for sec, ret in sorted_secs:
        col  = "#00d4a0" if ret > 0 else "#ff4b4b"
        bar  = min(abs(ret) / 5 * 60, 60)  # max 60px for 5% move
        is_leader = sec in r.get("leaders", [])
        badge = " 🏆" if is_leader else ""
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;padding:3px 0;"
            f"border-bottom:1px solid rgba(255,255,255,.03)'>"
            f"<span style='font-size:.68rem;color:#c9d1e0;width:70px'>{sec}{badge}</span>"
            f"<div style='height:4px;width:{bar:.0f}px;background:{col};border-radius:2px'></div>"
            f"<span style='font-size:.68rem;color:{col};font-weight:700'>{ret:+.2f}%</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_breakout_health_block(bh: dict | None) -> None:
    st.markdown(
        "<span style='font-size:.65rem;color:#f59e0b;text-transform:uppercase;"
        "letter-spacing:.1em;font-family:JetBrains Mono,monospace'>BREAKOUT HEALTH</span>",
        unsafe_allow_html=True,
    )
    if not bh:
        st.caption("Breakout health unavailable")
        return

    sig_col = {"BUY_BREAKOUTS": "#00d4a0", "WAIT": "#f59e0b", "AVOID_BREAKOUTS": "#ff4b4b"}.get(bh["signal"], "#8892a4")
    env_col = {"FAVORABLE": "#00d4a0", "NEUTRAL": "#f59e0b", "UNFAVORABLE": "#ff4b4b"}.get(bh["environment"], "#8892a4")

    st.markdown(
        f"<div style='background:#111827;border:1px solid rgba(245,158,11,.2);"
        f"border-radius:8px;padding:10px 12px'>"
        f"<div style='display:flex;justify-content:space-between;margin-bottom:6px'>"
        f"<span style='color:{sig_col};font-size:.82rem;font-weight:700'>"
        f"Signal: {bh['signal'].replace('_',' ')}</span>"
        f"<span style='color:{env_col};font-size:.72rem'>{bh['environment']}</span>"
        f"</div>"
        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:6px'>"
        f"<div><span style='font-size:.58rem;color:#4a5568'>Success Rate</span><br>"
        f"<span style='color:#00d4a0;font-weight:700'>{bh['success_rate']*100:.0f}%</span></div>"
        f"<div><span style='font-size:.58rem;color:#4a5568'>Failure Rate</span><br>"
        f"<span style='color:#ff4b4b;font-weight:700'>{bh['failure_rate']*100:.0f}%</span></div>"
        f"<div><span style='font-size:.58rem;color:#4a5568'>Avg Extension</span><br>"
        f"<span style='color:#f59e0b;font-weight:700'>{bh['avg_extension']:.1f}%</span></div>"
        f"<div><span style='font-size:.58rem;color:#4a5568'>Follow-Through</span><br>"
        f"<span style='color:#00d4ff;font-weight:700'>{bh['score']:.0f}/100</span></div>"
        f"</div>"
        f"<div style='font-size:.6rem;color:#4a5568;margin-top:6px'>"
        f"Sample: {bh['sample_size']} recent breakouts</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if bh.get("recent_fail"):
        st.markdown(
            f"<div style='font-size:.62rem;color:#ff4b4b;margin-top:4px'>"
            f"Recent fails: {', '.join(bh['recent_fail'])}</div>",
            unsafe_allow_html=True,
        )
    if bh.get("recent_win"):
        st.markdown(
            f"<div style='font-size:.62rem;color:#00d4a0;margin-top:2px'>"
            f"Recent winners: {', '.join(bh['recent_win'])}</div>",
            unsafe_allow_html=True,
        )


def _render_deepseek_narrative(r: dict | None, bh: dict | None) -> None:
    st.markdown(
        "<span style='font-size:.65rem;color:#a78bfa;text-transform:uppercase;"
        "letter-spacing:.1em;font-family:JetBrains Mono,monospace'>AI MARKET BRIEF</span>",
        unsafe_allow_html=True,
    )

    if st.button("📊 Generate Market Brief", key="narrative_gen", use_container_width=False):
        with st.spinner("Generating institutional brief…"):
            brief = _generate_brief(r, bh)
        st.session_state["narrative_brief"] = brief

    brief = st.session_state.get("narrative_brief")
    if brief:
        # Parse structured sections for clean display
        st.markdown(
            f"<div style='background:#111827;border:1px solid rgba(167,139,250,.2);"
            f"border-radius:10px;padding:12px 16px;font-size:.75rem;color:#c9d1e0;"
            f"line-height:1.7;white-space:pre-wrap;font-family:JetBrains Mono,monospace'>"
            f"{brief}</div>",
            unsafe_allow_html=True,
        )


@st.cache_data(ttl=3600, show_spinner=False)
def _generate_brief(regime_dict, bh_dict) -> str:
    import os, requests
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        return "⚠️ DeepSeek API key not configured."

    if not regime_dict:
        return "⚠️ Regime data unavailable. Check yfinance connectivity."

    r = regime_dict
    bh_line = ""
    if bh_dict:
        bh_line = (f"\nBreakout Health: {bh_dict['environment']} "
                   f"({bh_dict['success_rate']*100:.0f}% success, "
                   f"{bh_dict['failure_rate']*100:.0f}% failure, "
                   f"n={bh_dict['sample_size']})")

    prompt = (
        f"Generate a concise institutional market brief for NSE India. "
        f"Use structured format with headers. No fluff, no motivational text.\n\n"
        f"REGIME DATA:\n"
        f"Market: {r['market']}\n"
        f"Volatility: {r['vix_state']} (VIX {r['vix']:.1f})\n"
        f"Breadth: {r['breadth']} ({r['breadth_score']:.0f}/100)\n"
        f"Risk Mode: {r['risk_mode']}\n"
        f"Institutional: {r['inst_activity']}\n"
        f"Leaders: {', '.join(r['leaders'])}\n"
        f"Laggards: {', '.join(r['laggards'])}\n"
        f"Rotation: {r['rotation']}\n"
        f"Nifty: {r['nifty']:,.0f} ({r['nifty_1d']:+.2f}% today, {r['nifty_5d']:+.2f}% 5d){bh_line}\n\n"
        f"Recommended playbooks: {', '.join(r['playbooks'])}\n"
        f"Avoid: {', '.join(r['avoid'])}\n\n"
        f"Structure the response as:\n"
        f"MARKET STATE\n[3-4 bullets on current conditions]\n\n"
        f"BEST SETUPS TODAY\n[2-3 playbook recommendations with brief rationale]\n\n"
        f"AVOID\n[What NOT to trade and why]\n\n"
        f"REGIME RISK\n[1-2 sentences on what could invalidate the current regime]\n\n"
        f"Keep each section tight. Total max 200 words."
    )

    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are a senior equity strategist at an institutional fund. Output is displayed in a terminal. Be terse, precise, probabilistic. No retail commentary."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.15,
                "max_tokens": 500,
            },
            timeout=25,
        )
        data = resp.json()
        if "choices" not in data:
            return f"API error: {data.get('error', {}).get('message', 'Unknown')}"
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Brief unavailable: {e}"
