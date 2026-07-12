"""
🤖 Autopilot page — arm, limits, ledger, activity. Safety-first UI:
PAPER default, LIVE arming needs the exact phrase, kill switch on top.
"""
from __future__ import annotations

import streamlit as st

_CARD = ("background:#0d1421;border:1px solid #1e293b;border-radius:10px;"
         "padding:14px 18px;margin-bottom:10px")


def render_autopilot() -> None:
    from execution.autopilot import (get_status, set_config, arm, disarm,
                                     ARM_PHRASE)

    st.markdown("### 🤖 Autopilot")
    st.caption("System khud trade karega — tumhare set kiye limits ke andar, "
               "sniper/scanner ke signals pe, sirf strong sectors mein, "
               "har trade GTT-protected, +3% target. "
               "**PAPER mode se shuru karo — kam se kam ek hafta.**")

    s = get_status()

    # ── Status / kill switch ─────────────────────────────────────────────────
    armed = s["armed"]
    mode = s["mode"]
    a_col = "#00d4a0" if (armed and mode == "PAPER") else \
            ("#ff4b4b" if (armed and mode == "LIVE") else "#8892a4")
    a_txt = (f"ARMED — {mode}" if armed else
             f"OFF{(' · ' + s['disarmed_reason']) if s['disarmed_reason'] else ''}")
    st.markdown(
        f"<div style='{_CARD};border-left:4px solid {a_col}'>"
        f"<div style='display:flex;justify-content:space-between;flex-wrap:wrap'>"
        f"<span style='font-size:1.05rem;font-weight:800;color:{a_col}'>"
        f"● {a_txt}</span>"
        f"<span style='font-size:.8rem;color:#8892a4'>entries "
        f"{s['start_time']}–{s['end_time']} · max {s['max_trades_per_day']}/day · "
        f"target +{s['target_pct']}%</span></div>"
        f"</div>", unsafe_allow_html=True)

    # ── Money ledger ─────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pool (compounded)", f"₹{s['pool']:,.0f}",
              delta=f"₹{s['realized_pnl']:+,.0f} realized" if s["realized_pnl"] else None)
    m2.metric("Deployed", f"₹{s['deployed']:,.0f}")
    m3.metric("Available", f"₹{s['available']:,.0f}",
              help=f"Reserve {s['cash_reserve_pct']*100:.0f}% kabhi deploy nahi hota")
    m4.metric("Trades today", f"{s['trades_today_count']}/{s['max_trades_per_day']}")

    # ── Arm / Disarm ─────────────────────────────────────────────────────────
    c1, c2 = st.columns([1.2, 2.8])
    with c1:
        if armed:
            if st.button("⛔ DISARM (kill switch)", type="primary",
                         width="stretch", key="ap_disarm"):
                disarm("user")
                st.rerun()
        else:
            if mode == "LIVE":
                phrase = st.text_input(
                    f"LIVE arm karne ke liye type karo: {ARM_PHRASE}",
                    key="ap_phrase")
                if st.button("🔴 ARM — LIVE (real paisa)", width="stretch",
                             key="ap_arm_live"):
                    ok, msg = arm(phrase)
                    (st.success if ok else st.error)(msg)
                    if ok:
                        st.rerun()
            else:
                if st.button("🟢 ARM — Paper mode", type="primary",
                             width="stretch", key="ap_arm_paper"):
                    ok, msg = arm()
                    (st.success if ok else st.error)(msg)
                    if ok:
                        st.rerun()
    with c2:
        st.caption("**Gates (har trade se pehle, isi order mein):** valid "
                   "stop zaroori → armed → 9:30 ke baad → daily limit → "
                   "position limit → symbol ek-baar/din → score+edge → "
                   "sector top-N positive mein → market regime theek → "
                   "capital available (reserve untouched) → LIVE mein broker "
                   "cash double-check. Din ka loss limit cross → khud DISARM. "
                   "Profit mein: stop khud breakeven pe (free trade). "
                   "LIVE fills exchange se reconcile hote hain, estimate nahi.")

    # ── Limits (tumhari safety, tumhare numbers) ──────────────────────────────
    with st.expander("⚙️ Limits — apne numbers set karo", expanded=not s["allocation"]):
        l1, l2, l3 = st.columns(3)
        with l1:
            alloc = st.number_input("Allocation X (₹)", min_value=0.0,
                                    value=float(s["allocation"]), step=5000.0,
                                    key="ap_alloc",
                                    help="Zerodha ka jitna paisa autopilot ko dena hai")
            mode_pick = st.radio("Mode", ["PAPER", "LIVE"],
                                 index=0 if mode == "PAPER" else 1,
                                 horizontal=True, key="ap_mode",
                                 help="Mode badalte hi autopilot DISARM ho jata hai")
            reserve = st.slider("Cash reserve % (kabhi deploy nahi)", 5, 50,
                                int(s["cash_reserve_pct"] * 100), key="ap_res")
        with l2:
            per_trade = st.slider("Per-trade cap (% of pool)", 5, 50,
                                  int(s["per_trade_cap_pct"] * 100), key="ap_cap")
            max_pos = st.number_input("Max open positions", 1, 10,
                                      int(s["max_open_positions"]), key="ap_pos")
            max_day = st.number_input("Max trades / day", 1, 15,
                                      int(s["max_trades_per_day"]), key="ap_day")
        with l3:
            day_loss = st.slider("Daily loss circuit-breaker (%)", 1, 10,
                                 int(s["daily_loss_limit_pct"] * 100), key="ap_cb")
            min_score = st.slider("Min signal score", 40, 95,
                                  int(s["min_score"]), key="ap_score")
            top_n = st.number_input("Top-N strong sectors", 1, 8,
                                    int(s["sector_top_n"]), key="ap_topn")
            start_t = st.text_input("Entries start (HH:MM)", s["start_time"],
                                    key="ap_start",
                                    help="Market khulne ke baad settle hone do — 09:30 se pehle mat karo")
            end_t = st.text_input("Entries end (HH:MM)", s["end_time"],
                                  key="ap_end",
                                  help="Iske baad naye entries nahi (exits GTT se chalte rahenge)")
        t1, t2, t3 = st.columns(3)
        with t1:
            trail_on = st.toggle(
                "🛡 Breakeven trail", value=bool(s.get("trailing_enabled", True)),
                key="ap_trail",
                help="Profit trigger pe stop entry pe chala jata hai — "
                     "worst case scratch, upside khula")
        with t2:
            trail_trig = st.slider(
                "Trail trigger (+% profit)", 0.5, 8.0,
                float(s.get("breakeven_trigger_pct", 2.0)), 0.5,
                key="ap_trailtrig")
        with t3:
            regime_on = st.toggle(
                "📡 Regime gate", value=bool(s.get("regime_gate", True)),
                key="ap_regime",
                help="DISTRIBUTION / BEAR tape mein naye entries band — "
                     "breakouts wahan sabse zyada fail hote hain")
        if st.button("💾 Save limits", key="ap_save", width="stretch"):
            set_config(allocation=alloc, mode=mode_pick,
                       cash_reserve_pct=reserve / 100,
                       per_trade_cap_pct=per_trade / 100,
                       max_open_positions=int(max_pos),
                       max_trades_per_day=int(max_day),
                       daily_loss_limit_pct=day_loss / 100,
                       min_score=float(min_score),
                       sector_top_n=int(top_n),
                       start_time=start_t.strip() or "09:30",
                       end_time=end_t.strip() or "14:45",
                       trailing_enabled=trail_on,
                       breakeven_trigger_pct=float(trail_trig),
                       regime_gate=regime_on)
            st.success("Limits saved. (Mode change hua ho toh dobara ARM karna hoga.)")
            st.rerun()

    # ── Open autopilot positions ──────────────────────────────────────────────
    if s["open_trades"]:
        st.markdown("#### 📍 Autopilot positions")
        for t in s["open_trades"]:
            st.markdown(
                f"<div style='{_CARD}'>"
                f"<span style='color:#e2e8f0;font-weight:700;"
                f"font-family:JetBrains Mono,monospace'>{t['symbol']}</span>"
                f"<span style='font-size:.75rem;color:#8892a4'> ({t['mode']}) · "
                f"{t['qty']} sh @ ₹{float(t['entry_price'] or 0):,.1f} · "
                f"stop ₹{float(t['stop_price'] or 0):,.1f} · "
                f"target ₹{float(t['target_price'] or 0):,.1f} · "
                f"{t['status']}</span></div>", unsafe_allow_html=True)

    # ── Report Card — autopilot ka apna track record ──────────────────────────
    _render_report_card()

    # ── Activity log ──────────────────────────────────────────────────────────
    st.markdown("#### 📜 Activity")
    acts = s.get("activity", [])
    if acts:
        st.markdown(
            f"<div style='{_CARD};font-size:.78rem;color:#c9d1d9;"
            f"font-family:JetBrains Mono,monospace'>"
            + "<br>".join(acts[:20]) + "</div>", unsafe_allow_html=True)
    else:
        st.caption("Abhi koi activity nahi — allocation set karke ARM karo "
                   "(PAPER se shuru!), phir har decision yahan dikhega.")


def _render_report_card() -> None:
    """Autopilot ke closed trades ki equity curve + stats + LIVE-readiness
    verdict. Manual trades isme nahi aate — machine apne record pe judge ho."""
    from execution.autopilot import report_card

    st.markdown("#### 📊 Report Card — kya autopilot paisa deserve karta hai?")
    rc = report_card()
    stats, trades = rc["stats"], rc["trades"]

    v_style = {
        "COLLECTING_EVIDENCE": ("#f59e0b", "🟡 EVIDENCE JAMA HO RAHA HAI"),
        "READY_CANDIDATE":     ("#00d4a0", "🟢 LIVE CANDIDATE"),
        "NOT_READY":           ("#ff4b4b", "🔴 NOT READY"),
    }
    v_col, v_label = v_style.get(rc["verdict"], ("#8892a4", rc["verdict"]))
    st.markdown(
        f"<div style='{_CARD};border-left:4px solid {v_col}'>"
        f"<span style='font-weight:800;color:{v_col}'>{v_label}</span> "
        f"<span style='font-size:.8rem;color:#c9d1d9'>· {rc['verdict_reason']}"
        f"</span></div>", unsafe_allow_html=True)

    if not trades:
        st.caption("Closed trades abhi zero hain — curve pehle trade ke "
                   "close hone pe shuru hogi.")
        return

    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Closed trades", f"{stats['n']}",
              delta=f"{stats['paper_n']} paper · {stats['live_n']} live",
              delta_color="off")
    r2.metric("Win rate", f"{stats['win_rate']:.0f}%",
              delta=f"{stats['wins']}W / {stats['losses']}L", delta_color="off")
    r3.metric("Total P&L", f"₹{stats['total_pnl']:+,.0f}")
    r4.metric("Expectancy", f"{stats['expectancy_r']:+.2f}R",
              help="Average R-multiple per trade — positive hona zaroori")
    r5.metric("Profit factor", f"{stats['profit_factor']:.2f}",
              delta=f"max DD ₹{stats['max_drawdown']:,.0f}", delta_color="off")

    try:
        import plotly.graph_objects as go
        eq = [0.0] + [t["equity"] for t in trades]
        fig = go.Figure(go.Scatter(
            y=eq, mode="lines+markers", line=dict(color=v_col, width=2),
            marker=dict(size=5), hovertemplate="₹%{y:,.0f}<extra></extra>"))
        fig.update_layout(
            height=220, margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Trade #", color="#8892a4", gridcolor="#1e293b"),
            yaxis=dict(title="Cumulative P&L (₹)", color="#8892a4",
                       gridcolor="#1e293b"))
        st.plotly_chart(fig, width="stretch",
                        config={"displayModeBar": False})
    except Exception:
        pass

    with st.expander(f"📒 Trade ledger ({stats['n']} closed)"):
        for t in reversed(trades):        # newest first
            p_col = "#00d4a0" if t["win"] else "#ff4b4b"
            st.markdown(
                f"<div style='font-size:.78rem;color:#c9d1d9;"
                f"font-family:JetBrains Mono,monospace;padding:2px 0'>"
                f"{t['date']} · <b>{t['symbol']}</b> ({t['mode']}·{t['source']}) "
                f"· {t['qty']} sh ₹{t['entry']:,.1f}→₹{t['exit']:,.1f} · "
                f"<span style='color:{p_col}'>₹{t['pnl']:+,.0f} "
                f"({t['r']:+.1f}R)</span></div>", unsafe_allow_html=True)
