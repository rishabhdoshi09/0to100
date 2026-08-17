"""
🤖 Autopilot page — arm, limits, ledger, activity. Safety-first UI:
PAPER default, LIVE arming needs the exact phrase, kill switch on top.
"""
from __future__ import annotations

import streamlit as st

_CARD = ("background:#0d1421;border:1px solid #1e293b;border-radius:10px;"
         "padding:14px 18px;margin-bottom:10px")

_PRESET_HELP = {
    "Conservative": "Sirf A+ setups — kam trades, sabse saaf. Capital safety pehle.",
    "Balanced":     "Default — evidence-first, steady. Zyada logon ke liye sahi.",
    "Aggressive":   "Wider net — zyada shots, lekin har trade pe wahi discipline "
                    "(stop, 1% risk, regime gate). Frequency badhti hai, safety nahi girti.",
}


def _render_funnel_and_preset(mod, key_prefix: str) -> None:
    """Shared: aggressiveness dial + 'aaj kitne dekhe vs liye' funnel.
    `mod` is the autopilot module (execution.autopilot or execution.us_autopilot).
    Answers the real question — 'itne kam trades kyun?' — with data, not a shrug."""
    try:
        s = mod.get_status()
        cur_preset = s.get("preset", "Balanced")
        names = mod.presets()
        st.markdown("#### 🎛️ Aggressiveness")
        st.caption("Kitni **baar** trade le — safety rails (exchange-side stop, "
                   "1% risk, regime gate, live-price anchor) kabhi nahi badalte. "
                   "Ye sirf breadth hai: score bar, roz ke slots, chase room.")
        pick = st.radio(
            "Preset", names, index=names.index(cur_preset)
            if cur_preset in names else 1, horizontal=True,
            key=f"{key_prefix}_preset",
            label_visibility="collapsed")
        st.caption(_PRESET_HELP.get(pick, ""))
        if pick != cur_preset:
            if st.button(f"✅ Apply {pick}", key=f"{key_prefix}_preset_apply",
                         width="stretch"):
                ok, msg = mod.apply_preset(pick)
                (st.success if ok else st.error)(msg)
                if ok:
                    st.rerun()
    except Exception:
        pass

    # ── The funnel — WHY so few trades today ──────────────────────────────────
    try:
        f = mod.reject_funnel()
        considered = f.get("considered", 0)
        rejects = f.get("rejects", {})
        taken = max(0, considered - sum(rejects.values()))
        if not considered:
            st.caption("📊 Aaj abhi koi candidate evaluate nahi hua — scanner ke "
                       "setups aane par funnel yahan bharega.")
            return
        st.markdown("#### 📊 Aaj ka funnel — itne kam trades kyun?")
        st.markdown(
            f"<div style='{_CARD};font-size:.82rem;color:#c9d1d9'>"
            f"<b style='color:#e2e8f0'>{considered}</b> candidates dekhe → "
            f"<b style='color:#00d4a0'>{taken}</b> liye · "
            f"<b style='color:#f59e0b'>{sum(rejects.values())}</b> reject."
            + (("<div style='margin-top:6px'>" + "".join(
                f"<div>• {cat} — <b>{n}</b></div>"
                for cat, n in sorted(rejects.items(),
                                     key=lambda x: -x[1])) + "</div>")
               if rejects else "")
            + "</div>", unsafe_allow_html=True)
        st.caption("Ye discipline hai, bug nahi. Kam-lekin-accha (positive "
                   "expectancy) > zyada-lekin-kachra. Agar reasons mostly "
                   "'sector/score' hain aur aap zyada trades chahte ho → "
                   "Aggressive preset breadth badha dega, bina safety chhode.")
    except Exception:
        pass


def render_autopilot() -> None:
    st.markdown("### 🤖 Autopilot")
    _tab_in, _tab_us = st.tabs(["🇮🇳 India (Zerodha)", "🇺🇸 US (paper)"])
    with _tab_in:
        _render_india_autopilot()
    with _tab_us:
        _render_us_autopilot()


def _render_india_autopilot() -> None:
    from execution.autopilot import (get_status, set_config, arm, disarm,
                                     ARM_PHRASE)

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

    # ── Brain survival gate — abhi naye entries ruke hue hain? ────────────────
    if s.get("brain_gate", True):
        try:
            from execution.autopilot import _brain_posture
            _bp, _br = _brain_posture()
            if _bp == "STAND_ASIDE":
                st.markdown(
                    f"<div style='{_CARD};border-left:4px solid #ff4b4b;"
                    f"background:#ff4b4b12'>"
                    f"<b style='color:#ff4b4b'>🧠 Brain: STAND ASIDE</b> "
                    f"<span style='font-size:.82rem;color:#c9d1d9'>— naye "
                    f"entries abhi pause. {_br}</span><br>"
                    f"<span style='font-size:.72rem;color:#8892a4'>Purani "
                    f"positions + GTT exits chalte rahenge. Survival-first.</span>"
                    f"</div>", unsafe_allow_html=True)
        except Exception:
            pass

    # ── Money ledger ─────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pool (compounded)", f"₹{s['pool']:,.0f}",
              delta=f"₹{s['realized_pnl']:+,.0f} realized" if s["realized_pnl"] else None)
    m2.metric("Deployed", f"₹{s['deployed']:,.0f}")
    m3.metric("Available", f"₹{s['available']:,.0f}",
              help=f"Reserve {s['cash_reserve_pct']*100:.0f}% kabhi deploy nahi hota")
    m4.metric("Trades today", f"{s['trades_today_count']}/{s['max_trades_per_day']}")

    # ── 💰 P&L — aaj ka scoreboard, live ─────────────────────────────────────
    try:
        from execution.autopilot import pnl_snapshot
        pnl = pnl_snapshot()
    except Exception:
        pnl = None
    if pnl and (pnl["positions"] or pnl["day_closed"] or pnl["realized_total"]):
        p1, p2, p3, p4 = st.columns(4)
        _dc = "normal" if pnl["day_pnl"] >= 0 else "inverse"
        p1.metric("📅 Day P&L", f"₹{pnl['day_pnl']:+,.0f}",
                  delta=f"{pnl['day_closed']} closed today"
                        if pnl["day_closed"] else "koi close nahi aaj",
                  delta_color="off",
                  help="Aaj ke closed trades + open positions ka unrealized")
        p2.metric("📈 Unrealized (open)", f"₹{pnl['unrealized']:+,.0f}",
                  delta=f"{len(pnl['positions'])} open", delta_color="off")
        p3.metric("✅ Realized today", f"₹{pnl['day_realized']:+,.0f}")
        p4.metric("🏦 Realized total", f"₹{pnl['realized_total']:+,.0f}",
                  help="Poora compounded ledger — pool isi se badhta hai")

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
                   "LIVE price anchor (fresh quote zaroori, stop-toota/"
                   "chase reject — kal ke close pe trade KABHI nahi) → "
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
            max_sec = st.number_input("Max per sector", 1, 5,
                                      int(s.get("max_per_sector", 2)),
                                      key="ap_sec",
                                      help="Ek sector mein itni se zyada open "
                                           "positions nahi — correlation cap, "
                                           "sector bika toh sab saath na girein")
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
            brain_on = st.toggle(
                "🧠 Brain survival gate",
                value=bool(s.get("brain_gate", True)), key="ap_brain",
                help="Brain STAND_ASIDE (book DANGER, ya bad tape + negative "
                     "edge) → naye entries khud pause. Purani positions chalti "
                     "rahengi, sirf naye trade rukte hain.")
        u1, u2, u3 = st.columns(3)
        with u1:
            conv_on = st.toggle(
                "🎯 Conviction sizing", value=bool(s.get("conviction_sizing", True)),
                key="ap_conv",
                help="Risk 0.5×–1.5× scale hota hai measured score+edge se — "
                     "strong evidence pe zyada, weak pe aadha")
        with u2:
            adapt_on = st.toggle(
                "🧠 Adaptive source gate",
                value=bool(s.get("adaptive_source_gate", True)), key="ap_adapt",
                help="Scanner/sniper apne 15+ trades ke record se negative "
                     "nikla toh woh source khud pause ho jata hai")
        with u3:
            hold_days = st.number_input(
                "Time-stop (days)", 2, 30, int(s.get("max_hold_days", 15)),
                key="ap_hold",
                help="Soft ceiling. With thesis-hold ON, a healthy runner is "
                     "NOT force-closed just because days elapsed.")
        thesis_on = st.toggle(
            "📌 Thesis hold (recommended)",
            value=bool(s.get("thesis_hold", True)), key="ap_thesis",
            help="Hold while technicals + fundamentals look good. High RSI "
                 "on an OPEN trade is not a sell — tighten a GTT 2–3% below "
                 "LTP and hold until the setup actually breaks. NOT a fixed "
                 "₹1500 / 3% scalp.")
        # 🎯 Target % — used when thesis_hold is OFF; runner ceiling when ON
        _rec_txt = ""
        try:
            from scan.signal_backtest import load_report
            _bt = load_report() or {}
            _rec = _bt.get("recommended_target_pct")
            if _rec:
                _sw = (_bt.get("target_sweep") or {}).get(f"+{_rec:.0f}%", {})
                _rec_txt = (f"Backtest recommends +{_rec:.0f}% "
                            f"({_sw.get('expectancy_r', 0):+.2f}R measured, "
                            f"{_sw.get('trades', 0)} samples)")
        except Exception:
            pass
        tg1, tg2 = st.columns(2)
        with tg1:
            if thesis_on:
                runner_pct = st.slider(
                    "Runner GTT ceiling (%)", 5.0, 25.0,
                    float(s.get("runner_target_pct", 10.0)), 0.5,
                    key="ap_runner",
                    help="Wide exchange-side ceiling so a healthy runner is "
                         "not cut early. Real exit = thesis break / trail / stop.")
                rsi_protect_pct = st.slider(
                    "RSI-protect GTT (%)", 2.0, 3.0,
                    float(s.get("rsi_protect_pct", 2.5)), 0.1,
                    key="ap_rsi_protect",
                    help="If an open trade's RSI spikes, do NOT sell. "
                         "Tighten the stop this % below LTP and hold until "
                         "technicals deteriorate.")
                target_pct = float(s.get("target_pct", 3.0))
            else:
                target_pct = st.slider(
                    "Profit target (%)", 1.0, 10.0, float(s.get("target_pct", 3.0)),
                    0.5, key="ap_target",
                    help=(_rec_txt or "Scalp mode: GTT target = entry × (1+%)."))
                runner_pct = float(s.get("runner_target_pct", 10.0))
                rsi_protect_pct = float(s.get("rsi_protect_pct", 2.5))
        with tg2:
            chase_pct = st.slider(
                "Max chase (%)", 0.25, 5.0,
                float(s.get("max_chase_pct", 1.0)), 0.25, key="ap_chase",
                help="Live price signal-entry se isse zyada upar → trade "
                     "SKIP. Bhaagti train ka peecha nahi — extension pe "
                     "buy karna edge kha jata hai")
        st.caption(
            "Optional scalp booking (usually leave at 0 — thesis-hold is the default):"
        )
        book_pct = st.number_input(
            "💰 Optional profit-book AIM (% of pool, 0 = off)", 0.0, 15.0,
            float(s.get("profit_book_pct", 0.0)), 0.5, key="ap_book_pct",
            help="Optional. Default OFF. Thesis-hold exits on tech/fund break "
                 "instead of a fixed profit AIM.")
        book_rs = st.number_input(
            "💰 Absolute ₹ override (0 = use % / thesis-hold)", 0.0, 100000.0,
            float(s.get("profit_book_rupees", 0.0)), 250.0, key="ap_book",
            help="Legacy fixed-rupee AIM. When >0 this overrides % mode.")
        bf1, bf2 = st.columns(2)
        with bf1:
            book_floor_pct = st.number_input(
                "🛡️ Min floor (% of pool)", 0.0, 10.0,
                float(s.get("profit_book_min_pct", 1.5)), 0.25,
                key="ap_book_floor_pct",
                help="Pct-mode worst-case floor after trail arms.")
            book_floor = st.number_input(
                "🛡️ Min floor ₹ (absolute mode)", 0.0, 100000.0,
                float(s.get("profit_book_min_rupees", 1000.0)), 100.0,
                key="ap_book_floor",
                help="Used only when absolute ₹ override is set.")
        with bf2:
            book_give_pct = st.number_input(
                "📉 Trail give-back (% of pool)", 0.05, 5.0,
                float(s.get("profit_trail_giveback_pct", 0.6)), 0.05,
                key="ap_book_give_pct",
                help="Pct-mode: peak NET se itna % pool neeche aane par lock.")
            book_give = st.number_input(
                "📉 Trail give-back ₹ (absolute mode)", 50.0, 50000.0,
                float(s.get("profit_trail_giveback_rupees", 300.0)), 50.0,
                key="ap_book_give",
                help="Used only when absolute ₹ override is set.")
        if _rec_txt:
            st.caption(f"📊 {_rec_txt} — slider tumhara hai, evidence humara.")
        if st.button("💾 Save limits", key="ap_save", width="stretch"):
            set_config(allocation=alloc, mode=mode_pick,
                       cash_reserve_pct=reserve / 100,
                       per_trade_cap_pct=per_trade / 100,
                       max_open_positions=int(max_pos),
                       max_per_sector=int(max_sec),
                       max_trades_per_day=int(max_day),
                       daily_loss_limit_pct=day_loss / 100,
                       min_score=float(min_score),
                       sector_top_n=int(top_n),
                       start_time=start_t.strip() or "09:30",
                       end_time=end_t.strip() or "14:45",
                       trailing_enabled=trail_on,
                       breakeven_trigger_pct=float(trail_trig),
                       regime_gate=regime_on,
                       brain_gate=brain_on,
                       conviction_sizing=conv_on,
                       adaptive_source_gate=adapt_on,
                       max_hold_days=int(hold_days),
                       thesis_hold=bool(thesis_on),
                       runner_target_pct=float(runner_pct),
                       rsi_protect_pct=float(rsi_protect_pct),
                       target_pct=float(target_pct),
                       max_chase_pct=float(chase_pct),
                       profit_book_pct=float(book_pct),
                       profit_book_rupees=float(book_rs),
                       profit_book_min_pct=float(book_floor_pct),
                       profit_book_min_rupees=float(book_floor),
                       profit_trail_giveback_pct=float(book_give_pct),
                       profit_trail_giveback_rupees=float(book_give))
            st.success("Limits saved. (Mode change hua ho toh dobara ARM karna hoga.)")
            st.rerun()

    # ── Open autopilot positions — live price + P&L ke saath ─────────────────
    _pos = (pnl or {}).get("positions") or []
    if not _pos and s["open_trades"]:      # snapshot fail hua toh bhi dikhao
        _pos = [{"symbol": t["symbol"], "mode": t["mode"],
                 "qty": int(t["qty"] or 0),
                 "entry": float(t["entry_price"] or 0), "live": None,
                 "pnl": None, "pnl_pct": None,
                 "stop": float(t["stop_price"] or 0),
                 "target": float(t["target_price"] or 0),
                 "status": t["status"]} for t in s["open_trades"]]
    if _pos:
        st.markdown("#### 📍 Autopilot positions")
        for p in _pos:
            if p["pnl"] is not None:
                _pc = "#00d4a0" if p["pnl"] >= 0 else "#ff4b4b"
                _live_bit = (f" &nbsp; live <b style='color:#e2e8f0'>"
                             f"₹{p['live']:,.1f}</b> &nbsp; "
                             f"<b style='color:{_pc}'>₹{p['pnl']:+,.0f} "
                             f"({p['pnl_pct']:+.2f}%)</b>")
            else:
                _live_bit = (" &nbsp; <span style='color:#f59e0b;"
                             "font-size:.72rem'>live quote nahi — P&L "
                             "unknown (zero NAHI)</span>")
            st.markdown(
                f"<div style='{_CARD}'>"
                f"<span style='color:#e2e8f0;font-weight:700;"
                f"font-family:JetBrains Mono,monospace'>{p['symbol']}</span>"
                f"<span style='font-size:.75rem;color:#8892a4'> ({p['mode']}) · "
                f"{p['qty']} sh @ ₹{p['entry']:,.1f}</span>{_live_bit}"
                f"<div style='font-size:.72rem;color:#8892a4;margin-top:3px'>"
                f"stop ₹{p['stop']:,.1f} · target ₹{p['target']:,.1f} · "
                f"{p['status']}</div></div>", unsafe_allow_html=True)

    # ── Aggressiveness dial + rejection funnel ────────────────────────────────
    import execution.autopilot as _ap_in
    _render_funnel_and_preset(_ap_in, "ap_in")

    # ── Report Card — autopilot ka apna track record ──────────────────────────
    _render_report_card()

    # ── 🧪 Simulation Lab + capital scaling — change se PEHLE futures dekho ───
    _render_sim_lab()

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

    # 💰 Cost transparency — gross vs net, kitna slippage+charges kha gaya
    _gross = stats.get("gross_pnl", 0.0)
    _costs = stats.get("total_costs", 0.0)
    if _costs:
        _drag = (_costs / abs(_gross) * 100) if _gross else 0.0
        st.caption(
            f"💰 **Reality check** — gross ₹{_gross:+,.0f} · slippage + "
            f"charges ₹{_costs:,.0f} kha gaye ({_drag:.0f}% of gross) · "
            f"**net ₹{stats['total_pnl']:+,.0f}**. Yeh actual Zerodha costs "
            f"hain — paper bhi ab jhooth nahi bolta, LIVE jaisa hi feel.")

    try:
        from execution.autopilot import execution_quality
        eq = execution_quality()
        if eq["n_entry"] or eq["n_exit"]:
            _eff = ""
            if eq["n_entry"] >= 5:
                from execution.autopilot import get_status as _gs
                _tp = float(_gs().get("target_pct", 3.0))
                _eff = (f" · asli target ≈ +{_tp - eq['avg_entry_slip_pct'] + eq['avg_exit_slip_pct']:.2f}%"
                        f" (set +{_tp:.0f}%)")
            st.caption(
                f"⚙️ **Execution quality (LIVE fills)** — entry slip avg "
                f"{eq['avg_entry_slip_pct']:+.2f}% (worst "
                f"{eq['worst_entry_slip_pct']:+.2f}%, n={eq['n_entry']}) · "
                f"exit slip avg {eq['avg_exit_slip_pct']:+.2f}% "
                f"(n={eq['n_exit']}) · slippage cost ₹{eq['slippage_cost']:,.0f}"
                + _eff)
    except Exception:
        pass

    by_src = rc.get("by_source", {})
    if by_src:
        bits = []
        for src, d in sorted(by_src.items()):
            icon = "🔭" if src == "scanner" else "🎯"
            bits.append(f"{icon} {src}: {d['n']}t · "
                        f"{d['wins']}/{d['n']} wins · ₹{d['total_pnl']:+,.0f} "
                        f"· {d['expectancy_r']:+.2f}R")
        st.caption("**By source** — " + "  |  ".join(bits) +
                   "  (negative source 15+ trades pe khud pause)")

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


def _render_sim_lab() -> None:
    """🧪 Monte Carlo pre-flight: risk slider badalne se pehle hazaaron
    plausible futures + capital-scaling advice — sab apne REAL closed trades
    ke R-distribution se. Evidence-gated (≥30 closed); advice-only."""
    try:
        from execution.autopilot import report_card, get_status
        from core.sim_lab import compare, scaling_advice, MIN_TRADES
        rc = report_card()
        trades = rc.get("trades") or []
        stats = rc.get("stats") or {}
        rs = [float(t.get("r") or 0) for t in trades if t.get("r") is not None]
    except Exception:
        return
    with st.expander("🧪 Simulation Lab — parameter badalne se pehle"):
        if len(rs) < MIN_TRADES:
            st.caption(f"Monte Carlo ke liye {MIN_TRADES}+ closed trades "
                       f"chahiye (abhi {len(rs)}). System chalta rahe — Lab "
                       f"khud khul jayega.")
            return
        s = get_status()
        cur_risk = float(s.get("risk_per_trade_pct", 0.01))
        alt = st.slider("Compare risk/trade (%)", 0.25, 2.0,
                        round(min(2.0, max(0.25, cur_risk * 100 * 1.5)), 2),
                        0.25, key="sim_alt_risk",
                        help="Current risk vs yeh alternative — same 500 "
                             "simulated futures, sirf sizing alag")
        cmp = compare(rs, cur_risk, alt / 100)
        if cmp:
            a, b = cmp["a"], cmp["b"]
            c1, c2 = st.columns(2)
            for col, lab, d in ((c1, f"Current {cur_risk*100:.2f}%", a),
                                (c2, f"Alt {alt:.2f}%", b)):
                col.markdown(
                    f"<div style='{_CARD};font-size:.78rem;color:#c9d1d9'>"
                    f"<b style='color:#e2e8f0'>{lab}</b><br>"
                    f"median growth <b>{d['median_growth_pct']:+.1f}%</b> "
                    f"(p5 {d['p05_growth_pct']:+.1f} / p95 "
                    f"{d['p95_growth_pct']:+.1f})<br>"
                    f"max DD median {d['median_max_dd_pct']:.1f}% · worst-case "
                    f"(p95) {d['p95_max_dd_pct']:.1f}%<br>"
                    f"P(loss) {d['prob_loss_pct']:.0f}% · P(DD≥20%) "
                    f"{d['prob_dd20_pct']:.0f}%</div>", unsafe_allow_html=True)
            st.caption(f"🎲 {a['n_paths']} paths × {a['horizon']} trades, apne "
                       f"{a['n_trades_evidence']} real outcomes se bootstrap. "
                       f"**{cmp['verdict']}**")
        # capital scaling — earn it, don't wish it
        try:
            pool = float(s.get("pool") or s.get("allocation") or 1)
            dd_pct = (float(stats.get("max_drawdown", 0)) / pool * 100
                      if pool > 0 else 0.0)
            adv = scaling_advice(float(stats.get("profit_factor", 0)),
                                 dd_pct, int(stats.get("n", 0)))
            _ac = {"INCREASE": "#00d4a0", "REDUCE": "#ff4b4b",
                   "HOLD": "#8892a4"}[adv["action"]]
            _chg = (f" {adv['change_pct']:+d}%" if adv["change_pct"] else "")
            st.markdown(
                f"<div style='{_CARD};border-left:3px solid {_ac}'>"
                f"<b style='color:{_ac}'>💰 Capital scaling: {adv['action']}"
                f"{_chg}</b>"
                f" <span style='font-size:.8rem;color:#c9d1d9'>— "
                f"{adv['reason']} (Advice — allocation tum badloge.)</span>"
                f"</div>", unsafe_allow_html=True)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# 🇺🇸 US Autopilot — paper-only
# ══════════════════════════════════════════════════════════════════════════════

def _render_us_autopilot() -> None:
    from execution.us_autopilot import (get_status, set_config, arm, disarm,
                                        report_card)

    st.caption("US equities pe wahi engine + discipline (confirmed breakouts, "
               "conviction, base quality), **PAPER only** — koi US broker nahi "
               "chahiye. Entries US market hours (ET) mein, S&P benchmark, "
               "+4% target. US LIVE ke liye aage Alpaca chahiye hoga.")
    s = get_status()

    armed = s["armed"]
    a_col = "#00d4a0" if armed else "#8892a4"
    a_txt = (f"ARMED — PAPER" if armed else
             f"OFF{(' · ' + s['disarmed_reason']) if s['disarmed_reason'] else ''}")
    st.markdown(
        f"<div style='{_CARD};border-left:4px solid {a_col}'>"
        f"<span style='font-size:1.05rem;font-weight:800;color:{a_col}'>"
        f"● {a_txt}</span> <span style='font-size:.8rem;color:#8892a4'>· "
        f"entries {s['start_time']}–{s['end_time']} ET · target "
        f"+{s['target_pct']}%</span></div>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pool (USD)", f"${s['pool']:,.0f}",
              delta=f"${s['realized_pnl']:+,.0f}" if s["realized_pnl"] else None)
    m2.metric("Deployed", f"${s['deployed']:,.0f}")
    m3.metric("Available", f"${s['available']:,.0f}")
    m4.metric("Trades today", f"{s['trades_today_count']}/{s['max_trades_per_day']}")

    c1, c2 = st.columns([1.2, 2.8])
    with c1:
        if armed:
            if st.button("⛔ DISARM", type="primary", width="stretch",
                         key="us_disarm"):
                disarm("user"); st.rerun()
        else:
            if st.button("🟢 ARM — US Paper", type="primary", width="stretch",
                         key="us_arm"):
                ok, msg = arm()
                (st.success if ok else st.error)(msg)
                if ok:
                    st.rerun()
    with c2:
        st.caption("Same gates: armed → US window (settle ke baad) → daily/"
                   "position limits → symbol once/day → score+conviction → "
                   "live anchor (chase nahi) → capital+reserve. Din ka loss "
                   "limit → khud DISARM. Costs: Alpaca-style (~$0 commission).")

    with st.expander("⚙️ US limits", expanded=not s["allocation"]):
        l1, l2, l3 = st.columns(3)
        with l1:
            alloc = st.number_input("Allocation ($)", 0.0,
                                    value=float(s["allocation"]), step=1000.0,
                                    key="us_alloc")
            target = st.slider("Target %", 1.0, 15.0,
                               float(s["target_pct"]), 0.5, key="us_tgt")
        with l2:
            max_pos = st.number_input("Max open positions", 1, 10,
                                      int(s["max_open_positions"]), key="us_pos")
            max_day = st.number_input("Max trades / day", 1, 15,
                                      int(s["max_trades_per_day"]), key="us_day")
        with l3:
            min_sc = st.slider("Min score", 40, 95, int(s["min_score"]),
                               key="us_score")
            min_cv = st.slider("Min conviction", 0, 90, int(s["min_conviction"]),
                               key="us_conv")
        if st.button("💾 Save US limits", key="us_save", width="stretch"):
            set_config(allocation=alloc, target_pct=float(target),
                       max_open_positions=int(max_pos),
                       max_trades_per_day=int(max_day),
                       min_score=float(min_sc), min_conviction=float(min_cv))
            st.success("US limits saved."); st.rerun()

    if s["open_trades"]:
        st.markdown("#### 📍 US positions")
        for t in s["open_trades"]:
            st.markdown(
                f"<div style='{_CARD}'>"
                f"<span style='color:#e2e8f0;font-weight:700;"
                f"font-family:JetBrains Mono,monospace'>{t['symbol']}</span>"
                f"<span style='font-size:.75rem;color:#8892a4'> (PAPER) · "
                f"{t['qty']} sh @ ${float(t['entry_price'] or 0):,.2f} · "
                f"stop ${float(t['stop_price'] or 0):,.2f} · "
                f"target ${float(t['target_price'] or 0):,.2f}</span></div>",
                unsafe_allow_html=True)

    # ── Aggressiveness dial + rejection funnel ────────────────────────────────
    import execution.us_autopilot as _ap_us
    _render_funnel_and_preset(_ap_us, "ap_us")

    rc = report_card()
    v_style = {"COLLECTING_EVIDENCE": "#f59e0b", "READY_CANDIDATE": "#00d4a0",
               "NOT_READY": "#ff4b4b"}
    st.markdown(
        f"<div style='{_CARD};border-left:4px solid "
        f"{v_style.get(rc['verdict'], '#8892a4')}'>"
        f"<b style='color:{v_style.get(rc['verdict'], '#8892a4')}'>"
        f"{rc['verdict']}</b> · <span style='font-size:.82rem;color:#c9d1d9'>"
        f"{rc['verdict_reason']}</span></div>", unsafe_allow_html=True)
    if rc["trades"]:
        stt = rc["stats"]
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Closed", stt["n"])
        r2.metric("Win rate", f"{stt['win_rate']:.0f}%")
        r3.metric("Net P&L", f"${stt['total_pnl']:+,.2f}")
        r4.metric("Expectancy", f"{stt['expectancy_r']:+.2f}R")

    st.markdown("#### 📜 Activity")
    acts = s.get("activity", [])
    if acts:
        st.markdown(f"<div style='{_CARD};font-size:.78rem;color:#c9d1d9;"
                    f"font-family:JetBrains Mono,monospace'>"
                    + "<br>".join(acts[:20]) + "</div>", unsafe_allow_html=True)
    else:
        st.caption("Allocation set karke ARM karo — US market hours mein "
                   "decisions yahan aayenge.")
