"""
Telegram Alerts page — manage price/RSI/breakout alert rules and view history.
"""
from __future__ import annotations

import streamlit as st


def render_alerts_page() -> None:
    # ── Lazy imports so the page is renderable even if alerts/ isn't set up ──
    from alerts.telegram_alerts import AlertEngine, AlertManager

    engine  = AlertEngine()
    manager = AlertManager()

    # ── Page header ──────────────────────────────────────────────────────────
    st.markdown(
        "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;"
        "font-size:1.3rem;letter-spacing:2px;margin-bottom:2px'>"
        "🔔 TELEGRAM ALERT ENGINE</h2>"
        "<p style='color:#4a5568;font-size:.75rem;margin-bottom:1rem'>"
        "Price, RSI, and breakout alerts delivered to Telegram in real time.</p>",
        unsafe_allow_html=True,
    )

    # ── Setup / status section ────────────────────────────────────────────────
    if not engine.is_configured():
        st.warning(
            "⚠️ **Telegram not configured**\n\n"
            "Add the following to your `.env` file:\n\n"
            "```\n"
            "TELEGRAM_BOT_TOKEN=your_bot_token\n"
            "TELEGRAM_CHAT_ID=your_chat_id\n"
            "```",
            icon="⚠️",
        )
        with st.expander("📖 How to get your token and chat ID", expanded=True):
            st.markdown(
                """
**Step 1 — Create a bot**
1. Open Telegram and search for [@BotFather](https://t.me/BotFather)
2. Send `/newbot` and follow the prompts
3. Copy the **API token** (looks like `123456789:ABCdef…`)

**Step 2 — Get your Chat ID**
1. Send any message to your new bot
2. Visit this URL in your browser (replace `<token>` with your token):
   `https://api.telegram.org/bot<token>/getUpdates`
3. Find `"chat": {"id": <number>}` — that number is your **TELEGRAM_CHAT_ID**

**Step 3 — Add to .env**
```
TELEGRAM_BOT_TOKEN=123456789:ABCdef...
TELEGRAM_CHAT_ID=987654321
```

Then restart the app.
                """,
                unsafe_allow_html=True,
            )
    else:
        _c1, _c2 = st.columns([3, 1])
        with _c1:
            st.success("✅ **Telegram connected** — alerts will be delivered to your chat.", icon="✅")
        with _c2:
            if st.button("📤 Send Test Alert", key="tg_test_btn", use_container_width=True):
                with st.spinner("Sending…"):
                    ok = engine.send_test()
                if ok:
                    st.success("Test alert sent!", icon="✅")
                else:
                    st.error("Failed to send — check your token/chat ID.", icon="❌")

    st.markdown("---")

    # ── Add Alert form ────────────────────────────────────────────────────────
    st.markdown(
        "<h3 style='color:#e8eaf0;font-size:1rem;font-weight:700;"
        "letter-spacing:1px;margin-bottom:.5rem'>➕ ADD NEW ALERT</h3>",
        unsafe_allow_html=True,
    )

    _ALERT_TYPE_LABELS = {
        "Price Cross Above":  "PRICE_CROSS",
        "Price Cross Below":  "PRICE_CROSS_BELOW",
        "RSI Above":          "RSI_CROSS",
        "RSI Breakout":       "BREAKOUT",
    }

    with st.form("add_alert_form", clear_on_submit=True):
        _fa1, _fa2, _fa3, _fa4 = st.columns([2, 2, 2, 1])
        with _fa1:
            new_symbol = st.text_input(
                "Symbol",
                placeholder="e.g. RELIANCE",
                help="NSE symbol, uppercase",
            ).strip().upper()
        with _fa2:
            new_type_label = st.selectbox(
                "Alert Type",
                options=list(_ALERT_TYPE_LABELS.keys()),
            )
        with _fa3:
            new_threshold = st.number_input(
                "Threshold (Price ₹ or RSI level)",
                min_value=0.0,
                max_value=1_000_000.0,
                value=0.0,
                step=1.0,
                help="For price alerts enter ₹ value; for RSI alerts enter 0–100",
            )
        with _fa4:
            st.markdown("<br>", unsafe_allow_html=True)   # vertical align
            submitted = st.form_submit_button(
                "➕ Add Alert",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            if not new_symbol:
                st.error("Please enter a symbol.", icon="❌")
            elif new_threshold <= 0:
                st.error("Threshold must be greater than 0.", icon="❌")
            else:
                alert_type_key = _ALERT_TYPE_LABELS[new_type_label]
                rule_id = manager.add_rule(new_symbol, alert_type_key, new_threshold)  # type: ignore[arg-type]
                st.success(
                    f"Alert #{rule_id} added: {new_symbol} — {new_type_label} @ {new_threshold}",
                    icon="✅",
                )

    st.markdown("---")

    # ── Active Alerts table ───────────────────────────────────────────────────
    st.markdown(
        "<h3 style='color:#e8eaf0;font-size:1rem;font-weight:700;"
        "letter-spacing:1px;margin-bottom:.5rem'>📋 ACTIVE ALERTS</h3>",
        unsafe_allow_html=True,
    )

    rules = manager.get_rules()
    active_rules = [r for r in rules if not r.triggered]

    if not active_rules:
        st.markdown(
            "<div style='text-align:center;padding:1.5rem;color:#4a5568;font-size:.82rem'>"
            "No active alerts. Add one above.</div>",
            unsafe_allow_html=True,
        )
    else:
        # Table header
        st.markdown(
            "<div style='display:grid;grid-template-columns:80px 160px 110px 70px;"
            "gap:6px;padding:4px 8px;font-size:.6rem;color:#4a5568;font-weight:700;"
            "text-transform:uppercase;letter-spacing:.05em;"
            "border-bottom:1px solid rgba(255,255,255,.08);margin-bottom:4px'>"
            "<span>Symbol</span><span>Type</span><span>Threshold</span><span>Delete</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        for rule in active_rules:
            _rc1, _rc2, _rc3, _rc4 = st.columns([1, 2, 1.4, 1])
            with _rc1:
                st.markdown(
                    f"<span style='color:#e8eaf0;font-weight:700;"
                    f"font-family:JetBrains Mono,monospace;font-size:.78rem'>"
                    f"{rule.symbol}</span>",
                    unsafe_allow_html=True,
                )
            with _rc2:
                _type_display = rule.alert_type.replace("_", " ").title()
                st.markdown(
                    f"<span style='color:#8892a4;font-size:.75rem'>{_type_display}</span>",
                    unsafe_allow_html=True,
                )
            with _rc3:
                st.markdown(
                    f"<span style='color:#00d4a0;font-family:JetBrains Mono,monospace;"
                    f"font-size:.78rem'>{rule.threshold:,.2f}</span>",
                    unsafe_allow_html=True,
                )
            with _rc4:
                if st.button(
                    "🗑️",
                    key=f"del_rule_{rule.rule_id}",
                    help=f"Delete alert #{rule.rule_id}",
                    use_container_width=True,
                ):
                    manager.delete_rule(rule.rule_id)
                    st.rerun()

    st.markdown("---")

    # ── Recent Fires log ──────────────────────────────────────────────────────
    st.markdown(
        "<h3 style='color:#e8eaf0;font-size:1rem;font-weight:700;"
        "letter-spacing:1px;margin-bottom:.5rem'>🔥 RECENT FIRES (last 20)</h3>",
        unsafe_allow_html=True,
    )

    fires = manager.get_recent_fires(limit=20)

    if not fires:
        st.markdown(
            "<div style='text-align:center;padding:1.5rem;color:#4a5568;font-size:.82rem'>"
            "No alerts have fired yet.</div>",
            unsafe_allow_html=True,
        )
    else:
        # Header
        st.markdown(
            "<div style='display:grid;grid-template-columns:160px 80px 170px 90px 70px 70px;"
            "gap:6px;padding:4px 8px;font-size:.6rem;color:#4a5568;font-weight:700;"
            "text-transform:uppercase;letter-spacing:.05em;"
            "border-bottom:1px solid rgba(255,255,255,.08);margin-bottom:4px'>"
            "<span>Fired At</span><span>Symbol</span><span>Type</span>"
            "<span>Threshold</span><span>Price</span><span>RSI</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        for f in fires:
            price_str = f"₹{f['price']:,.2f}" if f["price"] is not None else "—"
            rsi_str   = f"{f['rsi']:.1f}"     if f["rsi"]   is not None else "—"
            _type_display = f["alert_type"].replace("_", " ").title()
            st.markdown(
                f"<div style='display:grid;grid-template-columns:160px 80px 170px 90px 70px 70px;"
                f"gap:6px;padding:5px 8px;font-size:.72rem;"
                f"font-family:JetBrains Mono,monospace;"
                f"border-bottom:1px solid rgba(255,255,255,.03);align-items:center'>"
                f"<span style='color:#4a5568'>{f['fired_at']}</span>"
                f"<span style='color:#e8eaf0;font-weight:700'>{f['symbol']}</span>"
                f"<span style='color:#8892a4;font-size:.67rem'>{_type_display}</span>"
                f"<span style='color:#c9d1e0'>{f['threshold']:,.2f}</span>"
                f"<span style='color:#00d4a0'>{price_str}</span>"
                f"<span style='color:#f59e0b'>{rsi_str}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
