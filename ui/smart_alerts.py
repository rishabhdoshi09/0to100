"""Smart Alerts — multi-condition alert builder, manager, and checker."""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# ── Theme ──────────────────────────────────────────────────────────────────────
_CYAN   = "#00d4ff"
_GREEN  = "#00ff88"
_RED    = "#ff4466"
_AMBER  = "#ffb800"
_WHITE  = "#e8eaf0"
_MUTED  = "#8892a4"

# ── Paths ──────────────────────────────────────────────────────────────────────
_ALERTS_FILE = Path("logs/smart_alerts.json")

# ── Condition types ────────────────────────────────────────────────────────────
CONDITION_TYPES = [
    "Price > X",
    "Price < X",
    "RSI > X",
    "RSI < X",
    "Volume spike (> 2x avg)",
    "% change > X%",
    "% change < -X%",
    "Above 20 SMA",
    "Below 20 SMA",
    "Above 50 SMA",
    "Below 50 SMA",
    "Above 200 SMA",
    "Below 200 SMA",
    "Price near 52W high (within 3%)",
    "Price near 52W low (within 3%)",
]

TRIGGER_TYPES = ["Once", "Every cycle", "Until cancelled"]

CONDITION_NEEDS_X = {
    "Price > X", "Price < X",
    "RSI > X", "RSI < X",
    "% change > X%", "% change < -X%",
}

# ── Pre-built templates ────────────────────────────────────────────────────────
TEMPLATES = {
    "Breakout Alert": {
        "description": "Price breaks above 52W high with volume surge",
        "conditions": [
            {"type": "Price near 52W high (within 3%)", "value": None, "logic": "AND"},
            {"type": "Volume spike (> 2x avg)", "value": None, "logic": "AND"},
        ],
        "trigger": "Once",
    },
    "Oversold Bounce": {
        "description": "RSI deeply oversold + price near 52W low support",
        "conditions": [
            {"type": "RSI < X", "value": 35, "logic": "AND"},
            {"type": "Price near 52W low (within 3%)", "value": None, "logic": "AND"},
        ],
        "trigger": "Once",
    },
    "Momentum Alert": {
        "description": "Strong RSI + price trading above short-term average",
        "conditions": [
            {"type": "RSI > X", "value": 65, "logic": "AND"},
            {"type": "Above 20 SMA", "value": None, "logic": "AND"},
        ],
        "trigger": "Every cycle",
    },
    "Big Move Alert": {
        "description": "Stock moved more than 3% in a session",
        "conditions": [
            {"type": "% change > X%", "value": 3.0, "logic": "OR"},
            {"type": "% change < -X%", "value": 3.0, "logic": "OR"},
        ],
        "trigger": "Every cycle",
    },
}


# ── Persistence ────────────────────────────────────────────────────────────────
def _load_alerts() -> dict:
    if _ALERTS_FILE.exists():
        try:
            return json.loads(_ALERTS_FILE.read_text())
        except Exception:
            pass
    return {"alerts": [], "history": []}


def _save_alerts(data: dict) -> None:
    _ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ALERTS_FILE.write_text(json.dumps(data, indent=2, default=str))


# ── Market data fetch ──────────────────────────────────────────────────────────
def _nse(sym: str) -> str:
    s = sym.strip().upper()
    return s if s.endswith(".NS") else s + ".NS"


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_market_data(symbol: str) -> dict[str, Any] | None:
    """Fetch latest price, RSI, SMAs, 52W high/low, volume from yfinance."""
    try:
        import yfinance as yf
        ticker  = yf.Ticker(_nse(symbol))
        hist    = ticker.history(period="1y", interval="1d")
        if hist.empty or len(hist) < 10:
            return None

        hist.index = pd.to_datetime(hist.index).tz_localize(None) if hist.index.tz else pd.to_datetime(hist.index)
        close   = hist["Close"]
        volume  = hist["Volume"]

        # Current price
        price = float(close.iloc[-1])

        # % change today
        pct_change = float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100) if len(close) >= 2 else 0.0

        # SMAs
        sma20  = float(close.rolling(20).mean().iloc[-1])  if len(close) >= 20  else price
        sma50  = float(close.rolling(50).mean().iloc[-1])  if len(close) >= 50  else price
        sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else price

        # RSI (14-period)
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, float("nan"))
        rsi   = float(100 - 100 / (1 + rs.iloc[-1])) if not rs.isna().iloc[-1] else 50.0

        # 52W high/low
        w52_high = float(close.rolling(252).max().iloc[-1])
        w52_low  = float(close.rolling(252).min().iloc[-1])

        # Volume spike: today vs 20-day avg
        avg_vol    = float(volume.rolling(20).mean().iloc[-1])
        today_vol  = float(volume.iloc[-1])
        vol_ratio  = today_vol / avg_vol if avg_vol > 0 else 1.0

        return {
            "symbol":    symbol,
            "price":     price,
            "pct_change": pct_change,
            "sma20":     sma20,
            "sma50":     sma50,
            "sma200":    sma200,
            "rsi":       rsi,
            "52w_high":  w52_high,
            "52w_low":   w52_low,
            "vol_ratio": vol_ratio,
        }
    except Exception as e:
        return None


# ── Condition evaluator ────────────────────────────────────────────────────────
def _evaluate_condition(cond: dict, mkt: dict) -> bool:
    ctype = cond.get("type", "")
    val   = cond.get("value")

    try:
        if ctype == "Price > X":
            return mkt["price"] > float(val)
        if ctype == "Price < X":
            return mkt["price"] < float(val)
        if ctype == "RSI > X":
            return mkt["rsi"] > float(val)
        if ctype == "RSI < X":
            return mkt["rsi"] < float(val)
        if ctype == "Volume spike (> 2x avg)":
            return mkt["vol_ratio"] > 2.0
        if ctype == "% change > X%":
            return mkt["pct_change"] > float(val)
        if ctype == "% change < -X%":
            return mkt["pct_change"] < -float(val)
        if ctype == "Above 20 SMA":
            return mkt["price"] > mkt["sma20"]
        if ctype == "Below 20 SMA":
            return mkt["price"] < mkt["sma20"]
        if ctype == "Above 50 SMA":
            return mkt["price"] > mkt["sma50"]
        if ctype == "Below 50 SMA":
            return mkt["price"] < mkt["sma50"]
        if ctype == "Above 200 SMA":
            return mkt["price"] > mkt["sma200"]
        if ctype == "Below 200 SMA":
            return mkt["price"] < mkt["sma200"]
        if ctype == "Price near 52W high (within 3%)":
            return abs(mkt["price"] - mkt["52w_high"]) / mkt["52w_high"] <= 0.03
        if ctype == "Price near 52W low (within 3%)":
            return abs(mkt["price"] - mkt["52w_low"]) / mkt["52w_low"] <= 0.03
    except Exception:
        pass
    return False


def _evaluate_alert(alert: dict, mkt: dict) -> bool:
    """Evaluate all conditions with AND/OR logic."""
    conditions = alert.get("conditions", [])
    if not conditions:
        return False

    # First condition is always evaluated; subsequent ones use their logic op
    result = _evaluate_condition(conditions[0], mkt)
    for cond in conditions[1:]:
        logic = cond.get("logic", "AND").upper()
        c_result = _evaluate_condition(cond, mkt)
        if logic == "OR":
            result = result or c_result
        else:
            result = result and c_result
    return result


# ── Telegram notification ──────────────────────────────────────────────────────
def _send_telegram(message: str) -> bool:
    token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False
    try:
        import urllib.request, urllib.parse
        url  = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": message, "parse_mode": "HTML"}).encode()
        req  = urllib.request.Request(url, data=data)
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception:
        return False


# ── Condition builder widget ───────────────────────────────────────────────────
def _condition_builder_row(idx: int, default: dict | None = None) -> dict | None:
    """Render a single condition row. Returns the condition dict or None to remove."""
    default = default or {}
    cols = st.columns([3, 2, 1, 1])

    ctype = cols[0].selectbox(
        "Condition",
        CONDITION_TYPES,
        index=CONDITION_TYPES.index(default.get("type", CONDITION_TYPES[0]))
        if default.get("type") in CONDITION_TYPES else 0,
        key=f"ctype_{idx}",
        label_visibility="collapsed",
    )

    needs_val = ctype in CONDITION_NEEDS_X
    if needs_val:
        val = cols[1].number_input(
            "Value",
            value=float(default.get("value") or 0),
            step=0.5,
            format="%.2f",
            key=f"cval_{idx}",
            label_visibility="collapsed",
        )
    else:
        cols[1].markdown(
            "<span style='color:#8892a4;font-size:.8rem;padding-top:8px;display:block;'>"
            "— no value needed —</span>",
            unsafe_allow_html=True,
        )
        val = None

    logic = cols[2].selectbox(
        "Logic",
        ["AND", "OR"],
        index=0 if default.get("logic", "AND") == "AND" else 1,
        key=f"clogic_{idx}",
        label_visibility="collapsed",
    )

    remove = cols[3].button("✕", key=f"remove_{idx}", help="Remove this condition")
    if remove:
        return None

    return {"type": ctype, "value": val, "logic": logic}


# ── Main render ────────────────────────────────────────────────────────────────
def render_smart_alerts() -> None:
    st.markdown(
        """
        <style>
        .alert-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 1rem 1.4rem;
            margin-bottom: 0.7rem;
        }
        .alert-active  { border-left: 3px solid #00d4ff; }
        .alert-triggered { border-left: 3px solid #00ff88; }
        .alert-cancelled { border-left: 3px solid #8892a4; opacity: .6; }
        .badge {
            display: inline-block;
            padding: .15rem .55rem;
            border-radius: 6px;
            font-size: .72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: .04em;
        }
        .badge-active    { background: rgba(0,212,255,.15); color: #00d4ff; border: 1px solid rgba(0,212,255,.3); }
        .badge-triggered { background: rgba(0,255,136,.12); color: #00ff88; border: 1px solid rgba(0,255,136,.3); }
        .badge-cancelled { background: rgba(136,146,164,.1); color: #8892a4; border: 1px solid rgba(136,146,164,.25); }
        .template-card {
            background: rgba(255,184,0,0.05);
            border: 1px solid rgba(255,184,0,0.18);
            border-radius: 12px;
            padding: .8rem 1.1rem;
            margin: .3rem 0;
        }
        .hist-row {
            background: rgba(0,255,136,0.04);
            border-left: 2px solid #00ff88;
            border-radius: 0 8px 8px 0;
            padding: .4rem .9rem;
            margin: .25rem 0;
            font-size: .83rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## 🔔 Smart Alert Builder")
    st.markdown(
        "<span style='color:#8892a4;font-size:.85rem;'>"
        "Multi-condition alerts · AND/OR logic · In-app + Telegram · Templates"
        "</span>",
        unsafe_allow_html=True,
    )

    # Load alerts data
    if "alerts_data" not in st.session_state:
        st.session_state.alerts_data = _load_alerts()

    alerts_data = st.session_state.alerts_data

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_build, tab_active, tab_history, tab_templates = st.tabs([
        "🔨 Build Alert", "📋 Active Alerts", "📜 History", "⚡ Templates"
    ])

    # ════════════════════════════════════════════════════════════════════════
    # Tab 1 — Build Alert
    # ════════════════════════════════════════════════════════════════════════
    with tab_build:
        st.markdown("#### Define a new alert")

        # Symbol
        symbol_input = st.text_input(
            "Stock Symbol (NSE)",
            placeholder="e.g. RELIANCE",
            help="Enter the NSE ticker symbol (without .NS suffix)",
        )

        st.markdown("##### Conditions")
        st.caption(
            "Add one or more conditions. Use AND/OR to combine them. "
            "The first condition's logic operator is ignored."
        )

        # Condition rows — managed in session state
        if "builder_conditions" not in st.session_state:
            st.session_state.builder_conditions = [
                {"type": "Price > X", "value": 0.0, "logic": "AND"}
            ]

        new_conditions = []
        for i, cond in enumerate(st.session_state.builder_conditions):
            st.markdown(
                f"<span style='color:{_MUTED};font-size:.75rem;'>Condition {i+1}</span>",
                unsafe_allow_html=True,
            )
            result = _condition_builder_row(i, cond)
            if result is not None:
                new_conditions.append(result)

        st.session_state.builder_conditions = new_conditions

        if st.button("➕ Add condition"):
            st.session_state.builder_conditions.append(
                {"type": "RSI > X", "value": 60.0, "logic": "AND"}
            )
            st.rerun()

        st.divider()

        col_trigger, col_method = st.columns(2)
        trigger_type = col_trigger.selectbox("Trigger type", TRIGGER_TYPES)
        alert_method = col_method.multiselect(
            "Alert method",
            ["In-app (toast)", "Telegram"],
            default=["In-app (toast)"],
        )

        alert_name = st.text_input("Alert name (optional)", placeholder="My breakout alert")

        # Telegram check
        tg_configured = bool(os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"))
        if "Telegram" in alert_method and not tg_configured:
            st.warning(
                "Telegram selected but `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` "
                "not found in environment. Set them in `.env` to enable Telegram alerts.",
                icon="⚠️",
            )

        save_col, preview_col = st.columns([1, 2])
        with save_col:
            if st.button("💾 Save Alert", type="primary"):
                if not symbol_input.strip():
                    st.error("Please enter a stock symbol.")
                elif not st.session_state.builder_conditions:
                    st.error("Add at least one condition.")
                else:
                    new_alert = {
                        "id":          str(uuid.uuid4())[:8],
                        "name":        alert_name.strip() or f"{symbol_input.upper()} alert",
                        "symbol":      symbol_input.strip().upper(),
                        "conditions":  st.session_state.builder_conditions.copy(),
                        "trigger":     trigger_type,
                        "methods":     alert_method,
                        "status":      "Active",
                        "created_at":  datetime.now().isoformat(),
                        "triggered_at": None,
                        "trigger_count": 0,
                    }
                    alerts_data["alerts"].append(new_alert)
                    _save_alerts(alerts_data)
                    st.session_state.alerts_data = alerts_data
                    st.session_state.builder_conditions = [
                        {"type": "Price > X", "value": 0.0, "logic": "AND"}
                    ]
                    st.success(f"Alert '{new_alert['name']}' saved!", icon="✅")
                    st.rerun()

        with preview_col:
            if st.session_state.builder_conditions and symbol_input:
                parts = []
                for i, c in enumerate(st.session_state.builder_conditions):
                    prefix = "" if i == 0 else f" {c.get('logic','AND')} "
                    val_str = f" {c['value']}" if c.get("value") is not None else ""
                    parts.append(f"{prefix}{c['type']}{val_str}")
                preview_text = "".join(parts)
                st.markdown(
                    f"""<div class="alert-card alert-active">
                    <span style="color:{_MUTED};font-size:.72rem;">PREVIEW</span><br>
                    <span style="color:{_CYAN};font-weight:600;">{symbol_input.upper()}</span>
                    <span style="color:{_WHITE};font-size:.85rem;"> — {preview_text}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

    # ════════════════════════════════════════════════════════════════════════
    # Tab 2 — Active Alerts + Check
    # ════════════════════════════════════════════════════════════════════════
    with tab_active:
        alerts = alerts_data.get("alerts", [])

        # Check alerts button
        check_col, info_col = st.columns([1, 3])
        with check_col:
            check_now = st.button("🔍 Check all alerts now", type="primary")

        if check_now:
            with st.spinner("Checking all active alerts…"):
                triggered_names = []
                for alert in alerts:
                    if alert.get("status") != "Active":
                        continue
                    mkt = _fetch_market_data(alert["symbol"])
                    if mkt is None:
                        info_col.warning(
                            f"Could not fetch data for {alert['symbol']}.", icon="⚠️"
                        )
                        continue
                    fired = _evaluate_alert(alert, mkt)
                    if fired:
                        alert["trigger_count"] = alert.get("trigger_count", 0) + 1
                        alert["triggered_at"]  = datetime.now().isoformat()
                        triggered_names.append(alert["name"])

                        # Log to history
                        hist_entry = {
                            "alert_id":    alert["id"],
                            "alert_name":  alert["name"],
                            "symbol":      alert["symbol"],
                            "triggered_at": datetime.now().isoformat(),
                            "price":       mkt["price"],
                            "rsi":         round(mkt["rsi"], 1),
                            "pct_change":  round(mkt["pct_change"], 2),
                        }
                        alerts_data.setdefault("history", []).append(hist_entry)

                        # Update status for "Once" alerts
                        if alert.get("trigger") == "Once":
                            alert["status"] = "Triggered"

                        # In-app toast
                        if "In-app (toast)" in alert.get("methods", []):
                            st.toast(
                                f"🔔 {alert['name']} triggered! "
                                f"{alert['symbol']} @ ₹{mkt['price']:,.2f}",
                                icon="🔔",
                            )

                        # Telegram
                        if "Telegram" in alert.get("methods", []):
                            msg = (
                                f"<b>🔔 SimpleQuant Alert</b>\n"
                                f"<b>{alert['name']}</b>\n"
                                f"{alert['symbol']} @ ₹{mkt['price']:,.2f}\n"
                                f"RSI: {mkt['rsi']:.1f} | Chg: {mkt['pct_change']:+.2f}%"
                            )
                            _send_telegram(msg)

                _save_alerts(alerts_data)
                st.session_state.alerts_data = alerts_data

            if triggered_names:
                st.success(
                    f"Triggered: {', '.join(triggered_names)}",
                    icon="🔔",
                )
            else:
                st.info("No alerts triggered at current prices.", icon="ℹ️")

        st.divider()

        if not alerts:
            st.markdown(
                f"<div style='color:{_MUTED};text-align:center;padding:2rem;'>"
                "No alerts configured yet. Use the Build Alert tab to create one."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            # Status filter
            status_filter = st.multiselect(
                "Filter by status",
                ["Active", "Triggered", "Cancelled"],
                default=["Active", "Triggered"],
            )
            filtered = [a for a in alerts if a.get("status") in status_filter]

            for alert in filtered:
                status   = alert.get("status", "Active")
                css_cls  = f"alert-{status.lower()}"
                badge_cls= f"badge-{status.lower()}"
                conds    = alert.get("conditions", [])
                cond_str = " | ".join(
                    f"{c.get('logic','AND')} {c['type']}"
                    + (f" {c['value']}" if c.get("value") is not None else "")
                    for c in conds
                )

                triggered_at_str = ""
                if alert.get("triggered_at"):
                    triggered_at_str = (
                        f"<span style='color:{_GREEN};font-size:.75rem;'>"
                        f"Last triggered: {alert['triggered_at'][:16]}</span>"
                    )

                st.markdown(
                    f"""<div class="alert-card {css_cls}">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                      <div>
                        <span style="color:{_WHITE};font-weight:600;font-size:.95rem;">
                          {alert.get('name','Unnamed')}</span>
                        <span style="color:{_MUTED};font-size:.78rem;margin-left:.6rem;">
                          #{alert.get('id','?')}</span><br>
                        <span style="color:{_CYAN};font-size:.82rem;">{alert.get('symbol','')}</span>
                        <span style="color:{_MUTED};font-size:.78rem;margin-left:.5rem;">
                          {cond_str}</span><br>
                        <span style="color:{_MUTED};font-size:.75rem;">
                          Trigger: {alert.get('trigger','')} ·
                          Methods: {', '.join(alert.get('methods',[]))} ·
                          Fired: {alert.get('trigger_count',0)}x</span><br>
                        {triggered_at_str}
                      </div>
                      <span class="badge {badge_cls}">{status}</span>
                    </div>
                    </div>""",
                    unsafe_allow_html=True,
                )

                act_col1, act_col2, _ = st.columns([1, 1, 5])
                alert_id = alert.get("id")
                if status == "Active":
                    if act_col1.button("❌ Cancel", key=f"cancel_{alert_id}"):
                        alert["status"] = "Cancelled"
                        _save_alerts(alerts_data)
                        st.session_state.alerts_data = alerts_data
                        st.rerun()
                else:
                    if act_col1.button("♻️ Reactivate", key=f"reactivate_{alert_id}"):
                        alert["status"] = "Active"
                        alert["triggered_at"] = None
                        _save_alerts(alerts_data)
                        st.session_state.alerts_data = alerts_data
                        st.rerun()
                if act_col2.button("🗑️ Delete", key=f"delete_{alert_id}"):
                    alerts_data["alerts"] = [
                        a for a in alerts_data["alerts"] if a.get("id") != alert_id
                    ]
                    _save_alerts(alerts_data)
                    st.session_state.alerts_data = alerts_data
                    st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # Tab 3 — History
    # ════════════════════════════════════════════════════════════════════════
    with tab_history:
        history = alerts_data.get("history", [])
        if not history:
            st.markdown(
                f"<div style='color:{_MUTED};text-align:center;padding:2rem;'>"
                "No alerts have been triggered yet."
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"**{len(history)} trigger events logged**")
            for entry in reversed(history[-50:]):   # show last 50
                ts      = entry.get("triggered_at", "")[:16]
                sym     = entry.get("symbol", "")
                name    = entry.get("alert_name", "")
                price   = entry.get("price", 0)
                rsi     = entry.get("rsi", 0)
                chg     = entry.get("pct_change", 0)
                chg_col = _GREEN if chg >= 0 else _RED

                st.markdown(
                    f"""<div class="hist-row">
                    <span style="color:{_MUTED};font-size:.72rem;">{ts}</span>
                    <span style="color:{_CYAN};font-weight:600;margin-left:.5rem;">{sym}</span>
                    <span style="color:{_WHITE};margin-left:.4rem;">— {name}</span><br>
                    <span style="color:{_MUTED};font-size:.78rem;">
                      ₹{price:,.2f} · RSI {rsi:.1f} ·
                      <span style="color:{chg_col};">{chg:+.2f}%</span>
                    </span>
                    </div>""",
                    unsafe_allow_html=True,
                )

            if st.button("🗑️ Clear history"):
                alerts_data["history"] = []
                _save_alerts(alerts_data)
                st.session_state.alerts_data = alerts_data
                st.success("History cleared.")
                st.rerun()

    # ════════════════════════════════════════════════════════════════════════
    # Tab 4 — Templates
    # ════════════════════════════════════════════════════════════════════════
    with tab_templates:
        st.markdown("#### Pre-built alert templates")
        st.caption(
            "Click 'Use template' to load it into the builder. "
            "You can then set the symbol and save."
        )

        for tname, tmpl in TEMPLATES.items():
            cond_preview = " AND ".join(
                f"{c['type']}" + (f" {c['value']}" if c.get("value") else "")
                for c in tmpl["conditions"]
            )
            col_desc, col_btn = st.columns([4, 1])
            col_desc.markdown(
                f"""<div class="template-card">
                <span style="color:{_AMBER};font-weight:600;font-size:.95rem;">{tname}</span><br>
                <span style="color:{_MUTED};font-size:.8rem;">{tmpl['description']}</span><br>
                <span style="color:{_WHITE};font-size:.78rem;margin-top:.3rem;display:block;">
                  Conditions: <span style="color:{_CYAN};">{cond_preview}</span>
                </span>
                <span style="color:{_MUTED};font-size:.75rem;">Trigger: {tmpl['trigger']}</span>
                </div>""",
                unsafe_allow_html=True,
            )
            with col_btn:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button(f"Use", key=f"tpl_{tname}"):
                    st.session_state.builder_conditions = [
                        dict(c) for c in tmpl["conditions"]
                    ]
                    st.session_state["_tpl_trigger"] = tmpl["trigger"]
                    st.session_state["_tpl_name"]    = tname
                    st.toast(
                        f"Template '{tname}' loaded — go to Build Alert tab to set symbol & save.",
                        icon="⚡",
                    )
                    st.rerun()

        # Telegram setup status
        st.divider()
        st.markdown("#### Telegram Integration Status")
        tg_token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
        tg_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        if tg_token and tg_chat_id:
            st.success(
                "Telegram is configured. Alerts with the Telegram method will send messages.",
                icon="✅",
            )
            if st.button("🧪 Send test Telegram message"):
                ok = _send_telegram(
                    "<b>SimpleQuant</b> — Telegram integration test successful! 🔔"
                )
                if ok:
                    st.success("Test message sent!")
                else:
                    st.error("Failed to send. Check your BOT_TOKEN and CHAT_ID.")
        else:
            st.warning(
                "Telegram not configured. Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` "
                "in your `.env` file to enable Telegram notifications.",
                icon="⚠️",
            )
            st.code(
                "# Add to .env\nTELEGRAM_BOT_TOKEN=your_bot_token_here\n"
                "TELEGRAM_CHAT_ID=your_chat_id_here",
                language="bash",
            )
