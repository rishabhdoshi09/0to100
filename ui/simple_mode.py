"""
🧒 Simple Mode — the beginner-friendly presentation layer.

Thin Streamlit rendering over `core.simple_language` (all the words + logic live there,
pure and tested). This layer only:
  • remembers the presentation depth (Simple is the new-user default);
  • gathers the CURRENT system state READ-ONLY (autopilot status, market open, data
    health) and hands it to the pure logic;
  • draws large, plain cards.

It changes no trading behaviour, no configuration and no permission. It imports NO
order path — it can read `autopilot.get_status()` (a pure read) but never `arm`,
`consider`, `place_trade`, Telegram order actions, or the broker. Switching Simple↔
Advanced only changes how much detail is shown, never what the system is allowed to do.
"""
from __future__ import annotations

import streamlit as st

from core import simple_language as S

_DEPTH_KEY = "ui_depth"           # 'simple' | 'advanced' (session-only, presentation)


# ── presentation depth (Simple default; never changes permissions) ─────────────

def get_ui_depth() -> str:
    return st.session_state.get(_DEPTH_KEY, S.DEFAULT_MODE)


def is_simple() -> bool:
    return S.is_simple(get_ui_depth())


def render_depth_toggle() -> None:
    """Sidebar toggle. Default OFF = Simple (the new-user default). Turning it on shows
    Advanced detail only — it does not touch trading, config or permissions."""
    advanced = st.toggle(
        "🔧 Advanced Mode",
        value=(get_ui_depth() == S.ADVANCED),
        key="ui_depth_toggle",
        help="Show technical detail (metrics, hashes, dataset ids, diagnostics). "
             "This only changes what you SEE — never what the system can do.")
    st.session_state[_DEPTH_KEY] = S.ADVANCED if advanced else S.SIMPLE
    if not advanced:
        st.caption("Simple Mode — plain language. Risk & safety info is always shown.")


# ── read-only state gathering (fail safe: unknown ⇒ conservative, never falsely READY)

def gather_state() -> dict:
    """Compose the plain Home state from existing read-only sources. Every lookup is
    defensive: if a source is unavailable, the safe assumption is made (data NOT ok),
    so a green 'Ready' is never shown on missing information."""
    state: dict = {"mode": "PAPER", "data_ok": False, "data_stale": False,
                   "market_open": False, "autopilot_armed": False, "safety_stop": False,
                   "trades_allowed": 0, "trades_used": 0, "open_positions": 0,
                   "attention": [], "live_enabled": False}
    try:
        from execution import autopilot as ap
        s = ap.get_status()                       # PURE READ — no arming, no orders
        state["mode"] = (s.get("mode") or "PAPER").upper()
        state["autopilot_armed"] = bool(s.get("armed"))
        state["trades_allowed"] = int(s.get("max_trades_per_day", 0) or 0)
        state["trades_used"] = int(s.get("trades_today_count", 0) or 0)
        state["open_positions"] = len(s.get("open_trades", []) or [])
        state["live_enabled"] = bool(s.get("live_enabled", False))
        state["safety_stop"] = "circuit breaker" in (s.get("disarmed_reason") or "").lower()
    except Exception:
        state["attention"] = ["Could not read autopilot status."]
    try:
        from core.market_session import in_market_open
        state["market_open"] = bool(in_market_open())
    except Exception:
        pass
    try:
        from data.bhavcopy_store import is_ready
        state["data_ok"] = bool(is_ready())
    except Exception:
        state["data_ok"] = False
    return state


# ── render helpers ─────────────────────────────────────────────────────────────

def _card(title: str, body: str, tone: str = "info") -> None:
    colour = {"good": "#00d4a0", "warn": "#f59e0b", "bad": "#ff4b4b",
              "info": "#38bdf8"}.get(tone, "#38bdf8")
    st.markdown(
        f"<div style='background:{colour}11;border:1px solid {colour}44;"
        f"border-left:4px solid {colour};border-radius:12px;padding:.85rem 1.1rem;"
        f"margin:.4rem 0'><div style='font-size:1.05rem;font-weight:700;color:{colour}'>"
        f"{title}</div><div style='color:#c8cfe0;font-size:.9rem;margin-top:.25rem'>"
        f"{body}</div></div>", unsafe_allow_html=True)


def render_next_best_action(state: dict | None = None) -> None:
    state = state or gather_state()
    _card("👉 Next best action", S.next_best_action(state), "info")


def render_home(state: dict | None = None) -> None:
    """The Simple Mode Home — the whole 'where am I / is it safe' picture."""
    state = state or gather_state()
    h = S.home_status(state)
    a = h["answers"]
    hs = h["headline_status"]
    _card(f"{hs['icon']} {hs['label']}", hs["plain"], hs["tone"])
    render_next_best_action(state)
    st.markdown("#### What's going on right now")
    for line in (a["data"], a["market"], a["mode"]["label"] + " — " + a["mode"]["money"],
                 a["autopilot"], a["trading_allowed"], a["safety_stop"], a["trades"],
                 a["open_positions"], a["live"]):
        st.markdown(f"- {line}")
    if a["attention"] and a["attention"] != ["Nothing is waiting for you."]:
        st.markdown("#### Waiting for you")
        for item in a["attention"]:
            st.markdown(f"- ⚠️ {item}")
    render_daily_checklist()


def render_daily_checklist() -> None:
    st.markdown("#### Daily checklist")
    for item in S.DAILY_CHECKLIST:
        st.checkbox(item["label"], key=f"chk_{item['key']}")


def render_mode_explainer() -> None:
    st.markdown("### The three modes")
    for key in ("RESEARCH", "PAPER", "LIVE"):
        m = S.MODES[key]
        _card(m["label"], f"{m['meaning']}  \n**Money:** {m['money']}",
              "good" if key == "PAPER" else "info")


def render_onboarding() -> None:
    """First-run tour — skippable and reopenable. Completing it enables nothing."""
    st.markdown("## Welcome to QuantTerm")
    for i, step in enumerate(S.ONBOARDING_STEPS, 1):
        _card(f"{i}. {step['title']}", step["body"],
              "warn" if "cannot promise" in step["title"].lower() else "info")
    st.info("Finishing this tour does NOT switch on real-money trading. "
            "You stay in safe practice mode.")
    if st.button("Start the PAPER practice walkthrough →", key="onb_to_walk"):
        st.session_state["sidebar_nav"] = "Practice Walkthrough"
        st.rerun()


def render_walkthrough() -> None:
    """Safe interactive story on FICTIONAL data. Calls no broker, Telegram or live
    service — it renders `core.simple_language.WALKTHROUGH_STEPS` only."""
    st.markdown("## PAPER practice walkthrough")
    st.caption("A made-up example for learning. No real money, no orders, no broker.")
    for i, step in enumerate(S.WALKTHROUGH_STEPS, 1):
        with st.expander(f"{i}. {step['title']}", expanded=(i == 1)):
            st.write(step["body"])
    _card("Remember", S.GOOD_DAY, "good")


def render_decision(code: str, technical: str | None = None) -> None:
    """Plain 5-part explanation of a system decision. In Advanced Mode the exact code
    and numbers are also shown."""
    d = S.decision_for(code)
    _card(d["decision"], d["main_reason"],
          "warn" if d["decision"] in ("Skipped", "Waiting", "Paper trade blocked") else "info")
    if d["supporting_reasons"]:
        for r in d["supporting_reasons"]:
            st.markdown(f"- {r}")
    if d["risk"]:
        st.markdown(f"**What can still go wrong:** {d['risk']}")
    if d["next_step"]:
        st.markdown(f"**Next step:** {d['next_step']}")
    if not is_simple() and technical:
        st.caption(f"Technical: {technical}")


def render_data_unavailable(operator_step: str = "", advanced_detail: str = "") -> None:
    """Honest empty-state — never a stack trace or blank page."""
    p = S.data_unavailable_panel(operator_step, advanced_detail)
    _card("What happened", p["what_happened"], "warn")
    st.markdown(f"**What it means:** {p['what_it_means']}")
    st.markdown(f"**Current status:** `{p['current_status']}`")
    st.markdown("**What still works:**")
    for w in p["what_still_works"]:
        st.markdown(f"- {w}")
    st.markdown(f"**What to do next:** {p['what_to_do_next']}")
    if not is_simple() and p["advanced_detail"]:
        st.caption(f"Advanced: {p['advanced_detail']}")


def render_page_help(page: str) -> None:
    """Contextual 'What is this page?' panel for any major page."""
    q = S.page_help(page)
    with st.expander("❔ What is this page?", expanded=False):
        st.markdown(f"**What is this?** {q['what_is_this']}")
        st.markdown(f"**Why does it matter?** {q['why_it_matters']}")
        st.markdown(f"**What should I do?** {q['what_should_i_do']}")
        st.markdown(f"**What happens next?** {q['what_will_happen']}")
        st.markdown(f"**Then?** {q['what_next']}")
        st.caption("Full guide: docs/user-guide/ · Glossary: docs/user-guide/GLOSSARY.md")


def confirm_safety_change(setting, current, proposed, effect, affects, reverse,
                          max_consequence=None, key: str = "safety_confirm") -> bool:
    """Render a SPECIFIC safety confirmation (never a generic 'Are you sure?'). Returns
    True only when the user explicitly confirms. The caller still enforces the real
    permission underneath — this is presentation, not a new authority."""
    c = S.safety_confirmation(setting, current, proposed, effect, affects, reverse,
                              max_consequence)
    st.warning(c["message"])
    if c["max_consequence"]:
        st.markdown(f"**Most this can cost:** {c['max_consequence']}")
    st.caption(f"To reverse: {c['reverse']}")
    return st.checkbox("I understand this change.", key=key)


def render_glossary(search: str = "") -> None:
    st.markdown("### Plain-language glossary")
    q = (search or "").strip().lower()
    for term, plain in sorted(S.GLOSSARY.items()):
        if not q or q in term.lower() or q in plain.lower():
            st.markdown(f"- **{term}** → {plain}")
