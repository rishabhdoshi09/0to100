"""
🛰️ Control Room — one screen that answers "what is the system doing right now?"

The rest of QuantTerm runs a lot of machinery in the background (market scan, the research
brain, paper autonomy, alert daemons). This page makes that legible at a glance: system
vitals, the brain's live chain-of-thought, paper-autonomy state, and daemon health — so a
trader (or a buyer being shown the product) can see inside, not just trust a black box.

Read-only. It renders live state; it places no orders and changes no settings.
"""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from ui import theme as T


# ── small safe readers (every source wrapped — the page must never crash) ────────

def _market() -> tuple[str, str]:
    """(label, status) where status ∈ ok|warn|off. IST-explicit, matches app hours."""
    try:
        from core.market_clock import now_ist_naive
        now = now_ist_naive()
        t = now.strftime("%H:%M IST")
        if now.weekday() >= 5:
            return f"NSE closed · weekend · {t}", "off"
        hm = now.hour * 60 + now.minute
        if hm < 9 * 60 + 15:
            return f"NSE pre-market · {t}", "warn"
        if hm > 15 * 60 + 30:
            return f"NSE closed · {t}", "off"
        return f"NSE open · {t}", "ok"
    except Exception:
        return "Market status —", "off"


def _readiness() -> dict:
    try:
        from research.auto_research import canonical_readiness
        return canonical_readiness()
    except Exception as e:
        return {"color": "red", "can_run": False, "reasons": [str(e)]}


def _scan() -> dict:
    try:
        from scan.auto_scan import get_results
        results, universe, ts, status = get_results()
        age = (datetime.now().timestamp() - ts) if ts else -1
        return {"n": len(results), "universe": universe, "age_s": age, "status": status}
    except Exception:
        return {"n": 0, "universe": 0, "age_s": -1, "status": "—"}


def _pulse() -> dict:
    try:
        from core.health import pulse
        return pulse()
    except Exception:
        return {}


def _age(age_s: float) -> str:
    if age_s is None or age_s < 0:
        return "never"
    if age_s < 90:
        return f"{age_s:.0f}s ago"
    if age_s < 5400:
        return f"{age_s/60:.0f}m ago"
    return f"{age_s/3600:.1f}h ago"


# ── page ─────────────────────────────────────────────────────────────────────────

def render_control_room() -> None:
    _hero()
    _vitals()
    _brain_panel()
    _knowledge_panel()
    _daemons_panel()


def _hero() -> None:
    mlabel, mstatus = _market()
    try:
        from config.settings import Settings
        paper = getattr(Settings(), "sq_paper_trading", True)
    except Exception:
        paper = True
    mode_pill = T.pill("Paper", "off") if paper else T.pill("LIVE", "warn")
    st.markdown(
        "<div class='qt-card' style='display:flex;justify-content:space-between;"
        "align-items:center;flex-wrap:wrap;gap:.6rem;"
        "background:linear-gradient(135deg,rgba(56,189,248,0.08),rgba(45,212,191,0.04));"
        "border-color:rgba(56,189,248,0.2)'>"
        "<div><div class='qt-eyebrow'>Control Room</div>"
        "<div style='font-size:1.55rem;font-weight:800;letter-spacing:-.02em'>"
        "What the system is doing right now</div></div>"
        f"<div style='display:flex;gap:.45rem;flex-wrap:wrap'>"
        f"{T.pill(mlabel, mstatus)}{mode_pill}</div></div>",
        unsafe_allow_html=True)


def _vitals() -> None:
    p = _pulse()
    daemons = p.get("daemons", {})
    alive = sum(1 for d in daemons.values() if d.get("status") in ("OK", "SLOW"))
    total = len(daemons)
    r = _readiness()
    s = _scan()
    rmap = {"green": ("Research-ready", T.GREEN), "amber": ("Limited", T.AMBER),
            "red": ("Not ready", T.RED)}
    rlabel, rcol = rmap.get(r.get("color"), ("Unknown", T.DIM))

    # quote latency / source (best-effort)
    lat = p.get("latency", {})
    q = lat.get("quote_fetch", {})
    qtxt = f"{q['p50_ms']:.0f}ms p50" if q else "—"

    T.section("System vitals", eyebrow="Live")
    T.stat_grid([
        {"k": "Daemons alive", "v": f"{alive}/{total}" if total else "—",
         "d": "background workers", "color": (T.GREEN if total and alive == total
                                              else T.AMBER if alive else T.RED)},
        {"k": "Research data", "v": rlabel, "color": rcol,
         "d": (r.get("reasons") or ["ready"])[0][:34]},
        {"k": "Last market scan", "v": _age(s["age_s"]),
         "d": f"{s['n']} setups · {s['universe']} scanned"},
        {"k": "Quote latency", "v": qtxt, "d": "quote source health"},
    ], cols=4)


def _brain_panel() -> None:
    try:
        from research.auto_research import get_brain
        brain = get_brain()
    except Exception:
        return

    st_ = brain.state
    T.section("The Brain — live reasoning", eyebrow="Autonomous research",
              sub="Every idea it generates, judges, rejects or proposes — in the open.")

    pa = "ok" if st_.paper_autonomy else "off"
    T.pill_row([
        T.pill(f"{st_.cycles_run} cycles", "ok" if st_.cycles_run else "off"),
        T.pill(f"{getattr(st_,'days_grown',0)} days grown", "ok" if getattr(st_,'days_grown',0) else "off"),
        T.pill(f"{st_.total_proposals} parked", "ok" if st_.total_proposals else "off"),
        T.pill("Loop running" if st_.running else "Loop idle", "ok" if st_.running else "off"),
        T.pill(f"Paper autonomy {'ON' if st_.paper_autonomy else 'off'}", pa),
    ])

    # the live chain-of-thought — the literal "ander kya ho raha hai"
    try:
        entries = brain.thread.all()
    except Exception:
        entries = []
    if not entries:
        st.markdown("<div class='qt-card tight' style='color:var(--qt-muted)'>"
                    "The brain hasn't thought a cycle yet. Open <b>🧠 Research Brain</b> "
                    "and press <i>Think one cycle now</i> — or engage the loop.</div>",
                    unsafe_allow_html=True)
    else:
        rows = []
        for e in entries[-8:]:
            rows.append(f"<div class='qt-think {e.kind}'><div class='kind'>"
                        f"cycle {e.cycle} · {e.kind}</div>"
                        f"<div class='txt'>{_esc(e.text)}</div></div>")
        st.markdown("<div class='qt-card'>" + "".join(rows) + "</div>",
                    unsafe_allow_html=True)

    # paper book (only if engaged or something deployed)
    try:
        rep = brain.paper.performance_report()
    except Exception:
        rep = None
    if rep and (rep.get("engaged") or rep.get("deployed")):
        book = rep["book"]
        pnl = book["equity"] - book["capital"]
        T.section("Paper autonomy — simulated book", eyebrow="Hands-off")
        T.stat_grid([
            {"k": "Paper equity", "v": f"₹{book['equity']:,.0f}",
             "d": f"{pnl:+,.0f} vs start",
             "color": (T.GREEN if pnl > 0 else T.RED if pnl < 0 else T.TEXT)},
            {"k": "Deployed", "v": rep["deployed"], "d": "strategies live in paper"},
            {"k": "Active", "v": rep["active"], "d": "currently trading"},
            {"k": "Retired", "v": len(rep["retired"]), "d": "auto-dropped losers"},
        ], cols=4)
        # equity curve — real chart, not just a number
        curve = book.get("equity_curve") or []
        if len(curve) >= 2:
            st.markdown(_equity_svg(curve, book["capital"]), unsafe_allow_html=True)
        # recent autonomy actions (the decision journal — transparent + auditable)
        jrnl = rep.get("journal", [])
        if jrnl:
            items = []
            for j in reversed(jrnl[-6:]):
                act = j.get("action")
                col = T.GREEN if act == "DEPLOY" else T.RED
                extra = (f"bt {j.get('backtest_R', 0):+.2f}R" if act == "DEPLOY"
                         else str(j.get("reason", ""))[:52])
                items.append(
                    f"<div style='display:flex;gap:.5rem;padding:.28rem 0;font-size:.8rem'>"
                    f"<span style='color:{col};font-weight:700;font-family:JetBrains Mono,monospace;"
                    f"min-width:64px'>{act}</span>"
                    f"<span style='font-family:JetBrains Mono,monospace'>{j.get('strategy_id','')}</span>"
                    f"<span style='color:var(--qt-muted);flex:1;text-align:right'>{extra}</span></div>")
            st.markdown("<div class='qt-card tight'><div class='qt-eyebrow' "
                        "style='margin-bottom:.3rem'>Recent autonomy actions</div>"
                        + "".join(items) + "</div>", unsafe_allow_html=True)
        per = rep.get("per_strategy", {})
        if per:
            lines = []
            for sid, s2 in list(per.items())[:6]:
                col = T.GREEN if s2["expectancy_R"] > 0 else T.RED
                lines.append(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"padding:.3rem 0;border-bottom:1px solid var(--qt-border)'>"
                    f"<span style='font-family:JetBrains Mono,monospace;font-size:.8rem'>{sid}</span>"
                    f"<span style='font-size:.8rem;color:var(--qt-muted)'>{s2['n_trades']} trades · "
                    f"<b style='color:{col}'>{s2['expectancy_R']:+.2f}R</b> · "
                    f"₹{s2['net_pnl']:,.0f}</span></div>")
            st.markdown("<div class='qt-card tight'>" + "".join(lines) + "</div>",
                        unsafe_allow_html=True)


def _knowledge_panel() -> None:
    """What the system has learned holds up out-of-sample — trust per strategy family."""
    try:
        from research.auto_research import get_brain
        summary = get_brain().knowledge.summary()
    except Exception:
        return
    if not summary:
        return
    T.section("What it has learned", eyebrow="Backtest → forward test",
              sub="Trust rises when a family's edge survives forward testing on unseen days.")
    rows = []
    for fk in summary[:8]:
        trust = fk["trust"]
        col = T.GREEN if trust >= 0.6 else T.AMBER if trust >= 0.4 else T.RED
        bar = int(round(trust * 100))
        rows.append(
            f"<div style='display:flex;align-items:center;gap:.7rem;padding:.4rem 0;"
            f"border-bottom:1px solid var(--qt-border)'>"
            f"<span style='flex:1;font-size:.86rem'>{_fam(fk['family'])}</span>"
            f"<span style='font-size:.74rem;color:var(--qt-muted);font-family:JetBrains Mono,monospace'>"
            f"bt {fk['backtest_R']:+.2f}R · fwd {fk['forward_R']:+.2f}R</span>"
            f"<div style='width:88px;height:6px;border-radius:3px;background:rgba(255,255,255,.06)'>"
            f"<div style='width:{bar}%;height:6px;border-radius:3px;background:{col}'></div></div>"
            f"<span style='font-size:.74rem;color:{col};font-weight:700;width:34px;text-align:right'>{bar}%</span>"
            f"</div>")
    st.markdown("<div class='qt-card tight'>" + "".join(rows) + "</div>",
                unsafe_allow_html=True)


def _fam(s: str) -> str:
    return str(s).replace("_", " ").title()


def _daemons_panel() -> None:
    p = _pulse()
    daemons = p.get("daemons", {})
    if not daemons:
        return
    label = {"auto_scan": "Market scan", "autopilot": "Autopilot",
             "sniper": "Breakout sniper", "live_ticker": "Live stream",
             "telegram_listener": "Telegram buttons", "quotes": "Quote source"}
    dot = {"OK": "ok", "SLOW": "warn", "DEAD": "bad", "NEVER": "off"}
    T.section("Background workers", eyebrow="Daemons")
    rows = []
    for name, d in daemons.items():
        note = f" · {d['note']}" if d.get("note") else ""
        rows.append(
            f"<div style='display:flex;align-items:center;gap:.55rem;padding:.4rem 0;"
            f"border-bottom:1px solid var(--qt-border)'>"
            f"<span class='qt-dot {dot.get(d.get('status'),'off')}'></span>"
            f"<span style='flex:1;font-size:.86rem'>{label.get(name,name)}</span>"
            f"<span style='font-size:.74rem;color:var(--qt-muted)'>{_age(d.get('age_s',-1))}{note}</span>"
            f"</div>")
    st.markdown("<div class='qt-card tight'>" + "".join(rows) + "</div>",
                unsafe_allow_html=True)


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _equity_svg(curve: list, capital: float, w: int = 640, h: int = 60) -> str:
    """A compact inline equity sparkline with an area fill and an emphasised endpoint."""
    lo, hi = min(curve), max(curve)
    rng = (hi - lo) or 1.0
    n = len(curve)
    up = curve[-1] >= curve[0]
    col = T.GREEN if up else T.RED
    pts = [(i / (n - 1) * w, h - 4 - (v - lo) / rng * (h - 12)) for i, v in enumerate(curve)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"0,{h} " + line + f" {w},{h}"
    last = pts[-1]
    pnl = curve[-1] - capital
    return (
        f"<div class='qt-card tight' style='margin-bottom:.6rem'>"
        f"<div style='display:flex;justify-content:space-between;align-items:baseline'>"
        f"<span class='qt-eyebrow'>Paper equity curve</span>"
        f"<span style='font-family:JetBrains Mono,monospace;font-size:.8rem;color:{col}'>"
        f"{pnl:+,.0f}</span></div>"
        f"<svg viewBox='0 0 {w} {h}' preserveAspectRatio='none' "
        f"style='width:100%;height:{h}px;display:block;margin-top:.3rem'>"
        f"<polygon points='{area}' fill='{col}' fill-opacity='0.12'/>"
        f"<polyline points='{line}' fill='none' stroke='{col}' stroke-width='1.6'/>"
        f"<circle cx='{last[0]:.1f}' cy='{last[1]:.1f}' r='2.6' fill='{col}'/></svg></div>")
