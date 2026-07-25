"""
🧠 QuantTerm Brain — the one read that composes every subsystem.

The system already has the parts: a whole-market scanner, evidence-calibrated
signals, a live edge that learns from real outcomes, a regime engine, a
portfolio-risk gatekeeper, an autopilot, and daemon health. This is the
conductor: each cycle it reads all of them and produces ONE authoritative
picture —

  • posture      — AGGRESSIVE / NORMAL / DEFENSIVE / STAND_ASIDE, derived from
                   regime × edge-trend × expectancy × book risk
  • verdict_line — one plain sentence: kya karna hai aaj
  • vitals       — the numbers that matter, in one place
  • directives   — a prioritised, plain-language to-do list (critical first)

The Brain is READ-ONLY. It never places a trade or changes a limit — the
autopilot and the ticket own execution, behind their own gates. The Brain
tells you (and the daemons) what the whole board says; it doesn't seize the
wheel. Every probe is wrapped, so a dead subsystem degrades the picture
instead of blanking it.
"""
from __future__ import annotations

from logger import get_logger

log = get_logger(__name__)

# Regimes where fresh breakout longs have the worst odds (mirrors autopilot).
_BAD_REGIMES = {"DISTRIBUTION", "TRENDING_BEAR", "BEAR"}
_GOOD_REGIMES = {"TRENDING_BULL", "EXPANSION"}

_POSTURE_META = {
    "AGGRESSIVE":  ("#00d4a0", "🟢 GREEN LIGHT"),
    "NORMAL":      ("#38bdf8", "🔵 NORMAL"),
    "DEFENSIVE":   ("#f59e0b", "🟠 DEFENSIVE"),
    "STAND_ASIDE": ("#ff4b4b", "🔴 STAND ASIDE"),
}


def posture_meta(posture: str) -> tuple[str, str]:
    return _POSTURE_META.get(posture, ("#8892a4", posture))


# ── Pure decision core (fully testable, no I/O) ───────────────────────────────

def decide_posture(regime: str, edge_trend: str, expectancy_r: float,
                   book_verdict: str, closed: int,
                   breadth: str = "", macro_risk_off: bool = False) -> tuple[str, str]:
    """The whole board → one posture + one-line reason.

    Precedence is SURVIVAL-FIRST: an over-risked book stands us aside no matter
    how good the tape looks; a bad tape or rotting edge keeps us defensive; only
    when tape, edge and book all align do we lean in. Breadth AND macro-news
    risk are DEMOTE-ONLY extra evidence: a NARROW market or a RISK_OFF news tape
    (tariffs/crude/geopolitics driving the day) can hold us back from AGGRESSIVE
    (chase mat karo jab macro haawi ho), never push us forward.
    """
    reg = (regime or "").upper()
    br = (breadth or "").upper()
    # 1) Book risk overrides everything — protect capital before opportunity.
    if book_verdict == "DANGER":
        return "STAND_ASIDE", ("Portfolio open-risk DANGER — capital pehle. "
                               "Naye trade nahi jab tak book halka na ho.")
    # 2) Actively hostile tape with a negative edge — don't fight it.
    if reg in _BAD_REGIMES and expectancy_r < 0:
        return "STAND_ASIDE", (f"Tape {reg} + edge negative "
                               f"({expectancy_r:+.2f}R) — aaj naye longs off.")
    # 3) Any single caution flag → defensive (smaller, only A+ setups).
    if (reg in _BAD_REGIMES or edge_trend == "decaying"
            or expectancy_r < 0 or book_verdict == "CAUTION"):
        why = ("tape kamzor" if reg in _BAD_REGIMES else
               "edge decaying" if edge_trend == "decaying" else
               "edge negative" if expectancy_r < 0 else "book caution")
        return "DEFENSIVE", (f"{why} — chhota size, sirf A+ setups, "
                             f"stops tight.")
    # 4) Everything aligned AND enough evidence → lean in. Breadth veto:
    # NARROW internals mein lean-in nahi (demote to NORMAL, not DEFENSIVE —
    # baaki board healthy hai, bas extra size ka license nahi).
    if (reg in _GOOD_REGIMES and edge_trend != "decaying"
            and expectancy_r >= 0.15 and book_verdict == "OK" and closed >= 30):
        if br == "NARROW":
            return "NORMAL", ("Tape/edge/book strong PAR market andar se "
                              "narrow — index ke peechhe kamzori. Normal size, "
                              "lean-in nahi.")
        if macro_risk_off:
            return "NORMAL", ("Tape/edge/book strong PAR macro news risk-off "
                              "(tariff/crude/geopolitics haawi) — normal size, "
                              "aaj extra aggression nahi.")
        return "AGGRESSIVE", ("Tape + edge + book teeno strong — normal-plus "
                              "size, winners ko chalne do.")
    # 5) Otherwise steady state.
    return "NORMAL", "Kaam theek — normal size, discipline banaye rakho."


def build_directives(v: dict) -> list[dict]:
    """Vitals → prioritised, plain-language to-do list. Each item is
    {severity, text}; sorted critical → good. Bounded and de-duplicated."""
    out: list[dict] = []

    def add(sev: str, text: str) -> None:
        out.append({"severity": sev, "text": text})

    # critical — survival + plumbing
    if v.get("book_verdict") == "DANGER":
        add("critical", f"🛑 Portfolio open-risk {v.get('open_risk_pct', 0):.1f}% "
                        f"(DANGER). Ek position book karke risk halka karo — "
                        f"naya trade avoid.")
    for d in v.get("dead_daemons", []):
        add("critical", f"🔧 Daemon '{d}' DEAD — scan/alerts ruk sakte hain. "
                        f"App/host restart dekho.")

    # warn — edge + tape + freshness
    if v.get("edge_trend") == "decaying":
        add("warn", f"📉 Edge decaying (recent {v.get('recent_r', 0):+.2f}R vs "
                    f"lifetime {v.get('expectancy_r', 0):+.2f}R) — size mat "
                    f"badhao, filters ko kaam karne do.")
    if (v.get("regime") or "").upper() in _BAD_REGIMES:
        add("warn", f"🌡️ Tape {v['regime']} — fresh breakout longs yahan "
                    f"historically fail. Autopilot bhi yeh block karta hai.")
    if v.get("stale_min") and v["stale_min"] > 45:
        add("warn", f"⏱️ Setups {v['stale_min']} min purane — refresh/scan "
                    f"chalao taaki decisions taaza data pe hon.")
    if v.get("expectancy_r", 0) < 0 and v.get("closed", 0) >= 30:
        add("warn", f"⚠️ System expectancy abhi negative "
                    f"({v['expectancy_r']:+.2f}R) — real size mat badhao.")
    # breadth: index ke peechhe ki kamzori — narrow tape = breakout graveyard
    if v.get("breadth") == "NARROW":
        _bl = (v.get("breadth_line")
               or "Market andar se narrow hai — breakouts ko hawa nahi milegi.")
        add("warn", f"📊 {_bl}")
    # 🌍 macro news radar — tape aaj news ke haath mein hai kya?
    if v.get("macro_mood") == "RISK_OFF":
        _mt = v.get("macro_themes") or []
        _hit = ", ".join((_mt[0].get("sectors_hit") or [])[:3]) if _mt else ""
        add("warn", f"🌍 Macro RISK-OFF: {v.get('macro_note', '')}."
                    + (f" Sabse zyada asar: {_hit}." if _hit else "")
                    + " Fresh longs pe careful, size mat badhao.")
    elif v.get("macro_mood") == "CAUTIOUS":
        add("info", f"🌍 Macro watch: {v.get('macro_note', '')} — news-driven "
                    f"tape, setups ke saath macro bhi dekho.")
    # options market ka vote — context, order nahi
    if v.get("options_bias") in ("BULLISH", "BEARISH"):
        add("info", f"🎯 Options positioning: {v.get('options_note', '')} · "
                    f"max-pain ₹{v.get('max_pain', 0):,.0f}.")
    # 🏦 institutional flows — bade paise ka footprint
    _fb = v.get("flows_bias", "")
    if _fb == "DISTRIBUTION":
        add("warn", f"🏦 {v.get('flows_note', '')} ")
    elif _fb in ("SUPPORTIVE", "FII_SELLING_DII_ABSORBING"):
        add("info", f"🏦 {v.get('flows_note', '')}")
    _bulk = v.get("bulk_buys") or []
    if _bulk:
        add("info", f"🏦 Bulk-deal buying aaj: {', '.join(_bulk[:5])}"
                    + (f" +{len(_bulk) - 5} aur" if len(_bulk) > 5 else "")
                    + " — bade paise in naamon mein ghuse.")
    # 📉 concept-drift — signals whose measured edge is decaying / strengthening
    # (change-point located + permutation-confirmed by the Research OS, so this
    # only fires on a real shift, never noise). Decay warns, growth informs.
    for dd in (v.get("drift_directives") or [])[:2]:
        add(dd.get("severity", "info"), dd.get("text", ""))
    # 🎯 Confidence Ledger — buckets where our own probabilities are miscalibrated
    for cd in (v.get("calibration_directives") or [])[:1]:
        add(cd.get("severity", "info"), cd.get("text", ""))
    # ⚖️ Counterfactual gate attribution — a filter that's leaking money
    for gd in (v.get("gate_directives") or [])[:1]:
        add(gd.get("severity", "info"), gd.get("text", ""))
    # 🗂️ Edge Timeline — durable signal character from its drift HISTORY: retire
    # the dead, respect the cyclical (don't panic-kill an edge that always comes
    # back). Complements the momentary drift read above.
    for td in (v.get("timeline_directives") or [])[:1]:
        add(td.get("severity", "info"), td.get("text", ""))
    # 📚 Knowledge Base — a validated belief whose edge is now decaying (ACTIVE→
    # WATCH) or one just retired. The institution questioning its own knowledge.
    for bd in (v.get("belief_directives") or [])[:1]:
        add(bd.get("severity", "info"), bd.get("text", ""))
    # 🕳️ Non-event learning — a rejection reason the control group says is too
    # conservative (costing winners). A gate to loosen, from what we DIDN'T trade.
    for rd in (v.get("rejection_directives") or [])[:1]:
        add(rd.get("severity", "info"), rd.get("text", ""))
    # correlation lens: N positions that move together = fewer real bets
    if (v.get("corr_positions", 0) >= 2
            and v.get("corr_bets", 0) < v["corr_positions"]):
        big = v.get("corr_biggest") or []
        add("warn", f"🧩 {v['corr_positions']} positions, par correlation ke "
                    f"hisaab se sirf {v['corr_bets']} asli bets — "
                    f"{'/'.join(big)} ek saath move karte hain. Ek girega toh "
                    f"sab girenge; naya trade isi cluster mein mat jodo.")

    # info — opportunity / nudges
    swap = v.get("rotation_swap")
    if swap:
        add("info", f"🔄 Opportunity cost: {swap['note']}")
    if not v.get("autopilot_armed") and v.get("n_high_conviction", 0) > 0:
        add("info", f"🤖 Autopilot off, par {v['n_high_conviction']} high-"
                    f"conviction setups. Arm karo (paper) ya manually dekho.")
    if v.get("autopilot_armed") and v.get("day_pnl") is not None:
        _c = "🟢" if v["day_pnl"] >= 0 else "🔴"
        add("info", f"{_c} Autopilot aaj {v.get('cur', '₹')}{v['day_pnl']:+,.0f} "
                    f"({v.get('trades_today', 0)} trades). Discipline chal rahi.")

    # good — green light
    if v.get("posture") == "AGGRESSIVE" and v.get("top_pick"):
        tp = v["top_pick"]
        _ev_bit = (f" · EV {tp['ev_pct']:+.1f}% ({tp.get('p_win', 0):.0f}% win)"
                   if tp.get("ev_pct") is not None else "")
        add("good", f"✅ Green light. Aaj ka top pick: {tp['symbol']} — "
                    f"entry {v.get('cur', '₹')}{tp.get('entry', 0):,.1f} · "
                    f"stop {v.get('cur', '₹')}{tp.get('stop', 0):,.1f} · "
                    f"target {v.get('cur', '₹')}{tp.get('target', 0):,.1f}"
                    f"{_ev_bit}.")

    if not out:
        add("info", "Sab shaant — koi urgent action nahi. Normal routine.")

    order = {"critical": 0, "warn": 1, "info": 2, "good": 3}
    out.sort(key=lambda d: order.get(d["severity"], 9))
    return out[:6]


# ── Probes (each wrapped — a dead subsystem degrades, never blanks) ───────────

def _probe_regime() -> str:
    try:
        from core.regime_engine import compute_regime
        return str(getattr(compute_regime(), "market_regime", "") or "")
    except Exception:
        return ""


def _probe_edge(capital: float) -> dict:
    try:
        from reports.verdict_dashboard import build_equity_curve
        return build_equity_curve(start_capital=capital).get("stats", {})
    except Exception:
        return {}


def _probe_setups(market: str) -> tuple[list[dict], float]:
    try:
        if market == "US":
            from scan.us_scanner import get_us_results
            res, ts, _st = get_us_results()
            return res, ts
        from scan.auto_scan import get_results
        res, _n, ts, _st = get_results()
        return res, ts
    except Exception:
        return [], 0.0


def _probe_book() -> dict:
    try:
        from risk.portfolio_risk import portfolio_risk_report
        return portfolio_risk_report()
    except Exception:
        return {}


def _probe_autopilot(market: str) -> dict:
    try:
        if market == "US":
            import execution.us_autopilot as ap
        else:
            import execution.autopilot as ap
        s = ap.get_status()
        day = None
        try:
            day = ap.pnl_snapshot().get("day_pnl")
        except Exception:
            day = None
        return {"armed": bool(s.get("armed")),
                "trades_today": int(s.get("trades_today_count", 0)),
                "day_pnl": day}
    except Exception:
        return {}


def _probe_rotation(market: str) -> dict:
    """Portfolio intelligence (NSE book only) — opportunity-cost advice."""
    if market == "US":
        return {}
    try:
        from core.portfolio_intel import daily_review
        return daily_review() or {}
    except Exception:
        return {}


def _probe_breadth(market: str) -> dict:
    """Full-market internals (advance/decline, % above DMAs) — index ke
    peechhe ki sachchai, apne hi scan cache se muft."""
    if market == "US":
        return {}
    try:
        from scan.auto_scan import get_breadth
        return get_breadth() or {}
    except Exception:
        return {}


_macro_cache = {"ts": 0.0, "data": {}}


def _probe_macro(market: str) -> dict:
    """🌍 Macro news radar — the market-MOVING themes (tariffs, crude, rates,
    geopolitics) from the news stream the system already pulls. Cached ~10 min
    (RSS calls aren't free). Fail-open: news down → empty, never blocks the
    Brain. A RISK_OFF read leans the posture careful — demote-only, like
    breadth NARROW — it never forces a trade, only holds back aggression."""
    if market == "US":
        return {}
    import time as _t
    if _t.time() - _macro_cache["ts"] < 600 and _macro_cache["data"]:
        return _macro_cache["data"]
    data: dict = {}
    try:
        from news.fetcher import NewsFetcher
        from core.macro_pulse import macro_pulse
        arts = [a.to_dict() for a in NewsFetcher().fetch_all(max_age_hours=12)]
        data = macro_pulse(arts)
    except Exception:
        data = {}
    _macro_cache.update(ts=_t.time(), data=data)
    return data


_drift_cache = {"ts": 0.0, "data": []}


def _probe_drift(market: str) -> list:
    """📉 Concept-drift directives — signals whose live edge is decaying or
    strengthening (Research OS: change-point + permutation-confirmed). Cached
    30 min (the permutation test isn't free and drift moves slowly). Fail-open:
    no log / error → [], never blocks the Brain."""
    if market == "US":
        return []
    import time as _t
    if _drift_cache["ts"] > 0 and _t.time() - _drift_cache["ts"] < 1800:
        return _drift_cache["data"]
    data: list = []
    try:
        from research.drift import drift_directives
        data = drift_directives()
    except Exception:
        data = []
    _drift_cache.update(ts=_t.time(), data=data)
    return data


_calib_cache = {"ts": 0.0, "data": []}


def _probe_calibration(market: str) -> list:
    """🎯 Confidence-Ledger directives — buckets where the system's own win-
    probabilities are systematically over/under-confident (harness FDR-gated).
    Cached 1h (calibration moves slowly). Fail-open."""
    if market == "US":
        return []
    import time as _t
    if _calib_cache["ts"] > 0 and _t.time() - _calib_cache["ts"] < 3600:
        return _calib_cache["data"]
    data: list = []
    try:
        from research.calibration import calibration_directives
        data = calibration_directives()
    except Exception:
        data = []
    _calib_cache.update(ts=_t.time(), data=data)
    return data


_gates_cache = {"ts": 0.0, "data": []}


def _probe_gates(market: str) -> list:
    """⚖️ Counterfactual gate-attribution directives — filters that are COSTING
    money (rejecting winners) surfaced from the rejected-trade ledger, FDR-gated.
    Cached 1h (moves slowly). Fail-open."""
    if market == "US":
        return []
    import time as _t
    if _gates_cache["ts"] > 0 and _t.time() - _gates_cache["ts"] < 3600:
        return _gates_cache["data"]
    data: list = []
    try:
        from research.counterfactual import gate_directives
        data = gate_directives()
    except Exception:
        data = []
    _gates_cache.update(ts=_t.time(), data=data)
    return data


_timeline_cache = {"ts": 0.0, "data": []}


def _probe_timeline(market: str) -> list:
    """🗂️ Edge-Timeline directives — the durable CHARACTER of a signal from its
    recorded drift history (distinct from the momentary drift read): retire the
    dead, respect the cyclical (recovers every dip — don't kill it). Cached 1h.
    Fail-open."""
    if market == "US":
        return []
    import time as _t
    if _timeline_cache["ts"] > 0 and _t.time() - _timeline_cache["ts"] < 3600:
        return _timeline_cache["data"]
    data: list = []
    try:
        from research.edge_timeline import timeline_directives
        data = timeline_directives()
    except Exception:
        data = []
    _timeline_cache.update(ts=_t.time(), data=data)
    return data


_beliefs_cache = {"ts": 0.0, "data": []}


def _probe_beliefs(market: str) -> list:
    """📚 Knowledge-Base directives — the institution's own validated beliefs
    that just changed character: an ACTIVE belief now on WATCH (its edge is
    decaying) or a freshly RETIRED one. The KB owns the science; the Brain only
    renders it. Cached 1h. Fail-open."""
    if market == "US":
        return []
    import time as _t
    if _beliefs_cache["ts"] > 0 and _t.time() - _beliefs_cache["ts"] < 3600:
        return _beliefs_cache["data"]
    data: list = []
    try:
        from research.knowledge import belief_directives
        data = belief_directives()
    except Exception:
        data = []
    _beliefs_cache.update(ts=_t.time(), data=data)
    return data


_rejection_cache = {"ts": 0.0, "data": []}


def _probe_rejections(market: str) -> list:
    """🕳️ Non-event directives — a rejection REASON the counterfactual control
    group now says is TOO CONSERVATIVE (passing more winners than it saves), so a
    gate to loosen. Harness-gated, demote-only. Cached 1h. Fail-open."""
    if market == "US":
        return []
    import time as _t
    if _rejection_cache["ts"] > 0 and _t.time() - _rejection_cache["ts"] < 3600:
        return _rejection_cache["data"]
    data: list = []
    try:
        from research.non_event import rejection_directives
        data = rejection_directives()
    except Exception:
        data = []
    _rejection_cache.update(ts=_t.time(), data=data)
    return data


def _probe_options(market: str) -> dict:
    """Index options positioning (PCR / max-pain) — options market ka vote.
    Off-hours/chain-fail → empty, koi claim nahi."""
    if market == "US":
        return {}
    try:
        from options.analytics import nifty_options_summary
        return nifty_options_summary() or {}
    except Exception:
        return {}


def _probe_flows(market: str) -> dict:
    """🏦 Institutional footprint: FII/DII net + bulk-deal buys (NSE, free).
    Bade paise ka context — evidence stream, gate nahi (jab tak edge
    measure na ho)."""
    if market == "US":
        return {}
    try:
        from data.institutional_flows import get_flows
        return get_flows() or {}
    except Exception:
        return {}


def _probe_correlation(market: str) -> dict:
    """Book through the correlation lens — positions vs real bets."""
    if market == "US":
        return {}
    try:
        from risk.correlation import book_correlation_report
        return book_correlation_report() or {}
    except Exception:
        return {}


def _probe_dead_daemons() -> list[str]:
    try:
        from core.health import pulse
        daemons = pulse().get("daemons", {})
        return [n for n, d in daemons.items()
                if (d.get("status") if isinstance(d, dict) else d) == "DEAD"]
    except Exception:
        return []


# ── The one call the UI/daemons use ───────────────────────────────────────────

def assess(market: str = "IN", capital: float = 100_000.0) -> dict:
    """Compose every subsystem into one authoritative read. READ-ONLY."""
    import time as _t
    cur = "$" if market == "US" else "₹"

    regime = _probe_regime() if market != "US" else ""
    edge = _probe_edge(capital)
    setups, ts = _probe_setups(market)
    book = _probe_book() if market != "US" else {}
    ap = _probe_autopilot(market)
    dead = _probe_dead_daemons()
    rot = _probe_rotation(market)
    corr = _probe_correlation(market)
    breadth = _probe_breadth(market)
    opts = _probe_options(market)
    flows = _probe_flows(market)
    macro = _probe_macro(market)
    drift = _probe_drift(market)
    calib = _probe_calibration(market)
    gates = _probe_gates(market)
    timeline = _probe_timeline(market)
    beliefs = _probe_beliefs(market)
    rejections = _probe_rejections(market)

    buys = [r for r in setups if r.get("verdict") in ("STRONG BUY", "BUY")]
    # EV-first cross-sectional ranking (north star): setups with a measured
    # expected value outrank point-scores; conviction stays the fallback.
    try:
        from scan.ev_engine import ev_rank_key
        buys.sort(key=ev_rank_key, reverse=True)
    except Exception:
        buys.sort(key=lambda r: float(r.get("conviction_rank")
                                      or r.get("score") or 0), reverse=True)
    top = buys[0] if buys else None
    n_hc = sum(1 for r in setups if r.get("high_conviction"))
    stale_min = int((_t.time() - ts) / 60) if ts else None

    expectancy_r = float(edge.get("expectancy_r", 0.0) or 0.0)
    edge_trend = edge.get("edge_trend", "stable")
    closed = int(edge.get("closed", 0) or 0)
    book_verdict = book.get("verdict", "OK")

    posture, reason = decide_posture(regime, edge_trend, expectancy_r,
                                     book_verdict, closed,
                                     breadth=breadth.get("verdict", ""),
                                     macro_risk_off=bool(macro.get("risk_off")))

    vitals = {
        "market": market, "cur": cur,
        "regime": regime or "UNKNOWN",
        "expectancy_r": round(expectancy_r, 2),
        "recent_r": float(edge.get("recent_avg_r", 0.0) or 0.0),
        "edge_trend": edge_trend,
        "drawdown_pct": float(edge.get("max_drawdown_pct", 0.0) or 0.0),
        "closed": closed,
        "n_buys": len(buys),
        "n_high_conviction": n_hc,
        "top_pick": ({"symbol": top["symbol"],
                      "entry": float(top.get("entry") or 0),
                      "stop": float(top.get("stop") or 0),
                      "target": float(top.get("target") or 0),
                      "ev_pct": top.get("ev_pct"),
                      "p_win": top.get("p_win")} if top else None),
        "stale_min": stale_min,
        "book_verdict": book_verdict,
        "open_risk_pct": float(book.get("open_risk_pct", 0.0) or 0.0),
        "n_positions": int(book.get("n_positions", 0) or 0),
        "autopilot_armed": ap.get("armed", False),
        "trades_today": ap.get("trades_today", 0),
        "day_pnl": ap.get("day_pnl"),
        "dead_daemons": dead,
        "portfolio_ev_pct": rot.get("portfolio_ev_pct"),
        "rotation_swap": rot.get("swap"),
        "corr_positions": corr.get("n_positions", 0),
        "corr_bets": corr.get("n_bets", 0),
        "corr_biggest": corr.get("biggest"),
        "breadth": breadth.get("verdict", ""),
        "breadth_line": breadth.get("line", ""),
        "pct_above_50": breadth.get("pct_above_50", 0.0),
        "options_bias": opts.get("bias", ""),
        "options_note": opts.get("note", ""),
        "max_pain": opts.get("max_pain", 0.0),
        "flows_bias": (flows.get("fii_dii") or {}).get("bias", ""),
        "flows_note": (flows.get("fii_dii") or {}).get("note", ""),
        "bulk_buys": flows.get("bulk_buys") or [],
        "macro_mood": macro.get("mood", ""),
        "macro_heat": macro.get("heat", 0),
        "macro_note": macro.get("note", ""),
        "macro_themes": macro.get("themes", []),
        "drift_directives": drift,
        "calibration_directives": calib,
        "gate_directives": gates,
        "timeline_directives": timeline,
        "belief_directives": beliefs,
        "rejection_directives": rejections,
        "posture": posture,
    }
    directives = build_directives(vitals)
    return {
        "posture": posture,
        "posture_reason": reason,
        "verdict_line": f"{posture_meta(posture)[1]} — {reason}",
        "vitals": vitals,
        "directives": directives,
    }


def briefing_telegram(market: str = "IN", capital: float = 100_000.0) -> str:
    """The Brain's verdict as a compact Telegram briefing (HTML). This is the
    one message that lets the user skip opening the app: posture, why, vitals,
    and the prioritised to-do — the whole board in a phone-sized read."""
    from datetime import datetime

    from core.market_clock import IST
    a = assess(market=market, capital=capital)
    v = a["vitals"]
    _, label = posture_meta(a["posture"])
    cur = v.get("cur", "₹")
    _tr = {"improving": "📈", "decaying": "📉", "stable": "➖"}.get(
        v.get("edge_trend"), "")
    ap_bit = "🤖 ARMED" if v.get("autopilot_armed") else "🤖 off"
    day = v.get("day_pnl")
    day_bit = f" ({cur}{day:+,.0f})" if day is not None else ""
    _br = v.get("breadth", "")
    br_bit = (f" · breadth <b>{_br}</b> ({v.get('pct_above_50', 0):.0f}%>50DMA)"
              if _br else "")
    _op = v.get("options_bias", "")
    op_bit = f" · options <b>{_op}</b>" if _op and _op != "NEUTRAL" else ""
    _mm = v.get("macro_mood", "")
    macro_bit = (f" · macro <b>{_mm}</b>" if _mm in ("RISK_OFF", "CAUTIOUS")
                 else "")
    lines = [
        f"🧠 <b>QuantTerm Brain</b> · {datetime.now(IST).strftime('%d %b')}",
        f"<b>{label}</b>",
        a["posture_reason"], "",
        (f"tape <b>{v.get('regime', '?')}</b> · edge "
         f"<b>{v.get('expectancy_r', 0):+.2f}R</b> {_tr}{br_bit}{op_bit}{macro_bit} · book "
         f"<b>{v.get('book_verdict', 'OK')}</b> ({v.get('open_risk_pct', 0):.1f}%) "
         f"· setups <b>{v.get('n_buys', 0)}</b> ({v.get('n_high_conviction', 0)} 🎯) "
         f"· {ap_bit}{day_bit}"), "",
        "<b>Aaj:</b>",
    ]
    lines += [f"• {d['text']}" for d in a["directives"]]
    lines.append("\n<i>Detail app ke Pulse tab par · ye read-only verdict hai</i>")
    return "\n".join(lines)
