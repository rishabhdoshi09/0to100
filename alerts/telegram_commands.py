"""
📱 Telegram commands — phone se system ko hukum, kahin se bhi.

"Take more trades" / "abhi koi trade mat lo" jaise controls ab chat mein:

  /status        — armed? trades aaj, P&L, posture — ek nazar
  /pause         — 🛑 abhi koi NAYA trade nahi (autopilot disarm; open
                   positions + GTT exits chalte rehte hain)
  /resume        — 🟢 wapas chalu (SIRF paper — LIVE kabhi Telegram se
                   arm nahi hota, invariant #4)
  /aggressive    — zyada trades (10/day tak, wider net)
  /balanced      — default discipline
  /conservative  — sirf A+ setups, kam trades
  /book 1500     — har trade ₹1500 profit pe book (0 = off)
  /funnel        — aaj kitne dekhe/liye/kyun reject — poora hisaab
  /brain         — abhi ka Brain verdict on demand
  /help          — ye list

Suraksha: chat-id guard telegram_actions mein hai (sirf tumhara chat);
yahan bhi LIVE-arming hamesha refuse hoti hai. Har command ka jawab
seedha, chhota, mobile-sized. Router pure hai — poora testable.
"""
from __future__ import annotations

from logger import get_logger

log = get_logger(__name__)


def _status() -> str:
    from execution.autopilot import get_status, pnl_snapshot
    s = get_status()
    armed = "🟢 ARMED · " + s.get("mode", "PAPER") if s.get("armed") else "🔴 OFF"
    day = None
    try:
        day = pnl_snapshot().get("day_pnl")
    except Exception:
        pass
    day_bit = f"\nAaj ka P&L: ₹{day:+,.0f}" if day is not None else ""
    # 🎯 machine ka hisaab — target vs booked (1+1=2)
    try:
        from execution.autopilot import daily_scoreboard
        sb = daily_scoreboard()
        if sb["target"] > 0:
            day_bit += (f"\n🎯 <b>₹{sb['booked_net']:,.0f} / "
                        f"₹{sb['target']:,.0f}</b> booked ({sb['booked_n']} "
                        f"trades, {sb['slots']} slots baaki)")
        day_bit += f"\n{sb['line']}"
    except Exception:
        pass
    posture = ""
    try:
        # cached posture (5-min) — /status hamesha turant jawab de; fresh
        # full-board assess sirf /brain par (jab user ne khud maanga ho)
        from core.brain import posture_meta
        from execution.autopilot import _brain_posture
        p, _why = _brain_posture()
        posture = f"\nBrain: {posture_meta(p)[1]}"
    except Exception:
        pass
    return (f"🤖 <b>Autopilot</b>: {armed}\n"
            f"Trades aaj: {s.get('trades_today_count', 0)}/"
            f"{s.get('max_trades_per_day', 0)} · "
            f"Preset: {s.get('preset', 'Balanced')}\n"
            f"Pool: ₹{s.get('pool', 0):,.0f} · "
            f"Book: ₹{s.get('profit_book_rupees', 0):,.0f}"
            f"{day_bit}{posture}")


def _pause() -> str:
    from execution.autopilot import disarm, get_status
    if not get_status().get("armed"):
        return "🔴 Already OFF — koi naya trade nahi lega."
    disarm("telegram /pause")
    return ("🛑 <b>PAUSED</b> — ab koi NAYA trade nahi.\n"
            "Open positions + GTT exits chalte rahenge.\n"
            "Wapas: /resume")


def _resume() -> str:
    from execution.autopilot import arm, get_status
    s = get_status()
    if s.get("armed"):
        return f"🟢 Already ARMED ({s.get('mode')})."
    if s.get("mode") == "LIVE":
        return ("⛔ LIVE arming Telegram se KABHI nahi hoti (safety "
                "invariant). App kholo → Autopilot → phrase type karo.")
    ok, msg = arm()
    return f"🟢 <b>ARMED — PAPER</b>. Agla scan se trades shuru." if ok \
        else f"❌ {msg}"


def _preset(name: str) -> str:
    from execution.autopilot import apply_preset, get_status
    ok, msg = apply_preset(name)
    if not ok:
        return f"❌ {msg}"
    s = get_status()
    return (f"🎛️ <b>{name}</b> set — max {s.get('max_trades_per_day')}/day, "
            f"{s.get('max_open_positions')} open, "
            f"min score {s.get('min_score'):.0f}."
            + ("" if s.get("armed") else "\n(Autopilot OFF hai — /resume)"))


def _book(arg: str) -> str:
    try:
        amt = float(arg)
    except Exception:
        return "Aise: /book 1500  (ya /book 0 = off)"
    from execution.autopilot import set_config, get_status
    set_config(profit_book_rupees=amt)
    val = get_status().get("profit_book_rupees", 0)
    return (f"💰 Profit-book: <b>₹{val:,.0f}</b>/trade"
            + (" — OFF" if val <= 0 else
               " — is level pe PAPER khud book, LIVE pe turant alert."))


def _trade_now() -> str:
    """📈 'Abhi ek trade lo' — next 15-min scan ka wait nahi. Store ka
    best untraded BUY (prime/EV-ranked) autopilot ke SAARE gates se guzaar
    ke place karne ki koshish. Gates fail hue toh imaandaar wajah batata
    hai — force nahi, sirf timing manual."""
    from execution.autopilot import get_status, consider
    s = get_status()
    if not s.get("armed"):
        return ("🔴 Autopilot OFF hai — pehle /resume karo, phir /trade.")
    if s.get("mode") == "LIVE":
        return ("⛔ /trade sirf PAPER mein. LIVE trades app ke ticket se "
                "(safety invariant).")
    try:
        from scan.auto_scan import get_results
        from scan.ev_engine import ev_rank_key
        results, _n, _ts, _st = get_results()
    except Exception:
        return "Scan store abhi khali — thodi der mein /trade dobara."
    buys = [r for r in results if r.get("verdict") in ("STRONG BUY", "BUY")]
    if not buys:
        return "Abhi koi BUY setup store mein nahi. /status se dekho."
    # prime first, phir conservative-EV/conviction
    buys.sort(key=lambda r: (bool(r.get("prime")), ev_rank_key(r)),
              reverse=True)
    for r in buys[:8]:                          # top few try — pehla jo pass ho
        placed = consider(
            symbol=r["symbol"],
            entry=float(r.get("entry") or r.get("price") or 0),
            stop=float(r.get("stop") or 0), score=float(r.get("score") or 0),
            edge=r.get("edge_r"), sector=r.get("sector") or "",
            source="manual", ev_pct=r.get("ev_pct"), p_win=r.get("p_win"),
            ev_conf=r.get("ev_conf"), grade=str(r.get("breakout_grade") or ""))
        if placed:
            return (f"✅ Trade liya: <b>{r['symbol']}</b> — gates paar, order "
                    f"lag gaya. /status se dekho.")
    # koi pass nahi hua → funnel se wajah
    return ("⚠️ Best setups gates pe atke (limit/sector/regime/breadth). "
            "Wajah: /funnel · zyada trades chahiye toh /aggressive.")


def _funnel() -> str:
    from execution.autopilot import reject_funnel
    f = reject_funnel()
    n, rejects = f.get("considered", 0), f.get("rejects", {})
    if not n:
        return ("📊 Aaj abhi koi candidate evaluate nahi hua.\n"
                "(Autopilot OFF tha ya scan ab tak setups nahi laya — /status)")
    taken = max(0, n - sum(rejects.values()))
    lines = [f"📊 Aaj: <b>{n}</b> dekhe → <b>{taken}</b> liye"]
    for cat, c in sorted(rejects.items(), key=lambda x: -x[1])[:6]:
        lines.append(f"• {cat} — {c}")
    return "\n".join(lines)


def _brain() -> str:
    from core.brain import briefing_telegram
    return briefing_telegram("IN")


_HELP = ("📱 <b>Commands</b>\n"
         "/status — ek nazar sab\n"
         "/trade — 📈 ABHI ek trade lo (best setup, gates ke saath)\n"
         "/pause — 🛑 naye trades band\n"
         "/resume — 🟢 wapas chalu (paper)\n"
         "/aggressive · /balanced · /conservative — kitne trades\n"
         "/book 1500 — NET ₹ profit pe auto-book (charges ke baad)\n"
         "/funnel — aaj ka hisaab (kyun kam trades)\n"
         "/brain — abhi ka verdict")


def handle_command(text: str) -> str | None:
    """Router. None = not a command (ignore silently). Kabhi raise nahi
    karta — listener ko girana mana hai."""
    try:
        parts = (text or "").strip().split()
        if not parts or not parts[0].startswith("/"):
            return None
        cmd = parts[0].lower().split("@")[0]     # /status@BotName bhi chale
        arg = parts[1] if len(parts) > 1 else ""
        if cmd == "/status":
            return _status()
        if cmd in ("/pause", "/stop"):
            return _pause()
        if cmd in ("/resume", "/start_trading"):
            return _resume()
        if cmd in ("/trade", "/buy"):
            return _trade_now()
        if cmd == "/aggressive":
            return _preset("Aggressive")
        if cmd == "/balanced":
            return _preset("Balanced")
        if cmd == "/conservative":
            return _preset("Conservative")
        if cmd == "/book":
            return _book(arg)
        if cmd == "/funnel":
            return _funnel()
        if cmd == "/brain":
            return _brain()
        if cmd in ("/help", "/start"):
            return _HELP
        return _HELP                              # unknown → help
    except Exception as exc:
        log.warning("telegram_command_failed", cmd=text[:30], error=str(exc))
        return "❌ Command fail hua — logs dekho ya /status try karo."
