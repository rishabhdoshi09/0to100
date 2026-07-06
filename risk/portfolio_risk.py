"""
Portfolio Risk — positions alag-alag nahi, EK portfolio hain.

The 1%% rule protects each trade; this protects the account:
  - Total open risk: agar SAARE stops aaj hit ho jayein, kitna doobega
  - Deployment: capital ka kitna % laga hua hai
  - Sector concentration: 2+ positions in one sector = ek hi bet
  - Position count vs max_open_positions

Verdicts:
  OK      — sab limits ke andar
  CAUTION — open risk > 3%% of capital, ya ek sector mein 2 positions
  DANGER  — open risk > 5%%, ya max positions cross, ya 3+ same sector

check_new_trade() runs the same math BEFORE a new trade is placed so
the ticket can warn "is trade ke baad total risk X%% ho jayega".
"""
from __future__ import annotations

from logger import get_logger

log = get_logger(__name__)


def _capital() -> float:
    try:
        import streamlit as st
        cap = st.session_state.get("user_capital")
        if cap:
            return float(cap)
    except Exception:
        pass
    try:
        from config import settings
        return float(settings.trading_capital)
    except Exception:
        return 100_000.0


def _max_positions() -> int:
    try:
        from config import settings
        return int(settings.max_open_positions)
    except Exception:
        return 5


def portfolio_risk_report(extra_trade: dict | None = None) -> dict:
    """
    {deployed, deployed_pct, open_risk, open_risk_pct, n_positions,
     sector_packs: {sector: [syms]}, warnings: [...], verdict}
    extra_trade: {symbol, qty, entry, stop} — simulate adding a trade.
    """
    from risk.position_manager import _open_trades
    trades = _open_trades()
    rows = [{"symbol": t["symbol"], "qty": int(t["qty"] or 0),
             "entry": float(t["entry_price"] or 0),
             "stop": float(t["stop_price"] or 0)} for t in trades]
    if extra_trade:
        rows.append(extra_trade)

    cap = _capital()
    deployed = sum(r["qty"] * r["entry"] for r in rows)
    open_risk = sum(r["qty"] * max(0.0, r["entry"] - r["stop"]) for r in rows)

    # Sector packs
    sector_packs: dict[str, list[str]] = {}
    try:
        from scan.sector_heat import sector_of
        for r in rows:
            sec = sector_of(r["symbol"])
            if sec:
                sector_packs.setdefault(sec, []).append(r["symbol"])
    except Exception:
        pass
    packs = {s: syms for s, syms in sector_packs.items() if len(syms) >= 2}

    risk_pct = open_risk / cap * 100 if cap else 0
    dep_pct = deployed / cap * 100 if cap else 0
    n = len(rows)
    max_n = _max_positions()

    warnings: list[str] = []
    verdict = "OK"
    if packs:
        for sec, syms in packs.items():
            warnings.append(f"⚠️ {sec} mein {len(syms)} positions "
                            f"({', '.join(syms)}) — yeh EK hi bet hai, sector "
                            f"gira toh sab saath girenge")
        verdict = "CAUTION"
    if risk_pct > 3:
        warnings.append(f"⚠️ Total open risk {risk_pct:.1f}% of capital — saare "
                        f"stops hit hue toh ₹{open_risk:,.0f} jayega")
        verdict = "CAUTION"
    if n >= max_n:
        warnings.append(f"⚠️ {n} positions open — max {max_n} ki limit "
                        + ("cross ho gayi" if n > max_n else "pe ho"))
        verdict = "DANGER" if n > max_n else verdict
    if risk_pct > 5:
        warnings.append("🔴 Open risk 5% se upar — naya trade lene se pehle "
                        "kuch band karo. Survival > opportunity.")
        verdict = "DANGER"
    if any(len(s) >= 3 for s in packs.values()):
        verdict = "DANGER"

    return {"deployed": round(deployed, 0), "deployed_pct": round(dep_pct, 1),
            "open_risk": round(open_risk, 0), "open_risk_pct": round(risk_pct, 2),
            "n_positions": n, "max_positions": max_n,
            "sector_packs": packs, "warnings": warnings, "verdict": verdict}


def check_new_trade(symbol: str, qty: int, entry: float, stop: float) -> dict:
    """Portfolio impact IF this trade is added — for the trade ticket."""
    return portfolio_risk_report(
        extra_trade={"symbol": symbol, "qty": qty, "entry": entry, "stop": stop})
