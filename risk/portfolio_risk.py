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


def legacy_open_rows() -> list[dict]:
    """Open rows from the legacy trades journal. Empty when that book is quiet."""
    from risk.position_manager import _open_trades
    trades = _open_trades()
    return [{"symbol": t["symbol"], "qty": int(t["qty"] or 0),
             "entry": float(t["entry_price"] or 0),
             "stop": float(t["stop_price"] or 0)} for t in trades]


def assess_open_rows(rows: list[dict] | None, *,
                     capital: float | None = None,
                     max_positions: int | None = None) -> dict:
    """Same OK / CAUTION / DANGER rails as the legacy report, on any row list.

    Thresholds are unchanged: sector pack of 2 → CAUTION, open risk >3% →
    CAUTION, open risk >5% or 3 names in one sector or more than max positions
    → DANGER. A supplied ``sector`` is trusted; otherwise the sector map is
    consulted. Missing sectors stay ungrouped — they are not invented.
    """
    clean: list[dict] = []
    for raw in rows or []:
        if not isinstance(raw, dict):
            continue
        try:
            qty = int(raw.get("qty") or 0)
            entry = float(raw.get("entry") or 0)
            stop = float(raw.get("stop") or 0)
        except (TypeError, ValueError):
            continue
        symbol = str(raw.get("symbol") or "").strip().upper()
        if not symbol or qty <= 0 or entry <= 0:
            continue
        clean.append({
            "symbol": symbol,
            "qty": qty,
            "entry": entry,
            "stop": stop,
            "sector": str(raw.get("sector") or "").strip(),
        })

    cap = float(capital) if capital is not None else _capital()
    max_n = int(max_positions) if max_positions is not None else _max_positions()
    deployed = sum(r["qty"] * r["entry"] for r in clean)
    open_risk = sum(r["qty"] * max(0.0, r["entry"] - r["stop"]) for r in clean)

    sector_packs: dict[str, list[str]] = {}
    try:
        from scan.sector_heat import sector_of
    except Exception:
        sector_of = None  # type: ignore[assignment]
    for r in clean:
        sec = r["sector"]
        if not sec and sector_of is not None:
            try:
                sec = str(sector_of(r["symbol"]) or "")
            except Exception:
                sec = ""
        if sec:
            sector_packs.setdefault(sec, []).append(r["symbol"])
    packs = {s: syms for s, syms in sector_packs.items() if len(syms) >= 2}

    risk_pct = open_risk / cap * 100 if cap else 0
    dep_pct = deployed / cap * 100 if cap else 0
    n = len(clean)

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


def portfolio_risk_report(extra_trade: dict | None = None) -> dict:
    """
    {deployed, deployed_pct, open_risk, open_risk_pct, n_positions,
     sector_packs: {sector: [syms]}, warnings: [...], verdict}
    extra_trade: {symbol, qty, entry, stop} — simulate adding a trade.
    """
    rows = legacy_open_rows()
    if extra_trade:
        rows.append(extra_trade)
    return assess_open_rows(rows)


def check_new_trade(symbol: str, qty: int, entry: float, stop: float) -> dict:
    """Portfolio impact IF this trade is added — for the trade ticket."""
    return portfolio_risk_report(
        extra_trade={"symbol": symbol, "qty": qty, "entry": entry, "stop": stop})
