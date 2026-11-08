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

import contextlib
import math

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


class AccountRiskUnreadable(RuntimeError):
    """A paper book exists but its open positions cannot be read. Account
    exposure is UNKNOWN, which must block new entries rather than read as 0."""


FNO_BOOK = "FNO"


def _fno_open_rows() -> list[dict]:
    from core.runtime_paths import logs_path
    from product.fo_paper_store import FoPaperStore

    if not logs_path("product", "fo_paper.sqlite3").exists():
        return []
    with FoPaperStore(read_only=True) as store:
        positions = store.load_positions()
    rows = []
    for pos in positions:
        qty = int(pos.get("quantity") or 0)
        entry = float(pos.get("entry_price") or 0.0)
        stop = float(pos.get("stop_price") or 0.0)
        if qty <= 0 or not math.isfinite(entry) or not math.isfinite(stop) or entry <= 0:
            raise ValueError(f"F&O paper position {pos.get('trade_id')} has invalid exposure")
        rows.append({"symbol": str(pos.get("underlying") or "").upper(), "qty": qty,
                     "entry": entry, "stop": stop, "book": FNO_BOOK})
    return rows


def _combined_open_rows(extra_trade: dict | None = None, *,
                        include_fno: bool = True) -> list[dict]:
    """The ONE read of combined PAPER exposure across every book that spends
    the account's risk: legacy equity (trades.db), modern equity
    (intel_book.json) and, unless include_fno=False, the F&O options lane
    (fo_paper.sqlite3). portfolio_risk_report() (the display) and
    account_exposure_gate() (the pre-mutation check) are built on this same
    read, so the two can never disagree.

    A missing book is empty. A book that exists but cannot be read raises
    AccountRiskUnreadable: unknown exposure is not zero exposure."""
    from product.paper_status import modern_engine_open_positions
    from risk.position_manager import _open_trades

    try:
        trades = _open_trades(strict=True)
        # US_AUTOPILOT rows share trades.db but are a separate virtual USD pool
        # with its own rails (execution.us_autopilot); dollars are not rupees.
        rows = [{"symbol": t["symbol"], "qty": int(t["qty"] or 0),
                 "entry": float(t["entry_price"] or 0),
                 "stop": float(t["stop_price"] or 0)} for t in trades
                if "US_AUTOPILOT" not in str(t.get("note") or "")]
    except Exception as exc:
        raise AccountRiskUnreadable(f"legacy paper book (trades.db) unreadable: {exc}") from exc
    try:
        rows.extend(
            {"symbol": p["symbol"], "qty": p["qty"],
             "entry": p["entry_price"], "stop": p["stop_price"]}
            for p in modern_engine_open_positions(strict=True)
        )
    except Exception as exc:
        raise AccountRiskUnreadable(f"modern paper book (intel_book.json) unreadable: {exc}") from exc
    if include_fno:
        try:
            rows.extend(_fno_open_rows())
        except Exception as exc:
            raise AccountRiskUnreadable(f"F&O paper ledger unreadable: {exc}") from exc
    if extra_trade:
        rows.append(extra_trade)
    return rows


def _row_risk(row: dict) -> float:
    return row["qty"] * max(0.0, row["entry"] - row["stop"])


def equity_books_open_risk() -> float:
    """Rupees at risk across both equity paper books, for the F&O lane's
    share of the account ceiling. Raises AccountRiskUnreadable."""
    return sum(_row_risk(r) for r in _combined_open_rows(include_fno=False))


def portfolio_risk_report(extra_trade: dict | None = None) -> dict:
    """
    {deployed, deployed_pct, open_risk, open_risk_pct, n_positions,
     sector_packs: {sector: [syms]}, warnings: [...], verdict}
    extra_trade: {symbol, qty, entry, stop} — simulate adding a trade.
    """
    rows = _combined_open_rows(extra_trade)

    cap = _capital()
    deployed = sum(r["qty"] * r["entry"] for r in rows)
    open_risk = sum(_row_risk(r) for r in rows)

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
    # MAX_OPEN_POSITIONS is the equity slot limit; the F&O lane has its own.
    n = sum(1 for r in rows if r.get("book") != FNO_BOOK)
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


# Reason codes mirrored from product.paper_autopilot so both engines report
# a BLOCK with the same vocabulary; this module intentionally has no import
# of product.paper_autopilot (would be circular-ish and unnecessary) so the
# strings are the shared contract instead.
GATE_DUPLICATE_POSITION = "DUPLICATE_POSITION"
GATE_MAX_POSITIONS = "MAX_POSITIONS"
GATE_SECTOR_CAP = "SECTOR_CAP"
GATE_MAX_PORTFOLIO_RISK = "MAX_PORTFOLIO_RISK"
GATE_RISK_BOOK_UNREADABLE = "RISK_BOOK_UNREADABLE"


def account_exposure_gate(
    symbol: str, *, qty: int = 0, entry: float = 0.0, stop: float = 0.0,
    sector: str = "",
) -> dict:
    """The ONE pre-mutation check BOTH equity paper-trading engines
    (execution.autopilot and product.paper_autopilot) must pass before
    opening a new position.

    Built on the exact same combined-book read as portfolio_risk_report() --
    a trade the account-risk display would mark DANGER can no longer be
    approved by an execution path that only sees its own book. Two
    independent engines each querying only their own open positions created
    a real gap: either could open a symbol, hit the position cap, or breach
    the account's total-risk ceiling without ever seeing what the OTHER
    engine had already committed.

    qty/entry/stop default to 0 for an early probe (duplicate symbol /
    combined position cap / combined sector cap) that doesn't yet need a
    sized trade; call again with the real sized values right before the
    actual mutation (inside account_mutation_lock()) for the final
    open-risk-cap check and as the race-closing re-validation.

    Returns {"ok": bool, "reason_code": str, "detail": str, "report": dict}.
    """
    symbol = str(symbol or "").upper()
    try:
        existing = {r["symbol"] for r in _combined_open_rows()}
        if symbol in existing:
            return {
                "ok": False, "reason_code": GATE_DUPLICATE_POSITION,
                "detail": f"{symbol} already open in the account "
                          "(legacy/modern equity or F&O paper book)",
                "report": None,
            }

        probe = {"symbol": symbol, "qty": int(qty or 0),
                 "entry": float(entry or 0.0), "stop": float(stop or 0.0)}
        report = portfolio_risk_report(extra_trade=probe)
    except AccountRiskUnreadable as exc:
        log.warning("account_risk_book_unreadable", error=str(exc))
        return {"ok": False, "reason_code": GATE_RISK_BOOK_UNREADABLE,
                "detail": str(exc), "report": None}

    if report["n_positions"] > report["max_positions"]:
        return {
            "ok": False, "reason_code": GATE_MAX_POSITIONS,
            "detail": "combined account already at max open positions "
                      f"({report['max_positions']})",
            "report": report,
        }
    packs = report["sector_packs"]
    if sector and len(packs.get(sector, [])) >= 3:
        return {
            "ok": False, "reason_code": GATE_SECTOR_CAP,
            "detail": f"combined '{sector}' exposure would reach "
                      f"{len(packs[sector])} positions across both engines",
            "report": report,
        }
    if qty and report["open_risk_pct"] > 5:
        return {
            "ok": False, "reason_code": GATE_MAX_PORTFOLIO_RISK,
            "detail": f"combined open risk would reach {report['open_risk_pct']:.2f}% "
                      "of account capital — over the 5% ceiling",
            "report": report,
        }
    return {"ok": True, "reason_code": "", "detail": "", "report": report}


@contextlib.contextmanager
def account_mutation_lock():
    """Cross-process mutual exclusion around the equity PAPER
    check-exposure-then-mutate sequence.

    execution.autopilot and product.paper_autopilot can run in different OS
    processes (legacy via whatever hosts its manual entry point; modern via
    the autonomy supervisor process). A read-then-check-then-act sequence,
    however careful, cannot by itself close a race between two processes:
    both could pass account_exposure_gate() for the same symbol a moment
    apart and then both mutate. Both engines must acquire this BLOCKING
    flock immediately before their final gate re-check + mutation, and hold
    it across both, so only one engine's check-and-mutate can be in flight
    for the whole account at a time.
    """
    import fcntl
    from core.runtime_paths import ensure_logs_path

    path = ensure_logs_path("risk", "account_exposure.lock")
    with open(path, "a+", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
