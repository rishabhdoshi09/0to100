"""
🎯 Risk-first Trade Plan — the "R lens" a professional trader reads before every entry.

A READ-ONLY projection. It answers, for one candidate at a given capital: how many shares for the
configured risk, the rupee risk and reward:risk, the invalidation level, what the trade does to the
book's open-risk %, and whether it is a NEW bet or piling onto a correlated one. It owns no truth and
adds no risk math — it COMPOSES the authoritative functions (`risk.position_sizer.size_position`,
`risk.portfolio_risk.portfolio_risk_report`, `risk.correlation.clusters_from_corr`). Missing inputs
degrade honestly (an invalid stop is `tradeable=False`; correlation without history is `unknown`);
nothing is fabricated and no order is ever placed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

OK, CAUTION, DANGER = "OK", "CAUTION", "DANGER"
NEW_BET, ADDS_TO_BET, UNKNOWN = "new_bet", "adds_to_bet", "unknown"


@dataclass(frozen=True)
class TradePlan:
    symbol: str
    tradeable: bool
    reason: str
    entry: float
    stop: float
    target: float | None
    qty: int
    invested: float
    rupee_risk: float
    capped: bool
    pct_of_capital: float
    risk_pct_of_capital: float
    reward_risk: float | None            # R to target (reward:risk); None if no valid target
    invalidation_pct: float              # how far the stop is below entry, in %
    suggested_risk_pct: float            # after any regime throttle
    open_risk_pct_before: float | None
    open_risk_pct_after: float | None
    heat_verdict: str
    heat_warnings: tuple = ()
    correlation_status: str = UNKNOWN
    correlated_with: tuple = ()
    effective_bets_before: int | None = None
    effective_bets_after: int | None = None
    round_trip_cost_pct: float | None = None
    cost_drag_r: float | None = None
    summary: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _default_size(entry, stop, capital, risk_pct):
    from risk.position_sizer import size_position
    return size_position(entry, stop, capital=capital, risk_pct=risk_pct)


def build_trade_plan(symbol: str, entry: float, stop: float, target: float | None = None, *,
                     capital: float, risk_pct: float = 0.01, regime_factor: float = 1.0,
                     sizer=None, portfolio_report=None, correlated_with=None,
                     effective_bets_before: int | None = None,
                     effective_bets_after: int | None = None) -> TradePlan:
    """Compose a risk-first plan. `sizer` / `portfolio_report` are injected (default to the canonical
    functions) so this is deterministic in tests. `regime_factor` scales the risk budget (e.g. 0.5 in
    a weak tape). `correlated_with` is the list of ALREADY-OPEN symbols this candidate correlates with
    (from `clusters_from_corr`); None means correlation could not be assessed (honest `unknown`)."""
    sizer = sizer or _default_size
    entry = float(entry or 0.0); stop = float(stop or 0.0)
    target = float(target) if target not in (None, "", 0) else None
    suggested_risk_pct = round(max(0.0, float(risk_pct) * float(regime_factor)), 6)

    per_share_risk = entry - stop
    # honest untradeable states — never fabricate a size
    if entry <= 0 or per_share_risk <= 0:
        return TradePlan(symbol=symbol, tradeable=False,
                         reason=("stop must be below entry" if per_share_risk <= 0
                                 else "entry price unavailable"),
                         entry=entry, stop=stop, target=target, qty=0, invested=0.0, rupee_risk=0.0,
                         capped=False, pct_of_capital=0.0, risk_pct_of_capital=0.0,
                         reward_risk=None, invalidation_pct=0.0, suggested_risk_pct=suggested_risk_pct,
                         open_risk_pct_before=None, open_risk_pct_after=None, heat_verdict=OK,
                         summary="No valid trade plan: a stop below the entry is required.")

    size = sizer(entry, stop, capital, suggested_risk_pct)
    qty = int(size.get("qty", 0) or 0)
    invested = float(size.get("invested", 0.0) or 0.0)
    rupee_risk = float(size.get("max_loss", 0.0) or 0.0)
    capped = bool(size.get("capped", False))

    if qty < 1:
        return TradePlan(symbol=symbol, tradeable=False,
                         reason="capital too small to take even one share at this risk",
                         entry=entry, stop=stop, target=target, qty=0, invested=0.0, rupee_risk=0.0,
                         capped=capped, pct_of_capital=0.0, risk_pct_of_capital=0.0, reward_risk=None,
                         invalidation_pct=round(per_share_risk / entry * 100, 2),
                         suggested_risk_pct=suggested_risk_pct, open_risk_pct_before=None,
                         open_risk_pct_after=None, heat_verdict=OK,
                         summary="Capital is too small for a single share at the configured risk.")

    pct_cap = round(invested / capital * 100, 2) if capital else 0.0
    risk_pct_cap = round(rupee_risk / capital * 100, 2) if capital else 0.0
    reward_risk = (round((target - entry) / per_share_risk, 2)
                   if target and target > entry else None)
    invalidation_pct = round(per_share_risk / entry * 100, 2)

    # book heat before vs after — the AUTHORITATIVE account-risk function (composed, not re-derived)
    before_pct = after_pct = None
    heat_verdict = OK
    warnings: tuple = ()
    if portfolio_report is not None:
        try:
            before = portfolio_report(None) or {}
            after = portfolio_report({"symbol": symbol, "qty": qty, "entry": entry, "stop": stop}) or {}
            before_pct = before.get("open_risk_pct")
            after_pct = after.get("open_risk_pct")
            heat_verdict = str(after.get("verdict", OK) or OK)
            warnings = tuple(after.get("warnings", []) or ())
        except Exception:
            pass

    # correlation read — is this a NEW bet or adding to an existing one?
    if correlated_with is None:
        corr_status = UNKNOWN
        correlated = ()
    else:
        correlated = tuple(s for s in correlated_with if s and s != symbol)
        corr_status = ADDS_TO_BET if correlated else NEW_BET

    try:
        from core.costs import cost_drag_r, round_trip_cost_pct

        cost_pct = float(round_trip_cost_pct("CNC"))
        drag = cost_drag_r(entry, stop, product="CNC")
    except Exception:
        cost_pct = None
        drag = None

    summary = _summarize(
        symbol, qty, rupee_risk, risk_pct_cap, pct_cap, reward_risk, capped,
        before_pct, after_pct, heat_verdict, corr_status, correlated,
        suggested_risk_pct, risk_pct, regime_factor, cost_drag_r=drag,
    )

    return TradePlan(symbol=symbol, tradeable=True, reason="", entry=entry, stop=stop, target=target,
                     qty=qty, invested=round(invested, 0), rupee_risk=round(rupee_risk, 0),
                     capped=capped, pct_of_capital=pct_cap, risk_pct_of_capital=risk_pct_cap,
                     reward_risk=reward_risk, invalidation_pct=invalidation_pct,
                     suggested_risk_pct=suggested_risk_pct, open_risk_pct_before=before_pct,
                     open_risk_pct_after=after_pct, heat_verdict=heat_verdict, heat_warnings=warnings,
                     correlation_status=corr_status, correlated_with=correlated,
                     effective_bets_before=effective_bets_before,
                     effective_bets_after=effective_bets_after,
                     round_trip_cost_pct=cost_pct, cost_drag_r=drag, summary=summary)


def _summarize(symbol, qty, rupee_risk, risk_pct_cap, pct_cap, rr, capped, before, after,
               verdict, corr_status, correlated, suggested_risk_pct, base_risk_pct, regime_factor,
               *, cost_drag_r=None):
    parts = [f"Buy {qty} {symbol}: risking ₹{rupee_risk:,.0f} "
             f"({risk_pct_cap:.2f}% of capital) to hold ₹{pct_cap:,.1f}% of capital."]
    if rr is not None:
        parts.append(f"Reward:risk is {rr:.2f}R to target.")
    else:
        parts.append("No target set — reward:risk unknown.")
    if cost_drag_r is not None and cost_drag_r > 0:
        parts.append(f"Round-trip costs ≈ {cost_drag_r:.2f}R of the stop distance.")
    if regime_factor < 1.0:
        parts.append(f"Risk throttled to {suggested_risk_pct*100:.2f}% "
                     f"(from {base_risk_pct*100:.2f}%) for the current market condition.")
    if capped:
        parts.append("Size capped by the per-name concentration limit.")
    if before is not None and after is not None:
        parts.append(f"Open book risk goes {before:.2f}% → {after:.2f}% of capital.")
    if verdict == DANGER:
        parts.append("🔴 This pushes the book past a safety limit — close something first.")
    elif verdict == CAUTION:
        parts.append("🟠 Book risk is getting full; size down or wait.")
    if corr_status == ADDS_TO_BET and correlated:
        parts.append(f"This is NOT a new bet — it moves with {', '.join(correlated)}; "
                     "sized together they are one exposure.")
    elif corr_status == NEW_BET:
        parts.append("This looks like a genuinely new, independent bet.")
    return " ".join(parts)


# ── canonical wiring (production) ────────────────────────────────────────────────
def plan_for_candidate(candidate: dict, *, capital: float, risk_pct: float = 0.01,
                       regime_factor: float = 1.0, open_symbols=None,
                       price_history=None) -> TradePlan:
    """Gather the authoritative context and build the plan.

    ``open_symbols`` enables the correlation read (``pairwise_corr`` loads bhav
    itself). ``price_history`` is accepted for older callers but unused — history
    comes from the official bhav store. Without open symbols, correlation stays
    honestly ``unknown``.
    """
    from risk.portfolio_risk import portfolio_risk_report
    symbol = str(candidate.get("symbol", "")).upper()
    correlated = None
    bets_before = bets_after = None
    if open_symbols:
        try:
            from risk.correlation import pairwise_corr, clusters_from_corr
            syms = sorted({*[str(s).upper() for s in open_symbols if s], symbol})
            corr = pairwise_corr(syms) if price_history is None else pairwise_corr(syms)
            # price_history injection is reserved for tests that monkeypatch pairwise_corr
            _ = price_history
            clusters = clusters_from_corr(syms, corr)
            bets_after = len(clusters)
            others = [s for s in syms if s != symbol]
            bets_before = len(clusters_from_corr(others, corr)) if others else 0
            mine = next((c for c in clusters if symbol in c), [symbol])
            open_set = {str(x).upper() for x in open_symbols}
            correlated = [s for s in mine if s != symbol and s in open_set]
        except Exception:
            correlated = None
    return build_trade_plan(symbol, candidate.get("entry"), candidate.get("stop"),
                            candidate.get("target"), capital=capital, risk_pct=risk_pct,
                            regime_factor=regime_factor, portfolio_report=portfolio_risk_report,
                            correlated_with=correlated, effective_bets_before=bets_before,
                            effective_bets_after=bets_after)
