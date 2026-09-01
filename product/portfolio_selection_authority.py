"""Portfolio Selection Authority — combination of names, not just the best stock.

Sits AFTER individual-stock Selection Authority. Explainable ranker, not an
opaque optimizer. Hard caps remain hard and cannot be overridden.

Watch / Avoid / DD-fail names cannot be upgraded to BUY here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

ENTER_NOW = "ENTER_NOW"
WAIT = "WAIT"
PORTFOLIO_BLOCK = "PORTFOLIO_BLOCK"
NO_TRADE = "NO_TRADE"

WATCH_TIERS = {"watch", "avoid", "skip"}
HARD_CAPS = {
    "MAX_NAME_RISK",
    "SECTOR_CAP",
    "MAX_PORTFOLIO_RISK",
    "LIQUIDITY_BLOCK",
    "DD_BLOCK",
    "INVALID_STOP",
    "DUPLICATE_POSITION",
    "CORRELATION_CAP",
}

DEFAULT_SECTOR_CAP = 2  # max new names from one sector in a cycle when others exist
DEFAULT_CORR_CAP = 0.80


@dataclass
class PortfolioChoice:
    symbol: str
    decision: str
    reason_code: str
    detail: str
    individual_rank: int
    individual_score: float
    portfolio_rank: int | None
    adjusted_score: float
    marginal_contribution: float
    concentration_effect: str
    correlation_effect: str
    risk_contribution: float
    capital_consumed: float
    opportunity_cost: str
    why_over: str
    hard_cap_applied: str | None = None
    invents_buy: bool = False
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _f(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        return out if out == out else None
    except (TypeError, ValueError):
        return None


def _sym(row: Mapping[str, Any]) -> str:
    return str(row.get("symbol") or "").strip().upper()


def _tier(row: Mapping[str, Any]) -> str:
    return str(row.get("reco_tier") or row.get("tier") or "").strip().lower()


def _sector(row: Mapping[str, Any]) -> str:
    return str(row.get("sector") or "").strip() or "UNKNOWN"


def _score(row: Mapping[str, Any]) -> float:
    for key in ("selection_score", "score", "individual_score"):
        val = _f(row.get(key))
        if val is not None:
            return val
    return 0.0


def _held_symbols(book: Any) -> set[str]:
    out: set[str] = set()
    if book is None:
        return out
    for pos in getattr(book, "open", {}) .values() if isinstance(getattr(book, "open", None), dict) else []:
        out.add(str(getattr(pos, "symbol", "") or "").upper())
    return out


def _held_sectors(book: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if book is None:
        return counts
    opens = getattr(book, "open", {})
    if not isinstance(opens, dict):
        return counts
    for pos in opens.values():
        sector = str(getattr(pos, "sector", "") or "")
        if not sector:
            continue
        counts[sector] = counts.get(sector, 0) + 1
    return counts


def _corr_key(a: str, b: str) -> str:
    left, right = sorted((a.upper(), b.upper()))
    return f"{left}|{right}"


def _hard_cap(row: Mapping[str, Any], *, held: set[str], book: Any) -> str | None:
    symbol = _sym(row)
    if not symbol:
        return "NO_TRADE"
    if symbol in held:
        return "DUPLICATE_POSITION"
    dd = str(row.get("dd_verdict") or row.get("dd_status") or "").upper()
    if dd in {"FAIL", "FAILED", "BLOCK", "AVOID"}:
        return "DD_BLOCK"
    if _tier(row) in WATCH_TIERS:
        return "WATCH_TIER"
    entry = _f(row.get("entry") or row.get("entry_price"))
    stop = _f(row.get("stop") or row.get("stop_price"))
    if entry is None or stop is None or stop <= 0 or (entry is not None and stop >= entry):
        return "INVALID_STOP"
    vol = _f(row.get("volume_ratio"))
    if vol is not None and vol < 0.7:
        return "LIQUIDITY_BLOCK"
    if bool(row.get("hard_cap")):
        return str(row.get("hard_cap") or "HARD_CAP")
    if book is not None:
        max_pos = int(getattr(book, "max_positions", 5) or 5)
        if len(getattr(book, "open", {}) or {}) >= max_pos:
            return "MAX_PORTFOLIO_RISK"
    return None


def allocate(
    eligible: Sequence[Mapping[str, Any]],
    *,
    book: Any = None,
    capital: float | None = None,
    regime: str = "RISK_ON",
    max_new: int = 3,
    correlations: Mapping[str, float] | None = None,
    existing_sectors: Mapping[str, int] | None = None,
    name_risk_pct: float = 1.0,
    sector_cap: int = DEFAULT_SECTOR_CAP,
    correlation_cap: float = DEFAULT_CORR_CAP,
    portfolio_risk_used_pct: float = 0.0,
    portfolio_risk_cap_pct: float = 5.0,
) -> list[PortfolioChoice]:
    """Rank already-eligible names for scarce risk capital.

    Does not invent BUY. Highest individual score does not automatically win
    if it duplicates existing portfolio risk. If the caller does not provide a
    correlation matrix, we try the already-local official NSE bhavcopy store.
    No network fetch is started from the money path.
    """
    held = _held_symbols(book)
    held_sectors = dict(existing_sectors or {})
    if not held_sectors:
        held_sectors = _held_sectors(book)

    rows = [dict(r) for r in eligible]
    corr_meta: dict[str, Any] = {
        "source": "caller_supplied" if correlations is not None else "unavailable",
        "point_in_time": correlations is not None,
        "coverage": 1.0 if correlations is not None else 0.0,
        "as_of": "",
        "network_used": False,
    }
    if correlations is None:
        try:
            from product.pit_correlation import correlations_for_candidates
            corr_meta = correlations_for_candidates(rows, held_symbols=sorted(held))
            corr = dict(corr_meta.get("correlations") or {})
        except Exception as exc:
            corr = {}
            corr_meta = {
                "source": "unavailable",
                "point_in_time": False,
                "coverage": 0.0,
                "as_of": "",
                "network_used": False,
                "error": type(exc).__name__,
            }
    else:
        corr = dict(correlations)

    # Deterministic individual rank first (score desc, symbol asc)
    individual = sorted(rows, key=lambda r: (-_score(r), _sym(r)))
    ranked_index = {_sym(r): i + 1 for i, r in enumerate(individual)}

    scored: list[tuple[float, Mapping[str, Any], dict[str, Any]]] = []
    for row in individual:
        symbol = _sym(row)
        sector = _sector(row)
        individual_score = _score(row)
        cap = _hard_cap(row, held=held, book=book)
        notes = {
            "concentration_effect": "none",
            "correlation_effect": "unknown",
            "marginal_contribution": individual_score,
            "why_over": "",
            "hard_cap": cap,
            "correlation_source": corr_meta.get("source") or "unavailable",
            "correlation_as_of": corr_meta.get("as_of") or "",
            "correlation_coverage": corr_meta.get("coverage") or 0.0,
        }
        adj = individual_score
        # Sector duplication vs existing book — labelled estimated (sector proxy)
        existing_in_sector = int(held_sectors.get(sector, 0) or 0)
        if existing_in_sector > 0 and sector != "UNKNOWN":
            penalty = 8.0 + 4.0 * existing_in_sector
            adj -= penalty
            notes["concentration_effect"] = (
                f"duplicates {existing_in_sector} existing {sector} position(s); "
                f"penalty={penalty}"
            )
            notes["marginal_contribution"] = round(individual_score - penalty, 4)
        # Pairwise correlation from caller or local official PIT bhavcopy. Missing
        # stays unknown; sector remains a labelled proxy rather than fake corr.
        measured_corr = []
        for other in held:
            key = _corr_key(symbol, other)
            if key in corr:
                measured_corr.append(float(corr[key]))
        if measured_corr:
            peak = max(measured_corr)
            src = str(notes.get("correlation_source") or "measured")
            notes["correlation_effect"] = f"max_corr={peak:.2f} ({src})"
            if peak >= correlation_cap:
                notes["hard_cap"] = "CORRELATION_CAP"
            else:
                adj -= 10.0 * peak
        elif held:
            notes["correlation_effect"] = "unknown — no usable PIT correlation; sector used as proxy"
        scored.append((adj, row, notes))

    scored.sort(key=lambda item: (-item[0], _sym(item[1])))

    chosen_sectors: dict[str, int] = {}
    chosen_symbols: list[str] = []
    out: list[PortfolioChoice] = []
    entered = 0
    remaining_risk = float(portfolio_risk_cap_pct) - float(portfolio_risk_used_pct)
    cap_capital = float(capital if capital is not None else getattr(book, "capital", 0.0) or 0.0)

    for port_i, (adj, row, notes) in enumerate(scored, start=1):
        symbol = _sym(row)
        sector = _sector(row)
        individual_score = _score(row)
        hard = notes.get("hard_cap")
        decision = ENTER_NOW
        reason = "PORTFOLIO_SELECTED"
        detail = "adds independent risk capital use"
        why = notes.get("why_over") or ""

        if symbol in held:
            hard = "DUPLICATE_POSITION"
        elif chosen_sectors.get(sector, 0) + held_sectors.get(sector, 0) >= int(sector_cap) and any(
            _sector(r) != sector for _, r, __ in scored
        ):
            hard = "SECTOR_CAP"

        if hard == "WATCH_TIER":
            decision, reason, detail = NO_TRADE, "WATCH_TIER", "Watch/Avoid cannot become BUY"
        elif hard == "DD_BLOCK":
            decision, reason, detail = PORTFOLIO_BLOCK, "DD_BLOCK", "DD remains authoritative"
        elif hard == "INVALID_STOP":
            decision, reason, detail = PORTFOLIO_BLOCK, "INVALID_STOP", "invalid stop cannot be overridden"
        elif hard == "DUPLICATE_POSITION":
            decision, reason, detail = PORTFOLIO_BLOCK, "DUPLICATE_POSITION", "already held"
        elif hard == "LIQUIDITY_BLOCK":
            decision, reason, detail = PORTFOLIO_BLOCK, "LIQUIDITY_BLOCK", "liquidity block is hard"
        elif hard == "CORRELATION_CAP":
            decision, reason, detail = PORTFOLIO_BLOCK, "CORRELATION_CAP", "correlation safety cap"
        elif hard == "MAX_PORTFOLIO_RISK":
            decision, reason, detail = PORTFOLIO_BLOCK, "MAX_PORTFOLIO_RISK", "portfolio risk cap"
        elif hard:
            decision, reason, detail = PORTFOLIO_BLOCK, str(hard), "hard cap cannot be overridden"
        elif remaining_risk < name_risk_pct - 1e-9:
            decision, reason, detail = PORTFOLIO_BLOCK, "MAX_PORTFOLIO_RISK", "risk budget exhausted"
        elif entered >= int(max_new):
            decision, reason, detail = WAIT, "NOT_TOP_OF_PORTFOLIO", "capital already allocated this cycle"
        elif (
            chosen_sectors.get(sector, 0) + held_sectors.get(sector, 0) >= int(sector_cap)
            and any(_sector(r) != sector for _, r, __ in scored)
        ):
            # Sector cap among competing eligible names — still hard.
            decision, reason, detail = PORTFOLIO_BLOCK, "SECTOR_CAP", f"{sector} cap"

        if decision == ENTER_NOW:
            entered += 1
            remaining_risk -= name_risk_pct
            chosen_sectors[sector] = chosen_sectors.get(sector, 0) + 1
            chosen_symbols.append(symbol)
            held.add(symbol)
            # Explain why this beat a higher individual score if applicable
            higher = [
                _sym(r) for r in individual
                if _score(r) > individual_score + 1e-9 and _sym(r) not in chosen_symbols
            ]
            if higher:
                why = (
                    f"selected over {higher[0]} because competing higher-ranked name(s) "
                    f"duplicate existing portfolio risk while {symbol} adds independent edge"
                )
            elif notes.get("concentration_effect") not in {"none", ""}:
                why = f"selected despite concentration: {notes['concentration_effect']}"
            elif not why:
                why = "highest portfolio-adjusted score among remaining eligible names"

        capital_consumed = 0.0
        if decision == ENTER_NOW and cap_capital:
            capital_consumed = round(cap_capital * (name_risk_pct / 100.0), 4)

        opp_cost = ""
        if decision != ENTER_NOW and entered == 0 and port_i == len(scored):
            opp_cost = "no name received capital this cycle"

        out.append(
            PortfolioChoice(
                symbol=symbol,
                decision=decision,
                reason_code=reason,
                detail=detail,
                individual_rank=ranked_index.get(symbol, port_i),
                individual_score=individual_score,
                portfolio_rank=entered if decision == ENTER_NOW else None,
                adjusted_score=round(adj, 4),
                marginal_contribution=float(notes.get("marginal_contribution") or adj),
                concentration_effect=str(notes.get("concentration_effect") or "none"),
                correlation_effect=str(notes.get("correlation_effect") or "unknown"),
                risk_contribution=name_risk_pct if decision == ENTER_NOW else 0.0,
                capital_consumed=capital_consumed,
                opportunity_cost=opp_cost,
                why_over=why,
                hard_cap_applied=str(hard) if hard and decision != ENTER_NOW else None,
                invents_buy=False,
                fields={
                    "correlation_source": notes.get("correlation_source") or "unavailable",
                    "correlation_as_of": notes.get("correlation_as_of") or "",
                    "correlation_coverage": notes.get("correlation_coverage") or 0.0,
                    "network_used": bool(corr_meta.get("network_used")),
                },
            )
        )

    if not out:
        out.append(
            PortfolioChoice(
                symbol="",
                decision=NO_TRADE,
                reason_code="NO_ELIGIBLE_CANDIDATE",
                detail="no individually eligible names",
                individual_rank=0,
                individual_score=0.0,
                portfolio_rank=None,
                adjusted_score=0.0,
                marginal_contribution=0.0,
                concentration_effect="none",
                correlation_effect="unknown",
                risk_contribution=0.0,
                capital_consumed=0.0,
                opportunity_cost="no trade",
                why_over="",
                invents_buy=False,
                fields={"correlation_source": corr_meta.get("source") or "unavailable", "network_used": False},
            )
        )
    return out


def apply_portfolio_authority(
    ranked: Sequence[tuple[float, Any]],
    *,
    book: Any = None,
    max_new: int = 3,
    regime: str = "RISK_ON",
    correlations: Mapping[str, float] | None = None,
) -> tuple[list[tuple[float, Any]], list[Any]]:
    """Reorder Selection Authority ENTER_NOW names. Divert the rest.

    Identity when a single eligible name has no hard-cap conflict.
    """
    if not ranked:
        return [], []
    cards = []
    by_symbol: dict[str, Any] = {}
    for score, decision in ranked:
        card = dict(getattr(decision, "card", None) or {})
        card.setdefault("symbol", getattr(decision, "symbol", ""))
        card.setdefault("selection_score", getattr(decision, "selection_score", score))
        card.setdefault("sector", getattr(decision, "card", {}).get("sector") if getattr(decision, "card", None) else "")
        card.setdefault("reco_tier", (getattr(decision, "card", None) or {}).get("reco_tier"))
        card.setdefault("dd_verdict", (getattr(decision, "card", None) or {}).get("dd_verdict"))
        card.setdefault("entry", (getattr(decision, "card", None) or {}).get("entry"))
        card.setdefault("stop", (getattr(decision, "card", None) or {}).get("stop"))
        card.setdefault("volume_ratio", (getattr(decision, "card", None) or {}).get("volume_ratio"))
        context = dict(getattr(decision, "context", None) or {})
        if context.get("as_of"):
            card.setdefault("as_of", context.get("as_of"))
        cards.append(card)
        by_symbol[_sym(card)] = (score, decision)

    choices = allocate(
        cards,
        book=book,
        max_new=max_new,
        regime=regime,
        correlations=correlations,
        capital=float(getattr(book, "capital", 0.0) or 0.0) if book is not None else None,
    )
    kept: list[tuple[float, Any]] = []
    diverted: list[Any] = []
    for choice in choices:
        pair = by_symbol.get(choice.symbol)
        if pair is None:
            continue
        score, decision = pair
        if hasattr(decision, "portfolio"):
            decision.portfolio = choice.to_dict()
        if choice.decision == ENTER_NOW:
            kept.append((score, decision))
            continue
        if hasattr(decision, "decision"):
            decision.decision = choice.decision
            decision.reason_code = choice.reason_code
            decision.detail = choice.detail
            decision.group = "RECOMMENDED_BUT_NOT_FILLED" if choice.decision == WAIT else "REJECTED"
        diverted.append(decision)
    return kept, diverted
