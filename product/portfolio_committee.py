"""Portfolio committee — a stock BUY is not automatically a trade.

Runs after the stock-level committee. A portfolio wait must not rewrite
the stock thesis. Correlation is not just sector: price-return clusters
are inspected when official history is present.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from product import decision_taxonomy as T

WAIT_PORTFOLIO = "WAIT_PORTFOLIO"
ADMIT = "ADMIT"


def _f(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _open_symbols(book: Any) -> list[str]:
    if book is None:
        return []
    out = []
    for pos in getattr(book, "open", {}) or {}:
        if hasattr(pos, "symbol"):
            out.append(str(pos.symbol).upper())
        elif isinstance(pos, str):
            out.append(pos.upper())
    if not out and isinstance(getattr(book, "open", None), dict):
        for item in book.open.values():
            out.append(str(getattr(item, "symbol", item) or "").upper())
    return [s for s in out if s]


def evaluate_portfolio(
    rec: Mapping[str, Any],
    *,
    book: Any = None,
    open_positions: Sequence[Mapping[str, Any]] | None = None,
    as_of: str = "",
    corr_threshold: float = 0.70,
) -> dict[str, Any]:
    """Overlay only. Never changes rec['decision'] from BUY to AVOID."""
    symbol = str(rec.get("symbol") or "").upper()
    stock_decision = str(rec.get("decision") or "")
    sector = str(rec.get("sector") or (rec.get("references") or {}).get("sector") or "")
    opens = list(open_positions or [])
    if not opens and book is not None:
        for item in (getattr(book, "open", {}) or {}).values():
            if isinstance(item, Mapping):
                opens.append(dict(item))
            else:
                opens.append({
                    "symbol": getattr(item, "symbol", ""),
                    "sector": getattr(item, "sector", ""),
                    "risk_pct": getattr(item, "risk_pct", None),
                })
    held = {str(p.get("symbol") or "").upper() for p in opens if p.get("symbol")}
    reasons: list[str] = []
    clusters: list[str] = []

    if symbol in held:
        reasons.append(T.PORTFOLIO_CONCENTRATION)
    if book is not None:
        max_pos = int(getattr(book, "max_positions", 5) or 5)
        if len(held) >= max_pos:
            reasons.append(T.MAX_POSITIONS)
        try:
            open_risk = float(book.open_risk()) if hasattr(book, "open_risk") else 0.0
            cap = float(getattr(book, "capital", 0.0) or 0.0)
            max_total = cap * float(getattr(book, "max_total_risk_pct", 0.05) or 0.05)
            if cap > 0 and open_risk >= max_total:
                reasons.append(T.MAX_PORTFOLIO_RISK)
        except Exception:
            pass

    same_sector = [p for p in opens if sector and str(p.get("sector") or "").lower() == sector.lower()]
    if sector and len(same_sector) >= 2:
        reasons.append(T.SECTOR_CAP)
        clusters.append(f"sector:{sector}")

    corr_pairs: list[dict[str, Any]] = []
    if held and as_of:
        try:
            from product.pit_correlation import build_pit_correlations

            names = sorted(held | {symbol})
            pit = build_pit_correlations(names, as_of=as_of)
            for pair, value in (pit.get("correlations") or {}).items():
                try:
                    corr = float(value)
                except (TypeError, ValueError):
                    continue
                if symbol in str(pair) and corr >= corr_threshold:
                    corr_pairs.append({"pair": pair, "corr": corr})
                    reasons.append(T.CORRELATION_LIMIT)
                    clusters.append(pair)
        except Exception:
            pass

    blocked = bool(reasons) and stock_decision == T.BUY
    return {
        "symbol": symbol,
        "stock_decision": stock_decision,
        "portfolio_verdict": WAIT_PORTFOLIO if blocked else ADMIT,
        "execution_state": T.BLOCKED_PORTFOLIO if blocked else rec.get("execution_state"),
        "reason_codes": list(dict.fromkeys(reasons)),
        "clusters": clusters,
        "correlated_pairs": corr_pairs,
        "open_symbols": sorted(held),
        "sector": sector,
        "thesis_preserved": True,
        "note": (
            "Stock BUY stands. Portfolio capacity refuses admission."
            if blocked else "Portfolio admits the stock-level BUY."
        ),
    }


def apply_overlay(rec: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Attach portfolio overlay without mutating the stock decision."""
    out = dict(rec)
    out["portfolio"] = dict(overlay)
    if overlay.get("portfolio_verdict") == WAIT_PORTFOLIO:
        out["execution_state"] = T.BLOCKED_PORTFOLIO
        out["portfolio_verdict"] = WAIT_PORTFOLIO
        # candidate_state / decision stay BUY + READY
    return out
