"""Full NSE symbol directory for search / autocomplete (not scan setups)."""
from __future__ import annotations

from typing import Any


def build_symbol_directory(*, query: str = "", limit: int = 50) -> dict[str, Any]:
    """Return searchable equity symbols from the live universe (+ holdings).

    This is intentionally broader than the Momentum scan payload, which only
    keeps rows that produced setups. Broker holdings (including ``-BE`` series)
    are always merged so demat shares remain searchable.
    """
    from data.nse_universe import get_nse_universe_with_names, search_nse_symbols

    q = str(query or "").strip()
    lim = max(1, min(int(limit or 50), 5000))
    matches = search_nse_symbols(q, limit=lim)
    all_names = dict(get_nse_universe_with_names())

    held: list[str] = []
    try:
        from product.holdings_book import holdings_symbols

        held = holdings_symbols()
    except Exception:
        held = []
    for sym in held:
        all_names.setdefault(sym, "")
        if q and not (
            sym == q.upper()
            or sym.startswith(q.upper())
            or q.upper() in sym
        ):
            continue
        if not any(row["symbol"] == sym for row in matches):
            matches.insert(0, {"symbol": sym, "name": all_names.get(sym) or "Holding"})
            if len(matches) > lim:
                matches = matches[:lim]

    return {
        "schema_version": 1,
        "query": q,
        "limit": lim,
        "universe_size": len(all_names),
        "count": len(matches),
        "symbols": matches,
        "holdings_pinned": len(held),
        "source": "nse_universe+holdings",
        "note": (
            "Full equity directory for search — not limited to scan setups. "
            "Your demat holdings are pinned even when they use a -BE series symbol."
        ),
    }
