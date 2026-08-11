"""Full NSE symbol directory for search / autocomplete (not scan setups)."""
from __future__ import annotations

from typing import Any


def build_symbol_directory(*, query: str = "", limit: int = 50) -> dict[str, Any]:
    """Return searchable equity symbols from the live universe (+ holdings).

    This is intentionally broader than the Momentum scan payload, which only
    keeps rows that produced setups. Broker holdings (including ``-BE`` series)
    are always merged so demat shares remain searchable.

    Empty ``query`` with ``limit<=0`` (or a thin limit) expands to the full
    universe size so letters after M/N are not truncated out of autocomplete.
    """
    from data.nse_universe import get_nse_universe_with_names, search_nse_symbols

    q = str(query or "").strip()
    all_names = dict(get_nse_universe_with_names())
    requested = int(limit or 0)
    if not q and requested <= 0:
        requested = max(len(all_names), 5000)
    lim = max(1, min(requested if requested > 0 else 50, 20_000))
    # Always ask for at least the full universe size on empty query.
    if not q:
        lim = max(lim, len(all_names) or lim)
    matches = search_nse_symbols(q, limit=lim)

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

    letters = sorted({(row["symbol"][:1] or "?") for row in matches if row.get("symbol")})
    return {
        "schema_version": 1,
        "query": q,
        "limit": lim,
        "universe_size": len(all_names),
        "count": len(matches),
        "symbols": matches,
        "letter_coverage": letters,
        "truncated": (not q) and len(matches) < len(all_names),
        "holdings_pinned": len(held),
        "source": "nse_universe+holdings",
        "note": (
            "Full equity directory for search — not limited to scan setups. "
            "Your demat holdings are pinned even when they use a -BE series symbol."
        ),
    }
