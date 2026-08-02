"""Full NSE symbol directory for search / autocomplete (not scan setups)."""
from __future__ import annotations

from typing import Any


def build_symbol_directory(*, query: str = "", limit: int = 50) -> dict[str, Any]:
    """Return searchable equity symbols from the live universe (+ bhav when present).

    This is intentionally broader than the Momentum scan payload, which only
    keeps rows that produced setups.
    """
    from data.nse_universe import get_nse_universe_with_names, search_nse_symbols

    q = str(query or "").strip()
    lim = max(1, min(int(limit or 50), 5000))
    matches = search_nse_symbols(q, limit=lim)
    all_names = get_nse_universe_with_names()
    return {
        "schema_version": 1,
        "query": q,
        "limit": lim,
        "universe_size": len(all_names),
        "count": len(matches),
        "symbols": matches,
        "source": "nse_universe",
        "note": (
            "Full equity directory for search — not limited to scan setups. "
            "Stock Intelligence can open any of these symbols when bhav history exists."
        ),
    }
