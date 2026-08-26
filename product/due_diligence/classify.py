"""Map a company onto a sector due-diligence framework. Easy to extend."""
from __future__ import annotations

from typing import Any, Mapping

from product.due_diligence.sector_frameworks.classify_rules import classify_business
from product.due_diligence.series import find_row


def classify_company(
    symbol: str,
    *,
    sector: str = "",
    about: str = "",
    quarterly_rows: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Pick a framework. Unknown stays generic — never guessed into a bank."""
    sector_name = str(sector or "").strip()
    if not sector_name:
        try:
            from scan.sector_heat import sector_of
            sector_name = str(sector_of(symbol) or "")
        except Exception:
            sector_name = ""
    has_npa = find_row(quarterly_rows, ("gross npa", "net npa", "gnpa", "nnpa")) is not None
    return classify_business(
        symbol,
        sector=sector_name,
        about=about,
        has_npa=has_npa,
    )
