"""
Block deals and bulk deals — canonical read path via institutional_flows cache.

Streamlit/JARVIS must not fetch NSE independently; use get_flows() so React,
Brain and legacy UIs see the same bulk/block snapshot.
"""

from dataclasses import dataclass

from data.institutional_flows import get_flows


@dataclass
class BlockDeal:
    date: str
    symbol: str
    client_name: str
    deal_type: str       # "BUY" or "SELL"
    quantity: int
    price: float
    value_cr: float


def _deals_from_flows(flow_key: str) -> list[BlockDeal]:
    flows = get_flows()
    raw = list(flows.get(flow_key) or [])
    deals: list[BlockDeal] = []
    for rec in raw:
        try:
            sym = str(rec.get("symbol") or "").strip().upper()
            side = str(rec.get("side") or "BUY").strip().upper()
            qty = int(float(rec.get("qty") or 0))
            price = float(rec.get("price") or 0)
            if not sym or qty <= 0:
                continue
            deals.append(
                BlockDeal(
                    date=str(flows.get("fii_dii", {}).get("date") or ""),
                    symbol=sym,
                    client_name=str(rec.get("client") or "")[:60],
                    deal_type=side if side in ("BUY", "SELL") else "BUY",
                    quantity=qty,
                    price=price,
                    value_cr=round(qty * price / 1e7, 2),
                )
            )
        except Exception:
            continue
    return deals


def get_block_deals(days: int = 3) -> list[BlockDeal]:
    """Block deals from canonical NSE largedeal snapshot (BLOCK_DEALS_DATA)."""
    return _deals_from_flows("block_deals")


def get_bulk_deals(days: int = 3) -> list[BlockDeal]:
    """Bulk deals from canonical NSE largedeal snapshot (BULK_DEALS_DATA)."""
    return _deals_from_flows("bulk_deals")


def get_significant_deals(
    universe: list[str], min_value_cr: float = 50.0
) -> list[BlockDeal]:
    universe_upper = {s.strip().upper() for s in universe}
    all_deals = get_block_deals() + get_bulk_deals()
    filtered = [
        d
        for d in all_deals
        if d.symbol in universe_upper and d.value_cr >= min_value_cr
    ]
    return sorted(filtered, key=lambda d: d.value_cr, reverse=True)


def format_deal_insight(deals: list[BlockDeal]) -> str:
    lines: list[str] = []
    for d in deals:
        qty_lakhs = d.quantity / 1e5
        if qty_lakhs >= 1.0:
            qty_str = f"{qty_lakhs:.1f}L"
        else:
            qty_str = f"{d.quantity:,}"
        lines.append(
            f"{d.symbol} {d.deal_type} {qty_str} shares"
            f" @ ₹{d.price:,.0f}"
            f" (₹{d.value_cr:.0f}Cr)"
            f" — institutional transaction"
        )
    return "\n".join(lines)
