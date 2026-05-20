"""
Block deals and bulk deals from NSE India.
Institutional transactions — informed money moving.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

_SESSION: requests.Session | None = None
_BLOCK_CACHE: dict = {}   # {"date": str, "data": list[BlockDeal]}
_BULK_CACHE: dict = {}    # {"date": str, "data": list[BlockDeal]}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.nseindia.com/",
    "Accept": "application/json",
}


@dataclass
class BlockDeal:
    date: str
    symbol: str
    client_name: str
    deal_type: str       # "BUY" or "SELL"
    quantity: int
    price: float
    value_cr: float


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        try:
            _SESSION.get(
                "https://www.nseindia.com/",
                headers=_HEADERS,
                timeout=10,
            )
        except Exception:
            pass
    return _SESSION


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _fetch_deals(endpoint: str) -> list[dict]:
    session = _get_session()
    try:
        resp = session.get(endpoint, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            for key in ("data", "BLOCK_DEALS", "BULK_DEALS", "records"):
                if key in data:
                    return data[key]
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def _parse_deals(raw: list[dict]) -> list[BlockDeal]:
    deals: list[BlockDeal] = []
    for rec in raw:
        try:
            symbol = (
                rec.get("symbol") or rec.get("Symbol") or rec.get("scrip_name") or ""
            ).strip().upper()
            client = (
                rec.get("client_name")
                or rec.get("clientName")
                or rec.get("Client_Name")
                or ""
            ).strip()
            dt = (
                rec.get("date")
                or rec.get("deal_date")
                or rec.get("BD_DT_DATE")
                or ""
            ).strip()
            buy_sell_raw = (
                rec.get("deal_type")
                or rec.get("buyOrSell")
                or rec.get("BD_BUYSELL")
                or ""
            ).strip().upper()
            deal_type = "BUY" if buy_sell_raw.startswith("B") else "SELL"
            qty = int(
                rec.get("quantity")
                or rec.get("BD_QTY_TRD")
                or rec.get("qty")
                or 0
            )
            price = float(
                rec.get("price")
                or rec.get("BD_TP_WATP")
                or rec.get("tradePrice")
                or 0.0
            )
            value_cr = round(qty * price / 1e7, 2)
            deals.append(
                BlockDeal(
                    date=dt,
                    symbol=symbol,
                    client_name=client,
                    deal_type=deal_type,
                    quantity=qty,
                    price=price,
                    value_cr=value_cr,
                )
            )
        except Exception:
            continue
    return deals


def get_block_deals(days: int = 3) -> list[BlockDeal]:
    global _BLOCK_CACHE
    today = _today_str()
    if _BLOCK_CACHE.get("date") == today:
        return _BLOCK_CACHE["data"]
    try:
        raw = _fetch_deals("https://www.nseindia.com/api/block-deal")
        deals = _parse_deals(raw)
        _BLOCK_CACHE = {"date": today, "data": deals}
        return deals
    except Exception:
        return []


def get_bulk_deals(days: int = 3) -> list[BlockDeal]:
    global _BULK_CACHE
    today = _today_str()
    if _BULK_CACHE.get("date") == today:
        return _BULK_CACHE["data"]
    try:
        raw = _fetch_deals("https://www.nseindia.com/api/bulk-deal")
        deals = _parse_deals(raw)
        _BULK_CACHE = {"date": today, "data": deals}
        return deals
    except Exception:
        return []


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
