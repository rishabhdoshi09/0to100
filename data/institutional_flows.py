"""
🏦 Institutional flows — bade paise ka footprint, NSE se free, roz.

Do khazane jo NSE roz publish karta hai aur retail systematize nahi karta:

  1. FII/DII cash-market activity — videshi bech rahe ya kharid rahe?
     Desi institutions (DII) tape sambhaal rahe ya nahi? Market ka sabse
     bada single-day context.
  2. Bulk/block deals — kis STOCK mein kisi bade ne >0.5% equity ek
     saath uthaya. Symbol-level institutional footprint.

Data policy: NSE official API (wahi cookie-prime pattern jo nse_live/
options use karte hain). Fetch fail → None/empty, KOI claim nahi (no fake
data). Daily cache (logs/) — market band hone ke baad numbers final hote
hain, baar-baar maarne ka matlab nahi.

Parsers pure hain — network ke bina poora testable.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from logger import get_logger

log = get_logger(__name__)

_CACHE = Path(__file__).resolve().parent.parent / "logs" / "inst_flows.json"
_TTL_S = 3600 * 3              # 3h — din mein 2-3 refresh kaafi

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"),
    "Accept": "application/json",
    "Referer": "https://www.nseindia.com/reports/fii-dii",
}


# ── Pure parsers (testable) ──────────────────────────────────────────────────

def parse_fii_dii(rows: list[dict]) -> dict | None:
    """NSE fiidiiTradeReact rows → {date, fii_net_cr, dii_net_cr, bias, note}.
    bias: SUPPORTIVE (dono +) / FII_SELLING_DII_ABSORBING / DISTRIBUTION
    (dono −) / MIXED."""
    fii = dii = None
    date = ""
    for r in rows or []:
        cat = str(r.get("category", "")).upper()
        try:
            net = float(str(r.get("netValue", "0")).replace(",", ""))
        except Exception:
            continue
        date = r.get("date", date)
        if "FII" in cat or "FPI" in cat:
            fii = net
        elif "DII" in cat:
            dii = net
    if fii is None or dii is None:
        return None
    if fii > 0 and dii > 0:
        bias, note = "SUPPORTIVE", (f"FII +₹{fii:,.0f}cr aur DII +₹{dii:,.0f}cr "
                                    f"dono kharid rahe — tape ke neeche haath hai")
    elif fii < 0 <= dii:
        bias = "FII_SELLING_DII_ABSORBING"
        note = (f"FII −₹{abs(fii):,.0f}cr bech rahe, DII +₹{dii:,.0f}cr "
                f"absorb kar rahe — girawat pe sahara, par videshi paisa nikal raha")
    elif fii < 0 and dii < 0:
        bias, note = "DISTRIBUTION", (f"FII −₹{abs(fii):,.0f}cr AUR DII "
                                      f"−₹{abs(dii):,.0f}cr dono bech rahe — "
                                      f"bade paise bahar, breakouts se saavdhan")
    else:
        bias, note = "MIXED", (f"FII +₹{fii:,.0f}cr, DII ₹{dii:,.0f}cr — "
                               f"mixed haath")
    return {"date": date, "fii_net_cr": round(fii, 0),
            "dii_net_cr": round(dii, 0), "bias": bias, "note": note}


def parse_bulk_deals(payload: dict) -> list[dict]:
    """NSE bulk-deals JSON → [{symbol, client, side, qty, price}] — sirf
    saaf BUY/SELL rows."""
    out = []
    for r in (payload or {}).get("data", []) or []:
        sym = str(r.get("BD_SYMBOL") or r.get("symbol") or "").upper().strip()
        side = str(r.get("BD_BUY_SELL") or r.get("buySell") or "").upper()
        try:
            qty = float(str(r.get("BD_QTY_TRD") or r.get("qty") or 0)
                        .replace(",", ""))
            px = float(str(r.get("BD_TP_WATP") or r.get("watp") or 0)
                       .replace(",", ""))
        except Exception:
            continue
        if sym and side in ("BUY", "SELL") and qty > 0:
            out.append({"symbol": sym,
                        "client": str(r.get("BD_CLIENT_NAME")
                                      or r.get("clientName") or "")[:60],
                        "side": side, "qty": qty, "price": px})
    return out


def bulk_buy_symbols(deals: list[dict]) -> set[str]:
    """Symbols jahan NET bulk BUYING hui (buy qty > sell qty) — bade paise
    ka andar aana. Tag ke liye (evidence-context, score-change nahi jab tak
    edge measure na ho)."""
    net: dict[str, float] = {}
    for d in deals:
        net[d["symbol"]] = net.get(d["symbol"], 0) + (
            d["qty"] if d["side"] == "BUY" else -d["qty"])
    return {s for s, q in net.items() if q > 0}


# ── Fetch + cache ────────────────────────────────────────────────────────────

def _fetch() -> dict:
    """One NSE session → dono endpoints. Fail-safe: jo mile wahi."""
    out: dict = {}
    try:
        import requests
        s = requests.Session()
        s.headers.update(_HEADERS)
        s.get("https://www.nseindia.com", timeout=10)      # cookie prime
        try:
            r = s.get("https://www.nseindia.com/api/fiidiiTradeReact",
                      timeout=12)
            if r.ok:
                fd = parse_fii_dii(r.json())
                if fd:
                    out["fii_dii"] = fd
        except Exception as exc:
            log.debug("fii_dii_fetch_failed", error=str(exc)[:80])
        try:
            r = s.get("https://www.nseindia.com/api/snapshot-capital-market-largedeal",
                      timeout=12)
            if r.ok:
                data = r.json()
                deals = parse_bulk_deals(
                    {"data": (data.get("BULK_DEALS_DATA") or
                              data.get("data") or [])})
                out["bulk_deals"] = deals
                out["bulk_buys"] = sorted(bulk_buy_symbols(deals))
        except Exception as exc:
            log.debug("bulk_deals_fetch_failed", error=str(exc)[:80])
    except Exception as exc:
        log.debug("inst_flows_session_failed", error=str(exc)[:80])
    return out


def get_flows(max_age_s: int = _TTL_S) -> dict:
    """Cached institutional flows: {fii_dii: {...}|absent, bulk_deals: [...],
    bulk_buys: [...]}. Cache-first (3h TTL); fetch fail → purana cache
    (age ke saath) — stale ko fresh nahi bolte."""
    try:
        if _CACHE.exists():
            data = json.loads(_CACHE.read_text())
            if time.time() - float(data.get("ts", 0)) < max_age_s:
                return data
    except Exception:
        pass
    fresh = _fetch()
    if fresh:
        fresh["ts"] = time.time()
        try:
            _CACHE.parent.mkdir(parents=True, exist_ok=True)
            tmp = _CACHE.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(fresh))
            tmp.replace(_CACHE)
        except Exception:
            pass
        return fresh
    # fetch fail → stale cache (better than nothing, honestly aged)
    try:
        if _CACHE.exists():
            old = json.loads(_CACHE.read_text())
            old["stale"] = True
            return old
    except Exception:
        pass
    return {}
