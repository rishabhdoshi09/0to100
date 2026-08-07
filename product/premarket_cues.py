"""Premarket / GIFT Nifty cues — soft-fail only, never invents levels.

Sources tried (in order):
  1. Moneycontrol pre-market page text (Gift Nifty mentions when present)
  2. US index futures via yfinance (ES/NQ) as overnight risk proxy
  3. India VIX / Nifty last print when Kite/live quotes available

Missing stays missing. Not a buy ticket.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _moneycontrol_premarket_text() -> tuple[str | None, str | None]:
    try:
        import requests
        from bs4 import BeautifulSoup
    except Exception as exc:
        return None, f"scrape deps unavailable ({exc})"
    url = "https://www.moneycontrol.com/pre-market/"
    try:
        resp = requests.Session().get(
            url,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html"},
            timeout=12,
        )
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"
        soup = BeautifulSoup(resp.text, "html.parser")
        div = soup.find("div", class_="premarket_data") or soup.find("div", {"id": "premarket"})
        if div:
            return div.get_text("\n", strip=True)[:4000], None
        kws = ("Gift Nifty", "GIFT Nifty", "GIFTNIFTY", "S&P", "Nasdaq", "Crude", "Gold", "Asian")
        lines = [
            line.strip()
            for line in soup.get_text("\n").split("\n")
            if any(k.lower() in line.lower() for k in kws) and len(line.strip()) > 20
        ]
        if not lines:
            return None, "No Gift Nifty / pre-market lines on Moneycontrol"
        return "\n".join(lines[:40]), None
    except Exception as exc:
        return None, str(exc)


def _parse_gift_nifty(text: str) -> dict[str, Any] | None:
    """Extract Gift Nifty level / change when the page states them plainly."""
    # Prefer explicit level + (pct) before bare "up/down X%" — order matters.
    level_pct = re.compile(
        r"Gift\s*Nifty[^\n]{0,40}?(-?\d{2}(?:,\d{3})*(?:\.\d+)?)[^\n]{0,20}?\((-?\d+(?:\.\d+)?)\s*%\)",
        re.I,
    )
    m = level_pct.search(text or "")
    if m:
        try:
            level = float(str(m.group(1)).replace(",", ""))
            chg = float(m.group(2))
            return {"level": level, "chg_pct": chg, "source": "moneycontrol_text"}
        except Exception:
            pass

    direction_pct = re.compile(
        r"Gift\s*Nifty[^\n%]{0,80}?((?:up|down|rose|fell|gained|lost)\s+)?(-?\d+(?:\.\d+)?)\s*%",
        re.I,
    )
    m = direction_pct.search(text or "")
    if m:
        direction = (m.group(1) or "").lower()
        try:
            chg = float(m.group(2))
        except Exception:
            chg = None
        if chg is not None:
            if any(w in direction for w in ("down", "fell", "lost")) and chg > 0:
                chg = -chg
            elif any(w in direction for w in ("up", "rose", "gained")) and chg < 0:
                chg = abs(chg)
            return {"level": None, "chg_pct": chg, "source": "moneycontrol_text"}

    # Keep a short Gift-related sentence if present without a parseable %.
    for line in (text or "").split("\n"):
        if "gift" in line.lower() and "nifty" in line.lower():
            return {
                "level": None,
                "chg_pct": None,
                "snippet": line.strip()[:220],
                "source": "moneycontrol_text",
            }
    return None


def _yf_last(ticker: str) -> dict[str, Any] | None:
    try:
        import yfinance as yf

        fi = yf.Ticker(ticker).fast_info
        last = float(getattr(fi, "last_price", 0) or 0)
        prev = float(getattr(fi, "previous_close", 0) or 0)
        if last > 0 and prev > 0:
            return {
                "ticker": ticker,
                "price": round(last, 2),
                "chg_pct": round((last - prev) / prev * 100.0, 2),
            }
    except Exception:
        return None
    return None


def build_premarket_cues(*, allow_network: bool = True) -> dict[str, Any]:
    """Compose Gift/premarket cues. Soft-fail every network path."""
    out: dict[str, Any] = {
        "available": False,
        "title": "GIFT NIFTY / PREMARKET CUES",
        "generated_at": _utc_now(),
        "gift_nifty": None,
        "us_futures": [],
        "headline": "",
        "bullets": [],
        "gaps": [],
        "source": "",
        "places_orders": False,
        "honesty": (
            "Premarket cues come from Moneycontrol / Yahoo when reachable. "
            "Missing Gift Nifty stays missing — never invents overnight levels."
        ),
    }
    if not allow_network:
        out["gaps"].append("Network disabled for premarket cues")
        return out

    text, err = _moneycontrol_premarket_text()
    gift = _parse_gift_nifty(text or "") if text else None
    if gift:
        out["gift_nifty"] = gift
        out["source"] = "moneycontrol"
        chg = gift.get("chg_pct")
        if chg is not None:
            direction = "down" if float(chg) < 0 else "up"
            level_bit = f" near {gift['level']:,.0f}" if gift.get("level") else ""
            out["bullets"].append(
                f"Gift Nifty is {direction} {abs(float(chg)):.2f}%{level_bit} (Moneycontrol pre-market)."
            )
        elif gift.get("snippet"):
            out["bullets"].append(str(gift["snippet"]))
    elif err:
        out["gaps"].append(f"Gift Nifty text unavailable: {err}")

    for label, ticker in (("S&P futures", "ES=F"), ("Nasdaq futures", "NQ=F")):
        row = _yf_last(ticker)
        if not row:
            out["gaps"].append(f"{label} unavailable")
            continue
        out["us_futures"].append({"name": label, **row})
        out["bullets"].append(
            f"{label} {row['chg_pct']:+.2f}% at {row['price']:,.2f} (overnight risk proxy)."
        )

    # Commodities that often decide the India open
    for label, ticker in (("Brent crude", "BZ=F"), ("WTI crude", "CL=F"), ("Gold", "GC=F")):
        row = _yf_last(ticker)
        if not row:
            continue
        out["bullets"].append(f"{label} {row['chg_pct']:+.2f}% at {row['price']:,.2f}.")

    out["available"] = bool(out["bullets"] or out["gift_nifty"] or out["us_futures"])
    if out["available"]:
        gift = out.get("gift_nifty") or {}
        if gift.get("chg_pct") is not None:
            out["headline"] = (
                f"Gift Nifty {float(gift['chg_pct']):+.2f}% — watch the open with US futures + commodities."
            )
        elif out["us_futures"]:
            top = out["us_futures"][0]
            out["headline"] = (
                f"{top['name']} {top['chg_pct']:+.2f}% overnight — India open inherits global risk tone."
            )
        else:
            out["headline"] = "Premarket cues available — see bullets."
    else:
        out["gaps"].append("No premarket / Gift Nifty evidence yet")
    out["bullets"] = out["bullets"][:8]
    return out
