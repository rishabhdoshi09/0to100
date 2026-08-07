"""Premarket / GIFT Nifty cues — multi-source, soft-fail, never invents levels.

Source order (hard Gift print):
  1. Moneycontrol pre-market HTML (when reachable)
  2. Retail market RSS (ET / Moneycontrol / Mint / BS / CNBC)
  3. Google News Gift Nifty query (headline consensus on pts / %)
  4. Last-good durable cache (labeled stale)

Overnight risk (never labeled as Gift):
  • US index futures ES/NQ via yfinance
  • Brent / WTI / Gold via yfinance

Missing stays missing. Not a buy ticket.
"""
from __future__ import annotations

import json
import os
import re
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LAST_GOOD_PATH = ROOT / "logs" / "product" / "premarket_cues_last_good.json"

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_HEADERS = {
    "User-Agent": _UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
}

_RSS_FEEDS = (
    ("et_markets", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("moneycontrol_markets", "https://www.moneycontrol.com/rss/marketreports.xml"),
    ("mint_markets", "https://www.livemint.com/rss/markets"),
    ("bs_markets", "https://www.business-standard.com/rss/markets-106.rss"),
    ("cnbctv18", "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/market.xml"),
    (
        "google_gift",
        "https://news.google.com/rss/search?q=%22Gift+Nifty%22+OR+%22GIFT+Nifty%22+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _low_power() -> bool:
    return os.environ.get("QT_LOW_POWER", "").strip().lower() in {"1", "true", "yes", "on"}


def _timeout() -> float:
    return 8.0 if _low_power() else 12.0


def _http_get(url: str, *, timeout: float | None = None) -> tuple[str | None, str | None]:
    try:
        import requests
    except Exception as exc:
        return None, f"requests unavailable ({exc})"
    try:
        resp = requests.Session().get(url, headers=_HEADERS, timeout=timeout or _timeout())
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"
        return resp.text, None
    except Exception as exc:
        return None, str(exc)


def _save_last_good(payload: dict[str, Any]) -> None:
    try:
        LAST_GOOD_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = LAST_GOOD_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, LAST_GOOD_PATH)
    except Exception:
        pass


def _load_last_good() -> dict[str, Any] | None:
    try:
        if not LAST_GOOD_PATH.exists():
            return None
        data = json.loads(LAST_GOOD_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _nifty_spot() -> float | None:
    try:
        from data.live_quotes import get_index_quotes

        q = (get_index_quotes(["NIFTY"]) or {}).get("NIFTY") or {}
        px = q.get("price")
        if px:
            return float(px)
    except Exception:
        pass
    # Yahoo cash Nifty — used only as reference to convert Gift *points* → %
    try:
        row = _yf_last("^NSEI")
        if row and row.get("price"):
            return float(row["price"])
    except Exception:
        pass
    return None


def _parse_gift_nifty(text: str) -> dict[str, Any] | None:
    """Extract Gift Nifty level / change when the page states them plainly."""
    level_pct = re.compile(
        r"Gift\s*Nifty[^\n]{0,40}?(-?\d{2}(?:,\d{3})*(?:\.\d+)?)[^\n]{0,20}?\((-?\d+(?:\.\d+)?)\s*%\)",
        re.I,
    )
    m = level_pct.search(text or "")
    if m:
        try:
            level = float(str(m.group(1)).replace(",", ""))
            chg = float(m.group(2))
            return {"level": level, "chg_pct": chg, "chg_points": None, "source": "moneycontrol_text"}
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
            return {"level": None, "chg_pct": chg, "chg_points": None, "source": "moneycontrol_text"}

    for line in (text or "").split("\n"):
        if "gift" in line.lower() and "nifty" in line.lower():
            return {
                "level": None,
                "chg_pct": None,
                "chg_points": None,
                "snippet": line.strip()[:220],
                "source": "moneycontrol_text",
            }
    return None


def _parse_gift_headline(text: str, *, source: str) -> dict[str, Any] | None:
    """Parse Gift move from a news headline/description. Never invents."""
    blob = re.sub(r"<[^>]+>", " ", text or "")
    blob = re.sub(r"\s+", " ", blob).strip()
    if not re.search(r"gift\s*nifty", blob, re.I):
        return None

    # Explicit percent
    m = re.search(
        r"Gift\s*Nifty[^\n%]{0,60}?((?:up|down|rose|fell|gained|lost)\s+)?(-?\d+(?:\.\d+)?)\s*%",
        blob,
        re.I,
    )
    if m:
        direction = (m.group(1) or "").lower()
        chg = float(m.group(2))
        if any(w in direction for w in ("down", "fell", "lost")) and chg > 0:
            chg = -chg
        return {
            "level": None,
            "chg_pct": chg,
            "chg_points": None,
            "snippet": blob[:220],
            "source": source,
        }

    # "trades 97 points lower/higher"
    m = re.search(
        r"(?:trades?|trading)\s+(\d+(?:\.\d+)?)\s*points?\s+(lower|higher|down|up)",
        blob,
        re.I,
    )
    if m:
        pts = float(m.group(1))
        if m.group(2).lower() in {"lower", "down"}:
            pts = -pts
        return {
            "level": None,
            "chg_pct": None,
            "chg_points": pts,
            "snippet": blob[:220],
            "source": source,
        }

    # "falls/down 100 pts" / "jumps/up 150 pts" / "down 93 pts"
    m = re.search(
        r"(?:falls?|fell|down|drops?|slips?|declines?)\s+(\d+(?:\.\d+)?)\s*(?:pts?|points?)",
        blob,
        re.I,
    )
    if m:
        return {
            "level": None,
            "chg_pct": None,
            "chg_points": -float(m.group(1)),
            "snippet": blob[:220],
            "source": source,
        }
    m = re.search(
        r"(?:jumps?|rises?|gains?|up|climbs?)\s+(?:over\s+)?(\d+(?:\.\d+)?)\s*(?:pts?|points?)",
        blob,
        re.I,
    )
    if m:
        return {
            "level": None,
            "chg_pct": None,
            "chg_points": float(m.group(1)),
            "snippet": blob[:220],
            "source": source,
        }

    # Soft directional only — no number
    soft = None
    low = blob.lower()
    if any(w in low for w in ("gap-down", "gap down", "negative start", "muted start", "weak start")):
        soft = "negative"
    elif any(w in low for w in ("gap-up", "gap up", "positive start", "higher open")):
        soft = "positive"
    elif "flat" in low or "quiet" in low:
        soft = "flat"
    if soft:
        return {
            "level": None,
            "chg_pct": None,
            "chg_points": None,
            "bias": soft,
            "snippet": blob[:220],
            "source": source,
        }
    return {
        "level": None,
        "chg_pct": None,
        "chg_points": None,
        "snippet": blob[:220],
        "source": source,
    }


def _enrich_gift_points(gift: dict[str, Any], nifty_spot: float | None) -> dict[str, Any]:
    """Convert point move → approx % using Nifty spot when Gift % missing."""
    out = dict(gift)
    pts = out.get("chg_points")
    if out.get("chg_pct") is None and pts is not None and nifty_spot and nifty_spot > 0:
        out["chg_pct"] = round(float(pts) / float(nifty_spot) * 100.0, 2)
        out["chg_pct_method"] = "points_vs_nifty_spot"
        out["nifty_spot_ref"] = round(float(nifty_spot), 2)
    if out.get("level") is None and pts is not None and nifty_spot and nifty_spot > 0:
        out["level"] = round(float(nifty_spot) + float(pts), 2)
        out["level_method"] = "nifty_spot_plus_gift_points"
    return out


def _moneycontrol_premarket_text() -> tuple[str | None, str | None]:
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:
        return None, f"scrape deps unavailable ({exc})"
    url = "https://www.moneycontrol.com/pre-market/"
    text, err = _http_get(url)
    if not text:
        return None, err or "empty"
    try:
        soup = BeautifulSoup(text, "html.parser")
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


def _rss_gift_candidates(max_feeds: int | None = None) -> list[dict[str, Any]]:
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return []
    feeds = list(_RSS_FEEDS)
    if max_feeds is not None:
        feeds = feeds[:max_feeds]
    found: list[dict[str, Any]] = []

    def _one(key_url: tuple[str, str]) -> list[dict[str, Any]]:
        key, url = key_url
        body, err = _http_get(url, timeout=_timeout())
        if not body:
            return []
        out: list[dict[str, Any]] = []
        try:
            soup = BeautifulSoup(body, "xml")
            items = soup.find_all("item")[:25]
            if not items:
                soup = BeautifulSoup(body, "html.parser")
                items = soup.find_all("item")[:25]
            for item in items:
                title = item.title.get_text(" ", strip=True) if item.title else ""
                desc = item.description.get_text(" ", strip=True) if item.description else ""
                parsed = _parse_gift_headline(f"{title}. {desc}", source=f"rss:{key}")
                if parsed and (
                    parsed.get("chg_pct") is not None
                    or parsed.get("chg_points") is not None
                    or parsed.get("bias")
                ):
                    parsed["headline"] = title[:180]
                    out.append(parsed)
        except Exception:
            return []
        return out

    workers = 3 if _low_power() else 6
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one, fu) for fu in feeds]
        for fut in as_completed(futs):
            try:
                found.extend(fut.result() or [])
            except Exception:
                continue
    return found


def _consensus_gift(candidates: list[dict[str, Any]], nifty_spot: float | None) -> dict[str, Any] | None:
    """Median of numeric Gift prints across headlines — best-in-class retail consensus."""
    if not candidates:
        return None
    numeric = []
    for c in candidates:
        enriched = _enrich_gift_points(c, nifty_spot)
        if enriched.get("chg_pct") is not None or enriched.get("chg_points") is not None:
            numeric.append(enriched)
    if numeric:
        pts = [float(c["chg_points"]) for c in numeric if c.get("chg_points") is not None]
        pcts = [float(c["chg_pct"]) for c in numeric if c.get("chg_pct") is not None]
        med_pts = statistics.median(pts) if pts else None
        med_pct = statistics.median(pcts) if pcts else None
        base = dict(numeric[0])
        if med_pts is not None:
            base["chg_points"] = round(float(med_pts), 1)
        if med_pct is not None:
            base["chg_pct"] = round(float(med_pct), 2)
        elif med_pts is not None:
            base = _enrich_gift_points(base, nifty_spot)
        base["source"] = "headline_consensus"
        base["evidence_count"] = len(numeric)
        base["evidence_sources"] = sorted({str(c.get("source")) for c in numeric})[:8]
        # Prefer a concrete snippet from a numeric headline
        for c in numeric:
            if c.get("headline"):
                base["snippet"] = c["headline"][:220]
                break
        return base

    # Soft bias only
    for c in candidates:
        if c.get("bias"):
            return {
                "level": None,
                "chg_pct": None,
                "chg_points": None,
                "bias": c.get("bias"),
                "snippet": c.get("snippet") or c.get("headline") or "",
                "source": c.get("source") or "rss",
                "evidence_count": 1,
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


def _fetch_futures_and_cmds(*, low_power: bool) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Parallel Yahoo prints. Returns (futures, commodity_bullets, gaps)."""
    futures: list[dict[str, Any]] = []
    cmd_bullets: list[str] = []
    gaps: list[str] = []
    pairs = [("S&P futures", "ES=F"), ("Nasdaq futures", "NQ=F")]
    cmds = [("Brent crude", "BZ=F"), ("Gold", "GC=F")]
    if not low_power:
        cmds.append(("WTI crude", "CL=F"))

    def _one(label: str, ticker: str) -> tuple[str, str, dict[str, Any] | None]:
        return label, ticker, _yf_last(ticker)

    with ThreadPoolExecutor(max_workers=4 if low_power else 6) as pool:
        futs = [pool.submit(_one, lab, tk) for lab, tk in pairs + cmds]
        for fut in as_completed(futs):
            try:
                label, _tk, row = fut.result()
            except Exception as exc:
                gaps.append(str(exc))
                continue
            if not row:
                gaps.append(f"{label} unavailable")
                continue
            if label.endswith("futures"):
                futures.append({"name": label, **row})
            else:
                cmd_bullets.append(f"{label} {row['chg_pct']:+.2f}% at {row['price']:,.2f}.")
    futures.sort(key=lambda r: 0 if "S&P" in r.get("name", "") else 1)
    return futures, cmd_bullets, gaps


def build_premarket_cues(*, allow_network: bool = True) -> dict[str, Any]:
    """Compose Gift/premarket cues. Soft-fail every network path."""
    out: dict[str, Any] = {
        "available": False,
        "title": "GIFT NIFTY / PREMARKET CUES",
        "generated_at": _utc_now(),
        "gift_nifty": None,
        "gift_hard": False,
        "us_futures": [],
        "headline": "",
        "bullets": [],
        "gaps": [],
        "source": "",
        "sources_tried": [],
        "stale": False,
        "places_orders": False,
        "honesty": (
            "Premarket cues come from Moneycontrol / retail RSS / Google News / Yahoo when reachable. "
            "Gift prints require a parseable level, %, or point move — never invents overnight levels. "
            "US futures are overnight risk proxies, not Gift Nifty."
        ),
    }
    if not allow_network:
        cached = _load_last_good()
        if cached and (cached.get("gift_nifty") or cached.get("us_futures") or cached.get("bullets")):
            cached = dict(cached)
            cached["stale"] = True
            cached["gaps"] = list(cached.get("gaps") or []) + ["Network disabled — serving last-good premarket cache"]
            cached["honesty"] = out["honesty"] + " Serving durable last-good cache (network off)."
            return cached
        out["gaps"].append("Network disabled for premarket cues")
        return out

    low_power = _low_power()
    nifty_spot = _nifty_spot()
    gift: dict[str, Any] | None = None

    # 1) Moneycontrol HTML
    text, err = _moneycontrol_premarket_text()
    out["sources_tried"].append("moneycontrol_premarket")
    if text:
        gift = _parse_gift_nifty(text)
        if gift and (gift.get("chg_pct") is not None or gift.get("level") is not None):
            gift = _enrich_gift_points(gift, nifty_spot)
            out["source"] = "moneycontrol"
        elif gift and not gift.get("chg_pct") and not gift.get("level"):
            # Keep snippet but keep searching for numeric
            snippet_only = gift
            gift = None
        else:
            snippet_only = None
    else:
        snippet_only = None
        out["gaps"].append(f"Moneycontrol pre-market: {err or 'unavailable'}")

    # 2) RSS + Google News consensus
    if gift is None or gift.get("chg_pct") is None:
        out["sources_tried"].append("rss_consensus")
        max_feeds = 3 if low_power else None
        candidates = _rss_gift_candidates(max_feeds=max_feeds)
        consensus = _consensus_gift(candidates, nifty_spot)
        if consensus and (
            consensus.get("chg_pct") is not None
            or consensus.get("chg_points") is not None
        ):
            gift = consensus
            out["source"] = consensus.get("source") or "headline_consensus"
        elif consensus and consensus.get("bias") and gift is None:
            gift = consensus
            out["source"] = consensus.get("source") or "rss"
        elif not candidates:
            out["gaps"].append("No Gift Nifty headlines with parseable move")

    if gift is None and snippet_only is not None:
        gift = snippet_only
        out["source"] = "moneycontrol"

    if gift:
        out["gift_nifty"] = gift
        hard = gift.get("chg_pct") is not None or gift.get("level") is not None or gift.get("chg_points") is not None
        out["gift_hard"] = bool(hard)
        chg = gift.get("chg_pct")
        pts = gift.get("chg_points")
        if chg is not None:
            direction = "down" if float(chg) < 0 else "up"
            level_bit = f" near {gift['level']:,.0f}" if gift.get("level") else ""
            pts_bit = f" ({float(pts):+.0f} pts)" if pts is not None else ""
            src = gift.get("source") or out["source"] or "wire"
            evidence = gift.get("evidence_count")
            evid_bit = f" · {evidence} headline prints" if evidence else ""
            out["bullets"].append(
                f"Gift Nifty is {direction} {abs(float(chg)):.2f}%{level_bit}{pts_bit} "
                f"({src}{evid_bit})."
            )
        elif pts is not None:
            direction = "down" if float(pts) < 0 else "up"
            out["bullets"].append(
                f"Gift Nifty {direction} {abs(float(pts)):.0f} pts "
                f"({gift.get('source') or 'headline'} — % pending Nifty spot)."
            )
        elif gift.get("bias"):
            out["bullets"].append(
                f"Gift Nifty headlines lean {gift['bias']} — no printed level/% yet "
                f"({gift.get('source') or 'rss'})."
            )
        elif gift.get("snippet"):
            out["bullets"].append(str(gift["snippet"]))

    # 3) Overnight risk proxies (never sold as Gift)
    futures, cmd_bullets, yf_gaps = _fetch_futures_and_cmds(low_power=low_power)
    out["us_futures"] = futures
    for row in futures:
        out["bullets"].append(
            f"{row['name']} {row['chg_pct']:+.2f}% at {row['price']:,.2f} (overnight risk proxy)."
        )
    out["bullets"].extend(cmd_bullets[:3])
    out["gaps"].extend(yf_gaps[:4])

    out["available"] = bool(out["bullets"] or out["gift_nifty"] or out["us_futures"])
    if out["available"]:
        g = out.get("gift_nifty") or {}
        if g.get("chg_pct") is not None:
            out["headline"] = (
                f"Gift Nifty {float(g['chg_pct']):+.2f}% — watch the open with US futures + commodities."
            )
        elif g.get("chg_points") is not None:
            out["headline"] = (
                f"Gift Nifty {float(g['chg_points']):+.0f} pts — open inherits Gift + global risk tone."
            )
        elif futures:
            top = futures[0]
            out["headline"] = (
                f"{top['name']} {top['chg_pct']:+.2f}% overnight — India open inherits global risk tone "
                "(Gift print incomplete)."
            )
        else:
            out["headline"] = "Premarket cues available — see bullets."
    else:
        # 4) Last-good fallback
        cached = _load_last_good()
        if cached and cached.get("available"):
            cached = dict(cached)
            cached["stale"] = True
            cached["gaps"] = list(cached.get("gaps") or []) + ["Fresh scrape empty — serving last-good cache"]
            return cached
        out["gaps"].append("No premarket / Gift Nifty evidence yet")

    out["bullets"] = out["bullets"][:8]
    if out["available"] and out.get("gift_hard"):
        _save_last_good(out)
    elif out["available"] and out.get("us_futures"):
        _save_last_good(out)
    return out
