"""Holdings desk — Zerodha book → fundamentals → technicals → news → research verdict.

Pipeline per held symbol:
  1. Load demat row (qty / avg / LTP)
  2. Fundamentals snapshot (Screener cache — PE/ROE/debt/growth; missing stays missing)
  3. Technical health + price range / stop / short-term target
  4. News good/bad bias from curated articles
  5. Market FII/DII context
  6. Compose a research stance with a 30-year desk brief (not a live order)

Horizon rule: prefer short-term trade framing for most names.
Long-term compounder exceptions: BSE, CDSL only.

Never places orders. Stances are research suggestions for Daily Pulse.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "logs" / "product" / "holdings_desk.json"

# Research stances — never order verbs for the live desk.
STANCE_HOLD = "HOLD"
STANCE_ADD_WATCH = "ADD_WATCH"
STANCE_TRIM_WATCH = "TRIM_WATCH"
STANCE_EXIT_WATCH = "EXIT_WATCH"
STANCE_INCOMPLETE = "INCOMPLETE"

# Prefer short-term trades everywhere except these compounder / franchise names.
LONG_TERM_EXCEPTIONS = frozenset({"BSE", "CDSL"})

_STANCE_RANK = {
    STANCE_EXIT_WATCH: 0,
    STANCE_TRIM_WATCH: 1,
    STANCE_ADD_WATCH: 2,
    STANCE_HOLD: 3,
    STANCE_INCOMPLETE: 4,
}


def desk_path(path: Path | None = None) -> Path:
    env = os.environ.get("QT_HOLDINGS_DESK_FILE", "").strip()
    if path is not None:
        return Path(path)
    if env:
        return Path(env)
    return DEFAULT_PATH


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def empty_desk(*, message: str = "") -> dict[str, Any]:
    return {
        "schema_version": 2,
        "available": False,
        "title": "HOLDINGS DESK",
        "generated_at": _utc_now(),
        "holdings_count": 0,
        "rows": [],
        "summary": {
            "HOLD": 0,
            "ADD_WATCH": 0,
            "TRIM_WATCH": 0,
            "EXIT_WATCH": 0,
            "INCOMPLETE": 0,
        },
        "market_flows": {},
        "message": message or "No holdings desk yet — sync Zerodha holdings, then Run holdings desk.",
        "places_orders": False,
        "honesty": (
            "Holdings desk composes demat book + fundamentals cache + technicals + curated news + FII/DII. "
            "Verdicts are research suggestions with price range/target — never live orders. "
            "Short-term trade bias except BSE/CDSL. Paper-first."
        ),
    }


def load_desk(path: Path | None = None) -> dict[str, Any]:
    target = desk_path(path)
    if not target.exists():
        return empty_desk()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return empty_desk(message="Holdings desk file unreadable")
        payload.setdefault("places_orders", False)
        return payload
    except Exception as exc:
        return empty_desk(message=f"Holdings desk load failed: {exc}")


def save_desk(payload: Mapping[str, Any], path: Path | None = None) -> dict[str, Any]:
    target = desk_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    out = dict(payload)
    out["places_orders"] = False
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return out


def _market_flows() -> dict[str, Any]:
    out: dict[str, Any] = {
        "available": False,
        "bias": "",
        "bias_label": "",
        "bias_note": "",
        "fii_net_cr": None,
        "dii_net_cr": None,
        "as_of": "",
    }
    try:
        from data.fii_dii_store import workspace_payload
        from data.institutional_flows import humanize_flow_bias

        flows = workspace_payload(days=30, allow_network=False, include_nifty_options=False)
        cash = flows.get("cash") if isinstance(flows.get("cash"), Mapping) else {}
        bias = str(cash.get("bias") or flows.get("bias") or "")
        plain = humanize_flow_bias(bias)
        today = cash.get("today") if isinstance(cash.get("today"), Mapping) else {}
        out.update(
            {
                "available": bool(flows.get("available") or cash.get("available")),
                "bias": bias,
                "bias_label": str(cash.get("bias_label") or flows.get("bias_label") or plain["bias_label"]),
                "bias_note": str(
                    cash.get("bias_note") or flows.get("bias_note") or flows.get("insight") or plain["bias_note"]
                ),
                "fii_net_cr": today.get("fii_net"),
                "dii_net_cr": today.get("dii_net"),
                "as_of": str(today.get("date") or ""),
            }
        )
    except Exception as exc:
        out["bias_note"] = f"FII/DII unavailable ({exc})"
    return out


def _news_bias(symbol: str, *, hours: int = 72) -> dict[str, Any]:
    """Score recent curated news for one symbol into GOOD / BAD / MIXED / NONE."""
    out: dict[str, Any] = {
        "available": False,
        "bias": "NONE",
        "label": "No recent news",
        "positive": 0,
        "negative": 0,
        "mixed": 0,
        "unclear": 0,
        "headlines": [],
        "hours": hours,
    }
    try:
        from news.curator_store import CuratorStore

        store = CuratorStore()
        articles = store.recent(hours=hours, limit=40, symbol=symbol)
    except Exception as exc:
        out["label"] = f"News store unavailable ({exc})"
        return out

    if not articles:
        return out

    pos = neg = mixed = unclear = 0
    headlines: list[dict[str, Any]] = []
    for art in articles:
        direction = str(getattr(art, "direction", "") or "unclear").lower()
        if direction == "likely_positive":
            pos += 1
            tone = "GOOD"
        elif direction == "likely_negative":
            neg += 1
            tone = "BAD"
        elif direction == "mixed":
            mixed += 1
            tone = "MIXED"
        else:
            unclear += 1
            tone = "UNCLEAR"
        headlines.append(
            {
                "title": str(getattr(art, "headline", "") or "")[:160],
                "direction": direction,
                "tone": tone,
                "source": str(getattr(art, "source", "") or getattr(art, "source_name", "") or ""),
                "impact_score": getattr(art, "impact_score", None),
                "published_at": str(getattr(art, "published_at", "") or ""),
            }
        )

    out.update(
        {
            "available": True,
            "positive": pos,
            "negative": neg,
            "mixed": mixed,
            "unclear": unclear,
            "headlines": headlines[:6],
        }
    )
    if neg > pos and neg >= 1:
        out["bias"] = "BAD"
        out["label"] = f"News lean negative ({neg} bad / {pos} good)"
    elif pos > neg and pos >= 1:
        out["bias"] = "GOOD"
        out["label"] = f"News lean positive ({pos} good / {neg} bad)"
    elif pos or neg or mixed:
        out["bias"] = "MIXED"
        out["label"] = f"News mixed ({pos} good / {neg} bad / {mixed} mixed)"
    else:
        out["bias"] = "NONE"
        out["label"] = f"{len(articles)} headline(s), direction unclear"
    return out


def _fund_brief(fund: Mapping[str, Any]) -> str:
    """One-line fundamentals talk-through from cached ratios — never invent numbers."""
    if not fund.get("available"):
        return "Fundamentals cache missing — no PE/ROE/debt call until Screener refresh."
    ratios = fund.get("ratios") if isinstance(fund.get("ratios"), Mapping) else {}
    bits: list[str] = []
    pe = ratios.get("pe")
    roe = ratios.get("roe")
    roce = ratios.get("roce")
    debt = ratios.get("debt_to_equity")
    sales = ratios.get("sales_growth_pct")
    profit = ratios.get("profit_growth_pct")
    if pe is not None:
        bits.append(f"PE {pe:.1f}x")
    if roe is not None:
        bits.append(f"ROE {roe:.1f}%")
    if roce is not None:
        bits.append(f"ROCE {roce:.1f}%")
    if debt is not None:
        bits.append(f"D/E {debt:.2f}")
    if sales is not None:
        bits.append(f"sales {sales:+.1f}%")
    if profit is not None:
        bits.append(f"profit {profit:+.1f}%")
    sev = str(fund.get("severity") or "info")
    tone = {
        "good": "constructive on cache",
        "info": "usable, not a screaming bargain or disaster",
        "warn": "flags on valuation/leverage/growth — respect them",
        "critical": "fundamental stress on cached ratios",
        "unknown": "incomplete",
    }.get(sev, "incomplete")
    if bits:
        return f"Fundamentals ({tone}): " + " · ".join(bits)
    return f"Fundamentals present but thin on ratios ({tone})."


def _price_plan(
    *,
    symbol: str,
    price: float | None,
    avg: float | None,
    health: Mapping[str, Any],
    long_term: bool,
) -> dict[str, Any]:
    """Support / resistance / stop / target from real averages — missing stays missing."""
    averages = health.get("averages") if isinstance(health.get("averages"), Mapping) else {}
    supports = health.get("supports") if isinstance(health.get("supports"), Mapping) else {}
    resistances = health.get("resistances") if isinstance(health.get("resistances"), Mapping) else {}
    if not resistances and isinstance(health.get("technicals"), Mapping):
        resistances = (health.get("technicals") or {}).get("resistances") or {}

    px = float(price) if price and price > 0 else None
    ema20 = averages.get("ema20")
    ema50 = averages.get("ema50")
    ema200 = averages.get("ema200")
    sup20 = supports.get("swing_20d")
    sup60 = supports.get("swing_60d")
    res20 = resistances.get("swing_20d")
    res60 = resistances.get("swing_60d")

    stop = None
    if sup20 and px:
        stop = round(float(sup20) * 0.995, 2)
    elif avg and avg > 0:
        stop = round(float(avg) * (0.92 if long_term else 0.96), 2)

    target = None
    target_note = ""
    if px:
        candidates: list[float] = []
        if ema20 and px < float(ema20):
            candidates.append(float(ema20))
            target_note = "first reclaim: 20-DMA"
        elif ema50 and px < float(ema50):
            candidates.append(float(ema50))
            target_note = "first reclaim: 50-DMA"
        if res20 and float(res20) > px:
            candidates.append(float(res20))
            if not target_note:
                target_note = "20-session swing high"
        if long_term and res60 and float(res60) > px:
            candidates.append(float(res60))
            if not target_note:
                target_note = "60-session swing high"
        if not candidates:
            # Short-term stretch target only when already above nearby averages — never invent alpha.
            stretch = 1.06 if not long_term else 1.12
            candidates.append(px * stretch)
            target_note = "measured stretch from last close (research only)"
        target = round(min(candidates), 2) if not long_term else round(max(candidates), 2)

    range_low = round(float(sup20), 2) if sup20 else (round(float(sup60), 2) if sup60 else None)
    range_high = round(float(res20), 2) if res20 else (round(float(res60), 2) if res60 else None)

    rr = None
    if px and stop and target and px > stop and target > px:
        rr = round((target - px) / (px - stop), 2)

    horizon = "LONG_TERM" if long_term else "SHORT_TERM"
    return {
        "horizon": horizon,
        "price": round(px, 2) if px else None,
        "average_cost": round(float(avg), 2) if avg else None,
        "range_low": range_low,
        "range_high": range_high,
        "stop_watch": stop,
        "target_watch": target,
        "target_note": target_note,
        "reward_risk": rr,
        "ema20": round(float(ema20), 2) if ema20 else None,
        "ema50": round(float(ema50), 2) if ema50 else None,
        "ema200": round(float(ema200), 2) if ema200 else None,
        "note": (
            f"{'Long-term franchise book' if long_term else 'Short-term trade book'}: "
            f"range ₹{range_low or '—'}–₹{range_high or '—'} · "
            f"stop watch ₹{stop or '—'} · target watch ₹{target or '—'}"
            + (f" · R:R {rr}" if rr is not None else "")
        ),
    }


def _compose_verdict(
    *,
    symbol: str,
    tech_sev: str,
    fund_sev: str,
    news_bias: str,
    vs_entry_pct: float | None,
    tech_available: bool,
    fund_brief: str,
    price_plan: Mapping[str, Any],
    flows: Mapping[str, Any],
    long_term: bool,
) -> tuple[str, float, str, list[str]]:
    """Return (stance, confidence, thesis, suggestions). Research language only."""
    horizon = "long-term compounder book (BSE/CDSL exception)" if long_term else "short-term trade book"
    flow_label = str(flows.get("bias_label") or flows.get("bias") or "Flows unknown")
    flow_code = str(flows.get("bias") or "")
    suggestions: list[str] = []

    if not tech_available and fund_sev == "unknown" and news_bias == "NONE":
        return (
            STANCE_INCOMPLETE,
            0.15,
            f"{symbol}: not enough fundamentals, technicals, or news for a {horizon} call.",
            [
                "Retry fundamentals on Stock Intelligence; ensure bhav history exists.",
                "Do not Telegram a verdict until the desk has numbers — analyse first.",
            ],
        )

    # EXIT
    if tech_sev == "critical" and (fund_sev in {"warn", "critical"} or news_bias == "BAD"):
        suggestions = [
            f"Research watch ({horizon}): plan an exit / hard stop review — structure is broken.",
            fund_brief,
            f"Price plan: {price_plan.get('note')}",
            "Confirm with your own risk plan — QuantTerm never sells for you.",
        ]
        if news_bias == "BAD":
            suggestions.append("Headlines lean negative — do not average down on hope.")
        if flow_code == "DISTRIBUTION":
            suggestions.append(f"Market flows: {flow_label} — institutional distribution argues against fighting the tape.")
        thesis = (
            f"{symbol}: technical damage + soft fundamentals/news. "
            f"{fund_brief} Prefer exit-watch over heroics on a {horizon}."
        )
        return STANCE_EXIT_WATCH, 0.78, thesis, suggestions

    if tech_sev == "critical" and (vs_entry_pct is not None and vs_entry_pct <= (-12.0 if long_term else -8.0)):
        return (
            STANCE_EXIT_WATCH,
            0.72,
            (
                f"{symbol}: hard drawdown vs your average and trend stack is damaged. "
                f"{fund_brief}"
            ),
            [
                f"Research watch ({horizon}): exit plan or stop review before adding risk.",
                f"Price plan: {price_plan.get('note')}",
                "Paper-first — no live sell is placed.",
            ],
        )

    # TRIM
    if tech_sev in {"warn", "critical"} or fund_sev == "critical" or (
        news_bias == "BAD" and tech_sev in {"warn", "info", "critical"}
    ):
        suggestions = [
            f"Research watch ({horizon}): trim / tighten risk — do not add size into weakness.",
            fund_brief,
            f"Price plan: {price_plan.get('note')}",
        ]
        if fund_sev in {"warn", "critical"}:
            suggestions.append("Fundamental flags are elevated on cached Screener ratios — valuation/leverage/growth matter here.")
        if news_bias == "BAD":
            suggestions.append("News bias negative — treat bounce attempts as suspect until tone improves.")
        if vs_entry_pct is not None and vs_entry_pct >= (40 if long_term else 18):
            suggestions.append("Large unrealized gain — booking partials into strength is a valid research option.")
        if flow_code == "FII_SELLING_DII_ABSORBING":
            suggestions.append(f"Market flows: {flow_label} — dips may find DII bids, but respect foreign selling.")
        elif flow_code == "DISTRIBUTION":
            suggestions.append(f"Market flows: {flow_label} — both sides selling; stay light.")
        thesis = (
            f"{symbol}: tape or fundamentals soft for a {horizon}. "
            f"{fund_brief} Prefer risk reduction over fresh adds."
        )
        conf = 0.66 if tech_sev == "critical" else 0.58
        return STANCE_TRIM_WATCH, conf, thesis, suggestions

    # ADD / HOLD
    pullback = vs_entry_pct is not None and ((-10.0 if long_term else -6.0) <= vs_entry_pct <= (3.0 if long_term else 2.0))
    near_support = False
    px = price_plan.get("price")
    low = price_plan.get("range_low")
    if px and low and float(low) > 0:
        near_support = (float(px) - float(low)) / float(low) <= 0.025

    fund_ok = fund_sev in {"good", "info", "unknown"}
    news_ok = news_bias in {"GOOD", "NONE", "MIXED"}
    if tech_sev in {"good", "info"} and fund_ok and news_ok and news_bias != "BAD":
        if long_term:
            suggestions = [
                "Long-term franchise (BSE/CDSL exception): hold core; accumulate only on planned dips near support.",
                fund_brief,
                f"Price plan: {price_plan.get('note')}",
                "Do not treat this like a 3-day momentum clip — franchise compounding needs patience.",
            ]
            if news_bias == "GOOD":
                suggestions.append("News constructive — still buy the level, not the headline.")
            if flow_code == "SUPPORTIVE":
                suggestions.append(f"Market flows: {flow_label} — institutional bid supportive for holds.")
            thesis = (
                f"{symbol}: franchise hold. {fund_brief} "
                f"Structure supports keeping the position; adds only near ₹{low or 'support'}."
            )
            stance = STANCE_ADD_WATCH if (pullback or near_support) and fund_sev in {"good", "info"} else STANCE_HOLD
            return stance, 0.64, thesis, suggestions

        # Short-term book
        suggestions = [
            "Short-term trade bias: define entry/stop/target before sizing — prefer a clean R:R ≥ 1.5.",
            fund_brief,
            f"Price plan: {price_plan.get('note')}",
        ]
        rr = price_plan.get("reward_risk")
        if rr is not None and float(rr) < 1.2:
            suggestions.append("Reward/risk looks thin from here — wait for a better level; do not chase.")
            stance = STANCE_HOLD
            thesis = (
                f"{symbol}: healthy but stretched for a short-term clip. {fund_brief} "
                "Hold / wait — add only if price gives a cleaner R:R."
            )
            return stance, 0.55, thesis, suggestions

        if pullback or near_support or (tech_sev == "good" and news_bias == "GOOD"):
            suggestions.append("Watch add zone near support / 20-DMA — not a market-order buy ticket.")
            if news_bias == "GOOD":
                suggestions.append("News lean positive — still wait for your level.")
            if flow_code == "DISTRIBUTION":
                suggestions.append(f"Market flows: {flow_label} — keep size small even if the chart looks fine.")
            thesis = (
                f"{symbol}: short-term setup watch. {fund_brief} "
                f"Plan toward ₹{price_plan.get('target_watch') or '—'} with stop ₹{price_plan.get('stop_watch') or '—'}."
            )
            return STANCE_ADD_WATCH, 0.62, thesis, suggestions

        suggestions.append("Hold the existing lot; fresh adds only on a planned dip.")
        thesis = f"{symbol}: structure OK for a hold. {fund_brief} Short-term adds stay on a watchlist only."
        return STANCE_HOLD, 0.58, thesis, suggestions

    suggestions = [
        f"Research stance ({horizon}): hold / monitor — no clear add or exit edge yet.",
        fund_brief,
        f"Price plan: {price_plan.get('note')}",
        "Re-run after fundamentals retry or news refresh — analyse before Telegram.",
    ]
    if news_bias == "MIXED":
        suggestions.append("News mixed — wait for a cleaner story before changing size.")
    if fund_sev == "unknown":
        suggestions.append("Fundamentals cache missing — Retry fundamentals on Stock Intelligence.")
    if flow_label:
        suggestions.append(f"Market flows: {flow_label}.")
    thesis = f"{symbol}: mixed evidence for a {horizon}. {fund_brief} Default = hold and monitor."
    return STANCE_HOLD, 0.5, thesis, suggestions


def evaluate_holding(
    row: Mapping[str, Any],
    *,
    flows: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run fund → tech → news → range/target → verdict for one demat row."""
    from product.buy_health import evaluate_symbol
    from product.holdings_book import research_symbol

    tradingsymbol = str(row.get("tradingsymbol") or row.get("symbol") or "").strip().upper()
    symbol = str(row.get("research_symbol") or research_symbol(tradingsymbol)).strip().upper()
    avg = float(row.get("average_price") or 0) or None
    ltp = float(row.get("last_price") or 0) or None
    qty = int(row.get("quantity") or 0)
    long_term = symbol in LONG_TERM_EXCEPTIONS
    market_flows = dict(flows or _market_flows())

    health = evaluate_symbol(symbol, entry_price=avg, live_price=ltp)
    fund = dict(health.get("fundamentals") or {})
    tech = dict(health.get("technicals") or {})
    news = _news_bias(symbol)
    fund_brief = _fund_brief(fund)

    tech_sev = str(tech.get("severity") or health.get("severity") or "unknown")
    fund_sev = str(fund.get("severity") or "unknown")
    vs_entry = health.get("vs_entry_pct")
    if vs_entry is None and avg and ltp and avg > 0:
        vs_entry = round((ltp / avg - 1.0) * 100.0, 2)

    price = health.get("price") or ltp
    price_plan = _price_plan(
        symbol=symbol,
        price=float(price) if price else None,
        avg=avg,
        health=health if health.get("averages") is not None else {**health, "averages": tech.get("averages") or {}, "supports": tech.get("supports") or health.get("supports") or {}, "resistances": tech.get("resistances") or health.get("resistances") or {}},
        long_term=long_term,
    )

    stance, confidence, thesis, suggestions = _compose_verdict(
        symbol=symbol,
        tech_sev=tech_sev,
        fund_sev=fund_sev,
        news_bias=str(news.get("bias") or "NONE"),
        vs_entry_pct=float(vs_entry) if vs_entry is not None else None,
        tech_available=bool(tech.get("available") or health.get("available")),
        fund_brief=fund_brief,
        price_plan=price_plan,
        flows=market_flows,
        long_term=long_term,
    )

    suggestion_label = {
        STANCE_HOLD: "HOLD",
        STANCE_ADD_WATCH: "BUY / ADD (watch)",
        STANCE_TRIM_WATCH: "TRIM / REDUCE (watch)",
        STANCE_EXIT_WATCH: "SELL / EXIT (watch)",
        STANCE_INCOMPLETE: "INCOMPLETE",
    }.get(stance, "HOLD")

    return {
        "tradingsymbol": tradingsymbol,
        "symbol": symbol,
        "quantity": qty,
        "average_price": avg,
        "last_price": price,
        "pnl": row.get("pnl"),
        "pnl_pct": row.get("pnl_pct"),
        "vs_entry_pct": vs_entry,
        "horizon": "LONG_TERM" if long_term else "SHORT_TERM",
        "stance": stance,
        "suggestion": suggestion_label,
        "confidence": round(float(confidence), 2),
        "thesis": thesis,
        "fund_brief": fund_brief,
        "suggestions": suggestions,
        "price_plan": price_plan,
        "fundamentals": {
            "available": bool(fund.get("available")),
            "severity": fund_sev,
            "status": fund.get("status"),
            "ratios": fund.get("ratios") or {},
            "flags": (fund.get("flags") or [])[:4],
            "brief": fund_brief,
            "note": fund.get("note") or "",
        },
        "technicals": {
            "available": bool(tech.get("available") or health.get("available")),
            "severity": tech_sev,
            "status_label": tech.get("status_label") or health.get("status_label") or "INCOMPLETE",
            "risk_score": tech.get("risk_score") or health.get("risk_score"),
            "warnings": (tech.get("warnings") or health.get("warnings") or [])[:5],
            "structure": tech.get("structure") or health.get("structure") or {},
            "supports": tech.get("supports") or health.get("supports") or {},
            "resistances": tech.get("resistances") or health.get("resistances") or {},
            "as_of": tech.get("as_of") or health.get("as_of") or "",
        },
        "news": news,
        "market_flows": {
            "bias": market_flows.get("bias"),
            "bias_label": market_flows.get("bias_label"),
            "as_of": market_flows.get("as_of"),
        },
        "places_orders": False,
        "honesty": "Research suggestion only — not a live buy/sell order. Analyse before Telegram.",
    }


def run_holdings_desk(
    *,
    persist: bool = True,
    path: Path | None = None,
    holdings_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate the full Zerodha holdings book into research verdicts."""
    from product.holdings_book import build_holdings_payload, connection_status

    book = build_holdings_payload(holdings_path)
    holdings = list(book.get("holdings") or [])
    conn = connection_status(holdings_path)
    flows = _market_flows()

    if not holdings:
        desk = empty_desk(
            message=conn.get("message")
            or "Holdings book empty — Sync Zerodha holdings on My Holdings, then re-run."
        )
        desk["connection"] = conn
        desk["market_flows"] = flows
        if persist:
            save_desk(desk, path)
        return desk

    rows = [evaluate_holding(row, flows=flows) for row in holdings]
    rows.sort(key=lambda r: (_STANCE_RANK.get(str(r.get("stance")), 9), str(r.get("symbol") or "")))

    summary = {k: 0 for k in _STANCE_RANK}
    for row in rows:
        key = str(row.get("stance") or STANCE_INCOMPLETE)
        summary[key] = int(summary.get(key) or 0) + 1

    flow_bit = flows.get("bias_label") or "flows n/a"
    desk = {
        "schema_version": 2,
        "available": True,
        "title": "HOLDINGS DESK",
        "generated_at": _utc_now(),
        "holdings_count": len(rows),
        "rows": rows,
        "summary": summary,
        "market_flows": flows,
        "connection": conn,
        "message": (
            f"{len(rows)} holding(s) scored · FII/DII: {flow_bit} · "
            f"EXIT {summary.get(STANCE_EXIT_WATCH, 0)} · "
            f"TRIM {summary.get(STANCE_TRIM_WATCH, 0)} · "
            f"ADD {summary.get(STANCE_ADD_WATCH, 0)} · "
            f"HOLD {summary.get(STANCE_HOLD, 0)} · "
            f"INCOMPLETE {summary.get(STANCE_INCOMPLETE, 0)}"
        ),
        "places_orders": False,
        "honesty": (
            "Fund → tech → news → range/target → research verdict. "
            "Short-term trade bias except BSE/CDSL. "
            "BUY/SELL/HOLD labels are suggestion watches only — never live orders. "
            "Analyse the desk before sending Telegram."
        ),
    }
    if persist:
        save_desk(desk, path)
    return desk


def desk_telegram_message(desk: Mapping[str, Any] | None = None) -> str:
    """HTML Telegram brief — escape dynamic text so parse_mode stays valid."""
    from alerts.telegram_alerts import escape_html as esc

    payload = dict(desk or load_desk())
    rows = list(payload.get("rows") or [])
    flows = payload.get("market_flows") if isinstance(payload.get("market_flows"), Mapping) else {}
    flow_label = esc(flows.get("bias_label") or flows.get("bias") or "n/a")

    lines = [
        "<b>Holdings Desk</b> — research brief",
        esc(str(payload.get("message") or f"{len(rows)} holding(s)")),
        f"FII/DII: <b>{flow_label}</b>"
        + (f" · {esc(flows.get('as_of'))}" if flows.get("as_of") else ""),
        "<i>Analyse-first · not live orders · short-term bias except BSE/CDSL</i>",
        "",
    ]
    if not rows:
        lines.append(esc(str(payload.get("message") or "No holdings scored yet.")))
        return "\n".join(lines)

    for row in rows[:12]:
        sym = esc(row.get("symbol") or row.get("tradingsymbol"))
        stance = esc(row.get("suggestion") or row.get("stance"))
        horizon = esc(row.get("horizon") or "SHORT_TERM")
        plan = row.get("price_plan") if isinstance(row.get("price_plan"), Mapping) else {}
        news = ((row.get("news") or {}).get("bias")) or "NONE"
        lines.append(f"• <b>{sym}</b> → {stance} · {horizon}")
        lines.append(f"  {esc(row.get('fund_brief') or (row.get('fundamentals') or {}).get('brief') or '')}")
        if plan:
            lines.append(
                f"  Range ₹{esc(plan.get('range_low') or '—')}–₹{esc(plan.get('range_high') or '—')} · "
                f"stop ₹{esc(plan.get('stop_watch') or '—')} · "
                f"target ₹{esc(plan.get('target_watch') or '—')} · news {esc(news)}"
            )
        thesis = str(row.get("thesis") or "").strip()
        if thesis:
            lines.append(f"  {esc(thesis[:180])}")
    if len(rows) > 12:
        lines.append(f"… +{len(rows) - 12} more")
    lines.append("\n<i>Paper-first · missing stays missing · no rush Telegram without analysis</i>")
    return "\n".join(lines)


def notify_holdings_desk_telegram(desk: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(desk or load_desk())
    if not payload.get("available") or not list(payload.get("rows") or []):
        return {
            "sent": False,
            "configured": True,
            "reason": "Holdings desk not analysed yet — Run holdings desk first, then send.",
            "count": 0,
        }
    incomplete = sum(1 for r in payload.get("rows") or [] if r.get("stance") == STANCE_INCOMPLETE)
    scored = len(payload.get("rows") or [])
    if scored and incomplete == scored:
        return {
            "sent": False,
            "configured": True,
            "reason": "All rows incomplete — retry fundamentals/bhav before Telegram.",
            "count": scored,
        }
    try:
        from alerts.telegram_alerts import AlertEngine

        engine = AlertEngine()
        if not engine.is_configured():
            return {
                "sent": False,
                "reason": "Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)",
                "configured": False,
            }
    except Exception as exc:
        return {"sent": False, "reason": str(exc), "configured": False}

    ok = bool(engine.send(desk_telegram_message(payload)))
    return {
        "sent": ok,
        "configured": True,
        "count": len(payload.get("rows") or []),
        "reason": None if ok else (engine.last_error or "Telegram send failed"),
        "source": getattr(engine, "cred_source", "") or "",
    }
