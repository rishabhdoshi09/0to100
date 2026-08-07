"""Holdings desk — Zerodha book → fundamentals → technicals → news → research verdict.

Pipeline per held symbol:
  1. Load demat row (qty / avg / LTP)
  2. Fundamentals snapshot (Screener cache only — missing stays missing)
  3. Technical health (official daily history via buy_health.evaluate_symbol)
  4. News good/bad bias from curated articles mentioning the symbol
  5. Compose a research stance suggestion (HOLD / ADD_WATCH / TRIM_WATCH / EXIT_WATCH)

Never places orders. Stances are research suggestions for the Daily Pulse desk,
not live buy/sell tickets.
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
        "schema_version": 1,
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
        "message": message or "No holdings desk yet — sync Zerodha holdings, then Run holdings desk.",
        "places_orders": False,
        "honesty": (
            "Holdings desk composes demat book + fundamentals cache + technicals + curated news. "
            "Verdicts are research suggestions (buy/sell/hold-style watches) — never live orders. Paper-first."
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


def _compose_verdict(
    *,
    tech_sev: str,
    fund_sev: str,
    news_bias: str,
    vs_entry_pct: float | None,
    tech_available: bool,
) -> tuple[str, float, str, list[str]]:
    """Return (stance, confidence, thesis, suggestions). Research language only."""
    suggestions: list[str] = []
    if not tech_available and fund_sev == "unknown" and news_bias == "NONE":
        return (
            STANCE_INCOMPLETE,
            0.15,
            "Not enough fundamentals, technicals, or news to form a research stance.",
            ["Open Stock Intelligence and Retry fundamentals; ensure bhav history exists."],
        )

    # EXIT: hard technical damage + weak fund or bad news
    if tech_sev == "critical" and (fund_sev in {"warn", "critical"} or news_bias == "BAD"):
        suggestions = [
            "Research watch: plan an exit review — trend support looks broken.",
            "Confirm with your stop / risk plan before acting; desk never sells for you.",
        ]
        if news_bias == "BAD":
            suggestions.append("Recent headlines lean negative — read the top stories before deciding.")
        thesis = "Technicals at risk and fundamentals/news do not support holding blindly."
        return STANCE_EXIT_WATCH, 0.78, thesis, suggestions

    if tech_sev == "critical" and (vs_entry_pct is not None and vs_entry_pct <= -8.0):
        return (
            STANCE_EXIT_WATCH,
            0.72,
            "Price is in a hard drawdown vs your average and technical structure is damaged.",
            [
                "Research watch: consider an exit plan or hard stop review.",
                "Paper-first — no live sell is placed by QuantTerm.",
            ],
        )

    # TRIM: weakening tape, rich/weak fund, or bad news on a soft tape
    if tech_sev in {"warn", "critical"} or fund_sev == "critical" or (
        news_bias == "BAD" and tech_sev in {"warn", "info", "critical"}
    ):
        suggestions = [
            "Research watch: trim / tighten risk rather than add size.",
            "Re-check supports and your average cost before any action.",
        ]
        if fund_sev in {"warn", "critical"}:
            suggestions.append("Fundamentals flags are elevated on cached Screener ratios.")
        if news_bias == "BAD":
            suggestions.append("News bias is negative — treat bounce attempts as suspect until tone improves.")
        if vs_entry_pct is not None and vs_entry_pct >= 25:
            suggestions.append("Large unrealized gain — trimming into strength is a valid research option.")
        thesis = "Tape or fundamentals are soft — prefer risk reduction over fresh adds."
        conf = 0.66 if tech_sev == "critical" else 0.58
        return STANCE_TRIM_WATCH, conf, thesis, suggestions

    # ADD: healthy / info tech with constructive fund + non-negative news, preferably pullback
    pullback = vs_entry_pct is not None and -6.0 <= vs_entry_pct <= 2.0
    fund_ok = fund_sev in {"good", "info", "unknown"}
    news_ok = news_bias in {"GOOD", "NONE", "MIXED"}
    if tech_sev in {"good", "info"} and fund_ok and news_ok and news_bias != "BAD":
        if pullback or tech_sev == "good":
            suggestions = [
                "Research watch: hold core; consider adding only on a planned dip / confirmed support.",
                "Not a buy ticket — size and entry remain your decision; paper-first.",
            ]
            if news_bias == "GOOD":
                suggestions.append("Recent news lean positive — still wait for your level, don't chase.")
            if fund_sev == "good":
                suggestions.append("Cached fundamentals look constructive vs peers on available ratios.")
            thesis = (
                "Structure and fundamentals support keeping the position; adds only as a watch, not a chase."
                if pullback
                else "Healthy structure — hold; adds stay on a planned watchlist only."
            )
            return STANCE_ADD_WATCH if pullback or news_bias == "GOOD" else STANCE_HOLD, 0.62, thesis, suggestions

    # Default HOLD
    suggestions = [
        "Research stance: hold / monitor — no clear add or exit edge from current stores.",
        "Re-run holdings desk after fresh news refresh or fundamentals retry.",
    ]
    if news_bias == "MIXED":
        suggestions.append("News is mixed — wait for a cleaner story before changing size.")
    if fund_sev == "unknown":
        suggestions.append("Fundamentals cache missing — Retry fundamentals on Stock Intelligence.")
    thesis = "Mixed or incomplete evidence — default research stance is hold and monitor."
    return STANCE_HOLD, 0.5, thesis, suggestions


def evaluate_holding(row: Mapping[str, Any]) -> dict[str, Any]:
    """Run fund → tech → news → verdict for one demat row."""
    from product.buy_health import evaluate_symbol
    from product.holdings_book import research_symbol

    tradingsymbol = str(row.get("tradingsymbol") or row.get("symbol") or "").strip().upper()
    symbol = str(row.get("research_symbol") or research_symbol(tradingsymbol)).strip().upper()
    avg = float(row.get("average_price") or 0) or None
    ltp = float(row.get("last_price") or 0) or None
    qty = int(row.get("quantity") or 0)

    health = evaluate_symbol(symbol, entry_price=avg, live_price=ltp)
    fund = dict(health.get("fundamentals") or {})
    tech = dict(health.get("technicals") or {})
    news = _news_bias(symbol)

    tech_sev = str(tech.get("severity") or health.get("severity") or "unknown")
    fund_sev = str(fund.get("severity") or "unknown")
    vs_entry = health.get("vs_entry_pct")
    if vs_entry is None and avg and ltp and avg > 0:
        vs_entry = round((ltp / avg - 1.0) * 100.0, 2)

    stance, confidence, thesis, suggestions = _compose_verdict(
        tech_sev=tech_sev,
        fund_sev=fund_sev,
        news_bias=str(news.get("bias") or "NONE"),
        vs_entry_pct=float(vs_entry) if vs_entry is not None else None,
        tech_available=bool(tech.get("available") or health.get("available")),
    )

    # Map research stance to plain buy/sell/hold-style suggestion label for UI.
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
        "last_price": health.get("price") or ltp,
        "pnl": row.get("pnl"),
        "pnl_pct": row.get("pnl_pct"),
        "vs_entry_pct": vs_entry,
        "stance": stance,
        "suggestion": suggestion_label,
        "confidence": round(float(confidence), 2),
        "thesis": thesis,
        "suggestions": suggestions,
        "fundamentals": {
            "available": bool(fund.get("available")),
            "severity": fund_sev,
            "status": fund.get("status"),
            "ratios": fund.get("ratios") or {},
            "flags": (fund.get("flags") or [])[:4],
            "note": fund.get("note") or "",
        },
        "technicals": {
            "available": bool(tech.get("available") or health.get("available")),
            "severity": tech_sev,
            "status_label": tech.get("status_label") or health.get("status_label") or "INCOMPLETE",
            "risk_score": tech.get("risk_score") or health.get("risk_score"),
            "warnings": (tech.get("warnings") or health.get("warnings") or [])[:5],
            "structure": tech.get("structure") or health.get("structure") or {},
            "as_of": tech.get("as_of") or health.get("as_of") or "",
        },
        "news": news,
        "places_orders": False,
        "honesty": "Research suggestion only — not a live buy/sell order.",
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

    if not holdings:
        desk = empty_desk(
            message=conn.get("message")
            or "Holdings book empty — Sync Zerodha holdings on My Holdings, then re-run."
        )
        desk["connection"] = conn
        if persist:
            save_desk(desk, path)
        return desk

    rows = [evaluate_holding(row) for row in holdings]
    rows.sort(key=lambda r: (_STANCE_RANK.get(str(r.get("stance")), 9), str(r.get("symbol") or "")))

    summary = {k: 0 for k in _STANCE_RANK}
    for row in rows:
        key = str(row.get("stance") or STANCE_INCOMPLETE)
        summary[key] = int(summary.get(key) or 0) + 1

    desk = {
        "schema_version": 1,
        "available": True,
        "title": "HOLDINGS DESK",
        "generated_at": _utc_now(),
        "holdings_count": len(rows),
        "rows": rows,
        "summary": summary,
        "connection": conn,
        "message": (
            f"{len(rows)} holding(s) scored · "
            f"EXIT {summary.get(STANCE_EXIT_WATCH, 0)} · "
            f"TRIM {summary.get(STANCE_TRIM_WATCH, 0)} · "
            f"ADD {summary.get(STANCE_ADD_WATCH, 0)} · "
            f"HOLD {summary.get(STANCE_HOLD, 0)} · "
            f"INCOMPLETE {summary.get(STANCE_INCOMPLETE, 0)}"
        ),
        "places_orders": False,
        "honesty": (
            "Fund → tech → news → research verdict for your demat book. "
            "BUY/SELL/HOLD labels are suggestion watches only — QuantTerm never places orders. Paper-first."
        ),
    }
    if persist:
        save_desk(desk, path)
    return desk


def desk_telegram_message(desk: Mapping[str, Any] | None = None) -> str:
    payload = dict(desk or load_desk())
    rows = list(payload.get("rows") or [])
    lines = [
        "<b>Holdings Desk</b>",
        str(payload.get("message") or f"{len(rows)} holding(s)"),
        "<i>Research suggestions · not live orders</i>",
        "",
    ]
    if not rows:
        lines.append(str(payload.get("message") or "No holdings scored yet."))
        return "\n".join(lines)

    for row in rows[:14]:
        sym = row.get("symbol") or row.get("tradingsymbol")
        stance = row.get("suggestion") or row.get("stance")
        tech = ((row.get("technicals") or {}).get("status_label")) or "—"
        news = ((row.get("news") or {}).get("bias")) or "NONE"
        fund = ((row.get("fundamentals") or {}).get("severity")) or "—"
        lines.append(
            f"• <b>{sym}</b> → {stance}\n"
            f"  tech {tech} · fund {fund} · news {news}"
        )
        thesis = str(row.get("thesis") or "").strip()
        if thesis:
            lines.append(f"  {thesis[:140]}")
    if len(rows) > 14:
        lines.append(f"… +{len(rows) - 14} more")
    lines.append("\n<i>Paper-first · missing stays missing</i>")
    return "\n".join(lines)


def notify_holdings_desk_telegram(desk: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(desk or load_desk())
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
