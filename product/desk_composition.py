"""Bloomberg-style desk composition — reuse every store constructively.

Nothing here invents prices, CAs, guidance, or LIVE fills. Each block fails soft
so a missing store never blanks the whole desk. Heavy network probes are skipped
under QT_LOW_POWER.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
PULSE_PATH = ROOT / "logs" / "product" / "latest_street_pulse.json"

_BRAIN_CACHE: dict[str, Any] = {"ts": 0.0, "payload": None}
_BRAIN_TTL_S = 90.0


def _low_power() -> bool:
    return str(os.getenv("QT_LOW_POWER", "") or "").strip().lower() in {"1", "true", "yes"}


def _f(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _evidence_completeness(symbol: str) -> dict[str, Any]:
    try:
        from reporting.evidence_intake import evidence_requirements

        status = evidence_requirements(symbol)
    except Exception as exc:
        return {
            "available": False,
            "score_pct": None,
            "fresh": 0,
            "missing": [],
            "note": f"Evidence desk unavailable ({exc})",
        }
    reqs = list(status.get("requirements") or [])
    fresh = sum(1 for r in reqs if r.get("available") and r.get("status") == "FRESH")
    attached = sum(1 for r in reqs if r.get("source_attached") or r.get("available"))
    missing = [str(r.get("label") or r.get("key")) for r in reqs if not r.get("available") and not r.get("source_attached")]
    score = int(status.get("coverage_pct") or 0)
    return {
        "available": True,
        "score_pct": score,
        "fresh": fresh,
        "attached": attached,
        "total": len(reqs),
        "missing": missing[:8],
        "note": (
            f"{score}% research coverage · {fresh}/{len(reqs)} fresh · "
            f"{attached} sources attached"
        ),
    }


def _buy_context(symbol: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "in_buy_book": False,
        "in_holdings": False,
        "entry_price": None,
        "stop_price": None,
        "quantity": None,
        "avg_price": None,
        "health": None,
    }
    entry = None
    stop = None
    try:
        from product.buy_book import load_book

        for item in load_book().get("items") or []:
            if str(item.get("symbol") or "").upper() != symbol:
                continue
            if str(item.get("status") or "active").lower() not in {"active", "watching", ""}:
                continue
            out["in_buy_book"] = True
            entry = _f(item.get("entry_price"))
            stop = _f(item.get("stop_price"))
            out["entry_price"] = entry
            out["stop_price"] = stop
            out["quantity"] = _f(item.get("quantity"))
            break
    except Exception:
        pass
    try:
        from product.holdings_book import build_holdings_payload

        from product.holdings_book import research_symbol

        holdings = build_holdings_payload()
        for item in holdings.get("holdings") or holdings.get("items") or []:
            held = research_symbol(str(item.get("tradingsymbol") or item.get("symbol") or ""))
            if held != symbol:
                continue
            out["in_holdings"] = True
            out["avg_price"] = _f(item.get("average_price") or item.get("avg_price"))
            out["quantity"] = out["quantity"] or _f(item.get("quantity"))
            if entry is None:
                entry = out["avg_price"]
            break
    except Exception:
        pass
    if out["in_buy_book"] or out["in_holdings"]:
        try:
            from product.buy_health import evaluate_symbol

            out["health"] = evaluate_symbol(symbol, entry_price=entry, stop_price=stop)
        except Exception as exc:
            out["health"] = {"available": False, "warnings": [{"severity": "info", "text": str(exc)}]}
    return out


def _earnings_context(symbol: str) -> dict[str, Any]:
    if _low_power():
        return {
            "available": False,
            "risk_level": "NONE",
            "note": "Earnings probe skipped in low-power mode (no extra network).",
            "confidence": "SKIPPED",
        }
    try:
        from data.earnings_calendar import get_earnings_risk

        risk = get_earnings_risk(symbol)
        return {**risk, "available": risk.get("risk_level") not in (None, "")}
    except Exception as exc:
        return {
            "available": False,
            "risk_level": "NONE",
            "note": f"Earnings date unavailable ({exc})",
            "confidence": "UNKNOWN",
        }


def _flow_context(symbol: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "available": False,
        "bias": "",
        "note": "",
        "latest_fii_net_cr": None,
        "latest_dii_net_cr": None,
        "as_of": "",
        "bulk_deals": [],
    }
    try:
        from data.fii_dii_store import workspace_payload

        # Never block the desk on NSE network — cache/SQLite only.
        flows = workspace_payload(days=30, allow_network=False, include_nifty_options=False)
        cash = flows.get("cash") if isinstance(flows.get("cash"), Mapping) else {}
        out["available"] = bool(flows.get("available") or cash.get("available"))
        out["bias"] = str(cash.get("bias") or "")
        out["note"] = str(flows.get("insight") or cash.get("note") or "")
        today = cash.get("today") if isinstance(cash.get("today"), Mapping) else {}
        out["as_of"] = str(today.get("date") or "")
        out["latest_fii_net_cr"] = _f(today.get("fii_net"))
        out["latest_dii_net_cr"] = _f(today.get("dii_net"))
        if symbol:
            for deal in list(flows.get("bulk_deals") or []):
                if not isinstance(deal, Mapping):
                    continue
                if str(deal.get("symbol") or deal.get("tradingsymbol") or "").upper() == symbol:
                    out["bulk_deals"].append(
                        {
                            "side": deal.get("side") or deal.get("buy_sell") or "",
                            "qty": deal.get("qty") or deal.get("quantity"),
                            "price": deal.get("price") or deal.get("avg_price"),
                            "client": deal.get("client") or deal.get("client_name") or "",
                            "date": deal.get("date") or deal.get("as_of") or "",
                        }
                    )
            # Also surface membership in bulk_buy_symbols list.
            if not out["bulk_deals"]:
                for buy_sym in list(flows.get("bulk_buy_symbols") or []):
                    if str(buy_sym).upper() == symbol:
                        out["bulk_deals"].append({"side": "BUY", "symbol": symbol, "client": "bulk_buy_tag"})
                        break
            out["bulk_deals"] = out["bulk_deals"][:5]
    except Exception as exc:
        out["note"] = f"FII/DII store unavailable ({exc})"
    return out


def _scan_context(symbol: str, scan_row: Mapping[str, Any] | None, technical: Mapping[str, Any] | None) -> dict[str, Any]:
    row = dict(scan_row or {})
    tech = dict(technical or {})
    score = _f(row.get("score"))
    return {
        "setup": row.get("status") or row.get("setup_label") or "",
        "score": score,
        "signals": list(row.get("signals") or [])[:8],
        "chase_risk": bool(row.get("chase_risk")),
        "entry": _f(row.get("entry")),
        "stop": _f(row.get("stop")),
        "target": _f(row.get("target")),
        "relative_strength_proxy": score,
        "liquidity": {
            "volume_ratio": _f(tech.get("volume_ratio") or row.get("volume_ratio")),
            "atr_pct": _f(tech.get("atr_pct") or row.get("atr_pct")),
            "from_high_pct": _f(tech.get("from_high_pct") or row.get("from_high_pct")),
        },
        "note": (
            "RS shown as scan-score proxy from the last whole-market scan — not a live Yahoo RS pull."
            if score is not None
            else "No scan row yet — run Scan now to unlock setup + RS proxy."
        ),
    }


def _headline_bullets(
    *,
    workspace_state: str,
    confidence_pct: int,
    news: Sequence[Mapping[str, Any]],
    buy_ctx: Mapping[str, Any],
    earnings: Mapping[str, Any],
    flows: Mapping[str, Any],
    scan_ctx: Mapping[str, Any],
    outlook: Mapping[str, Any] | None,
) -> list[str]:
    bullets: list[str] = []
    bullets.append(f"Desk state {workspace_state.replace('_', ' ')} · data confidence {confidence_pct}%")
    if scan_ctx.get("setup"):
        bullets.append(f"Setup: {scan_ctx['setup']}" + (f" · score {scan_ctx['score']}" if scan_ctx.get("score") is not None else ""))
    health = buy_ctx.get("health") or {}
    if buy_ctx.get("in_buy_book") or buy_ctx.get("in_holdings"):
        label = health.get("status_label") or health.get("severity") or "watching"
        bullets.append(f"Position context: {'buy book' if buy_ctx.get('in_buy_book') else 'holdings'} · {label}")
        for warn in list(health.get("warnings") or [])[:2]:
            if isinstance(warn, Mapping) and warn.get("text"):
                bullets.append(f"Health: {warn.get('text')}")
    if earnings.get("available") and str(earnings.get("risk_level") or "").upper() in {"HIGH", "MEDIUM"}:
        bullets.append(str(earnings.get("note") or f"Earnings risk {earnings.get('risk_level')}"))
    if flows.get("bias"):
        bullets.append(f"Market FII/DII bias: {flows.get('bias')}" + (f" · {flows.get('as_of')}" if flows.get("as_of") else ""))
    if flows.get("bulk_deals"):
        bullets.append(f"Bulk deal tag on this symbol ({len(flows['bulk_deals'])} recent)")
    for item in list(news or [])[:2]:
        title = str(item.get("title") or item.get("headline") or "").strip()
        if title:
            bullets.append(f"News: {title[:110]}")
    thesis = (outlook or {}).get("thesis") or {}
    if thesis.get("text"):
        bullets.append(f"Outlook: {thesis.get('label') or ''} — {str(thesis.get('text'))[:120]}".strip(" —"))
    elif outlook and outlook.get("summary"):
        bullets.append(f"Outlook: {str(outlook.get('summary'))[:120]}")
    return bullets[:10]


def build_stock_desk_tape(
    symbol: str,
    *,
    workspace: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose a single-stock information tape from every relevant store."""
    symbol = str(symbol or "").strip().upper()
    ws = dict(workspace or {})
    evidence = _evidence_completeness(symbol)
    buy_ctx = _buy_context(symbol)
    earnings = _earnings_context(symbol)
    flows = _flow_context(symbol)
    scan_ctx = _scan_context(symbol, ws.get("scanner") if isinstance(ws.get("scanner"), Mapping) else {}, ws.get("technical") if isinstance(ws.get("technical"), Mapping) else {})
    news = list(ws.get("news") or [])
    outlook = ws.get("growth_outlook") if isinstance(ws.get("growth_outlook"), Mapping) else {}
    peers = ws.get("peers") if isinstance(ws.get("peers"), Mapping) else {}
    confidence = int(ws.get("confidence_pct") or 0)
    # Blend evidence coverage into an information thirst score for the hero.
    evidence_score = int(evidence.get("score_pct") or 0) if evidence.get("available") else 0
    info_score = round(confidence * 0.7 + evidence_score * 0.3)

    bullets = _headline_bullets(
        workspace_state=str(ws.get("state") or "DATA_INCOMPLETE"),
        confidence_pct=confidence,
        news=news,
        buy_ctx=buy_ctx,
        earnings=earnings,
        flows=flows,
        scan_ctx=scan_ctx,
        outlook=outlook,
    )

    return {
        "schema_version": 1,
        "symbol": symbol,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "title": "STOCK DESK TAPE",
        "info_score_pct": info_score,
        "bullets": bullets,
        "evidence": evidence,
        "position": buy_ctx,
        "earnings": earnings,
        "flows": flows,
        "scan": scan_ctx,
        "peers_snapshot": {
            "average_pe": peers.get("average_pe"),
            "pe_vs_peer_avg": peers.get("pe_vs_peer_avg"),
            "peer_rank": peers.get("peer_rank"),
            "total_peers": peers.get("total_peers"),
            "peer_rank_verdict": peers.get("peer_rank_verdict"),
            "sector_leader": peers.get("sector_leader"),
        },
        "top_news": [
            {
                "title": item.get("title") or item.get("headline"),
                "published_at": item.get("published_at") or item.get("fetched_at"),
                "impact_score": item.get("impact_score"),
                "source": item.get("source") or item.get("publisher"),
            }
            for item in news[:5]
            if isinstance(item, Mapping)
        ],
        "outlook_thesis": (outlook or {}).get("thesis") or {},
        "honesty": (
            "Desk tape composes existing QuantTerm stores only. Missing stays missing. "
            "Not a buy/sell order. Paper-first."
        ),
        "places_orders": False,
    }


def _load_pulse() -> dict[str, Any]:
    if not PULSE_PATH.exists():
        return {}
    try:
        import json

        payload = json.loads(PULSE_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _brain_posture_light() -> dict[str, Any]:
    now = time.time()
    cached = _BRAIN_CACHE.get("payload")
    if cached is not None and now - float(_BRAIN_CACHE.get("ts") or 0) < _BRAIN_TTL_S:
        return cached
    if _low_power():
        # Avoid full brain fan-out on Air — use market view + FII bias only.
        light = {
            "available": True,
            "mode": "low_power_light",
            "posture": "",
            "posture_reason": "Full Brain posture skipped in low-power mode",
            "regime": "",
            "flows_bias": "",
            "options_bias": "",
            "breadth": "",
            "verdict_line": "",
        }
        try:
            from product.market_view import current_market_view

            mv = current_market_view()
            light["regime"] = str(getattr(mv, "health", "") or "")
            light["breadth"] = str(getattr(mv, "breadth", "") or "")
            light["verdict_line"] = str(getattr(mv, "summary", "") or "")[:160]
        except Exception:
            pass
        try:
            from data.fii_dii_store import workspace_payload

            flows = workspace_payload(days=10, allow_network=False, include_nifty_options=False)
            cash = flows.get("cash") if isinstance(flows.get("cash"), Mapping) else {}
            light["flows_bias"] = str(cash.get("bias") or "")
        except Exception:
            pass
        _BRAIN_CACHE["ts"] = now
        _BRAIN_CACHE["payload"] = light
        return light
    try:
        from core.brain import assess

        assessed = assess("IN")
        vitals = assessed.get("vitals") or {}
        payload = {
            "available": True,
            "mode": "full",
            "posture": assessed.get("posture"),
            "posture_reason": assessed.get("posture_reason"),
            "verdict_line": assessed.get("verdict_line"),
            "regime": vitals.get("regime"),
            "flows_bias": vitals.get("flows_bias"),
            "options_bias": vitals.get("options_bias"),
            "breadth": vitals.get("breadth"),
            "n_buys": vitals.get("n_buys"),
            "top_pick": vitals.get("top_pick"),
        }
        _BRAIN_CACHE["ts"] = now
        _BRAIN_CACHE["payload"] = payload
        return payload
    except Exception as exc:
        payload = {
            "available": False,
            "mode": "error",
            "note": str(exc),
            "posture": "",
            "verdict_line": "",
        }
        _BRAIN_CACHE["ts"] = now
        _BRAIN_CACHE["payload"] = payload
        return payload


def build_market_command(*, home: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Home-strip composition: pulse + FII/DII + brain posture + active-buy warnings."""
    base = dict(home or {})
    pulse = _load_pulse()
    brain = _brain_posture_light()
    flows = _flow_context("")  # market-level; symbol bulk deals ignored
    buy_warnings = 0
    buy_critical = 0
    try:
        from product.buy_health import evaluate_book

        book = evaluate_book()
        summary = book.get("summary") or {}
        buy_critical = int(summary.get("critical") or 0)
        buy_warnings = int(summary.get("critical") or 0) + int(summary.get("warn") or 0)
    except Exception:
        pass

    takeaways = list(pulse.get("takeaways") or pulse.get("headlines") or [])[:4]
    if not takeaways and brain.get("verdict_line"):
        takeaways = [brain.get("verdict_line")]

    return {
        **base,
        "command": {
            "available": True,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "posture": brain.get("posture") or "",
            "posture_reason": brain.get("posture_reason") or "",
            "verdict_line": brain.get("verdict_line") or "",
            "regime": brain.get("regime") or base.get("market_health") or "",
            "flows_bias": flows.get("bias") or brain.get("flows_bias") or "",
            "flows_note": flows.get("note") or "",
            "fii_net_cr": flows.get("latest_fii_net_cr"),
            "dii_net_cr": flows.get("latest_dii_net_cr"),
            "flows_as_of": flows.get("as_of"),
            "options_bias": brain.get("options_bias") or "",
            "takeaways": [
                (t.get("text") if isinstance(t, Mapping) else str(t))
                for t in takeaways
            ][:4],
            "pulse_as_of": pulse.get("generated_at") or pulse.get("as_of") or "",
            "active_buy_warnings": buy_warnings,
            "active_buy_critical": buy_critical,
            "lane_counts": base.get("counts") or {},
            "honesty": "Market command reuses Pulse, FII/DII, Brain/market view and Active Buys — no invented tape.",
            "low_power": _low_power(),
        },
    }


def build_watchlist_briefing(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Attach what-changed badges using scan / pulse / sniper / news / buy-health."""
    pulse = _load_pulse()
    movers = set()
    for key in ("gainers", "losers", "strength", "weak", "relative_strength", "breakouts_today"):
        for row in pulse.get(key) or []:
            if isinstance(row, Mapping) and row.get("symbol"):
                movers.add(str(row["symbol"]).upper())
            elif isinstance(row, str):
                movers.add(row.upper())

    sniper_hits: set[str] = set()
    try:
        from product.sniper_board import load_board

        board = load_board()
        for row in board.get("hits") or board.get("rows") or board.get("items") or []:
            if isinstance(row, Mapping) and row.get("symbol"):
                sniper_hits.add(str(row["symbol"]).upper())
    except Exception:
        pass

    briefed: list[dict[str, Any]] = []
    for item in items:
        row = dict(item)
        symbol = str(row.get("symbol") or "").upper()
        badges: list[str] = []
        snapshot = row.get("snapshot") if isinstance(row.get("snapshot"), Mapping) else {}
        setup = str(snapshot.get("setup_label") or snapshot.get("status") or "")
        if setup and setup.upper() not in {"", "NONE", "—", "-"}:
            badges.append(f"SETUP:{setup}")
        if symbol in movers:
            badges.append("PULSE_MOVER")
        if symbol in sniper_hits:
            badges.append("SNIPER")
        # News buzz (cheap local query)
        news_n = 0
        try:
            from news.curator_store import NewsCuratorStore

            store = NewsCuratorStore(ROOT / "logs" / "news_curator.sqlite3")
            news_n = len(list(store.recent(hours=72, limit=5, symbol=symbol)))
            if news_n:
                badges.append(f"NEWS:{news_n}")
        except Exception:
            pass
        health_label = ""
        try:
            from product.buy_book import load_book

            in_book = any(str(b.get("symbol") or "").upper() == symbol for b in (load_book().get("items") or []))
            if in_book:
                from product.buy_health import evaluate_symbol

                health = evaluate_symbol(symbol)
                health_label = str(health.get("status_label") or "")
                if health_label:
                    badges.append(f"HEALTH:{health_label}")
        except Exception:
            pass
        briefed.append(
            {
                **row,
                "briefing": {
                    "badges": badges,
                    "in_pulse_movers": symbol in movers,
                    "sniper_hit": symbol in sniper_hits,
                    "news_72h": news_n,
                    "health_label": health_label,
                    "changed_today": bool(badges),
                },
            }
        )

    changed = sum(1 for r in briefed if (r.get("briefing") or {}).get("changed_today"))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": briefed,
        "count": len(briefed),
        "changed_today": changed,
        "honesty": "Watchlist briefing badges reuse scan, pulse, sniper, news and buy-health — missing stays quiet.",
    }
