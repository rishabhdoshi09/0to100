"""Market Decision Brief — retail competition with Motilal-style morning desks.

Sections (compose-only, never invent):
  1. Three things that will decide the market today
     • GIFT Nifty / premarket cues
     • Macro / global cues (US, Asia, Europe, crude, gold)
     • Options data & key levels (Nifty / Bank Nifty / Sensex zones)
  2. Fundamental picks (long-term store — multi-month research)
  3. Technical picks (scan store — short-term research with entry/target when present)

Honesty: missing stays missing. Research suggestions ≠ buy tickets. Paper-first.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "logs" / "product" / "market_decision_brief.json"


def brief_path(path: Path | None = None) -> Path:
    env = os.environ.get("QT_MARKET_BRIEF_FILE", "").strip()
    if path is not None:
        return Path(path)
    if env:
        return Path(env)
    return DEFAULT_PATH


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _f(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def empty_brief(*, message: str = "") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "available": False,
        "report_type": "MARKET_DECISION_BRIEF",
        "title": "3 Things That Will Decide the Market Today",
        "generated_at": _utc_now(),
        "deciders": [],
        "fundamental_picks": {"available": False, "rows": [], "message": ""},
        "technical_picks": {"available": False, "rows": [], "message": ""},
        "gaps": [message] if message else [],
        "places_orders": False,
        "live_locked": True,
        "signal_desk": False,
        "honesty": (
            "QuantTerm Market Decision Brief composes premarket, global cues, options levels, "
            "long-term fundamentals and scanner technicals from real stores only. "
            "Missing sections stay missing. Not Motilal — better: retail-honest, paper-first."
        ),
        "message": message or "Brief not built yet",
    }


def load_brief(path: Path | None = None) -> dict[str, Any]:
    target = brief_path(path)
    if not target.exists():
        return empty_brief()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return empty_brief(message="Brief file unreadable")
        payload.setdefault("places_orders", False)
        return payload
    except Exception as exc:
        return empty_brief(message=str(exc))


def save_brief(payload: Mapping[str, Any], path: Path | None = None) -> dict[str, Any]:
    target = brief_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    out = dict(payload)
    out["places_orders"] = False
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return out


def _global_macro_cues(*, allow_network: bool = True) -> dict[str, Any]:
    """US / Asia / Europe / commodities — soft-fail each ticker."""
    cues: list[dict[str, Any]] = []
    gaps: list[str] = []
    # Prefer product us_retail when it has chg_pct
    try:
        from product import us_retail

        ov = us_retail.overview() or {}
        for item in ov.get("indices") or []:
            if item.get("name") and item.get("chg_pct") is not None:
                cues.append(
                    {
                        "name": item.get("name"),
                        "price": item.get("price"),
                        "chg_pct": item.get("chg_pct"),
                        "source": "us_retail",
                    }
                )
    except Exception:
        pass

    tickers = (
        ("S&P 500", "^GSPC"),
        ("Nasdaq", "^IXIC"),
        ("Dow", "^DJI"),
        ("FTSE 100", "^FTSE"),
        ("DAX", "^GDAXI"),
        ("Nikkei", "^N225"),
        ("Hang Seng", "^HSI"),
        ("Kospi", "^KS11"),
        ("Gold", "GC=F"),
        ("Brent crude", "BZ=F"),
        ("WTI crude", "CL=F"),
        ("India VIX", "^INDIAVIX"),
    )
    have = {str(c.get("name") or "").lower() for c in cues}
    if allow_network:
        try:
            import yfinance as yf

            for name, tk in tickers:
                if name.lower() in have:
                    continue
                try:
                    fi = yf.Ticker(tk).fast_info
                    last = float(getattr(fi, "last_price", 0) or 0)
                    prev = float(getattr(fi, "previous_close", 0) or 0)
                    if last > 0 and prev > 0:
                        cues.append(
                            {
                                "name": name,
                                "price": round(last, 2),
                                "chg_pct": round((last - prev) / prev * 100.0, 2),
                                "source": "yfinance",
                            }
                        )
                except Exception:
                    gaps.append(f"{name} unavailable")
        except Exception as exc:
            gaps.append(f"yfinance unavailable ({exc})")
    else:
        gaps.append("Network disabled for global prints")

    # Macro theme keywords from news (no prices invented)
    themes: list[str] = []
    try:
        from core.macro_pulse import macro_pulse
        from news.curator_store import CuratorStore

        articles = []
        for art in CuratorStore().recent(hours=36, limit=80):
            articles.append(
                {
                    "headline": getattr(art, "headline", "") or "",
                    "summary": getattr(art, "summary", "") or "",
                    "title": getattr(art, "headline", "") or "",
                }
            )
        mp = macro_pulse(articles) if articles else {}
        for t in list((mp or {}).get("themes") or [])[:4]:
            if isinstance(t, Mapping):
                label = t.get("label") or t.get("theme") or t.get("name")
                direction = str(t.get("direction") or "").strip()
                if label:
                    themes.append(f"{label} ({direction})" if direction else str(label))
            elif t:
                themes.append(str(t))
    except Exception:
        pass

    bullets: list[str] = []
    us = [c for c in cues if c["name"] in {"S&P 500", "Nasdaq", "Dow"}]
    asia = [c for c in cues if c["name"] in {"Nikkei", "Hang Seng", "Kospi"}]
    europe = [c for c in cues if c["name"] in {"FTSE 100", "DAX"}]
    cmds = [c for c in cues if c["name"] in {"Gold", "Brent crude", "WTI crude"}]

    def _band(rows: list[dict[str, Any]], label: str) -> None:
        if not rows:
            return
        worst = min(rows, key=lambda r: float(r.get("chg_pct") or 0))
        best = max(rows, key=lambda r: float(r.get("chg_pct") or 0))
        span = max(abs(float(r.get("chg_pct") or 0)) for r in rows)
        direction = "fell" if float(worst.get("chg_pct") or 0) < 0 else "rose"
        bullets.append(
            f"{label} {direction} up to {span:.1f}% "
            f"(range {float(worst['chg_pct']):+.1f}% to {float(best['chg_pct']):+.1f}%)."
        )

    _band(us, "US markets")
    _band(asia, "Asian markets")
    _band(europe, "European markets")
    for c in cmds:
        unit = "$" if "crude" in c["name"].lower() or c["name"] == "Gold" else ""
        bullets.append(
            f"{c['name']} {float(c['chg_pct']):+.1f}% at {unit}{float(c['price']):,.2f}."
        )
    for theme in themes[:3]:
        bullets.append(f"Macro theme on the tape: {theme}.")

    # FII/DII as India-specific macro
    try:
        from data.institutional_flows import humanize_flow_bias
        from data.fii_dii_store import workspace_payload

        flows = workspace_payload(days=10, allow_network=False, include_nifty_options=False)
        cash = flows.get("cash") if isinstance(flows.get("cash"), Mapping) else {}
        bias = str(cash.get("bias") or flows.get("bias") or "")
        if bias:
            plain = humanize_flow_bias(bias)
            bullets.append(f"India FII/DII: {plain['bias_label']} — {plain['bias_note']}")
    except Exception:
        gaps.append("FII/DII unavailable")

    headline = bullets[0] if bullets else "Global / macro cues incomplete"
    return {
        "available": bool(bullets),
        "key": "macro_global",
        "title": "Macro / Global Cues",
        "icon": "global",
        "headline": headline,
        "bullets": bullets[:8],
        "cues": cues,
        "themes": themes,
        "gaps": gaps,
    }


def _index_zone(spot: float | None, support: float | None, resistance: float | None) -> str | None:
    if not spot:
        return None
    lo = support if support and support < spot else spot * 0.985
    hi = resistance if resistance and resistance > spot else spot * 1.015
    # Round to trader-friendly zones
    step = 50 if spot < 30000 else 100
    lo_r = int(lo // step * step)
    hi_r = int((hi + step - 1) // step * step)
    return f"{lo_r} to {hi_r}"


def _options_levels_block() -> dict[str, Any]:
    """Nifty / Bank Nifty / Sensex options zones from real chain + quotes."""
    gaps: list[str] = []
    levels: list[dict[str, Any]] = []
    bullets: list[str] = []

    quotes: dict[str, dict] = {}
    try:
        from data.live_quotes import get_index_quotes

        quotes = get_index_quotes(["NIFTY", "BANKNIFTY", "SENSEX", "VIX"]) or {}
    except Exception as exc:
        gaps.append(f"Index quotes unavailable ({exc})")

    def _chain_read(symbol: str) -> dict[str, Any]:
        try:
            from options.chain_fetch import chain_workspace_cached
            from options.positioning_read import attach_positioning_read

            chain = chain_workspace_cached(symbol, force=False)
            if not chain.get("available"):
                return {"available": False, "symbol": symbol, "message": chain.get("message")}
            attached = attach_positioning_read(chain)
            spot = _f(attached.get("spot")) or _f((quotes.get(symbol) or {}).get("price"))
            # OI walls
            support = resistance = None
            try:
                import pandas as pd
                from options.analytics import get_oi_buildup

                rows = attached.get("chain") or chain.get("chain") or []
                if rows and spot:
                    df = pd.DataFrame(rows)
                    walls = get_oi_buildup(df, float(spot)) if not df.empty else {}
                    pe = walls.get("support_levels") or []
                    ce = walls.get("resistance_levels") or []
                    if pe:
                        support = _f(pe[0].get("strike"))
                    if ce:
                        resistance = _f(ce[0].get("strike"))
            except Exception:
                pass
            # Fallback walls from top put/call OI lists
            if support is None:
                top_pe = attached.get("top_put_oi") or chain.get("top_put_oi") or []
                if top_pe:
                    support = _f(top_pe[0].get("strike"))
            if resistance is None:
                top_ce = attached.get("top_call_oi") or chain.get("top_call_oi") or []
                if top_ce:
                    resistance = _f(top_ce[0].get("strike"))

            zone = _index_zone(spot, support, resistance)
            read = attached.get("positioning_read") or {}
            return {
                "available": True,
                "symbol": symbol,
                "spot": spot,
                "pcr": attached.get("pcr"),
                "max_pain": attached.get("max_pain"),
                "atm_iv": attached.get("atm_iv"),
                "bias": attached.get("bias"),
                "stance": read.get("stance"),
                "support": support,
                "resistance": resistance,
                "zone": zone,
                "note": attached.get("note") or read.get("headline") or "",
            }
        except Exception as exc:
            return {"available": False, "symbol": symbol, "message": str(exc)}

    for sym, label in (("NIFTY", "Nifty"), ("BANKNIFTY", "Bank Nifty")):
        block = _chain_read(sym)
        levels.append(block)
        if block.get("available") and block.get("zone"):
            bullets.append(f"{label} {block['zone']} zones")
            extras = []
            if block.get("pcr") is not None:
                extras.append(f"PCR {float(block['pcr']):.2f}")
            if block.get("max_pain"):
                extras.append(f"max pain {float(block['max_pain']):,.0f}")
            if block.get("stance") or block.get("bias"):
                extras.append(str(block.get("stance") or block.get("bias")))
            if extras:
                bullets.append(f"{label}: " + " · ".join(extras))
        else:
            gaps.append(f"{label} options chain incomplete")

    # Sensex — quote zone only (no NSE FO chain in this stack)
    sx = quotes.get("SENSEX") or {}
    sx_spot = _f(sx.get("price"))
    if sx_spot:
        zone = _index_zone(sx_spot, sx_spot * 0.99, sx_spot * 1.01)
        levels.append(
            {
                "available": True,
                "symbol": "SENSEX",
                "spot": sx_spot,
                "zone": zone,
                "chg_pct": sx.get("chg_pct"),
                "note": "Sensex zone from live/index quote band — options chain not attached.",
            }
        )
        if zone:
            bullets.append(f"Sensex {zone} zones")
    else:
        gaps.append("Sensex quote unavailable")

    vix = quotes.get("VIX") or {}
    if _f(vix.get("price")) is not None:
        bullets.append(f"India VIX at {float(vix['price']):.2f}.")

    headline = (
        " · ".join(b for b in bullets if "zones" in b)[:160]
        if any("zones" in b for b in bullets)
        else "Options / key levels incomplete"
    )
    return {
        "available": bool(bullets),
        "key": "options_levels",
        "title": "Options Data & Key Levels",
        "icon": "options",
        "headline": headline,
        "bullets": bullets[:8],
        "levels": levels,
        "gaps": gaps,
    }


def _fundamental_picks(limit: int = 5) -> dict[str, Any]:
    """Long-term research picks from durable store — multi-month horizon."""
    try:
        from product.long_term_store import load_long_term_scan

        payload = load_long_term_scan() or {}
    except Exception as exc:
        return {
            "available": False,
            "horizon": "LONG_TERM",
            "rows": [],
            "message": f"Long-term store unavailable ({exc})",
        }
    records = [r for r in (payload.get("records") or []) if isinstance(r, Mapping)]
    if not records:
        return {
            "available": False,
            "horizon": "LONG_TERM",
            "rows": [],
            "message": "No long-term picks yet — run Long-Term scan / autonomy weekly pass.",
            "as_of": payload.get("generated_at") or payload.get("as_of") or "",
        }

    ranked = sorted(
        records,
        key=lambda r: float(r.get("score") or r.get("long_term_score") or 0),
        reverse=True,
    )
    rows: list[dict[str, Any]] = []
    for rec in ranked:
        verdict = str(rec.get("verdict") or rec.get("status") or "").upper()
        if "SKIP" in verdict or "AVOID" in verdict:
            continue
        if verdict and verdict not in {"LONG_TERM_BUY", "BUY", "WATCH", "LONG_TERM", ""}:
            continue
        price = _f(rec.get("price") or rec.get("last_price"))
        # Long-term scan: from_high_pct = % below 52w high (positive when below).
        from_high = _f(rec.get("from_high_pct"))
        target_watch = None
        upside = None
        if price is not None and from_high is not None and 0 < from_high < 95:
            target_watch = round(price / (1.0 - from_high / 100.0), 2)  # prior high
            upside = round((target_watch / price - 1.0) * 100.0, 1)
        thesis = str(rec.get("thesis") or rec.get("why") or "")
        if not thesis and rec.get("factors"):
            thesis = "; ".join(str(f) for f in list(rec.get("factors") or [])[:3])
        rows.append(
            {
                "symbol": str(rec.get("symbol") or "").upper(),
                "price": price,
                "score": _f(rec.get("score") or rec.get("long_term_score")),
                "verdict": verdict or "LONG_TERM",
                "thesis": thesis[:200],
                "target_watch": target_watch,
                "upside_to_prior_high_pct": upside,
                "from_high_pct": from_high,
                "horizon": "LONG_TERM",
                "note": (
                    f"Prior-high watch ₹{target_watch:,.0f} (~{upside:.0f}% upside)"
                    if target_watch is not None and upside is not None
                    else "Long-term research watch — no invented broker target"
                ),
            }
        )
        if len(rows) >= limit:
            break

    return {
        "available": bool(rows),
        "horizon": "LONG_TERM",
        "title": "Fundamental / Long-Term Picks",
        "subtitle": "For more than a year · QuantTerm long-term store · research only",
        "rows": rows,
        "as_of": payload.get("generated_at") or payload.get("as_of") or "",
        "message": "" if rows else "Long-term shortlist empty",
        "places_orders": False,
        "honesty": (
            "Targets shown as prior-high watch levels from official history when available — "
            "not sell-side broker target prices."
        ),
    }


def _technical_picks(limit: int = 5) -> dict[str, Any]:
    """Short-term technical research from latest market scan."""
    try:
        from product.scan_store import load_scan, watchlist_rows

        payload = load_scan() or {}
        records = watchlist_rows(payload, limit=40) if payload else []
    except Exception as exc:
        return {
            "available": False,
            "horizon": "SHORT_TERM",
            "rows": [],
            "message": f"Scan store unavailable ({exc})",
        }

    preferred = [
        r
        for r in records
        if isinstance(r, Mapping)
        and str(r.get("status") or "") in {"Ready to trade", "Watch for breakout"}
        and not r.get("chase_risk")
    ]
    if not preferred:
        preferred = [r for r in records if isinstance(r, Mapping) and not r.get("chase_risk")]

    rows: list[dict[str, Any]] = []
    for rec in preferred:
        entry = _f(rec.get("entry"))
        stop = _f(rec.get("stop"))
        target = _f(rec.get("target"))
        price = _f(rec.get("price"))
        upside = None
        if price and target and target > price:
            upside = round((target / price - 1.0) * 100.0, 1)
        elif entry and target and target > entry:
            upside = round((target / entry - 1.0) * 100.0, 1)
        rows.append(
            {
                "symbol": str(rec.get("symbol") or "").upper(),
                "status": rec.get("status"),
                "verdict": rec.get("verdict"),
                "price": price,
                "entry": entry,
                "stop": stop,
                "target": target,
                "upside_pct": upside,
                "score": _f(rec.get("score")),
                "why": str(rec.get("why") or (rec.get("reasons") or ["setup"])[0])[:160],
                "signals": list(rec.get("signals") or [])[:6],
                "horizon": "SHORT_TERM",
            }
        )
        if len(rows) >= limit:
            break

    return {
        "available": bool(rows),
        "horizon": "SHORT_TERM",
        "title": "Technical Picks",
        "subtitle": "Short-term research from last whole-market scan · entry/stop/target when present",
        "rows": rows,
        "as_of": (payload or {}).get("generated_at") or (payload or {}).get("as_of") or "",
        "scan_source": (payload or {}).get("source") or "",
        "message": "" if rows else "No scanner setups yet — run Scan now.",
        "places_orders": False,
        "honesty": (
            "Technical picks reuse QuantTerm scan entry/stop/target only. "
            "No invented upside. Research watch — not a live buy order."
        ),
    }


def build_market_decision_brief(
    *,
    persist: bool = True,
    allow_network: bool = True,
    path: Path | None = None,
) -> dict[str, Any]:
    """Assemble the Motilal-competitive retail market decision brief."""
    gaps: list[str] = []

    # 1) GIFT / premarket
    try:
        from product.premarket_cues import build_premarket_cues

        premarket = build_premarket_cues(allow_network=allow_network)
    except Exception as exc:
        premarket = {
            "available": False,
            "headline": "",
            "bullets": [],
            "gaps": [str(exc)],
        }
    decider_gift = {
        "available": bool(premarket.get("available")),
        "key": "gift_premarket",
        "title": "GIFT Nifty (Gift City Cues)",
        "icon": "gift",
        "headline": premarket.get("headline") or "Gift / premarket cues incomplete",
        "bullets": list(premarket.get("bullets") or [])[:6],
        "gift_nifty": premarket.get("gift_nifty"),
        "us_futures": premarket.get("us_futures") or [],
        "gaps": list(premarket.get("gaps") or []),
    }
    gaps.extend(decider_gift["gaps"][:2])

    # 2) Macro / global
    macro = _global_macro_cues(allow_network=allow_network)
    gaps.extend(list(macro.get("gaps") or [])[:2])

    # 3) Options levels (local/cached chains — no scrape required)
    options = _options_levels_block()
    gaps.extend(list(options.get("gaps") or [])[:2])

    fund = _fundamental_picks(limit=5)
    tech = _technical_picks(limit=5)
    if not fund.get("available"):
        gaps.append(fund.get("message") or "Fundamental picks missing")
    if not tech.get("available"):
        gaps.append(tech.get("message") or "Technical picks missing")

    deciders = [decider_gift, macro, options]
    available = any(d.get("available") for d in deciders) or fund.get("available") or tech.get("available")

    why_better = [
        "Every print traces to a QuantTerm store — Gift/premarket, Yahoo globals, options OI, long-term + scan picks.",
        "Fundamental 'targets' are prior-high watches from official history — not invented broker TP numbers.",
        "Technical picks ship real entry / stop / target from the last whole-market scan when present.",
        "Missing Gift Nifty, chain walls, or scans stay missing — never padded to look like a sell-side note.",
        "Research brief only · paper-first · never places LIVE orders.",
    ]

    brief = {
        "schema_version": 1,
        "available": bool(available),
        "report_type": "MARKET_DECISION_BRIEF",
        "title": "3 Things That Will Decide the Market Today",
        "generated_at": _utc_now(),
        "deciders": deciders,
        "fundamental_picks": fund,
        "technical_picks": tech,
        "why_better": why_better,
        "gaps": [g for g in gaps if g][:12],
        "places_orders": False,
        "live_locked": True,
        "signal_desk": False,
        "competitor_note": (
            "Built to out-honest sell-side morning notes: every number traces to a QuantTerm store. "
            "No invented Gift prints, broker targets, or buy tickets."
        ),
        "honesty": (
            "Market Decision Brief composes Gift/premarket text, global prints, options OI zones, "
            "long-term fundamental shortlist and scanner technical picks. "
            "Missing stays missing. Research only — paper-first — never places orders."
        ),
        "message": (
            f"{sum(1 for d in deciders if d.get('available'))}/3 deciders ready · "
            f"fund picks {len(fund.get('rows') or [])} · tech picks {len(tech.get('rows') or [])}"
        ),
    }
    if persist:
        save_brief(brief, path)
    return brief


def brief_telegram_message(brief: Mapping[str, Any] | None = None) -> str:
    from alerts.telegram_alerts import escape_html as esc

    payload = dict(brief or load_brief())
    lines = [
        f"<b>{esc(payload.get('title') or 'Market Decision Brief')}</b>",
        f"<i>{esc(payload.get('message') or '')}</i>",
        "",
    ]
    for i, decider in enumerate(payload.get("deciders") or [], 1):
        if not isinstance(decider, Mapping):
            continue
        mark = "●" if decider.get("available") else "○"
        lines.append(f"{mark} <b>{i}. {esc(decider.get('title'))}</b>")
        if decider.get("headline"):
            lines.append(esc(decider.get("headline")))
        for b in list(decider.get("bullets") or [])[:4]:
            lines.append(f"• {esc(b)}")
        lines.append("")

    fund = payload.get("fundamental_picks") or {}
    if fund.get("available"):
        lines.append("<b>Fundamental picks</b> <i>(long-term research)</i>")
        for row in list(fund.get("rows") or [])[:5]:
            sym = esc(row.get("symbol"))
            bits = [sym]
            if row.get("upside_to_prior_high_pct") is not None:
                bits.append(f"room to high ~{row['upside_to_prior_high_pct']:.0f}%")
            if row.get("target_watch"):
                bits.append(f"prior-high watch {row['target_watch']}")
            lines.append(" · ".join(str(x) for x in bits))
            if row.get("thesis"):
                lines.append(f"  {esc(str(row['thesis'])[:140])}")
        lines.append("")

    tech = payload.get("technical_picks") or {}
    if tech.get("available"):
        lines.append("<b>Technical picks</b> <i>(short-term research)</i>")
        for row in list(tech.get("rows") or [])[:5]:
            sym = esc(row.get("symbol"))
            line = f"{sym}"
            if row.get("entry"):
                line += f" · entry {row['entry']}"
            if row.get("target"):
                line += f" · tgt {row['target']}"
            if row.get("upside_pct") is not None:
                line += f" (~{row['upside_pct']}%)"
            if row.get("stop"):
                line += f" · stop {row['stop']}"
            lines.append(line)
            if row.get("why"):
                lines.append(f"  {esc(str(row['why'])[:120])}")
        lines.append("")

    if payload.get("gaps"):
        lines.append(f"Missing: {esc((payload.get('gaps') or [''])[0])}")
    lines.append("<i>Research brief · not a buy ticket · paper-first · beat the sell-side with honesty</i>")
    return "\n".join(lines)


def notify_market_decision_brief(brief: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(brief or load_brief())
    if not payload.get("available"):
        payload = build_market_decision_brief(persist=True)
    try:
        from alerts.telegram_alerts import AlertEngine

        engine = AlertEngine()
        if not engine.is_configured():
            return {
                "sent": False,
                "configured": False,
                "reason": "Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID in .env)",
            }
    except Exception as exc:
        return {"sent": False, "configured": False, "reason": str(exc)}

    ok = bool(engine.send(brief_telegram_message(payload)))
    return {
        "sent": ok,
        "configured": True,
        "reason": None if ok else (engine.last_error or "Telegram send failed"),
        "message": payload.get("message"),
        "cred_source": getattr(engine, "cred_source", "") or "",
    }
