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


def _low_power() -> bool:
    return os.environ.get("QT_LOW_POWER", "").strip().lower() in {"1", "true", "yes", "on"}


def _global_macro_cues(
    *,
    allow_network: bool = True,
    skip_commodity_names: set[str] | None = None,
) -> dict[str, Any]:
    """US / Asia / Europe / commodities — soft-fail each ticker."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cues: list[dict[str, Any]] = []
    gaps: list[str] = []
    skip_cmds = {n.lower() for n in (skip_commodity_names or set())}

    # Prefer product us_retail (name/label + chg_pct after us_live_prices fix)
    try:
        from product import us_retail

        ov = us_retail.overview() or {}
        for item in ov.get("indices") or []:
            name = item.get("name") or item.get("label")
            chg = item.get("chg_pct")
            if name and chg is not None:
                cues.append(
                    {
                        "name": str(name).replace("NASDAQ", "Nasdaq").replace("Dow 30", "Dow"),
                        "price": item.get("price"),
                        "chg_pct": chg,
                        "source": "us_retail",
                    }
                )
    except Exception:
        pass

    tickers = [
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
        ("India VIX", "^INDIAVIX"),
    ]
    if not _low_power():
        tickers.append(("WTI crude", "CL=F"))

    have = {str(c.get("name") or "").lower() for c in cues}
    if allow_network:
        def _one(name: str, tk: str) -> dict[str, Any] | None:
            try:
                import yfinance as yf

                fi = yf.Ticker(tk).fast_info
                last = float(getattr(fi, "last_price", 0) or 0)
                prev = float(getattr(fi, "previous_close", 0) or 0)
                if last > 0 and prev > 0:
                    return {
                        "name": name,
                        "price": round(last, 2),
                        "chg_pct": round((last - prev) / prev * 100.0, 2),
                        "source": "yfinance",
                    }
            except Exception:
                return None
            return None

        pending = [(n, t) for n, t in tickers if n.lower() not in have]
        workers = 3 if _low_power() else 6
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_one, n, t): n for n, t in pending}
            for fut in as_completed(futs):
                name = futs[fut]
                try:
                    row = fut.result()
                except Exception:
                    gaps.append(f"{name} unavailable")
                    continue
                if row:
                    cues.append(row)
                else:
                    gaps.append(f"{name} unavailable")
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

    # Day-story global / macro lines (curator-backed, no invented prices)
    day_lines: list[str] = []
    try:
        from news.day_story_engine import build_day_stories

        payload = build_day_stories(limit=8) or {}
        stories = list(payload.get("stories") or []) if isinstance(payload, Mapping) else []
        for st in stories:
            if not isinstance(st, Mapping):
                continue
            cat = str(st.get("category") or "")
            et = str(st.get("event_type") or "")
            line = str(st.get("wrap_line") or st.get("headline") or "")
            if line and (cat in {"global", "economy"} or et in {"macro", "global"}):
                day_lines.append(line[:160])
    except Exception:
        pass

    bullets: list[str] = []
    us = [c for c in cues if c["name"] in {"S&P 500", "Nasdaq", "Dow"}]
    asia = [c for c in cues if c["name"] in {"Nikkei", "Hang Seng", "Kospi"}]
    europe = [c for c in cues if c["name"] in {"FTSE 100", "DAX"}]
    cmds = [
        c
        for c in cues
        if c["name"] in {"Gold", "Brent crude", "WTI crude"}
        and c["name"].lower() not in skip_cmds
    ]

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
    for theme in themes[:2]:
        bullets.append(f"Macro theme on the tape: {theme}.")
    for line in day_lines[:2]:
        bullets.append(f"Day story: {line}")

    # FII/DII with ₹ Cr when SQLite has today
    try:
        from data.institutional_flows import humanize_flow_bias
        from data.fii_dii_store import summarize

        summary = summarize(days=10, auto_refresh=False)
        if summary.get("available"):
            today = summary.get("today") or {}
            plain = humanize_flow_bias(str(summary.get("bias") or ""))
            fii = today.get("fii_net")
            dii = today.get("dii_net")
            as_of = summary.get("latest_date") or ""
            if fii is not None and dii is not None:
                bullets.append(
                    f"India FII/DII ({as_of}): FII ₹{float(fii):+.0f} Cr · DII ₹{float(dii):+.0f} Cr "
                    f"— {plain.get('bias_label') or summary.get('bias')}"
                )
            elif summary.get("bias"):
                bullets.append(
                    f"India FII/DII: {plain.get('bias_label')} — {plain.get('bias_note')}"
                )
        else:
            gaps.append("FII/DII store empty")
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


def _side_aware_walls(
    spot: float | None,
    top_calls: list[Mapping[str, Any]],
    top_puts: list[Mapping[str, Any]],
) -> tuple[float | None, float | None]:
    """Nearest heavy PE below spot (support) / CE above spot (resistance)."""
    if spot is None or spot <= 0:
        return None, None
    call_above: list[tuple[float, float]] = []
    for row in top_calls or []:
        strike = _f(row.get("strike"))
        oi = _f(row.get("ce_oi")) or 0.0
        if strike is not None and strike > spot and oi > 0:
            call_above.append((strike, oi))
    put_below: list[tuple[float, float]] = []
    for row in top_puts or []:
        strike = _f(row.get("strike"))
        oi = _f(row.get("pe_oi")) or 0.0
        if strike is not None and strike < spot and oi > 0:
            put_below.append((strike, oi))
    call_above.sort(key=lambda x: x[0])
    put_below.sort(key=lambda x: -x[0])
    support = put_below[0][0] if put_below else None
    resistance = call_above[0][0] if call_above else None
    return support, resistance


def _sensex_session_band(spot: float | None, *, allow_network: bool) -> tuple[float | None, float | None, str]:
    """Recent session high/low for Sensex — never invent OI walls."""
    if not spot:
        return None, None, "no spot"
    if allow_network:
        try:
            import yfinance as yf

            hist = yf.Ticker("^BSESN").history(period="5d")
            if hist is not None and not hist.empty and "High" in hist.columns:
                hi = float(hist["High"].max())
                lo = float(hist["Low"].min())
                if hi > 0 and lo > 0:
                    return lo, hi, "yahoo_5d_high_low"
        except Exception:
            pass
    # Honest quote band — labeled, not sold as OI
    return spot * 0.99, spot * 1.01, "quote_band_1pct"


def _options_levels_block(*, allow_network: bool = True) -> dict[str, Any]:
    """Nifty / Bank Nifty / Sensex key levels from real chain + quotes."""
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
            top_pe = list(attached.get("top_put_oi") or chain.get("top_put_oi") or [])
            top_ce = list(attached.get("top_call_oi") or chain.get("top_call_oi") or [])
            support, resistance = _side_aware_walls(spot, top_ce, top_pe)
            # Optional analytics backup only when side-aware empty
            if (support is None or resistance is None) and spot:
                try:
                    import pandas as pd
                    from options.analytics import get_oi_buildup

                    rows = attached.get("chain") or chain.get("chain") or []
                    if rows:
                        df = pd.DataFrame(rows)
                        walls = get_oi_buildup(df, float(spot)) if not df.empty else {}
                        if support is None:
                            pe = walls.get("support_levels") or []
                            if pe:
                                support = _f(pe[0].get("strike"))
                        if resistance is None:
                            ce = walls.get("resistance_levels") or []
                            if ce:
                                resistance = _f(ce[0].get("strike"))
                except Exception:
                    pass

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
                "wall_method": "side_aware_oi",
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
            if block.get("support") is not None:
                extras.append(f"put wall {float(block['support']):,.0f}")
            if block.get("resistance") is not None:
                extras.append(f"call wall {float(block['resistance']):,.0f}")
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

    # Sensex — session high/low when possible; never pretend BSE FO chain exists
    sx = quotes.get("SENSEX") or {}
    sx_spot = _f(sx.get("price"))
    if sx_spot is None and allow_network:
        try:
            import yfinance as yf

            fi = yf.Ticker("^BSESN").fast_info
            last = float(getattr(fi, "last_price", 0) or 0)
            if last > 0:
                sx_spot = last
                sx = {"price": last, "chg_pct": None, "source": "yfinance"}
        except Exception:
            pass
    if sx_spot:
        lo, hi, method = _sensex_session_band(sx_spot, allow_network=allow_network)
        zone = _index_zone(sx_spot, lo, hi)
        levels.append(
            {
                "available": True,
                "symbol": "SENSEX",
                "spot": sx_spot,
                "zone": zone,
                "support": lo,
                "resistance": hi,
                "chg_pct": sx.get("chg_pct"),
                "wall_method": method,
                "note": (
                    "Sensex session high/low band — not an options OI wall."
                    if method == "yahoo_5d_high_low"
                    else "Sensex quote band (±1%) — options chain not attached."
                ),
            }
        )
        if zone:
            label = "session band" if method == "yahoo_5d_high_low" else "quote band"
            bullets.append(f"Sensex {zone} ({label})")
    else:
        gaps.append("Sensex quote unavailable")

    vix = quotes.get("VIX") or {}
    if _f(vix.get("price")) is not None:
        bullets.append(f"India VIX at {float(vix['price']):.2f}.")

    headline = (
        " · ".join(b for b in bullets if "zones" in b or "band" in b)[:160]
        if any(("zones" in b or "band" in b) for b in bullets)
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


_FUND_CLASS_OK = {
    "QUALITY_COMPOUNDER",
    "GARP_CANDIDATE",
    "QUALITY_BUT_EXPENSIVE",
    "LONG_TERM_WATCH",
    "NEEDS_FUNDAMENTALS",
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
    as_of = (
        payload.get("scanned_at")
        or payload.get("generated_at")
        or payload.get("as_of")
        or ""
    )
    if not records:
        return {
            "available": False,
            "horizon": "LONG_TERM",
            "rows": [],
            "message": "No long-term picks yet — run Long-Term scan / autonomy weekly pass.",
            "as_of": as_of,
        }

    def _rank_key(r: Mapping[str, Any]) -> float:
        for key in ("combined_score", "score", "long_term_score", "technical_score"):
            val = _f(r.get(key))
            if val is not None:
                return val
        return 0.0

    ranked = sorted(records, key=_rank_key, reverse=True)
    rows: list[dict[str, Any]] = []
    for rec in ranked:
        verdict = str(rec.get("verdict") or rec.get("status") or "").upper()
        classification = str(rec.get("classification") or "").upper()
        if "SKIP" in verdict or classification == "AVOID_REVIEW":
            continue
        if classification and classification not in _FUND_CLASS_OK:
            # Still allow classic LONG_TERM_BUY / WATCH when classification absent
            if verdict not in {"LONG_TERM_BUY", "BUY", "WATCH", "LONG_TERM", ""}:
                continue
        elif not classification and verdict and verdict not in {
            "LONG_TERM_BUY", "BUY", "WATCH", "LONG_TERM", ""
        }:
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
        if not thesis and rec.get("quality_factors"):
            thesis = "; ".join(str(f) for f in list(rec.get("quality_factors") or [])[:3])
        if not thesis and rec.get("factors"):
            thesis = "; ".join(str(f) for f in list(rec.get("factors") or [])[:3])
        rows.append(
            {
                "symbol": str(rec.get("symbol") or "").upper(),
                "price": price,
                "score": _rank_key(rec),
                "combined_score": _f(rec.get("combined_score")),
                "technical_score": _f(rec.get("technical_score") or rec.get("score")),
                "fundamental_score": _f(rec.get("fundamental_score")),
                "verdict": verdict or "LONG_TERM",
                "classification": classification or None,
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
        "as_of": as_of,
        "message": "" if rows else "Long-term shortlist empty",
        "places_orders": False,
        "honesty": (
            "Targets shown as prior-high watch levels from official history when available — "
            "not sell-side broker target prices. Ranked by combined_score when present."
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
        why = str(rec.get("why") or "")
        if not why and rec.get("reasons"):
            why = str((rec.get("reasons") or ["setup"])[0])
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
                "edge_r": _f(rec.get("edge_r")),
                "score": _f(rec.get("score")),
                "why": why[:160],
                "signals": list(rec.get("signals") or [])[:6],
                "horizon": "SHORT_TERM",
            }
        )
        if len(rows) >= limit:
            break

    as_of = (
        (payload or {}).get("scanned_at")
        or (payload or {}).get("generated_at")
        or (payload or {}).get("as_of")
        or ""
    )
    return {
        "available": bool(rows),
        "horizon": "SHORT_TERM",
        "title": "Technical Picks",
        "subtitle": "Short-term research from last whole-market scan · entry/stop/target when present",
        "rows": rows,
        "as_of": as_of,
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
            "gift_hard": False,
        }
    gift_hard = bool(premarket.get("gift_hard"))
    gift_title = (
        "GIFT Nifty (Gift City Cues)"
        if gift_hard
        else "Premarket / Overnight Risk (Gift print incomplete)"
    )
    decider_gift = {
        "available": bool(premarket.get("available")),
        "key": "gift_premarket",
        "title": gift_title,
        "icon": "gift",
        "gift_hard": gift_hard,
        "headline": premarket.get("headline") or "Gift / premarket cues incomplete",
        "bullets": list(premarket.get("bullets") or [])[:6],
        "gift_nifty": premarket.get("gift_nifty"),
        "us_futures": premarket.get("us_futures") or [],
        "source": premarket.get("source") or "",
        "stale": bool(premarket.get("stale")),
        "gaps": list(premarket.get("gaps") or []),
    }
    gaps.extend(decider_gift["gaps"][:2])

    # Commodities already printed under premarket → skip duplicate bullets in macro
    skip_cmds: set[str] = set()
    for b in decider_gift["bullets"]:
        low = str(b).lower()
        if "brent" in low:
            skip_cmds.add("brent crude")
        if "wti" in low:
            skip_cmds.add("wti crude")
        if "gold" in low and "gift" not in low:
            skip_cmds.add("gold")

    # 2) Macro / global
    macro = _global_macro_cues(allow_network=allow_network, skip_commodity_names=skip_cmds)
    gaps.extend(list(macro.get("gaps") or [])[:2])

    # 3) Options levels — prefer cached chains; Sensex may use Yahoo session band
    options = _options_levels_block(allow_network=allow_network)
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
        "Gift Nifty: Moneycontrol → ET/Mint/BS/CNBC RSS → Google News consensus (median pts/%) — never cash-Nifty labeled as Gift.",
        "Macro prints parallel-fetch Yahoo + FII ₹ Cr from SQLite + day-story global lines.",
        "Options zones use side-aware put-below / call-above OI walls; Sensex is session/quote band, not fake FO.",
        "Fundamental watches = prior-high from official history · ranked by combined_score — not broker TPs.",
        "Technical picks ship real entry / stop / target from the last whole-market scan.",
        "Missing stays missing. Research only · paper-first · never places LIVE orders.",
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
