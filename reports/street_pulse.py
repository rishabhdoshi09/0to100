"""
Daily Street Pulse — auto-generated daily market report.

Modelled on analyst-style daily newsletters (e.g. Daily Street Pulse):
  1. Cover takeaways
  2. Market snapshot (+ options positioning stance when available)
  3. Sector heat (leaders / laggards from bhav)
  4. Top gainers / losers
  5. Buzzing stock
  6. Gaining strength
  7. Losing momentum
  8. Relative-strength leaders (from durable scan scores)
  9. Breakouts today / tomorrow watch
 10. Global / US cues (soft-fail)
 11. Top headlines

Honesty rules (hard):
  - composition over stores only — never invent prices, CA, or fills
  - missing sections stay missing / disclosed
  - not a buy ticket; paper-first research digest
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from logger import get_logger

log = get_logger(__name__)

DEFAULT_PULSE_PATH = Path("logs/product/latest_street_pulse.json")
SCHEMA_VERSION = 2


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _stock_card(row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise scan-store / auto_scan rows into one UI-safe card."""
    signals = [str(x) for x in (row.get("signals") or [])]
    categories = [str(x) for x in (row.get("categories") or [])]
    status = str(row.get("status") or "")
    reasons = [str(x) for x in (row.get("reasons") or []) if str(x).strip()]
    change = row.get("change_pct")
    if change is None:
        change = row.get("momentum_5d")
    pivot = row.get("pivot_distance_pct")
    if pivot is None and _f(row.get("entry")) > 0 and _f(row.get("price")) > 0:
        pivot = round((_f(row.get("entry")) - _f(row.get("price"))) / _f(row.get("entry")) * 100, 2)
    pre = (
        "PreBreakout" in categories
        or "PRE_BREAKOUT" in {s.upper() for s in signals}
        or status == "Watch for breakout"
    )
    return {
        "symbol": str(row.get("symbol") or "").upper(),
        "company": str(row.get("company") or row.get("symbol") or ""),
        "price": round(_f(row.get("price")), 2) if row.get("price") is not None else None,
        "change_pct": round(_f(change), 2) if change is not None else None,
        "volume_ratio": round(_f(row.get("volume_ratio")), 2) if row.get("volume_ratio") is not None else None,
        "score": round(_f(row.get("score")), 1) if row.get("score") is not None else None,
        "entry": round(_f(row.get("entry")), 2) if row.get("entry") is not None else None,
        "stop": round(_f(row.get("stop")), 2) if row.get("stop") is not None else None,
        "target": round(_f(row.get("target")), 2) if row.get("target") is not None else None,
        "pivot_distance_pct": round(_f(pivot), 2) if pivot is not None else None,
        "status": status,
        "verdict": str(row.get("verdict") or ""),
        "chase_risk": bool(row.get("chase_risk")),
        "signals": signals,
        "categories": categories,
        "reasons": reasons,
        "why": reasons[0] if reasons else str(row.get("why") or ""),
        "pre_breakout": pre,
        "fno_available": bool(row.get("fno_available")),
    }


def _scan_universe() -> tuple[list[dict[str, Any]], int, str, str]:
    """Prefer durable product scan; fall back to in-memory auto_scan."""
    try:
        from product.scan_store import load_scan

        payload = load_scan()
        if payload and payload.get("records"):
            rows = [_stock_card(r) for r in payload["records"] if r.get("symbol")]
            return (
                rows,
                int(payload.get("universe_size") or len(rows)),
                str(payload.get("scanned_at") or ""),
                "scan_store",
            )
    except Exception as exc:
        log.debug("pulse_scan_store_miss", error=str(exc))
    try:
        from scan.auto_scan import get_results

        results, universe_size, last_ts, _ = get_results()
        rows = [_stock_card(r) for r in (results or []) if r.get("symbol")]
        return rows, int(universe_size or len(rows)), str(last_ts or ""), "auto_scan"
    except Exception as exc:
        log.debug("pulse_auto_scan_miss", error=str(exc))
        return [], 0, "", "unavailable"


def _market_snapshot() -> dict[str, Any]:
    out: dict[str, Any] = {"indices": [], "commentary": "", "regime": "", "options_stance": None}
    try:
        from data.live_quotes import get_index_quotes

        q = get_index_quotes(["NIFTY", "BANKNIFTY"])
        for name, key in (("NIFTY 50", "NIFTY"), ("BANK NIFTY", "BANKNIFTY")):
            if q.get(key, {}).get("price"):
                out["indices"].append(
                    {
                        "name": name,
                        "price": q[key]["price"],
                        "chg_pct": q[key]["chg_pct"],
                    }
                )
    except Exception:
        pass
    try:
        from core.regime_engine import compute_regime

        regime = compute_regime()
        label = str(getattr(regime, "market_regime", "") or "")
        out["regime"] = label
        rmap = {
            "TRENDING_BULL": "Market trending up — breakout research favoured; still paper-first.",
            "CHOPPY": "Choppy tape — prefer smaller risk and clearer setups.",
            "TRENDING_BEAR": "Market weak — keep cash higher and chase risk low.",
        }
        out["commentary"] = rmap.get(label, "")
        leaders = list(getattr(regime, "leading_sectors", None) or [])[:4]
        laggards = list(getattr(regime, "lagging_sectors", None) or [])[:4]
        if leaders:
            out["regime_leaders"] = leaders
        if laggards:
            out["regime_laggards"] = laggards
    except Exception:
        pass
    try:
        from options.chain_fetch import chain_workspace_cached
        from options.positioning_read import attach_positioning_read

        chain = chain_workspace_cached("NIFTY", force=False)
        if chain.get("available"):
            attached = attach_positioning_read(chain)
            read = attached.get("positioning_read") or {}
            out["options"] = {
                "pcr": attached.get("pcr"),
                "max_pain": attached.get("max_pain"),
                "atm_iv": attached.get("atm_iv"),
                "bias": attached.get("bias"),
                "note": attached.get("note"),
                "spot": attached.get("spot"),
            }
            out["options_stance"] = {
                "stance": read.get("stance"),
                "score": read.get("score"),
                "confidence": read.get("confidence"),
                "headline": read.get("headline"),
                "consider_for": read.get("consider_for") or [],
                "honesty": read.get("honesty"),
            }
    except Exception:
        try:
            from options.analytics import nifty_options_summary

            opt = nifty_options_summary()
            if opt:
                out["options"] = opt
        except Exception:
            pass
    return out


def _movers_from_bhav(top_n: int = 5) -> tuple[list[dict], list[dict]]:
    """Top gainers/losers from the latest bhavcopy session (liquid stocks only)."""
    try:
        from data.bhavcopy_store import get_ohlcv, store_symbols
        from scan.unified_scanner import is_beaten_down_arr

        rows = []
        for sym in store_symbols():
            df = get_ohlcv(sym)
            if df is None or len(df) < 21:
                continue
            close = df["close"].values
            vol = df["volume"].values
            price = float(close[-1])
            turnover = float(np.nanmean(vol[-20:])) * price
            if turnover < 5e7:
                continue
            highs = df["high"].values[-250:] if "high" in df.columns else close[-250:]
            if is_beaten_down_arr(highs, price):
                continue
            chg = (close[-1] / close[-2] - 1) * 100
            rows.append({"symbol": sym, "price": round(price, 1), "chg_pct": round(float(chg), 2)})
        rows.sort(key=lambda r: r["chg_pct"], reverse=True)
        return rows[:top_n], rows[-top_n:][::-1]
    except Exception as exc:
        log.debug("pulse_movers_failed", error=str(exc))
        return [], []


def _losing_momentum() -> dict | None:
    """A liquid stock breaking down — fell hard and closed below its 50-day avg."""
    try:
        from data.bhavcopy_store import get_ohlcv, store_symbols
        from scan.unified_scanner import is_beaten_down_arr

        worst, worst_score = None, 0.0
        for sym in store_symbols():
            df = get_ohlcv(sym)
            if df is None or len(df) < 60:
                continue
            close = df["close"].values
            vol = df["volume"].values
            price = float(close[-1])
            if float(np.nanmean(vol[-20:])) * price < 2e8:
                continue
            highs = df["high"].values[-250:] if "high" in df.columns else close[-250:]
            if is_beaten_down_arr(highs, price):
                continue
            sma50 = float(close[-50:].mean())
            chg5 = (close[-1] / close[-6] - 1) * 100
            hi20 = float(np.max(close[-20:]))
            fall_from_high = (hi20 - price) / hi20 * 100
            if price < sma50 and chg5 < -4 and fall_from_high > worst_score:
                worst_score = fall_from_high
                worst = {
                    "symbol": sym,
                    "price": round(price, 1),
                    "chg_5d": round(float(chg5), 1),
                    "note": (
                        f"{fall_from_high:.0f}% off 20-day high, closed below 50-day average — "
                        "research caution, not a short ticket"
                    ),
                }
        return worst
    except Exception:
        return None


def _sector_heat(limit: int = 6) -> dict[str, Any]:
    try:
        from scan.sector_heat import sector_performance

        rows = sector_performance(min_members=3) or []
        if not rows:
            return {"available": False, "leaders": [], "laggards": [], "message": "Sector heat unavailable"}
        return {
            "available": True,
            "leaders": rows[:limit],
            "laggards": list(reversed(rows[-limit:])),
            "message": "",
        }
    except Exception as exc:
        return {"available": False, "leaders": [], "laggards": [], "message": str(exc)}


def _relative_strength(rows: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    """Top scan-score names as RS leaders — disclosed as scan-relative, not a new model."""
    ranked = sorted(
        [r for r in rows if _f(r.get("score")) > 0 and not r.get("chase_risk")],
        key=lambda r: _f(r.get("score")),
        reverse=True,
    )
    out = []
    for r in ranked[:limit]:
        out.append(
            {
                "symbol": r["symbol"],
                "company": r.get("company") or r["symbol"],
                "score": r.get("score"),
                "price": r.get("price"),
                "change_pct": r.get("change_pct"),
                "status": r.get("status"),
                "why": r.get("why") or "Leading current scan score vs peers in store",
                "note": "Relative strength proxy = durable scan score (not a separate RS model print)",
            }
        )
    return out


def _sniper_breakouts(limit: int = 4) -> list[dict[str, Any]]:
    try:
        from product.sniper_board import load_board

        board = load_board() or {}
        hits = list(board.get("hits") or [])
        hits.sort(key=lambda h: str(h.get("confirmed_at") or ""), reverse=True)
        out = []
        for hit in hits[:limit]:
            out.append(
                {
                    "symbol": str(hit.get("symbol") or "").upper(),
                    "price": hit.get("price") or hit.get("confirm_price"),
                    "change_pct": hit.get("change_pct"),
                    "entry": hit.get("entry") or hit.get("pivot"),
                    "stop": hit.get("stop"),
                    "target": hit.get("target"),
                    "signals": ["sniper_confirmed"],
                    "reasons": [str(hit.get("reason") or hit.get("note") or "Sniper confirmed breakout")],
                    "why": str(hit.get("reason") or "Sniper confirmed breakout"),
                    "status": "Confirmed breakout",
                    "source": "sniper_board",
                }
            )
        return out
    except Exception:
        return []


def _headlines(max_n: int = 5) -> list[str]:
    try:
        from news.day_story_engine import build_day_stories

        stories = build_day_stories(hours=20, limit=max_n, refresh_if_stale=False).get("stories") or []
        heads = [str(s.get("headline") or "")[:140] for s in stories if s.get("headline")]
        if heads:
            return heads
    except Exception:
        pass
    try:
        from news.curator_store import NewsCuratorStore

        store = NewsCuratorStore(Path("logs/news_curator.sqlite3"))
        try:
            rows = store.recent(hours=18, limit=max(max_n * 3, 15), min_impact=20)
            heads = [str(getattr(r, "headline", "") or "")[:140] for r in rows]
            heads = [h for h in heads if h]
            if heads:
                return heads[:max_n]
        finally:
            store.close()
    except Exception:
        pass
    try:
        from news.fetcher import NewsFetcher

        arts = NewsFetcher().fetch_all(max_age_hours=18)
        return [a.headline[:140] for a in arts[:max_n]]
    except Exception:
        return []


def _day_stories(max_n: int = 5, *, refresh_if_stale: bool = False) -> list[dict[str, Any]]:
    try:
        from news.day_story_engine import build_day_stories

        payload = build_day_stories(
            hours=20,
            limit=max_n,
            refresh_if_stale=refresh_if_stale,
        )
        return [dict(row) for row in (payload.get("stories") or []) if isinstance(row, dict)]
    except Exception:
        return []


def _global_cues() -> list[dict[str, Any]]:
    """Soft-fail global/US cues — omit when Yahoo/US path is down."""
    out: list[dict[str, Any]] = []
    try:
        from product import us_retail

        ov = us_retail.overview() or {}
        for item in ov.get("indices") or []:
            if item.get("name") and item.get("chg_pct") is not None:
                out.append(
                    {
                        "name": item.get("name"),
                        "price": item.get("price"),
                        "chg_pct": item.get("chg_pct"),
                        "source": "us_retail",
                    }
                )
        if out:
            return out
    except Exception:
        pass
    try:
        import yfinance as yf

        for name, tk in (
            ("S&P 500", "^GSPC"),
            ("Nasdaq", "^IXIC"),
            ("Gold", "GC=F"),
            ("Crude", "CL=F"),
            ("Nikkei", "^N225"),
            ("Hang Seng", "^HSI"),
        ):
            try:
                fi = yf.Ticker(tk).fast_info
                last = float(getattr(fi, "last_price", 0) or 0)
                prev = float(getattr(fi, "previous_close", 0) or 0)
                if last and prev:
                    out.append(
                        {
                            "name": name,
                            "price": last,
                            "chg_pct": round((last - prev) / prev * 100, 2),
                            "source": "yfinance",
                        }
                    )
            except Exception:
                continue
    except Exception:
        pass
    return out


def _takeaways(
    snapshot: Mapping[str, Any],
    gainers: list[dict],
    sectors: Mapping[str, Any],
    buzzing: dict | None,
) -> list[str]:
    takeaways: list[str] = []
    for idx in snapshot.get("indices") or []:
        arrow = "up" if _f(idx.get("chg_pct")) >= 0 else "down"
        takeaways.append(
            f"{idx['name']} {arrow} {_f(idx.get('chg_pct')):+.2f}% at {_f(idx.get('price')):,.0f}"
        )
    stance = (snapshot.get("options_stance") or {}).get("stance")
    if stance:
        takeaways.append(f"NIFTY options positioning stance: {stance} (research context, not a buy)")
    leaders = (sectors.get("leaders") or [])[:2]
    if leaders:
        takeaways.append(
            "Sector heat leaders: "
            + ", ".join(f"{r['sector']} ({r['chg_1d']:+.1f}%)" for r in leaders)
        )
    if gainers:
        takeaways.append(f"{gainers[0]['symbol']} top liquid gainer ({gainers[0]['chg_pct']:+.1f}%)")
    if buzzing:
        takeaways.append(
            f"Buzzing: {buzzing['symbol']} "
            f"({_f(buzzing.get('change_pct')):+.1f}% · {_f(buzzing.get('volume_ratio')):.1f}x vol)"
        )
    if snapshot.get("commentary"):
        takeaways.append(str(snapshot["commentary"]))
    return takeaways[:5]


def build_pulse(*, persist: bool = True) -> dict[str, Any]:
    """Assemble the full Daily Pulse from live system state."""
    rows, universe_size, scan_as_of, scan_source = _scan_universe()
    gainers, losers = _movers_from_bhav()
    snapshot = _market_snapshot()
    sectors = _sector_heat()
    gaps: list[str] = []

    buzzing = None
    movers = [
        r
        for r in rows
        if abs(_f(r.get("change_pct"))) >= 3 and _f(r.get("volume_ratio"), 1) >= 2
    ]
    if movers:
        b = max(movers, key=lambda r: _f(r.get("volume_ratio")) * abs(_f(r.get("change_pct"))))
        buzzing = {
            **b,
            "note": (
                f"{_f(b.get('change_pct')):+.1f}% move on {_f(b.get('volume_ratio')):.1f}× volume — "
                + (b.get("why") or (b.get("reasons") or ["strong interest"])[0])
            ),
        }
    else:
        gaps.append("No high-volume buzz name in the current scan (≥3% move and ≥2× volume)")

    pre = [r for r in rows if r.get("pre_breakout")]
    strength = None
    if pre:
        strength = min(pre, key=lambda r: _f(r.get("pivot_distance_pct"), 99))
        if not strength.get("why") and strength.get("reasons"):
            strength = {**strength, "why": strength["reasons"][0]}
    else:
        gaps.append("No pre-breakout / accumulation candidate in the current scan")

    scan_breakouts = [
        r
        for r in rows
        if any(
            s in ("52-week high breakout", "Resistance break on volume")
            or "BREAKOUT" in str(s).upper()
            for s in (r.get("signals") or [])
        )
        or str(r.get("status") or "") == "Ready to trade"
    ][:4]
    sniper = _sniper_breakouts()
    # Prefer confirmed sniper hits, then scan breakouts (dedupe by symbol).
    seen: set[str] = set()
    today_brk: list[dict[str, Any]] = []
    for r in sniper + scan_breakouts:
        sym = r.get("symbol")
        if not sym or sym in seen:
            continue
        seen.add(str(sym))
        today_brk.append(r)
        if len(today_brk) >= 4:
            break
    tomorrow_brk = sorted(pre, key=lambda r: _f(r.get("pivot_distance_pct"), 99))[:4]
    if not today_brk:
        gaps.append("No confirmed breakouts in sniper board / scan right now")
    if not tomorrow_brk:
        gaps.append("No near-pivot watch names for tomorrow")

    weak = _losing_momentum()
    if not weak:
        gaps.append("No liquid breakdown name matched the losing-momentum filter")

    rs = _relative_strength(rows)
    if not rs:
        gaps.append("Relative-strength leaders unavailable (empty or unscored scan)")

    day_stories = _day_stories(5, refresh_if_stale=False)
    headlines = [str(s.get("headline") or "") for s in day_stories if s.get("headline")]
    if not headlines:
        headlines = _headlines()
    if not headlines:
        gaps.append("No fresh headlines in curator/fetcher")
    if not day_stories:
        gaps.append("No wrap-ready day stories ranked yet — refresh market news")

    cues = _global_cues()
    if not cues:
        gaps.append("Global/US cues unavailable")

    if not gainers and not losers:
        gaps.append("Bhav movers unavailable — bhav store may be empty or stale")
    if not sectors.get("available"):
        gaps.append("Sector heat unavailable")

    takeaways = _takeaways(snapshot, gainers, sectors, buzzing)
    now = datetime.now(timezone.utc)
    pulse = {
        "schema_version": SCHEMA_VERSION,
        "available": True,
        "report_type": "DAILY_STREET_PULSE",
        "title": "Daily Street Pulse",
        "date": datetime.now().strftime("%d %B %Y"),
        "generated_at": now.isoformat(),
        "takeaways": takeaways,
        "wrap_of_the_day": {"available": False, "bullets": []},
        "snapshot": snapshot,
        "sectors": sectors,
        "gainers": gainers,
        "losers": losers,
        "buzzing": buzzing,
        "strength": strength,
        "weak": weak,
        "relative_strength": rs,
        "breakouts_today": today_brk,
        "breakouts_tomorrow": tomorrow_brk,
        "global_cues": cues,
        "day_stories": day_stories,
        "headlines": headlines,
        "scanned": universe_size,
        "scan_as_of": scan_as_of,
        "scan_source": scan_source,
        "gaps": gaps,
        "places_orders": False,
        "live_locked": True,
        "signal_desk": False,
        "honesty": (
            "QuantTerm Daily Street Pulse composes real scan, bhav, options, news and "
            "optional US/global cues. Missing sections stay missing. "
            "SUPPORTIVE / buzzing / breakout labels are research context — not buy tickets. Paper-first."
        ),
        "disclaimer": (
            "This digest organises evidence already in QuantTerm stores. "
            "It is not investment advice and does not place orders."
        ),
    }
    # System-composed Wrap of the Day (manual override wins when present).
    try:
        from product.wrap_of_the_day import compose_from_pulse, load_manual_override, save_wrap

        manual = load_manual_override()
        wrap = manual or compose_from_pulse(pulse)
        pulse["wrap_of_the_day"] = wrap
        if wrap.get("available") and wrap.get("bullets"):
            pulse["takeaways"] = [str(b) for b in wrap.get("bullets") or [] if str(b).strip()][:6]
            if not manual and persist:
                save_wrap(
                    wrap.get("bullets") or [],
                    date=str(wrap.get("date") or ""),
                    source="auto",
                    gaps=list(wrap.get("gaps") or []),
                )
    except Exception as exc:
        log.debug("pulse_wrap_miss", error=str(exc))
        pulse["wrap_of_the_day"] = {"available": False, "bullets": [], "message": str(exc)}

    if persist:
        try:
            save_pulse(pulse)
        except Exception as exc:
            log.debug("pulse_persist_failed", error=str(exc))
    return pulse


def save_pulse(pulse: Mapping[str, Any], path: str | Path | None = None) -> Path:
    target = Path(path or DEFAULT_PULSE_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(pulse), indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return target


def load_pulse(path: str | Path | None = None) -> dict[str, Any] | None:
    target = Path(path or DEFAULT_PULSE_PATH)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        return payload
    except Exception:
        return None


def _attach_live_wrap(pulse: dict[str, Any]) -> dict[str, Any]:
    """Refresh wrap on read — auto-compose from this pulse; manual override wins."""
    try:
        from product.wrap_of_the_day import compose_from_pulse, load_manual_override, save_wrap

        manual = load_manual_override()
        wrap = manual or compose_from_pulse(pulse)
        pulse["wrap_of_the_day"] = wrap
        if wrap.get("available") and wrap.get("bullets"):
            pulse["takeaways"] = [str(b) for b in wrap.get("bullets") or [] if str(b).strip()][:6]
            if not manual:
                save_wrap(
                    wrap.get("bullets") or [],
                    date=str(wrap.get("date") or ""),
                    source="auto",
                    gaps=list(wrap.get("gaps") or []),
                )
    except Exception:
        pass
    return pulse


def _attach_holdings_desk(pulse: dict[str, Any]) -> dict[str, Any]:
    """Attach last holdings-desk scorecard (does not re-run the engine on every read)."""
    try:
        from product.holdings_desk import load_desk

        desk = load_desk()
        pulse["holdings_desk"] = desk
    except Exception as exc:
        pulse["holdings_desk"] = {
            "available": False,
            "rows": [],
            "message": str(exc),
            "places_orders": False,
        }
    return pulse


def pulse_api_payload(*, force: bool = False) -> dict[str, Any]:
    """API helper — reuse persisted pulse unless force rebuild requested."""
    if not force:
        cached = load_pulse()
        if cached and cached.get("available"):
            return _attach_holdings_desk(_attach_live_wrap(cached))
    return _attach_holdings_desk(_attach_live_wrap(build_pulse(persist=True)))


def pulse_to_telegram(pulse: dict) -> str:
    """Compact plain-English Telegram pulse (HTML)."""
    lines = [f"<b>Daily Pulse</b> — {pulse.get('date') or 'today'}"]
    wrap = pulse.get("wrap_of_the_day") or {}
    wrap_bullets = list(wrap.get("bullets") or []) if isinstance(wrap, dict) else []
    takeaways = list(pulse.get("takeaways") or [])[:5]
    if wrap_bullets:
        lines.append("")
        lines.append("<b>Wrap of the Day</b>")
        for i, t in enumerate(wrap_bullets[:5], 1):
            lines.append(f"{i}) {t}")
    elif takeaways:
        lines.append("")
        lines.append("<b>Today in short</b>")
        for t in takeaways:
            lines.append(f"• {t}")
    stance = ((pulse.get("snapshot") or {}).get("options_stance") or {}).get("stance")
    if stance:
        meaning = {
            "SUPPORTIVE": "friendly for a long watch",
            "CAUTION": "mixed — wait for a cleaner entry",
            "HOSTILE": "not friendly for fresh longs",
            "NEUTRAL": "balanced",
            "INCOMPLETE": "not enough data",
        }.get(str(stance).upper(), "see app")
        lines.append(f"\nOptions mood: <b>{stance}</b> — {meaning} (not a buy)")
    sectors = pulse.get("sectors") or {}
    if sectors.get("available") and sectors.get("leaders"):
        top = ", ".join(
            f"{r.get('sector')} ({float(r.get('chg_1d') or 0):+.1f}%)"
            for r in (sectors.get("leaders") or [])[:3]
        )
        lines.append(f"Strong sectors today: {top}")
    if pulse.get("buzzing"):
        b = pulse["buzzing"]
        chg = b.get("change_pct")
        vol = b.get("volume_ratio")
        bit = []
        if chg is not None:
            bit.append(f"{float(chg):+.1f}%")
        if vol is not None:
            bit.append(f"{float(vol):.1f}× volume")
        lines.append(
            f"\nBuzzing stock: <b>{b.get('symbol')}</b>"
            + (f" ({', '.join(bit)})" if bit else "")
        )
    if pulse.get("strength"):
        s = pulse["strength"]
        dist = s.get("pivot_distance_pct")
        dist_txt = f"{float(dist):.1f}% below breakout level" if dist is not None else "near breakout level"
        lines.append(f"Gaining strength: <b>{s.get('symbol')}</b> — {dist_txt}")
    if pulse.get("weak"):
        w = pulse["weak"]
        lines.append(f"Losing steam: <b>{w.get('symbol')}</b> — stay away for now")
    if pulse.get("relative_strength"):
        syms = ", ".join(r.get("symbol") for r in (pulse.get("relative_strength") or [])[:4] if r.get("symbol"))
        if syms:
            lines.append(f"Relative strength leaders: {syms}")
    if pulse.get("breakouts_today"):
        syms = ", ".join(r.get("symbol") for r in (pulse.get("breakouts_today") or [])[:4] if r.get("symbol"))
        if syms:
            lines.append(f"Breakouts confirmed: {syms}")
    if pulse.get("breakouts_tomorrow"):
        syms = ", ".join(r.get("symbol") for r in (pulse.get("breakouts_tomorrow") or [])[:4] if r.get("symbol"))
        if syms:
            lines.append(f"\nWatch tomorrow: {syms}")
    desk = pulse.get("holdings_desk") or {}
    desk_rows = list(desk.get("rows") or []) if isinstance(desk, dict) else []
    if desk_rows:
        lines.append("\n<b>Holdings desk</b> (research watches)")
        for row in desk_rows[:8]:
            sym = row.get("symbol") or row.get("tradingsymbol")
            sug = row.get("suggestion") or row.get("stance")
            news = ((row.get("news") or {}).get("bias")) or "NONE"
            lines.append(f"• <b>{sym}</b> → {sug} · news {news}")
        if len(desk_rows) > 8:
            lines.append(f"… +{len(desk_rows) - 8} more")
    gaps = list(pulse.get("gaps") or [])
    if gaps:
        lines.append(f"\nMissing data: {gaps[0]}")
    lines.append("\n<i>Research digest only · not a buy ticket · paper-first</i>")
    return "\n".join(lines)


def send_pulse_telegram(*, force_build: bool = True) -> dict:
    """Build (optional) and send Daily Pulse to Telegram. Never places orders."""
    try:
        from alerts.telegram_alerts import AlertEngine
    except Exception as exc:
        return {"sent": False, "configured": False, "error": f"telegram module unavailable: {exc}"}
    engine = AlertEngine()
    if not engine.is_configured():
        return {
            "sent": False,
            "configured": False,
            "error": "Telegram not configured (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env)",
            "reason": "Telegram not configured (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env)",
        }
    pulse = build_pulse(persist=True) if force_build else (load_pulse() or build_pulse(persist=True))
    pulse = _attach_holdings_desk(pulse)
    ok = bool(engine.send(pulse_to_telegram(pulse)))
    err = None if ok else (engine.last_error or "Telegram send failed")
    return {
        "sent": ok,
        "configured": True,
        "date": pulse.get("date"),
        "scan_source": pulse.get("scan_source"),
        "error": err,
        "reason": err,
        "cred_source": getattr(engine, "cred_source", "") or "",
    }
