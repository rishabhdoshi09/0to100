"""Buy thesis for one clicked name — why it is on the desk, with live layers.

Never invents prices, sales, flows, or a book. Missing layers stay missing and
are fetched from the highest-grade source still allowed: Kite depth → NSE
quote-equity → Screener/Yahoo fundamentals resolver. Sector wave uses the
official NSE-universe sector map plus bhavcopy; FII/DII at the stock is
shareholding change + NSE bulk/block prints, never a guessed print.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from product.research_levels import attach_research_levels
from product.stock_workspace import build_stock_workspace, clean_symbol

_FUND_METRIC_KEYS = {
    "market_cap", "pe", "roe", "roce", "sales_growth_3y",
    "profit_growth_3y", "debt_to_equity", "promoter_holding",
    "fii_holding", "dii_holding", "promoter_pledge", "interest_coverage",
}

_INSTITUTION_NEEDLES = (
    "mutual fund", "life insurance", "general insurance", "pension",
    "provident", "investment authority", "world fund", "whiteoak",
    "morgan stanley", "goldman", "blackrock", "vanguard", "fidelity",
    "kuwait", "temasek", "gic private", "norway", "government of singapore",
    "tata aia", "sbi mutual", "icici prudential", "hdfc mutual",
    "kotak mahindra", "aditya birla", "dsp mutual", "hsbc mutual",
    "uti mutual", "nippon", "mirae", "axis mutual", "quant mutual",
    "smallcap world", "capital international", "aberdeen",
)

_DESK_NEEDLES = (
    "securities", "broking", "broker", "hft", "quantcap", "algoquant",
    "hrti", "junomoneta", "nk securities", "microcurves", "alphagrep",
    "irage", "jump trading", "prop",
)


def _f(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        n = float(str(value).replace(",", "").replace("₹", "").replace("%", "").strip())
        return n if n == n else None
    except (TypeError, ValueError):
        return None


def _row_label(row: Mapping[str, Any]) -> str:
    for key in ("", "row_label", "Particulars", "PARTICULARS"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _table_series(
    rows: Sequence[Mapping[str, Any]] | None,
    *needles: str,
    exclude: Sequence[str] = (),
) -> list[dict[str, Any]]:
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        label = _row_label(row).lower()
        if exclude and any(token in label for token in exclude):
            continue
        if not any(needle in label for needle in needles):
            continue
        series: list[dict[str, Any]] = []
        for key, value in row.items():
            if key in ("", "row_label", "Particulars", "PARTICULARS"):
                continue
            num = _f(value)
            if num is None:
                continue
            series.append({"period": str(key), "value": num})
        from fundamentals.period_freshness import normalize_period_points
        return normalize_period_points(series)
    return []


def _series_change(series: Sequence[Mapping[str, Any]], steps: int = 1) -> dict[str, Any] | None:
    if len(series) < steps + 1:
        return None
    latest = series[-1]
    prior = series[-1 - steps]
    latest_v = _f(latest.get("value"))
    prior_v = _f(prior.get("value"))
    if latest_v is None or prior_v is None:
        return None
    delta = round(latest_v - prior_v, 2)
    pct = round((latest_v / prior_v - 1.0) * 100.0, 1) if prior_v else None
    return {
        "latest": latest_v,
        "latest_period": latest.get("period"),
        "prior": prior_v,
        "prior_period": prior.get("period"),
        "delta": delta,
        "pct": pct,
    }


def _align_to_mapped_sector(nse_name: str) -> str:
    """Map an NSE industryInfo label onto the scanner's sector-heat names."""
    needle = str(nse_name or "").strip().lower()
    if not needle:
        return ""
    try:
        from scan.sector_heat import _load_map
        names = sorted({sec for sec in (_load_map() or {}).values() if sec})
    except Exception:
        names = []
    for sec in names:
        low = sec.lower()
        if needle == low or needle in low or low in needle:
            return sec
    tokens = [t for t in needle.replace("&", " ").replace("/", " ").split() if len(t) > 3]
    best, best_hits = "", 0
    for sec in names:
        hits = sum(1 for t in tokens if t in sec.lower())
        if hits > best_hits:
            best, best_hits = sec, hits
    return best if best_hits >= 1 else ""


_screener_sector_memo: dict[str, str] = {}


def _sector_from_screener(symbol: str, raw_data: Mapping[str, Any] | None = None) -> str:
    cached = str((raw_data or {}).get("sector") or "").strip()
    if cached:
        return cached
    if symbol in _screener_sector_memo:
        return _screener_sector_memo[symbol]
    try:
        from fundamentals.screener_deep import ScreenerDeepFetcher
        fetcher = ScreenerDeepFetcher()
        _url, soup = fetcher._fetch_page(symbol)
        path = fetcher._parse_sector_path(soup)
        label = str(path.get("sector") or path.get("industry") or "").strip()
    except Exception:
        label = ""
    _screener_sector_memo[symbol] = label
    return label


def resolve_sector(
    symbol: str,
    workspace_sector: str = "",
    raw_data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Identify the stock's sector before any wave claim."""
    mapped = ""
    try:
        from scan.sector_heat import sector_of
        mapped = str(sector_of(symbol) or "").strip()
    except Exception:
        mapped = ""
    ws = str(workspace_sector or "").strip()
    if ws.lower() in {"", "unclassified", "unknown", "other"}:
        ws = ""
    nse = {}
    nse_label = ""
    aligned = ""
    screener_label = ""
    if not mapped:
        try:
            from data.nse_live import fetch_equity_industry
            nse = fetch_equity_industry(symbol) or {}
        except Exception:
            nse = {}
        for key in ("sector", "industry", "basic_industry", "macro"):
            nse_label = str(nse.get(key) or "").strip()
            if nse_label:
                break
        aligned = _align_to_mapped_sector(nse.get("sector") or "") or _align_to_mapped_sector(nse_label)
        if not aligned and not nse_label:
            screener_label = _sector_from_screener(symbol, raw_data)
            aligned = _align_to_mapped_sector(screener_label)
    sector = mapped or aligned or nse_label or screener_label or ws
    if mapped:
        source = "nse_universe_map"
    elif aligned and nse_label:
        source = "nse_industry"
    elif aligned or screener_label:
        source = "screener"
    elif nse_label:
        source = "nse_industry"
    elif ws:
        source = "workspace"
    else:
        source = ""
    return {
        "sector": sector,
        "source": source,
        "mapped": mapped,
        "workspace": ws,
        "nse_sector": str((nse or {}).get("sector") or ""),
        "nse_industry": str((nse or {}).get("industry") or nse_label),
        "identified": bool(sector),
    }


def classify_client(name: str) -> str:
    """Label a bulk/block counterparty. Never upgrades a desk into a celebrity."""
    raw = str(name or "").strip()
    if not raw:
        return "unknown"
    n = raw.lower()
    if any(token in n for token in _INSTITUTION_NEEDLES):
        return "institution"
    if any(token in n for token in _DESK_NEEDLES):
        return "desk"
    words = [w for w in raw.replace(".", " ").split() if w]
    if 2 <= len(words) <= 4 and not any(
        token in n for token in ("limited", "ltd", "llp", "pvt", "private", "fund", "trust")
    ):
        return "named_person"
    return "named_buyer"


def _holding_change(
    rows: Sequence[Mapping[str, Any]] | None,
    *needles: str,
    exclude: Sequence[str] = (),
) -> dict[str, Any] | None:
    series = _table_series(rows, *needles, exclude=exclude)
    if len(series) < 1:
        return None
    latest = series[-1]
    out: dict[str, Any] = {
        "latest": latest["value"],
        "latest_period": latest["period"],
        "prior": None,
        "prior_period": "",
        "delta_pp": None,
        "action": "held",
    }
    if len(series) >= 2:
        prior = series[-2]
        out["prior"] = prior["value"]
        out["prior_period"] = prior["period"]
        delta = round(latest["value"] - prior["value"], 2)
        out["delta_pp"] = delta
        if delta > 0.05:
            out["action"] = "bought"
        elif delta < -0.05:
            out["action"] = "sold"
        else:
            out["action"] = "held"
    return out


def _index_return_pct(periods: int) -> float | None:
    try:
        from data.index_store import get_index_ohlcv
        frame = get_index_ohlcv("^NSEI")
        if frame is None or len(frame) <= periods:
            return None
        close = frame["Close"] if "Close" in frame.columns else frame["close"]
        last = float(close.iloc[-1])
        prev = float(close.iloc[-1 - periods])
        if prev <= 0:
            return None
        return round((last / prev - 1.0) * 100.0, 2)
    except Exception:
        return None


def _sector_avg_move(sector: str) -> dict[str, Any]:
    """Average 1d/5d move for mapped members of one sector — not the whole market."""
    empty = {"chg_1d": None, "chg_5d": None, "members": 0}
    if not sector:
        return empty
    try:
        from scan.sector_heat import _load_map
        smap = _load_map()
    except Exception:
        return empty
    members = [sym for sym, sec in (smap or {}).items() if sec == sector]
    if not members:
        return empty
    try:
        from data.bhavcopy_store import get_ohlcv
    except Exception:
        return empty
    moves: list[tuple[float, float]] = []
    for sym in members:
        try:
            df = get_ohlcv(sym)
        except Exception:
            continue
        if df is None or len(df) < 6:
            continue
        close = df["close"].values if "close" in df.columns else df["Close"].values
        try:
            c1 = (float(close[-1]) / float(close[-2]) - 1.0) * 100.0
            c5 = (float(close[-1]) / float(close[-6]) - 1.0) * 100.0
        except Exception:
            continue
        if c1 == c1 and c5 == c5:
            moves.append((c1, c5))
    if len(moves) < 3:
        return {"chg_1d": None, "chg_5d": None, "members": len(moves)}
    return {
        "chg_1d": round(sum(m[0] for m in moves) / len(moves), 2),
        "chg_5d": round(sum(m[1] for m in moves) / len(moves), 2),
        "members": len(moves),
    }


def _scan_pack(sector: str, scan_records: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    if not sector:
        return {"count": 0, "names": []}
    names: list[str] = []
    try:
        from scan.sector_heat import sector_of
    except Exception:
        sector_of = lambda _s: ""  # type: ignore[assignment, misc]
    for row in scan_records or []:
        if not isinstance(row, Mapping):
            continue
        sym = str(row.get("symbol") or "").upper()
        if not sym:
            continue
        row_sec = str(row.get("sector") or sector_of(sym) or "").strip()
        if row_sec == sector:
            names.append(sym)
    # unique, keep order
    seen: set[str] = set()
    uniq: list[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        uniq.append(name)
    return {"count": len(uniq), "names": uniq[:8]}


def _sector_bulk_net(sector: str, deals: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    if not sector:
        return {"net_qty": 0.0, "buy_prints": 0, "sell_prints": 0, "names": []}
    try:
        from scan.sector_heat import sector_of
    except Exception:
        return {"net_qty": 0.0, "buy_prints": 0, "sell_prints": 0, "names": []}
    net = 0.0
    buys = sells = 0
    names: list[str] = []
    for deal in deals or []:
        if not isinstance(deal, Mapping):
            continue
        sym = str(deal.get("symbol") or "").upper()
        if sector_of(sym) != sector:
            continue
        qty = _f(deal.get("qty")) or 0.0
        side = str(deal.get("side") or "").upper()
        if side == "BUY":
            net += qty
            buys += 1
        elif side == "SELL":
            net -= qty
            sells += 1
        if sym and sym not in names:
            names.append(sym)
    return {
        "net_qty": net,
        "buy_prints": buys,
        "sell_prints": sells,
        "names": names[:6],
    }


def sector_wave_verdict(wave: str) -> dict[str, str]:
    """First reply while discussing a sector wave: YES or NO, nothing in between.

    YES only when the tape is a clean INFLOW. OUTFLOW / MIXED / NO_CLAIM stay NO —
    never dress thin evidence as a supporting wave.
    """
    kind = str(wave or "NO_CLAIM").upper()
    if kind == "INFLOW":
        return {
            "verdict": "YES",
            "verdict_line": "YES — sector money is coming in around this name.",
        }
    if kind == "OUTFLOW":
        return {
            "verdict": "NO",
            "verdict_line": "NO — sector money is leaving, not supporting this name.",
        }
    if kind == "MIXED":
        return {
            "verdict": "NO",
            "verdict_line": "NO — sector tape is mixed; do not treat this as a supporting wave.",
        }
    return {
        "verdict": "NO",
        "verdict_line": "NO — not enough current sector evidence to claim a wave.",
    }


def build_sector_wave(
    symbol: str,
    workspace_sector: str = "",
    scan_records: Sequence[Mapping[str, Any]] | None = None,
    flows: Mapping[str, Any] | None = None,
    raw_data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ident = resolve_sector(symbol, workspace_sector, raw_data=raw_data)
    sector = ident["sector"]
    if not ident["identified"]:
        return {
            **ident,
            "wave": "NO_CLAIM",
            **sector_wave_verdict("NO_CLAIM"),
            "headline": "Sector not identified — no inflow claim.",
            "note": "Sector is unknown in the NSE map, NSE quote, and Screener peer header — no inflow claim.",
            "chg_1d": None,
            "chg_5d": None,
            "nifty_1d": None,
            "nifty_5d": None,
            "members": 0,
            "pack": {"count": 0, "names": []},
            "bulk": {"net_qty": 0.0, "buy_prints": 0, "sell_prints": 0, "names": []},
            "bullets": ["Cannot talk about a sector wave until the stock's sector is known."],
        }

    move = _sector_avg_move(sector)
    nifty_1d = _index_return_pct(1)
    nifty_5d = _index_return_pct(5)
    pack = _scan_pack(sector, scan_records)
    deals = list((flows or {}).get("bulk_deals") or []) + list((flows or {}).get("block_deals") or [])
    bulk = _sector_bulk_net(sector, deals)

    chg5 = move.get("chg_5d")
    vs_nifty = None
    if chg5 is not None and nifty_5d is not None:
        vs_nifty = round(chg5 - nifty_5d, 2)

    money_in = False
    money_out = False
    if chg5 is not None and chg5 > 0 and (vs_nifty is None or vs_nifty > 0):
        money_in = True
    if chg5 is not None and chg5 < 0 and (vs_nifty is None or vs_nifty < 0):
        money_out = True
    if bulk["net_qty"] > 0:
        money_in = True
    if bulk["net_qty"] < 0:
        money_out = True
    if pack["count"] >= 3 and chg5 is not None and chg5 > 0:
        money_in = True

    if money_in and not money_out:
        wave = "INFLOW"
    elif money_out and not money_in:
        wave = "OUTFLOW"
    elif chg5 is None and pack["count"] == 0 and bulk["buy_prints"] + bulk["sell_prints"] == 0:
        wave = "NO_CLAIM"
    else:
        wave = "MIXED"

    bullets: list[str] = [f"Sector: {sector} (from {ident['source'].replace('_', ' ')})"]
    if move["members"]:
        line = f"{sector} basket ({move['members']} mapped names)"
        if move["chg_1d"] is not None:
            line += f" {move['chg_1d']:+.2f}% 1d"
        if chg5 is not None:
            line += f" · {chg5:+.2f}% 5d"
        bullets.append(line)
    if nifty_5d is not None and vs_nifty is not None:
        vs = "ahead of" if vs_nifty > 0 else "behind" if vs_nifty < 0 else "in line with"
        bullets.append(f"Vs Nifty 50 5d ({nifty_5d:+.2f}%): {vs} by {abs(vs_nifty):.2f}pp")
    if pack["count"]:
        extra = ", ".join(pack["names"][:4])
        bullets.append(f"{pack['count']} names from this sector are on today's desk" + (f" ({extra})" if extra else ""))
    if bulk["buy_prints"] or bulk["sell_prints"]:
        side = "net bulk buying" if bulk["net_qty"] > 0 else "net bulk selling" if bulk["net_qty"] < 0 else "two-way bulk prints"
        bullets.append(
            f"NSE bulk/block in {sector}: {side} "
            f"({bulk['buy_prints']} buy / {bulk['sell_prints']} sell prints)"
        )
    if wave == "NO_CLAIM":
        bullets.append("Not enough members or prints to call a sector money wave.")

    headlines = {
        "INFLOW": f"{sector} is seeing money — basket up and/or bulk buying in the pack.",
        "OUTFLOW": f"{sector} is leaking money — basket down and/or bulk selling in the pack.",
        "MIXED": f"{sector} is mixed — some inflow evidence, some outflow. Not a clean wave.",
        "NO_CLAIM": f"{sector} identified, but there is not enough evidence to call a wave.",
    }
    if wave == "INFLOW" and bulk["net_qty"] > 0 and (chg5 is None or chg5 <= 0):
        headlines["INFLOW"] = (
            f"{sector}: NSE bulk/block buying in the pack, even though the basket is not up."
        )
    elif wave == "INFLOW" and chg5 is not None and chg5 > 0:
        headlines["INFLOW"] = f"{sector} basket is ahead of Nifty — money is rotating here."
    return {
        **ident,
        "wave": wave,
        **sector_wave_verdict(wave),
        "headline": headlines[wave],
        "note": (
            "NSE does not publish FII/DII by sector. This wave uses the stock's mapped "
            "sector, the sector basket vs Nifty, desk pack heat, and NSE bulk/block prints "
            "in that sector — not a guessed flow number."
        ),
        "chg_1d": move.get("chg_1d"),
        "chg_5d": move.get("chg_5d"),
        "nifty_1d": nifty_1d,
        "nifty_5d": nifty_5d,
        "vs_nifty_5d_pp": vs_nifty,
        "members": move.get("members") or 0,
        "pack": pack,
        "bulk": bulk,
        "bullets": bullets,
    }


def _deal_rows_for_symbol(symbol: str, flows: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for kind, key in (("bulk", "bulk_deals"), ("block", "block_deals")):
        for deal in list((flows or {}).get(key) or []):
            if not isinstance(deal, Mapping):
                continue
            if str(deal.get("symbol") or "").upper() != symbol:
                continue
            client = str(deal.get("client") or "").strip()
            out.append({
                "kind": kind,
                "client": client,
                "client_kind": classify_client(client),
                "side": str(deal.get("side") or "").upper(),
                "qty": _f(deal.get("qty")),
                "price": _f(deal.get("price")),
            })
    return out


def build_smart_money(
    symbol: str,
    shareholding: Sequence[Mapping[str, Any]] | None,
    flows: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fii = _holding_change(shareholding, "fii", "foreign institutional", "foreign portfolio")
    dii = _holding_change(shareholding, "dii", "domestic institutional")
    promoter = _holding_change(shareholding, "promoter", exclude=("pledge", "encumber"))
    deals = _deal_rows_for_symbol(symbol, flows)
    market = dict((flows or {}).get("fii_dii") or {}) if isinstance(flows, Mapping) else {}

    bullets: list[str] = []
    if fii:
        if fii["delta_pp"] is not None:
            bullets.append(
                f"FII holding {fii['latest']:.2f}% as of {fii['latest_period']} "
                f"({fii['action']} {abs(fii['delta_pp']):.2f}pp vs {fii['prior_period']})"
            )
        else:
            bullets.append(f"FII holding {fii['latest']:.2f}% as of {fii['latest_period']} — prior quarter not in table")
    else:
        bullets.append("FII holding for this stock is not in the filings pack yet.")
    if dii:
        if dii["delta_pp"] is not None:
            bullets.append(
                f"DII holding {dii['latest']:.2f}% as of {dii['latest_period']} "
                f"({dii['action']} {abs(dii['delta_pp']):.2f}pp vs {dii['prior_period']})"
            )
        else:
            bullets.append(f"DII holding {dii['latest']:.2f}% as of {dii['latest_period']} — prior quarter not in table")
    else:
        bullets.append("DII holding for this stock is not in the filings pack yet.")
    if promoter and promoter.get("delta_pp") not in (None, 0):
        bullets.append(
            f"Promoter holding {promoter['latest']:.2f}% "
            f"({promoter['action']} {abs(promoter['delta_pp']):.2f}pp last quarter)"
        )

    influencers: list[dict[str, Any]] = []
    for deal in deals:
        kind = deal["client_kind"]
        if kind in {"institution", "named_person"} and deal.get("client"):
            influencers.append(deal)
            who = deal["client"]
            label = "institution" if kind == "institution" else "named buyer"
            qty = f"{int(deal['qty']):,}" if deal.get("qty") else "?"
            px = f" @ ₹{deal['price']:.2f}" if deal.get("price") else ""
            bullets.append(
                f"NSE {deal['kind']} deal: {who} ({label}) {deal['side']} {qty} shares{px}"
            )
        elif kind == "desk" and deal.get("client"):
            bullets.append(
                f"NSE {deal['kind']} print via {deal['client']} — dealing desk, not treated as an influential holder"
            )

    if not deals:
        bullets.append("No NSE bulk or block print for this symbol in the latest cached session.")

    if market.get("bias"):
        bullets.append(
            f"All-India cash market ({market.get('date') or 'latest'}): "
            f"FII ₹{market.get('fii_net_cr')}cr · DII ₹{market.get('dii_net_cr')}cr · {market.get('bias')} "
            "— market-wide, not this stock."
        )

    stock_fii_bought = bool(fii and fii.get("action") == "bought")
    stock_dii_bought = bool(dii and dii.get("action") == "bought")
    inst_bought = any(d["side"] == "BUY" and d["client_kind"] == "institution" for d in deals)
    if stock_fii_bought or stock_dii_bought or inst_bought:
        headline = "Institutions added to this name recently (filings and/or NSE prints)."
        stance = "BOUGHT"
    elif (fii and fii.get("action") == "sold") or (dii and dii.get("action") == "sold"):
        headline = "Institutions reduced this name in the latest shareholding table."
        stance = "SOLD"
    elif deals:
        headline = "Bulk/block prints exist; counterparties are named below. Not the same as FII/DII holding change."
        stance = "PRINTS"
    else:
        headline = "No stock-level FII/DII buy evidence in cache yet."
        stance = "NO_CLAIM"

    stale = bool((flows or {}).get("stale"))
    return {
        "stance": stance,
        "headline": headline,
        "fii": fii,
        "dii": dii,
        "promoter": promoter,
        "deals": deals[:12],
        "influencers": influencers[:8],
        "market_fii_dii": {
            "date": market.get("date") or "",
            "fii_net_cr": market.get("fii_net_cr"),
            "dii_net_cr": market.get("dii_net_cr"),
            "bias": market.get("bias") or "",
            "note": market.get("note") or "",
            "stale": stale,
        } if market else None,
        "bullets": bullets,
        "note": (
            "Stock-level FII/DII is the quarterly shareholding table. "
            "Named buyers come from NSE bulk/block deals. "
            "All-India FII/DII cash is context only."
        ),
    }


def _earnings_block(
    raw_data: Mapping[str, Any],
    fund_metrics: Sequence[Mapping[str, Any]],
    *,
    as_of=None,
) -> dict[str, Any]:
    from fundamentals.period_freshness import (
        expected_latest_quarter,
        pack_latest_period,
        quarters_behind,
    )

    q_sales = _table_series(
        raw_data.get("quarterly_results"), "sales", "revenue", exclude=("growth",)
    )
    q_profit = _table_series(raw_data.get("quarterly_results"), "net profit", "profit after tax", "pat")
    opm = _table_series(raw_data.get("quarterly_results"), "opm")
    npm = _table_series(raw_data.get("quarterly_results"), "npm")
    if not opm:
        opm = _table_series(raw_data.get("profit_loss"), "opm")
    if not npm and q_sales and q_profit:
        by_p = {row["period"]: row["value"] for row in q_profit}
        computed = []
        for row in q_sales:
            s = _f(row.get("value"))
            p = _f(by_p.get(row["period"]))
            if s and s > 0 and p is not None:
                computed.append({"period": row["period"], "value": round(p / s * 100.0, 1)})
        npm = computed
    if not npm:
        npm = _table_series(raw_data.get("profit_loss"), "npm")
    if not opm:
        # Compute latest annual OPM from operating profit / sales when the % row is absent.
        op = _table_series(raw_data.get("profit_loss"), "operating profit", "ebit")
        sales_a = _table_series(raw_data.get("profit_loss"), "sales", "revenue")
        if op and sales_a:
            by_op = {row["period"]: row["value"] for row in op}
            computed = []
            for a in sales_a:
                s = _f(a.get("value"))
                o = _f(by_op.get(a["period"]))
                if s and s > 0 and o is not None:
                    computed.append({"period": a["period"], "value": round(o / s * 100.0, 1)})
            opm = computed
    if not npm:
        sales_a = _table_series(raw_data.get("profit_loss"), "sales", "revenue")
        pat = _table_series(raw_data.get("profit_loss"), "net profit", "profit after tax", "pat")
        if sales_a and pat:
            by_period = {row["period"]: row["value"] for row in pat}
            computed = []
            for row in sales_a:
                s = _f(row.get("value"))
                p = _f(by_period.get(row["period"]))
                if s and s > 0 and p is not None:
                    computed.append({"period": row["period"], "value": round(p / s * 100.0, 1)})
            npm = computed

    qoq = _series_change(q_sales, 1)
    yoy = _series_change(q_sales, 4)
    profit_qoq = _series_change(q_profit, 1)

    metric_map = {
        str(m.get("key")): m for m in fund_metrics if isinstance(m, Mapping)
    }

    def _metric(key: str) -> dict[str, Any] | None:
        m = metric_map.get(key)
        if not m or m.get("value") in (None, ""):
            return None
        return {"key": key, "label": m.get("label"), "value": m.get("value"), "unit": m.get("unit") or ""}

    valuations = [item for item in (
        _metric("pe"), _metric("market_cap"), _metric("roe"), _metric("roce"),
    ) if item]
    pb = None
    price = book = None
    for item in raw_data.get("key_ratios") or []:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "").lower().strip()
        if name in {"p/b", "pb"} or "p/b" in name or "price to book" in name or "price/book" in name:
            pb = _f(item.get("value"))
        elif name in {"current price", "price"}:
            price = _f(item.get("value"))
        elif name == "book value":
            book = _f(item.get("value"))
    if pb is None and price and book and book > 0:
        pb = round(price / book, 2)
    if pb is not None:
        valuations.append({"key": "pb", "label": "Price / book", "value": pb, "unit": "x"})

    earnings_metrics = [item for item in (
        _metric("sales_growth_3y"), _metric("profit_growth_3y"),
    ) if item]

    latest_stamp, latest_label = pack_latest_period(raw_data)
    behind = quarters_behind(latest_stamp, as_of)
    stale = behind is not None and behind >= 2
    current = behind == 0
    bullets: list[str] = []
    if qoq:
        period = qoq["latest_period"]
        amount = f"₹{qoq['latest']:,.0f} cr"
        extra = f" {qoq['pct']:+.1f}% QoQ" if qoq.get("pct") is not None else ""
        if current:
            bullets.append(f"Latest quarter sales {amount} ({period}){extra}")
        elif stale:
            bullets.append(f"As of {period} (stale) sales {amount}{extra}")
        else:
            bullets.append(f"As of {period} sales {amount}{extra}")
    if yoy and yoy.get("pct") is not None:
        bullets.append(f"Sales {yoy['pct']:+.1f}% YoY vs {yoy['prior_period']}")
    if profit_qoq:
        bullets.append(
            f"Net profit ₹{profit_qoq['latest']:,.0f} cr "
            + (f"{profit_qoq['pct']:+.1f}% QoQ" if profit_qoq.get("pct") is not None else "")
        )
    if opm:
        bullets.append(f"Operating margin {opm[-1]['value']:.1f}% ({opm[-1]['period']})")
    if npm:
        bullets.append(f"Net margin {npm[-1]['value']:.1f}% ({npm[-1]['period']})")
    pe = _metric("pe")
    if pe:
        bullets.append(f"P/E {pe['value']}x")
    if pb is not None:
        bullets.append(f"P/B {pb}x")
    if stale:
        need = expected_latest_quarter(as_of).strftime("%b %Y")
        bullets.append(
            f"Filings column is behind — current season expects {need} or later. "
            "Not treated as the latest quarter."
        )
    if not bullets:
        bullets.append("Earnings table not in cache yet — fetch fills this from Screener / Yahoo.")

    return {
        "available": bool(q_sales or opm or npm or valuations or earnings_metrics),
        "quarterly_sales": q_sales[-6:],
        "quarterly_profit": q_profit[-6:],
        "sales_qoq": qoq,
        "sales_yoy": yoy,
        "profit_qoq": profit_qoq,
        "opm": opm[-4:] if opm else [],
        "npm": npm[-4:] if npm else [],
        "valuations": valuations,
        "growth": earnings_metrics,
        "bullets": bullets,
        "stale": stale,
        "latest_period": latest_label or (qoq or {}).get("latest_period") or "",
        "quarters_behind": behind,
    }


def _why_chosen(scan: Mapping[str, Any], long_row: Mapping[str, Any], tech: Mapping[str, Any]) -> list[str]:
    bullets: list[str] = []
    reasons = scan.get("reasons") or scan.get("why") or []
    if isinstance(reasons, str) and reasons.strip():
        bullets.append(reasons.strip())
    elif isinstance(reasons, list):
        bullets.extend(str(item).strip() for item in reasons[:4] if str(item).strip())
    signals = [str(s).replace("_", " ") for s in (scan.get("signals") or []) if s]
    if signals:
        bullets.append("Scanner tags: " + ", ".join(signals[:6]))
    grade = str(scan.get("breakout_grade") or "").upper()
    if grade:
        bullets.append(f"Breakout grade {grade}" + (
            f" · conviction {scan.get('breakout_conviction')}"
            if scan.get("breakout_conviction") not in (None, "") else ""
        ))
    vol = _f(scan.get("volume_ratio") or tech.get("volume_ratio"))
    if vol is not None:
        bullets.append(f"Volume {vol:.1f}× the 20-day average")
    rsi = _f(scan.get("rsi") or tech.get("rsi14"))
    if rsi is not None:
        bullets.append(f"RSI {rsi:.0f}")
    cls = str(long_row.get("classification") or "")
    if cls:
        cov = long_row.get("fundamental_coverage")
        cov_s = f" · coverage {round(float(cov) * 100)}%" if cov not in (None, "") else ""
        bullets.append(f"Long-term class {cls.replace('_', ' ')}{cov_s}")
    factors = list(long_row.get("quality_factors") or [])[:3]
    bullets.extend(str(f) for f in factors if f)
    timing = str(long_row.get("timing") or "")
    if timing:
        bullets.append(f"Timing: {timing.replace('_', ' ').title()}")
    trend = str(tech.get("trend_explanation") or tech.get("trend") or "")
    if trend:
        bullets.append(trend)
    if not bullets:
        bullets.append("On the desk from the latest scan — open layers below for the evidence.")
    seen: set[str] = set()
    out: list[str] = []
    for item in bullets:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    out = out[:7]
    out.append("Research candidate, not an order. Invalidation is the stop.")
    return out


def _sales_from_raw(raw_record: Mapping[str, Any], *, as_of=None) -> dict[str, Any]:
    from fundamentals.period_freshness import (
        normalize_period_points,
        pack_filings_stale,
        pack_latest_period,
    )

    data = dict((raw_record or {}).get("data") or {})
    rows = list(data.get("profit_loss") or [])
    series: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        label = str(row.get("") or row.get("row_label") or "").lower()
        if "sales" not in label and "revenue" not in label:
            continue
        if "growth" in label:
            continue
        for key, value in row.items():
            if key in ("", "row_label"):
                continue
            num = _f(str(value).replace(",", "").replace("₹", "").replace("%", ""))
            if num is None:
                continue
            series.append({"period": str(key), "value": num, "sales_cr": num})
        break
    series = [
        {"period": item["period"], "sales_cr": item["sales_cr"]}
        for item in normalize_period_points(series)
    ]
    cagr = None
    window = series[-4:]
    if len(window) >= 2:
        start, end = window[0]["sales_cr"], window[-1]["sales_cr"]
        years = max(1, len(window) - 1)
        if start and start > 0 and end is not None and end >= 0:
            try:
                cagr = round(((end / start) ** (1.0 / years) - 1.0) * 100.0, 1)
            except Exception:
                cagr = None
    fetched = str((raw_record or {}).get("fetched_at") or "")
    _latest_stamp, latest_label = pack_latest_period({"profit_loss": rows})
    stale = pack_filings_stale({"profit_loss": rows}, as_of=as_of)
    note = "Annual sales from the company filings pack (Screener)."
    if stale and latest_label:
        note = (
            f"Annual table as of {latest_label} (stale vs current reporting season). "
            "Not a current-year filing."
        )
    elif not series:
        note = "Sales history not in cache yet — fetch fills this from Screener / Yahoo."
    return {
        "available": bool(series),
        "cagr_3y": cagr,
        "series": series[-6:],
        "source": "screener" if series else "",
        "as_of": fetched,
        "as_of_period": latest_label,
        "stale": stale,
        "note": note,
    }


def _order_book(symbol: str) -> dict[str, Any]:
    try:
        from data.order_book import fetch_order_book
        return fetch_order_book(symbol)
    except Exception as exc:
        return {
            "available": False,
            "status": "unavailable",
            "note": f"Order book unavailable ({type(exc).__name__})",
            "source": "",
            "bids": [],
            "asks": [],
        }


def _plan(scan: Mapping[str, Any], long_row: Mapping[str, Any], tech: Mapping[str, Any]) -> dict[str, Any]:
    row = attach_research_levels({
        **dict(long_row or {}),
        **dict(scan or {}),
        "price": scan.get("price") or long_row.get("price") or tech.get("close"),
        "atr": tech.get("atr14") or scan.get("atr") or long_row.get("atr"),
        "atr_pct": tech.get("atr_pct") or scan.get("atr_pct"),
        "vol_pct": long_row.get("vol_pct") or scan.get("vol_pct"),
    })
    buy = _f(row.get("entry"))
    stop = _f(row.get("stop"))
    target = _f(row.get("target"))
    upside = _f(row.get("upside_from_buy_pct"))
    return {
        "buy": buy,
        "stop": stop,
        "target": target,
        "upside_from_buy_pct": upside,
        "levels_source": str(row.get("levels_source") or ""),
    }


def _load_flows() -> dict[str, Any]:
    try:
        from data.institutional_flows import get_flows
        return dict(get_flows() or {})
    except Exception:
        return {}


def _load_scan_records() -> list[dict[str, Any]]:
    try:
        from product.scan_store import load_scan
        payload = load_scan() or {}
        return [dict(r) for r in (payload.get("records") or []) if isinstance(r, Mapping)]
    except Exception:
        return []


def build_buy_thesis(symbol: str, *, fetch_missing: bool = False) -> dict[str, Any]:
    symbol = clean_symbol(symbol)
    fetched = {"fundamentals": False, "source": "", "message": ""}
    if fetch_missing:
        try:
            from fundamentals.lazy import ensure_deep_fundamentals
            data = ensure_deep_fundamentals(symbol, force_refresh=False)
            fetched = {
                "fundamentals": bool(data),
                "source": str((data or {}).get("_qt_source") or (data or {}).get("source") or "resolver"),
                "message": "Filled from Screener / Yahoo / cache" if data else "Resolver returned nothing",
            }
        except Exception as exc:
            fetched = {
                "fundamentals": False,
                "source": "",
                "message": f"Could not fetch filings: {type(exc).__name__}: {exc}",
            }

    workspace = build_stock_workspace(symbol)
    scan = dict(workspace.get("scanner") or {})
    long_row = dict(workspace.get("long_term") or {})
    tech = dict(workspace.get("technical") or {})
    fund = dict(workspace.get("fundamentals") or {})
    raw = {}
    try:
        from reporting.evidence_intake import load_raw_fundamentals
        raw = load_raw_fundamentals(symbol) or {}
    except Exception:
        raw = {}
    raw_data = dict((raw or {}).get("data") or {})
    sales = _sales_from_raw(raw)
    if sales.get("cagr_3y") is None:
        for metric in fund.get("metrics") or []:
            if isinstance(metric, Mapping) and metric.get("key") == "sales_growth_3y" and metric.get("value") is not None:
                sales["cagr_3y"] = metric.get("value")
                sales["available"] = True
                break
    plan = _plan(scan, long_row, tech)
    fund_metrics = [
        m for m in (fund.get("metrics") or [])
        if isinstance(m, Mapping) and m.get("key") in _FUND_METRIC_KEYS
    ]
    # Fill FII/DII onto the metric list from shareholding when the workspace left them blank.
    shareholding = list(raw_data.get("shareholding") or [])
    fii_h = _holding_change(shareholding, "fii", "foreign institutional", "foreign portfolio")
    dii_h = _holding_change(shareholding, "dii", "domestic institutional")
    have_keys = {str(m.get("key")) for m in fund_metrics}
    if fii_h and "fii_holding" not in have_keys:
        fund_metrics.append({
            "key": "fii_holding", "label": "FII holding", "value": fii_h["latest"],
            "unit": "%", "meaning": "Latest disclosed foreign institutional ownership.",
        })
    if dii_h and "dii_holding" not in have_keys:
        fund_metrics.append({
            "key": "dii_holding", "label": "DII holding", "value": dii_h["latest"],
            "unit": "%", "meaning": "Latest disclosed domestic institutional ownership.",
        })

    flows = _load_flows()
    scan_records = _load_scan_records()
    sector_wave = build_sector_wave(
        symbol,
        str(workspace.get("sector") or long_row.get("sector") or scan.get("sector") or ""),
        scan_records,
        flows,
        raw_data=raw_data,
    )
    smart_money = build_smart_money(symbol, shareholding, flows)
    earnings = _earnings_block(raw_data, fund_metrics)
    filings_stale = bool(earnings.get("stale") or sales.get("stale"))
    filings_as_of = str(earnings.get("latest_period") or sales.get("as_of_period") or "")
    filings_refresh_attempted = bool(raw_data.get("_filings_refresh_attempted"))
    headline = (
        str(workspace.get("summary") or "")
        or "Clicked name — evidence layers below. Not a buy instruction."
    )
    return {
        "schema_version": 2,
        "symbol": symbol,
        "company": workspace.get("company") or symbol,
        "sector": sector_wave.get("sector") or workspace.get("sector") or "",
        "state": workspace.get("state"),
        "headline": headline,
        "why": _why_chosen(scan, long_row, tech),
        "plan": plan,
        "sector_wave": sector_wave,
        "smart_money": smart_money,
        "earnings": earnings,
        "filings_stale": filings_stale,
        "filings_as_of": filings_as_of,
        "filings_refresh_attempted": filings_refresh_attempted,
        "technical": {
            "available": bool(tech.get("available")),
            "close": tech.get("close"),
            "latest_date": tech.get("latest_date"),
            "trend": tech.get("trend"),
            "trend_explanation": tech.get("trend_explanation"),
            "rsi14": tech.get("rsi14"),
            "volume_ratio": tech.get("volume_ratio"),
            "from_high_pct": tech.get("from_high_pct"),
        },
        "fundamentals": {
            "available": bool(fund.get("available")),
            "coverage_pct": fund.get("coverage_pct") or 0,
            "classification": fund.get("classification") or long_row.get("classification") or "",
            "quality_factors": fund.get("quality_factors") or [],
            "risk_flags": fund.get("risk_flags") or [],
            "metrics": fund_metrics,
            "fetched_at": fund.get("fetched_at") or "",
            "about": (fund.get("company_about") or "")[:400],
        },
        "sales": sales,
        "order_book": _order_book(symbol),
        "gaps": workspace.get("gaps") or [],
        "fetched": fetched,
        "confidence_pct": workspace.get("confidence_pct"),
    }
