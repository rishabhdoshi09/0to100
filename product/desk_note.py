"""Daily desk note — assemble a research wrap from curated news + concepts.

This is the magazine the customer sees on Market Reports:

  wrap bullets (sourced headlines) → concept explainer → company desks
  → mix-shift theme.

Honesty:
  • Never invent a headline, order size, earnings print, or capex number.
  • Empty slot → empty copy. Do not backfill from a blog.
  • Company tiles are a research watch framework, not a BUY list.
  • Expected payoff / live BUY / GTT are not wired here.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

# Evergreen teach-ins used by the desk note AND Education.
GOLD_LOAN_CONCEPT = {
    "id": "concept-gold-loan-collateral",
    "title": "Why a jump in gold prices matters for gold-loan companies",
    "teach_point": (
        "Gold-loan lenders lend against jewellery. If pledged gold becomes more valuable, "
        "the existing loan is backed by stronger collateral. That can mean better protection "
        "for the lender, potentially higher eligible loan amounts, and more customer interest. "
        "High gold prices alone do not guarantee higher profits — loan demand, rates, "
        "competition, defaults and asset quality still matter."
    ),
    "why_it_matters": (
        "Read gold-loan rallies as a collateral-and-demand story, not an automatic earnings beat."
    ),
}

MIX_SHIFT_CONCEPT = {
    "id": "concept-mix-shift",
    "title": "Mix shift: higher-value products, not just more volume",
    "teach_point": (
        "The interesting thesis is often not “auto will grow” or “chemicals will recover”. "
        "It is that the company is trying to sell higher-margin, more specialised products. "
        "Watch the chain: revenue growth → product mix → margins → capital returns → earnings. "
        "A firm growing sales ~15% while mix improves can sometimes grow profit faster — "
        "only if the mix and returns actually show up in filings, not in a slide."
    ),
    "why_it_matters": (
        "Do not treat a sector recovery headline as a stock thesis. Ask what is being sold, "
        "at what margin, and whether new capex is earning its keep."
    ),
}

# Research watch — not recommendations. Numbers stay off this list on purpose.
MIX_SHIFT_DESKS: tuple[dict[str, Any], ...] = (
    {
        "symbol": "SSWL",
        "name": "Steel Strips Wheels",
        "lens": "Auto components — mix toward higher-value wheels/aluminium",
        "watch": [
            "Share of alloy / higher-value products vs steel wheels",
            "EBITDA per wheel (from filings, not a blog)",
            "Exports and new capacity utilisation",
            "Returns on announced capex",
        ],
        "risks": [
            "Auto OEM slowdown",
            "Execution at new capacity",
            "Competition, tariffs, margin volatility",
        ],
    },
    {
        "symbol": "TATVA",
        "name": "Tatva Chintan",
        "lens": "Speciality chemicals — less dependence on one product",
        "watch": [
            "SDA / emission-control chemicals vs one-product concentration",
            "Pharma PTC and any battery/electrolyte salt progress",
            "Customer concentration in filings",
        ],
        "risks": ["China supply", "Customer concentration", "Regulation timing (e.g. Euro 7)"],
    },
    {
        "symbol": "TANFACIND",
        "name": "Tanfac Industries",
        "lens": "Fluorochemicals — R32 / downstream grades",
        "watch": [
            "R32 capacity and actual utilisation (not pre-book talk alone)",
            "Solar / electronics-grade acid mix",
            "Whether valuation already prices a clean ramp",
        ],
        "risks": ["Execution vs booked demand", "Commodity-chemical pricing", "Valuation leaves little room for delay"],
    },
    {
        "symbol": "AETHER",
        "name": "Aether Industries",
        "lens": "CRAMS / CDMO mix vs traditional manufacturing",
        "watch": [
            "Contract-manufacturing revenue share",
            "Named customers only when the company discloses them",
            "R&D and capex converting into commercial sales",
        ],
        "risks": ["R&D spend that never becomes profit", "Capex timing"],
    },
    {
        "symbol": "JUBLINGREA",
        "name": "Jubilant Ingrevia",
        "lens": "Product-mix toward speciality / nutrition / CDMO",
        "watch": [
            "Speciality vs commodity/agro mix in reported segments",
            "Pyridine derivatives, agro-CDMO, electronics chemicals",
            "Evidence of execution after past delays",
        ],
        "risks": ["Execution delays", "Plans without filing evidence"],
    },
)

WRAP_SLOTS: tuple[dict[str, Any], ...] = (
    {
        "id": "policy",
        "label": "SEBI / policy",
        "needles": ("sebi studies", "sebi proposes", "related-party", "related party", "amfi "),
        "prefer_official": True,
        "reject_if": ("pursuant to the provisions of regulation", "listing obligations and disclosure"),
        "empty": "No sourced SEBI/policy headline in the curator yet — do not invent a circular.",
    },
    {
        "id": "flows",
        "label": "Mutual funds / household savings",
        "needles": ("mutual fund", " mf ", "amfi", "sip ", "household saving"),
        "prefer_official": False,
        "reject_if": (),
        "empty": "No sourced mutual-fund industry headline yet.",
    },
    {
        "id": "orders",
        "label": "Large orders",
        "needles": ("gas compression", "ultra-mega", "middle east", "bags ", "bagging/receiving of orders"),
        "symbols": ("LT",),
        "prefer_official": False,
        "reject_if": (),
        "empty": "No sourced large-order headline yet.",
    },
    {
        "id": "gold_loan",
        "label": "Gold-loan NBFCs",
        "needles": ("gold loan", "muthoot", "manappuram", "gold crosses", "gold prices"),
        "symbols": ("MUTHOOTFIN", "MANAPPURAM", "IIFL"),
        "prefer_official": False,
        "reject_if": (),
        "empty": "No sourced gold-loan / bullion headline yet.",
    },
    {
        "id": "global",
        "label": "US / global tape",
        "needles": (
            "us inflation", "fed chair", "federal reserve", "treasury yield",
            "us futures", "bond yield", "s&p", "nasdaq", "us markets", "nvidia",
        ),
        "prefer_official": False,
        "reject_if": (),
        "empty": "No sourced US/global tape headline yet.",
    },
)


def _blob(article: Mapping[str, Any]) -> str:
    return " ".join(
        str(article.get(key) or "")
        for key in ("headline", "summary", "why_it_matters", "source", "category", "event_type")
    ).lower()


def _syms(article: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for key in ("mentioned_symbols", "fno_symbols"):
        for item in article.get(key) or []:
            sym = str(item or "").strip().upper()
            if sym and sym not in out and len(sym) <= 16:
                out.append(sym)
    return out


def _rejected(article: Mapping[str, Any], reject_if: Sequence[str]) -> bool:
    text = _blob(article)
    return any(token in text for token in reject_if)


def _score_article(article: Mapping[str, Any], slot: Mapping[str, Any]) -> int:
    if _rejected(article, slot.get("reject_if") or ()):
        return -1
    text = _blob(article)
    needles = tuple(slot.get("needles") or ())
    hits = sum(1 for n in needles if n in text)
    distinctive = 0
    for token in ("gas compression", "ultra-mega", "middle east", "gold loan", "muthoot", "manappuram", "us inflation", "fed chair", "sebi studies", "related party"):
        if token in needles and token in text:
            distinctive += 1
    want_syms = {str(s).upper() for s in (slot.get("symbols") or ())}
    have = set(_syms(article))
    if want_syms and have & want_syms:
        hits += 3
    if hits <= 0:
        # Official SEBI RSS often has "SEBI" in source without the needle phrase.
        if slot["id"] == "policy" and (
            "sebi" in str(article.get("source") or "").lower()
            or "sebi.gov.in" in str(article.get("url") or "").lower()
        ):
            hits = 1
        else:
            return -1
    score = hits * 10 + distinctive * 18
    try:
        score += min(40, int(article.get("impact_score") or 0) // 3)
    except (TypeError, ValueError):
        pass
    if article.get("official"):
        score += 25 if slot.get("prefer_official") else 8
    if str(article.get("url") or "").startswith("http"):
        score += 2
    return score


def _bullet_from_article(article: Mapping[str, Any], slot: Mapping[str, Any]) -> dict[str, Any]:
    headline = str(article.get("headline") or "").strip()
    summary = str(article.get("summary") or article.get("why_it_matters") or "").strip()
    want = {str(s).upper() for s in (slot.get("symbols") or ())}
    symbols = [s for s in _syms(article) if s in want][:8] if want else []
    return {
        "id": slot["id"],
        "label": slot["label"],
        "available": True,
        "headline": headline,
        "summary": summary[:420],
        "source": str(article.get("source") or ""),
        "url": str(article.get("url") or ""),
        "official": bool(article.get("official")),
        "published_at": str(article.get("published_at") or ""),
        "symbols": symbols,
        "empty_detail": "",
    }


def _empty_slot(slot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": slot["id"],
        "label": slot["label"],
        "available": False,
        "headline": "",
        "summary": "",
        "source": "",
        "url": "",
        "official": False,
        "published_at": "",
        "symbols": [],
        "empty_detail": str(slot.get("empty") or "No sourced headline yet."),
    }


def _fnum(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


_SECTOR_INDICES: tuple[tuple[str, str], ...] = (
    ("^CNXPHARMA", "pharma"),
    ("^CNXFMCG", "FMCG"),
    ("^CNXAUTO", "auto"),
    ("^CNXIT", "IT"),
    ("^CNXMETAL", "metal"),
    ("^CNXENERGY", "energy"),
    ("^CNXREALTY", "realty"),
    ("^NSEBANK", "banking"),
)

_GLOBAL_NEEDLES: tuple[str, ...] = (
    "us inflation", "fed chair", "federal reserve", "treasury yield",
    "us futures", "bond yield", "s&p", "nasdaq", "us markets", "nvidia",
    "wall street", "dow jones", "s&p 500", "sp 500",
)

_FILING_REJECT: tuple[str, ...] = (
    "pursuant to the provisions of regulation",
    "listing obligations and disclosure",
)


def _join_names(names: Sequence[str]) -> str:
    items = [str(n).strip() for n in names if str(n).strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _pretty_sector(name: str) -> str:
    raw = str(name or "").strip()
    if not raw:
        return ""
    if raw.upper() in {"IT", "FMCG", "NBFC"}:
        return raw.upper()
    return raw[:1].upper() + raw[1:]


def _comma_int(value: float) -> str:
    return f"{int(round(value)):,}"


def _pct_label(chg: float) -> str:
    if abs(chg) >= 1.0:
        return f"{abs(chg):.0f}%"
    return f"{abs(chg):.1f}%"


def _move_gerund(chg: float) -> str:
    if chg < -0.05:
        return f"falling {_pct_label(chg)}"
    if chg > 0.05:
        return f"rising {_pct_label(chg)}"
    return f"little changed ({chg:+.1f}%)"


def _move_past(chg: float) -> str:
    if chg < -0.05:
        return f"fell {_pct_label(chg)}"
    if chg > 0.05:
        return f"rose {_pct_label(chg)}"
    return f"was little changed ({chg:+.1f}%)"


def _round_cross(close: float, prev: float, *, step: int = 100) -> str:
    if close <= 0 or prev <= 0 or step <= 0:
        return ""
    if close < prev:
        level = int(close // step + 1) * step
        if close < level <= prev:
            return f"slipping below {_comma_int(level)}"
        return ""
    level = int(close // step) * step
    if prev < level <= close:
        return f"crossing above {_comma_int(level)}"
    return ""


def _index_print(ticker: str) -> dict[str, Any]:
    try:
        from data.index_store import latest_index_print

        return dict(latest_index_print(ticker) or {})
    except Exception:
        return {}


def _recent_closes(ticker: str, n: int = 4) -> list[float]:
    try:
        from data.index_store import recent_index_closes

        return [float(x) for x in (recent_index_closes(ticker, n) or []) if float(x) > 0]
    except Exception:
        return []


def _session_changes(closes: Sequence[float]) -> list[float]:
    out: list[float] = []
    for idx in range(1, len(closes)):
        prev = float(closes[idx - 1])
        cur = float(closes[idx])
        if prev > 0:
            out.append((cur / prev - 1.0) * 100.0)
    return out


def _streak_phrase(changes: Sequence[float]) -> str:
    if not changes:
        return ""
    today = float(changes[-1])
    yesterday = float(changes[-2]) if len(changes) >= 2 else None
    if yesterday is not None and today < -0.05 and yesterday < -0.05:
        return "extended losses for the second straight session"
    if yesterday is not None and today > 0.05 and yesterday > 0.05:
        return "extended gains for the second straight session"
    if today < -0.05:
        return "ended lower"
    if today > 0.05:
        return "ended higher"
    return "ended little changed"


def _is_global_article(article: Mapping[str, Any]) -> bool:
    text = _blob(article)
    return any(token in text for token in _GLOBAL_NEEDLES)


def _is_filing_article(article: Mapping[str, Any]) -> bool:
    text = _blob(article)
    return any(token in text for token in _FILING_REJECT)


def _article_kind(article: Mapping[str, Any]) -> str:
    if _is_filing_article(article):
        return "skip"
    if _is_global_article(article):
        return "global"
    if _syms(article):
        return "stock"
    return "other"


def _line_from_article(
    article: Mapping[str, Any],
    *,
    scan_map: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    headline = str(article.get("headline") or "").strip()
    if not headline:
        return None
    summary = str(article.get("summary") or article.get("why_it_matters") or "").strip()
    text = headline if not summary else f"{headline} {summary}"
    symbols = _syms(article)[:6]
    scan_map = scan_map or {}
    prefix = ""
    if symbols and "%" not in text:
        row = dict(scan_map.get(symbols[0]) or {})
        chg = _fnum(row.get("change_pct"))
        if chg is not None and abs(chg) >= 0.4:
            name = str(row.get("company") or row.get("name") or "").strip() or symbols[0]
            if name.lower() not in text.lower():
                prefix = f"{name} {_move_past(chg)}. "
    return {
        "id": str(article.get("article_id") or headline[:24]),
        "text": " ".join((prefix + text).split())[:480],
        "source": str(article.get("source") or "Sourced news"),
        "official": bool(article.get("official")),
        "url": str(article.get("url") or ""),
        "symbols": symbols,
    }


def official_session_wrap(*, scan_payload: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Item 1 of the wrap: official India tape. Never invents Sensex, sectors, or news."""
    del scan_payload  # session line is indices + sectors, not scan counts
    try:
        from product.market_view import current_market_view

        view = current_market_view()
    except Exception:
        view = None

    nifty = _index_print("^NSEI")
    nifty_chg = _fnum((nifty or {}).get("chg_pct"))
    nifty_px = _fnum((nifty or {}).get("price"))
    if nifty_chg is None and view is not None:
        nifty_chg = _fnum(getattr(view, "nifty_change_1d", None))
    if nifty_px is None and view is not None:
        nifty_px = _fnum(getattr(view, "nifty_price", None))
        if nifty_px is not None and nifty_px <= 0:
            nifty_px = None

    closes = _recent_closes("^NSEI", 4)
    changes = _session_changes(closes)
    if nifty_chg is not None:
        if changes:
            changes = list(changes[:-1]) + [nifty_chg]
        else:
            changes = [nifty_chg]
    streak = _streak_phrase(changes)
    if not streak and nifty_chg is None and view is not None and str(getattr(view, "summary", "") or "").strip():
        return [{
            "id": "session_regime",
            "text": str(view.summary).strip(),
            "source": "QuantTerm regime on official bars",
            "official": True,
            "url": "",
            "symbols": [],
        }]
    if nifty_chg is None and not streak:
        return []

    head = f"Indian markets {streak or 'ended little changed'}"
    nifty_bit = ""
    if nifty_chg is not None:
        nifty_bit = f"the Nifty {_move_gerund(nifty_chg)}"
        if nifty_px:
            nifty_bit += f" to {_comma_int(nifty_px)}"
            prev = closes[-2] if len(closes) >= 2 else None
            crossed = _round_cross(nifty_px, prev) if prev else ""
            if crossed:
                nifty_bit += f", {crossed}"
    bank = _index_print("^NSEBANK")
    bank_chg = _fnum((bank or {}).get("chg_pct"))
    bank_bit = ""
    if bank_chg is not None:
        bank_bit = f"Bank Nifty {_move_past(bank_chg)}"

    clause = head
    extras = [bit for bit in (nifty_bit, bank_bit) if bit]
    if extras:
        if len(extras) == 1:
            clause = f"{head}, with {extras[0]}"
        else:
            clause = f"{head}, with {extras[0]}, while {extras[1]}"

    positive: list[str] = []
    negative: list[str] = []
    for ticker, label in _SECTOR_INDICES:
        if ticker == "^NSEBANK":
            continue
        print = _index_print(ticker)
        chg = _fnum((print or {}).get("chg_pct"))
        if chg is None:
            continue
        pretty = _pretty_sector(label)
        if chg > 0.05:
            positive.append(pretty)
        elif chg < -0.05:
            negative.append(pretty)
    sector = ""
    if positive:
        sector = f"{_join_names(positive[:3])} stocks ended positive"
        if negative:
            sector += f", while {_join_names(negative[:3])} ended lower"
    elif view is not None:
        leaders = [_pretty_sector(str(x)) for x in (getattr(view, "leaders", ()) or [])[:3]]
        laggards = [_pretty_sector(str(x)) for x in (getattr(view, "laggards", ()) or [])[:3]]
        if leaders:
            sector = f"{_join_names(leaders)} stocks led"
            if laggards:
                sector += f", while {_join_names(laggards)} lagged"
        elif laggards:
            sector = f"{_join_names(laggards)} lagged"

    text = clause + "."
    if sector:
        text = f"{text} {sector}."
    return [{
        "id": "session_indices",
        "text": " ".join(text.split()),
        "source": "Official NSE session",
        "official": True,
        "url": "",
        "symbols": [],
    }]


def news_wrap_lines(
    articles: Sequence[Mapping[str, Any]] | None,
    *,
    limit: int = 5,
    kinds: Sequence[str] | None = None,
    scan_payload: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Top sourced headlines. Headline required; no invented copy."""
    want = {str(k) for k in (kinds or ("stock", "other", "global"))}
    scan_map = _scan_map(scan_payload)
    ranked: list[tuple[int, Mapping[str, Any]]] = []
    for article in articles or []:
        if not isinstance(article, Mapping):
            continue
        if not str(article.get("headline") or "").strip():
            continue
        kind = _article_kind(article)
        if kind == "skip" or kind not in want:
            continue
        try:
            score = int(article.get("impact_score") or 0)
        except (TypeError, ValueError):
            score = 0
        ranked.append((score, article))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _score, article in ranked:
        headline = str(article.get("headline") or "").strip()
        key = headline.lower()
        if key in seen:
            continue
        seen.add(key)
        item = _line_from_article(article, scan_map=scan_map)
        if item:
            out.append(item)
        if len(out) >= max(1, int(limit)):
            break
    return out


def daily_wrap(
    *,
    articles: Sequence[Mapping[str, Any]] | None = None,
    scan_payload: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Here's the wrap of the day — official tape, then sourced stock stories, US last."""
    lines = official_session_wrap(scan_payload=scan_payload)
    existing = {str(item.get("text") or "").lower() for item in lines}
    seen_ids = {str(item.get("id") or "") for item in lines}

    def _take(kind: str, limit: int) -> None:
        for item in news_wrap_lines(articles, limit=limit, kinds=(kind,), scan_payload=scan_payload):
            key = str(item.get("text") or "").lower()
            aid = str(item.get("id") or "")
            if key in existing or (aid and aid in seen_ids):
                continue
            existing.add(key)
            if aid:
                seen_ids.add(aid)
            lines.append(item)

    _take("stock", 3)
    remain = max(0, 5 - len(lines) - 1)
    if remain:
        _take("other", remain)
    _take("global", 1)
    if len(lines) < 5:
        _take("other", 5 - len(lines))
    return lines[:5]


def wrap_from_news(articles: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    rows = [a for a in (articles or []) if isinstance(a, Mapping) and str(a.get("headline") or "").strip()]
    used: set[str] = set()
    out: list[dict[str, Any]] = []
    for slot in WRAP_SLOTS:
        ranked: list[tuple[int, Mapping[str, Any]]] = []
        for article in rows:
            aid = str(article.get("article_id") or article.get("url") or article.get("headline"))
            if aid in used:
                continue
            score = _score_article(article, slot)
            if score >= 0:
                ranked.append((score, article))
        if not ranked:
            out.append(_empty_slot(slot))
            continue
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        best = ranked[0][1]
        used.add(str(best.get("article_id") or best.get("url") or best.get("headline")))
        out.append(_bullet_from_article(best, slot))
    return out


def _scan_map(scan_payload: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in list((scan_payload or {}).get("records") or []):
        if not isinstance(row, Mapping):
            continue
        sym = str(row.get("symbol") or "").upper()
        if sym:
            out[sym] = dict(row)
    return out


def _news_for_symbol(articles: Sequence[Mapping[str, Any]], symbol: str) -> Mapping[str, Any] | None:
    want = symbol.upper()
    hits: list[Mapping[str, Any]] = []
    for article in articles:
        if want in _syms(article) or want.lower() in _blob(article):
            hits.append(article)
    if not hits:
        return None
    hits.sort(key=lambda a: int(a.get("impact_score") or 0), reverse=True)
    return hits[0]


def company_desks(
    *,
    articles: Sequence[Mapping[str, Any]] | None = None,
    scan_payload: Mapping[str, Any] | None = None,
    extra_symbols: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    news = [a for a in (articles or []) if isinstance(a, Mapping)]
    scan = _scan_map(scan_payload)
    frames = list(MIX_SHIFT_DESKS)
    extra = [str(s).upper() for s in (extra_symbols or []) if str(s).strip()]
    gold_syms = ["MUTHOOTFIN", "MANAPPURAM", "IIFL"]
    # Gold-loan names ride the wrap; mix-shift names are the watch pack.
    ordered_syms: list[str] = []
    for sym in (*gold_syms, *(f["symbol"] for f in frames), *extra):
        if sym not in ordered_syms:
            ordered_syms.append(sym)
    frame_by = {str(f["symbol"]).upper(): f for f in frames}
    out: list[dict[str, Any]] = []
    for sym in ordered_syms:
        frame = frame_by.get(sym) or {
            "symbol": sym,
            "name": sym,
            "lens": "Mentioned in today’s sourced wrap",
            "watch": ["Open Stock Intelligence for filings and coverage"],
            "risks": ["A headline is not a thesis"],
        }
        article = _news_for_symbol(news, sym)
        row = scan.get(sym)
        available = article is not None or row is not None
        reasons = (row or {}).get("reasons") if row else None
        scan_reason = ""
        if row:
            scan_reason = str(row.get("reason") or "")
            if not scan_reason and isinstance(reasons, list) and reasons:
                scan_reason = str(reasons[0])
        item = {
            "symbol": sym,
            "name": frame["name"],
            "lens": frame["lens"],
            "watch": list(frame.get("watch") or []),
            "risks": list(frame.get("risks") or []),
            "available": available,
            "source_headline": str((article or {}).get("headline") or ""),
            "source_summary": str((article or {}).get("summary") or "")[:280],
            "source": str((article or {}).get("source") or ""),
            "url": str((article or {}).get("url") or ""),
            "scan_status": str((row or {}).get("status") or ""),
            "scan_reason": scan_reason,
            "empty_detail": (
                ""
                if available
                else f"No sourced headline or saved scan row for {sym} today — the mix-shift questions stay, the numbers do not get invented."
            ),
            "is_recommendation": False,
        }
        out.append(item)
    return out


def build_desk_note(
    *,
    articles: Sequence[Mapping[str, Any]] | None = None,
    scan_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    news = [a for a in (articles or []) if isinstance(a, Mapping)]
    wrap = wrap_from_news(news)
    daily = daily_wrap(articles=news, scan_payload=scan_payload)
    gold_hit = next((b for b in wrap if b["id"] == "gold_loan" and b["available"]), None)
    order_hit = next((b for b in wrap if b["id"] == "orders" and b["available"]), None)
    extra = []
    if order_hit:
        known = {"LT"}
        have = [s for s in (order_hit.get("symbols") or []) if s in known]
        extra.extend(have or ["LT"])
    desks = company_desks(articles=news, scan_payload=scan_payload, extra_symbols=extra)
    sourced = sum(1 for b in wrap if b["available"])
    explainers = []
    if gold_hit:
        explainers.append({**GOLD_LOAN_CONCEPT, "attached_to": "gold_loan"})
    # Mix-shift theme is always the reading frame for the watch pack — not a signal.
    explainers.append({**MIX_SHIFT_CONCEPT, "attached_to": "mix_shift"})
    memory: dict[str, Any] = {}
    try:
        from product.case_memory import morning_digest
        memory = morning_digest()
    except Exception as exec_mem:
        memory = {
            "title": "What QuantTerm remembers this morning",
            "blurb": "Case memory unavailable.",
            "setups": [],
            "error": str(exec_mem)[:200],
            "places_orders": False,
        }
    decision_mem: dict[str, Any] = {}
    try:
        from product.decision_memory import morning_strip
        decision_mem = morning_strip()
    except Exception as exec_dm:
        decision_mem = {
            "title": "Decision Memory",
            "blurb": "Decision memory unavailable.",
            "error": str(exec_dm)[:200],
            "places_orders": False,
        }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "title": "Here's the wrap of the day",
        "blurb": (
            "Official Nifty and sector session first. Sourced stock stories next. "
            "US tape only when the curator has it. Empty stays empty."
        ),
        "wrap": wrap,
        "daily_wrap": daily,
        "wrap_sourced": sourced,
        "wrap_empty": 5 - sourced,
        "explainers": explainers,
        "desks": desks,
        "theme": {
            "id": "mix_shift",
            "title": MIX_SHIFT_CONCEPT["title"],
            "body": MIX_SHIFT_CONCEPT["teach_point"],
        },
        "memory": memory,
        "decision_memory": decision_mem,
        "disclaimer": (
            "Desk note is assembled from the news curator and saved scan — not a broker "
            "note, not Reco Wealth, not an order. Do not treat empty slots as “nothing happened”; "
            "they mean QuantTerm has no sourced line yet."
        ),
        "places_orders": False,
    }
