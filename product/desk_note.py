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
        "needles": ("us inflation", "fed chair", "federal reserve", "treasury yield", "us futures", "bond yield"),
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


def official_session_wrap(*, scan_payload: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Numbered wrap lines from official session + last scan. Never invents news."""
    lines: list[dict[str, Any]] = []
    try:
        from product.market_view import current_market_view

        view = current_market_view()
    except Exception:
        view = None
    if view is not None:
        chg = _fnum(view.nifty_change_1d)
        if chg is not None:
            if chg < -0.05:
                move = f"fell {abs(chg):.1f}%"
            elif chg > 0.05:
                move = f"rose {chg:.1f}%"
            else:
                move = f"was little changed ({chg:+.1f}%)"
            text = f"Indian markets {move} on the official Nifty session. {view.summary}"
            lines.append({
                "id": "session_indices",
                "text": " ".join(text.split()),
                "source": "Official NSE session",
                "official": True,
                "url": "",
                "symbols": [],
            })
        elif str(view.summary or "").strip():
            lines.append({
                "id": "session_regime",
                "text": str(view.summary).strip(),
                "source": "QuantTerm regime on official bars",
                "official": True,
                "url": "",
                "symbols": [],
            })
    records = [r for r in list((scan_payload or {}).get("records") or []) if isinstance(r, Mapping)]
    if records:
        ready = sum(1 for r in records if str(r.get("status") or "") == "Ready to trade")
        brk = []
        for row in records:
            sigs = {str(x).upper() for x in (row.get("signals") or [])}
            grade = str(row.get("breakout_grade") or "").upper()
            if sigs & {"BREAKOUT_52W", "BREAKOUT_RES"} or grade in {"A", "B"}:
                sym = str(row.get("symbol") or "").upper()
                if sym and sym not in brk:
                    brk.append(sym)
        bits = [f"Last market scan has {len(records)} name(s)"]
        if ready:
            bits.append(f"{ready} ready to trade")
        if brk:
            bits.append("breakouts " + ", ".join(brk[:6]))
        lines.append({
            "id": "session_scan",
            "text": ". ".join(bits) + ".",
            "source": "Saved market scan",
            "official": True,
            "url": "",
            "symbols": brk[:6],
        })
    return lines


def news_wrap_lines(articles: Sequence[Mapping[str, Any]] | None, *, limit: int = 5) -> list[dict[str, Any]]:
    """Top sourced headlines, ranked by impact. Headline required; no invented copy."""
    ranked: list[tuple[int, Mapping[str, Any]]] = []
    for article in articles or []:
        if not isinstance(article, Mapping):
            continue
        headline = str(article.get("headline") or "").strip()
        if not headline:
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
        summary = str(article.get("summary") or article.get("why_it_matters") or "").strip()
        text = headline if not summary else f"{headline} {summary}"
        symbols: list[str] = []
        for item in list(article.get("mentioned_symbols") or [])[:6]:
            sym = str(item or "").strip().upper()
            if sym and sym not in symbols:
                symbols.append(sym)
        out.append({
            "id": str(article.get("article_id") or f"news_{len(out)+1}"),
            "text": " ".join(text.split())[:420],
            "source": str(article.get("source") or "Sourced news"),
            "official": bool(article.get("official")),
            "url": str(article.get("url") or ""),
            "symbols": symbols,
        })
        if len(out) >= max(1, int(limit)):
            break
    return out


def daily_wrap(
    *,
    articles: Sequence[Mapping[str, Any]] | None = None,
    scan_payload: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Here's the wrap of the day — official session first, then sourced headlines."""
    lines = official_session_wrap(scan_payload=scan_payload)
    existing = {str(item.get("text") or "").lower() for item in lines}
    for item in news_wrap_lines(articles, limit=6):
        key = str(item.get("text") or "").lower()
        if key in existing:
            continue
        existing.add(key)
        lines.append(item)
    return lines[:8]


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
            "Official session first. Sourced headlines next. Empty slots stay empty — "
            "QuantTerm does not invent a lawsuit, order, or index print."
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
