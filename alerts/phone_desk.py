"""Optional Telegram Home text — the real phone desk is the browser.

Same evidence as the buy-thesis sheet, truncated for a 4096-char chat bubble.
Never invents prices, flows, or a book. Live orders stay off Telegram.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from product.research_levels import attach_research_levels
from product.stock_workspace import clean_symbol

PHONE_READ_COMMANDS = frozenset({
    "/desk", "/thesis", "/help", "/start", "/status", "/brain",
})


def _esc(value: Any) -> str:
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _f(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        n = float(value)
        return n if n == n else None
    except (TypeError, ValueError):
        return None


def _money(value: Any) -> str:
    n = _f(value)
    if n is None:
        return "—"
    return f"₹{n:,.0f}" if abs(n) >= 100 else f"₹{n:,.2f}"


def allowed_chat_ids(owner: str, extra: str = "") -> tuple[str, set[str]]:
    owner = str(owner or "").strip()
    extras = {part.strip() for part in str(extra or "").split(",") if part.strip()}
    allowed = set(extras)
    if owner:
        allowed.add(owner)
    return owner, allowed


def phone_may_run(chat_id: str, command: str, owner: str, extra: str = "") -> bool:
    """Owner: full paper command set. Extra phone chats: read-only desk/thesis."""
    chat = str(chat_id or "").strip()
    owner_id, allowed = allowed_chat_ids(owner, extra)
    if chat not in allowed:
        return False
    raw = (command or "").strip().split()
    if not raw:
        return False
    cmd = raw[0].lower().split("@")[0]
    if chat == owner_id:
        return True
    return cmd in PHONE_READ_COMMANDS


def _plan_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return attach_research_levels({
        "price": row.get("price") or row.get("entry") or row.get("close"),
        "entry": row.get("entry"),
        "stop": row.get("stop"),
        "target": row.get("target"),
        "atr": row.get("atr"),
        "atr_pct": row.get("atr_pct"),
        "vol_pct": row.get("vol_pct"),
        "upside_from_buy_pct": row.get("upside_from_buy_pct"),
        "levels_source": row.get("levels_source"),
    })


def _is_actionable(row: Mapping[str, Any]) -> bool:
    verdict = str(row.get("verdict") or "").upper()
    if verdict in {"BUY", "STRONG BUY"}:
        return True
    return str(row.get("status") or "") == "Ready to trade"


def _top_desk_rows(limit: int = 5) -> list[dict[str, Any]]:
    try:
        from product.scan_store import load_scan
        records = list((load_scan() or {}).get("records") or [])
    except Exception:
        records = []
    ranked = [dict(row) for row in records if isinstance(row, Mapping) and _is_actionable(row)]
    ranked.sort(key=lambda r: float(r.get("score") or 0), reverse=True)
    return ranked[:limit]


def format_desk(rows: Sequence[Mapping[str, Any]] | None = None) -> str:
    source = [dict(r) for r in rows] if rows is not None else _top_desk_rows()
    names = [row for row in source if _is_actionable(row)][:5]
    if not names:
        return (
            "📱 <b>Desk</b>\n"
            "Abhi koi actionable name nahi. System pe Start here dabao, "
            "ya yahan /status.\n"
            "Thesis: /thesis RELIANCE"
        )
    lines = [
        "📱 <b>Desk — phone se padho</b>",
        "Live order yahan se nahi. Thesis dekho, paper/watch owner chat se.",
        "",
    ]
    for i, row in enumerate(names, 1):
        plan = _plan_row(row)
        sym = _esc(row.get("symbol") or "")
        company = _esc(str(row.get("company") or "")[:28])
        upside = _f(plan.get("upside_from_buy_pct"))
        up = f" · +{upside:.1f}%" if upside is not None else ""
        lines.append(
            f"{i}. <b>{sym}</b> {company}\n"
            f"Buy {_money(plan.get('entry'))} · stop {_money(plan.get('stop'))} · "
            f"tgt {_money(plan.get('target'))}{up}\n"
            f"/thesis {sym}"
        )
    lines.append("\nTap Thesis neeche, ya type /thesis SYMBOL")
    return "\n".join(lines)


def desk_keyboard(rows: Sequence[Mapping[str, Any]] | None = None, *, thesis_only: bool = False) -> dict[str, Any]:
    names = [dict(r) for r in (rows if rows is not None else _top_desk_rows())]
    names = [row for row in names if _is_actionable(row)][:5]
    keyboard: list[list[dict[str, str]]] = []
    for row in names[:5]:
        plan = _plan_row(row)
        sym = str(row.get("symbol") or "").upper()
        if not sym:
            continue
        entry = _f(plan.get("entry")) or 0
        stop = _f(plan.get("stop")) or 0
        target = _f(plan.get("target")) or 0
        line = [{"text": f"📖 {sym}", "callback_data": f"th|{sym}"}]
        if not thesis_only:
            line.append({
                "text": "📝 paper",
                "callback_data": f"pt|{sym}|{entry:.0f}|{stop:.0f}|{target:.0f}",
            })
        keyboard.append(line)
    return {"inline_keyboard": keyboard}


def format_thesis(payload: Mapping[str, Any]) -> str:
    symbol = _esc(payload.get("symbol") or "")
    company = _esc(payload.get("company") or symbol)
    plan = dict(payload.get("plan") or {})
    wave = dict(payload.get("sector_wave") or {})
    money = dict(payload.get("smart_money") or {})
    earnings = dict(payload.get("earnings") or {})
    book = dict(payload.get("order_book") or {})
    lines = [
        f"📖 <b>{company}</b> ({symbol})",
        _esc(payload.get("headline") or "")[:180],
        "",
        f"Buy {_money(plan.get('buy'))} · stop {_money(plan.get('stop'))} · "
        f"tgt {_money(plan.get('target'))}",
    ]
    upside = _f(plan.get("upside_from_buy_pct"))
    if upside is not None:
        lines.append(f"Upside +{upside:.1f}% from buy — research, not an order.")
    lines.append("")
    lines.append("<b>Sector wave</b>")
    lines.append(_esc(wave.get("verdict_line") or wave.get("verdict") or "NO"))
    lines.append(_esc(wave.get("headline") or "Sector not identified."))
    for item in list(wave.get("bullets") or [])[:3]:
        lines.append("· " + _esc(item)[:140])
    lines.append("")
    lines.append("<b>FII / DII / named buyers</b>")
    lines.append(_esc(money.get("headline") or "No stock-level claim yet."))
    for item in list(money.get("bullets") or [])[:4]:
        lines.append("· " + _esc(item)[:160])
    lines.append("")
    lines.append("<b>Earnings / margins / valuations</b>")
    for item in list(earnings.get("bullets") or [])[:6]:
        lines.append("· " + _esc(item)[:140])
    note = str(book.get("note") or "No live depth.")
    src = str(book.get("source") or "")
    lines.append("")
    lines.append(f"<b>Order book</b> — {_esc(note)}" + (f" ({_esc(src)})" if src else ""))
    lines.append("Paper/watch: Home pe card, ya owner chat ke buttons.")
    text = "\n".join(lines)
    return text[:3900]


def load_thesis_text(symbol: str) -> str:
    try:
        clean = clean_symbol(symbol)
    except ValueError:
        return "Symbol samajh nahi aaya. Example: /thesis RELIANCE"
    try:
        from product.buy_thesis import build_buy_thesis
        payload = build_buy_thesis(clean, fetch_missing=False)
    except Exception as exc:
        return f"❌ Thesis load nahi hua ({type(exc).__name__})."
    return format_thesis(payload)
