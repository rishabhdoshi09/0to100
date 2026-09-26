"""Durable paper-cycle orchestration for NSE long-premium option candidates."""
from __future__ import annotations

from dataclasses import fields
from datetime import datetime
from typing import Any, Callable, Mapping

from data.nfo_market import quote_to_option_paper_mark, read_nfo_quotes
from product.fo_paper import FoPaperBook, FoPaperPosition
from product.fo_paper_store import FoPaperStore


def _restore_position(payload: Mapping[str, Any]) -> FoPaperPosition | None:
    allowed = {field.name for field in fields(FoPaperPosition)}
    kwargs = {key: payload[key] for key in allowed if key in payload}
    try:
        return FoPaperPosition(**kwargs)
    except (TypeError, ValueError):
        return None


def _candidate_rows(directional: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in list(directional.get("candidates") or [])
        if isinstance(row, Mapping)
    ]
    rows.sort(
        key=lambda row: (
            float((row.get("setup") or {}).get("score") or 0.0),
            float((row.get("selected_contract") or {}).get("score") or 0.0),
        ),
        reverse=True,
    )
    return rows


def run_fo_paper_cycle(
    directional: Mapping[str, Any],
    *,
    client,
    now_ist: datetime,
    allow_new_entries: bool,
    store: FoPaperStore | None = None,
    capital: float = 100_000.0,
    cost_model: Callable[[float, float, int], float] | None = None,
    cost_model_name: str = "",
) -> dict[str, Any]:
    """Settle existing paper options, then open fresh eligible paper candidates."""
    owned_store = store is None
    store = store or FoPaperStore()
    try:
        book = FoPaperBook(
            capital=capital,
            risk_per_trade_pct=0.01,
            max_premium_pct=0.10,
            max_positions=5,
            max_total_risk_pct=0.05,
            slippage_bps=5.0,
            cost_model=cost_model,
            cost_model_name=cost_model_name,
        )
        for raw in store.load_positions():
            pos = _restore_position(raw)
            if pos is not None:
                book.open[pos.option_symbol] = pos

        session = now_ist.date().isoformat()
        settled_rows: list[dict[str, Any]] = []
        settled_underlyings: set[str] = set()

        if book.open:
            raw_quotes = read_nfo_quotes(list(book.open), client=client)
            marks: dict[str, dict[str, float]] = {}
            for symbol, pos in book.open.items():
                quote = raw_quotes.get(symbol)
                if not isinstance(quote, Mapping):
                    continue
                mark = quote_to_option_paper_mark(quote)
                # Daily quote high/low include time before a same-day entry. Using
                # that range would create impossible post-entry stop/target hits.
                if str(pos.opened_at)[:10] == session:
                    last = float(mark.get("last_price") or mark.get("close") or 0.0)
                    mark["open"] = last
                    mark["high"] = last
                    mark["low"] = last
                    mark["close"] = last
                marks[symbol] = mark
            settled = book.mark(marks, session=session)
            for trade in settled:
                settled_underlyings.add(trade.underlying)
                row = trade.as_dict()
                row["production_evidence_eligible"] = book.fully_costed
                settled_rows.append(row)
            store.append_trades(settled_rows)

        opened: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        if allow_new_entries and bool(directional.get("available")):
            for candidate in _candidate_rows(directional):
                underlying = str(candidate.get("symbol") or "").upper()
                if not underlying:
                    skipped.append({"symbol": "", "reason": "MISSING_UNDERLYING"})
                    continue
                if underlying in settled_underlyings:
                    skipped.append({"symbol": underlying, "reason": "SETTLED_THIS_CYCLE"})
                    continue
                setup = candidate.get("setup") if isinstance(candidate.get("setup"), Mapping) else {}
                contract = (
                    candidate.get("selected_contract")
                    if isinstance(candidate.get("selected_contract"), Mapping)
                    else {}
                )
                plan = contract.get("trade_plan") if isinstance(contract.get("trade_plan"), Mapping) else {}
                option_symbol = str(contract.get("symbol") or "")
                entry = float(plan.get("entry") or 0.0)
                stop = float(plan.get("stop") or 0.0)
                target = float(plan.get("target") or 0.0)
                lot_size = int(contract.get("lot_size") or 0)
                if (
                    candidate.get("decision") != "PAPER_OPTION_CANDIDATE"
                    or not option_symbol
                    or not bool(contract.get("eligible"))
                    or entry <= 0
                    or stop <= 0
                    or target <= entry
                    or lot_size <= 0
                ):
                    skipped.append({"symbol": underlying, "reason": "CANDIDATE_NOT_EXECUTABLE"})
                    continue
                expected = setup.get("expected_move") if isinstance(setup.get("expected_move"), Mapping) else {}
                pos = book.open_position(
                    underlying=underlying,
                    option_symbol=option_symbol,
                    option_type=str(contract.get("option_type") or ""),
                    entry=entry,
                    stop=stop,
                    target=target,
                    lot_size=lot_size,
                    opened_at=now_ist.isoformat(),
                    max_holding_sessions=max(1, int(expected.get("holding_days") or 1)),
                    context_key=str(contract.get("context_key") or ""),
                    setup_score=float(setup.get("score") or 0.0),
                    option_score=float(contract.get("score") or 0.0),
                    ask=float(contract.get("ask") or 0.0),
                )
                if pos is None:
                    reason = book.refusals[-1][1] if book.refusals else "PAPER_BOOK_REJECTED"
                    skipped.append({"symbol": underlying, "reason": reason})
                    continue
                opened.append(pos.as_dict())

        store.replace_positions(pos.as_dict() for pos in book.open.values())
        status = store.status()
        return {
            "available": True,
            "paper_only": True,
            "live_execution_allowed": False,
            "session": session,
            "new_entries_allowed": bool(allow_new_entries),
            "opened_count": len(opened),
            "settled_count": len(settled_rows),
            "open_count": len(book.open),
            "opened": opened,
            "settled": settled_rows,
            "skipped": skipped,
            "open_positions": [pos.as_dict() for pos in book.open.values()],
            "store": status,
            "production_evidence_enabled": book.fully_costed,
            "evidence_cost_status": (
                f"CONFIGURED:{cost_model_name}" if book.fully_costed
                else "UNCONFIGURED_GROSS_ONLY"
            ),
        }
    finally:
        if owned_store:
            store.close()
