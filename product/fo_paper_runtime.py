"""Durable paper-cycle orchestration for NSE long-premium option candidates."""
from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping

from data.nfo_market import (
    quote_to_option_paper_mark,
    read_nfo_quotes,
    read_option_intraday_bars,
)
from product.fo_paper import FoPaperBook, FoPaperPosition
from product.fo_paper_store import FoPaperStore


def _restore_position(payload: Mapping[str, Any]) -> FoPaperPosition | None:
    allowed = {field.name for field in fields(FoPaperPosition)}
    kwargs = {key: payload[key] for key in allowed if key in payload}
    try:
        return FoPaperPosition(**kwargs)
    except (TypeError, ValueError):
        return None


def _post_entry_intraday_start(opened_at: str, now_ist: datetime) -> datetime | None:
    """First full minute that cannot contain time before the paper entry."""
    try:
        opened = datetime.fromisoformat(str(opened_at))
    except (TypeError, ValueError):
        return None
    if opened.tzinfo is None and now_ist.tzinfo is not None:
        opened = opened.replace(tzinfo=now_ist.tzinfo)
    if opened.tzinfo is not None and now_ist.tzinfo is not None:
        opened = opened.astimezone(now_ist.tzinfo)
    start = opened.replace(second=0, microsecond=0)
    if opened.second or opened.microsecond:
        start += timedelta(minutes=1)
    return start if start < now_ist else None


def _intraday_replay_start(pos: FoPaperPosition, now_ist: datetime) -> datetime | None:
    """Start at post-entry time for new positions, session open for overnight ones."""
    if str(pos.opened_at)[:10] == now_ist.date().isoformat():
        return _post_entry_intraday_start(pos.opened_at, now_ist)
    session_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    return session_open if session_open < now_ist else None


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
        # Closed-trade P&L is durable and must survive process restarts; otherwise
        # every cycle silently sizes from the original capital again.
        book.realized_pnl = float(store.realized_pnl())
        for raw in store.load_positions():
            pos = _restore_position(raw)
            if pos is not None:
                book.open[pos.option_symbol] = pos

        session = now_ist.date().isoformat()
        settled_rows: list[dict[str, Any]] = []
        settled_underlyings: set[str] = set()

        intraday_marks_used = 0
        intraday_marks_fallback = 0
        intraday_bars_replayed = 0
        overnight_intraday_marks_used = 0
        overnight_intraday_marks_fallback = 0
        overnight_intraday_bars_replayed = 0
        if book.open:
            raw_quotes = read_nfo_quotes(list(book.open), client=client)
            marks: dict[str, dict[str, float]] = {}
            settled = []
            for symbol, pos in list(book.open.items()):
                quote = raw_quotes.get(symbol)
                if not isinstance(quote, Mapping):
                    continue
                mark = quote_to_option_paper_mark(quote)
                last = float(mark.get("last_price") or mark.get("close") or 0.0)
                same_day = str(pos.opened_at)[:10] == session
                bars: list[dict[str, Any]] = []
                start_at = _intraday_replay_start(pos, now_ist)
                if int(pos.instrument_token or 0) > 0 and start_at is not None:
                    try:
                        bars = read_option_intraday_bars(
                            int(pos.instrument_token),
                            from_dt=start_at,
                            to_dt=now_ist,
                            client=client,
                            interval="minute",
                        )
                    except Exception:
                        bars = []

                if bars:
                    if same_day:
                        intraday_marks_used += 1
                    else:
                        overnight_intraday_marks_used += 1
                    first_bar = True
                    for bar in bars:
                        if symbol not in book.open:
                            break
                        replay = {
                            "open": float(bar.get("open") or bar.get("close") or 0.0),
                            "high": float(bar.get("high") or bar.get("close") or 0.0),
                            "low": float(bar.get("low") or bar.get("close") or 0.0),
                            "close": float(bar.get("close") or 0.0),
                            "last_price": float(bar.get("close") or 0.0),
                            # Never use today's current bid for a historical
                            # minute exit. STOP/TARGET/GAP fills come from the bar.
                            "bid": 0.0,
                        }
                        if first_bar and same_day:
                            # The partial entry minute is intentionally excluded;
                            # anchor the first complete bar to the paper entry
                            # rather than inventing a same-session gap.
                            replay["open"] = float(pos.entry_price)
                        first_bar = False
                        bar_session = str(bar.get("timestamp") or now_ist.isoformat())
                        bar_settled = book.mark(
                            {symbol: replay},
                            session=bar_session,
                            advance_session=False,
                        )
                        if same_day:
                            intraday_bars_replayed += 1
                        else:
                            overnight_intraday_bars_replayed += 1
                        if bar_settled:
                            settled.extend(bar_settled)
                            break

                    # Replay reconstructed historical path first. Only the live
                    # partial interval advances holding-session age / MAX_HOLD.
                    if symbol in book.open and last > 0:
                        ltp_mark = {
                            "open": last,
                            "high": last,
                            "low": last,
                            "close": last,
                            "last_price": last,
                            "bid": float(mark.get("bid") or 0.0),
                        }
                        settled.extend(
                            book.mark(
                                {symbol: ltp_mark},
                                session=now_ist.isoformat(),
                                advance_session=True,
                            )
                        )
                    continue

                if same_day:
                    # Full-day OHLC contains time before entry, so missing minute
                    # history must fail closed to current LTP only.
                    mark["open"] = last
                    mark["high"] = last
                    mark["low"] = last
                    mark["close"] = last
                    intraday_marks_fallback += 1
                else:
                    # The position existed from session open, so today's full-day
                    # range is valid evidence. Without ordered minute history we
                    # retain the conservative aggregated STOP-first semantics.
                    overnight_intraday_marks_fallback += 1
                marks[symbol] = mark

            if marks:
                settled.extend(book.mark(marks, session=now_ist.isoformat()))
            for trade in settled:
                settled_underlyings.add(trade.underlying)
                row = trade.as_dict()
                row["production_evidence_eligible"] = book.fully_costed
                settled_rows.append(row)

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
                    instrument_token=int(contract.get("instrument_token") or 0),
                    ask=float(contract.get("ask") or 0.0),
                )
                if pos is None:
                    reason = book.refusals[-1][1] if book.refusals else "PAPER_BOOK_REJECTED"
                    skipped.append({"symbol": underlying, "reason": reason})
                    continue
                opened.append(pos.as_dict())

        store.commit_cycle(
            settled_rows,
            (pos.as_dict() for pos in book.open.values()),
        )
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
            "realized_pnl": round(book.realized_pnl, 2),
            "equity_for_sizing": round(max(0.0, book.capital + book.realized_pnl), 2),
            "production_evidence_enabled": book.fully_costed,
            "same_day_intraday_marks_used": intraday_marks_used,
            "same_day_intraday_marks_fallback": intraday_marks_fallback,
            "same_day_intraday_bars_replayed": intraday_bars_replayed,
            "overnight_intraday_marks_used": overnight_intraday_marks_used,
            "overnight_intraday_marks_fallback": overnight_intraday_marks_fallback,
            "overnight_intraday_bars_replayed": overnight_intraday_bars_replayed,
            "evidence_cost_status": (
                f"CONFIGURED:{cost_model_name}" if book.fully_costed
                else "UNCONFIGURED_GROSS_ONLY"
            ),
        }
    finally:
        if owned_store:
            store.close()
