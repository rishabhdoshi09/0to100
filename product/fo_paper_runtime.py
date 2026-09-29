"""Durable paper-cycle orchestration for NSE long-premium option candidates."""
from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping

from data.nfo_market import (
    LIVE_QUOTE_MAX_SKEW_SECONDS,
    option_quote_implied_iv_pct,
    quote_provenance,
    quote_timestamp_skew_seconds,
    quote_to_option_paper_mark,
    read_market_quotes,
    read_nfo_quotes,
    read_option_intraday_bars,
)
from product.fo_paper import (
    ENTRY_MINUTE_AMBIGUOUS,
    ENTRY_MINUTE_CLEAR,
    ENTRY_MINUTE_EVIDENCE_OK,
    ENTRY_MINUTE_FULLY_OBSERVED,
    ENTRY_MINUTE_PENDING,
    ENTRY_MINUTE_UNAVAILABLE,
    FoPaperBook,
    FoPaperPosition,
)
from product.fo_paper_store import FoPaperStore


def _restore_position(payload: Mapping[str, Any]) -> FoPaperPosition | None:
    allowed = {field.name for field in fields(FoPaperPosition)}
    kwargs = {key: payload[key] for key in allowed if key in payload}
    try:
        return FoPaperPosition(**kwargs)
    except (TypeError, ValueError):
        return None


def _opened_at_datetime(opened_at: str, now_ist: datetime) -> datetime | None:
    try:
        opened = datetime.fromisoformat(str(opened_at))
    except (TypeError, ValueError):
        return None
    if opened.tzinfo is None and now_ist.tzinfo is not None:
        opened = opened.replace(tzinfo=now_ist.tzinfo)
    if opened.tzinfo is not None and now_ist.tzinfo is not None:
        opened = opened.astimezone(now_ist.tzinfo)
    return opened


def _post_entry_intraday_start(opened_at: str, now_ist: datetime) -> datetime | None:
    """First full minute that cannot contain time before the paper entry."""
    opened = _opened_at_datetime(opened_at, now_ist)
    if opened is None:
        return None
    start = opened.replace(second=0, microsecond=0)
    if opened.second or opened.microsecond:
        start += timedelta(minutes=1)
    return start if start < now_ist else None


def _audit_entry_minute(pos: FoPaperPosition, *, client, now_ist: datetime) -> str:
    """Prove whether the excluded partial entry minute could hide an exit.

    We never replay a partial minute because its extrema include time before the
    paper entry. Instead, its full range is used only as a proof test:
    - no stop/target touch anywhere => the post-entry slice also could not touch;
    - any boundary touch => timing is unknowable, so the outcome stays research-only;
    - an exact minute-boundary entry is fully observable and may be replayed.
    """
    current = str(pos.entry_minute_status or ENTRY_MINUTE_PENDING).upper()
    if current in ENTRY_MINUTE_EVIDENCE_OK or current == ENTRY_MINUTE_AMBIGUOUS:
        return current

    opened = _opened_at_datetime(pos.opened_at, now_ist)
    token = int(pos.instrument_token or 0)
    if opened is None or token <= 0:
        return ENTRY_MINUTE_UNAVAILABLE

    minute_start = opened.replace(second=0, microsecond=0)
    minute_end = minute_start + timedelta(minutes=1)
    if now_ist < minute_end:
        return ENTRY_MINUTE_PENDING

    try:
        bars = read_option_intraday_bars(
            token,
            from_dt=minute_start,
            to_dt=minute_end - timedelta(seconds=1),
            client=client,
            interval="minute",
        )
    except Exception:
        return ENTRY_MINUTE_UNAVAILABLE
    if not bars:
        return ENTRY_MINUTE_UNAVAILABLE

    # If the paper fill occurred exactly on the minute boundary, this entire
    # minute is post-entry and chronological replay can observe it directly.
    if opened.second == 0 and opened.microsecond == 0:
        return ENTRY_MINUTE_FULLY_OBSERVED

    bar = bars[0]
    close = float(bar.get("close") or 0.0)
    high = float(bar.get("high") or close)
    low = float(bar.get("low") or close)
    if close <= 0 or high <= 0 or low <= 0:
        return ENTRY_MINUTE_UNAVAILABLE
    if low <= float(pos.stop_price) or high >= float(pos.target_price):
        return ENTRY_MINUTE_AMBIGUOUS
    return ENTRY_MINUTE_CLEAR


def _intraday_replay_start(pos: FoPaperPosition, now_ist: datetime) -> datetime | None:
    """Start at post-entry time for new positions, session open for overnight ones."""
    if str(pos.opened_at)[:10] == now_ist.date().isoformat():
        return _post_entry_intraday_start(pos.opened_at, now_ist)
    session_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    return session_open if session_open < now_ist else None


FNO_EOD_EXIT_HOUR = 15
FNO_EOD_EXIT_MINUTE = 35
IV_CRUSH_RATIO = 0.75


def _eod_exit_due(now_ist: datetime) -> bool:
    return (now_ist.hour, now_ist.minute) >= (FNO_EOD_EXIT_HOUR, FNO_EOD_EXIT_MINUTE)


def _iv_crush_state(
    *,
    pos: FoPaperPosition,
    option_quote: Mapping[str, Any] | None,
    underlying_quote: Mapping[str, Any] | None,
    now_ist: datetime,
) -> tuple[bool, str, float | None]:
    """Current-quote IV-crush guard; never infers from stale or incoherent data."""
    if (
        pos.entry_iv_pct <= 0
        or pos.strike <= 0
        or not pos.expiry
        or not isinstance(option_quote, Mapping)
        or not isinstance(underlying_quote, Mapping)
    ):
        return False, "IV_CRUSH_PROVENANCE_UNAVAILABLE", None
    as_of = now_ist.date()
    option_state = quote_provenance(option_quote, as_of=as_of, now=now_ist)
    underlying_state = quote_provenance(underlying_quote, as_of=as_of, now=now_ist)
    if not option_state.get("ok") or not underlying_state.get("ok"):
        return False, "IV_CRUSH_QUOTE_UNTRUSTED", None
    skew = quote_timestamp_skew_seconds(option_quote, underlying_quote)
    if skew is None or skew > LIVE_QUOTE_MAX_SKEW_SECONDS:
        return False, "IV_CRUSH_QUOTE_SKEW_TOO_WIDE", None
    spot = float(underlying_quote.get("last_price") or 0.0)
    if spot <= 0:
        return False, "IV_CRUSH_SPOT_UNAVAILABLE", None
    current_iv = option_quote_implied_iv_pct(
        option_quote,
        spot=spot,
        strike=pos.strike,
        expiry=pos.expiry,
        as_of=as_of,
        option_type=pos.option_type,
    )
    if current_iv <= 0:
        return False, "IV_CRUSH_CURRENT_IV_UNAVAILABLE", None
    mark = quote_to_option_paper_mark(option_quote)
    last = float(mark.get("last_price") or mark.get("close") or 0.0)
    triggered = (
        current_iv <= pos.entry_iv_pct * IV_CRUSH_RATIO
        and last > 0
        and last < pos.entry_price
    )
    return triggered, "IV_CRUSH_TRIGGERED" if triggered else "IV_CRUSH_CLEAR", current_iv


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
        session = now_ist.date().isoformat()
        book = FoPaperBook(
            capital=capital,
            risk_per_trade_pct=0.01,
            max_premium_pct=0.10,
            max_daily_premium_pct=0.10,
            premium_deployed_today=store.premium_deployed_on_session(session),
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

        settled_rows: list[dict[str, Any]] = []
        settled_underlyings: set[str] = set()

        intraday_marks_used = 0
        intraday_marks_fallback = 0
        intraday_bars_replayed = 0
        overnight_intraday_marks_used = 0
        overnight_intraday_marks_fallback = 0
        overnight_intraday_bars_replayed = 0
        entry_minute_clear = 0
        entry_minute_ambiguous = 0
        entry_minute_unavailable = 0
        entry_minute_pending = 0
        partial_exit_interval_holdouts = 0
        iv_crush_evaluated = 0
        iv_crush_triggered = 0
        iv_crush_unavailable = 0
        eod_exit_due = _eod_exit_due(now_ist)
        if book.open:
            raw_quotes = read_nfo_quotes(list(book.open), client=client)
            underlying_quotes = read_market_quotes(
                sorted({f"NSE:{pos.underlying}" for pos in book.open.values()}),
                client=client,
            )
            iv_crush_symbols: set[str] = set()
            current_iv_by_symbol: dict[str, float] = {}
            for symbol, pos in list(book.open.items()):
                triggered, iv_reason, current_iv = _iv_crush_state(
                    pos=pos,
                    option_quote=raw_quotes.get(symbol),
                    underlying_quote=underlying_quotes.get(f"NSE:{pos.underlying}"),
                    now_ist=now_ist,
                )
                if current_iv is None:
                    iv_crush_unavailable += 1
                else:
                    iv_crush_evaluated += 1
                    current_iv_by_symbol[symbol] = current_iv
                if triggered:
                    iv_crush_triggered += 1
                    iv_crush_symbols.add(symbol)
            marks: dict[str, dict[str, float]] = {}
            settled = []
            for symbol, pos in list(book.open.items()):
                pos.entry_minute_status = _audit_entry_minute(
                    pos,
                    client=client,
                    now_ist=now_ist,
                )
                if pos.entry_minute_status in ENTRY_MINUTE_EVIDENCE_OK:
                    entry_minute_clear += 1
                elif pos.entry_minute_status == ENTRY_MINUTE_AMBIGUOUS:
                    entry_minute_ambiguous += 1
                elif pos.entry_minute_status == ENTRY_MINUTE_UNAVAILABLE:
                    entry_minute_unavailable += 1
                else:
                    entry_minute_pending += 1

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
                        # Same-day replay begins at the first complete minute
                        # that cannot contain pre-entry time. Its observed open is
                        # therefore genuine post-entry evidence; preserve it so
                        # gap-through-stop/target fills are not fabricated away.
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
                                observation_complete=False,
                                force_eod=eod_exit_due,
                                iv_crush_symbols=iv_crush_symbols,
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
                settled.extend(
                    book.mark(
                        marks,
                        session=now_ist.isoformat(),
                        observation_complete=False,
                        force_eod=eod_exit_due,
                        iv_crush_symbols=iv_crush_symbols,
                    )
                )
            for trade in settled:
                settled_underlyings.add(trade.underlying)
                row = trade.as_dict()
                row["production_evidence_eligible"] = bool(
                    book.fully_costed and trade.path_observation_complete
                )
                if not book.fully_costed:
                    row["evidence_exclusion_reason"] = "COST_MODEL_UNCONFIGURED"
                elif not trade.path_observation_complete:
                    row["evidence_exclusion_reason"] = (
                        trade.path_observation_reason
                        or f"ENTRY_MINUTE_{trade.entry_minute_status or ENTRY_MINUTE_PENDING}"
                    )
                if not trade.exit_observation_complete:
                    partial_exit_interval_holdouts += 1
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
                    horizon=str(expected.get("horizon") or ""),
                    exit_policy=str(expected.get("exit_policy") or "SESSION_HOLD"),
                    strike=float(contract.get("strike") or 0.0),
                    expiry=str(contract.get("expiry") or ""),
                    entry_iv_pct=float(contract.get("iv") or 0.0),
                    entry_underlying_spot=float(
                        (setup.get("underlying_trade_plan") or {}).get("entry") or 0.0
                    ),
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
            "daily_premium_cap_pct": round(book.max_daily_premium_pct * 100.0, 2),
            "premium_deployed_today": round(book.premium_deployed_today, 2),
            "daily_premium_budget": round(
                max(0.0, book.capital + book.realized_pnl) * book.max_daily_premium_pct,
                2,
            ),
            "daily_premium_remaining": round(
                max(
                    0.0,
                    max(0.0, book.capital + book.realized_pnl) * book.max_daily_premium_pct
                    - book.premium_deployed_today,
                ),
                2,
            ),
            "production_evidence_enabled": book.fully_costed,
            "same_day_intraday_marks_used": intraday_marks_used,
            "same_day_intraday_marks_fallback": intraday_marks_fallback,
            "same_day_intraday_bars_replayed": intraday_bars_replayed,
            "overnight_intraday_marks_used": overnight_intraday_marks_used,
            "overnight_intraday_marks_fallback": overnight_intraday_marks_fallback,
            "overnight_intraday_bars_replayed": overnight_intraday_bars_replayed,
            "entry_minute_clear_count": entry_minute_clear,
            "entry_minute_ambiguous_count": entry_minute_ambiguous,
            "entry_minute_unavailable_count": entry_minute_unavailable,
            "entry_minute_pending_count": entry_minute_pending,
            "partial_exit_interval_holdout_count": partial_exit_interval_holdouts,
            "iv_crush_evaluated_count": iv_crush_evaluated,
            "iv_crush_triggered_count": iv_crush_triggered,
            "iv_crush_unavailable_count": iv_crush_unavailable,
            "eod_exit_due": eod_exit_due,
            "eod_exit_cutoff_ist": f"{FNO_EOD_EXIT_HOUR:02d}:{FNO_EOD_EXIT_MINUTE:02d}",
            "evidence_cost_status": (
                f"CONFIGURED:{cost_model_name}" if book.fully_costed
                else "UNCONFIGURED_GROSS_ONLY"
            ),
        }
    finally:
        if owned_store:
            store.close()
