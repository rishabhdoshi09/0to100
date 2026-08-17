"""Filings-period honesty — cache-fresh is not the same as current results.

Screener's consolidated page can return HTTP 200 with a table that stopped
years ago (EIMCOELECO: last quarter Mar 2024 while standalone has Jun 2026).
These helpers parse `Mar 2024` / `Jun 2026` columns, skip TTM, and say when
the latest disclosed period is behind the current NSE reporting season.

Never invents a later quarter. Callers either fetch a newer pack or label
the old column stale.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Mapping, Sequence

_SKIP_KEYS = {"", "row_label", "Particulars", "PARTICULARS", "particulars"}
_Q_MONTH = {3: 0, 6: 1, 9: 2, 12: 3}


def _as_of_date(as_of: date | datetime | None = None) -> date:
    if isinstance(as_of, datetime):
        return as_of.date()
    if isinstance(as_of, date):
        return as_of
    try:
        from core.market_clock import today_ist
        return today_ist()
    except Exception:
        return date.today()


def parse_period_label(label: Any) -> date | None:
    text = str(label or "").strip()
    if not text or text.upper() == "TTM":
        return None
    for fmt in ("%b %Y", "%B %Y", "%b-%Y", "%Y"):
        try:
            parsed = datetime.strptime(text, fmt)
            month = 3 if fmt == "%Y" else parsed.month
            return date(parsed.year, month, 1)
        except ValueError:
            continue
    return None


def expected_latest_quarter(as_of: date | datetime | None = None) -> date:
    """Quarter-end we should already have on file by `as_of` (IST).

    Indian listed results typically land ~45 days after quarter end.
    After 16 Aug we expect the June quarter of that year.
    """
    today = _as_of_date(as_of)
    y, m, d = today.year, today.month, today.day
    md = (m, d)
    if md >= (11, 16):
        return date(y, 9, 1)
    if md >= (8, 16):
        return date(y, 6, 1)
    if md >= (5, 16):
        return date(y, 3, 1)
    if md >= (2, 16):
        return date(y - 1, 12, 1)
    return date(y - 1, 9, 1)


def quarter_index(stamp: date) -> int:
    month = stamp.month if stamp.month in _Q_MONTH else ((stamp.month - 1) // 3 + 1) * 3
    if month > 12:
        month = 12
    return stamp.year * 4 + _Q_MONTH.get(month, 0)


def quarters_behind(latest: date | None, as_of: date | datetime | None = None) -> int | None:
    if latest is None:
        return None
    expected = expected_latest_quarter(as_of)
    return max(0, quarter_index(expected) - quarter_index(latest))


def normalize_period_points(points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Sort by parsed period, drop TTM. Unparsed labels keep original order at the end."""
    dated: list[tuple[date, dict[str, Any]]] = []
    other: list[dict[str, Any]] = []
    for raw in points or []:
        if not isinstance(raw, Mapping):
            continue
        item = dict(raw)
        period = str(item.get("period") or "")
        if period.strip().upper() == "TTM":
            continue
        stamp = parse_period_label(period)
        if stamp is None:
            other.append(item)
        else:
            dated.append((stamp, item))
    dated.sort(key=lambda pair: pair[0])
    return [item for _, item in dated] + other


def _iter_table_periods(rows: Sequence[Mapping[str, Any]] | None):
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        for key in row.keys():
            if str(key) in _SKIP_KEYS:
                continue
            stamp = parse_period_label(key)
            if stamp is not None:
                yield stamp, str(key)


def pack_tables(pack: Mapping[str, Any] | None) -> Mapping[str, Any]:
    payload = dict(pack or {})
    nested = payload.get("data")
    if isinstance(nested, Mapping) and (
        nested.get("quarterly_results") or nested.get("profit_loss")
    ):
        return nested
    return payload


def pack_latest_period(
    pack: Mapping[str, Any] | None,
) -> tuple[date | None, str]:
    tables = pack_tables(pack)
    dated = list(_iter_table_periods(tables.get("quarterly_results")))
    if not dated:
        dated = list(_iter_table_periods(tables.get("profit_loss")))
    if not dated:
        return None, ""
    stamp, label = max(dated, key=lambda pair: pair[0])
    return stamp, label


def pack_filings_stale(
    pack: Mapping[str, Any] | None,
    *,
    as_of: date | datetime | None = None,
    min_quarters_behind: int = 2,
) -> bool:
    """True when the pack has a dated table that is two or more quarters late."""
    latest, _ = pack_latest_period(pack)
    behind = quarters_behind(latest, as_of)
    return behind is not None and behind >= min_quarters_behind


def pack_needs_filings_retry(
    pack: Mapping[str, Any] | None,
    *,
    as_of: date | datetime | None = None,
) -> bool:
    """Today's cache can still be a frozen Screener table — retry standalone once."""
    if not isinstance(pack, Mapping):
        return False
    if pack.get("_filings_refresh_attempted"):
        return False
    latest, _ = pack_latest_period(pack)
    if latest is None:
        return False
    behind = quarters_behind(latest, as_of)
    return behind is not None and behind >= 1


def prefer_fresher_pack(
    primary: Mapping[str, Any] | None,
    secondary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Keep the pack whose latest dated column is newer. Never invents periods."""
    left = dict(primary or {})
    right = dict(secondary or {})
    left_dt, _ = pack_latest_period(left)
    right_dt, _ = pack_latest_period(right)
    if right_dt is None:
        return left
    if left_dt is None or right_dt > left_dt:
        return right
    return left
