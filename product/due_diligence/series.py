"""Parse screener-style tables into dated series. Missing stays missing."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

_MONTH = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _f(value: Any) -> float | None:
    if value in (None, "", "-", "—"):
        return None
    try:
        text = str(value).replace(",", "").replace("%", "").replace("₹", "").strip()
        if not text or text.lower() in {"na", "n/a", "none"}:
            return None
        number = float(text)
        return number if number == number else None
    except (TypeError, ValueError):
        return None


def normalize_label(value: Any) -> str:
    return " ".join(str(value or "").lower().replace("+", " ").replace("%", " ").split())


def parse_period(key: str) -> datetime | None:
    text = str(key or "").strip()
    if not text or text.lower() in {"row_label", "particulars", "particular", ""}:
        return None
    parts = text.replace("-", " ").split()
    if len(parts) >= 2 and parts[0][:3].upper() in _MONTH:
        try:
            year = int(parts[1][:4])
            return datetime(_MONTH[parts[0][:3].upper()], 1, 1).replace(year=year, month=_MONTH[parts[0][:3].upper()])
        except (TypeError, ValueError):
            return None
    try:
        return datetime.fromisoformat(text[:10])
    except ValueError:
        return None


def row_label(row: Mapping[str, Any]) -> str:
    for key in ("row_label", "Particulars", "particulars", "Particular", "name", "Name", ""):
        if row.get(key) not in (None, ""):
            return str(row.get(key))
    return ""


def find_row(rows: Sequence[Mapping[str, Any]] | None, needles: Sequence[str]) -> dict[str, Any] | None:
    want = [normalize_label(n) for n in needles if n]
    candidates: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        label = normalize_label(row_label(row))
        if not label or label == "raw pdf":
            continue
        candidates.append(dict(row))
    for row in candidates:
        label = normalize_label(row_label(row))
        if label in want:
            return row
    for row in candidates:
        label = normalize_label(row_label(row))
        # Needle in label ("opm" in "opm", "gross npa" in "gross npa"). Never the reverse:
        # "operating profit" must not match the needle "operating profit margin".
        if any(n and n in label for n in want):
            return row
    return None


def dated_series(row: Mapping[str, Any] | None) -> list[tuple[str, float]]:
    if not row:
        return []
    items: list[tuple[datetime, str, float]] = []
    for key, value in row.items():
        if str(key) in {"row_label", "Particulars", "particulars", "Particular", ""}:
            continue
        stamp = parse_period(str(key))
        number = _f(value)
        if stamp is None or number is None:
            continue
        items.append((stamp, str(key), number))
    items.sort(key=lambda item: item[0])
    return [(label, number) for _stamp, label, number in items]


def snapshot(series: Sequence[tuple[str, float]], *, kind: str = "level") -> dict[str, Any]:
    """kind=level → percent change; kind=rate → percentage-point change."""
    if not series:
        return {
            "current": None, "current_period": "",
            "previous": None, "previous_period": "",
            "year_ago": None, "year_ago_period": "",
            "qoq_change": None, "yoy_change": None,
            "points": [],
        }
    current_label, current = series[-1]
    previous_label, previous = series[-2] if len(series) >= 2 else ("", None)
    year_label, year_ago = series[-5] if len(series) >= 5 else ("", None)

    def delta(latest: float | None, base: float | None) -> float | None:
        if latest is None or base is None:
            return None
        if kind == "rate":
            return round(latest - base, 2)
        if base == 0:
            return None
        return round((latest - base) / abs(base) * 100.0, 2)

    return {
        "current": current,
        "current_period": current_label,
        "previous": previous,
        "previous_period": previous_label,
        "year_ago": year_ago,
        "year_ago_period": year_label,
        "qoq_change": delta(current, previous),
        "yoy_change": delta(current, year_ago),
        "points": [{"period": p, "value": v} for p, v in series[-8:]],
    }


def direction(*, higher_is_better: bool, qoq: float | None, yoy: float | None) -> str:
    """improving / stable / deteriorating / unknown — from measured changes only."""
    moves = [x for x in (yoy, qoq) if x is not None]
    if not moves:
        return "unknown"
    signed = moves[0] if yoy is None else yoy
    if not higher_is_better:
        signed = -signed
    if signed > 0.25:
        return "improving"
    if signed < -0.25:
        return "deteriorating"
    return "stable"
