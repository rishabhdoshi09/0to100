"""Parse screener-style tables into dated series. Missing stays missing."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Mapping, Sequence

_MONTH = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

_QUARTER_RE = re.compile(r"\b(?:q[1-4]|quarter)\b", re.I)
_ANNUAL_RE = re.compile(r"\b(?:fy\s*\d{2,4}|year ended|annual|full[- ]year)\b", re.I)
_TTM_RE = re.compile(r"\b(?:ttm|trailing)\b", re.I)
_YTD_RE = re.compile(r"\b(?:ytd|year[- ]to[- ]date)\b", re.I)
_BASIS_RE = re.compile(r"\b(standalone|consolidated)\b", re.I)
_TABLE_PERIOD = {
    "quarterly_results": "quarterly",
    "profit_loss": "annual",
    "balance_sheet": "annual",
    "cash_flow": "annual",
    "key_ratios": "snapshot",
    "shareholding": "quarterly",
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


def infer_period_type(period: str, table: str = "") -> str:
    """quarterly / annual / ttm / ytd / snapshot / unknown — never guessed from a lone number."""
    blob = f"{period or ''} {table or ''}".strip()
    if _TTM_RE.search(blob):
        return "ttm"
    if _YTD_RE.search(blob):
        return "ytd"
    if _QUARTER_RE.search(blob):
        return "quarterly"
    if _ANNUAL_RE.search(blob):
        return "annual"
    if table in _TABLE_PERIOD:
        return _TABLE_PERIOD[table]
    if parse_period(str(period or "")) is not None and table != "key_ratios":
        # "Jun 2026" on a quarterly table is a quarter-end; without a table it stays unknown.
        return "unknown"
    return "unknown"


def infer_reporting_basis(text: str) -> str:
    match = _BASIS_RE.search(text or "")
    return match.group(1).lower() if match else ""


def periods_comparable(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == "snapshot" or right == "snapshot":
        return False
    if left == "unknown" and right == "unknown":
        return True
    if left == "unknown" or right == "unknown":
        return False
    return left == right


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


def snapshot(
    series: Sequence[tuple[str, float]],
    *,
    kind: str = "level",
    year_steps: int = 4,
    table: str = "",
) -> dict[str, Any]:
    """kind=level → percent change; kind=rate → percentage-point change.

    year_steps is how many points back counts as a year (4 for quarterly, 1 for annual).
    QoQ / YoY are only filled when the compared prints share a period type.
    """
    empty = {
        "current": None, "current_period": "",
        "previous": None, "previous_period": "",
        "year_ago": None, "year_ago_period": "",
        "qoq_change": None, "yoy_change": None,
        "points": [],
        "period_type": infer_period_type("", table) if table else "unknown",
        "previous_period_type": "",
        "year_ago_period_type": "",
        "reporting_basis": "",
    }
    if not series:
        return empty
    current_label, current = series[-1]
    previous_label, previous = series[-2] if len(series) >= 2 else ("", None)
    year_idx = -(int(year_steps) + 1)
    year_label, year_ago = series[year_idx] if len(series) >= (int(year_steps) + 1) else ("", None)
    current_type = infer_period_type(current_label, table)
    previous_type = infer_period_type(previous_label, table) if previous_label else ""
    year_type = infer_period_type(year_label, table) if year_label else ""

    def delta(latest: float | None, base: float | None) -> float | None:
        if latest is None or base is None:
            return None
        if kind == "rate":
            return round(latest - base, 2)
        if base == 0:
            return None
        return round((latest - base) / abs(base) * 100.0, 2)

    qoq = delta(current, previous) if periods_comparable(current_type, previous_type) else None
    yoy = delta(current, year_ago) if periods_comparable(current_type, year_type) else None
    return {
        "current": current,
        "current_period": current_label,
        "previous": previous,
        "previous_period": previous_label,
        "year_ago": year_ago,
        "year_ago_period": year_label,
        "qoq_change": qoq,
        "yoy_change": yoy,
        "points": [{"period": p, "value": v, "period_type": infer_period_type(p, table)} for p, v in series[-8:]],
        "period_type": current_type,
        "previous_period_type": previous_type,
        "year_ago_period_type": year_type,
        "reporting_basis": infer_reporting_basis(f"{current_label} {previous_label} {year_label}"),
    }


def direction(
    *,
    higher_is_better: bool,
    qoq: float | None,
    yoy: float | None,
    current_period_type: str = "",
    compare_period_type: str = "",
) -> str:
    """improving / stable / deteriorating / unknown — from comparable prints only."""
    if current_period_type and compare_period_type and not periods_comparable(current_period_type, compare_period_type):
        return "unknown"
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
