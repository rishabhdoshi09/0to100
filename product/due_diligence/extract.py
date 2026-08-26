"""Rule-based extraction from tables, filings and commentary.

This is the 'act like an LLM without one' layer: regex and table matching
only. Missing stays missing. Never calls a language model.
"""
from __future__ import annotations

import html
import re
from typing import Any, Mapping, Sequence

from product.due_diligence.series import _f, dated_series, find_row, normalize_label, snapshot

_GNPA_NEEDLES = ("gross npa", "gnpa", "gross non performing")
_NNPA_NEEDLES = ("net npa", "nnpa", "net non performing")
_PLEDGE_NEEDLES = ("pledge", "encumbrance")

_GNPA_LABEL = re.compile(r"(?:gross\s*npa[s]?|gnpa|gross\s*non[\s-]*performing)", re.I)
_NNPA_LABEL = re.compile(r"(?:net\s*npa[s]?|nnpa|net\s*non[\s-]*performing)", re.I)
_PLEDGE_LABEL = re.compile(r"(?:promoter\s*)?pledged?(?:\s+shares?|\s+equity|\s+holding)?", re.I)
_PERCENT_RE = re.compile(r"(\d{1,2}(?:\.\s*\d+)?)\s*%")
_SKIP_PREFIX = ("below", "under", "above", "over", "upto", "up to", "within")
_GUIDANCE_LINE = re.compile(
    r".{0,100}(?:guidance|outlook|we expect|we guide|we maintain|order book|"
    r"order[- ]book|raise(?:d)? guidance|cut guidance|lower(?:ed)? guidance|"
    r"headwind|tailwind|under pressure).{0,160}",
    re.I,
)
_CONSTRUCTIVE = (
    "raise", "raised", "strong demand", "confident", "ahead of",
    "better than", "robust", "tailwind", "increase guidance", "upgrade",
)
_CAUTIOUS = (
    "headwind", "pressure", "delay", "soft", "cautious", "uncertain",
    "muted", "challenging", "moderat",
)
_NEGATIVE = (
    "warning letter", "investigation", "cut guidance", "lower guidance",
    "withdraw", "usfda", "import alert", "downgrade", "default",
)

_TEXT_EXT = {".txt", ".html", ".htm", ".csv", ".json", ".vtt", ".srt", ".xml"}


def html_to_text(blob: str) -> str:
    text = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", blob or "")
    text = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _rate(value: Any) -> float | None:
    number = _f(value)
    if number is None or number < 0 or number > 100:
        return None
    return number


def _last_percent(label: re.Pattern[str], text: str) -> float | None:
    found: list[float] = []
    blob = text or ""
    for match in label.finditer(blob):
        window = blob[match.end(): match.end() + 96]
        for item in _PERCENT_RE.finditer(window):
            prefix = window[max(0, item.start() - 12): item.start()].lower()
            if any(tok in prefix for tok in _SKIP_PREFIX):
                continue
            number = _rate(item.group(1).replace(" ", ""))
            if number is not None:
                found.append(number)
                break
    return found[-1] if found else None


def key_ratio_value(rows: Sequence[Mapping[str, Any]] | None, needles: Sequence[str]) -> float | None:
    want = [normalize_label(n) for n in needles if n]
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        name = normalize_label(row.get("name") or row.get("row_label") or "")
        if not name:
            continue
        if name in want or any(n and n in name for n in want):
            return _rate(row.get("value") or row.get("current"))
    return None


def series_from_tables(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    needles: Sequence[str],
    *,
    kind: str = "rate",
    prefer: str = "",
) -> dict[str, Any] | None:
    order = []
    if prefer:
        order.append(prefer)
    order.extend(k for k in tables if k != prefer)
    for key in order:
        row = find_row(tables.get(key), needles)
        series = dated_series(row)
        if series:
            snap = snapshot(series, kind=kind)
            if snap.get("current") is not None:
                return {**snap, "table": key, "source": "results table on file"}
    return None


def extract_kpis_from_raw(raw: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Fill GNPA / NNPA / pledge from any table or key-ratio snapshot in the cache."""
    raw = dict(raw or {})
    tables = {
        "quarterly_results": list(raw.get("quarterly_results") or []),
        "profit_loss": list(raw.get("profit_loss") or []),
        "cash_flow": list(raw.get("cash_flow") or []),
        "shareholding": list(raw.get("shareholding") or []),
        "balance_sheet": list(raw.get("balance_sheet") or []),
        "key_ratios": list(raw.get("key_ratios") or []),
    }
    out: dict[str, dict[str, Any]] = {}
    specs = (
        ("gnpa", _GNPA_NEEDLES, "quarterly_results"),
        ("nnpa", _NNPA_NEEDLES, "quarterly_results"),
        ("pledge", _PLEDGE_NEEDLES, "shareholding"),
    )
    for kpi_id, needles, prefer in specs:
        snap = series_from_tables(tables, needles, kind="rate", prefer=prefer)
        if snap:
            out[kpi_id] = {**snap, "source": "Screener.in cache / company results table"}
            continue
        current = key_ratio_value(tables["key_ratios"], needles)
        if current is not None:
            out[kpi_id] = {
                "current": current,
                "current_period": "key-ratio snapshot",
                "previous": None,
                "previous_period": "",
                "year_ago": None,
                "year_ago_period": "",
                "qoq_change": None,
                "yoy_change": None,
                "points": [{"period": "key-ratio snapshot", "value": current}],
                "source": "Screener.in key-ratio snapshot",
            }
    return out


def extract_rates_from_text(text: str, *, source: str, source_url: str = "") -> dict[str, dict[str, Any]]:
    blob = html_to_text(text) if "<" in (text or "") and ">" in (text or "") else (text or "")
    out: dict[str, dict[str, Any]] = {}
    mapping = (("gnpa", _GNPA_LABEL), ("nnpa", _NNPA_LABEL), ("pledge", _PLEDGE_LABEL))
    for kpi_id, pattern in mapping:
        current = _last_percent(pattern, blob)
        if current is None:
            continue
        out[kpi_id] = {
            "current": current,
            "current_period": "extracted print",
            "previous": None,
            "previous_period": "",
            "year_ago": None,
            "year_ago_period": "",
            "qoq_change": None,
            "yoy_change": None,
            "points": [{"period": "extracted print", "value": current}],
            "source": source,
            "source_url": source_url,
        }
    return out


def guidance_tone(text: str) -> str:
    blob = (text or "").lower()
    if not blob.strip():
        return "Unmeasured"
    if any(tok in blob for tok in _NEGATIVE):
        return "Negative"
    if any(tok in blob for tok in _CAUTIOUS) and not any(tok in blob for tok in _CONSTRUCTIVE):
        return "Cautious"
    if any(tok in blob for tok in _CONSTRUCTIVE):
        return "Constructive"
    if _GUIDANCE_LINE.search(blob):
        return "Neutral"
    return "Unmeasured"


def extract_guidance(text: str, *, source: str, source_url: str = "", source_date: str = "") -> list[dict[str, Any]]:
    blob = html_to_text(text) if "<" in (text or "") else (text or "")
    if not blob.strip():
        return []
    lines: list[str] = []
    for match in _GUIDANCE_LINE.finditer(blob):
        line = re.sub(r"\s+", " ", match.group(0)).strip()
        cap = re.search(r"[A-Z]", line)
        if cap and cap.start() < 48:
            line = line[cap.start():]
        if line and line not in lines:
            lines.append(line)
        if len(lines) >= 6:
            break
    if not lines:
        return []
    joined = " ".join(lines)
    tone = guidance_tone(joined)
    if tone == "Unmeasured":
        return []
    return [{
        "tone": tone,
        "excerpt": lines[0][:280],
        "excerpts": lines[:4],
        "source": source,
        "source_url": source_url,
        "source_date": source_date,
        "method": "rule_extract",
        "not_an_llm": True,
    }]


def extract_from_html(html_blob: str, *, source: str, source_url: str = "") -> dict[str, Any]:
    text = html_to_text(html_blob)
    return {
        "kpis": extract_rates_from_text(text, source=source, source_url=source_url),
        "guidance": extract_guidance(text, source=source, source_url=source_url),
    }


def _read_local(path: str) -> str:
    from pathlib import Path

    file = Path(path)
    if not file.exists() or file.suffix.lower() not in _TEXT_EXT:
        return ""
    try:
        if file.stat().st_size > 2_000_000:
            return ""
        return file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def extract_from_uploads(symbol: str, uploads: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Read commentary / filing uploads already on disk. Does not download."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    items = list(uploads or [])
    if not items:
        try:
            from reporting.evidence_intake import list_uploads
            items = list_uploads(symbol)
        except Exception:
            items = []
    kpis: dict[str, dict[str, Any]] = {}
    guidance: list[dict[str, Any]] = []
    scanned = 0
    for item in items:
        kind = str(item.get("kind") or "")
        if kind not in {"management_commentary", "order_book_guidance", "annual_report", "financial_history"}:
            continue
        rel = str(item.get("path") or "")
        text = _read_local(str(root / rel)) if rel else ""
        if not text and item.get("commentary"):
            text = str(item.get("commentary") or "")
        if not text:
            continue
        scanned += 1
        source = f"Uploaded {kind} ({item.get('filename') or 'file'})"
        url = str(item.get("source_url") or "")
        date = str(item.get("as_of") or item.get("uploaded_at") or "")
        for kpi_id, snap in extract_rates_from_text(text, source=source, source_url=url).items():
            kpis.setdefault(kpi_id, {**snap, "source_date": date})
        guidance.extend(extract_guidance(text, source=source, source_url=url, source_date=date))
    return {"kpis": kpis, "guidance": guidance[:8], "files_read": scanned}


def merge_kpi_maps(*maps: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """First measured value wins. Later maps only fill holes."""
    out: dict[str, dict[str, Any]] = {}
    for payload in maps:
        for key, snap in dict(payload or {}).items():
            if not isinstance(snap, Mapping) or snap.get("current") is None:
                continue
            if key in out:
                continue
            out[key] = dict(snap)
    return out
