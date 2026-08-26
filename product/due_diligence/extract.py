"""Rule-based extraction from tables, filings and commentary.

This is the 'act like an LLM without one' layer: regex and table matching
only. Missing stays missing. Never calls a language model.
"""
from __future__ import annotations

import html
import re
from typing import Any, Mapping, Sequence

from product.due_diligence.series import _f, dated_series, find_row, infer_period_type, normalize_label, snapshot

_GNPA_NEEDLES = ("gross npa", "gnpa", "gross non performing")
_NNPA_NEEDLES = ("net npa", "nnpa", "net non performing")
_PLEDGE_NEEDLES = ("pledge", "encumbrance")
_CASA_NEEDLES = ("casa", "current account savings")
_NIM_NEEDLES = ("nim", "net interest margin")
_NIM_COLLISION = (
    "financing margin", "operating margin", "ebitda margin", "ebit margin",
    "gross margin", "yield on", "investment yield", "treasury yield",
)
_CET1_NEEDLES = ("cet1", "cet 1", "common equity tier")
_CRAR_NEEDLES = ("crar", "capital adequacy", "capital adequacy ratio")
_PCR_NEEDLES = ("pcr", "provision coverage")
_SLIPPAGE_NEEDLES = ("slippage", "slippages")
_CREDIT_COST_NEEDLES = ("credit cost", "credit costs")
_ROA_NEEDLES = ("roa", "return on assets")
_ROE_NEEDLES = ("roe", "return on equity")
_ADVANCES_NEEDLES = ("gross advances", "net advances", "total advances")
_DEPOSITS_NEEDLES = ("total deposits", "deposits")
_LDR_NEEDLES = ("credit deposit", "cd ratio", "loan deposit", "loan to deposit")

_GNPA_LABEL = re.compile(r"(?:gross\s*npa[s]?|gnpa|gross\s*non[\s-]*performing)", re.I)
_NNPA_LABEL = re.compile(r"(?:net\s*npa[s]?|nnpa|net\s*non[\s-]*performing)", re.I)
_PLEDGE_LABEL = re.compile(r"(?:promoter\s*)?pledged?(?:\s+shares?|\s+equity|\s+holding)?", re.I)
_CASA_LABEL = re.compile(r"\bcasa(?:\s+ratio)?\b", re.I)
_NIM_LABEL = re.compile(r"(?:net\s+interest\s+margin|\bnim\b)", re.I)
_CET1_LABEL = re.compile(r"(?:cet-?1|common\s+equity\s+tier(?:\s*1)?)", re.I)
_CRAR_LABEL = re.compile(r"(?:crar|capital\s+adequacy(?:\s+ratio)?)", re.I)
_PCR_LABEL = re.compile(r"(?:provision\s+coverage(?:\s+ratio)?|\bpcr\b)", re.I)
_SLIPPAGE_LABEL = re.compile(r"slippages?", re.I)
_CREDIT_COST_LABEL = re.compile(r"credit\s+costs?", re.I)
_ROA_LABEL = re.compile(r"(?:return\s+on\s+assets|\broa\b)", re.I)
_ROE_LABEL = re.compile(r"(?:return\s+on\s+equity|\broe\b)", re.I)
_PERCENT_RE = re.compile(
    r"(\d{1,2}(?:\.\s*\d+)?)\s*(?:%|(?:per\s*cent|percent)\b)",
    re.I,
)
_AMOUNT_RE = re.compile(
    r"(?:rs\.?|inr|₹)?\s*([\d,]+(?:\.\d+)?)\s*(lakh\s+crore|lakh\s+cr|crore|cr)\b",
    re.I,
)
_PERIOD_NEAR_RE = re.compile(
    r"(?:Q[1-4]\s*FY\s*\d{2,4}|FY\s*\d{2}(?:-?\d{2,4})?|"
    r"(?:quarter|year|period)\s+ended\s+[A-Za-z]{3,9}\s+\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})",
    re.I,
)
_BASIS_NEAR_RE = re.compile(r"\b(standalone|consolidated)\b", re.I)
_SKIP_PREFIX = ("below", "under", "above", "over", "upto", "up to", "within", "least")
_ADVANCES_LABEL = re.compile(r"(?:gross|net|total)\s+advances", re.I)
_DEPOSITS_LABEL = re.compile(r"(?:total|customer)\s+deposits", re.I)
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


def _context_ok(kpi_id: str, left: str, right: str, document_type: str = "") -> bool:
    window = f"{left} {right}".lower()
    if kpi_id == "nim":
        if any(tok in window for tok in _NIM_COLLISION):
            return False
        if "net interest margin" in window or re.search(r"\bnim\b", window):
            return True
        return False
    if kpi_id == "gnpa":
        if re.search(r"\bnet\s+npa|\bnnpa\b", left) and "gross" not in left:
            return False
        if "coverage" in window and "gross" not in left:
            return False
        return True
    if kpi_id == "nnpa":
        if re.search(r"\bgross\s+npa|\bgnpa\b", left) and "net" not in left:
            return False
        return True
    if kpi_id == "advances":
        if any(tok in window for tok in ("sanctioned", "under a special", "scheme advances")):
            return False
        return True
    if document_type in {"financial_media"} and kpi_id in {"nim", "gnpa", "nnpa", "casa", "cet1"}:
        # Media copy is a fallback; still require an explicit label already matched.
        return True
    return True


def _period_from_context(left: str, right: str) -> str:
    blob = f"{left} {right}"
    match = _PERIOD_NEAR_RE.search(blob)
    return re.sub(r"\s+", " ", match.group(0)).strip() if match else ""


def _basis_from_context(left: str, right: str) -> str:
    blob = f"{left} {right}"
    match = _BASIS_NEAR_RE.search(blob)
    return match.group(1).lower() if match else ""


def _last_percent(
    label: re.Pattern[str],
    text: str,
    *,
    kpi_id: str = "",
    document_type: str = "",
) -> tuple[float | None, str, str]:
    found: list[tuple[float, str, str]] = []
    blob = text or ""
    for match in label.finditer(blob):
        left = blob[max(0, match.start() - 80): match.start()]
        matched = match.group(0)
        window = blob[match.end(): match.end() + 96]
        if kpi_id and not _context_ok(kpi_id, f"{left} {matched}", window, document_type):
            continue
        for item in _PERCENT_RE.finditer(window):
            prefix = window[max(0, item.start() - 12): item.start()].lower()
            if any(tok in prefix for tok in _SKIP_PREFIX):
                continue
            number = _rate(item.group(1).replace(" ", ""))
            if number is None:
                continue
            period = _period_from_context(left, window)
            basis = _basis_from_context(left, window)
            found.append((number, period, basis))
            break
    if not found:
        return None, "", ""
    return found[-1]


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


def key_level_value(rows: Sequence[Mapping[str, Any]] | None, needles: Sequence[str]) -> float | None:
    want = [normalize_label(n) for n in needles if n]
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        name = normalize_label(row.get("name") or row.get("row_label") or "")
        if not name:
            continue
        if name in want or any(n and n in name for n in want):
            number = _f(row.get("value") or row.get("current"))
            if number is not None and number >= 0:
                return number
    return None


def _last_amount(label: re.Pattern[str], text: str, *, kpi_id: str = "") -> tuple[float | None, str, str]:
    found: list[tuple[float, str, str]] = []
    blob = text or ""
    for match in label.finditer(blob):
        left = blob[max(0, match.start() - 80): match.start()]
        window = blob[match.end(): match.end() + 64]
        if kpi_id and not _context_ok(kpi_id, f"{left} {match.group(0)}", window):
            continue
        item = _AMOUNT_RE.match(window.lstrip(" :,-")) or _AMOUNT_RE.search(window)
        if not item:
            continue
        try:
            number = float(item.group(1).replace(",", ""))
        except ValueError:
            continue
        unit = item.group(2).lower()
        if "lakh" in unit:
            number *= 100_000.0
        if number > 0:
            found.append((number, _period_from_context(left, window), _basis_from_context(left, window)))
    if not found:
        return None, "", ""
    return max(found, key=lambda row: row[0])


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
            snap = snapshot(series, kind=kind, year_steps=1 if key in {"profit_loss", "balance_sheet", "cash_flow"} else 4, table=key)
            current = snap.get("current")
            if current is None:
                continue
            if kind == "rate" and (current < 0 or current > 100):
                continue
            return {**snap, "table": key, "source": "results table on file"}
    return None


def extract_kpis_from_raw(raw: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Fill sector KPIs from any table or key-ratio snapshot in the cache."""
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
    rate_specs = (
        ("gnpa", _GNPA_NEEDLES, "quarterly_results"),
        ("nnpa", _NNPA_NEEDLES, "quarterly_results"),
        ("pledge", _PLEDGE_NEEDLES, "shareholding"),
        ("casa", _CASA_NEEDLES, "quarterly_results"),
        ("nim", _NIM_NEEDLES, "quarterly_results"),
        ("cet1", _CET1_NEEDLES, "quarterly_results"),
        ("crar", _CRAR_NEEDLES, "quarterly_results"),
        ("pcr", _PCR_NEEDLES, "quarterly_results"),
        ("slippages", _SLIPPAGE_NEEDLES, "quarterly_results"),
        ("credit_cost", _CREDIT_COST_NEEDLES, "quarterly_results"),
        ("roa", _ROA_NEEDLES, "key_ratios"),
        ("roe", _ROE_NEEDLES, "key_ratios"),
        ("loan_deposit", _LDR_NEEDLES, "quarterly_results"),
    )
    level_specs = (
        ("advances", _ADVANCES_NEEDLES, "balance_sheet"),
        ("deposits", _DEPOSITS_NEEDLES, "balance_sheet"),
    )
    for kpi_id, needles, prefer in rate_specs:
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
    for kpi_id, needles, prefer in level_specs:
        snap = series_from_tables(tables, needles, kind="level", prefer=prefer)
        if snap:
            out[kpi_id] = {**snap, "source": "Screener.in cache / company results table"}
            continue
        current = key_level_value(tables["key_ratios"], needles)
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


def _print_snap(
    current: float,
    *,
    source: str,
    source_url: str = "",
    period: str = "",
    reporting_basis: str = "",
    document_type: str = "",
) -> dict[str, Any]:
    period_label = period or "extracted print"
    period_type = infer_period_type(period_label, "")
    return {
        "current": current,
        "current_period": period_label,
        "previous": None,
        "previous_period": "",
        "year_ago": None,
        "year_ago_period": "",
        "qoq_change": None,
        "yoy_change": None,
        "points": [{"period": period_label, "value": current, "period_type": period_type}],
        "source": source,
        "source_url": source_url,
        "period_type": period_type,
        "previous_period_type": "",
        "year_ago_period_type": "",
        "reporting_basis": reporting_basis,
        "document_type": document_type,
        "source_count": 1,
        "source_consensus": "single",
        "agreeing_sources": [source] if source else [],
    }


def extract_rates_from_text(
    text: str,
    *,
    source: str,
    source_url: str = "",
    document_type: str = "",
) -> dict[str, dict[str, Any]]:
    blob = html_to_text(text) if "<" in (text or "") and ">" in (text or "") else (text or "")
    out: dict[str, dict[str, Any]] = {}
    mapping = (
        ("gnpa", _GNPA_LABEL),
        ("nnpa", _NNPA_LABEL),
        ("pledge", _PLEDGE_LABEL),
        ("casa", _CASA_LABEL),
        ("nim", _NIM_LABEL),
        ("cet1", _CET1_LABEL),
        ("crar", _CRAR_LABEL),
        ("pcr", _PCR_LABEL),
        ("slippages", _SLIPPAGE_LABEL),
        ("credit_cost", _CREDIT_COST_LABEL),
        ("roa", _ROA_LABEL),
        ("roe", _ROE_LABEL),
        ("loan_deposit", re.compile(r"(?:credit[\s-]*deposit|c[\s-]*d\s+ratio|loan[\s-]*deposit)", re.I)),
    )
    for kpi_id, pattern in mapping:
        current, period, basis = _last_percent(
            pattern, blob, kpi_id=kpi_id, document_type=document_type,
        )
        if current is None or not _in_bounds(kpi_id, current):
            continue
        out[kpi_id] = _print_snap(
            current, source=source, source_url=source_url,
            period=period, reporting_basis=basis, document_type=document_type,
        )
    for kpi_id, pattern in (("advances", _ADVANCES_LABEL), ("deposits", _DEPOSITS_LABEL)):
        current, period, basis = _last_amount(pattern, blob, kpi_id=kpi_id)
        if current is None:
            continue
        out[kpi_id] = _print_snap(
            current, source=source, source_url=source_url,
            period=period, reporting_basis=basis, document_type=document_type,
        )
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


_ORDER_RE = re.compile(
    r"(order[\s-]?book|order\s+inflow|order\s+intake|total\s+contract\s+value|\btcv\b)"
    r"[^\d%]{0,48}(\d[\d,]*(?:\.\d+)?)\s*(crore|cr|billion|bn)",
    re.I,
)
_SEGMENT_RE = re.compile(
    r"\b((?:[A-Za-z][A-Za-z0-9&/-]*\s+){0,2}[A-Za-z][A-Za-z0-9&/-]*)\s+"
    r"(?:segment|division|business)\s+"
    r"(?:contributed|accounted for|stood at|was|is)\s+(\d{1,2}(?:\.\d+)?)\s*%",
    re.I,
)
_SPEAKER_RE = re.compile(
    r"(?:(?:the\s+)?(?:md|ceo|cfo|chairman|management))\s+"
    r"(?:said|stated|added|noted|commented)[,:]?\s+(.{20,280})",
    re.I,
)


def extract_order_book(text: str, *, source: str, source_url: str = "", source_date: str = "") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen = set()
    for match in _ORDER_RE.finditer(text or ""):
        metric = re.sub(r"\s+", " ", match.group(1)).strip().title()
        raw = match.group(2).replace(",", "")
        unit = "₹ cr" if match.group(3).lower() in {"crore", "cr"} else match.group(3)
        try:
            value = float(raw)
        except ValueError:
            continue
        if value <= 0:
            continue
        key = (metric.lower(), value)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "metric": metric,
            "value": value,
            "unit": unit,
            "period": "",
            "wording": re.sub(r"\s+", " ", match.group(0)).strip()[:240],
            "as_of": source_date,
            "source_url": source_url,
            "source": source,
            "fact": f"{metric}: {value} {unit} ({source_date or 'date unavailable'})".strip(),
        })
        if len(out) >= 6:
            break
    return out


def extract_segments(text: str, *, source: str, source_url: str = "") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen = set()
    for match in _SEGMENT_RE.finditer(text or ""):
        name = re.sub(r"\s+", " ", match.group(1)).strip(" -")
        name = re.sub(r"^(?:the|a|an)\s+", "", name, flags=re.I).strip()
        if len(name) < 3 or name.lower() in {"the", "this", "our", "its"}:
            continue
        try:
            mix = float(match.group(2))
        except ValueError:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "segment": name,
            "revenue_cr": None,
            "revenue_mix_pct": mix,
            "source_url": source_url,
            "source": source,
        })
        if len(out) >= 6:
            break
    return out


def extract_commentary(text: str, *, source: str, source_url: str = "", source_date: str = "") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen = set()
    for match in _SPEAKER_RE.finditer(text or ""):
        speaker = re.sub(r"\s+", " ", match.group(0).split("said")[0] if "said" in match.group(0).lower() else "Management")
        speaker = speaker.strip(" :,") or "Management"
        if len(speaker) > 48:
            speaker = "Management"
        commentary = re.sub(r"\s+", " ", match.group(1)).strip()
        cap = re.search(r"[A-Z]", commentary)
        if cap and cap.start() < 24:
            commentary = commentary[cap.start():]
        if len(commentary) < 24:
            continue
        key = commentary[:80]
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "speaker": speaker[:48],
            "topic": "",
            "commentary": commentary[:400],
            "event_date": source_date,
            "source_url": source_url,
            "source": source,
        })
        if len(out) >= 6:
            break
    if not out and "annual report" not in (source or "").lower():
        for item in extract_guidance(text, source=source, source_url=source_url, source_date=source_date):
            out.append({
                "speaker": "Management",
                "topic": item.get("tone") or "",
                "commentary": item.get("excerpt") or "",
                "event_date": source_date,
                "source_url": source_url,
                "source": source,
            })
    return out


def extract_research_pack(
    text: str,
    *,
    source: str,
    source_url: str = "",
    source_date: str = "",
    document_type: str = "",
) -> dict[str, Any]:
    blob = html_to_text(text) if "<" in (text or "") and ">" in (text or "") else (text or "")
    return {
        "kpis": extract_rates_from_text(
            blob, source=source, source_url=source_url, document_type=document_type,
        ),
        "guidance": extract_guidance(blob, source=source, source_url=source_url, source_date=source_date),
        "commentary": extract_commentary(blob, source=source, source_url=source_url, source_date=source_date),
        "order_book": extract_order_book(blob, source=source, source_url=source_url, source_date=source_date),
        "segments": extract_segments(blob, source=source, source_url=source_url),
    }


def extract_from_html(html_blob: str, *, source: str, source_url: str = "") -> dict[str, Any]:
    return extract_research_pack(html_blob, source=source, source_url=source_url)


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
    commentary: list[dict[str, Any]] = []
    order_book: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    scanned = 0
    for item in items:
        kind = str(item.get("kind") or "")
        if kind not in {"management_commentary", "order_book_guidance", "annual_report", "financial_history", "business_segments"}:
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
        parsed = extract_research_pack(text, source=source, source_url=url, source_date=date)
        for kpi_id, snap in (parsed.get("kpis") or {}).items():
            kpis.setdefault(kpi_id, {**snap, "source_date": date})
        guidance.extend(parsed.get("guidance") or [])
        commentary.extend(parsed.get("commentary") or [])
        order_book.extend(parsed.get("order_book") or [])
        segments.extend(parsed.get("segments") or [])
    return {
        "kpis": kpis,
        "guidance": guidance[:8],
        "commentary": commentary[:6],
        "order_book": order_book[:6],
        "segments": segments[:6],
        "files_read": scanned,
    }


_RATE_KPI_IDS = {
    "gnpa", "nnpa", "pledge", "casa", "nim", "cet1", "crar", "pcr",
    "slippages", "credit_cost", "roa", "roe", "loan_deposit", "opm",
}
_RATE_BOUNDS = {
    "nim": (0.5, 10.0),
    "casa": (5.0, 80.0),
    "cet1": (5.0, 25.0),
    "crar": (8.0, 30.0),
    "gnpa": (0.0, 30.0),
    "nnpa": (0.0, 20.0),
    "pcr": (20.0, 100.0),
    "roa": (0.0, 10.0),
    "roe": (0.0, 50.0),
    "slippages": (0.0, 20.0),
    "credit_cost": (0.0, 10.0),
    "loan_deposit": (40.0, 130.0),
    "opm": (0.0, 100.0),
    "pledge": (0.0, 100.0),
}


def _in_bounds(kpi_id: str, current: Any) -> bool:
    try:
        number = float(current)
    except (TypeError, ValueError):
        return False
    lo, hi = _RATE_BOUNDS.get(kpi_id, (0.0, 100.0))
    return lo <= number <= hi


def merge_kpi_maps(*maps: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """First measured value wins. Later maps only fill holes or confirm.

    Agreeing prints increment source_count. Disagreements are flagged — never averaged.
    Invalid rates are dropped.
    """
    from product.due_diligence.provenance import material_disagreement

    out: dict[str, dict[str, Any]] = {}
    for payload in maps:
        for key, snap in dict(payload or {}).items():
            if not isinstance(snap, Mapping) or snap.get("current") is None:
                continue
            current = snap.get("current")
            if key in _RATE_KPI_IDS:
                if not _in_bounds(key, current):
                    continue
            incoming = dict(snap)
            source = str(incoming.get("source") or "")
            if key not in out:
                incoming.setdefault("source_count", 1)
                incoming.setdefault("source_consensus", "single")
                incoming.setdefault("agreeing_sources", [source] if source else [])
                out[key] = incoming
                continue
            existing = out[key]
            kind = "rate" if key in _RATE_KPI_IDS else "level"
            if material_disagreement(existing.get("current"), current, kind=kind):
                existing["source_consensus"] = "conflict"
                conflicts = list(existing.get("conflicting_sources") or [])
                conflicts.append({
                    "value": current,
                    "source": source,
                    "period": incoming.get("current_period"),
                    "source_url": incoming.get("source_url") or "",
                })
                existing["conflicting_sources"] = conflicts
                continue
            # Same figure from another source — record consensus, keep the first print.
            agreeing = list(existing.get("agreeing_sources") or [])
            if source and source not in agreeing:
                agreeing.append(source)
                existing["agreeing_sources"] = agreeing
                existing["source_count"] = len(agreeing)
                existing["source_consensus"] = "confirmed" if len(agreeing) >= 2 else "single"
    return out
