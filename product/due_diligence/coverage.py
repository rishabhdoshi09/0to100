"""Research-coverage inventory — files only; never fetches the internet.

Research Coverage is *not* Fundamental Quality. It answers: how much of the
sector-required *dataset* is actually on disk and still fresh enough to use.

Statuses are kept distinct on purpose:
  current              — present and within freshness policy
  stale                — present but past refresh policy
  not_yet_acquired     — never attempted
  acquisition_failed   — we tried; provider errored
  source_unavailable   — provider could not be reached / blocked
  metric_not_reported  — dataset exists; this particular metric is absent
  not_implemented      — framework lists the KPI; no validated acquisition path yet
  not_applicable       — sector does not need this dataset
  missing              — generic absent (legacy alias of not_yet_acquired)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from product.due_diligence.frameworks import get_framework
from product.due_diligence.series import dated_series

DATASET_IDS = (
    "company_master",
    "quarterly_results",
    "annual_financials",
    "sector_kpis",
    "shareholding",
    "promoter_pledge",
    "valuation",
    "peer_data",
    "exchange_filings",
    "corporate_announcements",
    "credit_ratings",
    "recent_news",
)

FRESHNESS: dict[str, timedelta] = {
    "company_master": timedelta(days=180),
    "quarterly_results": timedelta(days=95),
    "annual_financials": timedelta(days=370),
    "sector_kpis": timedelta(days=95),
    "shareholding": timedelta(days=95),
    "promoter_pledge": timedelta(days=95),
    "valuation": timedelta(days=14),
    "peer_data": timedelta(days=30),
    "exchange_filings": timedelta(days=3),
    "corporate_announcements": timedelta(days=3),
    "credit_ratings": timedelta(days=30),
    "recent_news": timedelta(hours=6),
}

LABELS: dict[str, str] = {
    "company_master": "Company classification",
    "quarterly_results": "Quarterly financials",
    "annual_financials": "Annual financials",
    "sector_kpis": "Sector KPIs",
    "shareholding": "Shareholding",
    "promoter_pledge": "Promoter pledge",
    "valuation": "Valuation",
    "peer_data": "Peer data",
    "exchange_filings": "Exchange filings",
    "corporate_announcements": "Corporate announcements",
    "credit_ratings": "Credit ratings",
    "recent_news": "News",
}

REQUIRED_FOR_COVERAGE = (
    "company_master",
    "quarterly_results",
    "annual_financials",
    "sector_kpis",
    "shareholding",
    "promoter_pledge",
    "valuation",
    "peer_data",
    "exchange_filings",
    "corporate_announcements",
    "recent_news",
)

OPTIONAL_DATASETS = ("credit_ratings",)

_GENERIC_KPI_IDS = {
    "pat", "sales", "opm", "eps", "promoter", "pledge", "fii", "dii", "public", "cfo", "roe", "roce", "borrowings",
}

_SCREENER_LANE = {
    "company_master", "quarterly_results", "annual_financials", "shareholding",
    "promoter_pledge", "valuation", "peer_data", "sector_kpis",
}
_NSE_FILINGS_LANE = {
    "exchange_filings", "corporate_announcements", "credit_ratings", "sector_kpis",
}
_NSE_ANNUAL_LANE = {"annual_financials", "sector_kpis"}
_NEWS_LANE = {"recent_news"}

STATUS_LABEL = {
    "current": "Current",
    "stale": "Stale",
    "not_yet_acquired": "Not yet acquired",
    "acquisition_failed": "Acquisition failed",
    "source_unavailable": "Source unavailable",
    "metric_not_reported": "Metric not reported",
    "not_implemented": "No validated acquisition path",
    "not_applicable": "Not applicable",
    "missing": "Not yet acquired",
}


def _parse_iso(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _age_label(acquired_at: datetime | None, now: datetime) -> str | None:
    if acquired_at is None:
        return None
    seconds = max(0, int((now - acquired_at).total_seconds()))
    if seconds < 90:
        return "just now"
    if seconds < 3600:
        mins = max(1, seconds // 60)
        return f"{mins} min old"
    if seconds < 86400:
        hours = max(1, seconds // 3600)
        return f"{hours} hour old" if hours == 1 else f"{hours} hours old"
    days = max(1, seconds // 86400)
    return f"{days} day old" if days == 1 else f"{days} days old"


def _table_has_values(rows: Any) -> bool:
    if not isinstance(rows, list):
        return False
    for row in rows:
        if isinstance(row, Mapping) and dated_series(row):
            return True
    return False


def _news_items(news: Any) -> list[dict[str, Any]]:
    if isinstance(news, dict):
        items = news.get("items")
        return [row for row in list(items or []) if isinstance(row, dict)]
    if isinstance(news, list):
        return [row for row in news if isinstance(row, Mapping)]
    return []


def _dataset_meta(facts: Mapping[str, Any], dataset_id: str) -> dict[str, Any]:
    meta = facts.get("dataset_meta")
    if isinstance(meta, dict):
        row = meta.get(dataset_id)
        if isinstance(row, dict):
            return row
    return {}


def _dataset_checked_at(
    facts: Mapping[str, Any],
    dataset_id: str,
    *,
    fetched_at: str = "",
) -> datetime | None:
    row = _dataset_meta(facts, dataset_id)
    stamp = _parse_iso(row.get("checked_at") or row.get("fetched_at") or row.get("acquired_at"))
    if stamp:
        return stamp
    pack = _parse_iso(facts.get("acquired_at") or facts.get("inspected_at"))
    if pack:
        return pack
    return _parse_iso(fetched_at)


def _dataset_error(facts: Mapping[str, Any], dataset_id: str) -> str | None:
    row = _dataset_meta(facts, dataset_id)
    status = str(row.get("status") or "")
    if status in {"acquisition_failed", "source_unavailable"}:
        return status
    if row.get("error"):
        return "acquisition_failed"
    return None


def _present_company_master(raw: Mapping[str, Any], facts: Mapping[str, Any]) -> bool:
    if str(facts.get("sector") or "").strip():
        return True
    about = str(raw.get("about") or raw.get("industry") or raw.get("sector") or "").strip()
    return bool(about)


def _present_quarterly(raw: Mapping[str, Any]) -> bool:
    return _table_has_values(raw.get("quarterly_results"))


def _present_annual(raw: Mapping[str, Any], facts: Mapping[str, Any]) -> bool:
    if _table_has_values(raw.get("profit_loss")) or _table_has_values(raw.get("balance_sheet")):
        return True
    if _table_has_values(raw.get("cash_flow")):
        return True
    for item in list(facts.get("downloads") or []):
        if not isinstance(item, Mapping):
            continue
        blob = f"{item.get('url') or ''} {item.get('path') or ''} {item.get('title') or ''}".lower()
        if "annual" in blob and item.get("ok"):
            return True
    return False


def _present_shareholding(raw: Mapping[str, Any], facts: Mapping[str, Any]) -> bool:
    if _table_has_values(raw.get("shareholding")):
        return True
    return bool(facts.get("shareholding_rows"))


def _finding_value(finding: Mapping[str, Any]) -> Any:
    if finding.get("latest") is not None:
        return finding.get("latest")
    snap = finding.get("snapshot")
    if isinstance(snap, Mapping):
        return snap.get("current")
    return None


def _present_pledge(
    raw: Mapping[str, Any],
    facts: Mapping[str, Any],
    findings: Sequence[Mapping[str, Any]],
) -> bool:
    for finding in findings:
        fid = str(finding.get("id") or finding.get("kpi_id") or "")
        if fid in {"pledge", "promoter_pledge_pct"} and _finding_value(finding) is not None:
            return True
    kpis = facts.get("kpis")
    if isinstance(kpis, Mapping) and isinstance(kpis.get("pledge"), Mapping):
        if kpis["pledge"].get("current") is not None:
            return True
    rows = raw.get("shareholding")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            label = str(row.get("row_label") or row.get("name") or "").lower()
            if "pledge" in label and dated_series(row):
                return True
    return False


def _present_valuation(raw: Mapping[str, Any]) -> bool:
    ratios = raw.get("key_ratios")
    if not isinstance(ratios, list) or not ratios:
        return False
    blob = " ".join(
        str(row.get("name") or row.get("row_label") or "").lower()
        for row in ratios if isinstance(row, Mapping)
    )
    return any(tok in blob for tok in ("pe", "p/e", "pb", "p/b", "ev/ebitda", "market cap", "roe"))


def _present_peers(raw: Mapping[str, Any], facts: Mapping[str, Any]) -> bool:
    peers = raw.get("peer_comparison") or raw.get("peers")
    if isinstance(peers, list) and len(peers) > 0:
        return True
    if isinstance(peers, dict) and (peers.get("rows") or peers.get("symbols")):
        return True
    return bool(facts.get("peer_symbols"))


def _present_filings(facts: Mapping[str, Any]) -> bool:
    filings = facts.get("filings")
    if isinstance(filings, list) and filings:
        return True
    for item in list(facts.get("downloads") or []):
        if isinstance(item, Mapping) and item.get("ok"):
            path = str(item.get("path") or item.get("url") or "").lower()
            if "option_chain" in path:
                continue
            if "nse_" in path or "filing" in path or "att" in path or "annual" in path:
                return True
    return False


def _present_announcements(facts: Mapping[str, Any]) -> bool:
    anns = facts.get("announcements")
    if isinstance(anns, list) and anns:
        return True
    headlines = facts.get("headlines")
    return isinstance(headlines, list) and len(headlines) > 0


def _present_ratings(facts: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> bool:
    if facts.get("credit_ratings"):
        return True
    for ev in list(events) + list(facts.get("announcements") or []):
        if not isinstance(ev, Mapping):
            continue
        blob = " ".join(
            str(ev.get(k) or "") for k in ("headline", "title", "why_it_matters", "event_type", "subject")
        ).lower()
        if "rating" in blob and any(
            tok in blob for tok in ("credit", "outlook", "upgrade", "downgrade", "crisil", "icra", "care")
        ):
            return True
    return False


def _present_news(news: Any, events: Sequence[Mapping[str, Any]]) -> bool:
    if _news_items(news):
        return True
    return any(str(ev.get("source_kind") or ev.get("category") or "") for ev in events)


def _present_sector_kpis(
    findings: Sequence[Mapping[str, Any]],
    facts: Mapping[str, Any],
    framework_id: str,
) -> bool:
    fw = get_framework(framework_id)
    sector_ids = {spec.id for spec in fw["kpis"] if spec.id not in _GENERIC_KPI_IDS}
    have: set[str] = set()
    for finding in findings:
        fid = str(finding.get("id") or finding.get("kpi_id") or "")
        if _finding_value(finding) is not None or finding.get("available"):
            have.add(fid)
    kpis = facts.get("kpis")
    if isinstance(kpis, Mapping):
        for key, snap in kpis.items():
            if isinstance(snap, Mapping) and snap.get("current") is not None:
                have.add(str(key))
    if not sector_ids:
        return bool(have)
    return bool(have & sector_ids)


def inspect_research_coverage(
    *,
    symbol: str,
    raw: Mapping[str, Any] | None,
    autonomy: Mapping[str, Any] | None,
    news: Any = None,
    framework_id: str,
    findings: Sequence[Mapping[str, Any]] | None = None,
    events: Sequence[Mapping[str, Any]] | None = None,
    fetched_at: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Pure cache inspection. Safe to call from GET handlers."""
    now = now or datetime.now(timezone.utc)
    raw = raw if isinstance(raw, Mapping) else {}
    facts = autonomy if isinstance(autonomy, Mapping) else {}
    findings = list(findings or [])
    events = list(events or [])
    fw = get_framework(framework_id)

    presenters = {
        "company_master": lambda: _present_company_master(raw, facts),
        "quarterly_results": lambda: _present_quarterly(raw),
        "annual_financials": lambda: _present_annual(raw, facts),
        "sector_kpis": lambda: _present_sector_kpis(findings, facts, framework_id),
        "shareholding": lambda: _present_shareholding(raw, facts),
        "promoter_pledge": lambda: _present_pledge(raw, facts, findings),
        "valuation": lambda: _present_valuation(raw),
        "peer_data": lambda: _present_peers(raw, facts),
        "exchange_filings": lambda: _present_filings(facts),
        "corporate_announcements": lambda: _present_announcements(facts),
        "credit_ratings": lambda: _present_ratings(facts, events),
        "recent_news": lambda: _present_news(news, events),
    }

    datasets: list[dict[str, Any]] = []
    to_fetch: list[str] = []
    available_n = 0
    required_n = 0
    latest_refresh: datetime | None = None

    for ds_id in DATASET_IDS:
        required = ds_id in REQUIRED_FOR_COVERAGE
        present = bool(presenters[ds_id]())
        checked_at = _dataset_checked_at(facts, ds_id, fetched_at=fetched_at)
        error_kind = _dataset_error(facts, ds_id)
        window = FRESHNESS[ds_id]
        stale = False
        if present and checked_at is not None:
            stale = (now - checked_at) > window
        elif present and checked_at is None:
            stale = False

        if present and checked_at is not None:
            if latest_refresh is None or checked_at > latest_refresh:
                latest_refresh = checked_at

        if present and stale:
            status = "stale"
        elif present:
            status = "current"
        elif error_kind == "source_unavailable":
            status = "source_unavailable"
        elif error_kind == "acquisition_failed":
            status = "acquisition_failed"
        elif str(_dataset_meta(facts, ds_id).get("status") or "") == "current" or _dataset_meta(facts, ds_id).get("fetched_at"):
            status = "metric_not_reported"
        else:
            status = "not_yet_acquired"

        if status == "current":
            if ds_id in {"recent_news", "exchange_filings", "corporate_announcements", "valuation"} and checked_at is not None:
                age = _age_label(checked_at, now) or "Current"
            else:
                age = "Current"
        else:
            age = STATUS_LABEL.get(status, status)
            if status == "stale":
                aged = _age_label(checked_at, now)
                age = f"Stale · {aged}" if aged else "Stale"

        label = LABELS[ds_id]
        if ds_id == "sector_kpis":
            label = f"{fw.get('label') or 'Sector'} KPIs"

        datasets.append({
            "id": ds_id,
            "label": label,
            "status": status,
            "required": required,
            "present": present,
            "age_label": age,
            "checked_at": checked_at.isoformat() if checked_at else None,
            "freshness_hours": round(window.total_seconds() / 3600, 1),
        })
        if required:
            required_n += 1
            if status == "current":
                available_n += 1
        if status in {"stale", "not_yet_acquired", "acquisition_failed", "source_unavailable"} and (
            required or ds_id in OPTIONAL_DATASETS and status == "stale"
        ):
            to_fetch.append(ds_id)

    coverage_pct = round(100.0 * available_n / required_n, 1) if required_n else 0.0
    return {
        "coverage_pct": coverage_pct,
        "available_n": available_n,
        "required_n": required_n,
        "summary": f"{available_n}/{required_n} datasets available",
        "needs_acquire": bool(to_fetch),
        "to_fetch": to_fetch,
        "datasets": datasets,
        "latest_data_refresh": latest_refresh.isoformat() if latest_refresh else None,
        "framework_id": framework_id,
        "symbol": str(symbol or "").upper(),
        "not_a_quality_score": True,
    }


def availability_state_for_kpi(
    *,
    kpi_id: str,
    has_value: bool,
    missing_ok: bool,
    coverage: Mapping[str, Any],
    implemented: bool = True,
) -> str:
    """Map a KPI onto the user-facing availability vocabulary."""
    if has_value:
        return "reported"
    if not implemented:
        return "not_implemented"
    by_id = {
        str(d.get("id")): d
        for d in list(coverage.get("datasets") or [])
        if isinstance(d, Mapping)
    }
    ds_id = "sector_kpis"
    if kpi_id in {"promoter", "promoter_holding_pct", "fii", "dii", "public"}:
        ds_id = "shareholding"
    elif kpi_id in {"pledge", "promoter_pledge_pct"}:
        ds_id = "promoter_pledge"
    status = str((by_id.get(ds_id) or {}).get("status") or "")
    if status == "not_yet_acquired":
        return "not_yet_acquired"
    if status == "acquisition_failed":
        return "acquisition_failed"
    if status == "source_unavailable":
        return "source_unavailable"
    if status == "not_applicable":
        return "not_applicable"
    if missing_ok:
        return "metric_not_reported"
    return "metric_not_reported"


def provider_lanes(dataset_ids: Sequence[str]) -> dict[str, bool]:
    want = set(dataset_ids)
    return {
        "screener": bool(want & _SCREENER_LANE),
        "nse_filings": bool(want & _NSE_FILINGS_LANE),
        "nse_annual": bool(want & _NSE_ANNUAL_LANE),
        "news": bool(want & _NEWS_LANE),
        "sector_fallback": "sector_kpis" in want,
    }


def display_status(status: str) -> str:
    return STATUS_LABEL.get(status, status or "Data unavailable")
