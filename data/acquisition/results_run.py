"""Resumable NSE results + XBRL ingest (ingest stage only)."""
from __future__ import annotations

import json
from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from typing import Any

from data.acquisition import PARSER_VERSION
from data.acquisition.anomalies import QUARANTINE, record as record_anomaly
from data.acquisition.cache import sha256_bytes, write_manifest, write_raw
from data.acquisition.http import get_bytes, nse_session
from data.nse_results_ingest import (
    _RESULTS_API,
    parse_xbrl_metrics,
    results_to_event_rows,
)
from data.period_alignment import classify_period, consol_label
from data.pit_events import write_events
from data.pit_fundamentals import merge_fundamentals

HOME = "https://www.nseindia.com/companies-listing/corporate-filings-financial-results"


def month_windows(start: date, end: date) -> list[tuple[str, str, str]]:
    """Inclusive month windows as DD-MM-YYYY pairs + yyyy-mm key."""
    out = []
    y, m = start.year, start.month
    while date(y, m, 1) <= end:
        last = monthrange(y, m)[1]
        fr = date(y, m, 1)
        to = date(y, m, last)
        if to > end:
            to = end
        out.append((fr.strftime("%d-%m-%Y"), to.strftime("%d-%m-%Y"), f"{y:04d}-{m:02d}"))
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return out


def fetch_window(period: str, fr: str, to: str, *, session) -> tuple[list[dict], dict]:
    url = (
        f"{_RESULTS_API}?index=equities&period={period}"
        f"&from_date={fr}&to_date={to}"
    )
    rel = f"results/{period}_{fr}_{to}.json"
    # resume from cache
    from data.acquisition.cache import raw_path
    cached = raw_path(rel)
    if cached.exists() and cached.stat().st_size > 20:
        try:
            rows = json.loads(cached.read_text(encoding="utf-8"))
            if isinstance(rows, list):
                return rows, {"cached": True, "n": len(rows), "url": url, "path": str(cached)}
        except Exception:
            pass
    blob, meta = get_bytes(url, session=session, timeout=180, retries=3)
    meta["period"] = period
    if not blob:
        return [], meta
    try:
        data = json.loads(blob.decode("utf-8"))
    except Exception as exc:
        meta["error"] = f"json:{exc}"
        return [], meta
    rows = data if isinstance(data, list) else []
    write_raw(rel, json.dumps(rows).encode(), meta={**meta, "n": len(rows)})
    meta["n"] = len(rows)
    meta["cached"] = False
    return rows, meta


def _enrich_fund_row(raw: dict, metrics: dict[str, float], xbrl_hash: str) -> dict | None:
    from data.nse_results_ingest import result_row_to_fundamentals

    row = result_row_to_fundamentals(raw, metrics)
    if not row:
        return None
    align = classify_period(
        period=raw.get("period"),
        period_start=row.get("period_start"),
        period_end=row.get("period_end"),
        cumulative=str(raw.get("cumulative") or ""),
        relating_to=str(raw.get("relatingTo") or ""),
    )
    row["period_kind"] = align["period_kind"]
    row["quarterly_usable"] = align["quarterly_usable"]
    row["consol_basis"] = consol_label(raw.get("consolidated"))
    row["reporting_frequency"] = raw.get("period")
    row["parser_version"] = PARSER_VERSION
    row["raw_hash"] = xbrl_hash
    row["ingested_at"] = datetime.now(timezone.utc).isoformat()
    row["source_id"] = raw.get("xbrl") or raw.get("seqNumber")
    from research.data_foundation.quality import fundamental_quality
    row["field_quality"] = fundamental_quality(row)
    if not align["quarterly_usable"] and str(raw.get("period") or "").lower() == "quarterly":
        record_anomaly(
            source="nse_xbrl",
            symbol=row.get("symbol"),
            period=str(row.get("period_end")),
            anomaly_type="non_quarter_labelled_quarterly",
            severity="warn",
            raw_evidence=align,
            parser=PARSER_VERSION,
            suggested=QUARANTINE,
        )
    # Impossible EPS
    eps = metrics.get("basic_eps")
    if eps is not None and abs(float(eps)) > 1e6:
        record_anomaly(
            source="nse_xbrl",
            symbol=row.get("symbol"),
            period=str(row.get("period_end")),
            anomaly_type="impossible_eps",
            severity="error",
            raw_evidence={"basic_eps": eps},
            parser=PARSER_VERSION,
        )
        return None
    return row


def ingest_results_metadata(
    *,
    start: date = date(2016, 1, 1),
    end: date | None = None,
    periods: tuple[str, ...] = ("Quarterly", "Annual"),
) -> dict[str, Any]:
    end = end or date.today()
    sess = nse_session()
    all_raw: list[dict] = []
    windows = []
    failed = []
    for period in periods:
        for fr, to, key in month_windows(start, end):
            rows, meta = fetch_window(period, fr, to, session=sess)
            windows.append({**meta, "period": period, "from": fr, "to": to, "key": key})
            if meta.get("error") or meta.get("http_status") not in (None, 200):
                failed.append(meta)
            all_raw.extend(rows)
    events = results_to_event_rows(all_raw)
    # DATE_ONLY vs timestamp
    for ev in events:
        ts = ev.get("available_at_ts") or ""
        if "T" in str(ts) and len(str(ts)) >= 16:
            ev["time_quality"] = "EVENT_TIMESTAMP_STRONG"
        else:
            ev["time_quality"] = "EVENT_DATE_ONLY"
            ev["causal_effective"] = "NEXT_SESSION"
    ev_status = write_events(events, source="nse_financial_results")
    man = {
        "source": "nse_corporates_financial_results",
        "requested_range": [str(start), str(end)],
        "periods": list(periods),
        "windows": len(windows),
        "failed_windows": failed,
        "raw_rows": len(all_raw),
        "normalized_event_rows": ev_status.get("rows"),
        "event_symbols": ev_status.get("symbols"),
        "parser_version": PARSER_VERSION,
        "duplicate_policy": "event_id first-write",
    }
    write_manifest("results_events", man)
    return {**man, "event_status": ev_status, "raw": all_raw}


def ingest_xbrl(
    raw_rows: list[dict],
    *,
    max_files: int | None = None,
    workers: int = 8,
    prefer_consolidated: bool = True,
    min_year: int = 2019,
) -> dict[str, Any]:
    from data.nse_results_ingest import fetch_xbrl
    from data.acquisition.xbrl_select import select_bounded

    cands = select_bounded(
        raw_rows, min_year=min_year, max_per_symbol=12,
        prefer_consolidated=prefer_consolidated,
    )
    if max_files is not None:
        cands = cands[: max_files]
    sess = nse_session()
    rows: list[dict] = []
    failed = 0
    hashes: list[str] = []

    def _one(raw: dict) -> dict | None:
        url = str(raw.get("xbrl") or "")
        blob = fetch_xbrl(url, session=sess, retries=2)
        if not blob:
            record_anomaly(
                source="nse_xbrl", symbol=str(raw.get("symbol") or ""),
                period=str(raw.get("toDate") or ""),
                anomaly_type="xbrl_download_failed", severity="warn",
                raw_evidence={"url": url}, parser=PARSER_VERSION,
            )
            return None
        digest = sha256_bytes(blob)
        rel = f"xbrl/{digest[:20]}.xml"
        write_raw(rel, blob, meta={"url": url, "symbol": raw.get("symbol")})
        try:
            metrics = parse_xbrl_metrics(blob)
        except Exception as exc:
            record_anomaly(
                source="nse_xbrl", symbol=str(raw.get("symbol") or ""),
                period=str(raw.get("toDate") or ""),
                anomaly_type="xbrl_parse_failed", severity="error",
                raw_evidence={"url": url, "error": str(exc)}, parser=PARSER_VERSION,
            )
            return None
        if not metrics:
            return None
        return _enrich_fund_row(raw, metrics, digest)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one, raw) for raw in cands]
        for fut in as_completed(futs):
            try:
                row = fut.result()
                if row:
                    rows.append(row)
                    hashes.append(row.get("raw_hash") or "")
                else:
                    failed += 1
            except Exception:
                failed += 1

    status = merge_fundamentals(rows, source="nse_xbrl_financial_results")
    man = {
        "source": "nse_xbrl",
        "attempted": len(cands),
        "normalized_row_count": status.get("rows"),
        "failed_objects": failed,
        "parser_version": PARSER_VERSION,
        "raw_hashes_sample": hashes[:20],
    }
    write_manifest("results_xbrl", man)
    return {**man, "fund_status": status}
