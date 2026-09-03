"""Idempotent staged historical-evidence backfill.

SYSTEM TRIES FIRST. Operator intervention only after automated acquisition
fails. Does not call acquire_symbol (that pulls today's Screener cache).
Does not re-download an identical source identity / content hash.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from product.pit_ingest import harvest_symbol
from product.pit_warehouse import persist_artifact

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "logs" / "product" / "pit_backfill_state.json"
EVIDENCE_ROOT = ROOT / "logs" / "research_evidence"

WALK_FORWARD_UNIVERSE = (
    "INFY", "TCS", "RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "ITC",
    "BHARTIARTL", "HINDUNILVR", "ASIANPAINT", "MARUTI", "TITAN",
    "SUNPHARMA", "DRREDDY", "TATASTEEL", "JSWSTEEL", "NTPC", "ONGC",
    "POWERGRID", "COALINDIA", "BAJFINANCE", "KOTAKBANK", "AXISBANK", "LT",
)

STAGE_WALK_FORWARD = "walk_forward_24"
STAGE_CANDIDATES = "candidates"
STAGE_LIQUID = "liquid"
STAGE_BROADER = "broader"

REASON_NOT_FOUND = "NOT_FOUND"
REASON_HTTP_FAILURE = "HTTP_FAILURE"
REASON_RATE_LIMITED = "RATE_LIMITED"
REASON_PARSER_FAILED = "PARSER_FAILED"
REASON_PUBLICATION_DATE_UNKNOWN = "PUBLICATION_DATE_UNKNOWN"
REASON_CORRUPT_DOCUMENT = "CORRUPT_DOCUMENT"
REASON_ACCESS_BLOCKED = "ACCESS_BLOCKED"
REASON_SOURCE_CONFLICT = "SOURCE_CONFLICT"

LANES = ("announcements", "results")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _reason_from_error(error: str) -> str:
    text = str(error or "").lower()
    if "403" in text or "401" in text or "blocked" in text:
        return REASON_ACCESS_BLOCKED
    if "429" in text or "rate" in text:
        return REASON_RATE_LIMITED
    if "http" in text or "timeout" in text or "connect" in text:
        return REASON_HTTP_FAILURE
    if "not on the official" in text:
        return REASON_ACCESS_BLOCKED
    return REASON_HTTP_FAILURE


def _already_have(symbol: str, name: str) -> bool:
    path = EVIDENCE_ROOT / symbol.upper() / "autonomy" / name
    if not path.exists():
        return False
    try:
        return path.stat().st_size > 2
    except OSError:
        return False


def _fetch_lane(symbol: str, lane: str, session) -> dict[str, Any]:
    from product.due_diligence.acquire import _download

    if lane == "announcements":
        url = f"https://www.nseindia.com/api/corporate-announcements?index=equities&symbol={symbol}"
        name = "nse_0.json"
    elif lane == "results":
        url = f"https://www.nseindia.com/api/corporates-financial-results?index=equities&symbol={symbol}"
        name = "nse_1.json"
    else:
        return {"ok": False, "reason": REASON_NOT_FOUND, "lane": lane}
    if _already_have(symbol, name):
        return {"ok": True, "skipped": True, "reason": "already_on_disk", "lane": lane, "path": name}
    item = _download(session, url, symbol=symbol, name=name)
    if not item.get("ok"):
        return {
            "ok": False,
            "lane": lane,
            "url": url,
            "reason": _reason_from_error(str(item.get("error") or "")),
            "error": item.get("error"),
        }
    persist_artifact({
        "symbol": symbol,
        "source_url": url,
        "local_path": item.get("path"),
        "bytes": item.get("bytes"),
        "document_type": "EXCHANGE_JSON",
        "content_sha": f"{name}:{item.get('bytes')}",
        "parser_version": "pit_backfill.v1",
    })
    return {"ok": True, "lane": lane, "path": item.get("path"), "bytes": item.get("bytes")}


def stage_universe(stage: str, extra: Sequence[str] | None = None) -> list[str]:
    if stage == STAGE_WALK_FORWARD:
        return list(WALK_FORWARD_UNIVERSE)
    if extra:
        return [str(s).upper() for s in extra if s]
    return list(WALK_FORWARD_UNIVERSE)


def backfill_symbol(
    symbol: str,
    *,
    session=None,
    warehouse_path=None,
    sleep_s: float = 0.0,
) -> dict[str, Any]:
    """Discover → retrieve → dedup → persist raw → parse → index."""
    from product.due_diligence.acquire import _nse_session

    name = str(symbol or "").upper()
    report = {
        "symbol": name,
        "attempted": 0,
        "acquired": 0,
        "parsed": 0,
        "failed": 0,
        "unavailable": 0,
        "skipped": 0,
        "reasons": {},
        "lanes": [],
    }
    sess = session or _nse_session()
    for lane in LANES:
        report["attempted"] += 1
        item = _fetch_lane(name, lane, sess)
        report["lanes"].append(item)
        if item.get("skipped"):
            report["skipped"] += 1
        elif item.get("ok"):
            report["acquired"] += 1
        else:
            report["failed"] += 1
            reason = str(item.get("reason") or REASON_HTTP_FAILURE)
            report["reasons"][reason] = report["reasons"].get(reason, 0) + 1
        if sleep_s:
            time.sleep(float(sleep_s))
    harvest = harvest_symbol(name, warehouse_path=warehouse_path)
    report["parsed"] = int(harvest.get("parsed") or 0)
    report["unverified"] = int(harvest.get("unverified") or 0)
    report["harvest"] = {k: harvest.get(k) for k in (
        "attempted", "acquired", "parsed", "unverified", "deduped", "unavailable",
    )}
    if harvest.get("unavailable"):
        report["unavailable"] += 1
        report["reasons"][REASON_NOT_FOUND] = report["reasons"].get(REASON_NOT_FOUND, 0) + 1
    return report


def backfill(
    *,
    stage: str = STAGE_WALK_FORWARD,
    symbols: Sequence[str] | None = None,
    resume: bool = True,
    limit: int | None = None,
    sleep_s: float = 0.8,
    warehouse_path=None,
    state_path: Path | None = None,
) -> dict[str, Any]:
    names = stage_universe(stage, symbols)
    if limit:
        names = names[: int(limit)]
    state_file = state_path or STATE_PATH
    state = _read_json(state_file) if resume else {}
    done = set(state.get("completed") or [])
    session = None
    details = []
    started = _now()
    for name in names:
        if name in done:
            details.append({"symbol": name, "skipped": True, "reason": "resume"})
            continue
        try:
            if session is None:
                from product.due_diligence.acquire import _nse_session
                session = _nse_session()
            row = backfill_symbol(name, session=session, warehouse_path=warehouse_path, sleep_s=sleep_s)
        except Exception as exc:
            row = {
                "symbol": name, "ok": False, "failed": 1,
                "reasons": {REASON_HTTP_FAILURE: 1}, "error": str(exc)[:240],
            }
        details.append(row)
        if row.get("acquired") or row.get("parsed") or row.get("skipped"):
            done.add(name)
        state = {
            "stage": stage,
            "completed": sorted(done),
            "updated_at": _now(),
            "last_symbol": name,
        }
        _write_json(state_file, state)
        reasons = row.get("reasons") or {}
        if reasons.get(REASON_RATE_LIMITED) or reasons.get(REASON_ACCESS_BLOCKED):
            time.sleep(max(sleep_s, 2.0))
    return {
        "stage": stage,
        "started_at": started,
        "finished_at": _now(),
        "universe": names,
        "n": len(names),
        "completed": sorted(done),
        "attempted": sum(int(r.get("attempted") or 0) for r in details),
        "acquired": sum(int(r.get("acquired") or 0) for r in details),
        "parsed": sum(int(r.get("parsed") or 0) for r in details),
        "failed": sum(int(r.get("failed") or 0) for r in details),
        "skipped": sum(int(r.get("skipped") or 0) for r in details),
        "details": details,
        "note": (
            "NSE JSON only. Screener was not fetched. "
            "Resume skips symbols already marked complete."
        ),
    }


def _parse_nse_stamp(value: Any) -> str:
    from product.pit_ingest import _parse_nse_date

    return _parse_nse_date(value)


def _persist_parsed_xbrl(
    symbol: str,
    *,
    xml_text: str,
    publication: str,
    period_end: str,
    source_url: str,
    source_identity: str,
    warehouse_path=None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    from product.pit_warehouse import DOC_QUARTERLY_RESULT, persist
    from product.pit_xbrl import PARSER_VERSION, parse_xbrl

    parsed = parse_xbrl(xml_text)
    meta = dict(extra or {})
    pub = publication or parsed.get("board_date") or ""
    return persist({
        "symbol": symbol,
        "evidence_type": DOC_QUARTERLY_RESULT,
        "document_type": DOC_QUARTERLY_RESULT,
        "period_start": parsed.get("period_start") or meta.get("period_start"),
        "period_end": parsed.get("period_end") or period_end,
        "publication_date": pub,
        "filing_date": pub,
        "exchange_timestamp": meta.get("broadcast") or pub,
        "available_from": pub,
        "source": "NSE XBRL",
        "source_url": source_url,
        "source_identity": source_identity,
        "parser_version": PARSER_VERSION,
        "extracted": {
            **parsed,
            "numbers_parsed": bool(parsed.get("numbers_parsed")),
            "consolidated": meta.get("consolidated"),
            "audited": meta.get("audited") or parsed.get("audited"),
        },
        "pit_status": "INDEXED" if pub and parsed.get("numbers_parsed") else (
            "PIT_UNVERIFIED" if not pub else "INDEXED"
        ),
        "reason_code": (
            "" if pub and parsed.get("numbers_parsed") else
            parsed.get("reason_code") or "PUBLICATION_DATE_UNKNOWN"
        ),
        "revision": 2 if meta.get("revised") else 1,
    }, path=warehouse_path)


def backfill_structured_financials(
    symbol: str,
    *,
    session=None,
    warehouse_path=None,
    max_xbrl: int = 8,
    sleep_s: float = 0.4,
) -> dict[str, Any]:
    """Official Integrated Filing + new-format quarterly XBRL. No Screener."""
    from product.due_diligence.acquire import _nse_session
    from product.pit_warehouse import persist_artifact

    name = str(symbol).upper()
    sess = session or _nse_session()
    report = {
        "symbol": name, "attempted": 0, "acquired": 0, "parsed": 0,
        "failed": 0, "skipped": 0, "reasons": {},
    }
    rows: list[dict[str, Any]] = []
    integrated = sess.get(
        "https://www.nseindia.com/api/integrated-filing-results",
        params={
            "index": "equities",
            "symbol": name,
            "period_ended": "all",
            "type": "Integrated Filing- Financials",
            "page": 1,
            "size": 50,
        },
        timeout=25,
    )
    if integrated.status_code == 200:
        payload = integrated.json() if integrated.content else {}
        for row in list((payload or {}).get("data") or []):
            if not isinstance(row, dict):
                continue
            rows.append({
                "xbrl": row.get("xbrl"),
                "publication": _parse_nse_stamp(row.get("broadcast_Date") or row.get("creation_Date")),
                "period_end": _parse_nse_stamp(row.get("qe_Date")),
                "consolidated": row.get("consolidated"),
                "audited": row.get("audited"),
                "seq": row.get("seq_Id"),
                "revised": bool(row.get("revised_Date")),
                "kind": "integrated",
            })
    else:
        report["reasons"][_reason_from_error(f"HTTP {integrated.status_code}")] = 1

    quarterly = sess.get(
        f"https://www.nseindia.com/api/corporates-financial-results?index=equities&symbol={name}&period=Quarterly",
        timeout=25,
    )
    if quarterly.status_code == 200:
        qrows = quarterly.json() if quarterly.content else []
        if isinstance(qrows, dict):
            qrows = qrows.get("data") or qrows.get("financialResults") or []
        for row in list(qrows or []):
            if not isinstance(row, dict):
                continue
            xbrl = str(row.get("xbrl") or "")
            if not xbrl or xbrl.endswith("xbrl/-"):
                continue
            rows.append({
                "xbrl": xbrl,
                "publication": _parse_nse_stamp(row.get("filingDate") or row.get("broadCastDate")),
                "period_end": _parse_nse_stamp(row.get("toDate")),
                "period_start": _parse_nse_stamp(row.get("fromDate")),
                "consolidated": row.get("consolidated"),
                "audited": row.get("audited"),
                "seq": row.get("seqNumber"),
                "revised": False,
                "kind": "quarterly",
            })

    # Newest publication first; integrated + consolidated win ties.
    def _rank(item: dict[str, Any]) -> tuple:
        consol = str(item.get("consolidated") or "").lower()
        pub = str(item.get("publication") or "")
        return (
            pub,
            1 if item.get("kind") == "integrated" else 0,
            1 if consol.startswith("consol") else 0,
        )

    seen_url = set()
    picked = []
    for item in sorted(rows, key=_rank, reverse=True):
        url = str(item.get("xbrl") or "")
        if not url or url in seen_url:
            continue
        seen_url.add(url)
        picked.append(item)
        if len(picked) >= int(max_xbrl):
            break

    from product.due_diligence.acquire import _download

    for item in picked:
        report["attempted"] += 1
        url = str(item["xbrl"])
        seq = str(item.get("seq") or url.rsplit("/", 1)[-1])
        fname = f"nse_xbrl_{item.get('kind')}_{seq}.xml"
        if _already_have(name, fname):
            xml_text = (EVIDENCE_ROOT / name / "autonomy" / fname).read_text(encoding="utf-8", errors="ignore")
            report["skipped"] += 1
        else:
            downloaded = _download(sess, url, symbol=name, name=fname)
            if not downloaded.get("ok"):
                report["failed"] += 1
                reason = _reason_from_error(str(downloaded.get("error") or ""))
                report["reasons"][reason] = report["reasons"].get(reason, 0) + 1
                continue
            report["acquired"] += 1
            persist_artifact({
                "symbol": name,
                "source_url": url,
                "local_path": downloaded.get("path"),
                "bytes": downloaded.get("bytes"),
                "document_type": "XBRL",
                "parser_version": "pit_xbrl.v1",
            })
            xml_text = Path(ROOT / str(downloaded["path"])).read_text(encoding="utf-8", errors="ignore")
        stored = _persist_parsed_xbrl(
            name,
            xml_text=xml_text,
            publication=str(item.get("publication") or ""),
            period_end=str(item.get("period_end") or ""),
            source_url=url,
            source_identity=f"nse_xbrl:{item.get('kind')}:{seq}",
            warehouse_path=warehouse_path,
            extra=item,
        )
        if (stored.get("extracted") or stored).get("numbers_parsed") or stored.get("pit_status") == "INDEXED":
            report["parsed"] += 1
        if sleep_s:
            time.sleep(float(sleep_s))
    return report


def consume_data_debt(
    *,
    symbols: Sequence[str] | None = None,
    limit: int = 2,
    entry_window: bool = False,
    warehouse_path=None,
) -> dict[str, Any]:
    """Bounded off-session work. Never runs inside the live entry window.

    Prefers structured XBRL (the remaining high-value debt) over
    re-fetching announcement metadata.
    """
    if entry_window:
        return {"skipped": True, "reason": "entry_window", "acquired": 0}
    names = [str(s).upper() for s in (symbols or WALK_FORWARD_UNIVERSE)][: max(1, int(limit))]
    reports = []
    for name in names:
        reports.append(
            backfill_structured_financials(
                name,
                max_xbrl=4,
                sleep_s=1.0,
                warehouse_path=warehouse_path,
            )
        )
    return {
        "mode": "structured_financials",
        "n": len(names),
        "acquired": sum(int(r.get("acquired") or 0) for r in reports),
        "parsed": sum(int(r.get("parsed") or 0) for r in reports),
        "failed": sum(int(r.get("failed") or 0) for r in reports),
        "details": reports,
    }
