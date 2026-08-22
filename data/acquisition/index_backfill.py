"""Download missing official NSE index daily CSVs (ingest stage)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Any

from data.acquisition.cache import write_manifest
from data.acquisition.http import HEADERS, get_bytes
from data.index_store import _DIR, _URL, _day_path

START_DEFAULT = date(2015, 1, 1)
END_LOCAL = date(2024, 4, 7)  # local store already has 2024-04-08+


def _weekdays(start: date, end: date) -> list[date]:
    out = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def backfill(*, start: date = START_DEFAULT, end: date = END_LOCAL, workers: int = 8) -> dict[str, Any]:
    import requests

    sess = requests.Session()
    sess.headers.update(HEADERS)
    days = [d for d in _weekdays(start, end) if not _day_path(d).exists()]
    ok = 0
    missing = 0
    failed = 0

    def _one(d: date) -> str:
        url = _URL.format(d=d.strftime("%d%m%Y"))
        blob, meta = get_bytes(url, session=sess, timeout=20, retries=1, sleep=0.2)
        if meta.get("http_status") == 404 or not blob:
            return "missing"
        if len(blob) < 400:
            return "missing"
        _DIR.mkdir(parents=True, exist_ok=True)
        _day_path(d).write_bytes(blob)
        return "ok"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one, d): d for d in days}
        for fut in as_completed(futs):
            try:
                st = fut.result()
            except Exception:
                st = "failed"
            if st == "ok":
                ok += 1
            elif st == "missing":
                missing += 1
            else:
                failed += 1

    man = {
        "source": "nse_ind_close_all",
        "requested_range": [str(start), str(end)],
        "attempted": len(days),
        "successful_objects": ok,
        "holiday_or_absent": missing,
        "failed_objects": failed,
        "already_on_disk_skipped": "existing files not re-downloaded",
    }
    write_manifest("index_backfill", man)
    return man
