"""Scheduler-safe post-session report job.

The OS service manager may call this every 15 minutes.  The job itself decides
in IST whether a report is due and de-duplicates by IST calendar date, so host
timezone does not matter.  It never manufactures a green result: the underlying
daily report exit code is persisted and returned on the first run for the day.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, time
from pathlib import Path
from typing import Any, Mapping

from core.market_clock import now_ist
from core.runtime_paths import logs_path, runtime_path

MARKER_PATH = Path("state") / "daily_report_job.json"
DUE_TIME_IST = time(16, 10)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def due_now(now: datetime | None = None, *, force: bool = False) -> tuple[bool, str]:
    if force:
        return True, "forced"
    now = now or now_ist()
    if now.weekday() >= 5:
        return False, "weekend"
    if now.timetz().replace(tzinfo=None) < DUE_TIME_IST:
        return False, f"before {DUE_TIME_IST.strftime('%H:%M')} IST"
    marker = _read_json(runtime_path(MARKER_PATH))
    if str(marker.get("ist_date") or "") == now.date().isoformat() and marker.get("report_written"):
        return False, "already generated for this IST date"
    return True, "post-session report due"


def _persist_text(report: Mapping[str, Any], text: str, *, ist_date: str) -> Path:
    target = logs_path("product", "operating_reports", f"{ist_date}.txt")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".txt.tmp")
    tmp.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
    os.replace(tmp, target)
    return target


def run_once(*, now: datetime | None = None, force: bool = False) -> dict[str, Any]:
    now = now or now_ist()
    should_run, reason = due_now(now, force=force)
    if not should_run:
        return {"ran": False, "reason": reason, "ist_date": now.date().isoformat(), "exit_code": 0}

    from product.daily_operating_report import (
        build_daily_operating_report,
        render_text,
        unmet_operating_proofs,
        write_daily_operating_report,
    )

    report = build_daily_operating_report()
    text = render_text(report)
    json_path = write_daily_operating_report(report)
    text_path = _persist_text(report, text, ist_date=now.date().isoformat())

    capital = report.get("capital_safety") or {}
    findings = report.get("silent_wrongness") or {}
    if not capital.get("live_locked") or int(capital.get("broker_mutations") or 0):
        exit_code = 2
    elif findings.get("available") and findings.get("critical"):
        exit_code = 2
    else:
        exit_code = 1 if unmet_operating_proofs(report) else 0

    alert = {"attempted": False, "delivered": False, "channel": "", "detail": "not required"}
    if exit_code == 2:
        try:
            from product.host_alerts import send_operational_alert

            alert = send_operational_alert(
                "QuantTerm CRITICAL operating report\n"
                f"SHA: {report.get('production_sha') or 'UNKNOWN'}\n"
                f"IST date: {now.date().isoformat()}\n"
                "A capital-safety or silent-wrongness proof failed. Read the persisted daily report."
            ).to_dict()
        except Exception as exc:
            alert = {"attempted": True, "delivered": False, "channel": "",
                     "detail": f"{type(exc).__name__}: {exc}"[:200]}

    marker = {
        "schema_version": 1,
        "ist_date": now.date().isoformat(),
        "generated_at_ist": now.isoformat(),
        "report_written": True,
        "json_path": str(json_path),
        "text_path": str(text_path),
        "exit_code": exit_code,
        "production_sha": report.get("production_sha") or "",
        "reason": reason,
        "alert": alert,
    }
    _atomic_json(runtime_path(MARKER_PATH), marker)
    return {"ran": True, **marker}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the post-session report once per IST date")
    parser.add_argument("--force", action="store_true", help="run even before the normal post-session time")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = run_once(force=args.force)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    elif result.get("ran"):
        print(f"daily operating report written: {result.get('json_path')} · exit={result.get('exit_code')}")
    else:
        print(f"daily operating report not due: {result.get('reason')}")
    return int(result.get("exit_code") or 0)


if __name__ == "__main__":
    raise SystemExit(main())
