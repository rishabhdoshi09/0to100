"""Machine-readable FEATURE-002 / production-scan logging health.

Does not change ranks, graduation, or production decisions.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.feature002.constants import (
    DB_PATH,
    FEATURE_SET_VERSION,
    FORWARD_START_DATE,
    FORWARD_START_TS_IST,
    LEDGER_DIR,
    OUT_DIR,
    PRIMARY_SOURCE,
    UNTIL_MATURE,
    protocol_hash,
)

SCAN_STATE = Path(__file__).resolve().parents[2] / "logs" / "scan_store.json"
PRODUCT_SCAN = Path(__file__).resolve().parents[2] / "logs" / "product" / "latest_momentum_scan.json"
HOOK_LOG = LEDGER_DIR / "hook_log.jsonl"
HEALTH_LOG_PATH = LEDGER_DIR / "research_logging_health.json"
HEALTH_DOC_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "data_program" / "research_logging_health.json"
)
STATUS_MD = OUT_DIR / "FEATURE_002_STATUS.md"
STATUS_MD_PROGRAM = (
    Path(__file__).resolve().parents[2] / "docs" / "data_program" / "FEATURE_002_STATUS.md"
)


def _ist_now() -> str:
    try:
        from core.market_clock import now_ist
        return now_ist().isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def _today_ist() -> str:
    try:
        from core.market_clock import today_ist
        return today_ist().isoformat()
    except Exception:
        return datetime.now().date().isoformat()


def _unix_to_ist_iso(ts: float | None) -> str | None:
    if not ts:
        return None
    try:
        from core.market_clock import IST
        return datetime.fromtimestamp(float(ts), tz=IST).isoformat()
    except Exception:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None


def _hook_events(limit: int = 400) -> list[dict[str, Any]]:
    if not HOOK_LOG.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        lines = HOOK_LOG.read_text(encoding="utf-8").splitlines()
        for line in lines[-limit:]:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                out.append(row)
    except Exception:
        return []
    return out


def _ledger_stats(path: Path | None = None) -> dict[str, Any]:
    db = path or DB_PATH
    empty = {
        "ledger_exists": db.exists(),
        "ledger_path": str(db),
        "observations": 0,
        "primary": 0,
        "candidate_sets": 0,
        "outcomes": 0,
        "resolved_outcomes": 0,
        "unresolved_outcomes": 0,
        "duplicates_suppressed": 0,
        "pre_freeze_refused": 0,
        "latest_observation_ts": None,
        "observations_today": 0,
        "candidate_sets_today": 0,
        "primary_today": 0,
        "by_source": {},
        "corrupt": False,
        "corrupt_reason": None,
        "feature_versions": [],
        "protocol_hashes": [],
    }
    if not db.exists():
        return empty
    try:
        c = sqlite3.connect(str(db), timeout=10)
        c.row_factory = sqlite3.Row
        today = _today_ist()
        n_obs = c.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        n_pri = c.execute(
            "SELECT COUNT(*) FROM observations WHERE eligible_primary=1"
        ).fetchone()[0]
        n_sets = c.execute("SELECT COUNT(*) FROM candidate_sets").fetchone()[0]
        n_out = c.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
        n_res = c.execute(
            "SELECT COUNT(*) FROM outcomes WHERE ret_5d IS NOT NULL"
        ).fetchone()[0]
        latest = c.execute(
            "SELECT MAX(recorded_at) FROM observations"
        ).fetchone()[0]
        n_today = c.execute(
            "SELECT COUNT(*) FROM observations WHERE session_date=?", (today,)
        ).fetchone()[0]
        n_sets_today = c.execute(
            "SELECT COUNT(*) FROM candidate_sets WHERE session_date=?", (today,)
        ).fetchone()[0]
        n_pri_today = c.execute(
            "SELECT COUNT(*) FROM observations WHERE eligible_primary=1 AND session_date=?",
            (today,),
        ).fetchone()[0]
        by_src = {
            str(r[0]): int(r[1])
            for r in c.execute(
                "SELECT source, COUNT(*) FROM observations GROUP BY source"
            )
        }
        versions = [str(r[0]) for r in c.execute(
            "SELECT DISTINCT feature_set_version FROM observations"
        )]
        hashes = [str(r[0]) for r in c.execute(
            "SELECT DISTINCT protocol_hash FROM observations"
        ) if r[0]]
        c.close()
        empty.update({
            "observations": int(n_obs),
            "primary": int(n_pri),
            "candidate_sets": int(n_sets),
            "outcomes": int(n_out),
            "resolved_outcomes": int(n_res),
            "unresolved_outcomes": max(int(n_obs) - int(n_res), 0),
            "latest_observation_ts": latest,
            "observations_today": int(n_today),
            "candidate_sets_today": int(n_sets_today),
            "primary_today": int(n_pri_today),
            "by_source": by_src,
            "feature_versions": versions,
            "protocol_hashes": hashes,
        })
        return empty
    except Exception as exc:
        empty["corrupt"] = True
        empty["corrupt_reason"] = str(exc)
        return empty


def _iso_to_ist_iso(value: str | None) -> str | None:
    if not value:
        return None
    try:
        from core.market_clock import IST
        raw = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(IST).isoformat()
    except Exception:
        return str(value)


def _scan_state() -> dict[str, Any]:
    """Either auto_scan store or the retail/autonomy product scan is a receipt."""
    auto = _load_json(SCAN_STATE)
    product = _load_json(PRODUCT_SCAN)
    auto_ts = (auto or {}).get("ts")
    auto_ist = _unix_to_ist_iso(auto_ts)
    product_ist = _iso_to_ist_iso((product or {}).get("scanned_at"))
    candidates = []
    if auto:
        candidates.append((str(auto_ist or ""), {
            "scan_store_exists": True,
            "latest_production_scan_ts_unix": auto_ts,
            "latest_production_scan_ts_ist": auto_ist,
            "n_results": len(auto.get("results") or []),
            "scanned_count": auto.get("count") or 0,
            "scan_artifact": "scan_store.json",
        }))
    if product:
        records = product.get("records") or []
        candidates.append((str(product_ist or ""), {
            "scan_store_exists": True,
            "latest_production_scan_ts_unix": None,
            "latest_production_scan_ts_ist": product_ist,
            "n_results": len(records),
            "scanned_count": product.get("universe_size") or len(records),
            "scan_artifact": "latest_momentum_scan.json",
        }))
    if not candidates:
        return {
            "scan_store_exists": False,
            "latest_production_scan_ts_unix": None,
            "latest_production_scan_ts_ist": None,
            "n_results": 0,
            "scanned_count": 0,
            "scan_artifact": None,
        }
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _clock_ok() -> dict[str, Any]:
    try:
        from core.market_clock import now_ist, today_ist, IST
        now = now_ist()
        return {
            "ok": now.tzinfo is not None and str(now.tzinfo) in {"IST", "Asia/Kolkata"}
            or getattr(now.tzinfo, "key", None) == "Asia/Kolkata",
            "now_ist": now.isoformat(),
            "today_ist": today_ist().isoformat(),
            "tz": str(IST),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def classify_empty_primary(scan: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    """Is zero primary a bug (scans after activation, no rows) or expected?"""
    latest = scan.get("latest_production_scan_ts_ist")
    n_pri = int(ledger.get("primary") or 0)
    if n_pri > 0:
        return {
            "is_bug": False,
            "reason": "primary_rows_present",
            "severity": "info",
        }
    if not scan.get("scan_store_exists") or not latest:
        return {
            "is_bug": False,
            "reason": "no_post_activation_production_scan",
            "severity": "info",
            "detail": (
                "No scan_store.json / latest_momentum_scan.json (or no timestamp). "
                f"Protocol activated {FORWARD_START_TS_IST}. Empty primary ledger "
                "is expected until a market-hours production scan runs after that instant."
            ),
        }
    if str(latest) < FORWARD_START_TS_IST:
        return {
            "is_bug": False,
            "reason": "latest_scan_before_protocol_ist",
            "severity": "info",
            "detail": f"latest scan {latest} is before {FORWARD_START_TS_IST}",
        }
    if int(scan.get("n_results") or 0) == 0:
        return {
            "is_bug": False,
            "reason": "production_scan_empty_candidate_set",
            "severity": "warn",
        }
    return {
        "is_bug": True,
        "reason": "production_scans_after_activation_but_no_primary_rows",
        "severity": "error",
        "detail": (
            "Valid production scan cards exist after protocol IST but the "
            "FEATURE-002 ledger has zero eligible_primary rows. Logging bug — "
            "do not change the experiment specification."
        ),
    }


def build_health(*, ledger_path: Path | None = None) -> dict[str, Any]:
    from research.feature002.evaluate import summarize

    scan = _scan_state()
    ledger = _ledger_stats(ledger_path)
    hooks = _hook_events()
    today = _today_ist()
    rejected: list[dict[str, Any]] = []
    exceptions: list[dict[str, Any]] = []
    duplicates = 0
    hook_today = 0
    for ev in hooks:
        kind = str(ev.get("kind") or "")
        src = str(ev.get("source") or "")
        ts = str(ev.get("ts") or "")
        # Implementation-test / unit-test receipts are not production health.
        if src in {"implementation_test", "synthetic", "replay"}:
            continue
        if ts.startswith(today) or str(ev.get("session_date") or "") == today:
            hook_today += 1
        if kind in {"persist_result", "hook_skipped", "pre_freeze_session"}:
            if ev.get("pre_freeze_refused"):
                rejected.append({"reason": "pre_freeze_refused", "n": ev.get("pre_freeze_refused")})
            if ev.get("skipped"):
                rejected.append({"reason": str(ev.get("skipped")), "event": ev})
            duplicates += int(ev.get("exists") or 0)
        if kind in {"observe_failed", "persist_failed", "hook_exception"}:
            exceptions.append(ev)
        if kind == "hook_skipped":
            rejected.append({"reason": ev.get("reason") or "hook_skipped", "event": ev})

    empty = classify_empty_primary(scan, ledger)
    clock = _clock_ok()
    summary = summarize(path=ledger_path)
    mat = summary.get("maturity") or {}

    reasons: dict[str, int] = {}
    for row in rejected:
        key = str(row.get("reason") or "unknown")
        reasons[key] = reasons.get(key, 0) + int(row.get("n") or 1)

    health = {
        "schema_version": 1,
        "generated_at": _ist_now(),
        "latest_production_scan_timestamp": scan.get("latest_production_scan_ts_ist"),
        "latest_feature002_observation_timestamp": ledger.get("latest_observation_ts"),
        "observations_today": ledger.get("observations_today"),
        "candidate_sets_today": ledger.get("candidate_sets_today"),
        "primary_today": ledger.get("primary_today"),
        "rejected_rows": sum(reasons.values()),
        "rejection_reasons": reasons,
        "duplicate_count": duplicates + int(ledger.get("duplicates_suppressed") or 0),
        "logging_exceptions": exceptions[-20:],
        "logging_exception_count": len(exceptions),
        "unresolved_outcomes": ledger.get("unresolved_outcomes"),
        "resolved_outcomes": ledger.get("resolved_outcomes"),
        "feature_version": FEATURE_SET_VERSION,
        "protocol_version": protocol_hash(),
        "protocol_activation_ist": FORWARD_START_TS_IST,
        "forward_start_date": FORWARD_START_DATE,
        "primary_source": PRIMARY_SOURCE,
        "ledger_path": ledger.get("ledger_path"),
        "ledger_version": "feature002.shadow.sqlite.v1",
        "ledger": ledger,
        "scan_state": scan,
        "hook_events_today": hook_today,
        "clock": clock,
        "empty_primary": empty,
        "status": summary.get("status") or UNTIL_MATURE,
        "maturity": {
            "stage": mat.get("stage"),
            "n_primary": mat.get("n_primary"),
            "n_resolved_5d": mat.get("n_resolved_5d"),
            "n_months": mat.get("n_months"),
            "decision_capable": mat.get("decision_capable"),
        },
        "user_summary": _user_summary(scan, ledger, empty, summary),
    }
    try:
        from research.feature002.acceptance import operational_state
        health["operational"] = operational_state(health)
    except Exception:
        health["operational"] = {
            "operational_state": "NO_POST_ACTIVATION_SCAN",
            "research_maturity": health.get("status"),
        }
    return health


def _user_summary(
    scan: dict[str, Any],
    ledger: dict[str, Any],
    empty: dict[str, Any],
    summary: dict[str, Any],
) -> str:
    status = summary.get("status") or UNTIL_MATURE
    if empty.get("is_bug"):
        return (
            f"FEATURE-002 logging bug: production scans exist after protocol "
            f"activation but primary live_scan rows are still {ledger.get('primary', 0)}. "
            f"Experiment status remains: {status}."
        )
    if int(ledger.get("primary") or 0) == 0:
        return (
            f"FEATURE-002 is active and isolated. Primary live_scan rows: 0. "
            f"Reason: {empty.get('reason')}. {status}."
        )
    return (
        f"FEATURE-002 collecting. Primary rows: {ledger.get('primary')}. "
        f"Resolved 5d: {ledger.get('resolved_outcomes')}. {status}."
    )


def write_health(*, ledger_path: Path | None = None, dest: Path | None = None) -> dict[str, Any]:
    health = build_health(ledger_path=ledger_path)
    blob = json.dumps(health, indent=2, default=str)
    targets = [HEALTH_LOG_PATH]
    if dest is not None:
        targets.append(Path(dest))
    else:
        targets.append(HEALTH_DOC_PATH)
    for p in targets:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(blob, encoding="utf-8")
        tmp.replace(p)
    return health


def write_status_md(health: dict[str, Any] | None = None, *, ledger_path: Path | None = None) -> dict[str, str]:
    health = health or build_health(ledger_path=ledger_path)
    mat = health.get("maturity") or {}
    empty = health.get("empty_primary") or {}
    scan = health.get("scan_state") or {}
    try:
        from core.market_clock import today_ist
        today = today_ist()
    except Exception:
        today = datetime.now().date()
    from datetime import date as _date
    try:
        start = _date.fromisoformat(FORWARD_START_TS_IST[:10])
        days = max((today - start).days, 0)
    except Exception:
        days = 0
    body = f"""# FEATURE-002 status

**{health.get('status') or UNTIL_MATURE}**

Do not peek at immature rank comparisons. Do not retune R3 weights.

| Item | Value |
|---|---|
| Days since protocol activation ({FORWARD_START_TS_IST[:10]}) | {days} |
| New primary live_scan rows | {mat.get('n_primary', 0)} |
| Resolved primary 5d rows | {mat.get('n_resolved_5d', 0)} |
| Candidate sets (all sources) | {(health.get('ledger') or {}).get('candidate_sets', 0)} |
| Candidate sets today | {health.get('candidate_sets_today', 0)} |
| Multi-candidate resolved sets | see maturity after INTERIM |
| Family coverage | withheld until DECISION-CAPABLE / n≥100 per family |
| Months observed (resolved) | {mat.get('n_months', 0)} |
| Feature version | `{health.get('feature_version')}` |
| Protocol hash | `{health.get('protocol_version')}` |
| Graduation gates | 2000 resolved · 250 multi-sets · 100/family · 6 months |
| Current classification | `{health.get('status')}` |
| Empty-primary reason | `{empty.get('reason')}` |
| Logging bug? | {'YES' if empty.get('is_bug') else 'no'} |
| Latest production scan (IST) | {scan.get('latest_production_scan_ts_ist') or 'none on disk'} |
| Latest shadow observation | {health.get('latest_feature002_observation_timestamp') or 'none'} |

## Operator summary

{health.get('user_summary')}

Until gates are reached the only allowed classification is:

`{UNTIL_MATURE}`
"""
    written: dict[str, str] = {}
    for p in (STATUS_MD, STATUS_MD_PROGRAM):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body.rstrip() + "\n", encoding="utf-8")
        written[str(p)] = "ok"
    return written
