"""Prospective shadow test of the pre-registered ranking challenger.

The walk-forward (scan/signal_backtest.confirmation_report) is historical
out-of-sample evidence. This is the forward half, and it is only evidence if
it is genuinely prospective:

* FROZEN BEFORE ENTRY. A session is frozen only while the first eligible
  forward entry (the next NSE session's 09:15 IST open, the evaluation
  rulebook's entry) is still in the future, from the scan the desk actually
  produced for that session. The scan must be the live session-close scan of
  the latest completed NSE session (not a historical replay, not stale input,
  identity present, completed after that session's close). Anything else
  freezes nothing. A historical observation can never be written as
  prospective: the deadline is checked when the row is written AND again
  when it is evaluated.
* ACTUAL PREDICTIONS + MODEL VERSION. Each row stores the champion's real
  output (scanner score, its rank in the scan, its verdict) and the
  challenger's composite score and rank, plus the hypothesis, the challenger
  feature-spec hash, the running build SHA and the scan identity. A content
  hash over all of it is re-verified before evaluation.
* DURABLE COMPLETION, CRASH-SAFE, IDEMPOTENT. One SQLite transaction writes
  the session marker and every prediction, or nothing; a second write for the
  same session is refused by the primary key. A session with ZERO candidates
  still gets its marker, so a quiet day is a recorded day, not a gap. A
  session whose deadline passed unfrozen is marked MISSED (never evidence),
  so gaps are visible rather than silent.

Evaluation settles a session only when its window is MATURED
(learning_scorecard.maturity_checker, fail-closed calendar) and measures each
candidate over EXACTLY the first ``h`` official NSE sessions after the
signal: entry at the open of the first, exit at the close of the h-th, and
every one of the h bars must exist for that symbol (a missing symbol bar
excludes it and is counted; a session with too many is excluded). The
decision statistic is PAIRED: challenger IC minus champion IC on the same
candidates, same session, with the overlap-robust inference of
research.overlap_inference. Two separately positive correlations prove
nothing about which ranker is better.

Shadow only: no paper order, no ranking change, no policy write.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import sqlite3
from contextlib import closing
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

SCHEMA_VERSION = 2
HYPOTHESIS = "2026-11-19:low_atr+near_high+not_extended"
LEDGER_FILE = "challenger_shadow.db"
#: Schema-1 append-only JSONL (never deployed). Never read as evidence.
LEGACY_FILE = "challenger_shadow.jsonl"

FROZEN = "FROZEN"
MISSED = "MISSED"
ENTRY_OPEN_HM = (9, 15)          # first eligible forward entry: next session's open (IST)
#: A freeze must land this long BEFORE the entry deadline: a host clock a few
#: minutes slow must not let a write after the real open pass as prospective.
FREEZE_MARGIN = timedelta(minutes=30)
SESSION_CLOSE_HM = (15, 30)
MIN_CANDIDATES_FOR_IC = 5
#: A session where more than this share of frozen candidates lack a bar in
#: their outcome window is excluded: missing bars are rarely random.
MAX_MISSING_BAR_SHARE = 0.20
MISSED_BACKFILL_LIMIT = 60
IST = timezone(timedelta(hours=5, minutes=30))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
    session TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('FROZEN', 'MISSED')),
    reason TEXT NOT NULL DEFAULT '',
    hypothesis TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    frozen_at TEXT NOT NULL,
    entry_deadline TEXT NOT NULL,
    scan_id TEXT NOT NULL DEFAULT '',
    scan_completed_at TEXT NOT NULL DEFAULT '',
    model_version TEXT NOT NULL,
    candidates INTEGER NOT NULL,
    frozen INTEGER NOT NULL,
    exclusions TEXT NOT NULL,
    universe TEXT NOT NULL DEFAULT '{}',
    payload_sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions(
    session TEXT NOT NULL,
    symbol TEXT NOT NULL,
    champion_score REAL,
    champion_rank INTEGER,
    champion_verdict TEXT,
    challenger_score REAL,
    challenger_rank INTEGER,
    low_atr REAL,
    near_high REAL,
    not_extended REAL,
    session_close REAL,
    PRIMARY KEY (session, symbol)
);
"""
_PRED_COLS = ("symbol", "champion_score", "champion_rank", "champion_verdict", "challenger_score",
              "challenger_rank", "low_atr", "near_high", "not_extended", "session_close")


def ledger_path() -> Path:
    from core.runtime_paths import logs_path

    return logs_path("product", LEDGER_FILE)


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30, isolation_level=None)
    conn.execute("PRAGMA synchronous=FULL")
    conn.executescript(_SCHEMA)
    return conn


# ── model identity ───────────────────────────────────────────────────────────

def challenger_spec_sha() -> str:
    """Hash of the challenger's feature definitions and composite rule."""
    from scan.signal_backtest import confirmation_features

    text = inspect.getsource(confirmation_features) + "|composite=sum(average_rank(low_atr,near_high,not_extended))"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def model_version(scan: Mapping[str, Any], horizon: int) -> dict[str, Any]:
    return {
        "hypothesis": HYPOTHESIS,
        "challenger": "composite of average ranks: low_atr + near_high + not_extended",
        "challenger_spec_sha": challenger_spec_sha(),
        "champion": "scan.unified_scanner score (scan record 'score')",
        "champion_calibration": str(scan.get("calibration_snapshot_id") or ""),
        "code_sha": os.environ.get("QT_BUILD_SHA", "").strip(),
        "horizon_sessions": int(horizon),
        "schema_version": SCHEMA_VERSION,
    }


def scan_universe(scan: Mapping[str, Any]) -> dict[str, Any]:
    """The instrument universe the champion actually ranked, frozen with the session."""
    symbols = sorted({str(r.get("symbol") or "").upper() for r in (scan.get("records") or [])
                      if isinstance(r, Mapping) and r.get("symbol")})
    return {"approved_universe": scan.get("approved_universe"), "scanned": scan.get("scanned"),
            "candidates": len(symbols),
            "candidates_sha256": hashlib.sha256("\n".join(symbols).encode("utf-8")).hexdigest()}


def _payload_sha(session: str, model: Mapping[str, Any], scan_id: str, rows: list[Mapping[str, Any]],
                 universe: Mapping[str, Any] | None = None) -> str:
    body = {"session": session, "model": dict(model), "scan_id": scan_id, "universe": dict(universe or {}),
            "rows": [[r.get(c) for c in _PRED_COLS] for r in sorted(rows, key=lambda r: str(r["symbol"]))]}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode("utf-8")).hexdigest()


# ── the prospective gate ─────────────────────────────────────────────────────

def _parse_utc(text: Any) -> datetime | None:
    try:
        stamp = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def entry_deadline(session: date, holidays) -> datetime:
    """First eligible forward entry for a signal on ``session`` (IST-aware)."""
    from research.intelligence.data.nse_calendar import next_session

    return datetime.combine(next_session(session, holidays), time(*ENTRY_OPEN_HM), tzinfo=IST)


def prospective_gate(scan: Mapping[str, Any] | None, *, now: datetime, holidays) -> dict[str, Any]:
    """Whether ``scan`` may be frozen NOW as prospective evidence.

    ``ok`` False with ``retry`` True means "not yet" (wait for the session-
    close scan); ``retry`` False means the session can no longer be frozen.
    """
    from research.intelligence.data import nse_calendar as CAL

    now = now if now.tzinfo else now.replace(tzinfo=IST)
    out: dict[str, Any] = {"ok": False, "retry": True, "reason": "", "session": ""}
    if not isinstance(scan, Mapping) or not isinstance(scan.get("records"), list):
        out["reason"] = "NO_SCAN"
        return out
    if scan.get("pit") or "replay" in str(scan.get("engine") or "").lower():
        # product.historical_replay marks its point-in-time rebuilds pit=True
        out.update(reason="HISTORICAL_REPLAY_IS_NOT_PROSPECTIVE", retry=False)
        return out
    prov = scan.get("provenance") if isinstance(scan.get("provenance"), Mapping) else {}
    scan_id = str(prov.get("scan_id") or "").strip()
    session_text = str(scan.get("as_of_session") or prov.get("market_session_date") or "")[:10]
    out["session"] = session_text
    if not scan_id:
        out["reason"] = "SCAN_IDENTITY_MISSING"
        return out
    try:
        session = date.fromisoformat(session_text)
    except ValueError:
        out["reason"] = "SCAN_SESSION_UNKNOWN"
        return out
    expected = str(prov.get("expected_session_date") or "")[:10]
    if not prov.get("data_current") or (expected and expected != session_text):
        out["reason"] = "SCAN_INPUT_STALE"
        return out
    # The freeze needs the calendar only through the next session (the entry
    # deadline). Evaluation separately waits until the whole window is covered.
    gap = CAL.calendar_gap(session, session, holidays) or (
        CAL.calendar_gap(session, CAL.next_session(session, holidays), holidays))
    if gap:
        out.update(reason=gap, retry=False)
        return out
    if not CAL.is_session(session, holidays):
        out.update(reason="SCAN_SESSION_NOT_AN_NSE_SESSION", retry=False)
        return out
    latest = CAL.completed_session(now.astimezone(IST).replace(tzinfo=None), holidays)
    if session != latest:
        out.update(reason="SCAN_NOT_THE_LATEST_COMPLETED_SESSION", retry=session > latest)
        return out
    deadline = entry_deadline(session, holidays)
    out["entry_deadline"] = deadline.isoformat()
    if now >= deadline - FREEZE_MARGIN:
        out.update(reason="ENTRY_DEADLINE_PASSED", retry=False)
        return out
    completed = _parse_utc(prov.get("scan_completed_at") or scan.get("scanned_at"))
    close_at = datetime.combine(session, time(*SESSION_CLOSE_HM), tzinfo=IST)
    if completed is None or completed > now + timedelta(minutes=5):
        out["reason"] = "SCAN_TIME_INCOHERENT"
        return out
    if completed < close_at:
        out["reason"] = "SCAN_BEFORE_SESSION_CLOSE"        # the session-close rescan will follow
        return out
    out.update(ok=True, retry=False, reason="PROSPECTIVE", scan_id=scan_id,
               scan_completed_at=completed.isoformat())
    return out


# ── freezing ─────────────────────────────────────────────────────────────────

def _champion_ranks(records: list[Mapping[str, Any]]) -> dict[str, int]:
    """The scan's own ordering: score descending, symbol as the stable tiebreak."""
    seen, ordered = set(), []
    for r in sorted(records, key=lambda r: (-float(r.get("score") or 0.0), str(r.get("symbol") or ""))):
        sym = str(r.get("symbol") or "").upper()
        if sym and sym not in seen:
            seen.add(sym)
            ordered.append(sym)
    return {s: i + 1 for i, s in enumerate(ordered)}


def build_predictions(session: str, records: Iterable[Mapping[str, Any]],
                      ohlcv_fn: Callable[[str], Any]) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    """(prediction rows, exclusion counts, candidates) for one session; PIT only."""
    from scan.signal_backtest import _average_ranks, confirmation_features
    import pandas as pd

    records = [r for r in records if isinstance(r, Mapping)]
    ranks = _champion_ranks(records)
    cut = pd.Timestamp(session)
    rows: list[dict[str, Any]] = []
    excluded: dict[str, int] = {}
    seen: set[str] = set()
    for c in records:
        sym = str(c.get("symbol") or "").upper()
        if not sym:
            excluded["NO_SYMBOL"] = excluded.get("NO_SYMBOL", 0) + 1
            continue
        if sym in seen:
            excluded["DUPLICATE_SYMBOL"] = excluded.get("DUPLICATE_SYMBOL", 0) + 1
            continue
        seen.add(sym)
        frame = ohlcv_fn(sym)
        if frame is None or getattr(frame, "empty", True):
            excluded["NO_HISTORY"] = excluded.get("NO_HISTORY", 0) + 1
            continue
        hist = frame[frame.index <= cut]
        if hist.empty or pd.Timestamp(hist.index[-1]).normalize() != cut:
            excluded["NO_SESSION_BAR"] = excluded.get("NO_SESSION_BAR", 0) + 1
            continue                               # never guessed from a stale bar
        feats = confirmation_features(hist)
        if feats is None:
            excluded["NO_FEATURES"] = excluded.get("NO_FEATURES", 0) + 1
            continue
        try:
            score = None if c.get("score") is None else float(c.get("score"))
        except (TypeError, ValueError):
            score = None
        rows.append({"symbol": sym, "champion_score": score, "champion_rank": ranks.get(sym),
                     "champion_verdict": str(c.get("verdict") or c.get("status") or ""),
                     "session_close": float(hist["close"].iloc[-1]), **{k: float(v) for k, v in feats.items()}})
    if len(rows) >= 3:
        parts = [_average_ranks([r[f] for r in rows]) for f in ("low_atr", "near_high", "not_extended")]
        for i, r in enumerate(rows):
            r["challenger_score"] = float(sum(p[i] for p in parts))
        order = sorted(range(len(rows)), key=lambda i: (-rows[i]["challenger_score"], rows[i]["symbol"]))
        for rank, i in enumerate(order, start=1):
            rows[i]["challenger_rank"] = rank
    return rows, excluded, len(records)


def _write(path: Path, header: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    """One transaction: marker + predictions, or nothing. False if already recorded."""
    with closing(_connect(path)) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            if conn.execute("SELECT 1 FROM sessions WHERE session=?", (header["session"],)).fetchone():
                conn.execute("ROLLBACK")
                return False
            conn.execute(
                "INSERT INTO sessions(session,status,reason,hypothesis,schema_version,frozen_at,entry_deadline,"
                "scan_id,scan_completed_at,model_version,candidates,frozen,exclusions,universe,payload_sha256) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (header["session"], header["status"], header.get("reason", ""), HYPOTHESIS, SCHEMA_VERSION,
                 header["frozen_at"], header["entry_deadline"], header.get("scan_id", ""),
                 header.get("scan_completed_at", ""), json.dumps(header["model_version"], sort_keys=True),
                 int(header["candidates"]), len(rows), json.dumps(header.get("exclusions") or {}, sort_keys=True),
                 json.dumps(header.get("universe") or {}, sort_keys=True), header["payload_sha256"]))
            conn.executemany(
                f"INSERT INTO predictions(session,{','.join(_PRED_COLS)}) VALUES(?{',?' * len(_PRED_COLS)})",
                [(header["session"], *[r.get(c) for c in _PRED_COLS]) for r in rows])
            conn.execute("COMMIT")
            return True
        except BaseException:
            conn.execute("ROLLBACK")
            raise


def _wall_clock() -> datetime:
    return datetime.now(timezone.utc)


def record_session(scan: Mapping[str, Any] | None, ohlcv_fn: Callable[[str], Any], *,
                   clock: Callable[[], datetime] = _wall_clock, holidays=None, session_published: bool = True,
                   horizon: int | None = None, path: Path | None = None) -> dict[str, Any]:
    """Freeze one session prospectively (idempotent, atomic).

    ``clock`` is the wall clock (tests inject one). The gate and frozen_at use
    the same reading, and evaluate() re-checks frozen_at against the deadline.
    """
    from research.intelligence.data.nse_calendar import load_holidays

    if horizon is None:
        from core.evaluation_spec import OUTCOME_HORIZON_SESSIONS as horizon  # type: ignore[no-redef]
    holidays = load_holidays() if holidays is None else holidays
    now = clock()
    if now.tzinfo is None:                 # a naive reading could be off by 5.5 h: refuse it
        return {"recorded": 0, "session": "", "reason": "CLOCK_NOT_TIMEZONE_AWARE", "retry": False}
    gate = prospective_gate(scan, now=now, holidays=holidays)
    if not gate["ok"]:
        return {"recorded": 0, "session": gate["session"], "reason": gate["reason"], "retry": gate["retry"]}
    ledger = path or ledger_path()
    if ledger.exists():
        with closing(_connect(ledger)) as conn:
            reused = conn.execute("SELECT session FROM sessions WHERE scan_id=? AND session<>?",
                                  (gate["scan_id"], gate["session"])).fetchone()
        if reused:
            return {"recorded": 0, "session": gate["session"], "reason": "SCAN_ID_ALREADY_FROZEN_FOR_ANOTHER_SESSION",
                    "retry": False}
    if not session_published:
        return {"recorded": 0, "session": gate["session"], "reason": "OFFICIAL_SESSION_BAR_NOT_PUBLISHED",
                "retry": True}
    session = gate["session"]
    rows, excluded, candidates = build_predictions(session, scan.get("records") or [], ohlcv_fn)
    model = model_version(scan, int(horizon))
    universe = scan_universe(scan)
    header = {"session": session, "status": FROZEN, "reason": "PROSPECTIVE",
              "frozen_at": now.astimezone(timezone.utc).isoformat(),
              "entry_deadline": gate["entry_deadline"], "scan_id": gate["scan_id"],
              "scan_completed_at": gate["scan_completed_at"], "model_version": model,
              "candidates": candidates, "exclusions": excluded, "universe": universe,
              "payload_sha256": _payload_sha(session, model, gate["scan_id"], rows, universe)}
    if not _write(path or ledger_path(), header, rows):
        return {"recorded": 0, "session": session, "reason": "already_recorded", "retry": False}
    return {"recorded": len(rows), "session": session, "reason": "PROSPECTIVE", "candidates": candidates,
            "exclusions": excluded, "retry": False}


def mark_missed(*, now: datetime, holidays, path: Path | None = None) -> list[str]:
    """Record MISSED for sessions after the ledger's first one whose deadline passed unfrozen."""
    from research.intelligence.data import nse_calendar as CAL

    path = path or ledger_path()
    if not path.exists():
        return []
    with closing(_connect(path)) as conn:
        known = {r[0] for r in conn.execute("SELECT session FROM sessions")}
    if not known:
        return []
    now = now if now.tzinfo else now.replace(tzinfo=IST)
    latest = CAL.completed_session(now.astimezone(IST).replace(tzinfo=None), holidays)
    cur, missed = date.fromisoformat(min(known)), []
    while cur < latest and len(missed) < MISSED_BACKFILL_LIMIT:
        cur = CAL.next_session(cur, holidays)
        if CAL.calendar_gap(cur, cur, holidays):
            break
        key = cur.isoformat()
        deadline = entry_deadline(cur, holidays)
        if key in known or now < deadline:
            continue
        header = {"session": key, "status": MISSED, "reason": "NOT_FROZEN_BEFORE_ENTRY_DEADLINE",
                  "frozen_at": now.astimezone(timezone.utc).isoformat(), "entry_deadline": deadline.isoformat(),
                  "model_version": {"hypothesis": HYPOTHESIS, "schema_version": SCHEMA_VERSION},
                  "candidates": 0, "payload_sha256": ""}
        if _write(path, header, []):
            missed.append(key)
    return missed


# ── evaluation ───────────────────────────────────────────────────────────────

def load_ledger(path: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    path = path or ledger_path()
    if not path.exists():
        return [], {}
    with closing(_connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        sessions = [dict(r) for r in conn.execute("SELECT rowid AS insert_order, * FROM sessions ORDER BY session")]
        preds: dict[str, list[dict[str, Any]]] = {}
        for r in conn.execute("SELECT * FROM predictions ORDER BY session, symbol"):
            preds.setdefault(r["session"], []).append(dict(r))
    return sessions, preds


def _admissible(meta: Mapping[str, Any], rows: list[Mapping[str, Any]], holidays) -> str:
    """'' when a stored session is admissible prospective evidence, else why not."""
    if meta.get("status") != FROZEN:
        return str(meta.get("status") or "UNKNOWN")
    if int(meta.get("schema_version") or 0) != SCHEMA_VERSION or meta.get("hypothesis") != HYPOTHESIS:
        return "OTHER_SCHEMA_OR_HYPOTHESIS"
    frozen = _parse_utc(meta.get("frozen_at"))
    try:
        deadline = entry_deadline(date.fromisoformat(str(meta["session"])), holidays)
    except Exception:
        return "DEADLINE_UNKNOWN"
    if frozen is None or frozen >= deadline - FREEZE_MARGIN:
        return "FROZEN_AFTER_ENTRY_DEADLINE"
    try:
        model = json.loads(meta.get("model_version") or "{}")
        universe = json.loads(meta.get("universe") or "{}")
    except ValueError:
        return "MODEL_VERSION_UNREADABLE"
    if _payload_sha(str(meta["session"]), model, str(meta.get("scan_id") or ""), list(rows),
                    universe) != meta.get("payload_sha256"):
        return "PAYLOAD_HASH_MISMATCH"
    return ""


def session_outcomes(session: str, rows: list[Mapping[str, Any]], ohlcv_fn: Callable[[str], Any],
                     window: list[date]) -> dict[str, Any]:
    """Exact-horizon outcomes: entry at the open of window[0], exit at the close of window[-1]."""
    import pandas as pd

    wanted = [pd.Timestamp(d) for d in window]
    complete, missing = [], 0
    for r in rows:
        frame = ohlcv_fn(str(r["symbol"]))
        if frame is None or getattr(frame, "empty", True):
            missing += 1
            continue
        idx = pd.DatetimeIndex(frame.index).normalize()
        bars = frame.loc[idx.isin(wanted)]
        if len(bars) != len(wanted) or set(pd.DatetimeIndex(bars.index).normalize()) != set(wanted):
            missing += 1                              # a symbol-level bar is missing in its window
            continue
        bars = bars.sort_index()
        entry, exit_ = float(bars["open"].iloc[0]), float(bars["close"].iloc[-1])
        if not (entry > 0 and exit_ > 0 and math.isfinite(entry) and math.isfinite(exit_)):
            missing += 1
            continue
        complete.append((r, exit_ / entry - 1.0))
    return {"complete": complete, "missing_bars": missing}


def evaluate(ohlcv_fn: Callable[[str], Any], *, maturity: Callable[[str], str], holidays=None,
             horizon: int | None = None, path: Path | None = None) -> dict[str, Any]:
    """Settle matured sessions; paired challenger-vs-champion test (read-only)."""
    from product.learning_scorecard import MATURED
    from research.intelligence.data import nse_calendar as CAL
    from scan.signal_backtest import spearman_ic

    if horizon is None:
        from core.evaluation_spec import OUTCOME_HORIZON_SESSIONS as horizon  # type: ignore[no-redef]
    holidays = CAL.load_holidays() if holidays is None else holidays
    metas, preds = load_ledger(path)
    counts: dict[str, int] = {}
    daily: dict[str, dict[str, float]] = {}
    missing_total = 0

    def bump(key: str) -> None:
        counts[key] = counts.get(key, 0) + 1

    # Freezing only ever writes the latest completed session, so FROZEN rows
    # arrive in session order. One inserted after a later session's row was
    # written retrospectively: never admissible, whatever its timestamps say.
    late_inserts, newest = set(), ""
    for meta in sorted((m for m in metas if m.get("status") == FROZEN), key=lambda m: int(m["insert_order"])):
        if str(meta["session"]) < newest:
            late_inserts.add(str(meta["session"]))
        newest = max(newest, str(meta["session"]))
    for meta in metas:
        session = str(meta["session"])
        rows = preds.get(session, [])
        why = "INSERTED_AFTER_A_LATER_SESSION" if session in late_inserts else _admissible(meta, rows, holidays)
        if why:
            bump(why)
            continue
        status = maturity(session)
        if status != MATURED:
            bump(f"WINDOW_{status}")
            continue
        window = CAL.forward_sessions(date.fromisoformat(session), int(horizon), holidays)
        if window is None:
            bump("EXCHANGE_CALENDAR_NOT_COVERED")
            continue
        got = session_outcomes(session, rows, ohlcv_fn, window)
        missing_total += got["missing_bars"]
        frozen_n = len(rows)
        if frozen_n and got["missing_bars"] / frozen_n > MAX_MISSING_BAR_SHARE:
            bump("TOO_MANY_MISSING_SYMBOL_BARS")
            continue
        fwd = got["complete"]
        if len(fwd) < MIN_CANDIDATES_FOR_IC:
            bump("TOO_FEW_CANDIDATES")
            continue
        mean = sum(x for _, x in fwd) / len(fwd)
        excess = [x - mean for _, x in fwd]
        stats: dict[str, float] = {}
        for key, col in (("challenger", "challenger_score"), ("champion", "champion_score")):
            xs = [r.get(col) for r, _ in fwd]
            if all(x is not None for x in xs):
                ic = spearman_ic([float(x) for x in xs], excess)
                if ic is not None:
                    stats[key] = ic
        if "challenger" in stats and "champion" in stats:
            stats["paired_ic_improvement"] = stats["challenger"] - stats["champion"]
            k = max(1, len(fwd) // 5)
            top = lambda col: sorted(range(len(fwd)), key=lambda i: (-float(fwd[i][0][col]), str(fwd[i][0]["symbol"])))[:k]
            stats["paired_top_quintile_excess"] = (sum(excess[i] for i in top("challenger_score"))
                                                   - sum(excess[i] for i in top("champion_score"))) / k
        if stats:
            daily[session] = stats
            bump("EVALUATED")
        else:
            bump("NO_RANK_INFORMATION")

    out: dict[str, Any] = {
        "hypothesis": HYPOTHESIS, "status": "SHADOW_ONLY", "schema_version": SCHEMA_VERSION,
        "sessions_recorded": sum(1 for m in metas if m.get("status") == FROZEN),
        "sessions_missed": sum(1 for m in metas if m.get("status") == MISSED),
        "sessions_evaluated": len(daily), "session_outcomes": counts,
        "missing_symbol_bars": missing_total, "results": {},
        "legacy_schema1_ledger_ignored": (Path(path).with_name(LEGACY_FILE) if path else
                                          ledger_path().with_name(LEGACY_FILE)).exists(),
        "outcome": f"open of NSE session 1 to close of NSE session {int(horizon)}, excess vs the session's candidates",
    }
    out["results"] = shadow_inference(daily, holidays=holidays, horizon=int(horizon))
    out["decision"] = out["results"]["paired_ic_improvement"].get("verdict")
    out["note"] = ("Prospective, frozen before the first eligible entry; shadow only. The decision is the PAIRED "
                   "IC improvement (challenger minus champion, same candidates, same session); separately "
                   "positive ICs are not evidence that the challenger is better. CHALLENGER_BETTER still needs "
                   "the walk-forward confirmation and the champion/challenger gate before PAPER.")
    return out


def shadow_inference(daily: Mapping[str, Mapping[str, float]], *, holidays, horizon: int) -> dict[str, Any]:
    """Overlap-robust tests of the per-session statistics; the paired one decides."""
    from research.intelligence.data import nse_calendar as CAL
    from research.overlap_inference import OK, _t_ppf, cells_from_pairs, overlap_robust_test

    ordinals = CAL.strict_session_ordinals(sorted(daily), holidays)
    results: dict[str, Any] = {}
    for key in ("challenger", "champion", "paired_ic_improvement", "paired_top_quintile_excess"):
        if ordinals is None:
            results[key] = {"verdict": "INSUFFICIENT EVIDENCE", "reason": "EXCHANGE_CALENDAR_NOT_COVERED"}
            continue
        rep = overlap_robust_test(cells_from_pairs((ordinals[s], v[key]) for s, v in daily.items() if key in v),
                                  int(horizon))
        res: dict[str, Any] = {"sessions": rep["sessions"], "independent_sessions": rep["independent_sessions"],
                               "mean": None if rep["mean"] is None else round(rep["mean"], 5)}
        if rep["status"] != OK:
            res.update({"verdict": "INSUFFICIENT EVIDENCE", "reason": rep["reason"]})
        else:
            upper = rep["mean"] + _t_ppf(0.95, rep["df"]) * rep["se"]
            res.update({"se": round(rep["se"], 5), "df": rep["df"], "t": round(rep["t"], 3),
                        "p_two_sided": round(rep["p_two_sided"], 6),
                        "lower_95_one_sided": round(rep["lower_95_one_sided"], 5),
                        "variance_inflation": rep["variance_inflation"],
                        "residual_dependence": rep["residual_dependence"]})
            if key.startswith("paired"):
                res["verdict"] = ("CHALLENGER_BETTER" if rep["lower_95_one_sided"] > 0 else
                                  "CHALLENGER_WORSE" if upper < 0 else "NOT_DISTINGUISHABLE")
            else:
                res["verdict"] = ("POSITIVE_IC" if rep["lower_95_one_sided"] > 0 else
                                  "NEGATIVE_IC" if upper < 0 else "NOT_DISTINGUISHABLE_FROM_ZERO")
        results[key] = res
    # Multiple testing: the decision family is the two paired tests (the
    # pre-registered IC difference and the top-quintile spread). One-sided
    # p-values for "challenger better", Holm-Bonferroni; CHALLENGER_BETTER
    # needs the primary test to survive the correction as well.
    # The family is fixed at two by pre-registration: an untestable member
    # counts as p = 1, it never shrinks the correction.
    family = ("paired_ic_improvement", "paired_top_quintile_excess")
    from research.overlap_inference import _t_sf

    pvals = {k: (_t_sf(float(results[k]["t"]), int(results[k]["df"])) if "t" in results.get(k, {}) else 1.0)
             for k in family}
    running = 0.0
    for rank, k in enumerate(sorted(family, key=lambda k: pvals[k])):
        running = max(running, min(1.0, (len(family) - rank) * pvals[k]))
        results[k]["p_one_sided_better"] = round(pvals[k], 6)
        results[k]["holm_adjusted_p"] = round(running, 6)
    primary = results.get("paired_ic_improvement", {})
    if primary.get("verdict") == "CHALLENGER_BETTER" and primary.get("holm_adjusted_p", 1.0) >= 0.05:
        primary["verdict"] = "NOT_DISTINGUISHABLE"
        primary["reason"] = "did not survive Holm correction across the paired tests"
    return results


def nightly() -> dict[str, Any]:
    """Learning-cycle hook: freeze tonight's session (if still prospective), mark gaps, evaluate."""
    from data import bhavcopy_runtime, bhavcopy_store
    from product.learning_scorecard import maturity_checker
    from product.scan_store import default_scan_path, load_scan
    from research.intelligence.data.nse_calendar import load_holidays

    bhavcopy_runtime.ensure_loaded(rebuild_from_local=False)
    if not bhavcopy_store.is_ready():
        return {"error": "OFFICIAL_STORE_NOT_LOADED"}
    holidays = load_holidays()
    scan = load_scan(default_scan_path())
    session = str((scan or {}).get("as_of_session") or "")[:10]
    try:
        published = date.fromisoformat(session) in set(bhavcopy_store._dates_on_disk())
    except Exception:
        published = False
    recorded = record_session(scan, bhavcopy_store.get_ohlcv, holidays=holidays, session_published=published)
    missed = mark_missed(now=_wall_clock(), holidays=holidays)
    report = evaluate(bhavcopy_store.get_ohlcv, maturity=maturity_checker(holidays=holidays), holidays=holidays)
    return {"recorded": recorded, "missed_marked": missed, "evaluation": report}
