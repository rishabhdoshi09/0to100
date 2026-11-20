"""Acceptance: a BUY logged today settles automatically after its horizon.

Runs the PRODUCTION handler chain (research.autonomy.install_runtime_handlers,
the exact sequence run_supervisor installs, including outcome_liveness and
outcome_integrity wrappers) in a fresh interpreter, with real frames in the
real in-memory store (data.bhavcopy_store._store), no patched get_ohlcv.

History: the installed handler never called update_outcomes, so no surfaced
BUY was settled from 2026-07-30 to 2026-11-19 while every job read
"outcomes resolved".
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = textwrap.dedent(
    r'''
    import json, sys
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    import pandas as pd

    import core.signal_outcome_tracker as T
    import data.bhavcopy_store as BS
    from research.autonomy import install_runtime_handlers
    from research.autonomy import jobs as J, schedules as SCH

    install_runtime_handlers()
    handler = J.HANDLERS[SCH.OUTCOME_RESOLUTION]

    IST = ZoneInfo("Asia/Kolkata")
    logged = datetime(2026, 9, 1, 10, 0)            # the BUY is surfaced on 1 Sep
    T._now = lambda: logged
    T.log_signal("WINCO", "UNIFIED_BUY", 100.0, 100.0, 96.0, 106.0, 72.0, 40.0, "VCP", "CHOPPY")
    T.log_signal("NOBARS", "UNIFIED_BUY", 50.0, 50.0, 48.0, 54.0, 60.0, 30.0, "VCP", "CHOPPY")
    T._now = lambda: datetime(2026, 9, 26, 19, 0)   # inside the horizon for this one
    T.log_signal("YOUNG", "UNIFIED_BUY", 200.0, 200.0, 192.0, 216.0, 65.0, 30.0, "VCP", "CHOPPY")

    idx = pd.bdate_range("2026-08-03", "2026-09-30")
    idx = idx[idx != pd.Timestamp("2026-09-14")]    # an NSE holiday has no bar
    close = [99.0 if d <= pd.Timestamp("2026-09-01") else 100.5 for d in idx]   # a close inside its bar
    high = [99.5 if d <= pd.Timestamp("2026-09-01") else 101.0 for d in idx]
    high[list(idx).index(pd.Timestamp("2026-09-04"))] = 107.0   # target 106 touched
    frame = pd.DataFrame({"open": 99.0, "high": high, "low": 98.5, "close": close, "volume": 1e6}, index=idx)
    young = pd.DataFrame({"open": 200.0, "high": 201.0, "low": 199.0, "close": 200.0, "volume": 1e6}, index=idx)
    BS._store["WINCO"] = frame                       # the real store, no patched get_ohlcv
    BS._store["YOUNG"] = young
    # The archive listing that names the official sessions: without it a loaded
    # store never settles positionally (core.session_window, QT-06).
    BS._dates_on_disk = lambda: [d.date() for d in idx]

    class Deps:
        def now_ist(self):
            return datetime(2026, 9, 30, 19, 0, tzinfo=IST)
        def holidays(self):
            return set()
        def official_history(self):
            return {"current": True, "available_session": "2026-09-30"}
        def resolve_outcomes(self, *_a, **_k):
            return {"status": "EOD_SETTLED", "positions_closed": [], "outcomes_recorded": []}

    class Ctx:
        deps = Deps()
        active_failures = set()

    T._now = lambda: datetime(2026, 9, 30, 19, 0)
    first = handler(Ctx())
    second = handler(Ctx())
    conn = T._get_conn()
    rows = {r["symbol"]: dict(r) for r in conn.execute("SELECT * FROM signal_log")}
    BS._store.clear()
    third = handler(Ctx())                            # no official bars in this process
    print(json.dumps({
        "handler": f"{handler.__module__}.{handler.__name__}",
        "first": {"status": first.status, "summary": first.summary,
                  "signal": (first.metadata or {}).get("signal_outcomes")},
        "second": {"status": second.status, "summary": second.summary,
                   "signal": (second.metadata or {}).get("signal_outcomes")},
        "third": {"status": third.status, "summary": third.summary,
                  "error": (third.metadata or {}).get("signal_outcomes_error")},
        "rows": rows,
    }, default=str))
    '''
)


def _run(tmp_path) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_CURRENT_TEST"}
    env.update({
        "QT_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "KITE_API_KEY": "", "KITE_ACCESS_TOKEN": "", "TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": "",
        "PYTHONPATH": str(ROOT),
    })
    (tmp_path / "runtime").mkdir()
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT], cwd=str(ROOT), env=env,
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_buy_logged_today_settles_after_horizon_through_the_installed_handler(tmp_path):
    out = _run(tmp_path)
    # The handler production actually runs is the outer integrity wrapper.
    assert out["handler"] == "research.autonomy.outcome_integrity.strict_outcome_resolution"

    rows = out["rows"]
    win = rows["WINCO"]
    assert win["worked"] == 1                                   # target touched on 4 Sep
    assert abs(float(win["outcome_pct"]) - 6.0) < 0.05          # filled at 100, out at 106
    assert win["eval_spec"] and win["outcome_checked_at"]
    assert rows["YOUNG"]["worked"] is None                       # still inside its horizon
    assert rows["NOBARS"]["worked"] is None                      # no bars: never invented

    first = out["first"]
    assert first["status"] == "SUCCEEDED"
    assert first["signal"]["written"] == 1
    assert first["signal"]["open_within_horizon"] == 1
    assert first["signal"]["stuck_past_horizon"] == 1           # NOBARS is named, not hidden
    assert first["signal"]["open_reasons"] == {"NO_BARS_OR_WITHIN_HORIZON": 1}
    assert "signal_log 1 settled/1 stuck" in first["summary"]


def test_rerun_is_idempotent_and_an_empty_store_is_reported(tmp_path):
    out = _run(tmp_path)
    second = out["second"]
    assert second["signal"]["written"] == 0                      # nothing settled twice
    assert second["signal"]["candidates"] == 2                   # only the still-open rows
    third = out["third"]
    assert third["error"] == "OFFICIAL_STORE_NOT_LOADED"
    assert "official store not loaded" in third["summary"]
    # A failed surfaced-BUY settlement is not a SUCCEEDED job: it retries and
    # does not release the learning dependency (GPT cross-review A-9).
    assert third["status"] == "RETRYABLE_FAILED"
