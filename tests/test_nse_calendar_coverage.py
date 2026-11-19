"""Session counting for evidence fails closed when the holiday calendar cannot vouch for a year."""

from __future__ import annotations

from datetime import date

import research.intelligence.data.nse_calendar as CAL
import scan.signal_backtest as SB

FULL_2026 = CAL.load_holidays()


def test_only_years_with_a_published_calendar_are_covered():
    assert 2026 in CAL.covered_years(FULL_2026)
    assert CAL.covered_years({"2027-01-26"}) == set()                    # one amendment is not a calendar
    assert CAL.covered_years({"garbage", "2026-02-30"}) == set()
    assert CAL.calendar_gap(date(2026, 3, 1), date(2026, 4, 1), FULL_2026) == ""
    assert CAL.calendar_gap(date(2026, 12, 20), date(2027, 1, 5), FULL_2026) == "EXCHANGE_CALENDAR_NOT_COVERED:2027"
    assert CAL.calendar_gap(date(2026, 3, 1), date(2026, 4, 1), set()) == "EXCHANGE_CALENDAR_UNAVAILABLE"


def test_forward_window_skips_holidays_and_refuses_uncovered_years():
    window = CAL.forward_sessions(date(2026, 11, 18), 10, FULL_2026)
    assert window[0] == date(2026, 11, 19) and window[-1] == date(2026, 12, 3)
    assert date(2026, 11, 24) not in window                               # Guru Nanak Jayanti
    assert CAL.forward_sessions(date(2026, 12, 22), 10, FULL_2026) is None  # reaches into 2027
    assert CAL.forward_sessions(date(2026, 11, 18), 10, set()) is None


def test_strict_ordinals_never_count_an_uncovered_year_as_weekdays():
    assert CAL.strict_session_ordinals(["2026-11-06", "2026-11-09", "2026-11-11"], FULL_2026) == {
        "2026-11-06": 1, "2026-11-09": 2, "2026-11-11": 3}
    assert CAL.strict_session_ordinals(["2025-06-02", "2025-06-03"], FULL_2026) is None
    assert SB._session_ordinals(["2025-06-02", "2026-06-03"]) is None     # the walk-forward fallback too


# ── provenance, not plausibility (2026-11-20 review) ─────────────────────────

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "nse_holiday_master_cm_2026.json"


def test_the_bundled_2026_calendar_is_verified_against_nse():
    manifest = CAL.load_manifest()
    entry = manifest["years"]["2026"]
    assert entry["status"] == CAL.VERIFIED_COMPLETE and "nseindia.com" in entry["source"]
    assert CAL.weekday_holiday_digest(FULL_2026, 2026) == (entry["weekday_holidays"], entry["weekday_holidays_sha256"])
    assert "2026-11-08 Diwali Laxmi Pujan*" in entry["special_session_markers"]


def test_a_plausible_but_partial_calendar_is_not_coverage():
    partial = set(sorted(FULL_2026)[:8])                                  # 8 real holidays of 16
    assert CAL.covered_years(partial) == set()
    assert CAL.calendar_gap(date(2026, 3, 1), date(2026, 3, 31), partial) == "EXCHANGE_CALENDAR_NOT_COVERED:2026"


def test_a_newly_announced_holiday_fails_closed_until_re_verified():
    amended = FULL_2026 | {"2026-12-15"}                                   # an unverified amendment
    assert CAL.covered_years(amended) == set()
    assert CAL.forward_sessions(date(2026, 12, 1), 10, amended) is None
    weekend_note = FULL_2026 | {"2026-11-08"}                              # weekend entries change nothing
    assert 2026 in CAL.covered_years(weekend_note)


def test_without_a_manifest_or_with_an_unverified_entry_nothing_is_covered(tmp_path):
    assert CAL.covered_years(FULL_2026, manifest={}) == set()
    entry = dict(CAL.load_manifest()["years"]["2026"], status="DRAFT")
    assert CAL.covered_years(FULL_2026, manifest={"years": {"2026": entry}}) == set()
    assert CAL.calendar_gap(date(2026, 3, 1), date(2026, 3, 2), FULL_2026, manifest={}) == \
        "EXCHANGE_CALENDAR_NOT_COVERED:2026"


def test_special_sessions_are_not_regular_sessions_in_outcome_windows():
    window = CAL.forward_sessions(date(2026, 11, 5), 3, FULL_2026)
    assert window == [date(2026, 11, 6), date(2026, 11, 9), date(2026, 11, 11)]   # no Sunday 8 Nov, no 10 Nov


def _verify(tmp_path, *extra, holidays=None):
    import uuid
    tmp_path = tmp_path / uuid.uuid4().hex[:8]
    tmp_path.mkdir()
    hol = tmp_path / "hol.json"
    hol.write_text(json.dumps(sorted(FULL_2026 if holidays is None else holidays)))
    man = tmp_path / "manifest.json"
    run = subprocess.run([sys.executable, str(ROOT / "scripts" / "verify_nse_calendar.py"), "--input", str(FIXTURE),
                          "--holidays", str(hol), "--manifest", str(man), *extra],
                         capture_output=True, text=True, timeout=60)
    return run, hol, man


def test_the_verifier_records_only_an_exact_match(tmp_path):
    run, _, man = _verify(tmp_path, "--year", "2026")
    assert run.returncode == 0 and json.loads(man.read_text())["years"]["2026"]["status"] == CAL.VERIFIED_COMPLETE
    run, _, man = _verify(tmp_path, "--year", "2026", holidays=sorted(FULL_2026)[:-1])   # one holiday missing
    assert run.returncode == 2 and '"official_only"' in run.stdout and not man.exists()
    run, _, man = _verify(tmp_path, "--year", "2027")                               # not published yet
    assert run.returncode == 2 and not man.exists()


def test_the_verifier_can_rebuild_a_year_from_nse(tmp_path):
    run, hol, man = _verify(tmp_path, "--year", "2026", "--write-holidays", holidays={"2026-01-26"})
    assert run.returncode == 0 and set(json.loads(hol.read_text())) == FULL_2026
