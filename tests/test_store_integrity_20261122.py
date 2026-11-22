"""Store integrity (22 Nov 2026): the saved OHLCV store must equal its verified official archives.

The audit of the production store (research.store_integrity_audit, on a copy)
found three defects in how the cache had been maintained:

* stale intraday overlay rows kept as history for symbols an archive did not
  list (fixed in 43cba9bb, RS-02);
* histories left out of date order when an archive arrived after an overlay row
  (``_incremental_append`` never re-sorted);
* young listings that lost their first sessions, because the full build drops
  names below 30 rows while appends admit them from their next row.

These tests pin the store fixes, the audit's classification, and the rebuild
procedure (plan, apply, rollback) on a synthetic runtime built from real CSV
archives parsed by the store's own parser.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
import pytest

from data import bhavcopy_store as BS


# ── store fixes ─────────────────────────────────────────────────────────────

def _frame(days, close=100.0):
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1000.0},
                        index=pd.DatetimeIndex(days))


def test_a_late_archive_leaves_every_history_in_date_order(monkeypatch):
    days = pd.bdate_range("2026-09-01", periods=6)
    held = _frame(list(days[:4]) + [days[5]])                       # day 5 missing, day 6 already held
    monkeypatch.setattr(BS, "_store", {"AAA": held})
    monkeypatch.setattr(BS, "_pending", {})
    monkeypatch.setattr(BS, "_store_last_day", days[5].date())
    monkeypatch.setattr(BS, "_store_sessions", 5)
    monkeypatch.setattr(BS, "_save_pkl", lambda: None)
    monkeypatch.setattr(BS, "_read_day", lambda d: pd.DataFrame(
        {"symbol": ["AAA"], "date": days[4], "open": 101.0, "high": 101.0, "low": 101.0,
         "close": 101.0, "volume": 900.0}))
    BS._incremental_append([days[4].date()])
    idx = BS._store["AAA"].index
    assert idx.is_monotonic_increasing and len(idx) == 6
    assert BS._store["AAA"].iloc[-1]["close"] == 100.0               # the latest bar is still the latest


def test_a_below_floor_name_rejoins_with_its_whole_history(monkeypatch):
    days = pd.bdate_range("2026-09-01", periods=6)
    monkeypatch.setattr(BS, "_store", {})
    monkeypatch.setattr(BS, "_pending", {"YOUNG": _frame(days[:5], 50.0)})
    monkeypatch.setattr(BS, "_store_last_day", days[4].date())
    monkeypatch.setattr(BS, "_store_sessions", 5)
    monkeypatch.setattr(BS, "_save_pkl", lambda: None)
    monkeypatch.setattr(BS, "_read_day", lambda d: pd.DataFrame(
        {"symbol": ["YOUNG"], "date": days[5], "open": 51.0, "high": 51.0, "low": 51.0,
         "close": 51.0, "volume": 10.0}))
    BS._incremental_append([days[5].date()])
    assert len(BS._store["YOUNG"]) == 6                               # not just the newest row
    assert "YOUNG" not in BS._pending


def test_the_pending_history_survives_a_save_and_a_load(tmp_path, monkeypatch):
    days = pd.bdate_range("2026-09-01", periods=3)
    monkeypatch.setattr(BS, "_PKL", tmp_path / "store_cache.pkl")
    monkeypatch.setattr(BS, "_store", {"AAA": _frame(days)})
    monkeypatch.setattr(BS, "_pending", {"YOUNG": _frame(days[1:], 5.0)})
    monkeypatch.setattr(BS, "_store_last_day", days[-1].date())
    monkeypatch.setattr(BS, "_store_sessions", 3)
    BS._save_pkl()
    monkeypatch.setattr(BS, "_store", {})
    monkeypatch.setattr(BS, "_pending", {})
    monkeypatch.setattr(BS, "_dates_on_disk", lambda: [d.date() for d in days])
    assert BS._load_pkl()
    assert set(BS._pending) == {"YOUNG"} and len(BS._pending["YOUNG"]) == 2


# ── a synthetic runtime with real archives ─────────────────────────────────────

def _sessions(start, n):
    """n NSE sessions from ``start`` (weekdays minus the store's own holiday list)."""
    holidays = BS._holidays()
    out = [d for d in pd.bdate_range(start, periods=n + 20) if d.date().isoformat() not in holidays]
    return pd.DatetimeIndex(out[:n])


DAYS = _sessions("2026-06-01", 35)


def _archive(day, rows):
    stamp = day.strftime("%d-%b-%Y")
    lines = ["SYMBOL, SERIES, DATE1, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, CLOSE_PRICE, TTL_TRD_QNTY, DELIV_PER"]
    for sym, close in rows:
        lines.append(f"{sym}, EQ, {stamp}, {close}, {close + 1}, {close - 1}, {close}, 1000, 45.5")
    return "\n".join(lines) + "\n"


def _official(sym, day_index):
    return 100.0 + day_index if sym == "AAA" else 50.0 + day_index if sym == "BBB" else 20.0 + day_index


@pytest.fixture
def runtime(tmp_path):
    """Archives: AAA all 35 sessions, BBB the first 34, YOUNG the last 12. The pickle is contaminated."""
    bhav = tmp_path / "rt" / "logs" / "bhav"
    bhav.mkdir(parents=True)
    for i, day in enumerate(DAYS):
        rows = [("AAA", _official("AAA", i))]
        if i < 34:
            rows.append(("BBB", _official("BBB", i)))
        if i >= 23:
            rows.append(("YOUNG", _official("YOUNG", i)))
        (bhav / f"{day.strftime('%d%m%Y')}.csv").write_text(_archive(day, rows), encoding="latin-1")

    def frame(sym, idx):
        closes = [_official(sym, i) for i in idx]
        return pd.DataFrame({"open": closes, "high": [c + 1 for c in closes], "low": [c - 1 for c in closes],
                             "close": closes, "volume": 1000.0, "deliv_per": 45.5},
                            index=pd.DatetimeIndex([DAYS[i] for i in idx]))
    bbb = frame("BBB", range(33))
    orphan = pd.DataFrame({"open": bbb["close"].iloc[-1], "high": bbb["close"].iloc[-1],
                           "low": bbb["close"].iloc[-1], "close": bbb["close"].iloc[-1],
                           "volume": 0.0, "deliv_per": float("nan")}, index=pd.DatetimeIndex([DAYS[34]]))
    bbb = pd.concat([bbb, orphan, frame("BBB", [33])])               # orphan, then out of order
    store = {"AAA": frame("AAA", range(35)), "BBB": bbb, "YOUNG": frame("YOUNG", range(29, 35))}
    with open(bhav / "store_cache.pkl", "wb") as fh:
        pickle.dump({"store": store, "last_day": DAYS[-1].date(), "sessions": 37}, fh)
    return tmp_path / "rt"


def test_the_audit_classifies_every_defect(runtime, tmp_path):
    from research.store_integrity_audit import audit

    rep = audit(runtime)
    assert rep["counts"].get("ORPHAN_ON_ARCHIVE_DATE") == 1
    assert rep["findings"]["ORPHAN_ON_ARCHIVE_DATE"] == [("BBB", DAYS[34].date().isoformat())]
    assert rep["counts"].get("MISSING_OFFICIAL_ROW") == 6                # YOUNG's first 6 sessions
    assert not rep["counts"].get("VALUE_MISMATCH") and not rep["counts"].get("NO_ARCHIVE_FOR_DATE")
    assert rep["affected_symbols"] == ["BBB"]


def test_the_audit_refuses_the_live_runtime():
    from research.store_integrity_audit import audit

    with pytest.raises(SystemExit):
        audit("/Volumes/QuantTermStorage/QuantTerm/runtime")


def test_plan_apply_audit_and_rollback_on_a_copy(runtime, tmp_path):
    from product import bhav_store_rebuild as RB
    from research.store_integrity_audit import audit

    target_pkl = runtime / "logs" / "bhav" / "store_cache.pkl"
    original = RB._sha256(target_pkl)
    manifest = tmp_path / "manifest.json"
    m = RB.plan(runtime, tmp_path / "staging", manifest)
    assert m["ready"], m["gates"]
    assert m["diff"]["removed"] == 1 and m["diff"]["added"] == 6 and m["diff"]["changed"] == 0
    assert m["diff"]["unsorted_in_old"] == ["BBB"]
    assert m["build"]["promoted_below_floor"] == 1                    # YOUNG (12 rows) keeps its place
    assert RB._sha256(target_pkl) == original                         # plan never writes the target

    with pytest.raises(SystemExit):
        RB.apply(manifest, "   ", target_is_copy=True)                # no approval in words
    RB.apply(manifest, "TEST", target_is_copy=True)
    assert RB._sha256(target_pkl) == m["new_sha256"]
    after = audit(runtime)
    assert not after["contamination_kinds"] and not after["counts"].get("MISSING_OFFICIAL_ROW")
    with open(target_pkl, "rb") as fh:
        store = pickle.load(fh)["store"]
    assert all(f.index.is_monotonic_increasing for f in store.values())
    assert len(store["YOUNG"]) == 12 and DAYS[34] not in store["BBB"].index

    RB.rollback(manifest, target_is_copy=True)
    assert RB._sha256(target_pkl) == original


def test_apply_refuses_a_target_that_changed_since_the_plan(runtime, tmp_path):
    from product import bhav_store_rebuild as RB

    manifest = tmp_path / "manifest.json"
    RB.plan(runtime, tmp_path / "staging", manifest)
    target_pkl = runtime / "logs" / "bhav" / "store_cache.pkl"
    target_pkl.write_bytes(target_pkl.read_bytes() + b"\0")            # the desk saved in between
    with pytest.raises(SystemExit, match="changed since the plan"):
        RB.apply(manifest, "TEST", target_is_copy=True)


def test_the_desk_check_is_never_skipped_under_volumes(monkeypatch):
    from product import bhav_store_rebuild as RB

    monkeypatch.setattr(RB, "_desk_running", lambda: ["terminal_product_api:123"])
    with pytest.raises(SystemExit, match="desk is running"):
        RB._check_desk(Path("/Volumes/QuantTermStorage/QuantTerm/runtime/logs/bhav/store_cache.pkl"), True)
    RB._check_desk(Path("/tmp/a-copy/logs/bhav/store_cache.pkl"), True)   # a declared copy elsewhere
    with pytest.raises(SystemExit, match="desk is running"):
        RB._check_desk(Path("/tmp/a-copy/logs/bhav/store_cache.pkl"), False)


def test_plan_refuses_a_staging_root_on_the_runtime_volume(runtime):
    from product import bhav_store_rebuild as RB

    with pytest.raises(SystemExit, match="/Volumes"):
        RB.plan(runtime, "/Volumes/QuantTermStorage/staging", "/tmp/m.json")


def test_rollback_without_a_backup_refuses(tmp_path):
    from product import bhav_store_rebuild as RB

    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"target_pickle": str(tmp_path / "x.pkl"), "target_sha256": "0" * 64}))
    with pytest.raises(SystemExit, match="nothing to roll back"):
        RB.rollback(manifest, target_is_copy=True)


# ── GPT cross-review of 8a52e49d (ST-01..ST-09): GPT's own sketches ───────────
# Each failed on 8a52e49d at its claim-bearing assertion. One adaptation: the
# tool resolves runtime paths through core.runtime_paths (the isolation gate),
# so ``runtime / rb.PICKLE`` is spelled ``rb._pickle(runtime)``.

@pytest.mark.parametrize("command", [
    "python -u -m product.host_entrypoint",
    "python -u -m product.host_launchd_entrypoint",
    "python -u -m product.host_bootstrap --child",
])
def test_review_apply_blocks_installed_host(runtime, tmp_path, monkeypatch, command):
    import re
    from subprocess import CompletedProcess
    from product import bhav_store_rebuild as rb

    manifest = tmp_path / "manifest.json"
    assert rb.plan(runtime, tmp_path / "stage", manifest)["ready"]

    def probe(args, **kwargs):
        hit = args[0] == "pgrep" and re.search(args[-1], command)
        return CompletedProcess(args, 0 if hit else 1, "999999\n" if hit else "", "")

    monkeypatch.setattr(rb.subprocess, "run", probe)
    with pytest.raises(SystemExit, match="desk is running"):
        rb.apply(manifest, "test approval")


def test_review_plan_uses_target_holidays(runtime, tmp_path):
    from product import bhav_store_rebuild as rb
    from research.store_integrity_audit import audit

    day = DAYS[10]
    (runtime / "logs/nse_holidays.json").write_text(json.dumps([day.date().isoformat()]))
    pkl = rb._pickle(runtime)
    payload = pickle.loads(pkl.read_bytes())
    for symbol, frame in payload["store"].items():
        payload["store"][symbol] = frame.drop(day, errors="ignore")
    pkl.write_bytes(pickle.dumps(payload))

    m = rb.plan(runtime, tmp_path / "stage", tmp_path / "manifest.json")
    assert m["ready"], m["gates"]
    assert not audit(runtime, pickle_path=m["new_pickle"])["contamination_kinds"]


def test_review_apply_rejects_archive_correction(runtime, tmp_path, monkeypatch):
    from product import bhav_store_rebuild as rb

    manifest = tmp_path / "manifest.json"
    assert rb.plan(runtime, tmp_path / "stage", manifest)["ready"]
    day = DAYS[10]
    csv = runtime / "logs/bhav" / day.strftime("%d%m%Y.csv")
    corrected = csv.with_suffix(".tmp")
    corrected.write_text(_archive(day, [("AAA", 999.0), ("BBB", _official("BBB", 10))]))
    corrected.replace(csv)
    monkeypatch.setattr(rb, "_held_open", lambda _: [])

    with pytest.raises(SystemExit):
        rb.apply(manifest, "test approval", target_is_copy=True)


def test_review_rollback_preserves_post_apply_update(runtime, tmp_path, monkeypatch):
    from product import bhav_store_rebuild as rb

    manifest = tmp_path / "manifest.json"
    assert rb.plan(runtime, tmp_path / "stage", manifest)["ready"]
    monkeypatch.setattr(rb, "_held_open", lambda _: [])
    rb.apply(manifest, "test approval", target_is_copy=True)
    target = rb._pickle(runtime)
    payload = pickle.loads(target.read_bytes())
    payload["store"]["AAA"].loc[DAYS[-1], "volume"] = 2000
    updated = pickle.dumps(payload)
    target.write_bytes(updated)

    with pytest.raises(SystemExit):
        rb.rollback(manifest, target_is_copy=True)
    assert target.read_bytes() == updated


def test_review_audit_detects_wrong_pending_close(runtime):
    from product import bhav_store_rebuild as rb
    from research.store_integrity_audit import audit

    pkl = rb._pickle(runtime)
    payload = pickle.loads(pkl.read_bytes())
    young = payload["store"].pop("YOUNG").reindex(DAYS[23:]).copy()
    for i in range(23, 29):
        close = _official("YOUNG", i)
        young.loc[DAYS[i]] = [close, close + 1, close - 1, close, 1000, 45.5]
    young.loc[DAYS[25], "close"] = 999.0
    payload["pending"] = {"YOUNG": young}
    pkl.write_bytes(pickle.dumps(payload))

    assert "YOUNG" in audit(runtime)["affected_symbols"]


def test_review_quote_cannot_overwrite_concurrent_archive(tmp_path, monkeypatch):
    from data import kite_client, nse_live, nse_universe
    from product import bhav_store_rebuild as rb

    prior, today, after = DAYS[:3]
    pkl = tmp_path / "store_cache.pkl"
    monkeypatch.setattr(BS, "_PKL", pkl)
    monkeypatch.setattr(BS, "_store", {"AAA": _frame([prior])})
    monkeypatch.setattr(BS, "_store_last_day", prior.date())
    monkeypatch.setattr(BS, "_store_sessions", 1)
    monkeypatch.setattr(BS, "_read_day", lambda d: pd.DataFrame({
        "symbol": ["AAA"], "date": pd.Timestamp(d), "open": 110.0,
        "high": 110.0, "low": 110.0, "close": 110.0, "volume": 1000.0,
    }))
    monkeypatch.setattr(nse_live, "date", type("Clock", (), {
        "today": staticmethod(today.date),
    }))
    from core import market_clock
    monkeypatch.setattr(market_clock, "today_ist", today.date)     # the producer dates by the NSE clock (C7-08)
    monkeypatch.setattr(nse_live, "_is_trading_now", lambda: True)
    monkeypatch.setattr(nse_universe, "get_nse_universe", lambda: [])
    monkeypatch.setattr(kite_client, "kite_credentials_available", lambda: False)

    def fetch():
        BS._incremental_append([today.date()])
        return {"AAA": dict(open=777.0, high=777.0, low=777.0, close=777.0, volume=7.0)}

    monkeypatch.setattr(nse_live, "fetch_live_snapshot", fetch)
    nse_live.apply_live_to_store()
    BS._incremental_append([after.date()])
    assert rb._frames(pkl)["AAA"].at[today, "close"] == 110.0


def test_review_rebuild_preserves_deployed_sampling_order(runtime, tmp_path):
    from product import bhav_store_rebuild as rb

    pkl = rb._pickle(runtime)
    payload = pickle.loads(pkl.read_bytes())
    order = ["BBB", "AAA", "YOUNG"]
    payload["store"] = {s: payload["store"][s] for s in order}
    pkl.write_bytes(pickle.dumps(payload))

    m = rb.plan(runtime, tmp_path / "stage", tmp_path / "manifest.json")
    assert m["ready"]
    assert list(rb._frames(m["new_pickle"])) == order


@pytest.mark.parametrize("defect", ["missing_delivery", "non_midnight", "overlay_marker"])
def test_review_audit_flags_reader_visible_differences(runtime, defect):
    from product import bhav_store_rebuild as rb
    from research.store_integrity_audit import audit

    pkl = rb._pickle(runtime)
    payload = pickle.loads(pkl.read_bytes())
    frame = payload["store"]["AAA"]
    if defect == "missing_delivery":
        frame.drop(columns="deliv_per", inplace=True)
    elif defect == "non_midnight":
        frame.index += pd.Timedelta(hours=15, minutes=30)
    else:
        frame.attrs["quantterm_live_overlay_date"] = DAYS[-1].date().isoformat()
    pkl.write_bytes(pickle.dumps(payload))

    assert "AAA" in audit(runtime)["affected_symbols"]


def test_review_public_refresh_incorporates_late_archive(monkeypatch):
    first, late, newest = DAYS[:3]
    monkeypatch.setattr(BS, "_store", {"AAA": _frame([first, newest])})
    monkeypatch.setattr(BS, "_store_last_day", newest.date())
    monkeypatch.setattr(BS, "_store_sessions", 2)
    monkeypatch.setattr(BS, "_MIN_DAYS", 1)
    monkeypatch.setattr(BS, "_trading_days_back",
                        lambda _: [newest.date(), late.date(), first.date()])
    monkeypatch.setattr(BS, "_is_valid_archive", lambda *_: True)
    monkeypatch.setattr(BS, "_read_day", lambda d: pd.DataFrame({
        "symbol": ["AAA"], "date": pd.Timestamp(d), "open": 100.0,
        "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0,
    }))
    monkeypatch.setattr(BS, "_save_pkl", lambda: None)

    BS.build_store(days=2)
    assert late in BS._store["AAA"].index


def test_review_complete_diff_can_exceed_display_cap(runtime, tmp_path):
    from product import bhav_store_rebuild as rb

    pkl = rb._pickle(runtime)
    payload = pickle.loads(pkl.read_bytes())
    extra = pd.date_range(end=DAYS[0] - pd.Timedelta(days=1), periods=2001)
    payload["store"]["AAA"] = pd.concat([_frame(extra), payload["store"]["AAA"]])
    pkl.write_bytes(pickle.dumps(payload))

    m = rb.plan(runtime, tmp_path / "stage", tmp_path / "manifest.json")
    assert m["diff"]["removed"] == 2002  # Includes the fixture's BBB orphan.
    assert m["gates"]["removed_rows_are_exactly_the_contamination"]


def _hold_publish_lock(path):
    """A separate process holding the store's publish lock (an in-flight writer or a repair)."""
    import subprocess
    import sys as _sys

    code = ("import fcntl, os, sys\n"
            f"fd = os.open({str(path)!r}, os.O_RDWR | os.O_CREAT, 0o644)\n"
            "fcntl.flock(fd, fcntl.LOCK_EX)\n"
            "print('locked', flush=True)\n"
            "sys.stdin.readline()\n")
    proc = subprocess.Popen([_sys.executable, "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "locked"
    return proc


def test_a_save_defers_while_a_repair_holds_the_publish_lock_and_retries_later(tmp_path, monkeypatch):
    monkeypatch.setattr(BS, "_PKL", tmp_path / "store_cache.pkl")
    monkeypatch.setattr(BS, "_store", {"AAA": _frame(DAYS[:3])})
    monkeypatch.setattr(BS, "_SAVE_PENDING", False)
    holder = _hold_publish_lock(tmp_path / BS.PUBLISH_LOCK)
    try:
        BS._save_pkl()
        assert not (tmp_path / "store_cache.pkl").exists() and BS._SAVE_PENDING
    finally:
        holder.communicate("\n", timeout=10)
    BS._save_pkl()                                               # the repair is over (the OS released the lock)
    assert (tmp_path / "store_cache.pkl").exists() and not BS._SAVE_PENDING


def test_a_repair_cannot_start_while_a_writer_holds_the_publish_lock(runtime, tmp_path):
    """Verification QT-05, adapted: an in-flight writer and a repair exclude each other."""
    from product import bhav_store_rebuild as rb

    manifest = tmp_path / "manifest.json"
    planned = rb.plan(runtime, tmp_path / "stage", manifest)
    assert planned["ready"]
    target = rb._pickle(runtime)
    holder = _hold_publish_lock(target.parent / rb.PUBLISH_LOCK)
    try:
        with pytest.raises(SystemExit, match="publish lock"):
            with rb._MaintenanceLock(target.parent, "test", wait_s=0.5):
                pass
        assert rb._sha256(target) == planned["target_sha256"]
    finally:
        holder.communicate("\n", timeout=10)
    rb.apply(manifest, "test approval", target_is_copy=True)
    assert rb._sha256(target) == planned["new_sha256"]


def test_a_corrected_archive_is_found_by_the_audit_and_left_to_the_gated_rebuild(runtime, tmp_path):
    """Verification QT-01 (corrected row) and QT-02, adapted: a periodic refresh never re-merges
    history; the audit detects a corrected archive, served or pending, and the gated rebuild repairs it."""
    from product import bhav_store_rebuild as rb
    from research.store_integrity_audit import audit

    day = DAYS[10]
    csv = rb._pickle(runtime).parent / day.strftime("%d%m%Y.csv")
    csv.write_text(_archive(day, [("AAA", 999.0), ("BBB", _official("BBB", 10))]))
    path = rb._pickle(runtime)
    payload = pickle.loads(path.read_bytes())
    payload["pending"] = {"GONE": _frame([day], 5.0)}               # the correction dropped this name
    path.write_bytes(pickle.dumps(payload))
    rep = audit(runtime)
    mismatches = [tuple(r[:3]) for r in rep["findings"]["VALUE_MISMATCH"]]
    assert ("AAA", day.date().isoformat(), "close") in mismatches
    assert "GONE" in rep["affected_symbols"]
    m = rb.plan(runtime, tmp_path / "stage", tmp_path / "manifest.json")
    assert m["ready"], m["gates"]
    assert rb._frames(m["new_pickle"])["AAA"].at[day, "close"] == 999.0


# ── GPT verification pass over 8a52e49d..9bfe1c70 (QT-01..QT-11): its sketches ──
# QT-03 (identity of re-parsed bytes) and QT-08 (same-date republication) do not
# apply: periodic refreshes no longer re-merge history; a correction is published
# only by the gated rebuild with the desk stopped. See the cycle-5 journal.
# Adaptations: the ``_incorporated`` monkeypatches are dropped (that mechanism
# was removed), and QT-05's in-process interleaving (apply inside a save's
# publish) is replaced by the two-process mutual-exclusion test above.

def _adversarial_store(monkeypatch, tmp_path, candidates, verified=None):
    monkeypatch.setattr(BS, "_BHAV_DIR", tmp_path)
    monkeypatch.setattr(BS, "_PKL", tmp_path / "store_cache.pkl")
    monkeypatch.setattr(BS, "_store", {})
    monkeypatch.setattr(BS, "_pending", {})
    monkeypatch.setattr(BS, "_store_last_day", None)
    monkeypatch.setattr(BS, "_store_sessions", 0)
    monkeypatch.setattr(BS, "_MIN_DAYS", 1)
    monkeypatch.setattr(BS, "_trading_days_back", lambda _: [d.date() for d in candidates])
    monkeypatch.setattr(BS, "_dates_on_disk",
                        lambda: [d.date() for d in (candidates if verified is None else verified)])
    monkeypatch.setattr(BS, "_is_valid_archive", lambda path, day: path.is_file())


@pytest.mark.parametrize("overlay", [True])   # False: see the adapted audit test
def test_adversarial_legacy_row_does_not_certify_archive(tmp_path, monkeypatch, overlay):
    prior, today = DAYS[:2]
    _adversarial_store(monkeypatch, tmp_path, [prior, today])
    for day, close in [(prior, 100.0), (today, 110.0)]:
        BS._day_path(day.date()).write_text(_archive(day, [("AAA", close)]))
    old = pd.concat([_frame([prior]), _frame([today], 777.0)])
    if overlay:
        old.attrs["quantterm_live_overlay_date"] = today.date().isoformat()
    BS._PKL.write_bytes(pickle.dumps({
        "store": {"AAA": old},
        "last_day": (prior if overlay else today).date(),
        "sessions": 1 if overlay else 2,
    }))

    BS.build_store(days=1)

    assert BS._store["AAA"].at[today, "close"] == 110.0


def test_adversarial_apply_rejects_correction_during_stage(runtime, tmp_path, monkeypatch):
    import os
    from product import bhav_store_rebuild as rb

    day = DAYS[10]
    path = rb._pickle(runtime).parent / day.strftime("%d%m%Y.csv")
    copy = rb.shutil.copy2
    manifest = tmp_path / "manifest.json"

    def copy_then_correct(src, dst):
        result = copy(src, dst)
        if Path(src) == path:
            old = path.stat()
            replacement = path.with_suffix(".new")
            replacement.write_text(_archive(day, [("AAA", 710.0), ("BBB", _official("BBB", 10))]))
            assert replacement.stat().st_size == old.st_size
            os.utime(replacement, ns=(old.st_atime_ns, old.st_mtime_ns + 5_000_000_000))
            replacement.replace(path)
        return result

    monkeypatch.setattr(rb.shutil, "copy2", copy_then_correct)
    monkeypatch.setattr(rb, "_held_open", lambda _: [])
    with pytest.raises(SystemExit, match="archive"):
        rb.plan(runtime, tmp_path / "stage", manifest)
        rb.apply(manifest, "test approval", target_is_copy=True)


def test_adversarial_loaded_launchd_job_blocks_maintenance(monkeypatch):
    from subprocess import CompletedProcess
    from product import bhav_store_rebuild as rb

    monkeypatch.setattr(rb.sys, "platform", "darwin")
    monkeypatch.setattr(rb.subprocess, "run", lambda args, **kw: (
        CompletedProcess(args, 0, "-\t9\tcom.quantterm.desk\n", "")
        if args[0] == "launchctl" else CompletedProcess(args, 1, "", "")
    ))

    assert rb._desk_running()


def test_adversarial_late_archive_does_not_freeze_live_quote(tmp_path, monkeypatch):
    """Verification QT-06, adapted: GPT's sketch set _store_sessions to 0, which sends
    build_store down the FULL-build path (a 1-archive window drops the name below the
    floor) instead of the claim's path: an older archive merged into a deep store
    while today's quote is live. The marker must survive and the next quote update."""
    from data import kite_client, nse_live, nse_universe

    late, prior, today = DAYS[:3]
    _adversarial_store(monkeypatch, tmp_path, [late, prior])
    BS._day_path(prior.date()).write_text(_archive(prior, [("AAA", 100.0)]))
    monkeypatch.setattr(BS, "_store", {"AAA": _frame([prior])})
    monkeypatch.setattr(BS, "_store_last_day", prior.date())
    monkeypatch.setattr(BS, "_store_sessions", 1)
    monkeypatch.setattr(nse_live, "_is_trading_now", lambda: True)
    monkeypatch.setattr(nse_live, "date", type("Clock", (), {"today": staticmethod(today.date)}))
    from core import market_clock
    monkeypatch.setattr(market_clock, "today_ist", today.date)     # the producer dates by the NSE clock (C7-08)
    monkeypatch.setattr(nse_universe, "get_nse_universe", lambda: [])
    monkeypatch.setattr(kite_client, "kite_credentials_available", lambda: False)
    prices = iter([777.0, 778.0])
    monkeypatch.setattr(nse_live, "fetch_live_snapshot", lambda: {
        "AAA": dict(open=100.0, high=800.0, low=100.0, close=next(prices), volume=1000.0),
    })
    assert nse_live.apply_live_to_store() == 1

    BS._day_path(late.date()).write_text(_archive(late, [("AAA", 95.0)]))
    BS.build_store(days=1)
    assert late in BS._store["AAA"].index                          # the late archive was merged
    assert BS._store["AAA"].attrs.get("quantterm_live_overlay_date") == today.date().isoformat()

    assert nse_live.apply_live_to_store() == 1
    assert BS._store["AAA"].at[today, "close"] == 778.0


@pytest.mark.parametrize("inside_build", [True, False])
def test_adversarial_unchanged_refresh_does_not_promote_historical_name(
    tmp_path, monkeypatch, inside_build,
):
    candidates = DAYS if inside_build else DAYS[1:]
    _adversarial_store(monkeypatch, tmp_path, candidates, verified=DAYS)
    for i, day in enumerate(DAYS):
        rows = [("OLD", 50.0)] if i == 0 else [("AAA", 100.0)]
        BS._day_path(day.date()).write_text(_archive(day, rows))
    BS._full_build([d.date() for d in candidates])
    order = BS.store_symbols()
    assert order == ["AAA"]
    reads = []
    read = BS._read_day

    def counted(day):
        reads.append(day)
        return read(day)

    monkeypatch.setattr(BS, "_read_day", counted)
    BS.build_store(days=len(candidates))

    assert (reads, BS.store_symbols()) == ([], order)


def test_adversarial_abandoned_maintenance_does_not_block_publication(tmp_path, monkeypatch):
    day = DAYS[0]
    _adversarial_store(monkeypatch, tmp_path, [day])
    monkeypatch.setattr(BS, "_store", {"AAA": _frame([day])})
    monkeypatch.setattr(BS, "_store_last_day", day.date())
    monkeypatch.setattr(BS, "_store_sessions", 1)
    (tmp_path / "MAINTENANCE.lock").write_text(
        '{"pid":2147483647,"since":"2000-01-01T00:00:00+0000"}'
    )

    BS._save_pkl()

    assert BS._PKL.is_file()


@pytest.mark.parametrize("defect", ["string_index", "eligible_name_only_pending"])
def test_adversarial_audit_reports_reader_visible_missing_history(runtime, defect):
    from product import bhav_store_rebuild as rb
    from research.store_integrity_audit import audit

    path = rb._pickle(runtime)
    payload = pickle.loads(path.read_bytes())
    if defect == "string_index":
        frame = payload["store"]["AAA"]
        frame.index = frame.index.strftime("%Y-%m-%d")
    else:
        payload["pending"] = {"AAA": payload["store"].pop("AAA")}
    path.write_bytes(pickle.dumps(payload))

    report = audit(runtime)
    kinds = report["contamination_kinds"] + report["incomplete_kinds"]
    assert any(row[0] == "AAA" for kind in kinds for row in report["findings"][kind])


def test_adversarial_rebuild_accounts_for_every_duplicate(runtime, tmp_path):
    from product import bhav_store_rebuild as rb

    path = rb._pickle(runtime)
    payload = pickle.loads(path.read_bytes())
    frame = payload["store"]["AAA"]
    one = frame.iloc[:1]
    payload["store"]["AAA"] = pd.concat([frame, one, one]).sort_index()
    path.write_bytes(pickle.dumps(payload))

    plan = rb.plan(runtime, tmp_path / "stage", tmp_path / "manifest.json")

    assert plan["diff"]["removed"] == 3  # Two duplicates and the fixture's BBB orphan.
    assert plan["gates"]["removed_rows_are_exactly_the_contamination"]


# ── GPT cycle-6 review (QT-01, QT-06): sketches, adapted only where noted ─────

def test_walk_forward_never_analyzes_future_bars(tmp_path, monkeypatch):
    """QT-01: an out-of-order frame is refused by the walk-forward, not fed to it."""
    from core import session_window
    from scan import signal_backtest as SB, unified_scanner as US

    days = pd.bdate_range("2025-01-01", periods=160)
    order = list(range(len(days)))
    order[69], order[99] = order[99], order[69]
    frame = pd.DataFrame(
        {"open": 100., "high": 101., "low": 99., "close": 100.},
        index=days[order],
    )
    seen = []

    class Scanner:
        def __init__(self, **kwargs):
            pass

        def _analyze(self, symbol, hist):
            seen.append((hist.index[-1], hist.index.max()))

    monkeypatch.setattr(US, "UnifiedScanner", Scanner)
    monkeypatch.setattr(BS, "store_symbols", lambda: ["AAA"])
    monkeypatch.setattr(BS, "get_ohlcv", lambda _: frame)
    monkeypatch.setattr(BS, "_dates_on_disk", lambda: list(days.date))
    monkeypatch.setattr(session_window, "official_session_index",
                        lambda _: {str(d.date()): i for i, d in enumerate(days)})
    monkeypatch.setattr(SB, "_nifty_regime_series", lambda: None)
    monkeypatch.setattr(SB, "_OUT", tmp_path / "report.json")
    monkeypatch.setenv("QT_ALPHA_LEDGER", str(tmp_path / "alpha.db"))

    SB.run_backtest()
    assert all(latest <= decision for decision, latest in seen)


def test_a_loaded_store_serves_every_frame_in_date_order(tmp_path, monkeypatch):
    """QT-01: the 6 production frames saved out of order are served in date order."""
    days = DAYS[:6]
    bad = _frame(days)
    bad = pd.concat([bad.iloc[:3], bad.iloc[[4, 3]], bad.iloc[5:]])
    bad.attrs["quantterm_live_overlay_date"] = days[-1].date().isoformat()
    monkeypatch.setattr(BS, "_PKL", tmp_path / "store_cache.pkl")
    (tmp_path / "store_cache.pkl").write_bytes(pickle.dumps(
        {"store": {"AAA": bad, "BBB": _frame(days)}, "pending": {"YOUNG": bad.copy()},
         "last_day": days[-1].date(), "sessions": 6}))
    monkeypatch.setattr(BS, "_dates_on_disk", lambda: [d.date() for d in days])
    assert BS._load_pkl()
    assert BS._store["AAA"].index.is_monotonic_increasing
    assert BS._pending["YOUNG"].index.is_monotonic_increasing
    assert BS._store["AAA"].attrs["quantterm_live_overlay_date"] == days[-1].date().isoformat()
    assert list(BS._store["AAA"].index) == list(days)


def test_rollback_recovers_an_interrupted_bad_swap(runtime, tmp_path, monkeypatch):
    """QT-06."""
    from product import bhav_store_rebuild as RB

    path = tmp_path / "manifest.json"
    m = RB.plan(runtime, tmp_path / "staging", path)
    assert m["ready"]
    target = RB._pickle(runtime)
    replace = RB.os.replace

    def damaged_swap(src, dst):
        replace(src, dst)
        if Path(dst) == target and Path(src).name.startswith(".rebuild-"):
            target.write_bytes(b"damaged after rename")

    monkeypatch.setattr(RB, "_held_open", lambda _: [])
    monkeypatch.setattr(RB.os, "replace", damaged_swap)
    with pytest.raises(SystemExit, match="run rollback"):
        RB.apply(path, "test", target_is_copy=True)
    assert RB.rollback(path, target_is_copy=True)["rollback_ok"]
    assert RB._sha256(target) == m["target_sha256"]


def test_rollback_restores_a_missing_target_from_a_verified_offline_copy(runtime, tmp_path, monkeypatch):
    """QT-06: the swap was interrupted, the target is gone AND the same-image
    backup is lost; the runbook's SSD copy (verified by sha256) restores it."""
    import shutil
    from product import bhav_store_rebuild as RB

    path = tmp_path / "manifest.json"
    m = RB.plan(runtime, tmp_path / "staging", path)
    target = RB._pickle(runtime)
    ssd_copy = tmp_path / "store_cache_pre_repair.pkl"
    shutil.copy2(target, ssd_copy)
    m.update({"state": "swapping", "backup_pickle": str(tmp_path / "gone.pkl")})
    path.write_text(json.dumps(m))
    target.unlink()
    monkeypatch.setattr(RB, "_held_open", lambda _: [])
    with pytest.raises(SystemExit, match="no backup"):
        RB.rollback(path, target_is_copy=True)
    wrong = tmp_path / "wrong.pkl"
    wrong.write_bytes(b"not it")
    with pytest.raises(SystemExit, match="does not match"):
        RB.rollback(path, target_is_copy=True, backup_path=wrong)
    out = RB.rollback(path, target_is_copy=True, backup_path=ssd_copy)
    assert out["rollback_ok"] and out["rolled_back_from"] == "interrupted swap"
    assert RB._sha256(target) == m["target_sha256"]


def test_a_verified_apply_still_never_discards_a_newer_cache(runtime, tmp_path, monkeypatch):
    """QT-06 must not weaken ST-03: after a VERIFIED apply, a newer save is kept."""
    from product import bhav_store_rebuild as RB

    path = tmp_path / "manifest.json"
    RB.plan(runtime, tmp_path / "staging", path)
    monkeypatch.setattr(RB, "_held_open", lambda _: [])
    assert RB.apply(path, "test", target_is_copy=True)["applied_ok"]
    RB._pickle(runtime).write_bytes(pickle.dumps({"store": {"NEW": 1}}))
    with pytest.raises(SystemExit, match="newer cache"):
        RB.rollback(path, target_is_copy=True)


# ── GPT cycle-6 verification pass (R1, R7) ──


# R1: a walk-forward from code without the chronology guard (schema < 6) cannot certify
def test_parent_walk_forward_cannot_certify_a_saved_buy(tmp_path, monkeypatch):
    import json
    from product import scan_store as SS
    from scan import calibration_snapshot as CS
    from tests.evidence_fixtures import proven_provenance, seal_snapshot

    snap = seal_snapshot({
        "snapshot_id": "parent",
        "backtest_provenance": proven_provenance(
            report_schema=5, code_sha="783f342f"
        ),
        "backtest_edges": {
            "VCP": {"mean": .5, "lower": .2, "n": 90, "blocks": 30,
                    "inference": "OVERLAP_ROBUST", "family": True}
        },
    })
    monkeypatch.setattr(CS, "DEFAULT_DIR", tmp_path)
    monkeypatch.setattr(SS, "scan_source_status", lambda _: (True, ""))
    (tmp_path / "current.json").write_text('{"snapshot_id":"parent"}')
    (tmp_path / "parent.json").write_text(json.dumps(snap))
    scan = tmp_path / "scan.json"
    scan.write_text(json.dumps({
        "schema_version": 2,
        "evidence_identity": {"id": "parent"},
        "records": [{"symbol": "TCS", "signals": ["VCP"], "verdict": "BUY",
                     "edge_evidence_id": "parent"}],
    }))
    assert SS.load_scan(scan)["records"][0]["verdict"] == "WATCH"

# R7
def test_unverified_rollback_can_be_retried(runtime, tmp_path, monkeypatch):
    from product import bhav_store_rebuild as RB

    path = tmp_path / "manifest.json"
    RB.plan(runtime, tmp_path / "staging", path)
    monkeypatch.setattr(RB, "_held_open", lambda _: [])
    RB.apply(path, "approve", target_is_copy=True)
    target = RB._pickle(runtime)
    replace = RB.os.replace

    def damage_first_restore(src, dst):
        replace(src, dst)
        if Path(dst) == target and Path(src).name.startswith(".rollback-"):
            target.write_bytes(b"failed read-back")

    with monkeypatch.context() as patch:
        patch.setattr(RB.os, "replace", damage_first_restore)
        with pytest.raises(SystemExit, match="did not restore"):
            RB.rollback(path, target_is_copy=True)
    assert json.loads(path.read_text())["state"] == "rollback_unverified"
    assert RB.rollback(path, target_is_copy=True)["rollback_ok"]


def test_a_different_case_volumes_spelling_is_still_production(monkeypatch):
    """macOS volumes are case-insensitive and realpath keeps the spelling given."""
    from product import bhav_store_rebuild as RB
    from research import store_integrity_audit as AU

    monkeypatch.setattr(RB, "_desk_running", lambda: ["launchd:com.quantterm.desk:1"])
    with pytest.raises(SystemExit, match="desk is running"):
        RB._check_desk(Path("/volumes/QuantTermStorage/QuantTerm/runtime/logs/bhav/store_cache.pkl"), True)
    with pytest.raises(SystemExit, match="refusing"):
        RB._not_live(Path("/VOLUMES/QuantTermStorage/x"), "the staging root")
    with pytest.raises(SystemExit, match="refusing"):
        AU._refuse_live("/volumes/QuantTermStorage/QuantTerm/runtime")


# ── GPT cycle-7 verification pass (C7V): sketches ──

# GPT C7V-05 (cycle-7 verification sketch)
# (adapted: the fake audit result gains 'affected_sessions', a key main() prints)
def test_c7v_rebuild_checks_the_entry_it_replaces(tmp_path, monkeypatch):
    from product import bhav_store_rebuild as rb

    live = tmp_path / "volumes" / "test-volume"
    live.mkdir(parents=True)
    backing = tmp_path / "old.pkl"
    backing.write_bytes(b"old")
    target = live / "store_cache.pkl"
    target.symlink_to(backing)
    monkeypatch.setattr(rb, "_on_live_volume",
                        lambda p: live in Path(p).parents)
    monkeypatch.setattr(rb, "_desk_running", lambda: ["desk"])

    with pytest.raises(SystemExit, match="desk is running"):
        rb._check_desk(target, target_is_copy=True)


def test_c7v_audit_refuses_live_output(tmp_path, monkeypatch):
    from research import store_integrity_audit as audit

    out = tmp_path / "report.json"
    resolve = Path.resolve
    monkeypatch.setattr(
        Path, "resolve",
        lambda p, *a, **k: Path("/volumes/test-volume/report.json")
        if p == out else resolve(p, *a, **k),
    )
    monkeypatch.setattr(audit, "audit", lambda *a, **k: dict.fromkeys(
        ("pickle", "archives", "counts", "contamination_kinds", "incomplete_kinds",
          "affected_sessions"), {}))
    with pytest.raises(SystemExit, match="live runtime"):
        audit.main(["--root", str(tmp_path), "--out", str(out)])
    assert not out.exists()
