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
# Each failed on 8a52e49d at its claim-bearing assertion.

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
    pkl = runtime / rb.PICKLE
    payload = pickle.loads(pkl.read_bytes())
    for symbol, frame in payload["store"].items():
        payload["store"][symbol] = frame.drop(day, errors="ignore")
    pkl.write_bytes(pickle.dumps(payload))

    m = rb.plan(runtime, tmp_path / "stage", tmp_path / "manifest.json")
    if m["ready"]:
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
    target = runtime / rb.PICKLE
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

    pkl = runtime / rb.PICKLE
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

    pkl = runtime / rb.PICKLE
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

    pkl = runtime / rb.PICKLE
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

    pkl = runtime / rb.PICKLE
    payload = pickle.loads(pkl.read_bytes())
    extra = pd.date_range(end=DAYS[0] - pd.Timedelta(days=1), periods=2001)
    payload["store"]["AAA"] = pd.concat([_frame(extra), payload["store"]["AAA"]])
    pkl.write_bytes(pickle.dumps(payload))

    m = rb.plan(runtime, tmp_path / "stage", tmp_path / "manifest.json")
    assert m["diff"]["removed"] == 2002  # Includes the fixture's BBB orphan.
    assert m["gates"]["removed_rows_are_exactly_the_contamination"]


def test_no_store_writer_saves_while_the_store_is_under_maintenance(tmp_path, monkeypatch):
    monkeypatch.setattr(BS, "_PKL", tmp_path / "store_cache.pkl")
    monkeypatch.setattr(BS, "_store", {"AAA": _frame(DAYS[:3])})
    (tmp_path / BS.MAINTENANCE_LOCK).write_text("{}")
    BS._save_pkl()
    assert not (tmp_path / "store_cache.pkl").exists()
    (tmp_path / BS.MAINTENANCE_LOCK).unlink()
    BS._save_pkl()
    assert (tmp_path / "store_cache.pkl").exists()
