"""Opt-in, prospective relative-strength paper observation in the close job."""
from __future__ import annotations

from contextlib import closing
from datetime import date, datetime, time, timedelta
from hashlib import sha256
import json
import random
import sqlite3
from types import SimpleNamespace

import numpy as np
import pandas as pd

from core.runtime_paths import REPO_ROOT, logs_path
from product.challenger_shadow import FREEZE_MARGIN, IST, entry_deadline
from research.opportunity_discovery import SLOTS, STOP, TARGET, _features, _path, _rules, _summary, _trade

SPEC = REPO_ROOT / "docs" / "research" / "opportunity_relative_strength_frozen.json"
SPEC_SHA = "3ad2d8db51b3afb3a456cf4d2ed2a601f0d8da86cc45075c0145b5555f9804f8"
REGISTRATION = "relative_strength_shadow_registration.json"
HORIZON = 10
COST = 0.0032
SCHEDULED_CYCLES = 24
MIN_MATURE_CYCLES = 20
DISCOVERY_COUNTS = {"target": 19, "stop": 44, "expiry": 6}


def _digest(payload):
    return sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def implementation_sha():
    paths = ("product/relative_strength_shadow.py", "research/opportunity_discovery.py",
             "core/evaluation_spec.py", "gauntlet/validator.py",
             "research/intelligence/data/nse_calendar.py")
    return _digest({name: sha256((REPO_ROOT / name).read_bytes()).hexdigest() for name in paths})


def historical_forecast():
    n = sum(DISCOVERY_COUNTS.values())
    return {"label": "Strategy-level historical estimate", "resolved_fills": n,
            "source_sha256": _spec()["discovery_report_sha256"],
            "exit_probability": {k: (v+.5)/(n+1.5) for k,v in DISCOVERY_COUNTS.items()}}


def _spec():
    content = SPEC.read_bytes()
    if sha256(content).hexdigest() != SPEC_SHA:
        raise ValueError("Frozen relative-strength specification changed")
    return json.loads(content)


def _dates(first, last):
    from research.intelligence.data.nse_calendar import (
        covered_years, documented_special_sessions, is_session, load_holidays, load_manifest,
    )
    manifest_path = logs_path("nse_calendar_manifest.json")
    manifest = load_manifest(manifest_path if manifest_path.is_file() else None)
    holidays = load_holidays()
    covered = covered_years(holidays, manifest)
    special = documented_special_sessions(manifest)
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    if any(day.year not in covered for day in days):
        raise ValueError("Verified exchange calendar coverage is required")
    return [day for day in days if is_session(day, holidays) or day.isoformat() in special]


def _inputs(days, symbols):
    from data.bhavcopy_store import _day_path
    from data.corporate_actions import _events_path, load_events
    from gauntlet.validator import scoped_equity_frames

    action_file = _events_path()
    action_bytes = action_file.read_bytes()
    actions = load_events(action_file)
    if not actions or action_file.read_bytes() != action_bytes:
        raise ValueError("Stable, verified corporate-action ledger is required")
    archives = []
    for day in days:
        path = _day_path(day)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            data = None
        archives.append((day, data))
    frames, report = scoped_equity_frames(archives, symbols, actions)
    report["archive_sha256"] = {d.isoformat(): sha256(data).hexdigest() if data else None
                                for d, data in archives}
    digest = _digest({"actions": sha256(action_bytes).hexdigest(),
                      "archives": report["archive_sha256"]})
    return frames, report, digest


def shortlist(day: date) -> dict:
    """Read official closes through the specified session, with no broker reads."""
    spec = _spec()
    # The frozen policy needs 60 sessions and the basket needs its trailing return.
    days = _dates(day - timedelta(days=110), day)
    if not days or days[-1] != day or len(days) < 60:
        raise ValueError("Decision date must be a verified session with sufficient history")
    frames, integrity, digest = _inputs(days, spec["cohort"])
    if integrity["market_wide_excluded_sessions"]:
        raise ValueError(f"Unverified archive window: {integrity['market_wide_excluded_sessions']}")
    index = pd.DatetimeIndex(days)
    prices = pd.DataFrame({symbol: f.close for symbol, f in frames.items()}).reindex(index)
    daily = prices.pct_change(fill_method=None).mean(axis=1)
    market_30 = float((1 + daily.iloc[-30:]).prod(min_count=30) - 1)
    if not np.isfinite(market_30):
        raise ValueError("Matched market return is unavailable")
    rows = _features(frames, index, len(index)-1, SimpleNamespace(_analyze=lambda *_: None),
                     {}, market_30*100)
    picked = _rules(rows, True, market_30)["relative_strength"][:SLOTS]
    levels = []
    for symbol in picked:
        close = float(frames[symbol].close.iloc[-1])
        levels.append({
            "symbol": symbol, "return_30d": float(rows[symbol]["r30"]),
            "market_return_30d": market_30, "entry": "Next official session open",
            "reference_close": close, "indicative_stop": round(close*(1-STOP), 2),
            "indicative_target": round(close*(1+TARGET), 2), "horizon_sessions": HORIZON,
            "reason": "30-session return beat the cohort and price closed above its 50-session average.",
        })
    return {"session": day.isoformat(), "eligible": sorted(rows), "selected": picked,
            "opportunities": levels, "history_sha": digest, "spec_sha": SPEC_SHA,
            "decision_archive_sha": integrity["archive_sha256"][day.isoformat()],
            "historical_forecast": historical_forecast()}


def _registration(value):
    first = date.fromisoformat(value["first_decision_session"])
    registered = datetime.fromisoformat(value["registered_at"])
    if (value["spec_sha256"] != SPEC_SHA or value["mode"] != "SHADOW_ONLY"
            or value["implementation_sha256"] != implementation_sha()
            or registered.tzinfo is None or registered.astimezone(IST).date() >= first):
        raise ValueError("Registration must bind the frozen spec before the first decision")
    return first, registered


def _result(record, frames):
    forward = pd.DatetimeIndex(record["forward_sessions"])
    trades = {s: _trade(frames.get(s, pd.DataFrame()), forward, COST) for s in record["eligible"]}
    selected = [trades[s] for s in record["selected"]]
    random_control = [trades[s] for s in record["random"]]
    all_trades = list(trades.values())
    market = np.array(_path(all_trades, HORIZON, max(1, len(all_trades))))
    exposure = len(selected) / SLOTS
    return {
        "ordinal": record["ordinal"], "day": record["session"], "opportunities": len(record["selected"]),
        "path": _path(selected, HORIZON), "market_path": (1 + (market-1)*exposure).tolist(),
        "random_path": _path(random_control, HORIZON), "trades": [[t[2], t[0]] for t in selected],
    }


def nightly() -> dict:
    """Called by the existing forward learning job. Absent registration is inert."""
    registration = logs_path("product", REGISTRATION)
    if not registration.is_file():
        return {"status": "DISABLED"}
    spec = _spec()
    raw = registration.read_bytes()
    reg = json.loads(raw)
    first, registered = _registration(reg)
    now = datetime.now(IST)
    if now.date() < first:
        return {"status": "AWAITING_START"}
    try:
        days = _dates(first, now.date() + timedelta(days=21))
    except ValueError:
        # Evaluate completed windows in covered years even if the next year lacks a calendar.
        days = _dates(first, date(now.year, 12, 31))
    if not days or days[0] != first:
        raise ValueError("Registered start is not an exchange session")
    paths = logs_path("product", "relative_strength_shadow.db")
    identity = sha256(raw).hexdigest()
    if not paths.parent.is_dir():
        raise ValueError("Approved registration directory missing")
    with closing(sqlite3.connect(paths, timeout=30)) as db, db:
        db.execute("PRAGMA synchronous=FULL")
        db.execute("CREATE TABLE IF NOT EXISTS registration (id INTEGER PRIMARY KEY, digest TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS cycles (session TEXT PRIMARY KEY, prediction TEXT, outcome TEXT)")
        db.execute("INSERT OR IGNORE INTO registration VALUES (1, ?)", (identity,))
        if db.execute("SELECT digest FROM registration WHERE id=1").fetchone()[0] != identity:
            raise ValueError("Registration changed after observation began")
        for i in range(0, min(len(days)-HORIZON, SCHEDULED_CYCLES*HORIZON), HORIZON):
            signal = days[i]
            if signal > now.date():
                break
            key = signal.isoformat()
            forward = days[i+1:i+HORIZON+1]
            read = db.execute("SELECT prediction, outcome FROM cycles WHERE session=?", (key,)).fetchone()
            if read is None:
                if now >= entry_deadline(signal)-FREEZE_MARGIN:
                    db.execute("INSERT OR IGNORE INTO cycles VALUES (?, NULL, NULL)", (key,))
                    continue
                if now < datetime.combine(signal, time(18), tzinfo=IST) or now < registered:
                    continue
                prediction = shortlist(signal)
                rng = random.Random(spec["random_seed"] + i//HORIZON)
                prediction.update(ordinal=i, frozen_at=datetime.now(IST).isoformat(),
                                  forward_sessions=[d.isoformat() for d in forward],
                                  random=rng.sample(prediction["eligible"], len(prediction["selected"])))
                if datetime.fromisoformat(prediction["frozen_at"]) >= entry_deadline(signal)-FREEZE_MARGIN:
                    continue
                prediction["payload_sha256"] = _digest(prediction)
                db.execute("INSERT OR IGNORE INTO cycles VALUES (?, ?, NULL)", (key, json.dumps(prediction)))
                continue
            if read[0] is None or read[1] is not None or now.date() <= forward[-1]:
                continue
            prediction = json.loads(read[0])
            payload_sha = prediction.pop("payload_sha256", None)
            if (payload_sha != _digest(prediction)
                    or prediction["spec_sha"] != SPEC_SHA
                    or not registered <= datetime.fromisoformat(prediction["frozen_at"])
                    < entry_deadline(signal)-FREEZE_MARGIN
                    or prediction["forward_sessions"] != [d.isoformat() for d in forward]):
                raise ValueError("Frozen cycle provenance or calendar differs")
            frames, integrity, evidence = _inputs([signal, *forward], prediction["eligible"])
            if integrity["market_wide_excluded_sessions"]:
                continue
            if integrity["archive_sha256"].get(key) != prediction["decision_archive_sha"]:
                raise ValueError("Decision archive changed after the cycle was frozen")
            result = _result(prediction, frames)
            result["outcome_sha"] = evidence
            db.execute("UPDATE cycles SET outcome=? WHERE session=? AND outcome IS NULL",
                       (json.dumps(result), key))
        records = [json.loads(row[0]) for row in db.execute(
            "SELECT outcome FROM cycles WHERE outcome IS NOT NULL ORDER BY session")]
        missed = db.execute("SELECT count(*) FROM cycles WHERE prediction IS NULL").fetchone()[0]
    stats = _summary(records, 0, HORIZON) if records else {}
    forecast = historical_forecast()["exit_probability"]
    scores = [sum((p - (outcome == k))**2 for k, p in forecast.items())
              for record in records for outcome, ret in record["trades"] if ret is not None]
    counts = stats.get("counts", {})
    resolved = stats.get("resolved", 0)
    stats["prospective_exit_rate"] = {k: counts.get(k, 0)/resolved for k in forecast} if resolved else {}
    stats["prospective_brier_using_frozen_historical_forecast"] = float(np.mean(scores)) if scores else None
    complete = len(records) + missed == SCHEDULED_CYCLES
    if not complete:
        status = "OBSERVING"
    elif len(records) < MIN_MATURE_CYCLES or stats.get("inference_status") != "OK":
        status = "INSUFFICIENT_EVIDENCE"
    elif (stats["matched_market_lower_95_pct_per_cycle"] > 0
          and (stats.get("net_r_per_resolved") or 0) > 0
          and stats["return_pct"] > 0):
        status = "MEETS_SHADOW_CRITERIA"
    else:
        status = "EDGE_NOT_CONFIRMED"
    return {"status": status, "mature_cycles": len(records), "missed_cycles": missed, "metrics": stats}
