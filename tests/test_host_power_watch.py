"""Battery warning for the Mac host (2026-11-11: battery drained to ~12%, the
USB runtime disk lost power and the desk stopped with no warning)."""
from __future__ import annotations

from product.host_power_watch import PowerWatch, read_power

AC = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t36%; charging; 1:34 remaining present: true\n"
BATT = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t{p}%; discharging; 2:10 remaining present: true\n"


def test_reads_pmset_power_source_and_percent():
    assert read_power(lambda argv: AC) == {"source": "AC", "percent": 36}
    assert read_power(lambda argv: BATT.format(p=12)) == {"source": "BATTERY", "percent": 12}
    assert read_power(lambda argv: "garbage") is None
    assert read_power(lambda argv: (_ for _ in ()).throw(OSError("no pmset"))) is None


def _watch(clock):
    sent = []
    return PowerWatch(sent.append, now=lambda: clock[0], low_pct=40, repeat_s=900), sent


def test_alerts_once_on_switch_to_battery_and_on_return():
    clock = [0.0]
    watch, sent = _watch(clock)
    watch.observe({"source": "AC", "percent": 90})
    assert sent == []                                       # AC at start: silent
    watch.observe({"source": "BATTERY", "percent": 80})
    assert len(sent) == 1 and "BATTERY (80%)" in sent[0]
    clock[0] = 2000
    watch.observe({"source": "BATTERY", "percent": 70})     # above low: no repeat
    assert len(sent) == 1
    watch.observe({"source": "AC", "percent": 70})
    assert len(sent) == 2 and "back on AC" in sent[1]


def test_repeats_while_low_on_battery_every_interval():
    clock = [0.0]
    watch, sent = _watch(clock)
    watch.observe({"source": "BATTERY", "percent": 35})
    clock[0] = 600
    watch.observe({"source": "BATTERY", "percent": 30})
    assert len(sent) == 1                                   # within 15 min
    clock[0] = 901
    watch.observe({"source": "BATTERY", "percent": 25})
    assert len(sent) == 2 and "25%" in sent[1]


def test_unknown_power_and_send_failure_are_harmless():
    def boom(_msg):
        raise RuntimeError("telegram down")

    watch = PowerWatch(boom, now=lambda: 0.0)
    assert watch.observe(None) is None
    assert watch.observe({"source": "BATTERY", "percent": 10})   # still reported, no raise


# ── Stage 1 finding F1 (cycle 8): flush before a lid-close sleep can drop the disk ──

def test_the_poll_sees_the_19_november_window():
    import product.host_power_watch as hpw
    assert hpw.POLL_S <= 10        # the charger went 21 s before the lid closed


def test_switching_to_battery_flushes_at_once_then_periodically():
    clock = [0.0]
    flushes = []
    watch = PowerWatch(lambda m: None, now=lambda: clock[0], flush=lambda: flushes.append(clock[0]), flush_s=30)
    watch.observe({"source": "AC", "percent": 90})
    assert flushes == []                                    # on AC: never
    watch.observe({"source": "BATTERY", "percent": 89})
    assert flushes == [0.0]                                 # immediately on the switch
    clock[0] = 20
    watch.observe({"source": "BATTERY", "percent": 88})
    assert flushes == [0.0]                                 # not yet
    clock[0] = 31
    watch.observe({"source": "BATTERY", "percent": 88})
    assert flushes == [0.0, 31]
    clock[0] = 35
    watch.observe({"source": "AC", "percent": 88})
    assert flushes == [0.0, 31]                             # on AC: never
    clock[0] = 40                                           # only 9 s after the last flush
    watch.observe({"source": "BATTERY", "percent": 88})     # a new switch flushes at once regardless
    assert flushes == [0.0, 31, 40]


def test_the_alert_names_the_lid_and_a_flush_failure_is_harmless():
    sent = []
    watch = PowerWatch(sent.append, now=lambda: 0.0, flush=lambda: (_ for _ in ()).throw(OSError("usb gone")))
    assert watch.observe({"source": "BATTERY", "percent": 50})
    assert "lid" in sent[0] and "OPEN" in sent[0]


def test_flush_runtime_writes_syncs_and_full_fsyncs_the_root(tmp_path, monkeypatch):
    import os
    import core.atomic_io as aio
    import product.host_power_watch as hpw
    calls = []
    monkeypatch.setattr(os, "sync", lambda: calls.append("sync"))
    monkeypatch.setattr(aio, "_sync_directory", lambda d: calls.append(("dir", str(d))))
    hpw.flush_runtime_writes(tmp_path)
    assert calls == ["sync", ("dir", str(tmp_path))]
