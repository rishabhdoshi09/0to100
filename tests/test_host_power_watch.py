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
