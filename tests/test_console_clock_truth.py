"""One clock in one log stream, and it says which clock it is.

Found by reading the stack's own log while it ran. market_ops stamped lines
with the host clock and the autonomy console stamped heartbeats in IST, both
as a bare HH:MM:SS into the same stdout:

    [10:57:24] MARKET OPS PROGRESS  ...
    [16:27:24] HEARTBEAT   pid=2784 · state=DATA_BLOCKED ...

Nothing crashed. Every line was individually correct. The log simply told an
operator the heartbeat was five and a half hours in the future — during an
incident, that is the difference between "the worker is alive" and "the clock
has jumped".
"""
from __future__ import annotations

import re

from core.market_clock import console_stamp, now_ist

STAMP = re.compile(r"^\d{2}:\d{2}:\d{2} IST$")


def test_the_shared_stamp_names_its_timezone():
    assert STAMP.match(console_stamp()), console_stamp()


def test_the_shared_stamp_is_ist():
    assert console_stamp().startswith(now_ist().strftime("%H:%M"))


def test_market_ops_and_the_autonomy_console_use_the_same_clock():
    from research.autonomy.console_runtime import _stamp

    ops = console_stamp()
    autonomy = _stamp()
    assert STAMP.match(autonomy), autonomy
    # Same minute is enough: the defect was a five-and-a-half-hour gap.
    assert ops[:5] == autonomy[:5]


def test_market_ops_lines_carry_the_labelled_stamp(capsys):
    from operations.market_ops import _emit

    _emit("PROGRESS", "sample")
    line = capsys.readouterr().out.strip()
    assert line.startswith("["), line
    assert "IST]" in line, line


def test_the_autonomy_console_lines_carry_the_labelled_stamp(capsys):
    from research.autonomy.console_runtime import _emit

    _emit("HEARTBEAT", "sample")
    line = capsys.readouterr().out.strip()
    assert "IST]" in line, line


def test_no_console_emitter_prints_a_bare_local_timestamp():
    """The invariant, not the two instances: an unlabelled stamp is how two
    different clocks coexisted in one stream without anyone noticing."""
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    listed = subprocess.run(
        ["git", "ls-files", "*.py"], cwd=root,
        capture_output=True, text=True, check=True,
    ).stdout.split()
    skip = ("tests/", "venv/", "ui/", "scripts/", "fintel/", "legacy_app.py")
    offenders = []
    for rel in listed:
        if rel.startswith(skip):
            continue
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), 1):
            code = line.split("#", 1)[0]
            if 'time.strftime("%H:%M:%S")' in code and "console_stamp" not in text[:2000]:
                # A bare wall-clock stamp is only acceptable as the fallback
                # inside the shared helper's own except branch.
                if "except" not in text[max(0, text.find(line) - 200):text.find(line)]:
                    offenders.append(f"{rel}:{number}")
    assert not offenders, (
        "these print an unlabelled host-clock timestamp into a shared log "
        f"stream: {offenders}"
    )
