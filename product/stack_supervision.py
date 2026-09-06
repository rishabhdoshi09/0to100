"""Testable stack-supervision primitive.

The production launchers ``scripts/run_quantterm.sh`` and
``scripts/run_quantterm_complete.sh`` implement this contract in their watch
loops: a dead supervised child is relaunched; the parent stays; live siblings
are not killed or duplicated.

This module is the deterministic form of that contract. Tests exercise it with
real child processes rather than inspecting launcher strings.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable


def pid_alive(pid: int | None) -> bool:
    """True when ``pid`` is a running (non-zombie) process.

    A zombie still occupies a PID and accepts ``kill(pid, 0)``. Supervisors must
    treat that as dead so the failed child is relaunched instead of trusted.
    """
    try:
        value = int(pid or 0)
    except (TypeError, ValueError):
        return False
    if value <= 1:
        return False
    try:
        os.kill(value, 0)
    except OSError:
        return False
    try:
        with open(f"/proc/{value}/stat", encoding="utf-8") as handle:
            stat = handle.read()
        state = stat[stat.rfind(")") + 2 :].split()[0]
    except OSError:
        return False
    except (IndexError, ValueError):
        return True
    return state not in {"Z", "X"}


@dataclass
class SupervisedChild:
    """One named child owned by a supervisor."""

    name: str
    launcher: Callable[[], int]
    pid: int | None = None


@dataclass
class TickResult:
    restarted: list[str] = field(default_factory=list)
    pids: dict[str, int | None] = field(default_factory=dict)
    parent_pid: int = 0


def supervise_tick(children: dict[str, SupervisedChild], *, parent_pid: int | None = None) -> TickResult:
    """One supervisor cycle.

    A child whose pid is missing or dead is launched exactly once. A live child
    is left untouched, so a tick cannot create a duplicate.
    """
    parent = int(parent_pid or os.getpid())
    restarted: list[str] = []
    for name, child in children.items():
        if pid_alive(child.pid):
            continue
        child.pid = int(child.launcher())
        restarted.append(name)
    return TickResult(
        restarted=restarted,
        pids={name: child.pid for name, child in children.items()},
        parent_pid=parent,
    )
