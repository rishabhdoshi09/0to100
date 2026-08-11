"""
🎚️ Automation operating modes (Phase P).

A single explicit ladder from OFF to FULL_AUTO. This milestone makes PAPER_AUTO work
end-to-end; every LIVE mode is DISABLED here and guarded by `assert_no_live()` so nothing can
route to a broker as a shortcut. The interfaces are live-compatible, but the user-owned live
gate (spec.USER_APPROVED) is the only door and it is never crossed automatically.
"""
from __future__ import annotations

OFF = "OFF"
RESEARCH_ONLY = "RESEARCH_ONLY"
SHADOW = "SHADOW"                 # evaluate + decide, but place NO paper orders
PAPER_AUTO = "PAPER_AUTO"        # full automatic PAPER loop (this milestone)
# Conceptual alias used by the forward-evidence system — same execution path as PAPER_AUTO.
PAPER_FORWARD_EVIDENCE = "PAPER_FORWARD_EVIDENCE"
PAPER_PAUSED = "PAPER_PAUSED"    # manage/exit existing, open NO new entries
LIMITED_LIVE = "LIMITED_LIVE"    # disabled this milestone
GUARDED_LIVE = "GUARDED_LIVE"    # disabled this milestone
FULL_AUTO = "FULL_AUTO"          # disabled this milestone
LIQUIDATE_ONLY = "LIQUIDATE_ONLY"  # only close positions
HALTED = "HALTED"                # do nothing

MODES = (OFF, RESEARCH_ONLY, SHADOW, PAPER_AUTO, PAPER_FORWARD_EVIDENCE, PAPER_PAUSED,
         LIMITED_LIVE, GUARDED_LIVE, FULL_AUTO, LIQUIDATE_ONLY, HALTED)

# modes that are permitted to run in THIS milestone (no live)
_ENABLED = {OFF, RESEARCH_ONLY, SHADOW, PAPER_AUTO, PAPER_FORWARD_EVIDENCE,
            PAPER_PAUSED, LIQUIDATE_ONLY, HALTED}
_LIVE = {LIMITED_LIVE, GUARDED_LIVE, FULL_AUTO}


class LiveModeDisabled(Exception):
    pass


def is_valid(mode: str) -> bool:
    return mode in MODES


def is_live(mode: str) -> bool:
    return mode in _LIVE


def assert_no_live(mode: str) -> None:
    """Hard stop: a live mode must never reach paper/broker execution in this milestone."""
    if mode in _LIVE:
        raise LiveModeDisabled(
            f"operating mode {mode!r} is a LIVE mode — disabled in this milestone; "
            "the user-owned live gate is the only path to live and is never crossed automatically")


def opens_new_entries(mode: str) -> bool:
    """Whether the loop may OPEN new paper positions in this mode."""
    return mode in (PAPER_AUTO, PAPER_FORWARD_EVIDENCE)


def manages_positions(mode: str) -> bool:
    """Whether the loop should still manage/exit open positions."""
    return mode in (PAPER_AUTO, PAPER_FORWARD_EVIDENCE, PAPER_PAUSED, LIQUIDATE_ONLY, SHADOW)
