"""Pre-registration helpers for Phase A.5 experiments."""
from __future__ import annotations

import json
from pathlib import Path

from research import registry as REG
from research import scientific_memory as SM

# Isolate Phase A.5 experiment DB under logs/phase_a5
_A5_DB = Path(__file__).resolve().parents[2] / "logs" / "phase_a5" / "experiments.db"
_A5_MEM = Path(__file__).resolve().parents[2] / "logs" / "phase_a5" / "scientific_memory.db"


def bind_phase_a5_stores() -> None:
    """Point registry + scientific memory at Phase A.5 isolated DBs."""
    REG._DB_PATH = _A5_DB
    SM._DB_PATH = _A5_MEM


def preregister(
    *,
    experiment_id: str,
    hypothesis: str,
    null_hypothesis: str,
    success_criteria: dict,
    data_window: dict,
    protocol: dict,
    seed: int = 42,
    code_hash: str = "phase_a5",
) -> str:
    """Register BEFORE seeing results. Success criteria are frozen here."""
    bind_phase_a5_stores()
    description = json.dumps({
        "experiment_id": experiment_id,
        "hypothesis": hypothesis,
        "null_hypothesis": null_hypothesis,
        "protocol": protocol,
    }, sort_keys=True)
    # Include experiment_id in name so ids are human-traceable
    return REG.register_hypothesis(
        name=f"{experiment_id}:{hypothesis[:80]}",
        success_criteria=success_criteria,
        data_window=data_window,
        description=description,
        seed=seed,
        code_hash=code_hash,
    )


def record(hid: str, metrics: dict) -> dict:
    bind_phase_a5_stores()
    return REG.record_result(hid, metrics)


def remember_negative(statement: str, *, signal: str, evidence_n: int, notes: str = "") -> None:
    bind_phase_a5_stores()
    SM.record_negative(statement, signal=signal, evidence_n=evidence_n, notes=notes)


def remember_watch(statement: str, *, signal: str, evidence_n: int, ev_r: float | None,
                   hypothesis_id: str, notes: str = "") -> None:
    bind_phase_a5_stores()
    SM.record_belief(
        statement, signal=signal, status=SM.WATCH, evidence_n=evidence_n,
        confidence="LOW", ev_r=ev_r, hypothesis_id=hypothesis_id, notes=notes,
    )
