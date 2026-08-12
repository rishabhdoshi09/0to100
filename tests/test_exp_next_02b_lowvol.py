"""Tests for EXP-NEXT-02B frozen protocol (network-free; no outcome peek required)."""
from __future__ import annotations

import json
from pathlib import Path

from research.data_expansion import exp_next_02b_lowvol as X
from research.phase_next import protocol as P0


def test_frozen_protocol_exists_and_matches_constants():
    frozen = X.load_frozen()
    assert frozen["experiment_id"] == "EXP-NEXT-02B"
    assert frozen["snapshot_id"] == "2f683be0c73eaa33"
    assert frozen["parent_experiment_id"] == "EXP-NEXT-02"
    assert frozen["frozen_before_outcome_inspection"] is True
    assert frozen["lookback"] == P0.LOWVOL_LOOKBACK == X.LOOKBACK
    assert frozen["rebalance_every_sessions"] == P0.LOWVOL_REBALANCE == X.REBALANCE
    assert frozen["holding_horizon_sessions"] == P0.LOWVOL_HOLD == X.HOLD
    assert frozen["cohort_quantile"] == P0.LOWVOL_Q == X.Q
    assert frozen["production_authority"] is False
    assert frozen["no_ml"] is True
    assert frozen["partitions"]["discovery_start"] == X.DISCOVERY_START
    assert frozen["partitions"]["confirm_start"] == X.CONFIRM_START


def test_does_not_overwrite_parent_experiment_id():
    assert X.EXPERIMENT_ID != "EXP-NEXT-02"
    frozen = X.load_frozen()
    assert frozen["parent_result"] == "INCONCLUSIVE"


def test_path_a_cannot_override_flag():
    frozen = X.load_frozen()
    assert frozen["path_a_secondary"]["cannot_override_primary_verdict"] is True
