"""Network-free smoke tests for Phase A.5 runners (synthetic fixture)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.phase_a5 import prereg
from research.phase_a5.exp_structure import run_exp_a5_01
from research.phase_a5.exp_horizons import run_exp_a2_01
from research.phase_a5.metrics import gate_research_grade


@pytest.fixture
def synth_panel(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n, m = 200, 8
    idx = pd.bdate_range("2024-01-01", periods=n)
    # two-factor closes
    f1 = rng.normal(scale=0.01, size=n).cumsum()
    f2 = rng.normal(scale=0.01, size=n).cumsum()
    data = {}
    sectors = {}
    for i in range(m):
        base = f1 if i < m // 2 else f2
        data[f"S{i}"] = 100 * np.exp(base + rng.normal(scale=0.005, size=n))
        sectors[f"S{i}"] = "A" if i < m // 2 else "B"
    closes = pd.DataFrame(data, index=idx)
    root = tmp_path / "phase_a5"
    root.mkdir()
    closes.to_csv(root / "exploratory_closes.csv")
    (root / "sector_map.json").write_text(json.dumps(sectors))
    (root / "dataset_manifest.json").write_text(json.dumps({
        "trust_class": "DISPLAY_ONLY",
        "research_tier": "LIMITED_RESEARCH",
        "source": "synthetic_fixture",
        "note": "test only",
        "n_symbols": m,
        "n_sessions": n,
    }))
    monkeypatch.setattr("research.phase_a5.dataset.ROOT", root)
    monkeypatch.setattr("research.phase_a5.dataset.CLOSES_CSV", root / "exploratory_closes.csv")
    monkeypatch.setattr("research.phase_a5.dataset.SECTOR_JSON", root / "sector_map.json")
    monkeypatch.setattr("research.phase_a5.dataset.MANIFEST_JSON", root / "dataset_manifest.json")
    monkeypatch.setattr("research.phase_a5.dataset.SNAPSHOT_ROOT", root / "snapshots")
    monkeypatch.setattr(prereg, "_A5_DB", root / "experiments.db")
    monkeypatch.setattr(prereg, "_A5_MEM", root / "scientific_memory.db")
    return closes, sectors


def test_gate_blocks_non_research_grade():
    g = gate_research_grade({"trust_class": "DISPLAY_ONLY", "research_grade": False})
    assert g["may_promote"] is False


def test_structure_and_horizon_runners_preregister(synth_panel):
    closes, sectors = synth_panel
    manifest = {
        "trust_class": "DISPLAY_ONLY",
        "research_grade": False,
        "snapshot_id": "testsnap",
        "source": "synthetic_fixture",
    }
    a5 = run_exp_a5_01(closes=closes, sectors=sectors, manifest=manifest,
                       lookback=40, step=20, n_clusters=2, oos_start_frac=0.6)
    assert a5["experiment_id"] == "EXP-A5-01"
    assert a5["hypothesis_id"]
    assert a5["verdict"] == "INCONCLUSIVE"  # blocked by trust class
    assert a5["production_authority"] is False

    a2 = run_exp_a2_01(closes=closes, manifest=manifest, oos_start_frac=0.6)
    assert a2["experiment_id"] == "EXP-A2-01"
    assert a2["hypothesis_id"]
    assert a2["verdict"] == "INCONCLUSIVE"
    assert a2["production_authority"] is False
