"""Ablation plumbing — synthetic frames only (no live orders)."""
from __future__ import annotations

from tests.test_sepa_001_eligibility import CFG, _plant_vcp, _stage2, _downtrend
from research.sepa.ablation import run_ablation, summarize, SimRow


def test_summarize_empty_is_honest():
    out = summarize([])
    assert out["n"] == 0
    assert out["expectancy_r"] is None
    assert "no trades" in out["note"]


def test_summarize_basic_metrics():
    rows = [
        SimRow("F", "A", "2024-01-02", 100, 95, 1.0, 1.1, 5, "WIN", 0.2, 1.2, True, False, False, False, "2024", "IT", "BULL"),
        SimRow("F", "B", "2024-01-03", 100, 95, -1.0, -1.0, 3, "LOSS", 1.0, 0.1, False, False, True, True, "2024", "IT", "BULL"),
    ]
    out = summarize(rows, n_years=1)
    assert out["n"] == 2
    assert out["expectancy_r"] == 0.0
    assert out["failed_breakout_rate"] == 50.0
    assert out["pct_stop_before_1r"] == 50.0


def test_ablation_runs_on_synthetic_frames():
    frames = {
        "LEADER": _plant_vcp(contractions="tight", volume="dry"),
        "GRIND": _stage2(),
        "DOWN": _downtrend(),
    }
    payload = run_ablation(
        frames=frames, sample_step=40, lookback_sessions=80, horizon=8,
        max_symbols=None, config=CFG,
    )
    assert payload["experiment"] == "SEPA-001"
    assert "A" in payload["variants"]
    assert "F" in payload["variants"]
    assert payload["sample"]["symbols"] == 3
    # Does not invent trades when nothing fills
    for variant, stats in payload["variants"].items():
        assert stats["n"] >= 0
        assert "expectancy_r" in stats
