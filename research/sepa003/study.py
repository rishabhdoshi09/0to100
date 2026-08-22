"""SEPA-003 study driver. Research only."""
from __future__ import annotations

import json
from typing import Any

from research.sepa003.analyze import run_analysis
from research.sepa003.constants import OUT_DIR
from research.sepa003.dataset import persist_dataset, reconstruct
from research.sepa003.report import write_all


def run_study_003(*, max_setups: int | None = None, collect_controls: bool = True) -> dict[str, Any]:
    print("SEPA-003 reconstruct start", flush=True)
    payload = reconstruct(max_setups=max_setups, collect_controls=collect_controls)
    paths = persist_dataset(payload)
    stats = run_analysis(payload)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stat_path = OUT_DIR / "sepa_003_stats.json"
    stat_path.write_text(json.dumps(stats, indent=2, default=str))
    reports = write_all(payload, stats)
    return {
        "paths": paths,
        "stats_path": str(stat_path),
        "reports": {k: str(v) for k, v in reports.items()},
        "n_fills": stats.get("n_reconstructed_fills"),
        "decay_verdict": stats.get("decay_verdict"),
        "index_source": payload.get("index_source"),
    }
