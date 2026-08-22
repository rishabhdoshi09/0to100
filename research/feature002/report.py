"""Write FEATURE-002 status notes. No manufactured graduation."""
from __future__ import annotations

import json

from research.feature002.constants import (
    FEATURE_001_FINAL_COMMIT,
    FEATURE_001_LAST_SAMPLE,
    FEATURE_SET_VERSION,
    FORWARD_START_DATE,
    FORWARD_START_TS_IST,
    OUT_DIR,
    UNTIL_MATURE,
    protocol_hash,
)
from research.feature002.evaluate import summarize
from research.feature002.ledger import counts


def write_all(*, path=None) -> dict[str, str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = summarize(path=path)
    ledger = counts(path=path)
    status = summary.get("status") or UNTIL_MATURE
    mat = summary.get("maturity") or {}

    (OUT_DIR / "feature_002_status.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    (OUT_DIR / "feature_002_manifest.json").write_text(json.dumps({
        "experiment": "FEATURE-002",
        "feature_set_version": FEATURE_SET_VERSION,
        "protocol_hash": protocol_hash(),
        "feature_001_final_commit": FEATURE_001_FINAL_COMMIT,
        "feature_001_last_sample": FEATURE_001_LAST_SAMPLE,
        "forward_start_date": FORWARD_START_DATE,
        "forward_start_ts_ist": FORWARD_START_TS_IST,
        "status": status,
        "ledger": ledger,
    }, indent=2))

    def w(name: str, body: str):
        p = OUT_DIR / name
        p.write_text(body.rstrip() + "\n")
        return str(p)

    w("FEATURE_002_MONTHLY.md", f"""# FEATURE-002 — Monthly

**Status:** {status}

Resolved primary 5d observations: {mat.get("n_resolved_5d", 0)}.
Months spanned: {mat.get("n_months", 0)}. Stage: {mat.get("stage")}.

No monthly graduation table is published until INTERIM.
""")

    w("FEATURE_002_STRATEGY_FAMILIES.md", f"""# FEATURE-002 — Strategy families

**Status:** {status}

Family resolved counts (primary live_scan only): `{mat.get("family_resolved") or {}}`.

Do not pool families. No family policy until DECISION-CAPABLE and n≥100 in that family.
""")

    w("FEATURE_002_RANKING_RESULTS.md", f"""# FEATURE-002 — Ranking results

**Status:** {status}

Primary comparisons (R0 vs R1 vs R2) are withheld while the sample is below INTERIM.

Rank metrics: `{summary.get("rank_metrics")}`.
""")

    w("FEATURE_002_DECISION.md", f"""# FEATURE-002 — Decision

**{status}**

This is the correct result. FEATURE-001 nominated Trend and RS. Only new
live-scan observations recorded on or after `{FORWARD_START_TS_IST}`, with
`session_date >= {FORWARD_START_DATE}`, can promote them.

## Classification (not yet allowed)

| Feature | Label |
|---|---|
| Trend (`trend_features_v1`) | *deferred — insufficient new data* |
| RS (`rs_features_v1`) | *deferred — insufficient new data* |

Allowed labels later (exactly one each): `GRADUATE_RANK_FEATURE` |
`EXTEND_FORWARD_VALIDATION` | `KEEP_RESEARCH_ONLY` | `RETIRE`.

`GRADUATE_RANK_FEATURE` would still **not** change production. FEATURE-003
is a separate milestone and is not started.

## The thirteen questions

All answers are: **not estimable on the primary live-scan sample yet.**

1. Does RS beat production rank? — insufficient new data
2. Does Trend beat production rank? — insufficient new data
3. Top-1? — insufficient new data
4. Top-3? — insufficient new data
5. Top-5? — insufficient new data
6. Tail loss? — insufficient new data
7. Families that benefit? — insufficient new data
8. Families that do not? — insufficient new data
9. Monthly stability? — insufficient new data
10. Regime stability? — insufficient new data
11. RS beyond production score? — insufficient new data
12. Trend beyond production score? — insufficient new data
13. Combined R3 vs RS alone? — insufficient new data (R3 remains exploratory)

Ledger: `{ledger}`.
""")
    return {"status": status}
